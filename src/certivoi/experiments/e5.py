"""E5 — explainability (04_EXPERIMENT_PLAN E5; Table 6, Figs. 7–8).

- Spearman(ρ, EDCP) per lab on test.
- Global importance: top-10 mean |φ^med| and mean |φ^unc| (Fig. 7).
- Faithfulness deletion test, k in ``explain.faithfulness_k`` (Table 6).
- SHAP additivity check (AC: max error <= 1e-3).
- Example explanations (Fig. 8 + text) ONLY in demo/synthetic mode — never on
  full-MIMIC rows (CONTRIBUTING.md §4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from certivoi.config import Config
from certivoi.conformal.crc import crc_threshold
from certivoi.explain.decompose import decompose
from certivoi.explain.faithfulness import deletion_test
from certivoi.explain.shap_attr import global_importance, tree_shap
from certivoi.explain.templates import reasons, render
from certivoi.features.build import manifest
from certivoi.models.train import edge_matrix, load_lab_frame, load_models, train_lab
from certivoi.triage import HIGH, LOW, POTENTIAL, assign_tiers, high_value_threshold
from certivoi.utils.io import Timer, write_results
from certivoi.viz.style import DOUBLE_COL, GRAYS, LAB_ORDER, figure

ADDITIVITY_TOL = 1e-3


def _qcol(tau: float) -> str:
    return f"q_{tau:g}"


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    lo, hi = (float(x) for x in cfg.get("explain.interval"))
    n_bg = int(cfg.get("explain.shap_background_rows"))
    n_ex = int(cfg.get("explain.shap_explain_rows"))
    ks = [int(k) for k in cfg.get("explain.faithfulness_k")]
    alpha = float(cfg.get("eval.main_alpha"))
    min_prec = float(cfg.get("triage.high_value_min_precision"))
    show_examples = not cfg.is_full
    rng = np.random.default_rng(cfg.seed)
    with Timer() as t:
        rho_rows, imp_rows, faith, add_rows, examples, ex_plot = [], [], [], [], [], []
        fig7, ax7 = figure(DOUBLE_COL, 4.2, ncols=4, nrows=2)
        for i, lab in enumerate(LAB_ORDER):
            preds = pd.read_parquet(train_lab(cfg, lab))
            frame, _ = load_lab_frame(cfg, lab)
            bundle = load_models(cfg, lab)
            fq, feats = bundle["models"]["ours"], bundle["features"]
            frame = frame.merge(preds[["event_id", "score_ours", "edcp_raw"]], on="event_id")
            te = frame[frame["split"] == "test"].reset_index(drop=True)
            tr = frame[frame["split"] == "train"]
            pte = preds[preds["split"] == "test"].set_index("event_id").loc[te["event_id"]]

            dec = decompose(pte[_qcol(lo)].to_numpy(), pte[_qcol(0.5)].to_numpy(),
                            pte[_qcol(hi)].to_numpy(), edge_matrix(te), te["a0"].to_numpy())
            finite = np.isfinite(dec["rho"].to_numpy())
            r = spearmanr(dec["rho"][finite], te["edcp_raw"][finite])
            rho_rows.append({"lab": lab, "spearman_rho_vs_edcp": float(r.statistic),
                             "p_value": float(r.pvalue), "n": int(finite.sum())})

            bg = tr[feats].sample(n=min(n_bg, len(tr)), random_state=cfg.seed)
            ex_idx = rng.choice(len(te), size=min(n_ex, len(te)), replace=False)
            ex = te.iloc[np.sort(ex_idx)].reset_index(drop=True)
            attr = tree_shap(fq, ex[feats], bg, (lo, 0.5, hi))
            for tau, err in attr.additivity_error().items():
                add_rows.append({"lab": lab, "tau": tau, "max_abs_error": err,
                                 "passes": bool(err <= ADDITIVITY_TOL)})
            gi = global_importance(attr, lo, hi)
            gi.insert(0, "lab", lab)
            imp_rows.append(gi)
            for row, col, title in ((0, "mean_abs_phi_med", "|φ med|"),
                                    (1, "mean_abs_phi_unc", "|φ unc|")):
                top = gi.nlargest(10, col)[::-1]
                ax = ax7[row, i]
                ax.barh(top["feature"], top[col], color=GRAYS[1 + row])
                ax.set_title(f"{lab}: mean {title}")
                ax.tick_params(axis="y", labelsize=5)

            importance = np.abs(attr.med) + np.abs(attr.unc(lo, hi))
            ft = deletion_test(fq, ex[feats], ex["y_last"].to_numpy(), edge_matrix(ex),
                               ex["a0"].to_numpy(), importance, bg, ks, cfg.seed)
            ft.insert(0, "lab", lab)
            faith.append(ft)

            if show_examples:
                cal = frame[frame["split"] == "calib"]
                va = frame[frame["split"] == "val"]
                lam = crc_threshold(cal["score_ours"].to_numpy(), cal["D"].to_numpy(), alpha)
                tau_h = high_value_threshold(va["score_ours"].to_numpy(), va["D"].to_numpy(),
                                             min_prec)
                pex = preds.set_index("event_id").loc[ex["event_id"]]
                dex = decompose(pex[_qcol(lo)].to_numpy(), pex[_qcol(0.5)].to_numpy(),
                                pex[_qcol(hi)].to_numpy(), edge_matrix(ex), ex["a0"].to_numpy())
                tiers = assign_tiers(ex["score_ours"].to_numpy(), lam, tau_h)
                desc = dict(zip(*manifest(cfg, lab)[["name", "description"]].T.values,
                                strict=True))
                unit = cfg.labs[lab].unit
                for tier in (LOW, POTENTIAL, HIGH):
                    cand = np.flatnonzero(tiers == tier)
                    if not len(cand):
                        continue
                    j = int(cand[0])
                    vals = ex[feats].to_numpy()[j]
                    text = render(lab, unit, tier, alpha, dex["y_hat"][j], dex["q_lo"][j],
                                  dex["q_hi"][j], dex["margin"][j], dex["boundary"][j],
                                  float(ex["score_ours"][j]),
                                  reasons(attr.med[j], vals, feats, desc),
                                  reasons(attr.unc(lo, hi)[j], vals, feats, desc,
                                          kind="uncertainty"))
                    examples.append(f"[{cfg.data_mode} data] {lab} / {tier}\n{text}\n")
                    if lab in ("HGB", "K"):
                        ex_plot.append((lab, tier, dex.iloc[j], edge_matrix(ex)[j], unit))

        ax7[1, 0].set_xlabel("mean |SHAP|")
        tables = {
            "rho_vs_edcp": pd.DataFrame(rho_rows),
            "global_importance": pd.concat(imp_rows, ignore_index=True),
            "table6_faithfulness": pd.concat(faith, ignore_index=True),
            "shap_additivity": pd.DataFrame(add_rows),
        }
        figs = {"fig7_global_importance": fig7}
        if ex_plot:
            fig8, ax8 = figure(DOUBLE_COL / 2, 0.45 * len(ex_plot) + 0.6)
            for k, (_lab, _tier, d, edges, _unit) in enumerate(ex_plot):
                ax8.plot([d["q_lo"], d["q_hi"]], [k, k], color="black", linewidth=2)
                ax8.plot(d["y_hat"], k, "o", color="white", markeredgecolor="black")
                for e in edges[1:-1]:
                    if np.isfinite(e):
                        ax8.plot([e, e], [k - 0.35, k + 0.35], color="0.5", linestyle=":")
            ax8.set_yticks(range(len(ex_plot)),
                           [f"{lab} · {tier.replace('_', ' ')}" for lab, tier, *_ in ex_plot])
            ax8.set_xlabel("Predicted value: median and 80% interval (dotted = thresholds)")
            figs["fig8_examples"] = fig8
    out = write_results(cfg, "E5", tables, figs, t.elapsed, examples_shown=show_examples)
    if examples:
        (out / "examples.txt").write_text(
            "Example explanations (open demo or synthetic data only).\n\n" + "\n".join(examples),
            encoding="utf-8")
    return tables
