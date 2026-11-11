"""E3 — certified reduction vs α (04_EXPERIMENT_PLAN E3; Table 4, Figs. 4–5).

- Validity: realised risk on test (main split) and over ``splits.crc_resamples``
  patient-level re-draws of the held-out (calib ∪ test) patients, model fixed.
- Utility: reduction rate; tests avoided per 1,000 patient-days.
- LTT for each β: realised FSR and reduction.
- Comparison: (a) baselines at natural thresholds; (b) CRC applied to every score.
Aggregate output only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.conformal.crc import crc_threshold, risk_and_reduction
from certivoi.conformal.ltt import fsr, ltt_threshold
from certivoi.models.train import train_lab
from certivoi.triage import assign_tiers, high_value_threshold
from certivoi.utils.io import Timer, write_results
from certivoi.viz.style import DOUBLE_COL, LAB_ORDER, LINESTYLES, MARKERS, figure

SCORES = ("ours", "B1", "B2", "B3", "B4", "B5")


def patient_days(df: pd.DataFrame) -> float:
    """Sum of admission lengths (days) over admissions present in ``df``."""
    return float(df.drop_duplicates("hadm_id")["los_days"].sum())


def resample_splits(held: pd.DataFrame, calib_share: float, reps: int, seed: int):
    """Yield (calib, test) frames from patient-level re-draws of held-out patients."""
    subj = held["subject_id"].unique()
    rng = np.random.default_rng(seed)
    n_cal = int(round(calib_share * len(subj)))
    for _ in range(reps):
        perm = rng.permutation(subj)
        cal_ids = set(perm[:n_cal])
        m = held["subject_id"].isin(cal_ids).to_numpy()
        yield held[m], held[~m]


def certify(cal: pd.DataFrame, te: pd.DataFrame, score: str, alpha: float) -> dict:
    lam = crc_threshold(cal[score].to_numpy(), cal["D"].to_numpy(), alpha)
    risk, red = risk_and_reduction(te[score].to_numpy(), te["D"].to_numpy(), lam)
    return {"lambda": lam, "risk": risk, "reduction": red}


def run(cfg: Config) -> pd.DataFrame:
    alphas = cfg.alphas
    betas = [float(b) for b in cfg.get("conformal.ltt.betas")]
    delta = float(cfg.get("conformal.ltt.delta"))
    grid = int(cfg.get("conformal.ltt.grid_size"))
    reps = int(cfg.get("splits.crc_resamples"))
    share = float(cfg.get("eval.resample_calib_share"))
    nat = float(cfg.get("baselines.natural_threshold"))
    min_prec = float(cfg.get("triage.high_value_min_precision"))
    with Timer() as t:
        t4, ltt_rows, nat_rows, all_rows, box, tiers = [], [], [], [], {}, []
        fig4, ax4 = figure(DOUBLE_COL, 2.0, ncols=4)
        fig5, ax5 = figure(DOUBLE_COL, 2.0, ncols=4)
        for i, lab in enumerate(LAB_ORDER):
            df = pd.read_parquet(train_lab(cfg, lab))
            va, cal, te = (df[df["split"] == s] for s in ("val", "calib", "test"))
            held = df[df["split"].isin(["calib", "test"])]
            pdays = patient_days(te)
            draws = list(resample_splits(held, share, reps, cfg.seed))
            tau_h = high_value_threshold(va["score_ours"].to_numpy(), va["D"].to_numpy(),
                                         min_prec)
            box[lab] = []
            for a in alphas:
                main = certify(cal, te, "score_ours", a)
                rs = [certify(c, tt, "score_ours", a) for c, tt in draws]
                risks = np.array([r["risk"] for r in rs])
                box[lab].append(risks)
                n_flag = main["reduction"] * len(te)
                t4.append({
                    "lab": lab, "alpha": a, "lambda": main["lambda"],
                    "risk_test": main["risk"], "reduction_test": main["reduction"],
                    "resample_risk_mean": float(np.nanmean(risks)),
                    "resample_risk_p05": float(np.nanquantile(risks, 0.05)),
                    "resample_risk_p95": float(np.nanquantile(risks, 0.95)),
                    "resample_reduction_mean": float(np.nanmean([r["reduction"] for r in rs])),
                    "valid_on_average": bool(np.nanmean(risks) <= a),
                    "avoided_per_1000_patient_days": 1000 * n_flag / pdays if pdays else np.nan,
                })
                tier = assign_tiers(te["score_ours"].to_numpy(), main["lambda"], tau_h)
                d = te["D"].to_numpy()
                for name in ("low_value", "potentially_useful", "high_value"):
                    m = tier == name
                    tiers.append({"lab": lab, "alpha": a, "tier": name,
                                  "share": float(m.mean()),
                                  "precision_D": float(d[m].mean()) if m.any() else np.nan})
                for model in SCORES:
                    col = f"score_{model}"
                    r = certify(cal, te, col, a)
                    rr = [certify(c, tt, col, a) for c, tt in draws]
                    all_rows.append({"lab": lab, "alpha": a, "model": model,
                                     "risk_test": r["risk"], "reduction_test": r["reduction"],
                                     "resample_risk_mean": float(np.nanmean(
                                         [x["risk"] for x in rr])),
                                     "resample_reduction_mean": float(np.nanmean(
                                         [x["reduction"] for x in rr]))})
            for b in betas:
                lam = ltt_threshold(cal["score_ours"].to_numpy(), cal["D"].to_numpy(), b,
                                    delta, grid)
                _, red = risk_and_reduction(te["score_ours"].to_numpy(), te["D"].to_numpy(), lam)
                rs = []
                for c, tt in draws:
                    lr = ltt_threshold(c["score_ours"].to_numpy(), c["D"].to_numpy(), b, delta,
                                       grid)
                    rs.append(fsr(tt["score_ours"].to_numpy(), tt["D"].to_numpy(), lr))
                rs = np.array(rs, float)
                ltt_rows.append({"lab": lab, "beta": b, "delta": delta, "lambda": lam,
                                 "fsr_test": fsr(te["score_ours"].to_numpy(),
                                                 te["D"].to_numpy(), lam),
                                 "reduction_test": red,
                                 "share_resamples_fsr_le_beta": float(np.mean(
                                     np.isnan(rs) | (rs <= b)))})
            for model in SCORES:
                r, red = risk_and_reduction(te[f"score_{model}"].to_numpy(), te["D"].to_numpy(),
                                            nat)
                nat_rows.append({"lab": lab, "model": model, "threshold": nat, "risk_test": r,
                                 "reduction_test": red,
                                 "fsr_test": fsr(te[f"score_{model}"].to_numpy(),
                                                 te["D"].to_numpy(), nat)})

            # Fig. 4: risk-reduction curve on test (λ sweep) with CRC operating points
            s_te, d_te = te["score_ours"].to_numpy(), te["D"].to_numpy()
            lams = np.unique(np.quantile(s_te, np.linspace(0, 1, 101)))
            curve = np.array([risk_and_reduction(s_te, d_te, lam) for lam in lams])
            ax = ax4[i]
            ax.plot(curve[:, 1], curve[:, 0], color="black", linestyle=LINESTYLES[0])
            for j, a in enumerate(alphas):
                row = next(r for r in t4 if r["lab"] == lab and r["alpha"] == a)
                ax.plot(row["reduction_test"], row["risk_test"], marker=MARKERS[j % 6],
                        color="black", linestyle="none", markersize=4, label=f"α={a:g}")
                ax.axhline(a, color="0.6", linestyle=":", linewidth=0.7)
            ax.set_title(lab)
            ax.set_xlabel("Reduction (share flagged)")
            ax.set_ylim(0, max(alphas) * 2)
            ax5[i].boxplot(box[lab], widths=0.5, medianprops={"color": "black"},
                           flierprops={"markersize": 2})
            ax5[i].set_xticks(range(1, len(alphas) + 1), [f"{a:g}" for a in alphas])
            ax5[i].plot(range(1, len(alphas) + 1), alphas, "k_", markersize=12)
            ax5[i].set_title(lab)
            ax5[i].set_xlabel("α")
        ax4[0].set_ylabel("Realised risk")
        ax4[0].legend(frameon=False, fontsize=6)
        ax5[0].set_ylabel("Realised risk (resamples)")
        table4 = pd.DataFrame(t4)
    write_results(cfg, "E3", {
        "table4_certified": table4, "ltt": pd.DataFrame(ltt_rows),
        "natural_thresholds": pd.DataFrame(nat_rows), "crc_all_scores": pd.DataFrame(all_rows),
        "tiers": pd.DataFrame(tiers)},
        {"fig4_risk_reduction": fig4, "fig5_risk_resamples": fig5}, t.elapsed,
        crc_resamples=reps)
    return table4
