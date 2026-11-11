"""E7 — certification extensions (04_EXPERIMENT_PLAN E7; Tables 8–10, Fig. 9).

(a) Patient-clustered CRC vs event-level CRC (02 §5.5): event-level and
    patient-level realised risk over patient-level re-splits.
(b) Panel (specimen-level) certification (02 §5.6): CBC = HGB+PLT, BMP = K+CREAT.
    Naive per-lab certificates at α, Bonferroni per-lab at α/k, and panel CRC on
    max / noisy-or panel scores; draw-level miss rate and draws avoided.
(c) Online risk control with audited feedback (02 §5.7) on ordered test streams
    (shuffled, era, ward→ICU shift, and the temporal model when the temporal split
    exists): static CRC vs online updates with audit rates p.
Aggregate output only (CONTRIBUTING.md §4); rolling curves use windows of >= 100 tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.conformal.clustered import crc_threshold_clustered, patient_risk
from certivoi.conformal.crc import crc_threshold, risk_and_reduction
from certivoi.conformal.online import (
    estimate_bound,
    online_risk_control,
    rank_transform,
    static_result,
)
from certivoi.conformal.panel import naive_flags, pair_panel, panel_score
from certivoi.data.split import split_path
from certivoi.experiments.e3 import resample_splits
from certivoi.models.train import train_lab
from certivoi.utils.io import Timer, privacy_suppress, write_results
from certivoi.utils.logging import get_logger
from certivoi.viz.style import DOUBLE_COL, GRAYS, LAB_ORDER, LINESTYLES, figure

log = get_logger("certivoi.e7")
MIN_WINDOW = 100


def _held(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["split"].isin(["calib", "test"])]


# ---------------- (a) clustered ----------------
def clustered_table(cfg: Config, preds: dict[str, pd.DataFrame], draws_seed: int) -> pd.DataFrame:
    reps = int(cfg.get("splits.crc_resamples"))
    share = float(cfg.get("eval.resample_calib_share"))
    rows = []
    for lab, df in preds.items():
        draws = list(resample_splits(_held(df), share, reps, draws_seed))
        for a in cfg.alphas:
            for method in ("event_crc", "patient_crc"):
                ev_r, pt_r, red, n_units = [], [], [], []
                for cal, te in draws:
                    n_units.append(cal["subject_id"].nunique() if method == "patient_crc"
                                   else len(cal))
                    s, d, g = (cal["score_ours"].to_numpy(), cal["D"].to_numpy(),
                               cal["subject_id"].to_numpy())
                    lam = (crc_threshold(s, d, a) if method == "event_crc"
                           else crc_threshold_clustered(s, d, g, a))
                    r, f = risk_and_reduction(te["score_ours"].to_numpy(), te["D"].to_numpy(),
                                              lam)
                    ev_r.append(r)
                    red.append(f)
                    pt_r.append(patient_risk(te["score_ours"].to_numpy(), te["D"].to_numpy(),
                                             te["subject_id"].to_numpy(), lam)[0])
                rows.append({"lab": lab, "alpha": a, "method": method,
                             "event_risk_mean": float(np.nanmean(ev_r)),
                             "patient_risk_mean": float(np.nanmean(pt_r)),
                             "patient_risk_p95": float(np.nanquantile(pt_r, 0.95)),
                             "reduction_mean": float(np.nanmean(red)),
                             "n_cal_units_mean": float(np.mean(n_units)),
                             # CRC can flag anything only if n >= 1/alpha - 1 units
                             "min_units_needed": int(np.ceil(1 / a - 1))})
    return pd.DataFrame(rows)


# ---------------- (b) panels ----------------
def panel_table(cfg: Config, preds: dict[str, pd.DataFrame], draws_seed: int) -> pd.DataFrame:
    reps = int(cfg.get("splits.crc_resamples"))
    share = float(cfg.get("eval.resample_calib_share"))
    how_list = list(cfg.get("extensions.panel_scores"))
    rows = []
    for name, labs in cfg.get("extensions.panels").items():
        if not all(lab in preds for lab in labs):
            continue
        p = pair_panel({lab: preds[lab] for lab in labs})
        held = _held(p)
        n_test = int((p["split"] == "test").sum())
        if held["subject_id"].nunique() < 4:
            rows.append({"panel": name, "status": "skipped: too few paired draws"})
            continue
        draws = list(resample_splits(held, share, reps, draws_seed))
        s_cols = [f"s_{lab}" for lab in labs]
        for a in cfg.alphas:
            methods = {"naive_per_lab_alpha": None, "bonferroni_per_lab": None,
                       **{f"panel_crc_{h}": h for h in how_list}}
            for m, how in methods.items():
                risks, reds = [], []
                for cal, ev in draws:
                    if m == "naive_per_lab_alpha":
                        f = naive_flags(cal, ev, labs, a)
                    elif m == "bonferroni_per_lab":
                        f = naive_flags(cal, ev, labs, a / len(labs))
                    else:
                        lam = crc_threshold(panel_score(cal[s_cols].to_numpy(), how),
                                            cal["D_panel"].to_numpy(), a)
                        f = panel_score(ev[s_cols].to_numpy(), how) <= lam
                    dp = ev["D_panel"].to_numpy()
                    risks.append(float((dp * f).mean()) if len(dp) else np.nan)
                    reds.append(float(f.mean()) if len(dp) else np.nan)
                rows.append({"panel": name, "labs": "+".join(labs), "alpha": a, "method": m,
                             "n_draws_test": n_test,
                             "prev_D_panel": float(p["D_panel"].mean()),
                             "risk_mean": float(np.nanmean(risks)),
                             "risk_p95": float(np.nanquantile(risks, 0.95)),
                             "draw_reduction_mean": float(np.nanmean(reds)),
                             "valid_on_average": bool(np.nanmean(risks) <= a),
                             "status": "ok"})
    return privacy_suppress(cfg, pd.DataFrame(rows), ["n_draws_test"]) if rows else \
        pd.DataFrame(rows)


# ---------------- (c) online ----------------
def _orders(stream: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    key = rng.random(len(stream))
    era = stream["anchor_year_group"].astype(str).to_numpy()
    icu = stream["icu"].fillna(False).astype(bool).to_numpy()
    return {"shuffled": np.argsort(key),
            "era": np.lexsort((key, era)),
            "icu_shift": np.lexsort((key, icu))}


def _temporal_available(cfg: Config) -> bool:
    sp = pd.read_parquet(split_path(cfg))
    counts = sp["split_temporal"].value_counts()
    return all(counts.get(s, 0) > 0 for s in ("train", "val", "calib", "test"))


def online_table(cfg: Config, preds: dict[str, pd.DataFrame]
                 ) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    eta = float(cfg.get("extensions.online.eta"))
    ps = [float(p) for p in cfg.get("extensions.online.audit_rates")]
    reps = int(cfg.get("extensions.online.reps"))
    wfrac = float(cfg.get("extensions.online.window_frac"))
    alphas = [float(a) for a in cfg.get("extensions.online.alphas")]
    main_alpha = float(cfg.get("eval.main_alpha"))
    temporal = _temporal_available(cfg)
    rows, curves = [], {}
    for lab, df in preds.items():
        sources = {"main": df}
        if temporal:
            sources["temporal_model"] = pd.read_parquet(train_lab(
                cfg, lab, tag="temporal", split_col="split_temporal", models=("ours",)))
        for src, frame in sources.items():
            cal = frame[frame["split"] == "calib"]
            stream = frame[frame["split"] == "test"].reset_index(drop=True)
            T = len(stream)
            if T < 2 * MIN_WINDOW or len(cal) == 0:
                rows.append({"lab": lab, "source": src, "status": f"skipped: stream {T}"})
                continue
            w = max(MIN_WINDOW, int(wfrac * T))
            f = rank_transform(cal["score_ours"].to_numpy())
            u_all, d_all = f(stream["score_ours"].to_numpy()), stream["D"].to_numpy()
            for a in alphas:
                lam_cal = crc_threshold(cal["score_ours"].to_numpy(), cal["D"].to_numpy(), a)
                lam1 = float(f(np.array([lam_cal]))[0]) if np.isfinite(lam_cal) else 0.0
                acc: dict[tuple[str, str], list] = {}
                for k in range(reps):
                    rng = np.random.default_rng(cfg.seed + 1000 * k)
                    orders = _orders(stream, rng)
                    if src == "temporal_model":
                        orders = {"temporal": orders["era"]}
                    for order, idx in orders.items():
                        u, d = u_all[idx], d_all[idx]
                        res = {"static": static_result(u, d, lam1)}
                        for p in ps:
                            res[f"online_p{p:g}"] = online_risk_control(u, d, a, eta, p, lam1,
                                                                        rng)
                        for m, r in res.items():
                            acc.setdefault((order, m), []).append(
                                (r.miss_rate, r.worst_window(w), r.flag_rate, r.reduction))
                            if k == 0 and a == main_alpha and m in ("static", "online_p0.2"):
                                c = np.convolve(r.loss, np.ones(w) / w, mode="valid")
                                curves.setdefault(f"{lab}|{order}", {})[m] = c
                for (order, m), vals in acc.items():
                    v = np.array(vals, float)
                    p = float(m.split("_p")[1]) if m.startswith("online") else np.nan
                    rows.append({"lab": lab, "source": src, "order": order, "alpha": a,
                                 "method": m, "audit_p": p, "n_stream": T, "window": w,
                                 "miss_rate_mean": float(np.nanmean(v[:, 0])),
                                 "miss_rate_p95": float(np.nanquantile(v[:, 0], 0.95)),
                                 "worst_window_mean": float(np.nanmean(v[:, 1])),
                                 "flag_rate_mean": float(np.nanmean(v[:, 2])),
                                 "reduction_mean": float(np.nanmean(v[:, 3])),
                                 "finite_T_bound": (estimate_bound(a, eta, p, lam1, T)
                                                    if np.isfinite(p) else np.nan),
                                 "status": "ok"})
    return privacy_suppress(cfg, pd.DataFrame(rows), ["n_stream"]), curves


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    with Timer() as t:
        preds = {lab: pd.read_parquet(train_lab(cfg, lab)) for lab in LAB_ORDER}
        tables = {"table8_clustered": clustered_table(cfg, preds, cfg.seed),
                  "table9_panel": panel_table(cfg, preds, cfg.seed)}
        online, curves = online_table(cfg, preds)
        tables["table10_online"] = online
        figs = {}
        shown = [k for k in curves if k.endswith("|icu_shift") or k.endswith("|temporal")]
        if shown:
            fig, ax = figure(DOUBLE_COL, 2.0, ncols=len(shown[:4]))
            ax = np.atleast_1d(ax)
            alpha = float(cfg.get("eval.main_alpha"))
            for i, key in enumerate(shown[:4]):
                for j, (m, c) in enumerate(curves[key].items()):
                    ax[i].plot(c, color=GRAYS[j * 2], linestyle=LINESTYLES[j], linewidth=0.9,
                               label=m.replace("_p", " p="))
                ax[i].axhline(alpha, color="0.6", linestyle=":", linewidth=0.8)
                ax[i].set_title(key.replace("|", " · "), fontsize=7)
                ax[i].set_xlabel("stream position")
            ax[0].set_ylabel("rolling miss rate")
            ax[0].legend(fontsize=6)
            figs["fig9_online_shift"] = fig
    write_results(cfg, "E7", tables, figs, t.elapsed)
    log.info("E7 done in %.1fs", t.elapsed)
    return tables
