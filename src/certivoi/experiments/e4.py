"""E4 — robustness (04_EXPERIMENT_PLAN E4; Table 5, Fig. 6).

1. Subgroups (sex, age band, race group, ICU/ward): global CRC vs Mondrian CRC.
2. Temporal shift: models trained/calibrated on earlier year groups, tested on
   later ones; then CRC re-calibrated on a small slice of the later period.
3. Threshold sensitivity: alternative action maps -> E1 summary + E3 summary,
   reusing the fixed quantile forecaster (its target does not depend on the map).
Aggregate output only; groups with < ``eval.min_cell_count`` events get no rates
in full mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.conformal.crc import crc_threshold, risk_and_reduction
from certivoi.conformal.mondrian import mondrian_flags, mondrian_thresholds
from certivoi.data.split import split_path
from certivoi.experiments.e1 import label_summary
from certivoi.experiments.e2 import quantile_cols
from certivoi.labels.actions import action_map
from certivoi.models.train import train_lab
from certivoi.utils.io import Timer, write_results
from certivoi.utils.logging import get_logger
from certivoi.viz.style import DOUBLE_COL, HATCHES, LAB_ORDER, figure
from certivoi.voi.edcp import IsotonicCalibrator, edcp

log = get_logger("certivoi.e4")


def _mask_small(cfg: Config, row: dict, n: int) -> dict:
    if cfg.is_full and n < int(cfg.get("eval.min_cell_count")):
        return {**row, "n_test": None, "risk": np.nan, "reduction": np.nan}
    return row


def subgroup_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for lab in LAB_ORDER:
        df = pd.read_parquet(train_lab(cfg, lab))
        cal, te = df[df["split"] == "calib"], df[df["split"] == "test"]
        for a in cfg.alphas:
            lam = crc_threshold(cal["score_ours"].to_numpy(), cal["D"].to_numpy(), a)
            for var in cfg.get("conformal.mondrian_groups"):
                thr = mondrian_thresholds(cal["score_ours"].to_numpy(), cal["D"].to_numpy(),
                                          cal[var].to_numpy(), a)
                mflag = mondrian_flags(te["score_ours"].to_numpy(), te[var].to_numpy(), thr)
                for g in sorted(te[var].astype(str).unique()):
                    m = (te[var].astype(str) == g).to_numpy()
                    d = te["D"].to_numpy()[m]
                    s = te["score_ours"].to_numpy()[m]
                    r_g, red_g = risk_and_reduction(s, d, lam)
                    base = {"lab": lab, "alpha": a, "group_var": var, "group": g,
                            "n_test": int(m.sum())}
                    rows.append(_mask_small(cfg, {**base, "method": "global", "risk": r_g,
                                                  "reduction": red_g}, int(m.sum())))
                    f = mflag[m]
                    rows.append(_mask_small(cfg, {
                        **base, "method": "mondrian",
                        "risk": float(np.mean(d * f)) if m.any() else np.nan,
                        "reduction": float(np.mean(f)) if m.any() else np.nan}, int(m.sum())))
    return pd.DataFrame(rows)


def temporal_table(cfg: Config) -> pd.DataFrame:
    sp = pd.read_parquet(split_path(cfg))
    counts = sp["split_temporal"].value_counts()
    if any(counts.get(s, 0) == 0 for s in ("train", "val", "calib", "test")):
        log.warning("E4 temporal: a temporal split is empty (%s); skipped", counts.to_dict())
        return pd.DataFrame([{"lab": lab, "status": "skipped: empty temporal split "
                              f"{counts.to_dict()}"} for lab in LAB_ORDER])
    rows = []
    frac = float(cfg.get("eval.temporal_recal_frac"))
    rng = np.random.default_rng(cfg.seed)
    for lab in LAB_ORDER:
        main = pd.read_parquet(train_lab(cfg, lab))
        tmp = pd.read_parquet(train_lab(cfg, lab, tag="temporal", split_col="split_temporal",
                                        models=("ours",)))
        cal, te = tmp[tmp["split"] == "calib"], tmp[tmp["split"] == "test"]
        mcal, mte = main[main["split"] == "calib"], main[main["split"] == "test"]
        subj = te["subject_id"].unique()
        recal_ids = set(rng.choice(subj, size=max(1, int(round(frac * len(subj)))),
                                   replace=False))
        rc = te[te["subject_id"].isin(recal_ids)]
        rt = te[~te["subject_id"].isin(recal_ids)]
        for a in cfg.alphas:
            for setting, (c, t_) in {"in_period": (mcal, mte), "shifted": (cal, te),
                                     "recalibrated": (rc, rt)}.items():
                lam = crc_threshold(c["score_ours"].to_numpy(), c["D"].to_numpy(), a)
                r, red = risk_and_reduction(t_["score_ours"].to_numpy(), t_["D"].to_numpy(),
                                            lam)
                rows.append({"lab": lab, "alpha": a, "setting": setting, "risk": r,
                             "reduction": red, "n_test": len(t_), "status": "ok"})
    return pd.DataFrame(rows)


def threshold_sensitivity(cfg: Config) -> pd.DataFrame:
    rows = []
    variants = cfg.get("actions.sensitivity_variants")
    for lab, override in variants.items():
        df = pd.read_parquet(train_lab(cfg, lab))
        amap = action_map(cfg.get("actions"), lab, override)
        a0 = amap.category(df["y_last"].to_numpy())
        cat = amap.category(df["y"].to_numpy())
        d = (a0 != cat).astype(int)
        cols, taus = quantile_cols(df)
        raw = edcp(df[cols].to_numpy(), taus, amap.edges(len(df)), a0)
        va = (df["split"] == "val").to_numpy()
        s = IsotonicCalibrator().fit(raw[va], d[va]).predict(raw)
        lab_df = pd.DataFrame({"D": d, "N": df["N"], "S": df["S"]})
        summ = label_summary(lab_df, lab)
        cal = (df["split"] == "calib").to_numpy()
        te = (df["split"] == "test").to_numpy()
        for a in cfg.alphas:
            lam = crc_threshold(s[cal], d[cal], a)
            r, red = risk_and_reduction(s[te], d[te], lam)
            rows.append({"lab": lab, "variant": str(override), "alpha": a,
                         "prev_D": summ["prev_D"], "pct_D_normal": summ["pct_D_normal"],
                         "pct_D_stable": summ["pct_D_stable"], "risk_test": r,
                         "reduction_test": red})
    return pd.DataFrame(rows)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    with Timer() as t:
        sub = subgroup_table(cfg)
        temp = temporal_table(cfg)
        sens = threshold_sensitivity(cfg)
        figs = {}
        if "setting" in temp.columns:
            fig, axes = figure(DOUBLE_COL, 2.0, ncols=4)
            settings = ["in_period", "shifted", "recalibrated"]
            for ax, lab in zip(axes, LAB_ORDER, strict=True):
                g = temp[temp["lab"] == lab]
                x = np.arange(len(cfg.alphas))
                for j, s in enumerate(settings):
                    v = g[g["setting"] == s].sort_values("alpha")["risk"].to_numpy()
                    ax.bar(x + (j - 1) * 0.27, v, width=0.27, color="white", edgecolor="black",
                           hatch=HATCHES[j], label=s.replace("_", " "))
                ax.plot(x, cfg.alphas, "k_", markersize=10)
                ax.set_xticks(x, [f"{a:g}" for a in cfg.alphas])
                ax.set_title(lab)
                ax.set_xlabel("α")
            axes[0].set_ylabel("Realised risk")
            axes[0].legend(frameon=False, fontsize=6)
            figs["fig6_temporal_shift"] = fig
    write_results(cfg, "E4", {"table5_subgroups": sub, "temporal": temp,
                              "threshold_sensitivity": sens}, figs, t.elapsed)
    return {"subgroups": sub, "temporal": temp, "sensitivity": sens}
