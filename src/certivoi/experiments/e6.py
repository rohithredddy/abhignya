"""E6 — ablations (04_EXPERIMENT_PLAN E6; Table 7).

Variants (same CRC procedure for all, every α):
- main: quantile LightGBM, target from config (CREAT: chosen by val pinball), isotonic.
- target_absolute: quantile model on y instead of Δ.
- no_trend / no_panel: drop the trend feature group / cross-lab panel.
- no_isotonic: raw EDCP fed to CRC (validity must hold either way).
- B5_direct / B4_logistic: direct classifier and logistic regression on D.
- icu_interventions: not run — needs ``icu.inputevents`` extraction (optional
  ablation in 03_DATA_AND_COHORT §7); reported as such rather than omitted.
Aggregate output only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.eval.metrics import safe_auprc, safe_auroc
from certivoi.experiments.e3 import certify, resample_splits
from certivoi.models.train import load_info, train_lab
from certivoi.utils.io import Timer, write_results
from certivoi.viz.style import LAB_ORDER

QUANTILE_VARIANTS = {
    "target_absolute": {"tag": "abl_absolute", "target": "absolute"},
    "no_trend": {"tag": "abl_no_trend", "drop_groups": ("trend",)},
    "no_panel": {"tag": "abl_no_panel", "drop_groups": ("panel",)},
}


def _evaluate(cfg: Config, df: pd.DataFrame, score: str, lab: str, variant: str,
              note: str = "") -> list[dict]:
    cal, te = df[df["split"] == "calib"], df[df["split"] == "test"]
    held = df[df["split"].isin(["calib", "test"])]
    draws = list(resample_splits(held, float(cfg.get("eval.resample_calib_share")),
                                 int(cfg.get("splits.crc_resamples")), cfg.seed))
    auroc = safe_auroc(te["D"].to_numpy(), te[score].to_numpy())
    auprc = safe_auprc(te["D"].to_numpy(), te[score].to_numpy())
    rows = []
    for a in cfg.alphas:
        m = certify(cal, te, score, a)
        rs = [certify(c, t, score, a) for c, t in draws]
        rows.append({"lab": lab, "variant": variant, "alpha": a, "auroc": auroc,
                     "auprc": auprc, "risk_test": m["risk"], "reduction_test": m["reduction"],
                     "resample_risk_mean": float(np.nanmean([r["risk"] for r in rs])),
                     "resample_reduction_mean": float(np.nanmean([r["reduction"] for r in rs])),
                     "note": note})
    return rows


def run(cfg: Config) -> pd.DataFrame:
    with Timer() as t:
        rows: list[dict] = []
        for lab in LAB_ORDER:
            main = pd.read_parquet(train_lab(cfg, lab))
            target = load_info(cfg, lab).get("target", "")
            rows += _evaluate(cfg, main, "score_ours", lab, "main", f"target={target}")
            rows += _evaluate(cfg, main, "edcp_raw", lab, "no_isotonic")
            rows += _evaluate(cfg, main, "score_B5", lab, "B5_direct")
            rows += _evaluate(cfg, main, "score_B4", lab, "B4_logistic")
            for name, kw in QUANTILE_VARIANTS.items():
                df = pd.read_parquet(train_lab(cfg, lab, models=("ours",), **kw))
                rows += _evaluate(cfg, df, "score_ours", lab, name)
            rows.append({"lab": lab, "variant": "icu_interventions", "alpha": np.nan,
                         "note": "not run: requires icu.inputevents extraction (optional)"})
        table7 = pd.DataFrame(rows)
    write_results(cfg, "E6", {"table7_ablations": table7}, None, t.elapsed)
    return table7
