"""E2 — predictive performance for target D (04_EXPERIMENT_PLAN E2; Table 3, Fig. 3).

Metrics on the test split with patient-level bootstrap CIs; forecaster pinball
loss per tau and empirical coverage of 80% / 90% intervals. Aggregate output only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.eval.metrics import bootstrap_ci, metric_fns, reliability_curve
from certivoi.models.quantile import pinball
from certivoi.models.train import load_info, train_lab
from certivoi.utils.io import Timer, privacy_suppress, write_results
from certivoi.viz.style import DOUBLE_COL, LAB_ORDER, LINESTYLES, MARKERS, figure

E2_MODELS = ("B2", "B3", "B4", "B5", "ours")
MIN_BIN_COUNT_PLOT = 10  # privacy: reliability bins with fewer events are not plotted


def quantile_cols(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    cols = [c for c in df.columns if c.startswith("q_")]
    taus = np.array([float(c[2:]) for c in cols])
    order = np.argsort(taus)
    return [cols[i] for i in order], taus[order]


def forecaster_metrics(te: pd.DataFrame, lab: str) -> list[dict]:
    cols, taus = quantile_cols(te)
    y = te["y"].to_numpy()
    rows = [{"lab": lab, "metric": f"pinball_{t:g}", "value": pinball(y, te[c].to_numpy(), t)}
            for c, t in zip(cols, taus, strict=True)]
    for lo, hi, name in ((0.10, 0.90, "coverage_80"), (0.05, 0.95, "coverage_90")):
        q = dict(zip(np.round(taus, 4), cols, strict=True))
        if lo in q and hi in q:
            inside = (y >= te[q[lo]]) & (y <= te[q[hi]])
            rows.append({"lab": lab, "metric": name, "value": float(inside.mean())})
    return rows


def run(cfg: Config) -> pd.DataFrame:
    with Timer() as t:
        fns = metric_fns(int(cfg.get("eval.ece_bins")))
        rows, frows, hyper = [], [], {}
        fig, axes = figure(DOUBLE_COL, 2.0, ncols=4)
        for ax, lab in zip(axes, LAB_ORDER, strict=True):
            df = pd.read_parquet(train_lab(cfg, lab))
            te = df[df["split"] == "test"]
            scores = {m: te[f"score_{m}"].to_numpy() for m in E2_MODELS}
            ci = bootstrap_ci(te["D"].to_numpy(), scores, te["subject_id"].to_numpy(), fns,
                              int(cfg.get("eval.bootstrap_reps")), cfg.seed)
            for m in E2_MODELS:
                r = {"lab": lab, "model": m, "n_test": len(te), "n_pos": int(te["D"].sum())}
                for k, (est, lo, hi) in ci[m].items():
                    r.update({k: est, f"{k}_lo": lo, f"{k}_hi": hi})
                rows.append(r)
            frows += forecaster_metrics(te, lab)
            hyper[lab] = load_info(cfg, lab)
            for j, m in enumerate(("ours", "B5")):
                mp, obs, cnt = reliability_curve(te["D"].to_numpy(), te[f"score_{m}"].to_numpy(),
                                                 int(cfg.get("eval.ece_bins")))
                keep = cnt >= MIN_BIN_COUNT_PLOT
                ax.plot(mp[keep], obs[keep], linestyle=LINESTYLES[j], marker=MARKERS[j],
                        markersize=3, color="black", label="Ours" if m == "ours" else "B5")
            ax.plot([0, 1], [0, 1], color="0.7", linewidth=0.8)
            ax.set_title(lab)
            ax.set_xlabel("Predicted P(D=1)")
        axes[0].set_ylabel("Observed rate")
        axes[0].legend(frameon=False)
        table3 = privacy_suppress(cfg, pd.DataFrame(rows), ["n_test", "n_pos"])
        forecast = pd.DataFrame(frows)
    out = write_results(cfg, "E2", {"table3_performance": table3, "forecaster": forecast},
                        {"fig3_reliability": fig}, t.elapsed)
    hp = {lab: {k: v for k, v in h.items() if k != "features"} for lab, h in hyper.items()}
    (out / "hyperparams.json").write_text(json.dumps(hp, indent=2, default=str),
                                          encoding="utf-8")
    return table3
