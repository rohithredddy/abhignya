"""E1 — label analysis: decision-anchored D vs proxies N (normal) and S (stable).

Implements 04_EXPERIMENT_PLAN E1 (Table 2, Fig. 2). Aggregate output only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.labels.build import build_labels
from certivoi.utils.io import Timer, privacy_suppress, write_results
from certivoi.viz.style import DOUBLE_COL, HATCHES, LAB_ORDER, figure


def label_summary(df: pd.DataFrame, lab: str) -> dict:
    """Prevalences and disagreement rates for one lab's labelled events."""
    d1 = df[df["D"] == 1]
    return {
        "lab": lab, "n_events": len(df), "n_action_changing": len(d1),
        "prev_D": df["D"].mean(), "prev_N": df["N"].mean(), "prev_S": df["S"].mean(),
        # share of action-changing results a proxy method would call "skip"
        "pct_D_normal": d1["N"].mean() if len(d1) else np.nan,
        "pct_D_stable": d1["S"].mean() if len(d1) else np.nan,
        "pct_D_normal_or_stable": ((d1["N"] == 1) | (d1["S"] == 1)).mean() if len(d1) else np.nan,
    }


def contingency(df: pd.DataFrame, lab: str) -> pd.DataFrame:
    rows = []
    for proxy in ("N", "S"):
        ct = df.groupby(["D", proxy]).size()
        for (d, p), n in ct.items():
            rows.append({"lab": lab, "proxy": proxy, "D": d, "proxy_value": p, "n": int(n)})
    return pd.DataFrame(rows)


def run(cfg: Config, labels: dict | None = None) -> pd.DataFrame:
    with Timer() as t:
        labels = labels or build_labels(cfg)
        summ, cont, fig_rows = [], [], []
        for lab in LAB_ORDER:
            df = pd.read_parquet(labels[lab], columns=["D", "N", "S"])
            summ.append(label_summary(df, lab))
            cont.append(contingency(df, lab))
            d1 = df[df["D"] == 1]
            n, s = d1["N"] == 1, d1["S"] == 1
            fig_rows.append([(n & s).mean(), (n & ~s).mean(), (~n & s).mean(), (~n & ~s).mean()])
        summary = privacy_suppress(cfg, pd.DataFrame(summ), ["n_events", "n_action_changing"])
        cont_df = privacy_suppress(cfg, pd.concat(cont, ignore_index=True), ["n"])

        fig, ax = figure(DOUBLE_COL / 2, 2.4)
        parts = ["normal & stable", "normal only", "stable only", "neither"]
        bottom = np.zeros(len(LAB_ORDER))
        vals = np.nan_to_num(np.array(fig_rows))
        for k, name in enumerate(parts):
            ax.bar(LAB_ORDER, vals[:, k], bottom=bottom, color="white", edgecolor="black",
                   hatch=HATCHES[k], label=name, linewidth=0.8)
            bottom += vals[:, k]
        ax.set_ylabel("Share of action-changing results")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    write_results(cfg, "E1", {"table2_labels": summary, "contingency": cont_df},
                  {"fig2_proxy_disagreement": fig}, t.elapsed)
    return summary
