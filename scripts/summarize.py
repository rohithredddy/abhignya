"""Print headline numbers from a results folder (aggregates only).

Usage: python scripts/summarize.py results/challenge2019 [--alpha 0.05]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n== {title} ==")
    print(df.round(4).to_string(index=False) if len(df) else "(empty)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    r, al = Path(a.results), a.alpha
    meta = json.loads((r / "E1" / "meta.json").read_text(encoding="utf-8"))
    print(f"results: {r}  data_mode: {meta.get('data_mode')}  alpha shown: {al}")
    if (p := r / "cohort" / "flow.csv").exists():
        f = pd.read_csv(p)
        show("repeat events after cap", f[f["step"] == f["step"].max()][
            ["lab", "n_events", "n_patients"]])
    if (p := r / "E1" / "table2_labels.csv").exists():
        show("E1 labels", pd.read_csv(p)[["lab", "n_events", "prev_D", "pct_D_normal",
                                          "pct_D_stable"]])
    if (p := r / "E2" / "table3_performance.csv").exists():
        t = pd.read_csv(p)
        show("E2 AUROC / AUPRC", t.pivot(index="lab", columns="model", values="auroc")
             .reset_index())
    if (p := r / "E3" / "table4_certified.csv").exists():
        t = pd.read_csv(p)
        show("E3 certified tier (all alpha)", t[["lab", "alpha", "resample_risk_mean",
                                                 "resample_reduction_mean", "reduction_test",
                                                 "valid_on_average"]])
    if (p := r / "E3" / "crc_all_scores.csv").exists():
        t = pd.read_csv(p)
        show(f"E3 CRC on every score, alpha={al}: reduction",
             t[t["alpha"] == al].pivot(index="lab", columns="model",
                                       values="resample_reduction_mean").reset_index())
    if (p := r / "E4" / "temporal.csv").exists():
        t = pd.read_csv(p)
        if "setting" in t:
            show(f"E4 external site / temporal, alpha={al}",
                 t[t["alpha"] == al][["lab", "setting", "risk", "reduction", "n_test"]])
    if (p := r / "E7" / "table9_panel.csv").exists():
        t = pd.read_csv(p)
        if "alpha" in t:
            show(f"E7 panels, alpha={al}", t[t["alpha"] == al][
                ["panel", "method", "risk_mean", "draw_reduction_mean"]])
    if (p := r / "E7" / "table10_online.csv").exists():
        t = pd.read_csv(p)
        if "alpha" in t:
            t = t[(t["alpha"] == al) & (t["order"].isin(["icu_shift", "temporal"]))]
            show(f"E7 online, alpha={al}", t[["lab", "order", "method", "miss_rate_mean",
                                              "worst_window_mean", "reduction_mean",
                                              "finite_T_bound"]])


if __name__ == "__main__":
    main()
