"""Leakage sanity check (CONTRIBUTING.md §11): AUROC of a model-free rule.

Score = -(distance from the last value to the nearest boundary of its action
interval). Writes results/<dataset>/E2/sanity_threshold_distance.csv (aggregate).
Usage: python scripts/sanity_distance.py --override configs/challenge2019.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from certivoi.config import load_config
from certivoi.explain.decompose import nearest_boundary
from certivoi.models.train import edge_matrix, train_lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, override_path=a.override)
    rows = []
    for lab in ("HGB", "K", "CREAT", "PLT"):
        df = pd.read_parquet(train_lab(cfg, lab))
        te = df[df["split"] == "test"]
        margin, _ = nearest_boundary(te["y_last"].to_numpy(float), edge_matrix(te),
                                     te["a0"].to_numpy())
        score = -np.abs(np.where(np.isfinite(margin), margin, 1e9))
        rows.append({"lab": lab, "n_test": len(te),
                     "auroc_distance_rule": float(roc_auc_score(te["D"], score)),
                     "auroc_ours": float(roc_auc_score(te["D"], te["score_ours"])),
                     "auroc_B5": float(roc_auc_score(te["D"], te["score_B5"]))})
    t = pd.DataFrame(rows)
    out = cfg.results_dir / "E2"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "sanity_threshold_distance.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
