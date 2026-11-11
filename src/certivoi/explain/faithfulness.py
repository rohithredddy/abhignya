"""Deletion-based faithfulness test. Implements 02_METHODOLOGY §6.

For each explained event and k: replace the top-k features (by |φ^med| + |φ^unc|)
with the values of a random background row and record |ΔEDCP|; repeat with k
random features (same background row). Compare with a paired one-sided
Wilcoxon signed-rank test (top-k > random-k).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from certivoi.models.quantile import QuantileForecaster
from certivoi.voi.edcp import edcp


def _edcp(fq: QuantileForecaster, X: pd.DataFrame, y_last, edges, a0) -> np.ndarray:
    return edcp(fq.predict_values(X, y_last), fq.taus, edges, a0)


def deletion_test(fq: QuantileForecaster, X: pd.DataFrame, y_last: np.ndarray,
                  edges: np.ndarray, a0: np.ndarray, importance: np.ndarray,
                  background: pd.DataFrame, ks: list[int], seed: int) -> pd.DataFrame:
    """Return rows (k, mean |ΔEDCP| top-k, mean |ΔEDCP| random-k, Wilcoxon p)."""
    rng = np.random.default_rng(seed)
    X = X[fq.feature_names].reset_index(drop=True)
    bg = background[fq.feature_names].to_numpy()
    base = _edcp(fq, X, y_last, edges, a0)
    n, p = X.shape
    donor = bg[rng.integers(0, len(bg), n)]
    order = np.argsort(-importance, axis=1)
    rows = []
    for k in ks:
        k = min(k, p)
        top_idx = order[:, :k]
        rand_idx = np.array([rng.choice(p, size=k, replace=False) for _ in range(n)])
        out = {}
        for name, idx in (("top", top_idx), ("random", rand_idx)):
            Xm = X.to_numpy(copy=True)
            r = np.arange(n)[:, None]
            Xm[r, idx] = donor[r, idx]
            e = _edcp(fq, pd.DataFrame(Xm, columns=fq.feature_names), y_last, edges, a0)
            out[name] = np.abs(e - base)
        diff = out["top"] - out["random"]
        pval = (float(wilcoxon(out["top"], out["random"], alternative="greater").pvalue)
                if np.any(diff != 0) else float("nan"))
        rows.append({"k": k, "mean_abs_delta_top": float(out["top"].mean()),
                     "mean_abs_delta_random": float(out["random"].mean()),
                     "share_top_gt_random": float(np.mean(diff > 0)), "wilcoxon_p": pval,
                     "n": n})
    return pd.DataFrame(rows)
