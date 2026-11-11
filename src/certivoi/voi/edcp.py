"""EDCP: expected decision-change probability from predicted quantiles.

Implements 02_METHODOLOGY §3 (rearrangement, piecewise-linear CDF with point
masses tau_min / 1 - tau_max at the extreme quantiles) and §4 (EDCP, utility VOI,
isotonic calibration). Pure, vectorised functions.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def rearrange(q: np.ndarray) -> np.ndarray:
    """Monotone rearrangement: sort predicted quantiles per row (non-crossing)."""
    return np.sort(np.asarray(q, dtype=float), axis=1)


def cdf_lt(values: np.ndarray, taus: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P(Y < y) under the piecewise-linear CDF through (values[:,k], taus[k]).

    Inputs: sorted quantile values (n, T), quantile levels (T,), thresholds (n,)
    (±inf allowed). Below the smallest quantile P = 0, above the largest P = 1,
    so mass tau_1 sits at values[:,0] and 1 - tau_T at values[:,-1].
    """
    v = np.asarray(values, dtype=float)
    taus = np.asarray(taus, dtype=float)
    y = np.asarray(y, dtype=float)
    n, t = v.shape
    k = (v < y[:, None]).sum(axis=1)
    out = np.zeros(n)
    out[k == t] = 1.0
    mid = (k > 0) & (k < t)
    if mid.any():
        rows = np.flatnonzero(mid)
        i = k[mid]
        lo, hi = v[rows, i - 1], v[rows, i]
        frac = (y[mid] - lo) / (hi - lo)  # lo < y <= hi, so hi > lo
        out[mid] = taus[i - 1] + frac * (taus[i] - taus[i - 1])
    return out


def category_probs(values: np.ndarray, taus: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Probability of each action category, p[:, c] = P(edges[c] <= Y < edges[c+1])."""
    g = np.column_stack([cdf_lt(values, taus, edges[:, j]) for j in range(edges.shape[1])])
    return np.clip(np.diff(g, axis=1), 0.0, 1.0)


def edcp(values: np.ndarray, taus: np.ndarray, edges: np.ndarray, a0: np.ndarray) -> np.ndarray:
    """EDCP = 1 - P(new value stays in the default category a0)  (02_METHODOLOGY §4)."""
    p = category_probs(values, taus, edges)
    a0 = np.asarray(a0, dtype=int)
    return np.clip(1.0 - p[np.arange(len(a0)), a0], 0.0, 1.0)


def voi_utility(probs: np.ndarray, a0: np.ndarray, utility: np.ndarray) -> np.ndarray:
    """Single-step VOI with utility matrix U[a, c]: sum_c p(c) [max_a U(a,c) - U(a0,c)].

    With U = -1[a != c] (0-1 regret) this equals EDCP (Proposition, §4).
    """
    utility = np.asarray(utility, dtype=float)
    best = utility.max(axis=0)                      # (K,)
    loss = best[None, :] - utility[np.asarray(a0, dtype=int)]  # (n, K)
    return (probs * loss).sum(axis=1)


class IsotonicCalibrator:
    """Isotonic map from raw score to P(D=1); fit on the validation split only."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, score: np.ndarray, d: np.ndarray) -> IsotonicCalibrator:
        self._iso.fit(np.asarray(score, float), np.asarray(d, float))
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(score, float))
