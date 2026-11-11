"""Conformal Risk Control for the low-value tier.

Implements 02_METHODOLOGY §5.1. Loss l_i(λ) = D_i · 1[s_i <= λ] (monotone in λ).
λ̂ = max{λ in Λ : n/(n+1) R̂_n(λ) + 1/(n+1) <= α}, Λ = unique calibration scores ∪ {0}.
Equivalent integer form: #{i : D_i = 1, s_i <= λ} + 1 <= α (n + 1).
Returns -inf when no λ is feasible (nothing is flagged).
"""

from __future__ import annotations

import numpy as np

_TOL = 1e-12


def crc_threshold(scores: np.ndarray, d: np.ndarray, alpha: float) -> float:
    """Largest λ satisfying the CRC bound on the calibration set (vectorised)."""
    s = np.asarray(scores, dtype=float)
    d = np.asarray(d, dtype=int)
    n = len(s)
    if n == 0:
        return -np.inf
    grid = np.unique(np.concatenate([s, [0.0]]))
    order = np.argsort(s, kind="stable")
    s_sorted, d_cum = s[order], np.cumsum(d[order])
    # number of positives with s <= λ for each grid value
    idx = np.searchsorted(s_sorted, grid, side="right")
    pos = np.where(idx > 0, d_cum[np.maximum(idx - 1, 0)], 0)
    feasible = pos + 1 <= alpha * (n + 1) + _TOL
    if not feasible.any():
        return -np.inf
    # risk is monotone in λ, so the feasible set is a prefix of the grid
    return float(grid[np.flatnonzero(feasible)[-1]])


def risk_and_reduction(scores: np.ndarray, d: np.ndarray, lam: float) -> tuple[float, float]:
    """Realised risk mean(D·1[s<=λ]) and reduction mean(1[s<=λ]) on an evaluation set."""
    s, d = np.asarray(scores, float), np.asarray(d, int)
    if len(s) == 0:
        return float("nan"), float("nan")
    flag = s <= lam
    return float(np.mean(d * flag)), float(np.mean(flag))
