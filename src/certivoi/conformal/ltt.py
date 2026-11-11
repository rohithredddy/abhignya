"""Learn-then-Test control of the false-skip rate among flagged tests.

Implements 02_METHODOLOGY §5.2. FSR(λ) = P(D=1 | s <= λ). For an increasing grid
λ_1 < λ_2 < ..., test H_j: FSR(λ_j) > β with the exact binomial p-value
p_j = P(Bin(n_j, β) <= k_j) (equivalently the Clopper–Pearson upper bound < β),
where n_j = #flagged and k_j = #positives among flagged on calibration data.
Fixed-sequence testing from the smallest λ; stop at the first non-rejection.
With probability >= 1 - δ, FSR(λ̂) <= β.

Grid: ``grid_size`` calibration-score quantiles whose flagged counts run evenly
from n_min to n, with n_min = ceil(log δ / log(1 - β)) — the smallest flagged
count for which H_j can be rejected at all (zero positives gives p = (1-β)^n).
The grid depends only on scores, β and δ, never on labels. A grid of every
unique score would start with n_1 = 1, whose p-value 1 - β is never <= δ, so
fixed-sequence testing would always stop at once.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def min_rejectable_n(beta: float, delta: float) -> int:
    """Smallest n for which P(Bin(n, β) <= 0) = (1-β)^n <= δ."""
    return int(np.ceil(np.log(delta) / np.log1p(-beta)))


def ltt_grid(scores: np.ndarray, grid_size: int, n_min: int = 1) -> np.ndarray:
    """Increasing λ grid: sorted scores at flagged counts spaced from n_min to n."""
    s = np.sort(np.asarray(scores, float))
    if len(s) < n_min or len(s) == 0:
        return np.array([])
    counts = np.unique(np.round(np.linspace(n_min, len(s), grid_size)).astype(int))
    return np.unique(s[counts - 1])


def ltt_pvalues(scores: np.ndarray, d: np.ndarray, grid: np.ndarray, beta: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_flagged, k_positive, p-value) for each grid λ."""
    s, d = np.asarray(scores, float), np.asarray(d, int)
    order = np.argsort(s, kind="stable")
    s_sorted, d_cum = s[order], np.cumsum(d[order])
    n = np.searchsorted(s_sorted, grid, side="right")
    k = np.where(n > 0, d_cum[np.maximum(n - 1, 0)], 0)
    p = np.where(n > 0, binom.cdf(k, np.maximum(n, 1), beta), 1.0)
    return n, k, p


def ltt_threshold(scores: np.ndarray, d: np.ndarray, beta: float, delta: float,
                  grid_size: int = 100) -> float:
    """Fixed-sequence LTT threshold; -inf if the first hypothesis is not rejected."""
    grid = ltt_grid(scores, grid_size, min_rejectable_n(beta, delta))
    if len(grid) == 0:
        return -np.inf
    _, _, p = ltt_pvalues(scores, d, grid, beta)
    fail = np.flatnonzero(p > delta)
    last_ok = (fail[0] if len(fail) else len(grid)) - 1
    return float(grid[last_ok]) if last_ok >= 0 else -np.inf


def fsr(scores: np.ndarray, d: np.ndarray, lam: float) -> float:
    """Realised false-skip rate among flagged events (NaN if none flagged)."""
    s, d = np.asarray(scores, float), np.asarray(d, int)
    flag = s <= lam
    return float(d[flag].mean()) if flag.any() else float("nan")
