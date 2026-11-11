"""Patient-clustered conformal risk control.

Implements 02_METHODOLOGY §5.5. Repeat tests of one patient are dependent, so
event-level exchangeability is doubtful; patients (the split unit) are
exchangeable. Treat each calibration patient i as one unit with loss

    L_i(λ) = (1/m_i) Σ_{e in i} D_e · 1[s_e <= λ]   ∈ [0, 1], monotone in λ,

and apply CRC over the n patients:
    λ̂ = max{λ : Σ_i L_i(λ) + 1 <= α (n + 1)}.
Guarantee (CRC, B = 1): E[L_new(λ̂)] <= α, i.e. the expected fraction of a new
patient's repeat tests that are flagged action changes is at most α.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TOL = 1e-12


def _weights(groups: np.ndarray) -> tuple[np.ndarray, int]:
    codes, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv], len(codes)


def crc_threshold_clustered(scores: np.ndarray, d: np.ndarray, groups: np.ndarray,
                            alpha: float) -> float:
    """Largest λ with Σ_i L_i(λ) + 1 <= α(n+1) over patients; -inf if none."""
    s = np.asarray(scores, dtype=float)
    d = np.asarray(d, dtype=float)
    if len(s) == 0:
        return -np.inf
    w, n = _weights(np.asarray(groups))
    grid = np.unique(np.concatenate([s, [0.0]]))
    order = np.argsort(s, kind="stable")
    s_sorted, risk_cum = s[order], np.cumsum((w * d)[order])
    idx = np.searchsorted(s_sorted, grid, side="right")
    total = np.where(idx > 0, risk_cum[np.maximum(idx - 1, 0)], 0.0)
    feasible = total + 1 <= alpha * (n + 1) + _TOL
    if not feasible.any():
        return -np.inf
    return float(grid[np.flatnonzero(feasible)[-1]])


def patient_risk(scores: np.ndarray, d: np.ndarray, groups: np.ndarray, lam: float
                 ) -> tuple[float, float]:
    """(mean over patients of L_i(λ), mean over patients of their flagged fraction)."""
    if len(scores) == 0:
        return float("nan"), float("nan")
    f = np.asarray(scores, float) <= lam
    df = pd.DataFrame({"g": groups, "loss": np.asarray(d, float) * f, "flag": f})
    per = df.groupby("g")[["loss", "flag"]].mean()
    return float(per["loss"].mean()), float(per["flag"].mean())
