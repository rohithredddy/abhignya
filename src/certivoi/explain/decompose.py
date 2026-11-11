"""Margin vs. uncertainty decomposition. Implements 02_METHODOLOGY §6.

ŷ = median forecast; margin m = signed distance from ŷ to the nearest boundary
of a0's interval (positive = inside); width w = q0.9 − q0.1; ρ = m / w.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_boundary(y_hat: np.ndarray, edges: np.ndarray, a0: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(signed margin, nearest finite boundary of a0's interval) per event."""
    a0 = np.asarray(a0, dtype=int)
    idx = np.arange(len(a0))
    lo, hi = edges[idx, a0], edges[idx, a0 + 1]
    d_lo, d_hi = y_hat - lo, hi - y_hat          # +inf where the side is open
    use_lo = d_lo <= d_hi
    margin = np.where(use_lo, d_lo, d_hi)
    boundary = np.where(use_lo, lo, hi)
    return margin, boundary


def decompose(q_lo: np.ndarray, q_med: np.ndarray, q_hi: np.ndarray, edges: np.ndarray,
              a0: np.ndarray) -> pd.DataFrame:
    """Per-event decomposition. Inputs are quantiles of the new value (y scale)."""
    margin, boundary = nearest_boundary(np.asarray(q_med, float), edges, a0)
    width = np.asarray(q_hi, float) - np.asarray(q_lo, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(width > 0, margin / width, np.sign(margin) * np.inf)
    return pd.DataFrame({"y_hat": q_med, "q_lo": q_lo, "q_hi": q_hi, "margin": margin,
                         "boundary": boundary, "width": width, "rho": rho})
