"""Three-tier triage. Implements 02_METHODOLOGY §5.4.

- low-value: s <= λ̂_α (certified by CRC on calibration data)
- high-value: s >= τ_H, τ_H chosen on validation so precision(D=1) >= target
- potentially useful: otherwise
"""

from __future__ import annotations

import numpy as np

LOW, POTENTIAL, HIGH = "low_value", "potentially_useful", "high_value"


def high_value_threshold(s_val: np.ndarray, d_val: np.ndarray, min_precision: float) -> float:
    """Smallest τ with precision of D=1 among {s >= τ} >= min_precision (inf if none).

    Choosing the smallest qualifying τ maximises the size of the high-value tier.
    """
    s, d = np.asarray(s_val, float), np.asarray(d_val, int)
    if len(s) == 0:
        return np.inf
    order = np.argsort(-s, kind="stable")
    s_desc, d_cum = s[order], np.cumsum(d[order])
    uniq = np.unique(s)[::-1]                      # candidate τ, descending
    n_ge = np.searchsorted(-s_desc, -uniq, side="right")
    prec = d_cum[n_ge - 1] / n_ge
    ok = np.flatnonzero(prec >= min_precision)
    return float(uniq[ok].min()) if len(ok) else np.inf


def assign_tiers(s: np.ndarray, lam: float, tau_h: float) -> np.ndarray:
    """Tier label per event; the certified low-value tier takes precedence."""
    s = np.asarray(s, float)
    out = np.full(len(s), POTENTIAL, dtype=object)
    out[s >= tau_h] = HIGH
    out[s <= lam] = LOW
    return out
