"""Mondrian (group-conditional) CRC. Implements 02_METHODOLOGY §5.3."""

from __future__ import annotations

import numpy as np

from certivoi.conformal.crc import crc_threshold


def mondrian_thresholds(scores: np.ndarray, d: np.ndarray, groups: np.ndarray, alpha: float
                        ) -> dict[str, float]:
    """CRC threshold computed separately within each calibration group."""
    s, d, g = np.asarray(scores, float), np.asarray(d, int), np.asarray(groups).astype(str)
    return {k: crc_threshold(s[g == k], d[g == k], alpha) for k in np.unique(g)}


def mondrian_flags(scores: np.ndarray, groups: np.ndarray, thresholds: dict[str, float]
                   ) -> np.ndarray:
    """Low-value flags using each event's group threshold (unseen group -> never flagged)."""
    s, g = np.asarray(scores, float), np.asarray(groups).astype(str)
    lam = np.array([thresholds.get(k, -np.inf) for k in g])
    return s <= lam
