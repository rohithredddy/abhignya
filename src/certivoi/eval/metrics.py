"""Discrimination / calibration metrics and patient-level bootstrap CIs.

Implements 04_EXPERIMENT_PLAN E2 (AUROC, AUPRC, Brier, ECE with 15 bins,
95% CIs by patient-level bootstrap).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(y)) if len(y) else float("nan")


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def metric_fns(n_bins: int) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {"auroc": safe_auroc, "auprc": safe_auprc, "brier": brier,
            "ece": lambda y, p: ece(y, p, n_bins)}


def bootstrap_ci(y: np.ndarray, scores: dict[str, np.ndarray], groups: np.ndarray,
                 fns: dict[str, Callable], reps: int, seed: int, level: float = 0.95
                 ) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Point estimate and percentile CI per (score, metric), resampling patients.

    Inputs: labels, {model: scores}, patient id per row, metric functions,
    replicates, seed. Output: {model: {metric: (estimate, lo, hi)}}.
    """
    y = np.asarray(y)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = np.split(np.argsort(inv, kind="stable"),
                             np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    draws = {m: {k: [] for k in fns} for m in scores}
    for _ in range(reps):
        g = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[i] for i in g])
        for m, s in scores.items():
            for k, f in fns.items():
                draws[m][k].append(f(y[idx], s[idx]))
    a = (1 - level) / 2
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for m, s in scores.items():
        out[m] = {}
        for k, f in fns.items():
            d = np.asarray(draws[m][k], float)
            d = d[~np.isnan(d)]
            lo, hi = (np.quantile(d, [a, 1 - a]) if len(d) else (np.nan, np.nan))
            out[m][k] = (f(y, s), float(lo), float(hi))
    return out


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean predicted, observed rate, count) per non-empty equal-width bin."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    mp, obs, cnt = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            mp.append(p[m].mean())
            obs.append(y[m].mean())
            cnt.append(int(m.sum()))
    return np.array(mp), np.array(obs), np.array(cnt)
