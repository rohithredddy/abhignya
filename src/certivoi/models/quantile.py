"""Per-quantile LightGBM forecaster with early stopping on validation.

Implements 02_METHODOLOGY §3. CPU only; threads from ``compute.threads``.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from certivoi.voi.edcp import rearrange

_LGB_VERSION = tuple(int(x) for x in lgb.__version__.split(".")[:2])


def eval_kwargs(X_val: Any, y_val: Any) -> dict[str, Any]:
    """Validation-set kwargs for LightGBM ``fit`` (``eval_set`` is deprecated in 4.7)."""
    if _LGB_VERSION >= (4, 7):
        return {"eval_X": (X_val,), "eval_y": (y_val,)}
    return {"eval_set": [(X_val, y_val)]}


_DEFAULT_LEFT_BIT = 2        # LightGBM decision_type bit 1: missing values go left
_MISSING_TYPE_SHIFT = 2      # bits 2-3: 0 = None, 1 = Zero, 2 = NaN


def float32_floor(t: float) -> float:
    """Largest float32 value <= t (returned as a Python float)."""
    d = np.float32(t)
    if float(d) > t:
        d = np.nextafter(d, np.float32(-np.inf))
    return float(d)


def make_shap_consistent(booster: lgb.Booster) -> lgb.Booster:
    """Rewrite a LightGBM booster so TreeSHAP routes samples exactly like LightGBM.

    Two mismatches break SHAP additivity (02 §6), both fixed without changing
    predictions on float32-exact features (features/build.py):

    1. LightGBM compares ``x <= t`` with float64 thresholds; TreeSHAP's C code
       rounds t to float32, which can move t past a float32 data value. Each t is
       replaced by its float32 floor d: for float32 x, ``x <= t`` iff ``x <= d``.
    2. For splits with missing type None (no NaN seen in training) LightGBM maps
       NaN to 0, whereas TreeSHAP sends NaN to the default child. The default
       direction is set to where 0 goes.

    Assumes numeric splits only (no categorical features in this project).
    """
    # drop the per-tree byte-size header: edited trees change length and LightGBM
    # parses trees sequentially when the header is absent
    lines = [ln for ln in booster.model_to_string().splitlines()
             if not ln.startswith("tree_sizes=")]
    thresholds: list[float] = []
    for i, line in enumerate(lines):
        if line.startswith("threshold="):
            thresholds = [float32_floor(float(v)) for v in line[len("threshold="):].split()]
            lines[i] = "threshold=" + " ".join(repr(v) for v in thresholds)
        elif line.startswith("decision_type="):
            types = [int(v) for v in line[len("decision_type="):].split()]
            for j, dt in enumerate(types):
                if (dt >> _MISSING_TYPE_SHIFT) & 3 == 0:          # missing type None
                    left = 0.0 <= thresholds[j]
                    types[j] = (dt | _DEFAULT_LEFT_BIT) if left else (dt & ~_DEFAULT_LEFT_BIT)
            lines[i] = "decision_type=" + " ".join(str(v) for v in types)
    return lgb.Booster(model_str="\n".join(lines) + "\n")


def lgb_params(cfg_lgb: dict[str, Any], threads: int, seed: int) -> dict[str, Any]:
    """LightGBM constructor kwargs from config (early stopping handled separately)."""
    p = {k: v for k, v in cfg_lgb.items() if k != "early_stopping_rounds"}
    return {**p, "n_jobs": threads, "random_state": seed, "verbose": -1}


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss."""
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


class QuantileForecaster:
    """One LightGBM regressor per tau; predicts sorted quantiles of the target.

    The target is ``y - y_last`` (``target='delta'``) or ``y`` (``'absolute'``);
    :meth:`predict_values` always returns quantiles of the new value y.
    """

    def __init__(self, taus: list[float], params: dict[str, Any], early_stopping: int,
                 target: str = "delta") -> None:
        if target not in ("delta", "absolute"):
            raise ValueError(f"target must be delta|absolute, got {target}")
        self.taus = np.asarray(taus, dtype=float)
        self.params = params
        self.early_stopping = early_stopping
        self.target = target
        self.models: list[lgb.Booster] = []
        self.best_iterations: list[int] = []
        self.feature_names: list[str] = []

    def _target(self, y: np.ndarray, y_last: np.ndarray) -> np.ndarray:
        return y - y_last if self.target == "delta" else y

    def fit(self, X: pd.DataFrame, y: np.ndarray, y_last: np.ndarray, X_val: pd.DataFrame,
            y_val: np.ndarray, y_last_val: np.ndarray) -> QuantileForecaster:
        """Fit one model per tau; each booster is truncated to its best iteration.

        Truncation makes the stored booster equal to what is used for prediction,
        so TreeSHAP attributions add up to the predicted quantile (02 §6).
        """
        self.feature_names = list(X.columns)
        z, zv = self._target(y, y_last), self._target(y_val, y_last_val)
        self.models, self.best_iterations = [], []
        for tau in self.taus:
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), **self.params)
            m.fit(X, z, **eval_kwargs(X_val, zv), eval_metric="quantile",
                  callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            best = int(m.best_iteration_ or m.n_estimators)
            booster = make_shap_consistent(lgb.Booster(model_str=m.booster_.model_to_string(
                num_iteration=best)))
            ref = m.predict(X_val, num_iteration=best)
            if not np.allclose(booster.predict(X_val[self.feature_names]), ref, atol=1e-9):
                raise RuntimeError("SHAP-consistency rewrite changed predictions")
            self.models.append(booster)
            self.best_iterations.append(best)
        return self

    def predict_target(self, X: pd.DataFrame) -> np.ndarray:
        """Sorted quantiles of the model target, shape (n, T)."""
        X = X[self.feature_names]
        return rearrange(np.column_stack([b.predict(X) for b in self.models]))

    def predict_values(self, X: pd.DataFrame, y_last: np.ndarray) -> np.ndarray:
        """Sorted quantiles of the new value y, shape (n, T)."""
        q = self.predict_target(X)
        return q + np.asarray(y_last)[:, None] if self.target == "delta" else q

    def booster_at(self, tau: float) -> lgb.Booster:
        """The (truncated) booster for quantile level ``tau`` (exact match)."""
        idx = int(np.flatnonzero(np.isclose(self.taus, tau))[0])
        return self.models[idx]

    def val_pinball(self, X: pd.DataFrame, y: np.ndarray, y_last: np.ndarray) -> float:
        """Mean pinball loss over all taus on the y scale."""
        v = self.predict_values(X, y_last)
        return float(np.mean([pinball(y, v[:, k], t) for k, t in enumerate(self.taus)]))
