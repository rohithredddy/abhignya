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
        self.models: list[lgb.LGBMRegressor] = []
        self.best_iterations: list[int] = []
        self.feature_names: list[str] = []

    def _target(self, y: np.ndarray, y_last: np.ndarray) -> np.ndarray:
        return y - y_last if self.target == "delta" else y

    def fit(self, X: pd.DataFrame, y: np.ndarray, y_last: np.ndarray, X_val: pd.DataFrame,
            y_val: np.ndarray, y_last_val: np.ndarray) -> QuantileForecaster:
        self.feature_names = list(X.columns)
        z, zv = self._target(y, y_last), self._target(y_val, y_last_val)
        self.models, self.best_iterations = [], []
        for tau in self.taus:
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), **self.params)
            m.fit(X, z, **eval_kwargs(X_val, zv), eval_metric="quantile",
                  callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            self.models.append(m)
            self.best_iterations.append(int(m.best_iteration_ or m.n_estimators))
        return self

    def predict_target(self, X: pd.DataFrame) -> np.ndarray:
        """Sorted quantiles of the model target, shape (n, T)."""
        q = np.column_stack([m.predict(X, num_iteration=m.best_iteration_) for m in self.models])
        return rearrange(q)

    def predict_values(self, X: pd.DataFrame, y_last: np.ndarray) -> np.ndarray:
        """Sorted quantiles of the new value y, shape (n, T)."""
        q = self.predict_target(X)
        return q + np.asarray(y_last)[:, None] if self.target == "delta" else q

    def model_at(self, tau: float) -> lgb.LGBMRegressor:
        """The fitted model for quantile level ``tau`` (exact match)."""
        idx = int(np.flatnonzero(np.isclose(self.taus, tau))[0])
        return self.models[idx]

    def val_pinball(self, X: pd.DataFrame, y: np.ndarray, y_last: np.ndarray) -> float:
        """Mean pinball loss over all taus on the y scale."""
        v = self.predict_values(X, y_last)
        return float(np.mean([pinball(y, v[:, k], t) for k, t in enumerate(self.taus)]))
