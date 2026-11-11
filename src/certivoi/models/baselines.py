"""Baselines B1–B5 (02_METHODOLOGY §7).

Score convention for every model: higher = more likely action-changing; an
event is flagged low-value when ``score <= lambda``.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from certivoi.models.quantile import eval_kwargs


def b1_rule_score(feats: pd.DataFrame, no_action: int, delta: float, max_hours: float
                  ) -> np.ndarray:
    """B1: 0 (skip) if a0 is the no-action category, |Δ last two| <= δ and last < max_hours.

    Missing second-last value -> rule does not fire (score 1).
    """
    skip = ((feats["a0"] == no_action) & (feats["delta_last2"].abs() <= delta)
            & (feats["hours_since_last"] < max_hours))
    return np.where(skip.fillna(False).to_numpy(), 0.0, 1.0)


class LGBMBinary:
    """LightGBM binary classifier with early stopping; constant fallback for one class."""

    def __init__(self, params: dict[str, Any], early_stopping: int) -> None:
        self.params, self.early_stopping = params, early_stopping
        self.model: lgb.LGBMClassifier | None = None
        self.const: float | None = None
        self.best_iteration: int = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray
            ) -> LGBMBinary:
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            self.const = float(y.mean()) if len(y) else 0.0
            return self
        self.model = lgb.LGBMClassifier(objective="binary", **self.params)
        self.model.fit(X, y, **eval_kwargs(X_val, np.asarray(y_val).astype(int)),
                       eval_metric="binary_logloss",
                       callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
        self.best_iteration = int(self.model.best_iteration_ or self.model.n_estimators)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self.const)
        return self.model.predict_proba(X, num_iteration=self.model.best_iteration_)[:, 1]


class Logistic:
    """B4: median imputation + standardisation + L2 logistic regression (fit on train)."""

    def __init__(self, seed: int) -> None:
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed))
        self.const: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Logistic:
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            self.const = float(y.mean()) if len(y) else 0.0
            return self
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.pipe.predict_proba(X)[:, 1]
