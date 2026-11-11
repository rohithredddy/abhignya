"""Interventional TreeSHAP for the median and interval-end quantile models.

Implements 02_METHODOLOGY §6: with one shared background, φ^med = SHAP(q0.5) and
φ^unc = SHAP(q0.9) − SHAP(q0.1) (valid by linearity of SHAP in model output).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from certivoi.models.quantile import QuantileForecaster


@dataclass
class Attributions:
    phi: dict[float, np.ndarray]        # tau -> (n, p) SHAP values
    base: dict[float, float]            # tau -> expected value over background
    pred: dict[float, np.ndarray]       # tau -> model output (target scale)
    features: list[str]

    @property
    def med(self) -> np.ndarray:
        return self.phi[0.5]

    def unc(self, lo: float, hi: float) -> np.ndarray:
        return self.phi[hi] - self.phi[lo]

    def additivity_error(self) -> dict[float, float]:
        """max |sum φ + base − prediction| per tau."""
        return {t: float(np.max(np.abs(self.phi[t].sum(1) + self.base[t] - self.pred[t])))
                for t in self.phi}


def tree_shap(fq: QuantileForecaster, X: pd.DataFrame, background: pd.DataFrame,
              taus: tuple[float, ...] = (0.1, 0.5, 0.9)) -> Attributions:
    """Interventional TreeSHAP for each tau with a common background sample."""
    X = X[fq.feature_names]
    background = background[fq.feature_names]
    phi, base, pred = {}, {}, {}
    for t in taus:
        booster = fq.booster_at(t)
        expl = shap.TreeExplainer(booster, data=background,
                                  feature_perturbation="interventional", model_output="raw")
        vals = np.asarray(expl.shap_values(X, check_additivity=False))
        phi[t] = vals
        base[t] = float(np.ravel(expl.expected_value)[0])
        pred[t] = booster.predict(X)
    return Attributions(phi, base, pred, list(fq.feature_names))


def global_importance(attr: Attributions, lo: float, hi: float) -> pd.DataFrame:
    """Mean |φ^med| and mean |φ^unc| per feature."""
    return pd.DataFrame({"feature": attr.features,
                         "mean_abs_phi_med": np.abs(attr.med).mean(0),
                         "mean_abs_phi_unc": np.abs(attr.unc(lo, hi)).mean(0)})
