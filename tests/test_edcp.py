"""Phase 4: EDCP math (02_METHODOLOGY §3–§4)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from certivoi.voi.edcp import (
    IsotonicCalibrator,
    category_probs,
    cdf_lt,
    edcp,
    rearrange,
    voi_utility,
)

TAUS13 = np.array([0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98])


def test_rearrange_sorts_rows() -> None:
    q = np.array([[1.0, 0.5, 2.0], [3.0, 2.0, 1.0]])
    assert (np.diff(rearrange(q), axis=1) >= 0).all()


def test_cdf_lt_tails_and_interpolation() -> None:
    v = np.array([[0.0, 1.0, 2.0]] * 5)
    taus = np.array([0.1, 0.5, 0.9])
    y = np.array([-1.0, 0.0, 0.5, 2.0, 2.5])
    np.testing.assert_allclose(cdf_lt(v, taus, y), [0.0, 0.0, 0.3, 0.9, 1.0])
    assert cdf_lt(v[:1], taus, np.array([np.inf]))[0] == 1.0
    assert cdf_lt(v[:1], taus, np.array([-np.inf]))[0] == 0.0


def test_cdf_lt_with_tied_quantiles() -> None:
    v = np.array([[1.0, 1.0, 1.0]])
    taus = np.array([0.1, 0.5, 0.9])
    assert cdf_lt(v, taus, np.array([1.0]))[0] == 0.0
    assert cdf_lt(v, taus, np.array([1.0001]))[0] == 1.0


@pytest.mark.parametrize("mu,sd,edges,a0,lo,hi", [
    (0, 1, [-np.inf, -1, 1, np.inf], 1, -1, 1),
    (7.5, 0.6, [-np.inf, 7.0, 8.0, np.inf], 1, 7.0, 8.0),
    (2, 1, [-np.inf, 3, np.inf], 1, 3, np.inf),          # open upper category
    (0, 2, [-np.inf, -1, np.inf], 0, -np.inf, -1),       # open lower category
])
def test_edcp_matches_gaussian_analytic(mu, sd, edges, a0, lo, hi) -> None:
    taus = np.linspace(0.0005, 0.9995, 1999)
    v = (mu + sd * norm.ppf(taus))[None, :]
    expected = 1 - (norm.cdf(hi, mu, sd) - norm.cdf(lo, mu, sd))
    got = edcp(v, taus, np.array([edges], dtype=float), np.array([a0]))[0]
    assert got == pytest.approx(expected, abs=1.5e-3)


def test_edcp_with_config_quantiles_close_to_gaussian() -> None:
    v = norm.ppf(TAUS13)[None, :]
    got = edcp(v, TAUS13, np.array([[-np.inf, -1.0, 1.0, np.inf]]), np.array([1]))[0]
    assert got == pytest.approx(1 - (norm.cdf(1) - norm.cdf(-1)), abs=0.02)


def test_edcp_in_unit_interval_and_probs_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    v = rearrange(rng.normal(size=(500, 13)) * 2 + 7)
    edges = np.tile([-np.inf, 7.0, 8.0, np.inf], (500, 1))
    p = category_probs(v, TAUS13, edges)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    e = edcp(v, TAUS13, edges, rng.integers(0, 3, 500))
    assert ((e >= 0) & (e <= 1)).all()


def test_voi_utility_equals_edcp_under_zero_one_regret() -> None:
    rng = np.random.default_rng(1)
    v = rearrange(rng.normal(size=(200, 13)) + 7.5)
    edges = np.tile([-np.inf, 7.0, 8.0, np.inf], (200, 1))
    a0 = rng.integers(0, 3, 200)
    p = category_probs(v, TAUS13, edges)
    u = -(1 - np.eye(3))
    np.testing.assert_allclose(voi_utility(p, a0, u), edcp(v, TAUS13, edges, a0), atol=1e-12)


def test_isotonic_calibrator_monotone() -> None:
    rng = np.random.default_rng(2)
    s = rng.random(2000)
    d = (rng.random(2000) < s).astype(int)
    c = IsotonicCalibrator().fit(s, d)
    out = c.predict(np.linspace(-1, 2, 50))
    assert (np.diff(out) >= -1e-12).all() and out.min() >= 0 and out.max() <= 1
