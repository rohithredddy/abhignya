"""Phase 5: CRC / LTT / Mondrian / triage (02_METHODOLOGY §5).

AC: on exchangeable synthetic data, mean realised CRC risk over 500 resamples
<= α + small tolerance; LTT FSR <= β in >= (1 - δ) of resamples.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import betainc

from certivoi.conformal.crc import crc_threshold, risk_and_reduction
from certivoi.conformal.ltt import (
    fsr,
    ltt_grid,
    ltt_pvalues,
    ltt_threshold,
    min_rejectable_n,
)
from certivoi.conformal.mondrian import mondrian_flags, mondrian_thresholds
from certivoi.triage import HIGH, LOW, POTENTIAL, assign_tiers, high_value_threshold


def _data(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    s = rng.beta(0.6, 3.0, n)                   # skewed like EDCP: many low scores
    d = (rng.random(n) < s).astype(int)          # calibrated: P(D=1|s) = s
    return s, d


def test_crc_hand_example() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    d = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1])
    # n = 9: need #pos(s<=λ) + 1 <= α·10
    assert crc_threshold(s, d, 0.1) == 0.2      # 0 + 1 <= 1 ok up to 0.2; 0.3 adds a positive
    assert crc_threshold(s, d, 0.2) == 0.5      # 1 + 1 <= 2 up to 0.5
    assert crc_threshold(s, d, 0.05) == -np.inf  # 1 > 0.5: infeasible
    assert crc_threshold(s, np.zeros(9, int), 0.1) == 0.9


def test_crc_ties_flag_all_equal_scores() -> None:
    s = np.array([0.2, 0.2, 0.2, 0.5])
    d = np.array([0, 1, 0, 0])
    assert crc_threshold(s, d, 0.3) < 0.2       # including 0.2 brings a positive: 1+1 > 1.5


def test_crc_monotone_in_alpha() -> None:
    rng = np.random.default_rng(0)
    s, d = _data(rng, 3000)
    lams = [crc_threshold(s, d, a) for a in (0.005, 0.01, 0.02, 0.05, 0.1)]
    assert all(a <= b for a, b in zip(lams, lams[1:], strict=False))


@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_crc_validity_simulation(alpha: float) -> None:
    rng = np.random.default_rng(42)
    risks, reds = [], []
    for _ in range(500):
        s, d = _data(rng, 2000)
        lam = crc_threshold(s[:1000], d[:1000], alpha)
        r, red = risk_and_reduction(s[1000:], d[1000:], lam)
        risks.append(r)
        reds.append(red)
    assert np.mean(risks) <= alpha + 0.1 * alpha
    assert np.mean(reds) > 0  # non-trivial


def test_ltt_pvalues_and_grid() -> None:
    s = np.linspace(0, 1, 101)
    g = ltt_grid(s, 10)
    assert len(g) == 10 and (np.diff(g) > 0).all()
    assert min_rejectable_n(0.05, 0.10) == 45      # 0.95^45 = 0.099 <= 0.1 < 0.95^44
    g2 = ltt_grid(s, 10, n_min=45)
    assert (s <= g2[0]).sum() == 45
    n, k, p = ltt_pvalues(s, np.zeros(101, int), g, 0.05)
    assert (n > 0).all() and (k == 0).all()
    np.testing.assert_allclose(p, 0.95 ** n)


def _population_fsr(lam: float, a: float = 0.6, b: float = 3.0) -> float:
    """True FSR(λ) = E[s | s <= λ] for s ~ Beta(a, b), P(D=1|s) = s."""
    if not np.isfinite(lam) or lam <= 0:
        return 0.0
    return a / (a + b) * betainc(a + 1, b, lam) / betainc(a, b, lam)


def test_ltt_simulation_fsr_control() -> None:
    """LTT guarantee: P(FSR(λ̂) <= β) >= 1 - δ, with FSR the population rate."""
    beta, delta = 0.05, 0.10
    rng = np.random.default_rng(7)
    ok, flagged = [], []
    for _ in range(500):
        s, d = _data(rng, 4000)
        lam = ltt_threshold(s, d, beta, delta, grid_size=100)
        ok.append(_population_fsr(lam) <= beta)
        flagged.append(np.mean(s <= lam))
    assert np.mean(ok) >= 1 - delta
    assert np.mean(flagged) > 0.05


def test_fsr_empirical() -> None:
    assert fsr(np.array([0.1, 0.2, 0.9]), np.array([0, 1, 1]), 0.5) == 0.5
    assert np.isnan(fsr(np.array([0.6]), np.array([1]), 0.5))


def test_ltt_returns_neg_inf_when_first_hypothesis_fails() -> None:
    s = np.linspace(0, 1, 50)
    assert ltt_threshold(s, np.ones(50, int), 0.05, 0.1, grid_size=10) == -np.inf


def test_mondrian_groupwise() -> None:
    rng = np.random.default_rng(3)
    s, d = _data(rng, 4000)
    g = np.where(rng.random(4000) < 0.5, "a", "b")
    d[(g == "b") & (s < 0.2)] = 1                 # group b is riskier at low scores
    thr = mondrian_thresholds(s, d, g, 0.02)
    assert thr["b"] < thr["a"]
    flags = mondrian_flags(np.array([0.0, 0.0]), np.array(["a", "zzz"]), thr)
    assert list(flags) == [True, False]


def test_high_value_threshold_and_tiers() -> None:
    s = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])
    d = np.array([0, 0, 0, 1, 0, 1])
    # s>=0.9: 1/1; >=0.8: 1/2; >=0.7: 2/3; >=0.6: 2/4 -> smallest with prec>=0.5 is 0.6
    assert high_value_threshold(s, d, 0.5) == 0.6
    assert high_value_threshold(s, np.zeros(6, int), 0.5) == np.inf
    tiers = assign_tiers(np.array([0.05, 0.3, 0.95]), lam=0.1, tau_h=0.6)
    assert list(tiers) == [LOW, POTENTIAL, HIGH]
