"""Clustered CRC, panel certification and audited online risk control (02 §5.5–5.7)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from certivoi.conformal.clustered import crc_threshold_clustered, patient_risk
from certivoi.conformal.crc import crc_threshold
from certivoi.conformal.online import (
    estimate_bound,
    online_risk_control,
    rank_transform,
    static_result,
)
from certivoi.conformal.panel import naive_flags, pair_panel, panel_score


# ---------- clustered CRC ----------
def test_clustered_equals_crc_with_singletons() -> None:
    rng = np.random.default_rng(0)
    s, d = rng.random(300), (rng.random(300) < 0.2).astype(int)
    for a in (0.01, 0.05, 0.1):
        assert crc_threshold_clustered(s, d, np.arange(300), a) == crc_threshold(s, d, a)


def test_clustered_brute_force() -> None:
    rng = np.random.default_rng(1)
    s = rng.random(40)
    d = (rng.random(40) < 0.3).astype(int)
    g = rng.integers(0, 12, 40)
    n = len(np.unique(g))
    a = 0.15
    best = -np.inf
    for lam in np.unique(np.concatenate([s, [0.0]])):
        tot = sum((d[g == k] * (s[g == k] <= lam)).mean() for k in np.unique(g))
        if tot + 1 <= a * (n + 1) + 1e-12:
            best = max(best, lam)
    assert crc_threshold_clustered(s, d, g, a) == best


def _clustered_population(rng, n_pat):
    """Patients with random effects: heavy within-patient dependence, varied m_i."""
    m = rng.integers(1, 30, n_pat)
    g = np.repeat(np.arange(n_pat), m)
    z = np.repeat(rng.normal(0, 1.5, n_pat), m)
    x = z + rng.normal(0, 0.5, len(g))
    p = 1 / (1 + np.exp(-(x - 2.5)))
    d = (rng.random(len(g)) < p).astype(int)
    s = 1 / (1 + np.exp(-(x - 2.5 + rng.normal(0, 0.7, len(g)))))
    return s, d, g


def test_clustered_validity_simulation() -> None:
    """Mean patient-level risk over resamples <= α (1 + 0.1)."""
    rng = np.random.default_rng(2)
    for a in (0.02, 0.05):
        risks = []
        for _ in range(300):
            s, d, g = _clustered_population(rng, 400)
            cal = g < 200
            lam = crc_threshold_clustered(s[cal], d[cal], g[cal], a)
            risks.append(patient_risk(s[~cal], d[~cal], g[~cal], lam)[0])
        assert np.mean(risks) <= a * 1.1, (a, np.mean(risks))


# ---------- panel ----------
def test_pair_panel_inner_join() -> None:
    a = pd.DataFrame({"subject_id": [1, 1, 2], "hadm_id": [10, 10, 20],
                      "t": pd.to_datetime(["2100-01-01 08:00", "2100-01-02 08:00",
                                           "2100-01-01 09:00"]),
                      "split": ["test"] * 3, "score_ours": [0.1, 0.2, 0.3], "D": [0, 1, 0]})
    b = a.iloc[[0, 2]].assign(score_ours=[0.5, 0.05], D=[1, 0])
    p = pair_panel({"HGB": a, "PLT": b})
    assert len(p) == 2
    assert p["D_panel"].tolist() == [1, 0]
    assert np.allclose(panel_score(p[["s_HGB", "s_PLT"]].to_numpy(), "max"), [0.5, 0.3])
    assert np.allclose(panel_score(np.array([[0.5, 0.5]]), "noisy_or"), [0.75])


def test_panel_crc_valid_naive_not() -> None:
    """Disjoint action changes in two labs: naive per-lab α certificates miss ~2α."""
    rng = np.random.default_rng(3)
    a, naive, panel = 0.05, [], []
    for _ in range(300):
        n = 4000
        which = rng.integers(0, 2, n)          # which lab can change
        d = (rng.random(n) < 0.15).astype(int)
        d1, d2 = d * (which == 0), d * (which == 1)
        s1 = np.where(which == 0, rng.random(n), rng.random(n) * 0.2)
        s2 = np.where(which == 1, rng.random(n), rng.random(n) * 0.2)
        df = pd.DataFrame({"s_A": s1, "s_B": s2, "D_A": d1, "D_B": d2,
                           "D_panel": np.maximum(d1, d2)})
        cal, ev = df.iloc[: n // 2], df.iloc[n // 2:]
        f = naive_flags(cal, ev, ["A", "B"], a)
        naive.append(float((ev["D_panel"].to_numpy() * f).mean()))
        sp_c = panel_score(cal[["s_A", "s_B"]].to_numpy(), "max")
        sp_e = panel_score(ev[["s_A", "s_B"]].to_numpy(), "max")
        lam = crc_threshold(sp_c, cal["D_panel"].to_numpy(), a)
        panel.append(float((ev["D_panel"].to_numpy() * (sp_e <= lam)).mean()))
    assert np.mean(panel) <= a * 1.1
    assert np.mean(naive) > a * 1.3          # union-bound violation shows up


# ---------- online ----------
def test_rank_transform_monotone() -> None:
    f = rank_transform(np.array([0.1, 0.2, 0.3, 0.4]))
    assert f(np.array([0.05, 0.2, 0.35, 1.0])).tolist() == [0.0, 0.5, 0.75, 1.0]


def test_online_deterministic_bound_adversarial() -> None:
    """Bound on mean IPW loss holds for adversarial streams and any audit outcome."""
    rng = np.random.default_rng(4)
    T = 3000
    streams = {
        "positives_late": (rng.random(T), (np.arange(T) > 2000).astype(int)),
        "all_positive": (rng.random(T), np.ones(T, int)),
        "low_scores_positive": (np.sort(rng.random(T)), (rng.random(T) < 0.5).astype(int)),
    }
    for (name, (u, d)), p, eta in itertools.product(streams.items(), (1.0, 0.2, 0.1),
                                                    (0.001, 0.01)):
        r = online_risk_control(u, d, 0.05, eta, p, 0.3, np.random.default_rng(5))
        assert r.loss_hat.mean() <= estimate_bound(0.05, eta, p, 0.3, T) + 1e-12, (name, p)


def test_online_tracks_shift_static_fails() -> None:
    """Abrupt shift: static CRC threshold violates α; audited online control does not."""
    rng = np.random.default_rng(6)
    a, T = 0.05, 20000
    cal_s = rng.random(5000)
    cal_d = (rng.random(5000) < 0.3 * cal_s).astype(int)  # P(D) grows with score
    f = rank_transform(cal_s)
    lam_cal = crc_threshold(cal_s, cal_d, a)
    lam1 = float(f(np.array([lam_cal]))[0])
    s = rng.random(T)
    shift = np.arange(T) >= T // 4                        # later: changes 3x more likely
    d = (rng.random(T) < np.where(shift, 0.9, 0.3) * s).astype(int)
    u = f(s)
    stat = static_result(u, d, lam1)
    misses = [online_risk_control(u, d, a, 0.002, 0.2, lam1, np.random.default_rng(k)).miss_rate
              for k in range(20)]
    assert stat.miss_rate > 1.5 * a
    # finite-T bound (transient from the warm start) + small martingale slack
    bound = estimate_bound(a, 0.002, 0.2, lam1, T)
    assert np.mean(misses) <= bound + 0.003, (np.mean(misses), bound)
    assert bound < stat.miss_rate
