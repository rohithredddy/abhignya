"""Phase 2: action maps and labels with hand-computed boundary cases (02_METHODOLOGY §2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from certivoi.config import Config
from certivoi.labels.actions import (
    KdigoParams,
    action_map,
    category_from_edges,
    kdigo_edges,
    kdigo_stage,
)
from certivoi.labels.build import label_events

P = KdigoParams()


@pytest.mark.parametrize("y,cat", [(6.9, 0), (7.0, 1), (7.9, 1), (8.0, 2), (12.0, 2)])
def test_hgb_boundaries(cfg: Config, y: float, cat: int) -> None:
    assert action_map(cfg.get("actions"), "HGB").category(np.array([y]))[0] == cat


@pytest.mark.parametrize("y,cat", [(3.4, 0), (3.5, 1), (4.2, 1), (5.5, 1), (5.6, 2)])
def test_k_boundaries(cfg: Config, y: float, cat: int) -> None:
    assert action_map(cfg.get("actions"), "K").category(np.array([y]))[0] == cat


@pytest.mark.parametrize("y,cat", [(9.0, 0), (10.0, 1), (49.0, 1), (50.0, 2)])
def test_plt_boundaries(cfg: Config, y: float, cat: int) -> None:
    assert action_map(cfg.get("actions"), "PLT").category(np.array([y]))[0] == cat


def test_no_action_index(cfg: Config) -> None:
    a = cfg.get("actions")
    assert [action_map(a, lab).no_action_index for lab in ("HGB", "K", "PLT", "CREAT")] == \
        [2, 1, 2, 0]


@pytest.mark.parametrize("y,b48,b7,stage", [
    (1.3, 1.0, 1.0, 1),            # +0.3 within 48 h (float 1.0 + 0.3 != 1.3)
    (1.29, 1.0, 1.0, 0),
    (1.3, np.nan, 1.0, 0),         # no 48 h baseline, ratio 1.3 < 1.5
    (1.5, np.nan, 1.0, 1),         # 1.5x
    (1.2, np.nan, 0.8, 1),         # 1.5 * 0.8 = 1.2000000000000002
    (2.0, 1.9, 1.0, 2),            # 2x
    (3.0, 2.9, 1.0, 3),            # 3x
    (4.0, 3.7, 3.5, 3),            # >= 4.0 with acute rise (b48 <= 3.7)
    (4.1, 3.8, 3.5, 1),            # >= 4.0 but b48 > 3.7 and < 1.5x b7 -> only +0.3 rule
    (4.0, np.nan, np.nan, 0),      # no baseline at all
])
def test_kdigo_stage_cases(y, b48, b7, stage) -> None:
    assert kdigo_stage(np.array([y]), np.array([b48]), np.array([b7]), P)[0] == stage


def test_kdigo_edges_match_direct_rule() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    b7 = rng.uniform(0.3, 6, n)
    b48 = b7 + rng.uniform(0, 2, n)
    b48[rng.random(n) < 0.2] = np.nan
    b7[rng.random(n) < 0.05] = np.nan
    y = np.round(rng.uniform(0.2, 15, n), 1)
    direct = kdigo_stage(y, b48, b7, P)
    via_edges = category_from_edges(y, kdigo_edges(b48, b7, P))
    assert (direct == via_edges).all()


def _creat_case(hours_between: float) -> pd.DataFrame:
    t0 = pd.Timestamp("2150-01-01 08:00")
    hist = pd.DataFrame({"labevent_id": [1, 2], "hadm_id": [9, 9],
                         "charttime": [t0, t0 + pd.Timedelta(hours=hours_between)],
                         "valuenum": [1.0, 1.3]})
    ev = pd.DataFrame({"event_id": [2], "subject_id": [5], "hadm_id": [9],
                       "t": [hist["charttime"][1]], "y": [1.3], "ref_lo": [0.5],
                       "ref_hi": [1.2]})
    return label_events(ev, hist, action_map({"CREAT": {"rule": "kdigo", "abs_rise_48h": 0.3,
                        "ratio_7d": [1.5, 2.0, 3.0], "stage3_abs": 4.0,
                        "names": ["no_aki", "aki_1", "aki_2", "aki_3"]}}, "CREAT"), 0.2, 1)


def test_creatinine_rise_47h_vs_49h() -> None:
    at47, at49 = _creat_case(47).iloc[0], _creat_case(49).iloc[0]
    assert at47["cat"] == 1 and at47["a0"] == 0 and at47["D"] == 1
    assert np.isnan(at49["b48"]) and at49["b7"] == 1.0
    assert at49["cat"] == 0 and at49["D"] == 0
    # edges at 47 h: stage 1 starts at 1.3
    assert at47["edge_1"] == pytest.approx(1.3)


def _hgb_events(times_min: list[float], values: list[float], cfg: Config) -> pd.DataFrame:
    t0 = pd.Timestamp("2150-01-01 08:00")
    ct = [t0 + pd.Timedelta(minutes=m) for m in times_min]
    hist = pd.DataFrame({"labevent_id": range(1, len(ct) + 1), "hadm_id": 9, "charttime": ct,
                         "valuenum": values})
    ev = pd.DataFrame({"event_id": [len(ct)], "subject_id": [5], "hadm_id": [9],
                       "t": [ct[-1]], "y": [values[-1]], "ref_lo": [12.0], "ref_hi": [16.0]})
    return label_events(ev, hist, action_map(cfg.get("actions"), "HGB"), 0.5, 1)


def test_y_last_ignores_values_within_gap(cfg: Config) -> None:
    # value 30 s before t is the same specimen -> must not be y_last
    r = _hgb_events([0, 600 - 0.5, 600], [9.0, 6.5, 6.8], cfg).iloc[0]
    assert r["y_last"] == 9.0 and r["a0"] == 2 and r["cat"] == 0 and r["D"] == 1


def test_labels_d_n_s(cfg: Config) -> None:
    r = _hgb_events([0, 360], [8.2, 7.9], cfg).iloc[0]
    assert (r["a0"], r["cat"], r["D"], r["N"], r["S"]) == (2, 1, 1, 0, 1)
    r = _hgb_events([0, 360], [13.0, 14.0], cfg).iloc[0]
    assert (r["D"], r["N"], r["S"]) == (0, 1, 0)


def test_sensitivity_variant_map(cfg: Config) -> None:
    a = action_map(cfg.get("actions"), "HGB", cfg.get("actions.sensitivity_variants.HGB"))
    assert a.n_categories == 2 and a.category(np.array([7.5]))[0] == 1
    k = action_map(cfg.get("actions"), "K", cfg.get("actions.sensitivity_variants.K"))
    assert list(k.category(np.array([2.9, 3.0, 6.0, 6.1]))) == [0, 1, 1, 2]
