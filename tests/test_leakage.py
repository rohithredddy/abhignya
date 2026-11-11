"""Leakage test (CONTRIBUTING.md §5; 05_IMPLEMENTATION_ROADMAP Phase 3).

Generate synthetic raw data, run extract -> cohort -> labels -> features. Then,
per admission, pick a cutoff time t* (a repeat-event time) and perturb every raw
lab row with charttime >= t* - 1 min (change values, add new rows). Re-run the
pipeline from raw. Features of every event (all labs) with t <= t* must be
identical: they may only use data strictly before t - 1 min.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from certivoi.config import Config
from certivoi.data.cohort import build_cohort
from certivoi.data.extract import extract
from certivoi.features.build import build_features, manifest
from synthetic import make_synthetic_mimic, write_mimic_layout


def _run(cfg: Config, root: Path, tables: dict) -> Config:
    c = cfg.with_overrides({"data": {"root": str(root)}})
    write_mimic_layout(tables, root)
    extract(c)
    build_cohort(c)
    build_features(c)
    return c


@pytest.fixture(scope="module")
def leak_setup(pipe_cfg: Config, tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("leak")
    tables = make_synthetic_mimic(n_patients=50, seed=21, cfg=pipe_cfg)
    c1 = _run(pipe_cfg, base / "orig", tables)

    rng = np.random.default_rng(0)
    lab = tables["labevents"]
    # cutoff per admission: a random chart time from the middle of its HGB draws
    hgb = lab[lab["itemid"] == pipe_cfg.labs["HGB"].itemid]
    cut = hgb.groupby("hadm_id")["charttime"].agg(
        lambda s: s.sort_values().iloc[len(s) // 2]).rename("t_star")
    lab2 = lab.merge(cut, left_on="hadm_id", right_index=True, how="left")
    after = (lab2["charttime"] >= lab2["t_star"] - pd.Timedelta(minutes=1)).to_numpy()
    assert after.sum() > 100
    lab2.loc[after, "valuenum"] = lab2.loc[after, "valuenum"] * rng.uniform(0.85, 1.15,
                                                                           after.sum())
    lab2.loc[after, "flag"] = np.where(rng.random(after.sum()) < 0.5, "abnormal", None)
    extra = lab2[after].sample(n=200, random_state=1).copy()
    extra["charttime"] = extra["charttime"] + pd.Timedelta(minutes=7)
    extra["labevent_id"] = lab2["labevent_id"].max() + 1 + np.arange(len(extra))
    lab2 = pd.concat([lab2, extra], ignore_index=True).drop(columns="t_star")
    tables2 = {**tables, "labevents": lab2}
    c2 = _run(pipe_cfg, base / "pert", tables2)
    return c1, c2, cut


@pytest.mark.parametrize("lab", ["HGB", "K", "CREAT", "PLT"])
def test_features_unchanged_by_future_data(leak_setup, lab: str) -> None:
    c1, c2, cut = leak_setup
    p1, p2 = build_features(c1)[lab], build_features(c2)[lab]
    f1, f2 = pd.read_parquet(p1["features"]), pd.read_parquet(p2["features"])
    m1 = pd.read_parquet(p1["meta"])[["event_id", "hadm_id", "t"]]
    m1 = m1.merge(cut, left_on="hadm_id", right_index=True, how="inner")
    early = m1.loc[m1["t"] <= m1["t_star"], "event_id"]
    assert len(early) > 30, "too few events before the cutoff to test"
    a = f1[f1["event_id"].isin(early)].sort_values("event_id").reset_index(drop=True)
    b = f2[f2["event_id"].isin(early)].sort_values("event_id").reset_index(drop=True)
    assert len(a) == len(b), "events before cutoff dropped/added by future data"
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_perturbation_changes_later_features(leak_setup) -> None:
    """Sanity: the perturbation is strong enough to change features after the cutoff."""
    c1, c2, cut = leak_setup
    p1, p2 = build_features(c1)["HGB"], build_features(c2)["HGB"]
    f1, f2 = pd.read_parquet(p1["features"]), pd.read_parquet(p2["features"])
    common = f1.merge(f2, on="event_id", suffixes=("_a", "_b"))
    assert (common["last_a"] != common["last_b"]).any()


def test_forbidden_columns_never_features(pipe_cfg: Config) -> None:
    banned = {"race", "insurance", "dischtime", "los_days", "discharge_location",
              "hospital_expire_flag", "y", "D", "N", "S", "cat", "race_group"}
    for lab in pipe_cfg.labs:
        assert not banned & set(manifest(pipe_cfg, lab)["name"])
