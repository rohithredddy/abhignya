"""Phase 1: extraction, cohort filters, dedup, repeat definition, splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from certivoi.config import Config
from certivoi.data.cohort import build_cohort, cap_by_patient
from certivoi.data.extract import extract
from certivoi.data.split import assign_random, assign_temporal, make_splits
from certivoi.data.verify import verify_items
from synthetic import DECOY_K_ITEMID


@pytest.fixture(scope="module")
def built(pipe_cfg: Config) -> dict:
    ex = extract(pipe_cfg)
    co = build_cohort(pipe_cfg)
    sp = make_splits(pipe_cfg)
    return {"ex": ex, "co": co, "sp": sp}


def test_verify_items_passes_on_synthetic(pipe_cfg: Config, built: dict) -> None:
    table, problems = verify_items(pipe_cfg)
    assert problems == []
    assert DECOY_K_ITEMID not in set(table["itemid"])


def test_flow_table_monotone(pipe_cfg: Config, built: dict) -> None:
    flow = pd.read_csv(pipe_cfg.results_dir / "cohort" / "flow.csv")
    assert (pipe_cfg.results_dir / "cohort" / "meta.json").exists()
    for _, g in flow.groupby("lab"):
        n = g.sort_values("step")["n_events"].to_numpy()
        assert (np.diff(n) <= 0).all()
        assert n[-1] > 0
    # each artifact type removes something on synthetic data
    hgb = flow[flow["lab"] == "HGB"].set_index("step")["n_events"]
    assert hgb["2_adult"] < hgb["1_extracted_in_admission"]
    assert hgb["3_admission_ge_24h"] < hgb["2_adult"]
    assert hgb["7_deduplicated"] < hgb["6_plausible"]


def test_clean_labs_filters(pipe_cfg: Config, built: dict) -> None:
    clean = pd.read_parquet(built["co"]["clean_labs"])
    assert not clean.duplicated(["hadm_id", "itemid", "charttime"]).any()
    assert clean["valuenum"].notna().all()
    assert DECOY_K_ITEMID not in set(clean["itemid"])
    for code, spec in pipe_cfg.labs.items():
        v = clean.loc[clean["code"] == code, "valuenum"]
        assert v.between(*spec.plausible).all(), code
    adm = pd.read_parquet(built["co"]["admissions"])
    assert (adm["anchor_age"] >= pipe_cfg.get("cohort.min_age")).all()
    los = (adm["dischtime"] - adm["admittime"]) / pd.Timedelta(hours=1)
    assert (los >= pipe_cfg.get("cohort.min_admission_hours")).all()
    assert set(clean["hadm_id"]) <= set(adm["hadm_id"])


def test_dedup_keeps_first_labevent(pipe_cfg: Config, built: dict) -> None:
    raw = pd.read_parquet(built["ex"]["labs"])
    clean = pd.read_parquet(built["co"]["clean_labs"])
    first = raw.sort_values("labevent_id").drop_duplicates(["hadm_id", "itemid", "charttime"])
    m = clean.merge(first[["hadm_id", "itemid", "charttime", "labevent_id"]],
                    on=["hadm_id", "itemid", "charttime"], suffixes=("", "_first"))
    assert (m["labevent_id"] == m["labevent_id_first"]).all()


def test_repeat_events_have_prior_within_lookback(pipe_cfg: Config, built: dict) -> None:
    clean = pd.read_parquet(built["co"]["clean_labs"])
    lb = pd.Timedelta(hours=pipe_cfg.get("cohort.repeat_lookback_hours"))
    gap = pd.Timedelta(minutes=pipe_cfg.get("cohort.min_gap_minutes"))
    for lab in pipe_cfg.labs:
        ev = pd.read_parquet(built["co"][f"events_{lab}"])
        hist = clean[clean["code"] == lab]
        assert len(ev) > 50
        m = ev.merge(hist[["hadm_id", "charttime"]], on="hadm_id")
        ok = (m["charttime"] < m["t"] - gap) & (m["charttime"] >= m["t"] - lb)
        has_prior = ok.groupby(m["event_id"]).any()
        assert has_prior.reindex(ev["event_id"]).fillna(False).all()
        # and every clean event NOT in ev (and not capped) lacks such a prior
        others = hist[~hist["labevent_id"].isin(ev["event_id"])]
        m2 = others.merge(hist[["hadm_id", "charttime"]], on="hadm_id", suffixes=("", "_h"))
        ok2 = ((m2["charttime_h"] < m2["charttime"] - gap)
               & (m2["charttime_h"] >= m2["charttime"] - lb))
        assert not ok2.any()


def test_splits_disjoint_and_complete(pipe_cfg: Config, built: dict) -> None:
    sp = pd.read_parquet(built["sp"])
    assert sp["subject_id"].is_unique
    assert set(sp["split"]) == {"train", "val", "calib", "test"}
    names = ("train", "val", "calib", "test")
    groups = [set(sp.loc[sp["split"] == s, "subject_id"]) for s in names]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not groups[i] & groups[j]
    for lab in pipe_cfg.labs:
        ev = pd.read_parquet(built["co"][f"events_{lab}"])
        assert set(ev["subject_id"]) <= set(sp["subject_id"])


def test_assign_random_fractions() -> None:
    fr = {"train": 0.6, "val": 0.1, "calib": 0.15, "test": 0.15}
    lab = assign_random(np.arange(1000), fr, seed=1)
    counts = pd.Series(lab).value_counts()
    assert counts["train"] == 600 and counts["val"] == 100 and counts["calib"] == 150
    assert (assign_random(np.arange(1000), fr, seed=1) == lab).all()


def test_assign_temporal() -> None:
    g = pd.Series(["2008 - 2010"] * 50 + ["2017 - 2019"] * 20 + ["2020 - 2022"] * 5)
    fr = {"train": 0.6, "val": 0.1, "calib": 0.15, "test": 0.15}
    lab = assign_temporal(g, fr, ["2008 - 2010"], ["2017 - 2019"], seed=0)
    assert (lab[50:70] == "test").all()
    assert all(x is None for x in lab[70:])
    assert set(lab[:50]) == {"train", "val", "calib"}


def test_cap_by_patient() -> None:
    counts = pd.Series([5, 5, 5, 5], index=[1, 2, 3, 4])
    assert cap_by_patient(counts, 100, 0) == {1, 2, 3, 4}
    kept = cap_by_patient(counts, 12, 0)
    assert len(kept) == 2
    assert counts.loc[list(kept)].sum() <= 12
