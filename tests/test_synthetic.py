"""Checks that the synthetic generator matches the MIMIC-IV schema and exercises edge cases."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from certivoi.config import Config
from synthetic import DECOY_K_ITEMID, make_synthetic_mimic, write_mimic_layout

# Columns from docs/03_DATA_AND_COHORT.md §3 that downstream code relies on.
REQUIRED = {
    "patients": {"subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"},
    "admissions": {"subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                   "race", "insurance"},
    "labevents": {"subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom",
                  "ref_range_lower", "ref_range_upper", "flag"},
    "d_labitems": {"itemid", "label", "fluid", "category"},
    "icustays": {"hadm_id", "intime", "outtime"},
}


def test_schema(synth: dict) -> None:
    for name, cols in REQUIRED.items():
        assert cols <= set(synth[name].columns), name
        assert len(synth[name]) > 0, name


def test_deterministic(cfg: Config) -> None:
    a = make_synthetic_mimic(n_patients=10, seed=5, cfg=cfg)
    b = make_synthetic_mimic(n_patients=10, seed=5, cfg=cfg)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    c = make_synthetic_mimic(n_patients=10, seed=6, cfg=cfg)
    assert not a["labevents"]["valuenum"].equals(c["labevents"]["valuenum"])


def test_keys_consistent(synth: dict) -> None:
    pats, adm, lab, icu = (synth[k] for k in ("patients", "admissions", "labevents", "icustays"))
    assert pats["subject_id"].is_unique and adm["hadm_id"].is_unique
    assert set(adm["subject_id"]) <= set(pats["subject_id"])
    assert set(lab["hadm_id"]) <= set(adm["hadm_id"])
    assert set(icu["hadm_id"]) <= set(adm["hadm_id"])
    assert set(lab["itemid"]) <= set(synth["d_labitems"]["itemid"])


def test_labs_within_admission_and_icu_within_admission(synth: dict) -> None:
    m = synth["labevents"].merge(synth["admissions"][["hadm_id", "admittime", "dischtime"]],
                                 on="hadm_id")
    assert (m["charttime"] > m["admittime"]).all()
    assert (m["charttime"] < m["dischtime"]).all()
    i = synth["icustays"].merge(synth["admissions"], on=["subject_id", "hadm_id"])
    assert (i["intime"] >= i["admittime"]).all() and (i["outtime"] <= i["dischtime"]).all()


def test_all_target_and_feature_labs_present_with_units(synth: dict, cfg: Config) -> None:
    lab = synth["labevents"]
    for spec in cfg.labs.values():
        rows = lab[lab["itemid"] == spec.itemid]
        assert len(rows) > 100, spec.code
        assert (rows["valueuom"] == spec.unit).mean() > 0.95, spec.code
    assert set(cfg.feature_labs.values()) <= set(lab["itemid"])


def _clean_target(synth: dict, cfg: Config, code: str) -> pd.Series:
    spec = cfg.labs[code]
    v = synth["labevents"].loc[synth["labevents"]["itemid"] == spec.itemid, "valuenum"]
    return v[(v >= spec.plausible[0]) & (v <= spec.plausible[1])]


def test_values_straddle_action_thresholds(synth: dict, cfg: Config) -> None:
    for code in ("HGB", "K", "PLT"):
        v = _clean_target(synth, cfg, code)
        for b in cfg.get(f"actions.{code}.boundaries"):
            assert (v < b).sum() >= 10 and (v >= b).sum() >= 10, (code, b)


def test_threshold_crossings_between_consecutive_draws(synth: dict, cfg: Config) -> None:
    lab = synth["labevents"]
    hgb = lab[(lab["itemid"] == cfg.labs["HGB"].itemid) & lab["valuenum"].between(2, 25)]
    hgb = hgb.drop_duplicates(["hadm_id", "charttime"]).sort_values(["hadm_id", "charttime"])
    below = hgb["valuenum"] < 7.0
    crossed = below != below.groupby(hgb["hadm_id"]).shift()
    first = hgb["hadm_id"] != hgb["hadm_id"].shift()
    assert (crossed & ~first).sum() >= 5


def test_creatinine_rises_meeting_kdigo(synth: dict, cfg: Config) -> None:
    lab = synth["labevents"]
    cr = lab[(lab["itemid"] == cfg.labs["CREAT"].itemid) & lab["valuenum"].between(0.1, 25)]
    ratio = cr.groupby("hadm_id")["valuenum"].agg(lambda s: s.max() / s.min())
    assert (ratio >= 1.5).sum() >= 5


def test_artifacts_present(synth: dict, cfg: Config) -> None:
    lab, pats, adm = synth["labevents"], synth["patients"], synth["admissions"]
    assert (pats["anchor_age"] < cfg.get("cohort.min_age")).any()
    los_h = (adm["dischtime"] - adm["admittime"]) / pd.Timedelta(hours=1)
    assert (los_h < cfg.get("cohort.min_admission_hours")).any()
    assert lab["valuenum"].isna().any()
    assert lab.duplicated(["hadm_id", "itemid", "charttime"]).any()
    assert (lab["itemid"] == DECOY_K_ITEMID).any()
    hgb = lab[lab["itemid"] == cfg.labs["HGB"].itemid]
    assert (hgb["valuenum"] > cfg.labs["HGB"].plausible[1]).any()
    assert (lab.loc[lab["itemid"].isin([s.itemid for s in cfg.labs.values()]), "valueuom"]
            == "mg/L").any()


def test_clean_mode_has_no_artifacts(cfg: Config) -> None:
    t = make_synthetic_mimic(n_patients=20, seed=1, cfg=cfg, inject_artifacts=False)
    lab = t["labevents"]
    assert lab["valuenum"].notna().all()
    assert not lab.duplicated(["hadm_id", "itemid", "charttime"]).any()
    assert (t["patients"]["anchor_age"] >= cfg.get("cohort.min_age")).all()


def test_write_layout_readable_by_duckdb(synth: dict, tmp_path: Path) -> None:
    root = write_mimic_layout(synth, tmp_path / "mimic")
    con = duckdb.connect()
    n = con.execute(
        f"SELECT count(*) FROM read_csv_auto('{(root / 'hosp' / 'labevents.csv.gz').as_posix()}')"
    ).fetchone()[0]
    assert n == len(synth["labevents"])
    assert (root / "icu" / "icustays.csv.gz").exists()
