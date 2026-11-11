"""Open-dataset converters (synthetic stand-ins only; CONTRIBUTING.md §4.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from certivoi.data.adapters import CHALLENGE_COLS, convert_challenge2019, convert_eicu_demo

ALL_COLS = ["HR", *CHALLENGE_COLS, "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
            "SepsisLabel"]


def _psv(path: Path, n: int, rng: np.random.Generator) -> None:
    df = pd.DataFrame(np.nan, index=range(n), columns=ALL_COLS)
    df["ICULOS"] = np.arange(1, n + 1)
    df["Age"], df["Gender"], df["HospAdmTime"] = 67.0, 1, -5.0
    df["SepsisLabel"] = 1                      # must never reach the output
    for h in range(0, n, 8):
        df.loc[h, ["Hgb", "Potassium", "Creatinine", "Platelets"]] = (
            9 + rng.normal(), 4 + 0.3 * rng.normal(), 1.1, 200.0)
    df.to_csv(path, sep="|", index=False, na_rep="NaN")


def test_convert_challenge(tmp_path: Path, cfg) -> None:
    rng = np.random.default_rng(0)
    for sub, ids in (("training_setA", [1, 2]), ("training_setB", [100001])):
        (tmp_path / "src" / sub).mkdir(parents=True)
        for i in ids:
            _psv(tmp_path / "src" / sub / f"p{i:06d}.psv", 60, rng)
    out = convert_challenge2019(tmp_path / "src", tmp_path / "out", cfg)
    pats = pd.read_csv(out / "hosp" / "patients.csv.gz")
    labs = pd.read_csv(out / "hosp" / "labevents.csv.gz")
    assert sorted(pats["anchor_year_group"]) == ["hospA", "hospA", "hospB"]
    assert set(labs["itemid"]) == {51222, 50971, 50912, 51265}
    assert labs.groupby("subject_id").size().eq(4 * 8).all()       # 8 draws x 4 labs
    assert set(labs["valueuom"]) == {"g/dL", "mEq/L", "mg/dL", "K/uL"}
    assert not any("sepsis" in c.lower() for t in ("patients", "admissions", "labevents")
                   for c in pd.read_csv(out / "hosp" / f"{t}.csv.gz", nrows=1).columns)


def test_convert_eicu(tmp_path: Path, cfg) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pt = pd.DataFrame({"patientunitstayid": [11, 12, 21], "patienthealthsystemstayid": [1, 1, 2],
                       "uniquepid": ["a", "a", "b"], "gender": ["Male", "Male", "Female"],
                       "age": ["> 89", "> 89", "54"], "ethnicity": ["Caucasian", None, "Other"],
                       "hospitalid": [5, 5, 9], "hospitaladmitoffset": [-120, -3000, 0],
                       "hospitaldischargeoffset": [6000, 3000, 4000],
                       "unitdischargeoffset": [2000, 2500, 3000],
                       "hospitaladmitsource": ["ED", "ED", "Floor"]})
    pt.to_csv(src / "patient.csv.gz", index=False)
    lab = pd.DataFrame({"patientunitstayid": [11, 11, 12, 21, 21],
                        "labresultoffset": [60, 600, 30, 10, 20],
                        "labname": ["Hgb", "Hgb", "potassium", "platelets x 1000", "foo"],
                        "labresult": [8.1, 7.6, 4.2, 150, 1]})
    lab.to_csv(src / "lab.csv.gz", index=False)
    out = convert_eicu_demo(src, tmp_path / "out", cfg, train_share=0.5)
    labs = pd.read_csv(out / "hosp" / "labevents.csv.gz", parse_dates=["charttime"])
    assert len(labs) == 4                                   # unknown lab dropped
    hgb = labs[labs["itemid"] == 51222].sort_values("charttime")
    # unit stay 11 starts 120 min after hospital admission -> labs at 180 and 720 min
    assert (hgb["charttime"].diff().dropna() == pd.Timedelta(minutes=540)).all()
    pats = pd.read_csv(out / "hosp" / "patients.csv.gz")
    assert dict(zip(pats["subject_id"], pats["anchor_year_group"], strict=True)) == {
        1: "siteA", 2: "siteB"}
    assert pats.loc[pats["subject_id"] == 1, "anchor_age"].item() == 90
