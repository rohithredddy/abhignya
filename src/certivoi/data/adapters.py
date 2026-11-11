"""Convert open datasets into the MIMIC-IV raw layout used by the pipeline.

Implements 03_DATA_AND_COHORT §8 (open datasets). Each converter writes
``<out>/hosp/{patients,admissions,labevents,d_labitems}.csv.gz`` and
``<out>/icu/icustays.csv.gz`` with MIMIC itemids and the config units, so the
cohort, label, feature, model and experiment code runs unchanged.

``anchor_year_group`` carries the *site group* (e.g. ``hospA``/``hospB``): the
"temporal" split machinery (``splits.temporal``) then becomes an external-site
split (train/val/calib on one site group, test on the other).

Sources (open access, no credentialing):
- PhysioNet/CinC Challenge 2019 (CC BY 4.0): 40,336 ICU patients, hospital A
  (training_setA) and B (training_setB); hourly bins; one ICU stay per patient.
  ``SepsisLabel`` is never read (outcome; leakage).
- eICU-CRD demo v2.0.1 (ODbL): ~2,500 unit stays from 20 hospitals; minute offsets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.data.synthetic import write_mimic_layout
from certivoi.utils.logging import get_logger

log = get_logger("certivoi.adapters")

BASE = pd.Timestamp("2150-01-01")

# MIMIC itemid and d_labitems label for each code used by the pipeline.
LABEL = {"HGB": "Hemoglobin", "K": "Potassium", "CREAT": "Creatinine", "PLT": "Platelet Count",
         "HCT": "Hematocrit", "WBC": "White Blood Cells", "NA": "Sodium", "CL": "Chloride",
         "HCO3": "Bicarbonate", "BUN": "Urea Nitrogen", "GLU": "Glucose", "CA": "Calcium, Total",
         "MG": "Magnesium", "PHOS": "Phosphate", "INR": "INR(PT)"}
FEATURE_UNITS = {"HCT": "%", "WBC": "K/uL", "NA": "mEq/L", "CL": "mEq/L", "HCO3": "mEq/L",
                 "BUN": "mg/dL", "GLU": "mg/dL", "CA": "mg/dL", "MG": "mg/dL", "PHOS": "mg/dL",
                 "INR": ""}

CHALLENGE_COLS = {"Hgb": "HGB", "Potassium": "K", "Creatinine": "CREAT", "Platelets": "PLT",
                  "Hct": "HCT", "WBC": "WBC", "Chloride": "CL", "HCO3": "HCO3", "BUN": "BUN",
                  "Glucose": "GLU", "Calcium": "CA", "Magnesium": "MG", "Phosphate": "PHOS"}
EICU_LABS = {"Hgb": "HGB", "potassium": "K", "creatinine": "CREAT", "platelets x 1000": "PLT",
             "Hct": "HCT", "WBC x 1000": "WBC", "sodium": "NA", "chloride": "CL",
             "bicarbonate": "HCO3", "BUN": "BUN", "glucose": "GLU", "calcium": "CA",
             "magnesium": "MG", "phosphate": "PHOS", "PT - INR": "INR"}


def _code_meta(cfg: Config) -> tuple[dict[str, int], dict[str, str]]:
    itemid = {c: s.itemid for c, s in cfg.labs.items()}
    itemid.update({c: int(i) for c, i in cfg.get("feature_labs").items()})
    unit = {c: s.unit for c, s in cfg.labs.items()}
    unit.update(FEATURE_UNITS)
    return itemid, unit


def _d_labitems(itemid: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"itemid": list(itemid.values()),
                         "label": [LABEL[c] for c in itemid], "fluid": "Blood",
                         "category": "Chemistry"})


def _labevents(long: pd.DataFrame, itemid: dict[str, int], unit: dict[str, str]) -> pd.DataFrame:
    """``long``: subject_id, hadm_id, charttime, code, valuenum -> MIMIC labevents."""
    long = long[long["code"].isin(itemid)].dropna(subset=["valuenum"])
    long = long.sort_values(["hadm_id", "charttime", "code"]).reset_index(drop=True)
    return pd.DataFrame({
        "labevent_id": np.arange(1, len(long) + 1), "subject_id": long["subject_id"],
        "hadm_id": long["hadm_id"], "specimen_id": pd.NA,
        "itemid": long["code"].map(itemid).astype(int), "charttime": long["charttime"],
        "value": long["valuenum"].astype(str), "valuenum": long["valuenum"],
        "valueuom": long["code"].map(unit), "ref_range_lower": np.nan,
        "ref_range_upper": np.nan, "flag": pd.NA})


# ---------------- PhysioNet/CinC Challenge 2019 ----------------
def read_challenge_patient(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", usecols=[*CHALLENGE_COLS, "Age", "Gender", "HospAdmTime",
                                             "ICULOS"])
    return df


def convert_challenge2019(src: Path, out: Path, cfg: Config) -> Path:
    """Convert ``src/training_set{A,B}/*.psv`` into the MIMIC layout under ``out``."""
    itemid, unit = _code_meta(cfg)
    pats, adms, icus, longs = [], [], [], []
    for site, sub in (("hospA", "training_setA"), ("hospB", "training_setB")):
        files = sorted((src / sub).glob("p*.psv"))
        log.info("challenge2019 %s: %d files", site, len(files))
        for f in files:
            sid = int(f.stem[1:]) + 1          # p000000-style ids; +1 avoids id 0
            df = read_challenge_patient(f)
            if df.empty:
                continue
            hadm = 10_000_000 + sid
            hours = df["ICULOS"].to_numpy(float)
            t = BASE + pd.to_timedelta(hours, unit="h")
            intime = BASE + pd.Timedelta(hours=float(np.nanmin(hours)) - 1)
            outtime = BASE + pd.Timedelta(hours=float(np.nanmax(hours)))
            hosp_adm = df["HospAdmTime"].dropna()
            admit = intime + pd.Timedelta(hours=float(min(hosp_adm.iloc[0], 0))
                                          if len(hosp_adm) else 0.0)
            age = float(df["Age"].iloc[0])
            pats.append({"subject_id": sid, "gender": "M" if df["Gender"].iloc[0] == 1 else "F",
                         "anchor_age": int(min(age, 91)), "anchor_year": 2150,
                         "anchor_year_group": site, "dod": None})
            adms.append({"subject_id": sid, "hadm_id": hadm, "admittime": admit,
                         "dischtime": outtime, "admission_type": "ICU", "race": "UNKNOWN",
                         "insurance": "UNKNOWN"})
            icus.append({"subject_id": sid, "hadm_id": hadm, "stay_id": hadm, "intime": intime,
                         "outtime": outtime})
            vals = df[list(CHALLENGE_COLS)].rename(columns=CHALLENGE_COLS)
            vals["charttime"] = t
            lg = vals.melt(id_vars="charttime", var_name="code", value_name="valuenum").dropna()
            lg["subject_id"], lg["hadm_id"] = sid, hadm
            longs.append(lg)
    tables = {"patients": pd.DataFrame(pats), "admissions": pd.DataFrame(adms),
              "icustays": pd.DataFrame(icus), "d_labitems": _d_labitems(itemid),
              "labevents": _labevents(pd.concat(longs, ignore_index=True), itemid, unit)}
    write_mimic_layout(tables, out)
    log.info("challenge2019 -> %s: %d patients, %d lab rows", out, len(pats),
             len(tables["labevents"]))
    return out


# ---------------- eICU-CRD demo ----------------
def _age(v: object) -> float:
    s = str(v).strip()
    if s.startswith(">"):
        return 90.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def convert_eicu_demo(src: Path, out: Path, cfg: Config, train_share: float = 0.6) -> Path:
    """Convert eICU demo ``patient`` + ``lab`` tables into the MIMIC layout.

    Hospital stay (``patienthealthsystemstayid``) -> admission; unit stays -> ICU
    stays; lab times = unit offset converted to hospital-admission time. Hospitals
    are ordered by id and the first ``train_share`` of them form site group
    ``siteA`` (train/val/calib), the rest ``siteB`` (external test).
    """
    itemid, unit = _code_meta(cfg)
    pt = pd.read_csv(src / "patient.csv.gz")
    lab = pd.read_csv(src / "lab.csv.gz", usecols=["patientunitstayid", "labresultoffset",
                                                    "labname", "labresult"])
    pt["subject_id"] = pd.factorize(pt["uniquepid"])[0] + 1
    pt["hadm_id"] = pt["patienthealthsystemstayid"]
    pt["age"] = pt["age"].map(_age)
    hosp = np.sort(pt["hospitalid"].unique())
    train_h = set(hosp[: int(round(train_share * len(hosp)))])
    pt["site"] = np.where(pt["hospitalid"].isin(train_h), "siteA", "siteB")
    # minutes from hospital admission to unit admission = -hospitaladmitoffset
    pt["unit_start"] = -pt["hospitaladmitoffset"].astype(float)
    pt["intime"] = BASE + pd.to_timedelta(pt["unit_start"], unit="m")
    pt["outtime"] = pt["intime"] + pd.to_timedelta(pt["unitdischargeoffset"].astype(float),
                                                   unit="m")
    pt["dischtime"] = BASE + pd.to_timedelta(
        (pt["hospitaldischargeoffset"] - pt["hospitaladmitoffset"]).astype(float), unit="m")

    first = pt.sort_values("unit_start").drop_duplicates("subject_id")
    pats = pd.DataFrame({"subject_id": first["subject_id"],
                         "gender": first["gender"].map({"Male": "M", "Female": "F"}),
                         "anchor_age": first["age"], "anchor_year": 2150,
                         "anchor_year_group": first["site"], "dod": None}).dropna(
        subset=["anchor_age", "gender"])
    pats["anchor_age"] = pats["anchor_age"].astype(int)
    adm = pt.drop_duplicates("hadm_id")
    adms = pd.DataFrame({"subject_id": adm["subject_id"], "hadm_id": adm["hadm_id"],
                         "admittime": BASE, "dischtime": adm["dischtime"],
                         "admission_type": adm["hospitaladmitsource"].fillna("UNKNOWN"),
                         "race": adm["ethnicity"].fillna("UNKNOWN"), "insurance": "UNKNOWN"})
    icus = pd.DataFrame({"subject_id": pt["subject_id"], "hadm_id": pt["hadm_id"],
                         "stay_id": pt["patientunitstayid"], "intime": pt["intime"],
                         "outtime": pt["outtime"]})
    lab = lab[lab["labname"].isin(EICU_LABS)].merge(
        pt[["patientunitstayid", "subject_id", "hadm_id", "unit_start"]], on="patientunitstayid")
    long = pd.DataFrame({"subject_id": lab["subject_id"], "hadm_id": lab["hadm_id"],
                         "charttime": BASE + pd.to_timedelta(
                             lab["unit_start"] + lab["labresultoffset"].astype(float), unit="m"),
                         "code": lab["labname"].map(EICU_LABS),
                         "valuenum": pd.to_numeric(lab["labresult"], errors="coerce")})
    tables = {"patients": pats, "admissions": adms, "icustays": icus,
              "d_labitems": _d_labitems(itemid), "labevents": _labevents(long, itemid, unit)}
    write_mimic_layout(tables, out)
    log.info("eicu_demo -> %s: %d patients, %d admissions, %d lab rows (%d/%d hospitals in "
             "siteA)", out, len(pats), len(adms), len(tables["labevents"]), len(train_h),
             len(hosp))
    return out


CONVERTERS = {"challenge2019": convert_challenge2019, "eicu_demo": convert_eicu_demo}
