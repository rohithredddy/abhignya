"""Synthetic MIMIC-IV-schema generator for tests, fixtures and the demo (no real patient data).

Implements 05_IMPLEMENTATION_ROADMAP Phase 0 (synthetic generator) and CONTRIBUTING.md §4.5.
Produces ``patients``, ``admissions``, ``labevents``, ``d_labitems`` and ``icustays``
with the MIMIC-IV column names, realistic mean-reverting lab trajectories, and
clinical episodes (bleeding, AKI, thrombocytopenia, hypokalaemia) that push values
across the action thresholds in ``configs/default.yaml``.

With ``inject_artifacts=True`` it also adds the dirt the cohort step must handle:
minors, admissions < 24 h, duplicate (hadm_id, itemid, charttime) rows, null
``valuenum``, implausible values, wrong units, and blood-gas potassium (50822).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from certivoi.config import Config, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"

DECOY_K_ITEMID = 50822  # blood-gas potassium; must never be mixed with 50971
YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
ADMISSION_TYPES = ["EW EMER.", "URGENT", "ELECTIVE", "OBSERVATION ADMIT", "DIRECT EMER."]
RACES = ["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN", "OTHER", "UNKNOWN"]
INSURANCE = ["Medicare", "Medicaid", "Other"]
CAREUNITS = ["Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)",
             "Cardiac Vascular Intensive Care Unit (CVICU)"]


@dataclass(frozen=True)
class _LabModel:
    """Generative parameters for one lab (log-scale labs model log(value))."""

    label: str
    category: str
    unit: str | None
    mean: float          # population mean (on natural scale)
    sd_between: float    # between-admission sd of the baseline (model scale)
    sd_within: float     # stationary OU sd (model scale)
    tau_h: float         # OU mean-reversion time (hours)
    ref: tuple[float, float]
    decimals: int
    log: bool = False


# Natural-scale means / reference ranges loosely follow inpatient MIMIC distributions.
_FEATURE_LAB_MODELS: dict[str, _LabModel] = {
    "HCT": _LabModel("Hematocrit", "Hematology", "%", 31.0, 0, 1.0, 1, (36.0, 48.0), 1),
    "WBC": _LabModel("White Blood Cells", "Hematology", "K/uL", 9.5, 0.35, 0.2, 36,
                     (4.0, 11.0), 1, log=True),
    "NA": _LabModel("Sodium", "Chemistry", "mEq/L", 139.0, 3.0, 2.0, 48, (133.0, 145.0), 0),
    "CL": _LabModel("Chloride", "Chemistry", "mEq/L", 103.0, 3.5, 2.0, 48, (96.0, 108.0), 0),
    "HCO3": _LabModel("Bicarbonate", "Chemistry", "mEq/L", 24.5, 3.0, 2.0, 36, (22.0, 32.0), 0),
    "BUN": _LabModel("Urea Nitrogen", "Chemistry", "mg/dL", 22.0, 0.5, 0.15, 72,
                     (6.0, 20.0), 0, log=True),
    "GLU": _LabModel("Glucose", "Chemistry", "mg/dL", 125.0, 0.2, 0.2, 12,
                     (70.0, 100.0), 0, log=True),
    "CA": _LabModel("Calcium, Total", "Chemistry", "mg/dL", 8.6, 0.5, 0.3, 48, (8.4, 10.3), 1),
    "MG": _LabModel("Magnesium", "Chemistry", "mg/dL", 2.0, 0.2, 0.15, 36, (1.6, 2.6), 1),
    "PHOS": _LabModel("Phosphate", "Chemistry", "mg/dL", 3.5, 0.7, 0.4, 36, (2.7, 4.5), 1),
    "INR": _LabModel("INR(PT)", "Hematology", None, 1.3, 0.25, 0.1, 48, (0.9, 1.1), 1, log=True),
}

# Target-lab generative parameters; itemid/unit come from the config.
_TARGET_PARAMS: dict[str, dict] = {
    "HGB": dict(label="Hemoglobin", category="Hematology", mean=10.0, sd_between=1.6,
                sd_within=0.45, tau_h=48, ref=(12.0, 16.0), decimals=1),
    "K": dict(label="Potassium", category="Chemistry", mean=4.1, sd_between=0.35,
              sd_within=0.3, tau_h=24, ref=(3.3, 5.1), decimals=1),
    "CREAT": dict(label="Creatinine", category="Chemistry", mean=1.1, sd_between=0.45,
                  sd_within=0.06, tau_h=72, ref=(0.5, 1.2), decimals=1, log=True),
    "PLT": dict(label="Platelet Count", category="Hematology", mean=200.0, sd_between=0.5,
                sd_within=0.1, tau_h=72, ref=(150.0, 440.0), decimals=0, log=True),
}

# Labs drawn on the same specimen share a specimen_id.
_PANELS: dict[str, tuple[list[str], float]] = {
    "CBC": (["HGB", "HCT", "PLT", "WBC"], 0.85),
    "CHEM": (["K", "NA", "CL", "HCO3", "BUN", "CREAT", "GLU", "CA"], 0.90),
    "EXT": (["MG", "PHOS"], 0.55),
    "COAG": (["INR"], 0.30),
}


def _lab_models(cfg: Config) -> dict[str, tuple[int, _LabModel]]:
    """Map lab code -> (itemid, model) for all target and feature labs."""
    out: dict[str, tuple[int, _LabModel]] = {}
    for code, spec in cfg.labs.items():
        out[code] = (spec.itemid, _LabModel(unit=spec.unit, **_TARGET_PARAMS[code]))
    for code, itemid in cfg.feature_labs.items():
        out[code] = (itemid, _FEATURE_LAB_MODELS[code])
    return out


def _ou_path(rng: np.random.Generator, t: np.ndarray, mu: float, sd: float, tau: float
             ) -> np.ndarray:
    """Exact Ornstein-Uhlenbeck sample at sorted times ``t`` (hours), stationary start."""
    x = np.empty(len(t))
    x[0] = mu + sd * rng.standard_normal()
    for k in range(1, len(t)):
        rho = np.exp(-(t[k] - t[k - 1]) / tau)
        x[k] = mu + (x[k - 1] - mu) * rho + sd * np.sqrt(1 - rho**2) * rng.standard_normal()
    return x


def _ramp(t: np.ndarray, start: float, dur: float) -> np.ndarray:
    """0 before ``start``, linear to 1 over ``dur`` hours, then 1."""
    return np.clip((t - start) / dur, 0.0, 1.0)


def _episode_effects(rng: np.random.Generator, t: np.ndarray, los_h: float
                     ) -> dict[str, np.ndarray]:
    """Additive effects (model scale) from random clinical episodes in one admission."""
    eff = {k: np.zeros(len(t)) for k in ("HGB", "K", "CREAT", "PLT", "BUN")}

    def start() -> float:
        return float(rng.uniform(0.0, max(los_h * 0.7, 1.0)))

    if rng.random() < 0.30:  # bleeding: HGB drop 1.5-4 g/dL over 6-24 h
        r = _ramp(t, start(), rng.uniform(6, 24))
        eff["HGB"] -= rng.uniform(1.5, 4.0) * r
        eff["PLT"] -= 0.15 * r
    if rng.random() < 0.25:  # AKI: creatinine x1.5-3.5 over 24-72 h, K and BUN up
        r = _ramp(t, start(), rng.uniform(24, 72))
        eff["CREAT"] += np.log(rng.uniform(1.5, 3.5)) * r
        eff["K"] += rng.uniform(0.5, 1.6) * r
        eff["BUN"] += 0.6 * r
    if rng.random() < 0.20:  # thrombocytopenia: PLT x0.01-0.4 (log-uniform)
        drop = rng.uniform(np.log(0.01), np.log(0.4))
        eff["PLT"] += drop * _ramp(t, start(), rng.uniform(24, 96))
    if rng.random() < 0.20:  # hypokalaemia: K down 0.7-1.4
        eff["K"] -= rng.uniform(0.7, 1.4) * _ramp(t, start(), rng.uniform(6, 24))
    return eff


def _draw_times(rng: np.random.Generator, los_h: float) -> np.ndarray:
    """Blood-draw times (hours since admission): first within 4 h, then ~12 h gaps."""
    times = [rng.uniform(0.5, 4.0)]
    while True:
        nxt = times[-1] + rng.gamma(shape=2.5, scale=5.0) + 0.25
        if nxt >= los_h - 0.5:
            break
        times.append(nxt)
    return np.round(np.asarray(times) * 60) / 60  # whole minutes


def make_synthetic_mimic(
    n_patients: int = 100,
    seed: int = 0,
    cfg: Config | None = None,
    inject_artifacts: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate MIMIC-IV-schema tables.

    Inputs: number of patients, RNG seed, config (itemids/units; default config if
    None), whether to inject data-quality artifacts. Output: dict with keys
    ``patients``, ``admissions``, ``labevents``, ``d_labitems``, ``icustays``.
    Deterministic for a given (n_patients, seed, cfg, inject_artifacts).
    """
    cfg = cfg or load_config(DEFAULT_CONFIG)
    rng = np.random.default_rng(seed)
    models = _lab_models(cfg)

    patients, admissions, icustays, labs = [], [], [], []
    hadm_id, stay_id, specimen_id = 20_000_000, 30_000_000, 1
    min_age = 15 if inject_artifacts else int(cfg.get("cohort.min_age"))

    for i in range(n_patients):
        subject_id = 10_000_000 + i
        gender = "F" if rng.random() < 0.45 else "M"
        age = int(rng.integers(min_age, 92))
        anchor_year = int(rng.integers(2110, 2190))
        patients.append(dict(subject_id=subject_id, gender=gender, anchor_age=age,
                             anchor_year=anchor_year,
                             anchor_year_group=YEAR_GROUPS[int(rng.integers(len(YEAR_GROUPS)))],
                             dod=pd.NaT))
        race = RACES[int(rng.integers(len(RACES)))]
        insurance = INSURANCE[int(rng.integers(len(INSURANCE)))]
        admit = pd.Timestamp(f"{anchor_year}-01-01") + pd.Timedelta(
            hours=float(rng.uniform(0, 24 * 365)))

        for _ in range(int(rng.integers(1, 4))):
            hadm_id += 1
            short = inject_artifacts and rng.random() < 0.08
            los_h = float(rng.uniform(4, 23)) if short else 24 + float(rng.gamma(2.0, 55.0))
            admit = admit.floor("min")
            disch = admit + pd.Timedelta(hours=los_h)
            died = rng.random() < 0.05
            admissions.append(dict(
                subject_id=subject_id, hadm_id=hadm_id, admittime=admit, dischtime=disch,
                deathtime=disch if died else pd.NaT,
                admission_type=ADMISSION_TYPES[int(rng.integers(len(ADMISSION_TYPES)))],
                admission_location="EMERGENCY ROOM",
                discharge_location="DIED" if died else "HOME",
                insurance=insurance, race=race, hospital_expire_flag=int(died)))

            if rng.random() < 0.4 and los_h > 12:  # one ICU stay
                stay_id += 1
                intime = admit + pd.Timedelta(hours=float(rng.uniform(0, min(24, los_h / 2))))
                outtime = min(intime + pd.Timedelta(hours=float(rng.uniform(24, 120))), disch)
                unit = CAREUNITS[int(rng.integers(len(CAREUNITS)))]
                icustays.append(dict(subject_id=subject_id, hadm_id=hadm_id, stay_id=stay_id,
                                     first_careunit=unit, last_careunit=unit, intime=intime,
                                     outtime=outtime,
                                     los=(outtime - intime) / pd.Timedelta(days=1)))

            t = _draw_times(rng, los_h)
            eff = _episode_effects(rng, t, los_h)
            values: dict[str, np.ndarray] = {}
            for code, (_, m) in models.items():
                if code == "HCT":
                    continue
                mu = (np.log(m.mean) if m.log else m.mean) + m.sd_between * rng.standard_normal()
                x = _ou_path(rng, t, mu, m.sd_within, m.tau_h) + eff.get(code, 0.0)
                values[code] = np.exp(x) if m.log else x
            values["HCT"] = 3.0 * values["HGB"] + rng.normal(0, 1.0, len(t))

            for k, hours in enumerate(t):
                charttime = admit + pd.Timedelta(hours=float(hours))
                priority = "STAT" if rng.random() < 0.3 else "ROUTINE"
                for panel_labs, p in _PANELS.values():
                    if rng.random() >= p:
                        continue
                    specimen_id += 1
                    store = charttime + pd.Timedelta(minutes=int(rng.integers(30, 180)))
                    for code in panel_labs:
                        if code not in models:
                            continue
                        itemid, m = models[code]
                        v = round(float(values[code][k]), m.decimals)
                        lo, hi = m.ref
                        if code == "HGB" and gender == "M":
                            lo, hi = lo + 1.5, hi + 1.5
                        labs.append(dict(
                            subject_id=subject_id, hadm_id=hadm_id, specimen_id=specimen_id,
                            itemid=itemid, charttime=charttime, storetime=store,
                            value=f"{v:.{m.decimals}f}", valuenum=v, valueuom=m.unit,
                            ref_range_lower=lo, ref_range_upper=hi,
                            flag="abnormal" if (v < lo or v > hi) else None,
                            priority=priority, comments=None))
            admit = disch + pd.Timedelta(days=float(rng.uniform(10, 400)))

    lab_df = pd.DataFrame(labs)
    if inject_artifacts:
        lab_df = _inject_lab_artifacts(rng, lab_df, cfg)
    lab_df = lab_df.sort_values(["subject_id", "charttime", "itemid"], kind="stable")
    lab_df.insert(0, "labevent_id", np.arange(1, len(lab_df) + 1))
    lab_df = lab_df.reset_index(drop=True)

    d_labitems = pd.DataFrame(
        [dict(itemid=itemid, label=m.label, fluid="Blood", category=m.category)
         for itemid, m in models.values()]
        + [dict(itemid=DECOY_K_ITEMID, label="Potassium, Whole Blood", fluid="Blood",
                category="Blood Gas")]
    ).sort_values("itemid").reset_index(drop=True)

    icu_cols = ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit",
                "intime", "outtime", "los"]
    return {
        "patients": pd.DataFrame(patients),
        "admissions": pd.DataFrame(admissions),
        "labevents": lab_df,
        "d_labitems": d_labitems,
        "icustays": pd.DataFrame(icustays, columns=icu_cols),
    }


def _inject_lab_artifacts(rng: np.random.Generator, df: pd.DataFrame, cfg: Config
                          ) -> pd.DataFrame:
    """Add duplicates, null valuenum, implausible values, wrong units, blood-gas K."""
    n = len(df)
    target_ids = {s.itemid: s for s in cfg.labs.values()}
    is_target = df["itemid"].isin(list(target_ids)).to_numpy()
    df = df.copy()

    # null valuenum (~0.5%)
    idx = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[idx, "valuenum"] = np.nan
    df.loc[idx, "value"] = "___"

    # implausible target values (~0.3% of target rows): 10x the upper plausible bound
    tgt_rows = np.flatnonzero(is_target)
    idx = rng.choice(tgt_rows, size=max(1, len(tgt_rows) // 300), replace=False)
    for j in idx:
        hi = target_ids[int(df.at[j, "itemid"])].plausible[1]
        df.at[j, "valuenum"] = hi * 10
        df.at[j, "value"] = f"{hi * 10:.1f}"

    # wrong unit on a few target rows
    idx = rng.choice(tgt_rows, size=max(1, len(tgt_rows) // 500), replace=False)
    df.loc[idx, "valueuom"] = "mg/L"

    # exact duplicates on (hadm_id, itemid, charttime) (~1%)
    dup = df.iloc[rng.choice(n, size=max(1, n // 100), replace=False)].copy()
    dup["valuenum"] = dup["valuenum"] + 0.1  # differing value: dedup must keep first

    # blood-gas potassium decoy rows copied from some K rows
    k_rows = df[df["itemid"] == cfg.labs["K"].itemid]
    decoy = k_rows.sample(n=max(1, len(k_rows) // 20), random_state=int(rng.integers(1 << 31)))
    decoy = decoy.assign(itemid=DECOY_K_ITEMID, valuenum=decoy["valuenum"] - 0.2)
    return pd.concat([df, dup, decoy], ignore_index=True)


def write_mimic_layout(tables: dict[str, pd.DataFrame], root: str | Path) -> Path:
    """Write tables as ``<root>/hosp/*.csv.gz`` and ``<root>/icu/icustays.csv.gz``."""
    root = Path(root)
    (root / "hosp").mkdir(parents=True, exist_ok=True)
    (root / "icu").mkdir(parents=True, exist_ok=True)
    for name in ("patients", "admissions", "labevents", "d_labitems"):
        tables[name].to_csv(root / "hosp" / f"{name}.csv.gz", index=False)
    tables["icustays"].to_csv(root / "icu" / "icustays.csv.gz", index=False)
    return root
