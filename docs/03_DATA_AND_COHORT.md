# 03 — Data, Cohort and Features

## 1. Datasets

| Dataset | Access | Use |
|---|---|---|
| **MIMIC-IV Clinical Database Demo** (v2.2, 100 patients) | Open (ODbL), direct download from PhysioNet | Pipeline development, unit/integration testing, demo UI. **Results from the demo are not publishable** (too small). |
| **MIMIC-IV** (v3.x, `hosp` + `icu` modules) | Credentialed: PhysioNet account + CITI "Data or Specimens Only Research" course + signed DUA | All reported experiments |
| eICU-CRD (optional stretch) | Credentialed | External validation only if time allows |

**Action item (week 1, human):** register on PhysioNet, complete CITI training, request MIMIC-IV access. Approval can take days to weeks. Also check PhysioNet's current access policies for your country.

## 2. Privacy rules (DUA compliance)

PhysioNet's DUA forbids sharing credentialed data with third parties, and its 2025 guidance on LLMs requires zero data retention / no training / no human review for any online service, and recommends local models. Practical rule for this project:

- **Row-level full-MIMIC data never leaves the local machine.** Code is developed on synthetic or open data and run locally on the full data; only **aggregate** outputs (counts with n ≥ 10, metrics, plots of aggregates) are shared back.
- Row-level inspection is allowed only in `data_mode: demo` (the open demo dataset).
- No MIMIC rows in tests, fixtures, notebooks committed to git, the paper, or the demo UI. Tests use a **synthetic generator** with the same schema.

## 3. Tables used (MIMIC-IV schema)

| Module.table | Columns needed | Purpose |
|---|---|---|
| `hosp.patients` | subject_id, gender, anchor_age, anchor_year, anchor_year_group | demographics, temporal split |
| `hosp.admissions` | subject_id, hadm_id, admittime, dischtime, admission_type, race, insurance | admission context, subgroups |
| `hosp.labevents` | subject_id, hadm_id, itemid, charttime, valuenum, valueuom, ref_range_lower, ref_range_upper, flag | targets + lab history |
| `hosp.d_labitems` | itemid, label, fluid, category | item verification |
| `icu.icustays` | hadm_id, intime, outtime | ICU flag at time t |
| `icu.inputevents` (optional) | stay_id, starttime, itemid, amount | transfusion / KCl between draws (ICU subset ablation) |

**Excluded for leakage:** `hosp.diagnoses_icd`, `hosp.drgcodes` (assigned at/after discharge), `dischtime`-derived LOS, discharge location, hospital_expire_flag.

## 4. Lab item IDs (verify against `d_labitems` before use)

| Code | itemid (MIMIC-IV, blood) | Role |
|---|---|---|
| HGB | 51222 | target + feature |
| K | 50971 | target + feature (do **not** mix with blood-gas potassium 50822) |
| CREAT | 50912 | target + feature |
| PLT | 51265 | target + feature |
| HCT | 51221 | feature |
| WBC | 51301 | feature |
| NA | 50983 | feature |
| CL | 50902 | feature |
| HCO3 | 50882 | feature |
| BUN | 51006 | feature |
| GLU | 50931 | feature |
| CA | 50893 | feature |
| MG | 50960 | feature |
| PHOS | 50970 | feature |
| INR | 51237 | feature |

Script `scripts/01_verify_items.py` must print label/fluid/category/unit counts for each itemid (aggregate only) and fail loudly on mismatch.

## 5. Cohort

1. Admissions of patients with `anchor_age ≥ 18`.
2. Admission duration ≥ 24 h.
3. Lab events with non-null `valuenum`, expected unit, and values within plausibility bounds (config; drop outliers beyond physiologic limits, log counts).
4. Deduplicate: same `(hadm_id, itemid, charttime)` → keep first.
5. **Repeat events:** at least one prior same-lab value in the same admission within 72 h.
6. Optional cap per lab (config `max_events_per_lab`, default 400k) by **patient-level** random sampling to bound CPU time.

Report a cohort flow table (counts after each step, per lab) — Table 1 of the paper.

## 6. Splits

- **Main:** random by `subject_id` → train 60% / val 10% / calib 15% / test 15% (seeded).
- **Temporal shift:** train/val/calib on `anchor_year_group` ≤ "2014 - 2016"; test on "2017 - 2019" (and later groups if present in the version used).
- **CRC resampling:** keep trained model fixed; re-draw calib/test partition of the held-out patients 100 times to report the distribution of realized risk.

## 7. Features (computed at time t from strictly-prior data)

**Demographic/context (≈8):** age, sex, admission_type (one-hot, few levels), hours since admission, in-ICU flag, hour-of-day and weekday of draw, number of prior admissions.

**Target-lab history (≈15):** last value; second-last; Δ(last, second-last); hours since last; OLS slope over last 48 h; min, max, std over 48 h; count in 24 h and 72 h; default category $a^0$; signed distance of last value to nearest action boundary; number of category changes so far in admission; last value's reference-range flag. CREAT extra: 48 h min, 7 d min, current AKI stage.

**Cross-lab panel (≈2 per lab × 15):** last value and hours-since for each lab in §4 (NaN if none; LightGBM handles missing natively; also add missing-indicators only if helpful in ablation).

**Optional (ablation, ICU subset):** PRBC / platelet transfusion amount and KCl given since the last draw.

**Never as features:** race, insurance (subgroup evaluation only); anything from the future; discharge-time codes.

**Implementation notes.** History windows use same-admission values strictly before t − 1 min; the target-lab window is the 72 h look-back (count/trend features use 24 h / 48 h sub-windows); cross-lab features use an ASOF join with the same strict cut-off. `in_icu` = an ICU stay that started before t − 1 min and has not ended at t. Admission length (`dischtime`) is used only for the cohort filter and as the patient-days denominator in E3, never as a feature. Feature values are rounded (4 decimals; hours 2) and stored float32-exact (see 02 §6). The leakage test (`tests/test_leakage.py`) perturbs all raw lab rows at/after a per-admission cut-off, re-runs the pipeline from raw files, and asserts that features of all earlier events are unchanged; it was checked to fail when a leak is planted.

## 8. Storage & compute plan (CPU)

- Use **DuckDB** to read `labevents.csv.gz` directly, filter by itemid and cohort, and write **Parquet** to `data/interim/` once. Never load the full labevents table into pandas.
- Feature engineering in DuckDB SQL window functions or Polars, partitioned by lab.
- Set `threads` in config (default: physical cores − 1). LightGBM `n_jobs` from same setting.
- Expected: extraction 15–60 min one-time; per-lab model training minutes, not hours.
