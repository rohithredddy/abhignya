# 05 — Implementation Roadmap

Work **one phase at a time**. Each phase lists tasks and **acceptance criteria (AC)**. A phase is done only when all AC pass and `PROGRESS.md` is updated. Suggested calendar: 16 weeks.

---

## Phase 0 — Scaffold (Week 1)
Tasks:
- Create repo layout from `README.md`, `pyproject.toml` (package `certivoi`, src layout), `requirements.txt`, `.gitignore` (ignore `data/`, `results/**/rows*`, `.env`).
- `src/certivoi/config.py`: load `configs/default.yaml` (+ optional override file), expose typed access; compute and store a config hash.
- `src/certivoi/utils/logging.py`, `utils/seed.py`, `utils/io.py` (write results with `meta.json`).
- `tests/synthetic.py`: generator that produces MIMIC-schema tables (patients, admissions, labevents, d_labitems, icustays) with realistic lab trajectories and occasional threshold crossings.
- `make`-style entry points or a `scripts/run.py` CLI with subcommands.

AC: `pytest -q` passes on synthetic data; `ruff check` clean; `python scripts/run.py --help` works.

## Phase 1 — Data extraction (Weeks 1–3)
Tasks:
- `scripts/00_download_demo.sh` (wget the open demo).
- `data/extract.py`: DuckDB reads raw CSV.gz → filters itemids/cohort → Parquet in `data/interim/`.
- `scripts/01_verify_items.py`: aggregate check of itemids (label, fluid, unit counts).
- `data/cohort.py`: cohort steps in `03_DATA_AND_COHORT.md` §5 with a flow table saved to `results/cohort/flow.csv`.
- `data/split.py`: patient-level splits + temporal split; saved as subject_id lists.

AC: runs end-to-end on demo and synthetic; flow table produced; no subject appears in two splits (test asserts); unit tests for dedup and plausibility filtering.

## Phase 2 — Labels (Week 4)
Tasks:
- `labels/actions.py`: action maps per lab from config; KDIGO creatinine staging using prior-48 h and prior-7 d minima.
- `labels/build.py`: $a^0$, $D$, $N$, $S$ per repeat event.
- Experiment **E1** script.

AC: unit tests with hand-computed cases for every boundary (e.g., HGB exactly 7.0, K exactly 5.5, creatinine +0.3 at 47 h vs 49 h); E1 outputs on demo.

## Phase 3 — Features (Weeks 5–6)
Tasks:
- `features/build.py`: features in `03_DATA_AND_COHORT.md` §7, computed with strictly-prior data.
- **Leakage test:** a test that perturbs any data at or after $t$ and asserts features for event $e$ are unchanged.

AC: leakage test passes; feature matrix per lab saved as Parquet with a feature manifest (name, description, group).

## Phase 4 — Models (Weeks 7–8)
Tasks:
- `models/baselines.py`: B1–B5.
- `models/quantile.py`: per-τ LightGBM, early stopping on val, rearrangement, CDF builder.
- `voi/edcp.py`: EDCP from CDF + action intervals; optional utility-matrix VOI; isotonic calibration.
- Experiment **E2** script with bootstrap CIs.

AC: EDCP ∈ [0,1]; unit test that EDCP equals analytic value on a toy Gaussian case; E2 table generated; hyperparameters logged.

## Phase 5 — Conformal certification (Week 9)
Tasks:
- `conformal/crc.py`: CRC threshold selection (§5.1), vectorized.
- `conformal/ltt.py`: LTT with Clopper–Pearson + fixed-sequence (§5.2).
- `conformal/mondrian.py`: group-wise CRC.
- `triage.py`: three-tier assignment.
- Experiments **E3**, **E4**.

AC: simulation test — on synthetic exchangeable data, mean realized risk over 500 resamples ≤ α + small tolerance; LTT FSR ≤ β in ≥ (1−δ) of resamples.

## Phase 6 — Explainability (Week 10)
Tasks:
- `explain/decompose.py`: ŷ, margin, width, ρ.
- `explain/shap_attr.py`: interventional TreeSHAP for q0.5, q0.1, q0.9 with shared background; φ^med, φ^unc.
- `explain/templates.py`: text explanation per tier.
- `explain/faithfulness.py`: deletion test.
- Experiment **E5**.

AC: SHAP additivity check (sum φ + base ≈ model output, tolerance 1e-3); template renders for synthetic examples; E5 outputs.

## Phase 7 — Full runs, ablations (Weeks 11–12)
- Run E1–E6 on full MIMIC-IV (**run locally**, share only `results/`).
- Ablations E6.
- Freeze results: tag git commit `results-v1`.

AC: all tables/figures in `results/` reproducible via `python scripts/run.py all --config configs/default.yaml`.

## Phase 8 — Minimal demo (Week 13, optional)
- `app/streamlit_app.py`: select a **synthetic or demo-dataset** admission + lab → show tier, EDCP, interval vs thresholds plot, explanation text. Banner: "Research prototype — not for clinical use."

AC: runs with `streamlit run app/streamlit_app.py` on synthetic data only.

## Phase 9 — Paper & thesis (Weeks 14–16)
- Fill `paper/` using `docs/06_PAPER_OUTLINE.md`; figures from `results/` via `scripts/make_figures.py`.
- Thesis chapters reuse docs 01–04.
