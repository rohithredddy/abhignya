# PROGRESS.md — living log (updated every session)

## Current status
- **Phase:** 0–8 implemented and tested on demo + synthetic data; Phase 9 paper prose written (numbers `TODO(result)`), thesis outline in `docs/07_THESIS_OUTLINE.md`. **Next: full MIMIC-IV run (human, locally)**, then fill numbers from `results/`.
- **Data access:** [ ] PhysioNet account  [ ] CITI training  [ ] MIMIC-IV approved  [x] Demo usable (open, downloaded to `data/raw/mimic-iv-demo`)
- **data.mode in config:** demo
- **Repo:** https://github.com/rohithredddy/abhignya (branch `main`, one commit per phase)
- **Checks:** `pytest -q` → 125 passed; `ruff check src tests scripts app` clean; `python scripts/run.py all` on demo ≈ 2 min. CI: `.github/workflows/ci.yml` (ruff + pytest on synthetic data; paper compile → `paper-pdf` artifact).

## Phase checklist (see docs/05_IMPLEMENTATION_ROADMAP.md)
- [x] Phase 0 — Scaffold
- [x] Phase 1 — Data extraction
- [x] Phase 2 — Labels (+ E1)
- [x] Phase 3 — Features (+ leakage test)
- [x] Phase 4 — Models (+ E2)
- [x] Phase 5 — Conformal (+ E3, E4)
- [x] Phase 6 — Explainability (+ E5)
- [~] Phase 7 — Full runs + ablations (E6): code + demo run done; **full MIMIC-IV run and `results-v1` tag pending (human)**
- [x] Phase 8 — Demo: `app/streamlit_app.py` + `certivoi/demo.py` (synthetic/demo data only)
- [~] Phase 9 — Paper & thesis: prose, pipeline figure, auto-included tables/figures, refs, thesis outline done; result numbers need the full run

## Phase tasks and acceptance criteria
### Phase 0 — Scaffold ✅
- [x] pyproject (src layout), config loader (typed, validated, hashed), logging/seed/io utils, synthetic MIMIC-schema generator, CLI.
- AC: tests pass, ruff clean, `run.py --help` works.

### Phase 1 — Data extraction ✅
- [x] `scripts/00_download_demo.sh` (wget or curl) — demo downloaded.
- [x] `data/extract.py` (DuckDB → Parquet; `--sample-frac` by patient), `data/verify.py` + `scripts/01_verify_items.py` / `run.py verify` (all 15 itemids verified on demo: labels, fluid, units).
- [x] `data/cohort.py` (adult, ≥24 h, non-null, unit, plausibility, dedup keep-first by labevent_id, repeat ≤72 h, patient-level cap) → `results/cohort/flow.csv`.
- [x] `data/split.py` (60/10/15/15 by subject; temporal ≤2016 vs 2017–2019).
- AC: runs on demo + synthetic; flow table; disjoint splits asserted; dedup + plausibility tests ✅

### Phase 2 — Labels ✅
- [x] `labels/actions.py` (interval maps with per-boundary closure; KDIGO stage + per-event edges), `labels/build.py` (y_last, a0, D, N, S, edges), E1.
- AC: hand-computed boundary tests (HGB 7.0, K 5.5, PLT 10/50, creatinine +0.3 at 47 h vs 49 h, float 1.0+0.3) ✅; E1 on demo ✅

### Phase 3 — Features ✅
- [x] `features/build.py` (DuckDB windows + ASOF joins, 53–55 features/lab, manifest name/description/group).
- [x] `tests/test_leakage.py`: perturbs all raw lab rows at/after a per-admission cut-off, re-runs from raw; features of all earlier events identical (all 4 labs). Mutation-checked: a planted `<= t` leak makes it fail.
- AC ✅

### Phase 4 — Models ✅
- [x] B1 rule, B2/B3/B5 LightGBM, B4 logistic; per-τ quantile LightGBM with early stopping, rearrangement, CDF, EDCP, isotonic; CREAT target auto-selected; E2 with 1000× patient bootstrap; hyperparameters → `results/E2/hyperparams.json`.
- AC: EDCP ∈ [0,1]; EDCP = analytic Gaussian value (|err| ≤ 1.5e-3 dense τ; ≤ 0.02 with config τ); VOI(0–1 regret) = EDCP; E2 table ✅

### Phase 5 — Conformal ✅
- [x] `conformal/crc.py`, `ltt.py`, `mondrian.py`, `triage.py`; E3 (100 resamples, LTT, natural thresholds, CRC on all scores, tiers), E4 (subgroups global vs Mondrian, temporal + recalibration, threshold sensitivity).
- AC: 500-resample simulation: mean CRC risk ≤ α(1+0.1) for α ∈ {0.01, 0.05}; LTT population FSR ≤ β in ≥ 90% of resamples ✅

### Phase 6 — Explainability ✅
- [x] `explain/decompose.py`, `shap_attr.py` (interventional TreeSHAP), `templates.py`, `faithfulness.py`; E5.
- AC: SHAP additivity ≤ 1e-3 (demo: ≤ 3e-6 all labs/τ) ✅; templates render ✅; E5 outputs ✅

### Phase 7 — Full runs, ablations (partial)
- [x] E6 ablations (Δ vs absolute, no trend, no panel, no isotonic, B5, B4; ICU-intervention ablation reported as not run).
- [x] `run.py all` reproduces every table/figure (tested end-to-end on synthetic via the CLI).
- [ ] Human: full MIMIC-IV run → share `results/` only → tag `results-v1`.

### Phase 8 — Demo ✅
- [x] `src/certivoi/demo.py` (pure back end: `prepare_lab`, `explain_event`, `interval_figure`; refuses `data.mode: full`), `app/streamlit_app.py` (lab, α, admission, repeat test → tier, P(action change), predicted CDF vs thresholds, template explanation, banner "Research prototype — not for clinical use").
- AC: `streamlit run app/streamlit_app.py -- --override configs/synthetic.yaml` serves the app; exercised headlessly with Streamlit `AppTest` (no exceptions, HGB and CREAT); `tests/test_demo.py` ✅

### Phase 9 — Paper (partial)
- [x] Intro, related work, setup, discussion, conclusion written (no result numbers); TikZ pipeline figure (Fig. 1); `\maybetable`/`\maybefig` include generated tables/figures when present. `docs/07_THESIS_OUTLINE.md` (chapters ↔ docs ↔ results files, submission checklist).
- [x] `paper/main.tex` (IEEEtran; method written; all numbers `TODO(result)`), `paper/refs.bib` (only entries from docs/REFERENCES.md; verify all), `scripts/make_figures.py` (figures + booktabs tables with source-CSV comments; writes `paper/tables/DATA_MODE.txt`).

## Decision log
| Date | Decision | Reason | Doc updated |
|---|---|---|---|
| (start) | Scope fixed to 4 labs, repeat tests, CPU tree models + CRC | M.Tech scope, CPU constraint | 01, 02 |
| 2026-11-11 | Dev env: `.venv` with CPython 3.11.15 via `uv` (machine has 3.10/3.14) | CONTRIBUTING.md §7 requires 3.11 | — |
| 2026-11-11 | `write_results` refuses tables with subject_id/hadm_id/stay_id/labevent_id/specimen_id columns or `rows*` names; small-cell suppression (n<10) applied automatically in `full` mode | Enforce CONTRIBUTING.md §4 in code | — |
| 2026-11-11 | Interim caches keyed per stage by only the config keys that stage uses + code versions of it and upstream stages (`utils/io.py: STAGE_KEYS, STAGE_CODE_VERSION`) | Changing α must not force a 60-min re-extraction; code changes must invalidate stale models | — |
| 2026-11-11 | Synthetic generator moved to `certivoi/data/synthetic.py`; `tests/synthetic.py` re-exports it | `data.mode: synthetic` runs the full CLI pipeline (and Phase 8 can use it) | — |
| 2026-11-11 | K boundary: exactly 5.5 = no action (`at_boundary_upper: [true, false]`); other labs left-closed | Match 02 §2 table (">5.5 treat") — config comment said "left-closed" | 02 §2, config |
| 2026-11-11 | Fallback reference ranges in config (`labs.*.ref_range`) for label N when labevents ref range is null | 02 §2 "falling back to config" | config |
| 2026-11-11 | KDIGO: rule as MIMIC-code `kdigo_creatinine`; baselines within the same admission; 1e-9 float tolerance | Standard implementation; float 1.0+0.3 ≠ 1.3 | 02 §2 |
| 2026-11-11 | CREAT target: `model.creat_target: auto` fits Δ and absolute, keeps lower val pinball (demo chose Δ) | 02 §3 asks to pick by validation pinball | 02 §3 |
| 2026-11-11 | LTT grid = 100 score order statistics from n_min = ⌈log δ / log(1−β)⌉ to n (label-free) | A grid of all unique scores starts at n=1 and can never reject → LTT always flags nothing | 02 §5.2, config |
| 2026-11-11 | LTT test of the AC uses population FSR (analytic for Beta scores), not finite-test empirical FSR | LTT guarantees FSR(λ̂) ≤ β w.p. 1−δ; empirical test FSR adds sampling noise | — |
| 2026-11-11 | Quantile boosters truncated at best iteration; features rounded (4 dp, hours 2 dp) and stored float32-exact; boosters rewritten (thresholds → float32 floor, NaN default = where 0 goes) | Needed for exact TreeSHAP additivity (diagnosed by brute-force Shapley on a failing tree); predictions unchanged (asserted) | 02 §6, 03 §7 |
| 2026-11-11 | B1 rule does not fire when the second-last value is missing | Δ(last two) undefined | — |
| 2026-11-11 | E3 natural threshold for every score = 0.5 (`baselines.natural_threshold`) | 04 E3 mode (a) | config |
| 2026-11-11 | E5 example explanations written only when `data_mode != full` | CONTRIBUTING.md §4 / 06 writing rules | — |
| 2026-11-11 | E6 ICU-intervention ablation reported as "not run" | Needs `icu.inputevents` extraction (optional in 03 §7) | — |
| 2026-11-11 | LaTeX tables written by a small helper, not `DataFrame.to_latex` | pandas needs jinja2 for that; no new dependency | — |
| 2026-11-11 | `streamlit>=1.35` added as optional extra `[demo]` (not a core dependency) | Human: "add what are necessary and complete project" (taken as approval for the Phase 8 dependency) | README |
| 2026-11-11 | Demo refuses `data.mode: full` and shows only test-split events | CONTRIBUTING.md §4 (row-level display) | README |
| 2026-11-11 | Override files `configs/synthetic.yaml`, `configs/full.yaml` | One-flag switch between modes | README |
| 2026-11-11 | Template explanations print "missing" for NaN feature values | "= n/a" read poorly in the demo | — |

## Observations on demo data (NOT publishable; 100 patients)
- E1: proxies disagree with D strongly — e.g. 75% of action-changing K results were inside the reference range (`results/E1/table2_labels.csv`).
- E3: mean realised risk over 100 resamples ≤ α for every lab and α. Proxy baselines at natural thresholds miss more action-changing results (e.g. stability classifier B3, HGB: risk 0.119).
- **Isotonic calibration on the tiny demo val split (10 patients) collapses scores into ties**: HGB AUROC 0.84 (raw EDCP) → 0.71 (calibrated), and CRC then certifies nothing for HGB/K at α ≤ 2%. E6 `no_isotonic` certifies far more (HGB: 77% at α = 5%). Expected to disappear with full-MIMIC val sizes — check E6 on the full run; if it persists, consider CRC on raw EDCP (validity holds either way). **Do not change the default without the user.**
- PLT has ≈1 action-changing result in the demo test split → its metrics are meaningless on demo.
- E4 temporal shift skipped on demo: demo has no 2017–2019 patients.
- E5: Spearman(ρ, EDCP) ≈ −0.93 (HGB, K, CREAT); top-k SHAP deletions change EDCP more than random-k (Wilcoxon p < 0.01, all labs).

## Open questions
- Confirm action thresholds with a clinician (docs/02 §2), in particular K = 5.5 → no action.
- Target conference and deadline?
- After the full run: if isotonic still hurts CRC reduction (E6), decide whether to certify on raw EDCP.
- `results/` currently holds demo aggregates and is committed. Keep committing results (fine for aggregates) or ignore and attach to releases?

## How to run the full experiment
1. Get MIMIC-IV access; put `hosp/` and `icu/` under `data/raw/mimiciv/`.
2. Override file `configs/full.yaml` (already in the repo): `data: {mode: full, root: data/raw/mimiciv}`.
3. `python scripts/run.py verify --override configs/full.yaml` → inspect `results/verify_items/items.csv` if it fails.
4. Smoke test: `python scripts/run.py all --override configs/full.yaml --sample-frac 0.05`.
5. Full: `python scripts/run.py all --override configs/full.yaml`, then `python scripts/make_figures.py`.
6. Share only `results/` (aggregates; small cells already suppressed in full mode).

## Results snapshot (aggregate only; DEMO data — not publishable)
Source: `results/E2/table3_performance.csv`, `results/E3/table4_certified.csv` (resample mean; main test split in brackets).

| Lab | AUROC(D) | AUPRC(D) | Reduction @ α=1% | Realized risk @ α=1% |
|---|---|---|---|---|
| HGB | 0.712 | 0.328 | 0.000 (0.000) | 0.0000 (0.0000) |
| K | 0.791 | 0.372 | 0.000 (0.000) | 0.0000 (0.0000) |
| CREAT | 0.777 | 0.386 | 0.215 (0.277) | 0.0058 (0.0061) |
| PLT | 0.500 | 0.003 | 0.410 (0.000) | 0.0082 (0.0000) |
