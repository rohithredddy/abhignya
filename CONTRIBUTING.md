# CONTRIBUTING.md — Working rules for CertiVOI

Read this file before making changes, then read `PROGRESS.md` to see where the project stands.

## 1. What this project is (in 6 lines)
CertiVOI predicts, **before a repeat lab test is drawn**, whether its result is likely to **change the clinical action** (decision-anchored information value, from the DECIDE-Lab base paper). A LightGBM quantile model forecasts the new value; the probability mass outside the current action category is the **EDCP** score. **Conformal risk control** certifies the "low-value" tier (missed action-changing results ≤ α). A **margin vs. uncertainty** decomposition with TreeSHAP explains each prediction. M.Tech project → IEEE mid-tier conference paper.

## 2. Source-of-truth documents (read the relevant one before coding)
| Need | File |
|---|---|
| Why / novelty / scope | `docs/01_RESEARCH_PROPOSAL.md` |
| Definitions, equations, baselines | `docs/02_METHODOLOGY.md` ← code must match this |
| Data tables, itemids, cohort, features | `docs/03_DATA_AND_COHORT.md` |
| Experiments E1–E7, metrics, outputs | `docs/04_EXPERIMENT_PLAN.md` |
| Phases, tasks, acceptance criteria | `docs/05_IMPLEMENTATION_ROADMAP.md` |
| Paper structure | `docs/06_PAPER_OUTLINE.md` |
| Citations | `docs/REFERENCES.md` |

If code must deviate from a doc, **update the doc and log the decision in `PROGRESS.md`** in the same change.

## 3. Scope guardrails
**In scope:** 4 labs (HGB, K, CREAT, PLT); repeat tests only; MIMIC-IV (+ open demo, PhysioNet 2019 Challenge, eICU demo); LightGBM / logistic regression / isotonic; CRC, LTT, Mondrian CRC and the E7 extensions; TreeSHAP; template explanations; optional minimal Streamlit demo on synthetic/demo data.

**Out of scope:**
- RL, POMDP solvers, deep learning (LSTM/Transformer/GNN), PyTorch/TensorFlow.
- Generated free-text explanations or calls to external text-generation services inside the project code.
- Extra labs, imaging, free-text notes, first-time (non-repeat) tests.
- Hyperparameter sweeps beyond the small grid in config.
- New datasets beyond those listed above without discussion.
- Web services, database servers, Docker, cloud deployment.

Anything out of scope needs a one-paragraph justification and agreement before work starts.

## 4. Privacy rules (PhysioNet DUA) — hard rules
Full MIMIC-IV is credentialed data. Row-level full-MIMIC data never leaves the local machine.
1. Check `data_mode` in config. If `full`: never print, preview or open row-level files under `data/raw`, `data/interim`, `data/processed`, and never write code that prints rows to stdout.
2. In `full` mode only schemas (column names/dtypes), row counts and aggregate statistics where every cell count ≥ 10 may be inspected or shared.
3. Full-data commands run locally; only the contents of `results/` (aggregates) are shared.
4. Row-level inspection is allowed only for open-access data (`data_mode: demo` or `open`) or synthetic data.
5. Tests and fixtures use `tests/synthetic.py` only. Never commit MIMIC rows, never put them in notebooks, figures, the paper, or the demo app.
6. Any script that writes per-event outputs writes them under `data/` (git-ignored), never `results/`.

## 5. Leakage rules — hard rules
- Features for an event at time *t* use only data with timestamps `< t − 1 min`.
- Never use: `diagnoses_icd`, `drgcodes`, `dischtime`/LOS, discharge location, `hospital_expire_flag`, or any future lab.
- Splits are by `subject_id`. Fit imputers/scalers/calibrators only on their designated split.
- The **calibration split** is used only for conformal threshold selection. The **test split** is read only by final evaluation scripts. Never tune on test.
- Race and insurance are **never features**; only for subgroup evaluation.
- The leakage unit test (`tests/test_leakage.py`) must pass before any model work is merged.

## 6. Compute rules
- No GPU code paths.
- Use DuckDB for anything touching raw `labevents`; never load raw labevents into pandas.
- Thread count from `config.compute.threads`; pass to DuckDB and LightGBM.
- Before any full run, test with `--sample-frac` (e.g., 0.05).
- Keep peak RAM < 12 GB; use Parquet + column selection; process labs one at a time.
- Cache intermediate artifacts in `data/interim/` keyed by config hash; skip recompute if present (`--force` to override).

## 7. Tech stack & conventions
- Python 3.11, src layout, package `certivoi`.
- Libraries: duckdb, polars (or pandas for small frames), pyarrow, numpy, scipy, scikit-learn, lightgbm, shap, matplotlib, pyyaml, pytest, ruff. Discuss before adding any other dependency.
- Type hints on public functions; docstrings state inputs, outputs, and which doc section they implement (e.g., `Implements 02_METHODOLOGY §5.1`).
- Pure functions for math (CRC, LTT, EDCP, action maps) — easy to unit test.
- Config-driven: no magic numbers in code; thresholds/α/grids live in `configs/default.yaml`.
- Seeds via `certivoi.utils.seed.set_all(seed)`.
- Every experiment writes `results/<exp_id>/` with CSV/PNG + `meta.json` (config hash, git commit, data_mode, seed, runtime, sample_frac).
- Plots: matplotlib only, consistent style from `certivoi/viz/style.py`.

## 8. Commands
```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                  # all tests (synthetic data)
ruff check src tests scripts
python scripts/run.py all       --config configs/default.yaml --sample-frac 0.05
python scripts/run.py convert   --override configs/challenge2019.yaml
python scripts/run.py all       --override configs/challenge2019.yaml
python scripts/paper_tables.py                       # results/ -> paper/tables, paper/figures
streamlit run app/streamlit_app.py -- --override configs/synthetic.yaml   # optional demo
```

## 9. Workflow
1. Read `PROGRESS.md` → pick the next unchecked task of the current phase in `docs/05_IMPLEMENTATION_ROADMAP.md`.
2. Read the doc sections that task implements.
3. Write tests first for math/labels/leakage; then implementation.
4. Run `pytest -q` and `ruff check`. Fix before moving on.
5. Update `PROGRESS.md`: tick the task, note decisions and open questions.
6. Keep changes small and focused; one task per commit message (`phaseN: <task>`).
7. Do not start the next phase until the current phase's acceptance criteria pass.

## 10. Scientific honesty rules
- Never fabricate numbers, results, or citations. If a number is needed for text and no `results/` file exists, write `TODO(result)`.
- Report all labs and all α levels, including unfavorable outcomes.
- When reduction is small at tight α, report it plainly; it is a valid finding.
- Phrase novelty claims as "to our knowledge".
- The system is decision support; never write that it diagnoses or that it is clinically validated.

## 11. Stop and review before proceeding when
- `scripts/01_verify_items.py` reports any itemid/unit mismatch.
- Action thresholds or label definitions would change.
- A new dependency, dataset, or out-of-scope idea comes up.
- Results look too good (e.g., AUROC > 0.97 for D) → suspect leakage; investigate and document before continuing.
