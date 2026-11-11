# PROGRESS.md — living log (updated every session)

## Current status
- **Phase:** 1 — Data extraction (Phase 0 done 2026-11-11)
- **Data access:** [ ] PhysioNet account  [ ] CITI training  [ ] MIMIC-IV approved  [x] Demo usable (open)
- **data.mode in config:** demo

## Phase checklist (see docs/05_IMPLEMENTATION_ROADMAP.md)
- [x] Phase 0 — Scaffold
- [ ] Phase 1 — Data extraction
- [ ] Phase 2 — Labels (+ E1)
- [ ] Phase 3 — Features (+ leakage test)
- [ ] Phase 4 — Models (+ E2)
- [ ] Phase 5 — Conformal (+ E3, E4)
- [ ] Phase 6 — Explainability (+ E5)
- [ ] Phase 7 — Full runs + ablations (E6)
- [ ] Phase 8 — Demo (optional)
- [ ] Phase 9 — Paper & thesis

## Current phase tasks
### Phase 0 — Scaffold (done 2026-11-11)
- [x] Repo layout, `pyproject.toml` (src layout, package `certivoi`, ruff + pytest config), `.gitignore` (+ egg-info, caches)
- [x] `src/certivoi/config.py`: YAML + optional override file + dict overrides, validation, typed access (`LabSpec`), 12-char SHA-256 config hash
- [x] `utils/logging.py`, `utils/seed.py` (`set_all`), `utils/io.py` (`write_results` → CSV/PNG + `meta.json`; refuses identifier columns / `rows*` names; `suppress_small_counts`; `interim_path` keyed by config hash)
- [x] `tests/synthetic.py`: MIMIC-schema generator (patients, admissions, labevents, d_labitems, icustays); OU trajectories + bleed/AKI/thrombocytopenia/hypoK episodes; optional artifacts (minors, <24 h stays, duplicates, null/implausible values, wrong units, blood-gas K 50822); `write_mimic_layout` → hosp/ + icu/ csv.gz
- [x] `scripts/run.py` CLI: info, extract, cohort, split, labels, features, train, exp, all (`--config`, `--override`, `--sample-frac`, `--force`, `--lab`)
- AC: `pytest -q` → 39 passed; `ruff check src tests scripts` clean; `python scripts/run.py --help` works ✅

### Phase 1 — Data extraction (next)
- [ ] `scripts/00_download_demo.sh` (exists; human runs it — needs wget/bash)
- [ ] `data/extract.py` (DuckDB raw CSV.gz → Parquet in `data/interim/`)
- [ ] `scripts/01_verify_items.py`
- [ ] `data/cohort.py` + `results/cohort/flow.csv`
- [ ] `data/split.py` (patient-level + temporal)

## Decision log
| Date | Decision | Reason | Doc updated |
|---|---|---|---|
| (start) | Scope fixed to 4 labs, repeat tests, CPU tree models + CRC | M.Tech scope, CPU constraint | 01, 02 |
| 2026-11-11 | Dev env: `.venv` with CPython 3.11.15 created via `uv` (machine has 3.10/3.14 only) | CONTRIBUTING.md §7 requires 3.11 | — |
| 2026-11-11 | Config hash covers the whole resolved config (incl. `sample_frac`), so interim caches differ per sample fraction | Avoid mixing sampled and full caches | — |
| 2026-11-11 | `write_results` raises if a table has subject_id/hadm_id/stay_id/labevent_id/specimen_id columns | Enforce CONTRIBUTING.md §4.6 in code | — |
| 2026-11-11 | Added CLI subcommands `info` (prints config hash, no data access) and `split` beyond the CONTRIBUTING.md §8 list | Split is its own Phase 1 task | — |
| 2026-11-11 | Synthetic generator lives in `tests/synthetic.py` (per roadmap). Unimplemented CLI stages exit with code 2 naming their phase | Keep tests-only data out of the package | — |

## Open questions
- Confirm action thresholds with a clinician (docs/02 §2) if one is available.
- Target conference and deadline?
- The folder is not a git repository yet, so Phase 0 is uncommitted. Run `git init` (and commit `phase0: scaffold`)?
- Phase 1 needs the MIMIC-IV demo in `data/raw/mimic-iv-demo` (`bash scripts/00_download_demo.sh`, needs wget).

## Results snapshot (aggregate only; fill from results/)
| Lab | AUROC(D) | AUPRC(D) | Reduction @ α=1% | Realized risk @ α=1% |
|---|---|---|---|---|
| HGB | TODO | TODO | TODO | TODO |
| K | TODO | TODO | TODO | TODO |
| CREAT | TODO | TODO | TODO | TODO |
| PLT | TODO | TODO | TODO | TODO |
