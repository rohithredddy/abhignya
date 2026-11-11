# CertiVOI

**Decision-anchored, conformally certified and explainable prediction of the information value of repeat laboratory tests.**

> Research prototype for an M.Tech thesis and IEEE conference paper. **Not for clinical use.** It is a decision-support research tool; it does not diagnose and does not cancel orders.

## Idea in one picture

```
 Patient history before draw ──► Quantile LightGBM ──► predictive distribution of new value
                                                          │
             action thresholds (e.g., Hgb < 7 g/dL) ──────┤
                                                          ▼
                              EDCP = P(new result changes the action category)
                                                          │
                       isotonic calibration ─► Conformal Risk Control (α)
                                                          ▼
                 LOW-VALUE (certified) │ POTENTIALLY USEFUL │ HIGH-VALUE
                                                          │
                 Explanation: margin to threshold vs. uncertainty (+ TreeSHAP reasons)
```

Builds on the value-of-information view of **DECIDE-Lab** (Allenbrand, *Diagnostics* 2026): a test is valuable only if its result can change the action. Unlike prior lab-reduction models that predict *normal* or *stable* results, CertiVOI predicts *action change*, guarantees a bound on missed action-changing results among tests it flags as low-value, and explains each prediction.

## Scope
- Labs: hemoglobin, potassium, creatinine (KDIGO AKI staging), platelets.
- Repeat tests within an adult hospital admission.
- Data: MIMIC-IV (credentialed) for experiments; MIMIC-IV demo (open) and synthetic data for development.
- CPU only.

## Repository layout
```
certivoi/
├── CONTRIBUTING.md         # working rules (read first)
├── PROGRESS.md             # living task/decision log
├── README.md
├── requirements.txt
├── pyproject.toml          # (created in Phase 0)
├── configs/default.yaml
├── docs/                   # proposal, methodology, data, experiments, roadmap, paper outline, references
├── src/certivoi/
│   ├── config.py
│   ├── data/      (extract.py, verify.py, cohort.py, split.py, synthetic.py)
│   ├── labels/    (actions.py, build.py)
│   ├── features/  (build.py)
│   ├── models/    (baselines.py, quantile.py, train.py)
│   ├── voi/       (edcp.py)
│   ├── conformal/ (crc.py, ltt.py, mondrian.py)
│   ├── triage.py
│   ├── explain/   (decompose.py, shap_attr.py, templates.py, faithfulness.py)
│   ├── eval/      (metrics.py)
│   ├── experiments/ (e1.py … e6.py)
│   ├── viz/       (style.py, figures.py)
│   └── utils/     (seed.py, io.py, logging.py)
├── scripts/       (00_download_demo.sh, 01_verify_items.py, run.py, make_figures.py)
├── tests/         (synthetic.py, test_actions.py, test_leakage.py, test_crc.py, ...)
├── app/           (streamlit_app.py — optional, synthetic/demo data only)
├── data/          # git-ignored: raw/, interim/, processed/
├── results/       # aggregate outputs only
└── paper/         # LaTeX (IEEEtran), figures/, tables/
```

## Quick start (development on open demo data)
```bash
# Python 3.11 (CONTRIBUTING.md §7). With uv:  uv venv --python 3.11 .venv
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scriptsctivate
pip install -r requirements.txt && pip install -e .
bash scripts/00_download_demo.sh                       # MIMIC-IV demo → data/raw/mimic-iv-demo
pytest -q                                              # 122 tests, synthetic data only
python scripts/run.py all --config configs/default.yaml   # ~2 min on the demo
python scripts/make_figures.py                         # results/ → paper/figures, paper/tables
```

No data at all? Set `data.mode: synthetic` (and a fresh `data.root`) in an override file:
the pipeline generates a synthetic MIMIC-schema dataset and runs end to end.

### Commands
| Command | What it does | Output |
|---|---|---|
| `run.py verify` | itemid / unit check against `d_labitems` (fails loudly) | `results/verify_items/` |
| `run.py extract` | raw CSV.gz → filtered Parquet (DuckDB) | `data/interim/extract/` |
| `run.py cohort` | cohort steps, repeat events, flow table | `results/cohort/flow.csv` |
| `run.py split` | patient-level + temporal splits | `data/interim/split/` |
| `run.py labels` | a0, D, N, S (+ KDIGO for creatinine) | `data/interim/labels/` |
| `run.py features` | strictly-prior features + manifest | `results/features/` |
| `run.py train [--lab L]` | quantile LightGBM, EDCP, isotonic, B1–B5 | `data/interim/models/` |
| `run.py exp E1..E6` | one experiment | `results/E*/` |
| `run.py all` | everything above, then E1–E6 | `results/` |

Common flags: `--override extra.yaml`, `--sample-frac 0.05`, `--force`. Stages cache
their outputs under `data/interim/<stage>/<hash>/`, keyed by the config keys (and code
version) each stage depends on.

## Status
Phases 0–7 and the Phase 9 paper skeleton are implemented and tested (see `PROGRESS.md`).
Results in `results/` currently come from the **open demo (100 patients) and are not
publishable**; the full MIMIC-IV run is done locally by the data owner. Phase 8
(Streamlit demo) is not started — it needs the `streamlit` dependency approved.

## Full MIMIC-IV runs
1. Obtain credentialed access on PhysioNet (CITI training + DUA).
2. Download `hosp/` and `icu/` to `data/raw/mimiciv/`.
3. Set `data.mode: full` and `data.root` in `configs/default.yaml`.
4. Smoke-test: `python scripts/run.py all --sample-frac 0.05`.
5. Full: `python scripts/run.py all`.
6. Share only `results/` with collaborators (see `CONTRIBUTING.md` §4).

## Experiments
| ID | Question | Main output |
|---|---|---|
| E1 | Do proxy labels (normal/stable) disagree with action-change labels? | Table 2, Fig. 2 |
| E2 | How well is action change predicted? | Table 3, Fig. 3 |
| E3 | Certified reduction vs α | Table 4, Figs. 4–5 |
| E4 | Subgroup & temporal robustness | Table 5, Fig. 6 |
| E5 | Explanation faithfulness | Table 6, Figs. 7–8 |
| E6 | Ablations | Table 7 |

## Hardware
Tested target: 8-core CPU, 16 GB RAM (8 GB works with `--sample-frac`). No GPU.

## Ethics & data
Uses de-identified MIMIC-IV data under the PhysioNet Credentialed Health Data Use Agreement. No row-level data leaves the local machine. Outputs are aggregate. Research prototype only.

## Citation
If you use this code, cite the paper (TBA) and the base paper:
C. Allenbrand, "DECIDE-Lab: A Value-of-Information and POMDP Framework for Diagnostic Laboratory Test Selection," *Diagnostics*, 16(15):2356, 2026.
