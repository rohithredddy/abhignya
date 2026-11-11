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
│   ├── data/      (extract.py, cohort.py, split.py)
│   ├── labels/    (actions.py, build.py)
│   ├── features/  (build.py)
│   ├── models/    (baselines.py, quantile.py)
│   ├── voi/       (edcp.py)
│   ├── conformal/ (crc.py, ltt.py, mondrian.py)
│   ├── triage.py
│   ├── explain/   (decompose.py, shap_attr.py, templates.py, faithfulness.py)
│   ├── eval/      (metrics.py, bootstrap.py)
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
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
bash scripts/00_download_demo.sh                       # MIMIC-IV demo → data/raw/mimic-iv-demo
pytest -q
python scripts/run.py all --config configs/default.yaml
```

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
