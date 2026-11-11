# 07 — M.Tech Thesis Outline

The thesis reuses docs 01–04 (05_IMPLEMENTATION_ROADMAP Phase 9: "Thesis chapters reuse docs 01–04") and the paper (`paper/main.tex`). The same writing rules apply as for the paper (06 "Writing rules"): every number comes from `results/` produced by the **full** MIMIC-IV run; demo and synthetic numbers are never reported as findings; novelty is phrased "to our knowledge"; no patient-level data anywhere.

| Ch. | Title | Source material | Results files |
|---|---|---|---|
| 1 | Introduction: problem, motivation, research questions RQ1–RQ4, contributions C1–C4, thesis organisation | 01 §1, §4, §6, §7; paper §I | — |
| 2 | Background and literature review: value of information and the threshold approach; DECIDE-Lab; ML for low-yield labs; RL / active feature acquisition; conformal prediction, CRC, LTT, Mondrian; quantile regression; SHAP | 01 §2–§3; REFERENCES.md (verify every entry); paper §II | — |
| 3 | Data and cohort: MIMIC-IV and the open demo; itemids and units; cohort rules; leakage controls; splits; ethics and the PhysioNet DUA (how the workflow kept row-level data local) | 03 (all); CONTRIBUTING.md §4–§5 | `results/cohort/flow.csv`, `results/verify_items/` |
| 4 | Methodology: action maps (incl. KDIGO); label D and proxies N, S; quantile forecaster; EDCP and Proposition 1 with proof; isotonic calibration; CRC, LTT (grid from n_min), Mondrian; triage; margin/uncertainty decomposition; TreeSHAP (incl. the SHAP-consistency implementation note); templates | 02 (all); paper §III | — |
| 5 | Implementation: architecture (`src/certivoi` modules), config hashing and stage caching, CPU/RAM budget, testing strategy (synthetic generator, leakage mutation test, CRC simulation tests), reproducibility (`run.py all`, `meta.json`), demo app | README; PROGRESS decision log; `tests/` | `results/*/meta.json` (runtimes) |
| 6 | Experiments and results: E1 labels vs proxies; E2 prediction; E3 certified reduction (all labs, all α); E4 subgroups, temporal shift, threshold sensitivity; E5 explanations and faithfulness; E6 ablations | 04 (all); paper §IV–§V | `results/E1`–`results/E6` |
| 7 | Discussion: interpretation of the certificate; conservativeness at tight α; selection bias and confounding by intervention; threshold choice; exchangeability and single-centre data; myopic vs sequential value | paper §VI; 01 §10 | — |
| 8 | Conclusion and future work | paper §VII | — |
| App. A | Configuration (`configs/default.yaml`) and hyperparameters | config; `results/E2/hyperparams.json` | |
| App. B | Feature list with descriptions | `results/features/` (manifest) | |
| App. C | Additional tables: per-α, per-subgroup, sensitivity variants | `results/E3`, `results/E4` | |
| App. D | Demo app screenshots (synthetic data only) | `app/streamlit_app.py` | |

## Checklist before submission
- [ ] Full MIMIC-IV run done; `results-v1` tag created; `paper/tables/DATA_MODE.txt` says `full`.
- [ ] Every `TODO(result)` replaced by a number with its `results/` path in a comment.
- [ ] Every reference verified against the publisher (`paper/refs.bib` header).
- [ ] Action thresholds confirmed with a clinician (PROGRESS open questions).
- [ ] Unfavourable results reported (small reductions at tight α, any subgroup or temporal violations).
- [ ] Ethics statement: decision support only; not clinically validated.
