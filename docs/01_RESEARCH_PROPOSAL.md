# 01 — Research Proposal: CertiVOI

**Working title (paper):** *CertiVOI: Decision-Anchored, Conformally Certified and Explainable Prediction of the Information Value of Repeat Laboratory Tests*

**Program:** M.Tech research project → IEEE mid-tier conference paper (e.g., IEEE BIBM, ICHI, CBMS, BHI, EMBC — check current CFPs).
**Compute constraint:** CPU only (no GPU). Target: laptop/desktop with 16 GB RAM (8 GB workable with subsampling).

---

## 1. Problem statement

Many inpatient laboratory tests are repeated even when the evidence already available (prior values, trends, related labs, clinical context) makes it very unlikely that the new result will change any clinical decision. We want to predict, **before a repeat test is performed**, whether its result is likely to carry *action-changing* information, classify the test as **low-value / potentially useful / high-value**, attach a **statistical safety guarantee** to the low-value label, and **explain** the prediction to a clinician.

The system is decision support only. It never diagnoses, never cancels orders, and never replaces clinician judgment.

## 2. Base paper and what it leaves open

**Base paper:** C. Allenbrand, "DECIDE-Lab: A Value-of-Information and POMDP Framework for Diagnostic Laboratory Test Selection," *Diagnostics* 16(15):2356, 2026. doi:10.3390/diagnostics16152356.

Key ideas we inherit:
- Lab ordering is a **decision problem**, not a test-accuracy problem.
- A test has positive value **only if at least one possible result changes the action** chosen under the current belief.
- High accuracy does not imply high value when posterior beliefs do not cross **action thresholds**.
- Value is **contextual** (patient- and subgroup-specific), not intrinsic to the test.

What DECIDE-Lab does **not** do (our opening):
1. It is a theoretical framework with hand-specified priors, utilities and worked examples. It does not **learn** patient-specific beliefs from EHR data.
2. It gives no **empirical safety guarantee** for recommending that a test be skipped.
3. Its explanation is the decision model itself; there is no patient-level, feature-level explanation of *why* a given repeat test is low value.
4. Its POMDP machinery is heavy for a CPU M.Tech project; we deliberately use the **single-step (myopic) VOI** special case.

## 3. Related work (condensed)

| Line of work | Representative papers | What they predict | Limitation relative to our goal |
|---|---|---|---|
| Decision-theoretic VOI for labs | Allenbrand 2026 (DECIDE-Lab); Howard 1966 (VOI) | Expected value of information from specified priors/utilities | No learning from data; no guarantees |
| Predicting normal results | Xu et al., *JAMA Netw Open* 2019 (Stanford, 12 labs, AUROC ≥ 0.90) | P(result in reference range) | Normal ≠ "won't change decision"; no reduction policy or guarantee |
| Low-yield / stability prediction | Roy et al. (low-information labs, RF AUROC ~0.88); Aikens et al., AMIA 2023 (*Probabilistic Prediction of Lab Test Information Yield*) | Stability of repeat value, with uncertainty | Label is a proxy (stability), not action change; uncertainty not converted to a guarantee |
| Deep reduction strategies | Yu et al., *IJMI* 2020 (MIMIC-III, ~20% reduction); Huang et al., *BMC MIDM* 2023 (confidence-based Hgb reduction, ~10% reduction) | Normality / transition / value, with "confidence" | Heuristic confidence, no finite-sample guarantee; LSTMs (GPU-friendly, less transparent) |
| Change-prediction for specific labs | Lactate trajectory prediction (MIMIC-III/eICU, LSTM, arXiv 2107.07582) | Deterioration of one analyte | Single lab, no decision anchoring |
| Implementation studies | Rabbani et al., *Clin Biochem* 2023 (Stanford SMART-on-FHIR pre-implementation) | — | Physicians wanted ~85–90% certainty before discontinuing/alerting → motivates calibrated, certified outputs |
| RL / active feature acquisition | SM-DDPO (Yu et al., ICLR 2023); Li & Oliva-style AFA; IJCAI 2026 AFA survey; cost-aware AFA on MIMIC-IV (medRxiv 2026) | Sequential test-ordering policy | GPU-heavy, hard to explain, usually targets diagnosis accuracy rather than per-test value |
| Conformal decision support | Conformal risk control (Angelopoulos et al. 2022); Learn-then-Test (Angelopoulos et al. 2021); StratCP (act-or-defer with FDR control, 2026); ConfGuide (FNR-capped guidance, 2026) | Error-controlled selection | Not applied to lab-test deferral with decision-threshold labels (as far as our search found) |

> Note: before writing the paper's related-work section, re-verify each citation in `docs/REFERENCES.md` (authors, venue, year). Do not invent references.

## 4. Research gap (one paragraph for the paper)

Existing data-driven methods label a repeat test "unnecessary" when its result is predicted to be *normal* or *stable*. Decision theory (DECIDE-Lab) says the correct criterion is whether the result can **change the clinical action**. A normal-range result can still change an action (e.g., creatinine rising 0.3 mg/dL within 48 h → AKI stage change while still "normal"), and an abnormal result can be completely uninformative (e.g., a 4th consecutive stable Hgb of 7.8 g/dL). Moreover, no prior lab-reduction method provides a **distribution-free, finite-sample guarantee** on how many action-changing results are missed among tests flagged low-value, which is exactly what clinicians asked for in implementation studies. Finally, explanations are generic feature attributions rather than VOI-native reasons ("far from the threshold" vs. "low uncertainty").

## 5. Proposed approach (overview)

1. **Decision-anchored label.** For each repeat test of lab *L*, map values to guideline-based **action categories** (e.g., Hgb < 7 g/dL → transfusion consideration). The *default action* is the category implied by the last available value (carry-forward belief). Label `D = 1` if the new result falls in a **different action category** than the default.
2. **Distributional forecaster (CPU).** LightGBM **quantile regression** on Δ = (new value − last value) produces a predictive distribution of the new result from information strictly before the draw.
3. **EDCP score.** *Expected Decision-Change Probability* = predictive probability mass outside the default action's interval. We show EDCP equals single-step VOI under 0–1 action-regret, linking directly to DECIDE-Lab; a general utility matrix is an optional extension.
4. **Certified triage.** Isotonic calibration, then **Conformal Risk Control** chooses the low-value threshold so that the expected rate of missed action-changing results is ≤ α (finite-sample, distribution-free, under exchangeability). Learn-then-Test variant bounds the false-skip rate *among flagged tests*; Mondrian (per-group) variant addresses subgroup equity.
5. **VOI-native explanation.** Decompose each prediction into **margin** (how far the predicted value is from the nearest action boundary) and **uncertainty** (predictive interval width), and attribute each to features with TreeSHAP. Render a short template explanation. Evaluate faithfulness with deletion tests.

## 6. Research questions

- **RQ1 (Label validity):** How often do proxy labels (normal / stable) disagree with decision-anchored labels on real data?
- **RQ2 (Prediction):** Can a CPU-trained distributional model estimate EDCP with good discrimination and calibration, compared with proxy-label and direct classifiers?
- **RQ3 (Safety & utility):** Does conformal risk control achieve its target miss-rate on held-out data (including subgroups and a later time period) while still flagging a clinically useful fraction of repeat tests as low-value?
- **RQ4 (Explainability):** Are margin/uncertainty explanations faithful (deletion test) and consistent with clinical intuition?

## 7. Expected contributions

- **C1.** A decision-anchored formulation of repeat-test information value that operationalizes DECIDE-Lab's action-change principle on EHR data, with a simple proposition connecting EDCP to single-step VOI.
- **C2.** A model-agnostic, distribution-free **safety certificate** for low-value flags (CRC + LTT + Mondrian), evaluated for validity across labs, subgroups and temporal shift.
- **C3.** A **VOI-native explanation** (margin vs. uncertainty, each SHAP-attributed) with quantitative faithfulness evaluation.
- **C4.** A reproducible, CPU-only empirical study on MIMIC-IV over four common repeat labs, including the first (to our knowledge) quantification of proxy-vs-decision label disagreement.

**Honest novelty framing:** each ingredient (quantile GBMs, conformal risk control, SHAP) exists; the novelty is the **decision-anchored target**, the **certified triage for lab deferral**, and the **VOI-native explanation**, combined and evaluated together. Do not overclaim in the paper.

## 8. Mapping the project brief to this design

| Brief says | Realized as |
|---|---|
| Demographics | age, sex (race/insurance used **only** for subgroup evaluation, not as features) |
| Symptoms | Not available as structured data in MIMIC-IV hosp; proxied by admission type, ICU status and (optional) vitals. Free-text notes are **out of scope**. Stated as a limitation. |
| Medical history | Prior-admission count; prior-admission diagnoses (optional). **Current-admission ICD codes are excluded** (assigned at discharge → leakage). |
| Previous test results & temporal changes | Last value, deltas, slopes, variability, counts, time since last, cross-lab panel |
| Characteristics of candidate test | Lab identity, its action thresholds, distance of last value to nearest threshold, analytical variability (optional) |
| High-value / potentially useful / low-value | Three tiers from calibrated EDCP; the low-value tier carries the conformal guarantee |
| Explainability layer | Margin/uncertainty decomposition + TreeSHAP + template text |

## 9. Scope boundaries

**In scope:** repeat tests of 4 labs (hemoglobin, potassium, creatinine, platelets) in adult hospital admissions of MIMIC-IV; tree models on CPU; conformal certification; SHAP-based explanations; an optional minimal demo UI on the open demo dataset.

**Out of scope:** POMDP/RL policy learning; deep sequence models; LLM-generated explanations; imaging or non-lab tests; first-time (non-repeat) tests; clinical deployment or prospective validation; free-text notes; claims of clinical safety.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MIMIC-IV credentialing delay | Start PhysioNet credentialing + CITI training in week 1. Build the entire pipeline on the **open MIMIC-IV demo** (100 patients) meanwhile. |
| Data-use-agreement violation | Row-level full-MIMIC data never leaves the local machine (see `CONTRIBUTING.md` §4). Only code, schemas and aggregates. |
| Label imbalance (few action changes) | Report AUPRC; CRC still valid; per-lab reporting. |
| Low reduction at tight α | Report full α–reduction trade-off curve; this is itself a finding. |
| Thresholds not universally accepted | Thresholds are configurable; sensitivity analysis over alternative thresholds. |
| Confounding by intervention (e.g., transfusion between draws) | Optional intervention features (ICU subset); stated limitation. |
| Selection bias (only ordered tests observed) | Standard in this literature; explicitly discussed. |
