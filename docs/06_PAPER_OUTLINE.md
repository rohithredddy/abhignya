# 06 — IEEE Paper Outline (6 pages, IEEE conference template, two-column)

Target venues (mid-tier, verify current CFPs and page limits): IEEE BIBM, IEEE ICHI, IEEE CBMS, IEEE BHI, IEEE EMBC (4 pages), IEEE INDICON / IEEE region conferences.

**Title:** CertiVOI: Decision-Anchored, Conformally Certified and Explainable Prediction of the Information Value of Repeat Laboratory Tests

## Abstract (≈200 words)
Problem (low-value repeat labs) → gap (proxy labels, no guarantees) → method (decision-anchored label, quantile GBM → EDCP, conformal risk control, margin/uncertainty explanations) → data (MIMIC-IV, 4 labs, CPU) → key results (fill from E1/E3/E5) → takeaway.

## I. Introduction (~0.75 page)
- Burden of repeat testing; Choosing Wisely context.
- VOI view (DECIDE-Lab): value = ability to change action.
- Gap paragraph (from `01_RESEARCH_PROPOSAL.md` §4).
- Contributions C1–C4 as a bulleted list (IEEE style allows it here).

## II. Related Work (~0.6 page)
A. Decision-theoretic test valuation. B. ML for unnecessary lab prediction. C. Sequential/RL test acquisition (why we don't use it). D. Conformal decision support.

## III. Method (~1.6 pages)
A. Problem setup and notation. B. Action maps and decision-anchored label. C. Distributional forecaster. D. EDCP and Proposition 1 (VOI link). E. Certified triage: CRC, LTT, Mondrian. F. VOI-native explanation.
- Fig. 1: pipeline diagram.

## IV. Experimental Setup (~0.6 page)
Data & cohort (Table 1), splits, baselines, metrics, implementation details (CPU spec, runtimes).

## V. Results (~1.6 pages)
A. Label disagreement (Table 2/Fig. 2). B. Prediction (Table 3). C. Certified reduction (Table 4, Fig. 4). D. Robustness: subgroups & temporal (Table 5). E. Explanations (Fig. 7, Table 6, example box). F. Ablations (short text or Table 7).

## VI. Discussion (~0.5 page)
Clinical interpretation; when certification is conservative; selection bias and confounding by intervention; threshold choice; single-site data; relation to full POMDP (sequential) value — future work.

## VII. Conclusion (~0.15 page)

## Writing rules
- Every number in the text must come from a file in `results/`; cite the file path in a LaTeX comment.
- Claims of "first" must be phrased "to our knowledge".
- No patient-level MIMIC data anywhere. Example explanations use synthetic cases.
- Include an ethics/data statement (PhysioNet DUA, de-identified data, decision support only).
