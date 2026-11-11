# 04 — Experiment Plan

All experiments run per lab (HGB, K, CREAT, PLT), on CPU, with fixed seeds (default 42; robustness over seeds {42, 7, 2026} for headline numbers). Every experiment has a script in `scripts/` that writes **aggregate-only** outputs to `results/<exp_id>/` (CSV + PNG + a `meta.json` with config hash, git commit, data_mode, runtime).

## E1 — Label analysis (RQ1)
**Goal:** show decision-anchored labels differ from proxies.
- Prevalence of $D$, $N$, $S$ per lab.
- Contingency tables $D \times N$, $D \times S$; report % of action-changing results that were **normal** and % that were **stable** (these would be wrongly skipped by proxy methods).
- **Output:** Table 2 (label prevalence & disagreement), Fig. 2 (stacked bars).

## E2 — Predictive performance (RQ2)
**Models:** B2–B5, Ours (see `02_METHODOLOGY.md` §7).
**Metrics on test (target D):** AUROC, AUPRC, Brier score, ECE (15 bins), with 95% CIs by 1,000× patient-level bootstrap.
**Forecaster-specific:** pinball loss per τ, empirical coverage of 80% and 90% intervals.
**Output:** Table 3 (per lab × model), Fig. 3 (reliability diagrams for Ours vs B5).

## E3 — Certified reduction (RQ3, headline)
For α ∈ {0.005, 0.01, 0.02, 0.05}:
- **Validity:** realized risk on test (mean, 5th–95th pct over 100 calib/test resamples) vs α. Must be ≤ α on average.
- **Utility:** % of repeat tests flagged low-value (reduction rate); estimated tests avoided per 1,000 patient-days.
- **Conditional:** LTT with β ∈ {0.02, 0.05}: realized FSR and reduction.
- **Comparison:** (a) baselines at natural thresholds (expect proxies to exceed α); (b) CRC applied to all scores → reduction at equal guaranteed risk.
- **Output:** Table 4 (α × lab: risk, reduction), Fig. 4 (risk–reduction curve with α line), Fig. 5 (box plots of realized risk over resamples).

## E4 — Robustness (RQ3)
- **Subgroups:** sex, age band, race group, ICU vs ward. Global CRC vs Mondrian CRC: realized risk and reduction per group.
- **Temporal shift:** train ≤2016, test 2017–2019. Report degradation; then re-calibrate CRC on a small slice of the later period and show recovery.
- **Threshold sensitivity:** alternative action thresholds (e.g., HGB single threshold 7; K boundaries 3.0/6.0) → re-run E1 + E3 summary.
- **Output:** Table 5 (subgroup validity), Fig. 6 (temporal shift).

## E5 — Explainability (RQ4)
- Spearman(ρ, EDCP) per lab (justifies margin/uncertainty decomposition).
- Global importance: mean |φ^med| and mean |φ^unc| top-10 per lab (Fig. 7).
- Faithfulness deletion test (k = 1, 3, 5 vs random-k; Wilcoxon signed-rank) (Table 6).
- 4–6 **synthetic or demo-dataset** example explanations for the paper (never full-MIMIC rows) — one each for low/potential/high tiers (Fig. 8).

## E6 — Ablations
- Target: Δ vs absolute value.
- Features: remove trend features; remove cross-lab panel; add ICU intervention features (ICU subset).
- Model: quantile LightGBM vs direct classifier (B5) vs logistic (B4) — same CRC.
- Calibration: with vs without isotonic before CRC (validity should hold either way; reduction may change).
- **Output:** Table 7.

## E7 — Certification extensions (RQ3+, novelty)
- (a) **Patient-clustered CRC** (02 §5.5) vs event-level CRC: event-level and patient-level realised risk, reduction, over `splits.crc_resamples` patient re-splits. Output: Table 8.
- (b) **Panel certification** (02 §5.6) for CBC (HGB+PLT) and BMP (K+CREAT): naive per-lab α, Bonferroni per-lab α/k, panel CRC (max, noisy-or); draw-level miss rate and draws avoided. Output: Table 9.
- (c) **Audited online risk control** (02 §5.7): test stream ordered as shuffled (exchangeable), by era (`anchor_year_group`), ward→ICU (covariate shift), and the temporal model (train ≤2016, stream 2017–2019) when available; static CRC vs online with audit rates $p\in\{1, 0.2, 0.1\}$; miss rate, worst rolling-window miss rate, flag rate, effective reduction, finite-T bound; `extensions.online.reps` repeats. Output: Table 10, Fig. 9.
- Report honestly: clustered CRC is more conservative (fewer units); online control trades some reduction (audits) for validity under shift.

## Reporting checklist
- Cohort flow (Table 1), with counts per lab.
- All metrics with CIs; all randomness seeded and logged.
- State exchangeability assumption and where it may fail (temporal shift).
- Follow TRIPOD+AI items where applicable (development/validation of prediction model).
- No individual-level MIMIC data in any figure or table.

## Compute budget (estimate, 8-core CPU, 16 GB)
| Step | Est. time |
|---|---|
| Extraction to Parquet | 15–60 min (once) |
| Features per lab | 2–10 min |
| Quantile models (13 τ) per lab | 5–20 min |
| Baselines per lab | < 5 min |
| CRC/LTT + 100 resamples | < 2 min |
| TreeSHAP (5k explained rows, 3 models) per lab | 5–15 min |
| Bootstrap CIs | 5–10 min |
