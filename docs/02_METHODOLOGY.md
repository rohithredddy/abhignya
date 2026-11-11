# 02 — Methodology (formal specification)

This document is the **source of truth** for definitions. Code must match it; if code needs to deviate, update this file and log the decision in `PROGRESS.md`.

---

## 1. Units of analysis

- **Lab** $L \in \mathcal{L} = \{\text{HGB}, \text{K}, \text{CREAT}, \text{PLT}\}$.
- **Event** $e = (i, h, L, t)$: a resulted measurement of lab $L$ for patient $i$ in hospital admission $h$ at chart time $t$, with value $y_e$.
- **Repeat event:** an event with at least one earlier measurement of the same lab in the same admission within the look-back window $W_{\text{rep}}$ (default 72 h). Only repeat events are modeled.
- **History** $\mathcal{H}_e$: all data with timestamp **strictly before** $t$ (and at least $\epsilon$ = 1 min before, to exclude same-specimen draws). Let $y^{\text{last}}_e$ be the most recent prior value of $L$.

## 2. Action map and decision-anchored label

Each lab has an **action map** $A_L$ that converts a value (and, for creatinine, recent history) into a discrete action category $c \in \{1,\dots,K_L\}$. Defaults (configurable in `configs/default.yaml`; to be confirmed with a clinician co-advisor if available):

| Lab | Units | Categories (action implied) | Basis |
|---|---|---|---|
| HGB | g/dL | `<7` consider transfusion · `7–<8` borderline (patient-dependent) · `≥8` no transfusion action | Restrictive transfusion guidance (AABB) |
| K | mEq/L | `<3.5` replete · `3.5–5.5` no action · `>5.5` treat/monitor hyperkalemia | Common inpatient electrolyte protocols |
| PLT | K/µL | `<10` prophylactic transfusion · `10–<50` procedure-dependent · `≥50` no action | Platelet transfusion guidance (AABB) |
| CREAT | mg/dL | KDIGO AKI stage 0/1/2/3 computed from the new value relative to the min in prior 48 h (≥0.3 rise) and prior 7 days (≥1.5×, 2×, 3×; or ≥4.0) | KDIGO 2012 (creatinine criteria only; no urine output) |

**Boundary convention.** Intervals are left-closed by default (HGB exactly 7.0 → borderline; PLT exactly 50 → no action). Potassium follows the table above: exactly 5.5 is *no action* (config `actions.K.at_boundary_upper: [true, false]`). Creatinine staging uses a 1e-9 tolerance so that, e.g., 1.0 + 0.3 counts as a 0.3 mg/dL rise despite floating-point rounding. KDIGO baselines are the minimum prior creatinine in the same admission within 48 h / 7 d, strictly before t − 1 min (rule as in the MIMIC-code `kdigo_creatinine` concept).

**Default action (prior belief):** $a^0_e = A_L(y^{\text{last}}_e)$ (for CREAT: the AKI stage at the last measurement).

**Decision-anchored label:**
$$D_e = \mathbb{1}\big[A_L(y_e) \neq a^0_e\big].$$
$D_e=1$ means the test was **action-changing** (high information).

**Proxy labels (for RQ1 and baselines only):**
- $N_e = \mathbb{1}[y_e \in \text{reference range}]$ (normality, Xu et al. style; use `ref_range_lower/upper` from labevents, falling back to config).
- $S_e = \mathbb{1}[|y_e - y^{\text{last}}_e| \le \delta_L]$ (stability, Aikens et al. style; $\delta_L$ in config).

## 3. Distributional forecaster

Target: $\Delta_e = y_e - y^{\text{last}}_e$ (for CREAT, model $y_e$ directly or $\Delta$; pick by validation pinball loss and log the decision). *Implemented:* with `model.creat_target: auto`, CREAT fits both targets and keeps the one with lower mean validation pinball loss on the y scale; the choice is written to `results/E2/hyperparams.json` (`target`).

For each lab separately, train LightGBM quantile models $\hat q_\tau(x)$ for $\tau \in \mathcal{T} = \{0.02, 0.05, 0.10, \dots, 0.90, 0.95, 0.98\}$ (pinball loss). Post-process:
1. **Non-crossing:** sort predicted quantiles per row (monotone rearrangement).
2. **CDF:** $\hat F(y \mid x)$ by piecewise-linear interpolation between $(\hat q_\tau + y^{\text{last}}, \tau)$; constant tails beyond the extreme quantiles (mass $\tau_{\min}$ and $1-\tau_{\max}$ placed at the ends).

## 4. EDCP score and link to VOI

Let $[l_{a}, u_{a})$ be the value interval of action category $a$ (for CREAT, intervals are computed per event from the 48 h / 7 d baselines).

$$\text{EDCP}(x_e) = 1 - \big[\hat F(u_{a^0_e} \mid x_e) - \hat F(l_{a^0_e} \mid x_e)\big].$$

**Proposition (single-step VOI under 0–1 regret).** Suppose (i) the test reveals the true category $c$, (ii) after seeing $c$ the clinician takes the best action for $c$, (iii) without the test the clinician takes $a^0$, and (iv) the regret of action $a$ in category $c$ is $r(a,c) = \mathbb{1}[a\neq c]$. Then the expected value of the test's information equals $\Pr(c \neq a^0 \mid x)$, i.e., EDCP.

*Proof sketch:* $\text{VOI} = \mathbb{E}_c[r(a^0,c) - r(c,c)] = \mathbb{E}_c[\mathbb{1}[c\neq a^0]] - 0$.

**Generalization (optional, for discussion section):** with a utility matrix $U(a,c)$, $\text{VOI}(x) = \sum_c \hat p(c\mid x)\,[\max_a U(a,c) - U(a^0,c)]$. This is DECIDE-Lab's single-step VOI with learned, patient-specific category probabilities. Optionally add analytical measurement noise (lab CV) by convolving $\hat F$ with $\mathcal N(0, (\text{CV}\cdot y)^2)$ — this is where "test performance" enters, mirroring DECIDE-Lab's observation model.

**Calibration:** fit isotonic regression $g$ on the validation split: $s_e = g(\text{EDCP}(x_e))$ approximates $\Pr(D_e=1\mid x_e)$.

## 5. Certified triage (conformal)

Data splits are **by patient** (`subject_id`): train / validation / calibration / test (default 60/10/15/15).

### 5.1 Conformal Risk Control (primary)
Flag a test **low-value** if $s_e \le \lambda$. Per-event loss
$$\ell_e(\lambda) = D_e \cdot \mathbb{1}[s_e \le \lambda] \in \{0,1\},$$
which is non-decreasing in $\lambda$. With calibration set of size $n$ and empirical risk $\hat R_n(\lambda)$:
$$\hat\lambda = \max\Big\{\lambda \in \Lambda : \frac{n}{n+1}\hat R_n(\lambda) + \frac{1}{n+1} \le \alpha\Big\}.$$
**Guarantee** (exchangeability): $\mathbb{E}[\ell_{\text{new}}(\hat\lambda)] \le \alpha$. Interpretation: *per 100 repeat tests of lab L, on average at most 100α action-changing results fall in the low-value tier.*

Default $\alpha \in \{0.005, 0.01, 0.02, 0.05\}$; grid $\Lambda$ = unique calibration scores plus 0.

### 5.2 Learn-then-Test (secondary, conditional)
False-skip rate among flagged: $\text{FSR}(\lambda) = \Pr(D=1 \mid s \le \lambda)$. For an increasing grid $\lambda_1<\lambda_2<\dots$, test $H_j: \text{FSR}(\lambda_j) > \beta$ with an exact binomial (Clopper–Pearson) p-value on the flagged calibration subset; **fixed-sequence testing** from the smallest $\lambda$, stop at first non-rejection. Guarantee: with probability $\ge 1-\delta$, $\text{FSR}(\hat\lambda) \le \beta$. Defaults: $\beta \in \{0.02, 0.05\}$, $\delta = 0.1$.

*Grid (implementation decision).* A grid of every unique calibration score starts with one flagged event, whose p-value $1-\beta$ is never $\le\delta$, so fixed-sequence testing would always stop at once. The grid is therefore `conformal.ltt.grid_size` (=100) calibration-score order statistics whose flagged counts run evenly from $n_{\min}=\lceil \log\delta / \log(1-\beta)\rceil$ (the smallest count at which rejection is possible; 45 for β = 0.05, δ = 0.1) to $n$. The grid depends only on scores, β and δ, never on labels. With small calibration sets (e.g. the demo) LTT may flag nothing; this is reported, not hidden.

### 5.3 Mondrian (group-conditional) variant
Run 5.1 separately within groups $G$ (sex; age band <45/45–65/65–80/>80; race group; ICU vs ward). Report validity and reduction per group, global vs. Mondrian.

### 5.4 Three tiers
- **Low-value:** $s \le \hat\lambda_\alpha$ (certified).
- **High-value:** $s \ge \tau_H$, with $\tau_H$ chosen on validation so that precision of $D=1$ in this tier ≥ 0.5 (config).
- **Potentially useful:** otherwise.

## 6. VOI-native explanation

Per event, compute:
- **Predicted value** $\hat y = y^{\text{last}} + \hat q_{0.5}$.
- **Margin** $m$ = signed distance from $\hat y$ to the nearest boundary of $a^0$'s interval (positive = inside).
- **Uncertainty** $w = \hat q_{0.9} - \hat q_{0.1}$ (80% interval width).
- **Margin ratio** $\rho = m / w$. Report Spearman correlation between $\rho$ and EDCP to justify the decomposition empirically.

Feature attribution (interventional TreeSHAP, shared background of ~200 training rows):
- $\phi^{\text{med}} = \text{SHAP}(\hat q_{0.5})$ → *why the predicted value is where it is* (drives margin).
- $\phi^{\text{unc}} = \text{SHAP}(\hat q_{0.9}) - \text{SHAP}(\hat q_{0.1})$ → *why the model is (un)certain* (valid by linearity of SHAP in model output with a common background).

*Implementation notes (needed for exact additivity, checked in E5: max |Σφ + base − output| ≤ 1e-3):* (i) each quantile booster is truncated to its early-stopping iteration, so SHAP explains exactly the model used for prediction; (ii) features are rounded (4 decimals; hours 2) and stored as float32-representable values; (iii) each booster is rewritten so TreeSHAP routes samples like LightGBM: thresholds are replaced by their float32 floor (TreeSHAP compares in float32, LightGBM in float64), and splits that saw no NaN in training send missing values where 0 goes (LightGBM's NaN→0 rule). Predictions are unchanged (asserted at fit time).

**Template text** (no LLM), e.g.:
> *Low value (certified at α = 1%). Predicted Hgb 9.3 g/dL (80% range 8.8–9.8), 1.3 g/dL above the next action threshold (8 g/dL). Main reasons: last 3 values stable (slope −0.05 g/dL/day), last value 9.4, no large change in hematocrit.*

**Faithfulness evaluation:** for k ∈ {1,3,5}, replace the top-k |φ| features with background values and measure |ΔEDCP|; compare against k random features (paired test). Report also global importance (mean |φ|) per lab.

## 7. Baselines

| ID | Method | Score used for skipping |
|---|---|---|
| B0 | No reduction (all tests ordered) | — |
| B1 | Rule: skip if prior value in "no action" category and |Δ last two| ≤ δ_L and last test < 24 h ago | rule |
| B2 | Normality classifier (LightGBM on $N$) | $1 - \Pr(N=1)$ |
| B3 | Stability classifier (LightGBM on $S$) | $1 - \Pr(S=1)$ |
| B4 | Logistic regression on $D$ | $\Pr(D=1)$ |
| B5 | Direct LightGBM classifier on $D$ | $\Pr(D=1)$ |
| **Ours** | Quantile LightGBM → EDCP → isotonic | $s$ |

Two evaluation modes: (a) baselines at their natural thresholds (shows proxies can violate safety); (b) **CRC applied to every score** (CRC is model-agnostic → fair comparison of reduction at equal certified risk). Be explicit in the paper that our gain over B5 in mode (b) may be small; our main advantages are the decision-anchored target, the certificate, and the decomposable explanation.
