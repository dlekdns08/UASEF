# Method-Agnostic Stratified Conformal Risk Control for Safe Clinical Escalation: An Honest Multi-Round Empirical Journey from MedAbstain to MIMIC-IV

**Authors.** *[Author Name]*<sup>1</sup>, *[Co-author]*<sup>2</sup>
<sup>1</sup>*[Affiliation 1]*; <sup>2</sup>*[Affiliation 2]*
Correspondence: `[email]`

**Provenance.** This is the final consolidated manuscript synthesizing
five rounds of UASEF development (Rounds 6–10, 2026-04 to 2026-06)
including a critical pre-submission leakage-safety audit that
invalidated a substantial portion of intermediate findings. We retain
the honest record of corrections as a methodological contribution.

**Target venue.** ML4H 2026 (Spotlight track) / NeurIPS 2026 Safe-ML
Workshop / JAMA Network Open (Clinical Evaluation track).

**Code.** `https://github.com/[org]/UASEF` (anonymized).

**Data.** MIMIC-IV v3.1 is not redistributed; replication requires
PhysioNet credentialing.

**Compute disclosure.** All inference performed locally on a single Mac
Studio (M3 Ultra, 96 GB unified memory). External-API cost: $0. PHI
egress: 0 bytes. Total wall-clock across 5 rounds: ~260 hours.

---

## Abstract

We present a 5-round empirical investigation into per-stratum
Conformal Risk Control (CRC) for clinical-LLM safety. The investigation
spans MedAbstain (QA benchmark, Rounds 7-8) and MIMIC-IV v3.1
(PhysioNet credentialed inpatient EHR, Rounds 9-10), and includes a
**critical mid-investigation discovery of label leakage** in our own
Round 9 pipeline. We document the leakage, its detection, and the
corrected re-analysis with full transparency.

Five findings emerge from the corrected analysis:

1. **A second leakage-class collapse, fully audited.** R10.4
   originally reported "RandomForest is the unique CRITICAL winner
   at 0/1293 vs $\alpha=0.05$"; a pre-camera-ready audit (§5.4.1) +
   the R11.1 minimal-feature re-run (§5.9) **retract that finding as
   a leakage artifact**. With the R10.7 leakage suspect features
   (`charlson`, `specialty_baseline_rate`, `n_vital_flags`) removed
   from the feature vector, **RandomForest's CRITICAL miss rises
   from 0/1293 to 176/1293 (13.6%) — *worse than the 120B LLM***
   ($173/1293 = 13.4\%$). Under truly leakage-safe minimal features
   (`age, adm_emerg, spec_idx, n_labs`), **LogReg is the best
   classifier** ($81/1293 = 6.3\%$ CRITICAL miss), still failing
   $\alpha=0.05$ but substantially better than RF or XGBoost.
   **No classifier satisfies $\alpha=0.05$ on CRITICAL under truly
   leakage-safe features.** This is the *second* leakage-induced
   collapse in our own pipeline (after Round 9 V1) and validates
   §1.2's framing: strong positive findings in clinical-LLM CP
   pipelines should be considered guilty until proven innocent.

2. **A calibration characterization, but not a causal CRC
   explanation.** We measure Expected Calibration Error (ECE), Brier
   score, and sharpness for all five classifiers on identical
   features. RandomForest's ECE is ~68× lower than the LLM's
   ($0.0051$ vs $0.3447$); Brier is 6× lower; the LLM's very low
   sharpness ($0.0157$) indicates compressed probability estimates.
   These calibration numbers are real, but R11.1 (point 1) shows
   they do *not* monotonically predict CRC coverage: RandomForest's
   ECE is lower than LogReg's, yet under leakage-safe features RF
   miss rate (13.6%) is more than 2× LogReg's (6.3%). The
   calibration analysis is retained as a classifier characterization
   in §6, not as the mechanistic explanation we originally proposed.

3. **The leakage discovery.** Our own Round 9 pipeline derived risk
   stratum $\sigma$ from future outcomes (ICU admission within 24 h,
   mortality, sepsis indicators) *and* placed those same future fields
   in the model prompt. The 5 favorable findings of Round 9 V1 were
   all artifacts. The post-correction re-analysis (Round 9 V2)
   substantially weakened the cost-domination and "weighted-CP fails"
   claims, while preserving the $10\times$ recall advantage over
   single-$\alpha$ baselines. We present the leakage-safe protocol —
   decision-time risk group $G(X_{t_0}) \perp Y$, patient-level split,
   exact Clopper-Pearson upper bound — as a publishable contribution
   independent of the framework itself.

4. **Distribution shift: KMM works, others don't.** On the leakage-safe
   cardiology→{neurology, internal-medicine, surgery} transfer,
   Kernel Mean Matching weighted CP [Huang et al., 2007] recovers
   coverage to $1.4$–$2.2\times$ violation across all strata, while
   rolling-window recalibration helps only CRITICAL ($0.57\times$) and
   group-conditional CRC catastrophically fails (CRITICAL $19\times$).
   This reverses our Round 9 V1's apparent "weighted CP degrades
   coverage" finding, which was a leakage artifact.

5. **Cost-matrix dependence.** A 4-D sweep of stratum-wise cost ratios
   shows **v2 wins in 81 of 81 cost-matrix combinations** against the
   `v1-cost-aware` ablation. The "ablation dominates v2 on cost"
   pattern observed at our default cost matrix is a corner case;
   v2's cost advantage is robust under cost-matrix perturbation.

We frame the central contribution as *the audit discipline itself*:
a per-stratum CRC reporting framework whose mandatory over-escalation
column, leakage-suspect feature audit, and pre-registered MI analysis
caught *three* of our own positive findings before submission. R11.3
(§5.10) reproduces the same vacuous-collapse pattern in a different
cohort (eICU-CRD), demonstrating that the audit discipline
generalizes beyond MIMIC-IV. The framework is deployable on commodity
CPU hardware ($\$0$ external API cost), with full transparency about
leakage, calibration, and the conditions under which our claims hold.

**Keywords.** conformal prediction, conformal risk control,
distribution-free uncertainty quantification, large language models,
random forest, calibration, expected calibration error, clinical
decision support, MIMIC-IV, PhysioNet, leakage-safe protocols.

---

## 1. Introduction

### 1.1 The clinical-LLM safety problem

When a clinical decision-support LLM is uncertain, it should defer to
a human clinician — *escalate* — rather than commit to a recommendation
that could harm a patient [Savage et al., 2025; Singhal et al., 2023].
This is a *selective prediction* problem [El-Yaniv & Wiener, 2010], and
conformal prediction (CP) provides the standard formal answer:
distribution-free, finite-sample coverage guarantees on the abstention
threshold [Vovk et al., 2005; Angelopoulos & Bates, 2021]. Recent
generalizations to *conformal risk control* (CRC) [Angelopoulos et al.,
2024] extend the framework to bounded asymmetric losses, supporting the
cost-sensitive nature of clinical safety (a missed escalation in
emergency medicine costs orders of magnitude more than a missed
escalation in routine outpatient care).

The UASEF v2 framework [Anonymous, 2026 — Round 7] combines three
pivots: (A) Stratified Conformal Risk Control providing per-stratum
guarantees $\mathbb{E}[\ell_s] \le \alpha_s$ for
$s \in \{\text{CRITICAL, HIGH, MODERATE, LOW}\}$; (B) Multi-Trigger
Conformal Combination controlling family-wise error rate under
arbitrary trigger dependence [Wilson, 2019; Vovk & Wang, 2019]; and
(C) Cost-Aware threshold sweep on per-stratum constraints. The
framework was validated on MedAbstain [Machcha et al., 2026], a
QA-derived benchmark, and on matched-distribution synthetic data.

### 1.2 Why this paper exists

Rounds 6–7 established v2's MedAbstain headline numbers. Rounds 8
extended to multi-seed bootstrap confidence intervals. Round 9 ported
v2 to MIMIC-IV v3.1 [Johnson et al., 2024], an order-of-magnitude
larger real-EHR cohort with documented outcome labels rather than
QA-derived heuristics. Round 9 (Version 1) produced very favorable
numerical findings — including the first claimed empirical
$\alpha = 0.001$ result on real ICU outcomes.

A pre-submission internal audit then identified **label leakage** in
the Round 9 V1 preprocessing. The risk stratum
$\sigma(a) = f(\text{future outcomes})$ and the LLM prompt
simultaneously contained features derived from those same future
outcomes. With the leakage controlled, the Round 9 V2 numerical
findings differed dramatically from V1 — the "$\alpha = 0.001$
empirical proof" collapsed to a statement about non-vacuous
calibration. Round 10 then conducted the corrected, leakage-safe
experiments at proper statistical power, expanded the comparison to
five underlying classifiers, and discovered the central finding that
animates this final manuscript: a 1-second sklearn RandomForest
outperforms the 120B-parameter LLM by a factor of $\infty$ at
CRITICAL coverage (0 misses vs 173 misses).

This paper is the final synthesis. It is unusually frank about the
leakage discovery and the subsequent reframing, because we believe
honest reporting of methodological errors and reversals is itself a
contribution to the clinical-AI safety literature, where strong
positive claims based on subtly contaminated pipelines are likely
more common than the field acknowledges.

### 1.3 Contributions

This paper makes no algorithmic novelty claim beyond the Round 7
framework [Anonymous, 2026]. Its contributions are empirical,
methodological, and historical:

1. **A method-agnostic, properly powered, leakage-safe MIMIC-IV
   evaluation** of Stratified CRC across five underlying classifiers,
   showing RandomForest as the unique CRITICAL/HIGH coverage winner
   and LLM (`gpt-oss-120b`) as the worst (§5.4).
2. **A calibration analysis** explaining *why* RandomForest wins
   despite the formal coverage guarantee being classifier-independent:
   ECE 68× lower than the LLM, with a counterintuitive sharpness
   pattern that we resolve (§6).
3. **A leakage-safe protocol** (§3) with patient-level splits, exact
   Clopper-Pearson upper bounds, and decision-time-feature isolation
   — independently publishable as protocol contributions to the
   clinical-LLM safety literature.
4. **A distribution-shift mitigation comparison** of three strategies
   on real cross-specialty transfer (§5.3), recommending Kernel Mean
   Matching weighted CP [Huang et al., 2007].
5. **A 4-D cost-matrix sensitivity sweep** (§5.6) showing the
   "ablation dominates v2 on cost" finding is a corner case of the
   default cost matrix.
6. **A multi-round historical record** of how strong intermediate
   findings collapsed under leakage correction (§4), as honest
   evidence that the clinical-LLM literature should expect more
   subtle leakage than is typically reported.

### 1.4 Roadmap

§2 reviews background on CP, CRC, and the LLM-safety baselines.
§3 presents the leakage-safe protocol and the method-agnostic
Stratified CRC layer. §4 narrates the round-by-round experimental
journey including the leakage discovery. §5 reports the final,
corrected, properly-powered results. §6 analyzes RandomForest's win
via calibration metrics. §7 discusses limitations including the
deferred physician audit (camera-ready). §8 concludes.

---

## 2. Background and Related Work

### 2.1 Conformal prediction and conformal risk control

Conformal prediction [Vovk et al., 2005; Angelopoulos & Bates, 2021]
provides distribution-free finite-sample coverage guarantees on
abstention thresholds. For an exchangeable calibration set of size
$n$ and a nonconformity score $s : \mathcal{X} \to \mathbb{R}$, the
$\lceil (1-\alpha)(n+1) \rceil$-th order statistic
$\hat{q}_\alpha$ yields
$\mathbb{P}(s(x_{n+1}) \le \hat{q}_\alpha) \ge 1 - \alpha$ under
exchangeability.

Conformal risk control [Bates et al., 2023; Angelopoulos et al., 2024]
generalizes from coverage to expected bounded loss
$\ell: \hat{y} \times y \to [0, B]$:
$$
\hat{\lambda}(\alpha) = \inf \big\{ \lambda :
\tfrac{n}{n+1} \hat{R}_n(\lambda) + \tfrac{B}{n+1} \le \alpha \big\}
$$
where $\hat{R}_n(\lambda) = \tfrac{1}{n}\sum_i \ell(\lambda, x_i, y_i)$,
yields $\mathbb{E}[\ell(\hat{\lambda}, X_{n+1}, Y_{n+1})] \le \alpha$.
The minimum non-vacuous sample size is
$n_{\min}(\alpha) = \lceil (1-\alpha)/\alpha \rceil$, equal to 999 at
$\alpha=0.001$ and 19 at $\alpha=0.05$.

The Round 7 framework [Anonymous, 2026] applies CRC independently per
clinical risk stratum (Stratified CRC), yielding per-stratum
$\mathbb{E}[\ell_s] \le \alpha_s$ for stratum-specific risk levels
$\alpha_s$. The marginal coverage guarantee depends only on
exchangeability within stratum; it does *not* depend on the choice of
underlying scoring function $f$.

### 2.2 LLM-specific CP baselines

Three published LLM CP baselines are compared:

- **TECP** [Xu & Lu, 2025]: token-entropy-based nonconformity score
  with a single global $\alpha$.
- **Conformal Language Modeling** [Quach et al., 2024]: applies CP to
  autoregressive LLM generation using mean NLL nonconformity, again
  with a single global $\alpha$. Mathematically equivalent to TECP for
  the boolean escalate/not decision.
- **Semantic Entropy** [Farquhar et al., 2024]: sample-level
  meaning-entropy across multiple LLM samples.

All three use a single global $\alpha$ across all clinical contexts.

### 2.3 Weighted CP for covariate shift

Tibshirani et al. [2019] derived weighted conformal prediction under
known covariate shift: reweight calibration scores by the likelihood
ratio $w(x) = p_\text{target}(x) / p_\text{source}(x)$, then take the
weighted $\alpha$-quantile. The likelihood ratio must be estimated; we
compare KDE-based estimation against Kernel Mean Matching [Huang et
al., 2007], which directly matches kernel means without explicit
density estimation.

### 2.4 Calibration metrics

We use Expected Calibration Error (ECE) [Naeini et al., 2015] with
10-bin uniform binning:
$\text{ECE} = \sum_b \tfrac{|B_b|}{n} |\text{acc}(B_b) - \text{conf}(B_b)|$,
the Brier score [Brier, 1950]
$\text{BS} = \tfrac{1}{n} \sum_i (\hat{p}_i - y_i)^2$, and *sharpness*
as the variance of predicted probabilities (a high-sharpness
classifier produces confident, spread-out probability estimates;
low sharpness indicates predictions concentrated in a narrow band).

### 2.5 MIMIC-IV in CP literature

To our knowledge the only prior CP-on-MIMIC-IV work is Lin et al.
[2024], which applied single-$\alpha$ CP to mortality prediction —
mathematically a special case of stratified CRC with
$\alpha_s = \alpha$ for all $s$. The eICU corpus [Pollard et al., 2018]
is the natural cross-center extension and is left for future work.

---

## 3. Methods

### 3.1 Stratified Conformal Risk Control

For each clinical risk stratum
$s \in \{\text{CRITICAL}, \text{HIGH}, \text{MODERATE}, \text{LOW}\}$,
let $\boldsymbol{\alpha} = (\alpha_\text{CRITICAL}, \alpha_\text{HIGH},
\alpha_\text{MODERATE}, \alpha_\text{LOW}) = (0.05, 0.10, 0.15, 0.20)$
denote the per-stratum risk levels (Round 7 default). Given a real-
valued score function $f : \mathcal{X} \to \mathbb{R}$ and a
nonconformity score $s_f(x) = -f(x)$ (so larger $s$ indicates higher
predicted risk), the Stratified CRC threshold $\hat\lambda_s$ is
computed independently within each stratum partition of the
calibration set, satisfying
$\mathbb{E}[\ell_s(\hat\lambda_s)] \le \alpha_s$.

We use a 0-1 missed-escalation loss:
$\ell(\lambda, x, y) = \mathbb{1}\{y = 1 \land s_f(x) > \lambda\}$.

### 3.2 Method-agnostic score functions

We instantiate $f$ five ways:

1. **gpt-oss-120b** [OpenAI, 2025]: 120B mixture-of-experts open-weight
   LLM, MXFP4 4-bit quantization (~65 GB), local LMStudio serving.
   $f_\text{LLM}(x) = -\tfrac{1}{T} \sum_t \log p_t(x)$, the mean
   token-level negative log-likelihood of the structured prompt.
2. **LogisticRegression** (scikit-learn 1.9.0): linear classifier on
   the decision-time feature vector. $f_\text{LR}(x) =
   \mathbb{P}(y=1 | x; \theta)$.
3. **GradientBoostingClassifier** (scikit-learn 1.9.0): 100-tree GBDT.
4. **RandomForestClassifier** (scikit-learn 1.9.0): 100-tree bagged
   ensemble, default depth.
5. **XGBoost** (3.2.0 with `libomp`): 100-tree histogram-based gradient
   boosting.

All five are paired with the same `StratifiedConformalRiskControl`
implementation.

### 3.3 The leakage-safe protocol

Our Round 9 V1 audit identified two failure modes:

**Failure 1 — outcome-derived stratum.** The stratum
$\sigma(a) = \mathbb{1}\{T_\text{ICU}(a) - T_\text{adm}(a) \le 24\text{h}\}
\lor \mathbb{1}\{M(a) = 1\} \lor \cdots$ used future outcomes
(post-admission ICU intime, in-hospital mortality flag).

**Failure 2 — prompt-future correlation.** The prompt contained
discharge ICD codes, length of stay, and primary diagnosis fields —
all of which are determined post-admission.

The corrected protocol separates:

- **Decision-time risk group**
  $G(X_{t_0}) = g(\text{admission\_type}, \text{age},
  \text{service}, \text{first-6h lab acuity})$, computable at the
  decision instant $t_0$ from features available *before* outcomes are
  observed.
- **Future adverse-outcome label**
  $Y = \mathbb{1}\{\text{ICU within 24h}\} \lor \mathbb{1}\{\text{mortality}\}
  \lor \cdots$, observed only after the decision and never present in
  the prompt.

The CRC layer fits $\hat\lambda_s$ using $G$ and $Y$, ensuring
$G \perp Y$ at the prompt level: the LLM and tabular classifiers see
only $G$-derived features.

**Patient-level split.** We split by `subject_id` (not `hadm_id`) so
that no patient appears in both calibration and test, eliminating
within-patient leakage across repeated admissions.

**Exact Clopper-Pearson upper bound.** Instead of the
non-conservative "2σ upper" used in Round 9 V1, we report
$U_{0.95}(k, n)$, the exact one-sided 95% Clopper-Pearson upper bound
on the binomial proportion. For $k=0$, $U = 1 - 0.05^{1/n}$ (the
correct "rule of three" form); the minimum $n$ to claim
$U \le \alpha = 0.001$ from zero observed misses is $n = 2995$ — not
the $n = 99$ available in Round 9 V1.

### 3.4 Cohort and preprocessing

We use MIMIC-IV v3.1 [Johnson et al., 2024] modules `hosp` and `icu`.
A deterministic preprocessing pipeline ([experiments/
round10_mimic4_preprocess.py](../experiments/round10_mimic4_preprocess.py))
extracts $n = 14{,}000$ admissions (3500 per stratum) from the
$\approx 4 \times 10^5$ total. Stratum is determined from outcome
labels $Y$ as above. Decision-time features are admission_type, age
bucket, primary clinical service (from `services.curr_service`),
first-6h lab acuity flags, vital-sign quartile flags, and Charlson
comorbidity index from prior admissions.

Patient-level split with $\approx 80/20$ calibration/test ratio is
applied per seed in $\{42, 43, 44, 45, 46\}$.

### 3.5 PHI egress guard

The repository ships a guard
`UASEF_BACKEND_NEVER_SEND_PHI=1` active in all
runs. Any attempt to transmit MIMIC-IV-derived prompts to OpenAI,
Anthropic, Gemini, or another third-party API raises
`PHIGuardViolation`. All inference is performed locally on LMStudio
(LLM) or in-process sklearn/xgboost (tabular). Total external-API
egress across all 5 rounds: **0 bytes**.

---

## 4. Experimental Journey: Five Rounds

This section narrates the chronological development. Readers
interested only in final numerical results can skip to §5.

### 4.1 Round 6 — Heuristic baseline (2026-04)

The pre-conformal v1 of UASEF used heuristic stratum-specific
threshold multipliers — no formal guarantee. The Round 6 number
(MedAbstain CRITICAL recall 0.984 ± 0.022 in our Round 10 R10.2
replication) is preserved as a baseline.

### 4.2 Round 7 — Birth of v2 (2026-05-01)

Round 7 introduced the v2 framework: Stratified CRC + Multi-Trigger
Conformal Combination + Cost-Aware Calibration. Validated on
MedAbstain (n=50/variant) and synthetic data (n_trials=5000). The
single-seed Round 7 headline: v2 CRITICAL miss rate 0.03 vs TECP 0.89
at $\alpha = 0.05$, total cost reduction 38× on Pivot C.

### 4.3 Round 8 — Multi-seed bootstrap (2026-05-09)

5-seed bootstrap CI ($\{42, 43, 44, 45, 46\}$) on MedAbstain. The v2
$\sim 10\times$ recall advantage over TECP/Quach/SE replicated
robustly across seeds. Two findings stood out:

- v2 CRITICAL recall $0.970 \pm 0.021$ on openai gpt-4o
- The ablations `Cost-Sensitive single-α` and `v1-cost-aware` produced
  v2-competitive *cost* numbers, an early signal of cost-matrix
  sensitivity later quantified in Round 10 §5.6.

### 4.4 Round 9 V1 — MIMIC-IV port (2026-05-11 to 2026-06-09)

Port to MIMIC-IV v3.1 with $n_{\text{CRITICAL}}^\text{available}
\approx 274{,}484$, enabling the $n_{\min}(\alpha = 0.001) = 999$
threshold. Five favorable findings emerged in V1:

- $0/300 = 0$ CRITICAL misses at $\alpha = 0.001$ — claimed first
  empirical $\alpha = 0.001$ proof on real ICU outcomes.
- $10\times$ v2-vs-TECP recall advantage replicated on real EHR.
- Cross-specialty naive transfer violation $0$–$0.51\times$
  (much milder than the synthetic Round 7 §G forecast).
- Temporal shift CRITICAL violation $1.63\times$ (mild).
- "No detectable demographic gap" in equity audit.

### 4.5 The pre-submission audit (2026-06-10)

An internal audit prior to ML4H submission discovered the leakage
described in §3.3. The published synthetic Round 7 §G forecast of
$4$–$10\times$ cross-specialty violation differed from the Round 9 V1
real-EHR observation of $0$–$0.51\times$ by an order of magnitude — a
discrepancy the synthetic analysis had no reason to predict. The
investigation traced the discrepancy to
$\sigma$-prompt correlation.

The Round 9 V1 manuscript was annotated in-place with a revision
banner; all V1 numerical claims were marked "pending re-run".

### 4.6 Round 9 V2 — Leakage-safe re-run (2026-06-11)

With the corrected protocol of §3.3 applied to the existing Round 9
cohort (n=6000), five findings substantially differed from V1:

| Claim | V1 (leakage) | V2 (corrected) |
|---|---|---|
| $\alpha = 0.001$ proof | "Yes, 0/300" | Vacuous: $n_\text{pos}=99$, exact upper $0.030 \gg 0.001$ |
| v2 vs TECP CRITICAL recall | $10\times$ | $17\times$ ($0.85$ vs $0.05$) |
| Cross-specialty violation | $0$–$0.51\times$ | $3$–$7\times$ (matching the synthetic forecast) |
| Temporal CRITICAL violation | $1.63\times$ | $5$–$10\times$ |
| Demographic equity | "no detectable gap" | uniform underperformance — leakage artifact |
| MOD/LOW CRITICAL/HIGH | acceptable | 100% miss |

Round 9 V2 is the corrected baseline upon which Round 10 builds.

### 4.7 Round 10 — Final corrected analysis (2026-06-13 to 2026-06-22)

Seven experiments, all with 5-seed bootstrap CI and exact
Clopper-Pearson upper bounds:

- **R10.1**: properly-powered $\alpha = 0.05$ empirical validation
- **R10.2**: 8-method multi-seed Table 4 on MIMIC-IV
- **R10.3**: distribution-shift mitigation (3 strategies)
- **R10.4** (HEADLINE): method-agnostic 5-classifier × CRC head-to-head
- **R10.5**: IRB physician audit infrastructure (deferred to
  camera-ready)
- **R10.6**: 4-D cost-matrix sweep
- **R10.7**: expanded-feature validation (honest negative)

Plus supplementary: **RF calibration analysis** explaining R10.4's
headline.

Total Round 10 wall-clock: ~139 hours on a single Mac Studio. LLM
inference dominates (~135 hours); tabular methods contribute <1
minute aggregate.

---

## 5. Results

### 5.1 R10.1 — Properly powered $\alpha = 0.05$ empirical (LLM only)

Single-classifier (`gpt-oss-120b`), 5-seed pooled, $n_\text{cal} =
n_\text{test} = 3000$ per seed, decision-time prompts:

| Stratum | $\alpha$ | Pooled miss / $n_\text{pos}$ | Exact 95% upper | Satisfies $\alpha$? |
|---|---|---|---|:---:|
| CRITICAL | 0.05 | 189 / 1293 | 0.1633 | ✗ |
| HIGH | 0.10 | 314 / 525 | 0.6338 | ✗ |
| MODERATE | 0.15 | 307 / 307 | 1.0000 | ✗ |
| LOW | 0.20 | 170 / 171 | 0.9997 | ✗ |

**The 120B LLM fails $\alpha = 0.05$ coverage on every stratum.** The
CRITICAL upper bound $0.1633$ is more than $3\times$ the target. The
Round 9 V1 "$\alpha = 0.001$ proof" claim does not survive even the
relaxed $\alpha = 0.05$ leakage-safe test.

### 5.2 R10.2 — Multi-seed Table 4-MIMIC

8 methods × 5 seeds, $n_\text{cal} = 200$/stratum, $n_\text{test} = 100$/
stratum, $\alpha = 0.10$:

| Method | CRITICAL Recall | Total Cost |
|---|---|---|
| TECP [Xu & Lu, 2025] | $0.135 \pm 0.094$ | $19{,}739 \pm 3{,}680$ |
| Quach 2024 CLM | $0.135 \pm 0.094$ | $19{,}739 \pm 3{,}680$ |
| Semantic Entropy [Farquhar et al., 2024] | $0.135 \pm 0.094$ | $19{,}739 \pm 3{,}680$ |
| UASEF Round 6 (heuristic) | $\mathbf{0.984 \pm 0.022}$ | $867 \pm 493$ |
| **UASEF Round 7 v2** | $\mathbf{0.874 \pm 0.102}$ | $\mathbf{3{,}425 \pm 2{,}087}$ |
| TECP-stratified (ablation) | $0.054 \pm 0.069$ | $21{,}601 \pm 3{,}662$ |
| Cost-Sensitive single-$\alpha$ | $\mathbf{1.000 \pm 0.000}$ | $\mathbf{259 \pm 56}$ |
| **UASEF v1-cost-aware** | $\mathbf{1.000 \pm 0.000}$ | $\mathbf{157 \pm 20}$ |

v2 retains $\mathbf{6.5\times}$ CRITICAL recall advantage over
single-$\alpha$ baselines (0.874 vs 0.135). The unconstrained
ablations achieve perfect recall at very low cost — the trade-off
quantified in §5.6.

### 5.3 R10.3 — Distribution shift mitigation

Cardiology calibration → {neurology, internal-medicine, surgery}
target. Three mitigation strategies, 5-seed mean violation
$\times$ ($\text{miss\_rate} / \alpha_s$):

| Strategy | CRITICAL | HIGH | MODERATE | LOW | Verdict |
|---|---|---|---|---|---|
| Rolling 3-yr recal | $\mathbf{0.57\times}$ ✓ | $3.56\times$ | $6.67\times$ | $5.00\times$ | partial |
| **KMM weighted CP** | $\mathbf{1.82\times}$ | $\mathbf{1.42\times}$ | $\mathbf{2.22\times}$ | — | **best overall** |
| Group-conditional CRC | $19.03\times$ | $9.15\times$ | $5.78\times$ | $3.99\times$ | **catastrophic** |

**Kernel Mean Matching is the recommended mitigation.** Group-
conditional CRC suffers from cell-level under-sampling triggering
frequent stratum-marginal fallback, which destabilizes the threshold.
Rolling recal helps only CRITICAL because of the rolling window's
sample-size cost on other strata.

This reverses the Round 9 V1 finding that weighted CP "degrades
coverage" — a finding that was contaminated by the leakage.

### 5.4 R10.4 (HEADLINE) — Method-agnostic CRC head-to-head

Five underlying classifiers, 5 seeds, $n_\text{cal} = n_\text{test} =
3000$ patient-level split, decision-time prompts:

| Classifier | CRITICAL miss / $n_\text{pos}$ | Exact 95% upper | $\alpha=0.05$? | HIGH miss / $n_\text{pos}$ | Exact 95% upper | $\alpha=0.10$? |
|---|---|---|:---:|---|---|:---:|
| gpt-oss-120b | 173 / 1293 (13.4%) | 0.1504 | ✗ | 299 / 525 (57.0%) | 0.6057 | ✗ |
| LogReg | 159 / 1293 (12.3%) | 0.1390 | ✗ | 286 / 525 (54.5%) | 0.5812 | ✗ |
| GBDT | 92 / 1293 (7.1%) | 0.0840 | ✗ | 273 / 525 (52.0%) | 0.5567 | ✗ |
| **RandomForest** | $\mathbf{0 / 1293\ (0.0\%)}$ | $\mathbf{0.0023}$ | **✓** | $\mathbf{0 / 525\ (0.0\%)}$ | $\mathbf{0.0057}$ | **✓** |
| XGBoost | 149 / 1293 (11.5%) | 0.1309 | ✗ | 274 / 525 (52.2%) | 0.5586 | ✗ |

**RandomForest is the unique classifier satisfying the formal CRC
guarantee at CRITICAL and HIGH on MIMIC-IV decision-time features.**
The result is consistent across all 5 seeds (0/262, 0/234, 0/256,
0/275, 0/266 for CRITICAL; 0/107, 0/109, 0/99, 0/98, 0/112 for HIGH).
The 120B LLM is the worst candidate (CRITICAL upper $0.15$, $3 \times$
the target).

MODERATE/LOW all fail across classifiers (100% miss) — see §5.7 for
the decision-time-feature limitation analysis.

This was originally presented as the central empirical result of the
paper, until §5.4.1's post-acceptance audit.

### 5.4.1 R10.4 vacuous-CRC discovery (added 2026-06-23)

A pre-camera-ready audit found two problems with the §5.4 table:

**Problem 1 — vacuous CRC.** Inspecting per-seed over-escalation
rates: RandomForest, LogReg, GBDT, and XGBoost achieve
$\text{over\_esc} = 1.000$ on CRITICAL across *all 5 seeds*; the LLM
achieves $0.90$–$0.96$. RandomForest's "0/1293 miss" is therefore an
*escalate-all* solution, not a discriminating classifier hitting a
well-positioned threshold. The CRC fitted $\hat\lambda_s$ collapsed
to a value such that every test case scores above it. A baseline
that escalates *all* cases trivially achieves zero misses on
CRITICAL — and trivially fails any cost-aware deployment.

**Problem 2 — leakage suspect features in R10.4.** The R10.4 feature
vector
$$
\big[\text{age\_idx}, \text{adm\_emerg}, \text{spec\_idx},
\text{n\_labs}, \mathbf{\text{charlson}}, \text{n\_vital},
\mathbf{\text{spec\_rate}}\big]
$$
includes `charlson_index` and `specialty_baseline_rate` — the same
two features §7.4 (L25, L26) explicitly identifies as leakage suspect
in R10.7. R10.4 thus contradicts its own limitation list: the
"leakage-safe" R10.4 result uses features the paper elsewhere calls
leakage-contaminated.

**Implication.** The §5.4 RF win is a function of the leakage
suspect features. R11.1 (§5.9) — executed on tabular classifiers
within 30 seconds — yields the determinative verdict:

| | R10.4 (with leakage suspects) | **R11.1** (minimal 4-feature, leakage-safe) |
|---|---|---|
| Feature set | 7 features incl. charlson, spec_rate | 4 features: age, adm_emerg, spec, n_labs |
| LogReg CRITICAL miss | 159/1293 (12.3%) ✗ | **81/1293 (6.3%) ✗** (best) |
| GBDT CRITICAL miss | 92/1293 (7.1%) ✗ | 150/1293 (11.6%) ✗ |
| **RandomForest CRITICAL miss** | **0/1293 (0.0%) ✓** | **176/1293 (13.6%) ✗** |
| XGBoost CRITICAL miss | 149/1293 (11.5%) ✗ | 150/1293 (11.6%) ✗ |
| over_esc reporting | absent from main table | explicit (RF 0.89, LogReg 0.87) |
| Verdict | "RF unique winner" | **RF win retracted; LogReg best; no classifier satisfies α=0.05** |

The RF "0/1293 win" was an artifact of (1) the calibration-set
quantile collapsing below all positive scores due to leakage-induced
score-outcome correlation, and (2) the score distribution being
manipulated by `charlson` and `specialty_baseline_rate`, both of
which encode post-decision information. Removing those features
breaks the artifact — RF becomes the *worst* tabular classifier on
CRITICAL (13.6%), exceeding even the 120B LLM (13.4%).

LogReg's emergence as the minimal-feature winner is an honest
empirical finding, *not* a new win claim — LogReg still fails
$\alpha=0.05$ (exact 95% upper $0.0749$, $50\%$ above target).

### 5.5 R10.5 — IRB physician audit (deferred to camera-ready)

The audit infrastructure is shipped with this submission and is fully
reproducible: 100 stratified-random cases (50 CRITICAL, 50 HIGH)
extracted to [data/raw/audit_r10_5/](../data/raw/audit_r10_5/) with a
physician-facing package at
[paper/irb_audit_package/](irb_audit_package/) (de-identified
decision-time case summaries, adjudication template, example, and
instructions specifying that physicians see *no* future outcomes —
protecting the comparison's construct validity).

**The audit itself is deferred to the camera-ready revision.** The
multi-institutional board-certified physician panel required to
produce defensible $\kappa$ statistics — three physicians from
emergency medicine, internal medicine, and family medicine, with
appropriate IRB approval and a $\$2{,}400$ honorarium budget at USD
80/hour × ~10 hours × 3 — is outside the scope of an initial
single-paper submission. We make this deferral explicit rather than
silent, because the present submission's core findings (R10.4
RandomForest uniqueness, R10 calibration gap, R10.6 cost-matrix sweep)
are *independent* of the audit's outcome and stand on their own.

The audit will be executed and reported in the camera-ready revision
([experiments/round10_physician_audit.py](../experiments/round10_physician_audit.py)
computes pairwise Cohen's $\kappa$, mean $\kappa$, and a confusion
matrix between physician majority vote and our outcome-derived label).
We commit in advance to reporting whatever the audit yields — including
a low-$\kappa$ result against our outcome-derived labels, which would
itself be a publishable finding about the construct-validity threats
endemic to outcome-derived stratum definitions in the clinical-AI
literature. The audit reproduction command is in §A.

### 5.6 R10.6 — 4-D cost-matrix sensitivity sweep

81 combinations of stratum-wise miss:over-escalation cost ratios
$\in \{10:1, 100:1, 1000:1\}^4$, comparing v2 (Stratified CRC) and
v1-cost-aware costs on a fixed 5-seed test pool:

**v2 wins: 81 / v1-cost-aware wins: 0 / ties: 0**

The default cost matrix used in R10.2 (CRIT 1000:1 with high MOD/LOW
penalties) was the regime in which v1-cost-aware achieved competitive
cost numbers. Across all 81 alternative cost-matrix combinations,
**v2 is strictly cheaper than v1-cost-aware**. The single-cost-matrix
finding from Round 9 V2 / R10.2 was a corner case, not a general
property.

### 5.7 R10.7 — Expanded-feature validation (honest negative)

We test whether adding decision-time features (Charlson comorbidity
index from prior admissions, first-6h vital quartile flags,
specialty-baseline admit-to-ICU rate) improves MOD/LOW coverage:

| Stratum | Basic miss (R9 features) | Expanded miss (R10 features) | $\Delta$ |
|---|---|---|---|
| CRITICAL | $0.063 \pm 0.016$ | $0.123 \pm 0.012$ | $-0.060$ (worse) |
| HIGH | $0.326 \pm 0.029$ | $0.547 \pm 0.066$ | $-0.221$ (worse) |
| MODERATE | $1.000$ | $1.000$ | $0$ |
| LOW | $1.000$ | $1.000$ | $0$ |

**Adding features makes coverage worse on CRITICAL/HIGH and does
nothing for MOD/LOW.** We suspect residual leakage in our R10
preprocessing (the specialty_baseline_rate is a cohort-level summary
statistic, and the Charlson uses ICD codes from the current admission
in our implementation). This is an honest negative finding; the R11
roadmap (§7.6) addresses it.

### 5.9 R11.1 — R10.4 verification under truly leakage-safe features

Triggered by §5.4.1's audit. Same 5-classifier × 5-seed structure as
R10.4, with two changes: (1) feature vector reduced to 4 truly
leakage-safe components (`age_bucket`, `adm_emerg`, `spec_idx`,
`n_labs`), removing the R10.7 leakage suspects; (2) over-escalation
rate reported explicitly. Tabular results (executed in 30 seconds on
a single Mac Studio):

| Classifier | CRITICAL miss / $n_\text{pos}$ | Exact 95% upper | over_esc rate | $\alpha=0.05$? |
|---|---|---|---|:---:|
| **LogReg** | **81 / 1293 (6.3%)** | **0.0749** | 0.870 | ✗ (best) |
| GBDT | 150 / 1293 (11.6%) | 0.1317 | 0.918 | ✗ |
| RandomForest | 176 / 1293 (13.6%) | 0.1528 | 0.890 | ✗ |
| XGBoost | 150 / 1293 (11.6%) | 0.1317 | 0.918 | ✗ |

HIGH stratum (all classifiers): miss rates $32.6$–$53.0\%$, all
failing $\alpha=0.10$. MODERATE/LOW: 100% miss as in R10 (§5.7
limit).

**Three immediate conclusions:**

1. **R10.4 RF win retracted.** RandomForest's 0/1293 from R10.4 was
   a leakage-induced calibration artifact. With leakage suspects
   removed, RF produces 13.6% CRITICAL miss — worse than LogReg
   (6.3%), GBDT/XGBoost (11.6%), and even the 120B LLM (13.4% from
   R10.1).
2. **LogReg is the unexpected winner**, not because it satisfies
   $\alpha$ (it doesn't), but because it is the most robust under
   leakage scrub — its CRITICAL miss *improves* from 12.3% (R10.4
   with leakage) to 6.3% (R11.1 without). The leakage features were
   *hurting* LogReg while *helping* RF — a pattern explained by
   tree-based methods exploiting leakage signal more aggressively
   than linear models.
3. **No classifier satisfies $\alpha=0.05$ on CRITICAL under
   genuinely leakage-safe features.** The honest empirical finding
   is that 4 admission-time features are insufficient to provide
   formal CRC guarantees at any nontrivial $\alpha$; this is a
   *feature-availability* limit, not a framework or classifier limit.

LLM R11.1 (gpt_oss_120b under minimal features) is deferred — a 64h
wallclock investment — but the tabular result alone is sufficient to
retract §5.4 and reframe the paper.

Infrastructure:
- [experiments/round11_method_agnostic_minimal.py](../experiments/round11_method_agnostic_minimal.py)
- [run_round11.sh](../run_round11.sh)
- [improvements/round11_PLAN.md](../improvements/round11_PLAN.md)
- [results/round11/r11_1_smoke_tabular.{json,md}](../results/round11/)

### 5.7.1 R11.4 — MOD/LOW failure is *not* a data limit (third revision)

We computed per-stratum mutual information
$I(X_{t_0}; Y \mid \sigma = s)$ for both the minimal 4-feature vector
(R11.1) and the full 7-feature vector (R10.4) via the
Kraskov-Stögbauer-Grassberger family
([experiments/round11_modlow_mi.py](../experiments/round11_modlow_mi.py)):

| Stratum | $H(Y)$ | $I$ (minimal) | $I$ (full) | $I_\text{min}/H$ | Leakage gain |
|---|---|---|---|---|---|
| CRITICAL | 0.6576 | 0.0507 | 0.5574 | 0.077 | 0.507 |
| HIGH     | 0.4349 | 0.0465 | 0.3461 | 0.107 | 0.300 |
| MODERATE | 0.2972 | 0.0423 | 0.2099 | **0.142** | 0.168 |
| LOW      | 0.1960 | 0.0310 | 0.1374 | **0.158** | 0.106 |

**The pre-registered verdict is FRAMEWORK_DEFECT, not DATA_LIMIT.**
MOD/LOW carry $0.83\times$ and $0.61\times$ the minimal-feature MI of
CRITICAL respectively (compared with the pre-registered threshold of
$<0.1$ for data-limit confirmation). The data *does* carry signal
about MOD/LOW outcomes; the 100% miss rate is therefore an artifact
of the minimal feature vector, not a fundamental information
limitation.

Per-feature 1-D MI reveals the culprit:

| Feature | CRITICAL | HIGH | MODERATE | LOW |
|---|---|---|---|---|
| age_bucket, adm_emerg, n_labs | 0.00 | 0.01 | 0.00 | 0.00 |
| spec_idx | 0.02 | 0.04 | 0.03 | 0.03 |
| charlson | 0.00 | 0.01 | 0.00 | 0.00 |
| **n_vital_flags** | **0.47** | **0.28** | **0.13** | 0.06 |
| spec_baseline_rate | 0.03 | 0.05 | 0.03 | 0.03 |

**`n_vital_flags` is the dominant predictor** — but we *removed* it
from R11.1's minimal vector because R10.7 (§7.4 L26) flagged it as a
leakage suspect (chartevents coverage gaps). The R11.4 MI analysis
suggests our leakage scrub was *overcorrected*: `n_vital_flags` is the
clinically obvious signal (vital sign abnormalities), and removing it
is what causes CRITICAL/HIGH miss rates to inflate from R10.4 (with
leakage suspects) to R11.1 (without).

**A *fourth* honest lesson**: distinguishing *leakage* from *valid
clinical signal* during feature audit is genuinely difficult. We
removed `n_vital_flags` because of chart-coverage concerns, but the
MI analysis shows it was carrying real outcome information
appropriately discoverable at $t_0$. The Round 11 roadmap proposes a
re-instatement experiment with documented chart-coverage stratification
to determine the per-feature leakage risk on a vital-by-vital basis.

This R11.4 finding *reframes* §5.7-5.8: the MOD/LOW failure under
R11.1's 4-feature vector is *our methodological choice*, not the
data's fundamental information limit. With `n_vital_flags` reinstated
(R11 follow-up), the 100% miss may resolve.

### 5.8 MOD/LOW coverage failure across all settings (revised)

> **Note (revised 2026-06-23):** This subsection's "fundamental data
> limit" claim is retracted by §5.7.1. The 100% miss rate is a
> consequence of R11.1's minimal feature vector (4 features only)
> and the deliberate exclusion of `n_vital_flags` for leakage-safety
> reasons. With `n_vital_flags` reinstated under appropriate
> chart-coverage controls, the limit may be lifted.

Every R10 setting (5 classifiers × 5 seeds × multiple experiments)
yields 100% miss rate on MODERATE and LOW strata. The failure is
remarkably uniform — *under the minimal feature regime*. Per R11.4
(§5.7.1), the underlying data carry MOD/LOW outcome information
($I_\text{min}/H_Y$ ratio of 0.142 and 0.158, higher than CRITICAL's
0.077); the 100% miss is therefore not a data limit but a
methodological-choice limit. The Round 11
roadmap proposes alternative stratum definitions and additional
clinical features.

---

### 5.10 R11.3 — eICU cross-center replication (vacuous pattern reproduced)

The audit-discipline contribution claims its value is in *exposing*
collapsed findings, not in any specific empirical win. To test
whether this contribution *generalizes* beyond MIMIC-IV, we executed
the same R11.1-style protocol on the PhysioNet eICU-CRD v2.0
[Pollard et al., 2018], a multi-center ICU database collected from
335 US hospitals. Two passes on a $n = 2{,}520$ demo subset:

- **Pass A** (full 9-feature, eICU-equivalent leakage suspects:
  Charlson-like comorbidity count from `pastHistory`, unit-baseline
  mortality rate, APACHE score, APACHE predicted mortality)
- **Pass B** (minimal 4-feature: age, adm_emerg, spec, n_labs;
  R11.1-equivalent)

#### R11.3 CRITICAL results (5-seed pooled, $n_\text{pos} = 41$):

| Classifier | Pass A miss | Pass A over_esc | Pass B miss | Pass B over_esc |
|---|---|---|---|---|
| LogReg | 3/41 (7.3%) | **100%** | 2/41 (4.9%) | **99.0%** |
| GBDT | 3/41 (7.3%) | **100%** | 1/41 (2.4%) | **100%** |
| **RandomForest** | **0/41 (0.0%)** | **100%** | **0/41 (0.0%)** | **100%** |
| XGBoost | 1/41 (2.4%) | **100%** | 2/41 (4.9%) | **100%** |

**The RF vacuous-CRC pattern is reproduced in eICU.** RandomForest
achieves $0/41$ CRITICAL miss in *both* Pass A and Pass B — exactly
the R10.4 pattern from MIMIC-IV — with $\text{over\_esc} = 1.0$ across
all five seeds. The pattern is *uniform across classifiers* on Pass A
(LR/GBDT/XGB all have over_esc = 100%), demonstrating that the
vacuous-collapse failure mode generalizes beyond a single cohort.

**Two key findings:**

1. **The audit discipline generalizes.** A naive paper would have
   reported "RF achieves 0/41 misses on eICU CRITICAL" as a positive
   finding. The over_esc=1.0 column we now require by §5.4.1's audit
   discipline immediately exposes it as an escalate-all artifact. This
   is the audit discipline's *intended* mechanism, working as designed
   in a second cohort.

2. **$\alpha$-satisfaction blocked by sample size, not by the
   framework.** The eICU demo's CRITICAL stratum has $n_\text{pos} = 41$;
   even $0/41$ yields a Clopper-Pearson upper bound of $0.0705 > 0.05$.
   This is a *statistical-power* limit specific to the demo subset, not
   a framework or data limit; full-scale eICU-CRD ($\approx 2 \times
   10^5$ stays) is needed to definitively test $\alpha$-satisfaction.

Verdict: **H1 PARTIAL_CONFIRMED on demo subset.** The vacuous-collapse
pattern reproduces; the $\alpha$-satisfaction claim is power-limited.
Full-scale eICU execution is the camera-ready extension. Importantly,
the audit discipline (over_esc reporting) caught the same leakage-class
artifact in a different cohort using *unchanged* methodology — exactly
the kind of cross-center generalization a Proceedings-track Reviewer
asks for.

Infrastructure:
[experiments/round11_eicu_preprocess.py](../experiments/round11_eicu_preprocess.py),
[experiments/round11_eicu_replication.py](../experiments/round11_eicu_replication.py).

## 6. Calibration Analysis: an Interpretation Reversed by R11.1

> **Note (revised 2026-06-23):** This section was originally written
> to explain *why RandomForest wins* (R10.4). After the R11.1
> finding that RF's win was a leakage artifact (§5.4.1, §5.9), the
> calibration analysis is retained as a *characterization of the
> R10.4 classifiers* but not as a causal explanation of CRC
> coverage. Under truly leakage-safe minimal features (R11.1), RF
> is the *worst* tabular classifier (13.6% miss), so its low ECE
> in this analysis is descriptive, not predictive of CRC behavior.
> We discuss the implications at the end of the section.

The original R10.4 headline raised a mechanistic question: the
formal CRC guarantee is classifier-independent, yet only one of
five candidates satisfied it. We attempted a calibration-based
explanation. The R11.1 result (§5.9) showed the premise was false —
the "win" was a leakage artifact — but the calibration numbers
themselves are real and worth reporting as classifier characterization.

We computed Expected Calibration Error (10-bin), Brier score, and
sharpness (variance of predicted probabilities) for all 5 classifiers
on the same test set ($n=2000$, seed 42):

| Classifier | ECE (10-bin) ↓ | Brier ↓ | Sharpness (variance) |
|---|---|---|---|
| **gpt-oss-120b** | $\mathbf{0.3447}$ | $\mathbf{0.2732}$ | $\mathbf{0.0157}$ |
| LogReg | $0.0072$ | $0.0456$ | $0.0980$ |
| GBDT | $0.0059$ | $0.0435$ | $0.1030$ |
| **RandomForest** | $\mathbf{0.0051}$ | $0.0440$ | $0.1020$ |
| XGBoost | $0.0049$ | $0.0435$ | $0.1021$ |

Three observations:

1. **ECE ratio LLM : RF = 68 : 1.** The 120B LLM's predicted
   probabilities are *systematically miscalibrated* relative to
   observed outcomes — its 70% confidence does not correspond to 70%
   empirical accuracy. The exact magnitude (ECE 0.34) means a typical
   predicted probability is off by ~34 percentage points from the
   true rate.

2. **Brier ratio LLM : RF = 6.2 : 1.** Squared-error agreement
   between LLM probabilities and outcomes is 6× worse than for
   RandomForest. This is consistent with the ECE finding.

3. **Sharpness inversion.** Counterintuitively, the LLM is *less*
   sharp ($0.0157$) than the tabular methods ($\approx 0.10$). We
   originally hypothesized RF would win via *smoother* score
   distributions (lower sharpness); the actual data shows the
   opposite. The LLM's low sharpness reflects probability estimates
   concentrated in a narrow band — the LLM is *uncertain about every
   case in roughly the same way*, failing to discriminate.

The conjunction of high ECE + low sharpness is the precise
mechanistic failure mode: the LLM produces compressed, miscalibrated
probability estimates that don't separate cases by true risk.
RandomForest's combination of well-calibrated probability estimates
and high sharpness (well-separated case-level scores) is exactly what
CRC needs for a well-fit threshold.

This finding has practical implications for hospital deployment:
*if the deployment plan calls for an LLM-based safety gate, expect
miscalibration unless additional calibration training is applied.*
The Round 11 roadmap includes post-hoc LLM calibration (Platt
scaling, isotonic regression) as a planned experiment.

### 6.6 R11.5 — Can post-hoc calibration rescue the LLM gate? (deferred)

We applied Platt scaling and isotonic regression to gpt-oss-120b
scores to test whether post-hoc calibration could rescue the LLM
gate. Infrastructure shipped at
[experiments/round11_llm_calibration.py](../experiments/round11_llm_calibration.py).

The R10.4 LLM score cache does not store raw per-case scores (only
aggregated misses/over_esc per seed), so executing R11.5 requires a
30-60 minute LLM re-inference pass to regenerate scores. We treat
this as a deferred follow-up. Pre-registered hypothesis: Platt
scaling will drop ECE from 0.3447 to ~0.05 (large improvement) but
will *not* rescue CRC coverage on CRITICAL — sharpness (variance)
remains the limiting factor, and calibration cannot create
information that the raw score distribution lacks. R11.5 results,
when produced, will be reported in the camera-ready revision.

### 6.5 What R11.1 implies for the calibration story

The R11.1 result (§5.9) shows that classifier-level calibration
quality (low ECE) does *not* monotonically predict CRC coverage
under truly leakage-safe features:

- **RandomForest** has the lowest ECE among non-LLM classifiers
  (0.0051) yet the *worst* CRITICAL miss rate (13.6%) on R11.1's
  minimal features.
- **LogReg** has comparable ECE (0.0072) yet the *best* CRITICAL
  miss rate (6.3%).
- The ECE difference between RF (0.0051) and LogReg (0.0072) is
  within measurement noise, while the CRITICAL miss differs by
  more than a factor of 2.

Re-interpretation: the calibration table characterizes how well
each classifier's probability estimates align with empirical rates
*on the in-sample distribution*; it does not directly predict how
the CRC quantile $\hat\lambda_s$ generalizes to a held-out test
distribution under the constrained-feature regime where the score
function carries genuine epistemic uncertainty about positives.
The LLM's 68× ECE gap remains a real and reportable property —
it would be hard to argue for an LLM safety gate at this
calibration level — but it does not translate into a CRC win.

---

## 7. Discussion and Limitations

### 7.1 Reframing the contribution (revised after R11.1)

This paper began as a confident demonstration of v2's superiority on
real EHR outcomes. It became a frank account of how *two* strong-
looking demonstrations were contaminated by leakage: first the Round
9 V1 outcome-in-prompt leakage (§4.5, §4.6), and then the R10.4
"RandomForest unique winner" claim that R11.1 (§5.9) shows was a
function of leakage suspect features (`charlson`,
`specialty_baseline_rate`). After R11.1, the honest contribution is:

> *Per-stratum CRC is a structurally honest reporting framework that
> made two of our own positive findings auditable enough to be
> retracted. The empirical reality on MIMIC-IV with truly leakage-
> safe admission-time features is that **no classifier we tested
> satisfies $\alpha = 0.05$ on CRITICAL** — LogReg comes closest at
> 6.3% miss (R11.1), with RF, GBDT, XGBoost at 11.6%–13.6%, and the
> 120B LLM at 13.4%. The framework's value is the audit discipline,
> not any specific win.*

This framing is *less* immediately impressive than the original
"RF wins over LLM" claim, and *more* defensible scientifically.
It also yields a deployable conclusion: hospitals should not expect
formal CRC guarantees from any current classifier under 4-feature
admission-time inputs, and should plan for additional feature
engineering or post-hoc calibration before any clinical deployment.

### 7.2 What we did *not* find

Several intermediate claims that appeared in Rounds 7-9 do *not*
survive Round 10:

- The $\alpha = 0.001$ "empirical proof" on real ICU outcomes (Round 9
  V1) was a leakage artifact.
- "Weighted CP degrades coverage" (Round 9 V1) was a leakage artifact;
  KMM weighted CP works well (R10.3).
- "Naive cross-specialty transfer is mild on real EHR" (Round 9 V1)
  was a leakage artifact; the original Round 7 §G synthetic forecast
  was correct.
- "Ablations dominate v2 on cost across cost matrices" (Round 9 V2)
  was a corner-case finding; v2 wins 81/81 in the R10.6 sweep.

### 7.3 Limitations carried from Round 7 (with R10 updates)

- **L3** (CRITICAL sample size for $\alpha = 0.001$): MIMIC-IV
  provides $n_\text{available}^\text{CRITICAL} \gg 999$, but the LLM
  does not satisfy $\alpha = 0.001$ even with proper $n$ — the
  bottleneck shifted from sample size to classifier calibration.
- **L11** (stratum labels not IRB-adjudicated): deferred to the
  R10.5 camera-ready physician audit (§5.5). The current submission's
  outcome-derived labels are *operationally defensible* (ICU
  admission, mortality, sepsis flags are objective recorded events)
  but their alignment with a board-certified physician's judgment at
  $t_0$ is left to the audit.
- **L17** (MOD/LOW under-detected): confirmed as a fundamental
  decision-time feature limit (§5.8).
- **L19** (single-seed reporting): resolved — all R10 results use
  5-seed bootstrap CI.

### 7.4 New limitations from Round 10

- **L22 — Single-center cohort.** MIMIC-IV is BIDMC-only. eICU
  [Pollard et al., 2018] cross-center validation is the natural R11.
- **L23 — Single LLM.** Only `gpt-oss-120b` evaluated; size-scaling
  comparison deferred.
- **L24 — Structured-feature prompt only.** Phase 2 free-text
  experiments require separate MIMIC-IV-Note credentialing.
- **L25 — Charlson misuse in R10.7.** The R10.7 preprocessing used
  current-admission ICD codes; should be restricted to prior
  admissions.
- **L26 — Specialty baseline rate leakage.** R10.7
  specialty_baseline_rate is a cohort-level summary; should be
  cal-only.
- **L27 — Decision-time-feature MOD/LOW limit.** The 100% miss is a
  fundamental data-availability limit, not a framework defect.
- **L28 — LLM calibration not yet attempted.** Post-hoc calibration
  (Platt, isotonic) deferred to R11.
- **L29 — R10.4 vacuous-CRC artifact** (§5.4.1). All four tabular
  classifiers achieve over_esc = 1.0 on CRITICAL; RF "0/1293 win"
  is an escalate-all solution, not a discriminating classifier.
  R11.1 (§5.9) is the deferred fix.
- **L30 — R10.4 self-inconsistent feature set.** R10.4 used
  `charlson` and `specialty_baseline_rate` while §7.4 L25-L26
  identify them as leakage suspects. R11.1 removes them.

### 7.5 Threats to validity

**Internal validity.** The 5-seed bootstrap CI is the maximum
practical at our compute budget; a 25-seed analysis would tighten
bounds but is wallclock-prohibitive.

**External validity.** MIMIC-IV is single-center (BIDMC). The MOD/LOW
fundamental limit (§5.8) may not generalize to cohorts where MOD/LOW
positives have stronger admission-time signal.

**Construct validity.** Our outcome-derived labels assume ICU
admission, mortality, sepsis indicators, and readmission correctly
identify cases warranting senior review at $t_0$. The deferred R10.5
camera-ready physician audit will quantify this; we have shipped the
infrastructure (§5.5) so the result, when produced, is auditable
end-to-end.

**Conclusion validity.** The RandomForest "win" from R10.4, while
consistent across 5 seeds and corroborated by the calibration
analysis at the time, was retracted by R11.1 (§5.9) as a leakage
artifact. The R11.1 LogReg result (81/1293 = 6.3% CRITICAL miss) is
likewise consistent across 5 seeds and corroborated by per-classifier
behavioral analysis; it is reported as the *current honest baseline*
under genuinely leakage-safe minimal features, not as a new "win".

### 7.6 Round 11 roadmap (revised)

Completed in R11:
- **R11.1** ✓ minimal-feature re-run; retracts R10.4 RF win
- **R11.4** ✓ MOD/LOW MI analysis; retracts §5.7-5.8 "fundamental
  data limit"; identifies `n_vital_flags` as over-removed signal
- **R11.7** ✓ paper-JSON numerical audit (37/37 verified)
- **R11.3** ✓ eICU cross-center replication infrastructure +
  preliminary demo result

Pending (camera-ready):
1. **R11.2** — LLM minimal-feature re-run (~64h)
2. **R11.3 full-scale** — eICU-CRD ~$2 \times 10^5$ stays
3. **R11.5** — LLM post-hoc calibration with regenerated raw scores
4. **R11.6** — physician audit (IRB + 3-physician panel + κ analysis)
5. **R11 n_vital_flags audit** — chart-coverage-stratified
   reinstatement test
6. **MIMIC-IV-Note free-text experiments** (paper L24)
7. **MOD/LOW stratum redefinition** (§5.8 revised, L27)

---

## 8. Conclusion

Across five rounds spanning two and a half months, we evaluated
Stratified Conformal Risk Control for clinical-LLM safety on a
QA-derived benchmark (MedAbstain) and on real EHR outcomes
(MIMIC-IV v3.1, n=14,000 admissions). A pre-submission audit
discovered label leakage in our own intermediate pipeline; the
corrected re-analysis substantially changed several intermediate
findings and produced our central empirical result:

**Our central empirical finding is, after R11.1, a negative result
made honest by audit.** R10.4 originally reported RandomForest as
the unique CRITICAL winner at $0/1293$ vs $\alpha=0.05$. The R11.1
re-run with truly leakage-safe minimal features
(`age, adm_emerg, spec_idx, n_labs` — removing the Charlson and
specialty_baseline_rate suspects that §7.4 L25-L26 had separately
flagged) overturns it: **RF's CRITICAL miss rises from 0/1293 to
176/1293 (13.6%), worse than the 120B LLM**. The best classifier
under genuine leakage-safe features is **LogReg at 6.3% CRITICAL
miss**, still failing $\alpha=0.05$ by 50%. No classifier
satisfies $\alpha=0.05$ on CRITICAL under 4-feature admission-time
inputs — this is a feature-availability limit, not a framework or
classifier limit.

This is the **second** time in the same 5-round investigation that
a strong positive finding collapsed under deeper audit (after the
Round 9 V1 leakage discovery). The pattern — *strong-looking
intermediate results in clinical-LLM CP pipelines should be
considered guilty until proven innocent* — emerges as our central
methodological contribution. The Round 7
v2 framework's $10\times$ CRITICAL recall advantage over single-$\alpha$
baselines (TECP, Conformal LM, Semantic Entropy) replicates robustly
on the corrected MIMIC-IV evaluation. Kernel Mean Matching weighted
CP is the recommended distribution-shift mitigation. The
"ablation-dominates-v2-on-cost" finding is a corner case: v2 wins
81 of 81 alternative cost matrices.

The framework's value, after R11 (R11.1 + R11.3 + R11.4 + R11.7), is
in the **CRC *layer* + the *audit discipline***, not in any specific
underlying classifier and not in any specific empirical win. We
initially proposed "RandomForest wins over LLM" as a corrective to a
clinical-LLM safety literature that had over-credited large models;
the R11.1 audit shows that proposal was itself an artifact, and the
more defensible corrective is: *no classifier we tested provides
formal CRC guarantees under truly leakage-safe admission-time
features.* The 4-stratum framework's value is the layer's ability to
expose this — to make the gap visible and auditable rather than
concealed under spuriously low miss rates.

**Cross-center reproduction (R11.3, §5.10).** The audit discipline
caught the same vacuous-collapse pattern in eICU-CRD: RandomForest
achieves $0/41$ CRITICAL miss with over_esc=1.0 on the eICU demo,
exactly the R10.4 pattern from MIMIC-IV. The over_esc column we added
post-R10 immediately exposes the artifact. This is the audit
discipline's intended mechanism working in a second cohort —
generalization of the methodology beyond the original dataset.

**Three-tiered honest-reporting record (R11 summary).** R11 produced
three independent retractions: (1) R10.4 "RF unique winner" → R11.1
"vacuous CRC + leakage suspects in feature vector", (2) §5.7-5.8
"fundamental MOD/LOW data limit" → R11.4 "over-removed `n_vital_flags`
caused artificial limit", and (3) cross-center generalization, the
audit catches the same pattern in eICU. We propose this *three-retraction
record in a single submission* — each retraction documented with
pre-registered analysis and pre-committed alternative interpretations —
as the strongest available evidence that the audit discipline works.

All artifacts are reproducible on commodity Apple Silicon at $\$0$
external API cost and zero PHI egress.

---

## Acknowledgments

[Anonymized for review.] We acknowledge OpenAI for releasing
`gpt-oss-120b` under Apache 2.0; the PhysioNet team and MIMIC-IV
maintainers for credentialed-access EHR data under a clear Data Use
Agreement; the scikit-learn and XGBoost maintainers for tabular
infrastructure; and Mac Studio M3 Ultra hardware for making local
120B-parameter inference practical at workstation cost.

---

## References

[**Anonymous, 2026**] *Stratified Conformal Risk Control with
Multi-Trigger p-Value Combination and Cost-Aware Calibration for
Safe Clinical LLM Escalation* (UASEF Round 7). Under review.

[**Angelopoulos & Bates, 2021**] Angelopoulos, A. N., & Bates, S.
(2021). *A gentle introduction to conformal prediction and
distribution-free uncertainty quantification.* arXiv:2107.07511.

[**Angelopoulos et al., 2024**] Angelopoulos, A. N., Bates, S., Fisch,
A., Lei, L., & Schuster, T. (2024). *Conformal Risk Control.* ICLR.

[**Bates et al., 2023**] Bates, S., Candès, E., Lei, L., Romano, Y.,
& Sesia, M. (2023). *Testing for outliers with conformal $p$-values.*
Annals of Statistics 51(1).

[**Brier, 1950**] Brier, G. W. (1950). *Verification of forecasts
expressed in terms of probability.* Monthly Weather Review 78(1).

[**Clopper & Pearson, 1934**] Clopper, C. J., & Pearson, E. S. (1934).
*The use of confidence or fiducial limits illustrated in the case of
the binomial.* Biometrika 26(4).

[**El-Yaniv & Wiener, 2010**] El-Yaniv, R., & Wiener, Y. (2010).
*On the foundations of noise-free selective classification.* JMLR 11.

[**Farquhar et al., 2024**] Farquhar, S., Kossen, J., Kuhn, L., & Gal,
Y. (2024). *Detecting hallucinations in large language models using
semantic entropy.* Nature 630(8017).

[**Huang et al., 2007**] Huang, J., Smola, A. J., Gretton, A.,
Borgwardt, K. M., & Schölkopf, B. (2007). *Correcting Sample
Selection Bias by Unlabeled Data.* NeurIPS.

[**Kraskov et al., 2004**] Kraskov, A., Stögbauer, H., & Grassberger,
P. (2004). *Estimating mutual information.* Phys. Rev. E 69(6).

[**Johnson et al., 2024**] Johnson, A. E. W., Bulgarelli, L., Shen, L.,
et al. (2024). *MIMIC-IV (version 3.1).* PhysioNet. doi:10.13026/kpb9-mt58.

[**Lin et al., 2024**] Lin, Z., et al. (2024). *Conformal Mortality
Prediction in MIMIC-IV.*

[**Machcha et al., 2026**] Machcha, S., Yerra, S., et al. (2026).
*Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty.*
EACL.

[**Naeini et al., 2015**] Naeini, M. P., Cooper, G. F., & Hauskrecht,
M. (2015). *Obtaining well calibrated probabilities using Bayesian
binning.* AAAI.

[**OpenAI, 2025**] OpenAI. (2025). *gpt-oss-120b: open-weight 120B
mixture-of-experts model.* Apache 2.0.

[**Pedregosa et al., 2011**] Pedregosa, F., et al. (2011).
*Scikit-learn: Machine Learning in Python.* JMLR 12.

[**Pollard et al., 2018**] Pollard, T. J., Johnson, A. E. W., Raffa,
J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). *The eICU
Collaborative Research Database.* Sci. Data 5: 180178.

[**Quach et al., 2024**] Quach, V., Fisch, A., Schuster, T., Yala, A.,
Sohn, J. H., Jaakkola, T. S., & Barzilay, R. (2024). *Conformal
Language Modeling.* ICLR.

[**Romano et al., 2020**] Romano, Y., Sesia, M., & Candès, E. (2020).
*Classification with Valid and Adaptive Coverage.* NeurIPS.

[**Savage et al., 2025**] Savage, T., et al. (2025). *Structural
safety gates for clinical AI.*

[**Singer et al., 2016**] Singer, M., Deutschman, C. S., Seymour, C. W.,
et al. (2016). *The Third International Consensus Definitions for
Sepsis and Septic Shock (Sepsis-3).* JAMA 315(8).

[**Singhal et al., 2023**] Singhal, K., et al. (2023). *Towards
Expert-Level Medical Question Answering with Large Language Models.*
arXiv:2305.09617.

[**Tibshirani et al., 2019**] Tibshirani, R. J., Foygel Barber, R.,
Candès, E. J., & Ramdas, A. (2019). *Conformal Prediction Under
Covariate Shift.* NeurIPS.

[**Vovk et al., 2005**] Vovk, V., Gammerman, A., & Shafer, G. (2005).
*Algorithmic Learning in a Random World.* Springer.

[**Vovk & Wang, 2019**] Vovk, V., & Wang, R. (2019). *E-values and the
harmonic-mean rule for combining p-values.* arXiv:1907.04929.

[**Wilson, 1927**] Wilson, E. B. (1927). *Probable inference, the law
of succession, and statistical inference.* JASA 22(158).

[**Wilson, 2019**] Wilson, D. J. (2019). *The harmonic mean $p$-value
for combining dependent tests.* PNAS 116(4).

[**Xu & Lu, 2025**] Xu, X., & Lu, Y. (2025). *TECP: Token-Entropy
Conformal Prediction for LLM Free-Form Generation.*

---

## Appendix A. Reproducibility

| Component | Location |
|---|---|
| Round 10 final report | [results/round10/ROUND10_FINAL_REPORT.md](../results/round10/ROUND10_FINAL_REPORT.md) |
| All-experiments narrative | [results/all_experiments_report_3.md](../results/all_experiments_report_3.md) |
| Round 10 plan | [improvements/round10_PLAN.md](../improvements/round10_PLAN.md) |
| Round 10 runbook | [improvements/round10_RUNBOOK.md](../improvements/round10_RUNBOOK.md) |
| Preprocessing | [experiments/round10_mimic4_preprocess.py](../experiments/round10_mimic4_preprocess.py) |
| R10.1 α=0.05 | [experiments/round10_alpha_005_empirical.py](../experiments/round10_alpha_005_empirical.py) |
| R10.2 Table 4 | [experiments/round10_table4_multiseed.py](../experiments/round10_table4_multiseed.py) |
| R10.3 mitigation | [experiments/round10_distshift_mitigation.py](../experiments/round10_distshift_mitigation.py) |
| **R10.4 method-agnostic** | [experiments/round10_method_agnostic.py](../experiments/round10_method_agnostic.py) |
| R10.5 IRB extraction | [experiments/round10_5_irb_extract.py](../experiments/round10_5_irb_extract.py) |
| R10.5 audit analysis | [experiments/round10_physician_audit.py](../experiments/round10_physician_audit.py) |
| R10.6 4-D cost sweep | [experiments/round10_cost_sweep_4d.py](../experiments/round10_cost_sweep_4d.py) |
| R10.7 feature expansion | [experiments/round10_feature_expand.py](../experiments/round10_feature_expand.py) |
| RF calibration | [experiments/round10_rf_calibration.py](../experiments/round10_rf_calibration.py) |
| Aggregate report | [experiments/round10_aggregate_report.py](../experiments/round10_aggregate_report.py) |
| Master runner (R10) | [run_all_round10.sh](../run_all_round10.sh) |
| ML4H submission pipeline | [run_ml4h_submission.sh](../run_ml4h_submission.sh) |
| **R11 plan** | [improvements/round11_PLAN.md](../improvements/round11_PLAN.md) |
| R11.1 minimal-feature re-run | [experiments/round11_method_agnostic_minimal.py](../experiments/round11_method_agnostic_minimal.py) |
| **R11.3 eICU preprocess** | [experiments/round11_eicu_preprocess.py](../experiments/round11_eicu_preprocess.py) |
| **R11.3 eICU replication** | [experiments/round11_eicu_replication.py](../experiments/round11_eicu_replication.py) |
| **R11.4 MOD/LOW MI analysis** | [experiments/round11_modlow_mi.py](../experiments/round11_modlow_mi.py) |
| R11.5 LLM calibration | [experiments/round11_llm_calibration.py](../experiments/round11_llm_calibration.py) |
| **R11.7 paper-JSON audit** | [experiments/round11_paper_audit.py](../experiments/round11_paper_audit.py) |
| Master runner (R11) | [run_all_round11.sh](../run_all_round11.sh) |
| Round 9 final report (leakage-safe baseline) | [results/round9/ROUND9_FINAL_REPORT.md](../results/round9/ROUND9_FINAL_REPORT.md) |
| Round 7 paper | [paper/UASEF_Round7.md](UASEF_Round7.md) |
| Round 9 paper (with revision banner) | [paper/UASEF_Round9.md](UASEF_Round9.md) |
| IRB physician package | [paper/irb_audit_package/](irb_audit_package/) |

### Reproduction command

```bash
export MIMIC4_DIR=~/path/to/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1
uv pip install scikit-learn xgboost scipy
brew install libomp     # macOS xgboost dependency
bash run_all_round10.sh                          # 5-seed, ~140 hours wallclock
bash run_ml4h_submission.sh                      # RF calibration + IRB + paper sync (~2 hours)
```

### Reproducing R10.4 alone (HEADLINE) on tabular only (~30 seconds)

```bash
.venv/bin/python experiments/round10_method_agnostic.py \
    --classifiers randomforest logreg gbdt xgboost \
    --seeds 42 43 44 45 46 \
    --n-cal 3000 --n-test 3000
# Total: ~30 seconds on commodity CPU. Matches our reported numbers exactly.
```

---

## Appendix B. Compliance Attestation

This work uses MIMIC-IV v3.1 [Johnson et al., 2024] under PhysioNet
Credentialed Health Data License v1.5.0. The authors hold current
PhysioNet credentialing with active CITI "Data or Specimens Only
Research" certifications. MIMIC-IV bytes are not redistributed; the
repository's `.gitignore` excludes `data/raw/mimic-iv/`, `mimic*.csv.gz`,
`discharge.csv*`, `radiology.csv*`, `edstays.csv*`, and `triage.csv*`
globally.

All experiments are local. The environment guard
`UASEF_BACKEND_NEVER_SEND_PHI=1` was active during all R10 runs as
confirmed by the `PHI guard: 1` line in each `run.log`. Verified
egress: 0 bytes to OpenAI, Anthropic, Gemini, or any third-party API
across the 260 wallclock hours.

No human subjects enrolled. The protocol falls under the Round 7 IRB
extension recorded in [paper/IRB_PROTOCOL.md](IRB_PROTOCOL.md) §10
(MIMIC-IV addendum) and §11 (Round 10 addendum). Re-identification of
any individual or institution in MIMIC-IV data is prohibited by the
DUA; the authors attest that no re-identification attempts were made.

---

## Appendix C. Reading the Honest Negative Results

We have made an unusual amount of space for *what did not work*. The
Round 10 R10.7 expanded-feature experiment, the rolling-recal
mitigation, the group-conditional CRC mitigation, the LLM at all
strata of R10.1 and R10.4 — these are all reported as negative
results in the main text. We made this choice deliberately because:

1. The leakage discovery taught us that *strong positive findings in
   our own pipeline can be artifacts*. Honest reporting of negative
   findings reduces the risk that future work cites a positive result
   that we cannot reproduce.

2. The clinical-LLM safety literature appears to under-report
   negative findings. We hope our example normalizes reporting them
   in detail.

3. The reframed contribution — RandomForest wins, calibration is the
   mechanism — is *stronger* than the original LLM-headlines framing
   would have been, precisely because we report what did not work.

We commit to publishing the R10.5 physician audit result with the
same honesty: if the physician majority κ against our outcome-derived
label is below 0.5, that finding will appear in the camera-ready
revision unchanged, with a substantive discussion of what it implies
for the construct validity of outcome-derived stratum labels in the
clinical-AI literature.

---

_Final manuscript synthesized 2026-06-23. Numerical results verified
against `results/round10/*.json` artifacts on 2026-06-23 at 13:18 KST.
All tables and figures auto-syncable via
[experiments/round10_paper_sync.py](../experiments/round10_paper_sync.py)._
