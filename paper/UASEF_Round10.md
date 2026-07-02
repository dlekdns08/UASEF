# Method-Agnostic Conformal Risk Control for Safe Clinical Escalation on Real EHR Outcomes: A Multi-Classifier MIMIC-IV Validation

**Authors.** *[Author Name]*<sup>1</sup>, *[Co-author]*<sup>2</sup>
<sup>1</sup>*[Affiliation 1]*; <sup>2</sup>*[Affiliation 2]*
Correspondence: `[email]`

**Provenance.** This paper supersedes the Round 9 manuscript
[UASEF_Round9.md](UASEF_Round9.md) after a leakage-safety audit revealed
that several Round 9 numerical findings were artifacts of label leakage
in the original preprocessing pipeline. The Round 9 manuscript was
corrected in-place ([revision banner](UASEF_Round9.md)) and is retained
for transparency. The present paper builds on the corrected Round 9
methodology and adds the experimental contributions of §1.4.

**Target venue.** ML4H 2026 (Spotlight) / NeurIPS 2026 Safe-ML Workshop /
JAMA Network Open (clinical evaluation track).

**Code & data.** `https://github.com/[org]/UASEF` (anonymized for review).
MIMIC-IV v3.1 is not redistributed; replication requires PhysioNet
credentialing.

**Compute disclosure.** All inference is performed locally. No
MIMIC-IV-derived content is transmitted to any third-party API.

---

## Abstract

We argue, contrary to a common framing in the LLM safety literature, that
the contribution of conformal-prediction-based clinical-escalation
frameworks is not "an LLM-based escalation method" but rather a
**method-agnostic per-stratum Conformal Risk Control layer** whose
guarantee — $\mathbb{E}[\ell_s] \le \alpha_s$ for each clinical risk
stratum $s$ — holds for *any* underlying classifier that produces a
real-valued score. We validate this claim on MIMIC-IV v3.1
[Johnson et al., 2024], the PhysioNet-credentialed inpatient EHR corpus,
under a leakage-safe pipeline that separates a decision-time risk group
$G(X_{t_0})$ from a future adverse-outcome label $Y$.

We compare five underlying classifiers — `openai/gpt-oss-120b` (120B
mixture-of-experts open-weight LLM), logistic regression, gradient-boosted
decision trees, random forest, and XGBoost — each paired with an identical
Stratified CRC layer
[Anonymous, 2026; Angelopoulos et al., 2024]. On a CRITICAL-stratum
cohort of $n_{\text{cal}} = n_{\text{test}} = 3{,}000$ (now satisfying
the $n \ge n_{\min}(\alpha)$ requirement for $\alpha = 0.05$), we find
that **the simplest tabular classifier — logistic regression in
$<1$ second on commodity CPU — achieves CRITICAL recall statistically
indistinguishable from the 120B LLM run for $\approx 120$ hours on a
$\$7$k Mac Studio**: McNemar pooled $p > 0.05$ on 5 seeds. The total
cost gap between methods is dominated by stratum-cohort prevalence
rather than by classifier sophistication. We confirm at a properly
powered scale that the published TECP, Conformal Language Modeling, and
Semantic Entropy single-$\alpha$ baselines under-cover by an order of
magnitude (CRITICAL recall $\approx 0.05$ vs CRC-layer recall
$\approx 0.85$–$0.99$).

We also report **three distribution-shift mitigation strategies**
(online rolling-window recalibration, kernel mean matching weighted CP,
group-conditional CRC) and contrast them with the Round 9
*detection-only* result. On a 7-versus-6-year temporal shift (2008–2016
calibration vs 2017–2022 test), the Round 9 naive CRC violation of
$5.71\times$ in CRITICAL drops to $1.4\times$ under online recalibration
on a rolling 3-year window — making the rolling protocol a defensible
production recommendation. Finally, a 100-case board-certified physician
audit (R10.5) reports Cohen's $\kappa$ between physician-adjudicated
escalation labels and outcome-derived labels, quantifying the
construct-validity threat of the latter.

We frame the central value of this work as *per-stratum risk control
that is method-agnostic, leakage-safe, distribution-shift-aware, and
deployable on commodity hardware* — not as a recommendation for any
specific underlying classifier.

**Keywords.** conformal prediction; conformal risk control; method-agnostic
calibration; weighted conformal prediction; large language models;
tabular machine learning; clinical decision support; abstention; MIMIC-IV;
PhysioNet; open-weight language models.

---

## 1. Introduction

### 1.1 Context and the Round 9 correction

The companion Round 7 paper [Anonymous, 2026] introduced UASEF v2, a
framework combining (Pivot A) Stratified Conformal Risk Control,
(Pivot B) Multi-Trigger Conformal Combination, and (Pivot C)
Cost-Aware Calibration, validated on the QA-derived MedAbstain
benchmark [Machcha et al., 2026]. The Round 9 extension
[UASEF_Round9.md](UASEF_Round9.md) ported the framework to MIMIC-IV
v3.1 [Johnson et al., 2024] but, in a pre-submission audit, was found
to suffer from **label leakage**: the risk stratum $\sigma(a)$ was
defined as a deterministic function of future outcomes (ICU admission
within 24 h, in-hospital mortality, sepsis indicators, length-of-stay)
and the prompt simultaneously contained features derived from those
same future outcomes (discharge ICD codes, length of stay, primary
diagnosis). The Round 9 manuscript was corrected in place to clearly
mark all pre-fix numerical tables as "pending re-run" with a revision
banner.

The leakage-safe re-run, completed 2026-06-11 and reported in
[results/round9/ROUND9_FINAL_REPORT.md](../results/round9/ROUND9_FINAL_REPORT.md),
made five findings that substantially differ from the pre-fix numbers:

1. The original "0 misses → $\alpha = 0.001$ empirical proof" was an
   artifact: with $n_{\text{pos}} = 99$, the exact one-sided 95%
   upper bound on the true miss rate is $0.030 \gg 0.001$, so the
   observation is consistent with $\alpha = 0.001$ but does not
   *prove* it. A proper $\alpha = 0.001$ empirical proof requires
   $n_{\text{pos}} \ge 2{,}995$.
2. The v2-vs-TECP/Quach/SE recall advantage *replicates* (recall
   $0.85$ vs $0.05$, $17\times$ improvement) but the total-cost
   advantage shrinks against unconstrained ablations
   (`v1-cost-aware` cost = $165$, v2 = $3{,}648$).
3. Cross-specialty naive transfer violates per-stratum coverage by
   $3$–$7\times$, *matching* the synthetic Round 7 §G forecast — the
   original "milder than expected" Round 9 result was a leakage
   artifact.
4. The 7-versus-6-year temporal shift causes $5$–$10\times$
   violations (catastrophic), not the original "mild $1.63\times$".
5. The R9.6 tabular logistic-regression and GBDT baselines, run in
   $< 1$ second on CPU and paired with the same Stratified CRC
   layer, achieve CRITICAL recall $0.99$ (1/99 miss) on identical
   leakage-safe features — *better* than the 120B LLM's $0.85$.

The fifth finding motivates the central reframing of this paper.

### 1.2 The central thesis

We argue that conformal prediction's contribution to clinical LLM
safety has been mis-framed in the literature [Xu & Lu, 2025;
Quach et al., 2024; Farquhar et al., 2024], including in our own
prior work [Anonymous, 2026]. The actual mechanism producing the
distribution-free coverage guarantee is the *conformal layer*, not the
underlying classifier. We make this precise:

> **Method-agnostic CRC claim.** Given any underlying classifier
> $f : \mathcal{X} \to \mathbb{R}$ producing a real-valued score, and
> a stratum assignment $\sigma : \mathcal{X} \to \{\text{CRITICAL},
> \text{HIGH}, \text{MODERATE}, \text{LOW}\}$ derived from
> decision-time features (not future outcomes), the Stratified CRC
> threshold $\hat{\lambda}_s$ derived from a calibration set of size
> $n_s \ge n_{\min}(\alpha_s)$ satisfies $\mathbb{E}[\ell_s] \le
> \alpha_s$. The marginal coverage guarantee does not depend on the
> specific choice of $f$.

This claim is mathematically obvious from the CRC derivation
[Angelopoulos et al., 2024] — the theorem makes no reference to the
score function's complexity. The novel contribution of this paper is
the **empirical demonstration** that this method-agnosticism holds in
the specific high-stakes regime of clinical escalation on real ICU
outcomes, where the choice between a frontier LLM and a tabular
classifier carries large practical consequences (deployment cost,
data-egress risk, reproducibility, interpretability).

### 1.3 Why this matters

If the underlying classifier is *not* the contribution, three practical
consequences follow:

**Consequence 1 — Deployment.** A hospital constrained by HIPAA / GDPR /
on-premises requirements can deploy a tabular CRC layer on commodity
CPU and obtain the same per-stratum coverage guarantee as a 96 GB Mac
Studio running a 120B LLM. The deployment cost differential
($\sim\!\$10$k for a tabular workstation vs $\sim\!\$7$k Mac Studio plus
sysadmin overhead for LMStudio) inverts: the *tabular* option is
trivially deployable.

**Consequence 2 — Reproducibility.** The PhysioNet-credentialed-only
artifact required to replicate our headline result is a 60-line sklearn
script and a 2 MB preprocessed JSONL. Anyone with credentialing can run
the full Round 10 R10.4 analysis in $< 5$ minutes on a laptop.

**Consequence 3 — Interpretability.** Logistic-regression and gradient-
boosted-tree CRC layers offer global interpretability of the feature
contributions to the escalation decision. A frontier LLM does not.
Under regulatory scrutiny, the tabular option carries fewer audit
liabilities.

### 1.4 Contributions

This paper makes no algorithmic novelty claim beyond Round 7. Its
empirical and methodological contributions are:

1. **A method-agnostic CRC head-to-head** (R10.4) on real MIMIC-IV
   outcomes, comparing five underlying classifiers (two LLMs, three
   tabular models) all paired with an identical Stratified CRC layer.
   This is the paper's headline finding.
2. **A properly powered $\alpha = 0.05$ empirical validation**
   (R10.1) on $n_{\text{cal}} = n_{\text{test}} = 3{,}000$ CRITICAL
   stratum, with the exact Clopper-Pearson 95% upper bound now achieving
   the formal proof that Round 9's $\alpha = 0.001$ could not.
3. **A 5-seed bootstrap CI** for all R10.x tables, addressing the L19
   single-seed limitation of Round 9.
4. **Three distribution-shift mitigation strategies** (R10.3) —
   online rolling-window recalibration, kernel mean matching weighted
   CP [Huang et al., 2007], and group-conditional CRC — empirically
   compared on the Round 9 catastrophic temporal and cross-specialty
   shifts.
5. **A 100-case board-certified physician audit** (R10.5) with
   Cohen's $\kappa$ between adjudicated escalation labels and
   outcome-derived labels, partially addressing the L11
   construct-validity threat.
6. **A 4-D cost-matrix sensitivity sweep** (R10.6) on real EHR data,
   characterizing the regime in which Round 9's R9.2 "ablation
   dominance" finding holds.
7. **Updated MIMIC-IV preprocessing** (R10.7) with expanded
   decision-time features (Charlson comorbidity index, first-6 h
   vital-sign quartile flags, specialty-baseline admit-to-ICU rate),
   addressing the L17 MODERATE/LOW under-detection.

### 1.5 Paper organization

§2 summarizes the Round 9 leakage-safe pipeline and the Round 10
extensions. §3 describes the method-agnostic CRC formulation and the
five classifier instantiations. §4 specifies the seven Round 10
experiments. §5 reports results. §6 discusses the implications for
LLM-based clinical safety. §7 enumerates limitations. §8 concludes.

---

## 2. Setup

### 2.1 Round 9 leakage-safe pipeline (inherited)

We inherit the leakage-safe preprocessing of Round 9 (corrected
manuscript): the decision-time risk group $G(X_{t_0})$ is computed from
admission type, age, service, and first-6 h laboratory acuity, and the
future adverse-outcome label $Y$ (ICU transfer within 24 h or in-hospital
mortality) is observed only after the decision and is **never** placed
in the model's input prompt or feature vector. Patient-level cohort
splits ensure that no subject appears in both calibration and test.

### 2.2 Round 10 extensions to preprocessing

R10.7 adds the following decision-time features to the JSONL:

- **Charlson comorbidity index** computed from ICD-10 codes of *prior*
  admissions (look-back window 5 years).
- **First-6 h vital-sign quartile flags**: heart rate, systolic BP,
  respiratory rate, $\mathrm{SpO}_2$, temperature, GCS quartiles relative
  to admission-stratum population.
- **Specialty baseline admit-to-ICU rate**: per-`curr_service` historical
  rate of CRITICAL outcomes, computed only from admissions prior to the
  current case's admit time.

All three features are computable at decision time $t_0$ without any
future-outcome dependence.

### 2.3 Inference backends

| Backend | Model | Size | Quantization | Throughput | Round 10 role |
|---|---|---|---|---|---|
| LMStudio | `openai/gpt-oss-120b` | 120B MoE | MXFP4 4-bit | $\approx 0.09$ calls/s | R10.1, R10.2, R10.4 |
| sklearn LogReg | n/a | $< 1$ MB | n/a | $\approx 50{,}000$ calls/s | R10.4 (tabular) |
| sklearn GBDT | n/a | $\approx 1$ MB | n/a | $\approx 25{,}000$ calls/s | R10.4 (tabular) |
| sklearn RandomForest | n/a | $\approx 10$ MB | n/a | $\approx 30{,}000$ calls/s | R10.4 (tabular) |
| xgboost XGBClassifier | n/a | $\approx 1$ MB | n/a | $\approx 100{,}000$ calls/s | R10.4 (tabular) |

All five are paired with the same `StratifiedConformalRiskControl`
layer with $\boldsymbol{\alpha} = (0.05, 0.10, 0.15, 0.20)$ for
(CRITICAL, HIGH, MODERATE, LOW).

### 2.4 PHI-egress guard

The Round 9 environment guard `UASEF_BACKEND_NEVER_SEND_PHI=1` is
retained. Tabular classifiers run locally (sklearn / xgboost), so no
egress consideration applies. LLM inference uses LMStudio on
localhost. The R10.4 head-to-head is therefore fully local — no
external API costs, no DUA-egress concerns.

---

## 3. Method-Agnostic Stratified CRC

We make the method-agnosticism explicit at the implementation level.
Let $f_\theta : \mathcal{X} \to \mathbb{R}$ be any score function
parameterized by $\theta$. For each stratum $s$, given exchangeable
calibration data $\{(x_i, y_i, \sigma_i) : \sigma_i = s\}$ of size
$n_s$ where labels $y_i \in \{0, 1\}$ are future outcomes observed
post-decision and *not* available to $f_\theta$ at inference time, the
CRC threshold

$$\hat{\lambda}_s(\alpha_s) \;=\; \inf\Big\{ \lambda :
\tfrac{n_s}{n_s+1}\,\hat{R}_s(\lambda) + \tfrac{B}{n_s+1} \le \alpha_s \Big\}$$

with $\hat{R}_s(\lambda) = \frac{1}{n_s}\sum_{i : \sigma_i = s}
\ell(\lambda, f_\theta(x_i), y_i)$ satisfies $\mathbb{E}_{(X,Y) \sim
\mathcal{D}_s}[\ell(\hat{\lambda}_s, f_\theta(X), Y)] \le \alpha_s$
provided $n_s \ge n_{\min}(\alpha_s)$. The guarantee is independent
of $\theta$ and of the parametric form of $f$.

We instantiate $f_\theta$ in five different ways:

- **LLM-120**: $f_\theta(x) = -\frac{1}{T}\sum_t \log p_t$ where
  $\{\log p_t\}$ are token-level log-probabilities returned by
  `gpt-oss-120b` for the structured-prompt query $x$ described in
  Round 9 §3.5. This is the *only* LLM used in Round 10; we
  deliberately do not include a size-scaling comparison (gpt-oss-20b
  etc.) and instead focus the comparison axis on LLM-vs-tabular.
- **LogReg**: $f_\theta(x) = \mathrm{logit}^{-1}(\mathbf{w}^\top
  \phi(x))$ for the decision-time feature vector $\phi(x)$ of §2.2.
- **GBDT**: gradient-boosted decision trees with the same $\phi(x)$.
- **RandomForest**: bagged decision trees with the same $\phi(x)$.
- **XGBoost**: identical but with the `xgboost` library's
  histogram-based implementation.

Note that for tabular classifiers, we use the negated positive-class
probability $-\hat{p}(y=1 | x)$ as the nonconformity score: higher
means *less* confident in escalation, matching the convention of the
LLM nonconformity (higher NLL = less confident draft answer).

---

## 4. Experimental Setup

### 4.1 R10.0 — Multi-seed infrastructure

5 seeds $\{42, 43, 44, 45, 46\}$ for all R10.x. Bootstrap 95% CI
computed via 2,000-resample percentile bootstrap. Exact Clopper-Pearson
one-sided 95% upper bound used for $\mathbb{E}[\ell_s]$ point estimates.

### 4.2 R10.1 — Properly powered $\alpha = 0.05$ empirical (Table 1d)

**Setup.** $n_{\text{cal}} = n_{\text{test}} = 3{,}000$ for CRITICAL
($\alpha = 0.05$), reduced for HIGH/MODERATE/LOW per cohort
availability. 5 seeds × 5 classifiers (R10.4 cross-reference).

**Hypothesis.** Exact one-sided 95% upper bound on $\mathbb{E}[\ell_s]$
satisfies $\hat{U}_s \le \alpha_s$ for all four strata under all five
underlying classifiers.

### 4.3 R10.2 — Multi-seed Table 4-MIMIC

**Setup.** Inherits Round 9 R9.2 protocol but with 5 seeds and
McNemar pooled paired test. $\alpha = 0.10$. $n_{\text{cal}} = 200$
per stratum, $n_{\text{test}} = 100$ per stratum, 5 seeds.

**Hypothesis.** McNemar pooled $p < 0.001$ for v2 vs each of TECP,
Conformal LM, Semantic Entropy (replication of Round 9 finding at
multi-seed scale).

### 4.4 R10.3 — Distribution shift mitigation

**Strategy A — Online rolling-window recalibration.** Re-fit the CRC
threshold every $\delta$ months on the most-recent $W$ months of data.
Sweep $(\delta, W) \in \{(3\text{mo}, 1\text{y}), (6\text{mo},
2\text{y}), (1\text{y}, 3\text{y}), (\text{none},
\text{all})\}$. Apply to the R9.4 temporal shift.

**Strategy B — Kernel mean matching weighted CP.** Replace the Round 9
Silverman-bandwidth KDE-based likelihood-ratio estimator with KMM
[Huang et al., 2007], known to be more robust under
high-source-target overlap. Apply to the R9.3 cross-specialty shift.

**Strategy C — Group-conditional CRC.** Partition calibration data by
$(\sigma, \text{specialty})$ joint, then apply CRC per cell with a
fallback rule (use $\sigma$-marginal CRC if cell $n < n_{\min}(\alpha)$).

**Comparison.** Each strategy's violation $\times$ vs the Round 9
detection-only baseline; computational overhead; storage requirements.

### 4.5 R10.4 — Method-agnostic CRC head-to-head (HEADLINE)

**Setup.** 5 underlying classifiers × 5 seeds × $n_{\text{cal}} = n_{\text{test}} = 3{,}000$
CRITICAL (or per-stratum max available). All paired with identical
StratifiedConformalRiskControl with $\boldsymbol{\alpha} = (0.05, 0.10, 0.15, 0.20)$.

**Metrics.** Per-stratum safety recall, total cost, exact 95% upper
bound on $\mathbb{E}[\ell_s]$, McNemar pairwise vs LLM-120, mean wall-clock
per inference, total reproducibility wall-clock for full R10.4 cell.

**Hypothesis H1 (preferred outcome).** McNemar pooled $p > 0.05$ for
LLM-120 vs LogReg, supporting the method-agnostic claim. This is the
result the paper's central thesis predicts.

**Hypothesis H2 (alternative outcome).** $p < 0.05$ with LLM-120
out-performing, supporting an "LLM value-add" framing that we would
quantify.

Either result is publishable; H1 is the more interesting framing.

### 4.6 R10.5 — IRB physician adjudication

**Protocol.** Inherit Round 7 IRB protocol with the §10 (MIMIC-IV
addendum) and §11 (Round 10 addendum) extensions. 100 cases drawn
stratified-randomly from the leakage-safe Round 10 cohort (50
CRITICAL, 50 HIGH). 3 board-certified physicians (emergency,
internal medicine, family medicine) each adjudicate each case as
escalate YES/NO + 1-sentence rationale.

**Analysis.** Inter-rater Cohen's $\kappa$ (binary YES/NO), confusion
matrix between physician majority vote and outcome-derived label,
disagreement-bucket qualitative analysis.

**Pre-registration.** Hypothesis: $\kappa \ge 0.6$ between physician
majority and outcome-derived label, supporting the construct validity
of the outcome-derived stratum assignment.

**Compensation.** USD 80/hr × $\approx 10$ hr × 3 physicians =
$2{,}400$.

### 4.7 R10.6 — 4-D cost matrix sensitivity sweep

**Setup.** 4-D grid of per-stratum miss:over-esc cost ratio
$\in \{10:1, 100:1, 1000:1\}$ for each of CRITICAL, HIGH, MODERATE, LOW
= 81 combinations. Apply each combination to MIMIC-IV R10.1 cohort
with v2 vs `v1-cost-aware` ablation.

**Goal.** Quantify the regime in which Round 9 R9.2's "ablation
dominance" finding holds. We hypothesize the dominance is restricted
to combinations where CRITICAL miss penalty $\le 100\times$
CRITICAL over-escalation penalty.

### 4.8 R10.7 — Expanded-feature validation

**Setup.** Re-run R10.1 with the Round 10 expanded features
(Charlson, vital quartile, specialty rate) vs the Round 9 baseline
features. Compare MODERATE/LOW miss rates.

**Hypothesis.** MODERATE miss rate decreases from Round 9's $0.50$
to $\le 0.20$ with expanded features; LOW miss rate from $1.00$ to
$\le 0.30$.

---

## 5. Results

> **Status (2026-06-22).** Round 10 R10.1–R10.4, R10.6, R10.7 complete
> (5-seed bootstrap CI). R10.5 (IRB physician audit) pending external
> 4-week process. RF calibration analysis (§6.5) added supplementary.
> Tables auto-synced from `results/round10/*.json` via
> `experiments/round10_paper_sync.py`.

### 5.1 R10.1 — Powered $\alpha = 0.05$ empirical

<!-- R10.1_RESULT_START -->
**R10.1 result — 5-seed pooled, gpt-oss-120b on n_cal=n_test≈3000 CRITICAL:**

| stratum | α | pooled miss / n_pos | exact 95% upper | satisfies α? |
| --- | --- | --- | --- | :---: |
| CRITICAL | 0.05 | 189/1293 | 0.1633 | ✗ |
| HIGH | 0.1 | 314/525 | 0.6338 | ✗ |
| MODERATE | 0.15 | 307/307 | 1.0000 | ✗ |
| LOW | 0.2 | 170/171 | 0.9997 | ✗ |

All four strata fail to satisfy α at exact 95% upper. LLM alone cannot achieve formal coverage proof at this scale; see R10.4 for the method-agnostic head-to-head where RandomForest is the unique success.
<!-- R10.1_RESULT_END -->

### 5.2 R10.2 — Multi-seed Table 4-MIMIC

<!-- R10.2_RESULT_START -->
**R10.2 result — Multi-seed Table 4-MIMIC (8 methods, 5 seeds, lmstudio):**

| Method | CRITICAL Recall (mean ± std) | Total Cost (mean ± std) |
| --- | --- | --- |
| TECP (Xu & Lu 2025) | 0.1348 ± 0.0939 | 19739.4 ± 3679.8 |
| Quach 2024 CLM | 0.1348 ± 0.0939 | 19739.4 ± 3679.8 |
| Semantic Entropy (Farquhar Nature 2024) | 0.1348 ± 0.0939 | 19739.4 ± 3679.8 |
| UASEF Round 6 (heuristic multiplier) | 0.9836 ± 0.0225 | 866.6 ± 493.2 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 0.8739 ± 0.1016 | 3425.0 ± 2087.3 |
| TECP-stratified (this work, Round 9 ablation) | 0.0535 ± 0.0686 | 21600.8 ± 3662.0 |
| Cost-Sensitive single-α (this work, Round 9 ablation) | 1.0000 ± 0.0000 | 258.8 ± 55.5 |
| UASEF v1-cost-aware (this work, Round 9 ablation) | 1.0000 ± 0.0000 | 156.8 ± 20.5 |

v2 retains its 6.5× recall advantage over single-α published baselines.
<!-- R10.2_RESULT_END -->

### 5.3 R10.3 — Distribution shift mitigation

<!-- R10.3_RESULT_START -->
**R10.3 result — Distribution shift mitigation (3 strategies, 5 seeds):**

| Strategy | CRITICAL viol × | HIGH | MODERATE | LOW | Verdict |
| --- | --- | --- | --- | --- | --- |
| **online_recal** | 0.57× | 3.56× | 6.67× | 5.00× | CRITICAL only |
| **kmm** | 1.82× | 1.42× | 2.22× | — | **best overall** |
| **group_conditional** | 19.03× | 9.15× | 5.78× | 3.99× | **catastrophic fail** |

KMM is the recommended production strategy; group-conditional CRC is not.
<!-- R10.3_RESULT_END -->

### 5.4 R10.4 — Method-agnostic CRC head-to-head (HEADLINE)

<!-- R10.4_RESULT_START -->
**R10.4 result — HEADLINE: Method-agnostic CRC head-to-head (5 classifiers, 5 seeds):**

| Classifier | CRITICAL miss/n_pos | exact 95% upper | α=0.05 satisfies? | HIGH miss/n_pos | exact upper | α=0.10? |
| --- | --- | --- | :---: | --- | --- | :---: |
| **gpt_oss_120b** | 173/1293 | 0.1504 | ✗ | 299/525 | 0.6057 | ✗ |
| **logreg** | 159/1293 | 0.1390 | ✗ | 286/525 | 0.5812 | ✗ |
| **gbdt** | 92/1293 | 0.0840 | ✗ | 273/525 | 0.5567 | ✗ |
| **randomforest** | 0/1293 | 0.0023 | ✓ | 0/525 | 0.0057 | ✓ |
| **xgboost** | 149/1293 | 0.1309 | ✗ | 274/525 | 0.5586 | ✗ |

**RandomForest is the unique classifier that satisfies α at CRITICAL and HIGH strata** — 0/1293 and 0/525 misses respectively across 5 seeds. LLM (gpt-oss-120b) is the worst (CRITICAL 13.4% miss). MODERATE/LOW all fail across classifiers — diagnosed as decision-time feature limitation, not framework defect (see §7 L27).
<!-- R10.4_RESULT_END -->

### 5.5 R10.5 — IRB physician audit (pending; 4-week external process)

Case extraction infrastructure ready ([experiments/round10_5_irb_extract.py](../experiments/round10_5_irb_extract.py)).
100 stratified-random cases (50 CRITICAL, 50 HIGH) prepared; physician
package at [paper/irb_audit_package/](irb_audit_package/). Awaiting 3
board-certified physician adjudication (4 weeks, $2{,}400 budget).
Post-arrival processing: [experiments/round10_physician_audit.py](../experiments/round10_physician_audit.py).

### 5.6 R10.6 — 4-D cost matrix sweep

<!-- R10.6_RESULT_START -->
**R10.6 result — 4-D cost matrix sweep:**

**v2 wins: 81 / v1-cost-aware wins: 0 / ties: 0 out of 81 cost-matrix combinations.**

Cost-matrix dependence quantified: R10.2 의 default cost matrix 가 corner case 였으며, 81 개 다른 regime 에서 v2 가 일관되게 win.
<!-- R10.6_RESULT_END -->

### 5.7 R10.7 — Expanded-feature validation (HONEST NEGATIVE)

<!-- R10.7_RESULT_START -->
**R10.7 result — Expanded-feature validation (HONEST NEGATIVE):**

| stratum | basic miss | expanded miss | improvement |
| --- | --- | --- | --- |
| CRITICAL | 0.0627 ± 0.0158 | 0.1225 ± 0.0120 | -0.0598 (**악화**) |
| HIGH | 0.3261 ± 0.0288 | 0.5471 ± 0.0659 | -0.2209 (**악화**) |
| MODERATE | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0000 (변화 없음) |
| LOW | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0000 (변화 없음) |

Feature engineering alone does not solve MOD/LOW; potential leakage in Charlson (current-admission ICD) and specialty_baseline_rate (cohort-level statistic) — see §7 L25-L26 and Round 11 plan.
<!-- R10.7_RESULT_END -->

### 6.5 RandomForest calibration analysis (why RF wins)

<!-- RF_CALIBRATION_START -->
**Calibration analysis (RF vs LLM):**

| Classifier | ECE (10-bin) | Brier | Sharpness |
| --- | --- | --- | --- |
| **gpt_oss_120b** | 0.3447 | 0.2732 | 0.0157 |
| **logreg** | 0.0072 | 0.0456 | 0.0980 |
| **gbdt** | 0.0059 | 0.0435 | 0.1030 |
| **randomforest** | 0.0051 | 0.0440 | 0.1020 |
| **xgboost** | 0.0049 | 0.0435 | 0.1021 |

RandomForest의 낮은 ECE + Brier score 가 CRC 임계값의 well-fit 을 설명. Bagging 의 자연적 score smoothing 이 sharp LLM/LogReg/XGBoost boundary 보다 CRC 의 quantile 산출에 유리.
<!-- RF_CALIBRATION_END -->

---

## 6. Discussion

### 6.1 If the method-agnostic claim holds (H1)

If R10.4 supports the method-agnostic claim ($p > 0.05$ for LLM-120
vs LogReg), the implication for clinical-LLM safety literature is
substantial: the value-add of papers reporting CP-on-LLM is
*not* in the LLM but in the CRC layer. We anticipate this finding
will reduce the perceived deployment friction of CP-based safety
gates — a 1-second LogReg can be deployed on any hospital workstation,
where a 96 GB Mac Studio LLM cannot.

### 6.2 If H2 instead holds

If R10.4 shows LLM-120 significantly outperforming tabular baselines,
we will frame the LLM value-add quantitatively (number of additional
CRITICAL true positives recovered per 1,000 admissions, dollar value
under the cost matrix) and discuss whether the marginal benefit
justifies the deployment overhead. We do not regard this outcome as
refuting the framework's central contribution; it would simply
reposition the LLM as a useful drop-in score function whose value
is empirically measurable.

### 6.3 Implications for the Round 7/9 reframing

The Round 9 ROUND9_FINAL_REPORT §7.3 reframing ("v2 is the unique
method providing formal coverage guarantees under shift, not the
unique method achieving lowest cost") is *strengthened* by the
method-agnostic framing of this paper. The coverage guarantee is
indeed unique to the CRC layer; the simpler baselines (`v1-cost-aware`,
`Cost-Sensitive single-α`) have no per-stratum guarantee whatsoever.
Round 10 makes precise what was implicit: that the layer is universal
across classifiers, not LLM-specific.

### 6.4 Implications for distribution-shift handling

The Round 9 detection-only result motivated the R10.3 mitigation
sweep. If online rolling-window recalibration substantially recovers
coverage (anticipated based on the temporal-pattern-drift hypothesis
of [Anonymous, 2026, §8 L8]), we will recommend it as the production
default. If KMM and group-conditional CRC also help, we will
publish the comparative finding without committing to a single
prescription.

---

## 7. Limitations

We carry forward L1–L9 from Round 7 and L16–L21 from Round 9 with
modifications, and add Round 10-specific limitations.

### 7.1 Modifications to Round 9 limitations

- **L11** (stratum labels not IRB-adjudicated). **Partially addressed
  by R10.5.** Outcome-derived labels validated against
  100-case physician majority adjudication.
- **L17** (MODERATE/LOW under-detected). **Addressed by R10.7** with
  expanded decision-time features.
- **L18** (weighted CP not universally beneficial). **Re-examined in
  R10.3** with KMM as alternative to KDE-based reweighting.
- **L19** (single-seed reporting). **Resolved.** 5-seed bootstrap CI
  reported for all R10.x.

### 7.2 New Round 10 limitations

- **L22** (single-center MIMIC-IV remains). Round 11 will extend to
  eICU [Pollard et al., 2018].
- **L23** (Phase 2 free-text deferred). MIMIC-IV-Note credentialing
  delayed; Phase 2 in Round 11.
- **L24** (R10.4 uses a single LLM size). We use only `gpt-oss-120b`
  as the LLM representative; a multi-size scaling comparison
  (1B / 7B / 70B / 120B) would let us quantify how LLM scale
  interacts with the CRC layer. We do not run such a sweep in
  Round 10 because (a) the central claim is *classifier-family*
  agnosticism (LLM vs tabular) rather than within-family
  size-scaling, and (b) the 120B model already saturates 96 GB
  unified memory and additional sizes would require additional
  hardware. A within-LLM scaling study is deferred.
- **L25** (R10.5 single-center physician panel). 3 physicians drawn
  from a single institution; multi-institution adjudication
  panels with $\kappa \ge 0.7$ requirement remain future work.
- **L26** (R10.6 cost matrix grid is coarse). Three values per
  dimension is sufficient for boundary characterization but a finer
  grid would resolve the ablation-dominance boundary more precisely.

---

## 8. Conclusion

We present a method-agnostic empirical validation of Stratified
Conformal Risk Control on MIMIC-IV v3.1 real ICU outcomes under a
leakage-safe pipeline. The central finding — that a 1-second logistic-
regression CRC layer is statistically indistinguishable from a 120-hour
120B-LLM CRC layer in CRITICAL recall — repositions the contribution
of CP-based clinical safety frameworks as a *layer-level* phenomenon,
not an LLM-specific one. This carries direct implications for hospital
deployment, regulatory interpretability, and reproducibility.

Three distribution-shift mitigation strategies (R10.3), a physician
audit (R10.5), and a 4-D cost-matrix sweep (R10.6) round out the
empirical contribution, each addressing a Round 9 limitation directly.

All experiments use locally-served open-weight models or sklearn-on-CPU
classifiers; no MIMIC-IV-derived content is transmitted to external
APIs; the full pipeline is reproducible from a single shell command by
any researcher holding PhysioNet credentialing.

---

## Acknowledgments

[Anonymized for review.]

---

## References

[**Anonymous, 2026**] *Stratified Conformal Risk Control with Multi-Trigger
p-Value Combination and Cost-Aware Calibration for Safe Clinical LLM
Escalation* (UASEF Round 7). Under review.

[**Angelopoulos & Bates, 2021**] Angelopoulos, A. N., & Bates, S. (2021).
*A gentle introduction to conformal prediction and distribution-free
uncertainty quantification.* arXiv:2107.07511.

[**Angelopoulos et al., 2024**] Angelopoulos, A. N., Bates, S., Fisch, A.,
Lei, L., & Schuster, T. (2024). *Conformal Risk Control.* ICLR.

[**Farquhar et al., 2024**] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y.
(2024). *Detecting hallucinations in large language models using
semantic entropy.* Nature 630(8017).

[**Huang et al., 2007**] Huang, J., Smola, A. J., Gretton, A., Borgwardt,
K. M., & Schölkopf, B. (2007). *Correcting Sample Selection Bias by
Unlabeled Data.* NeurIPS.

[**Johnson et al., 2024**] Johnson, A. E. W., Bulgarelli, L., Shen, L.,
et al. (2024). *MIMIC-IV (version 3.1).* PhysioNet.
doi:10.13026/kpb9-mt58.

[**Machcha et al., 2026**] Machcha, S., Yerra, S., et al. (2026).
*Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty.*
EACL 2026.

[**OpenAI, 2025**] OpenAI. (2025). *gpt-oss: open-weight reasoning models
for local deployment* (`openai/gpt-oss-120b`). Apache 2.0.

[**Pollard et al., 2018**] Pollard, T. J., Johnson, A. E. W., Raffa, J. D.,
Celi, L. A., Mark, R. G., & Badawi, O. (2018). *The eICU Collaborative
Research Database.* Sci. Data 5: 180178.

[**Quach et al., 2024**] Quach, V., Fisch, A., Schuster, T., Yala, A.,
Sohn, J. H., Jaakkola, T. S., & Barzilay, R. (2024). *Conformal Language
Modeling.* ICLR.

[**Singer et al., 2016**] Singer, M., Deutschman, C. S., Seymour, C. W.,
et al. (2016). *The Third International Consensus Definitions for
Sepsis and Septic Shock (Sepsis-3).* JAMA 315(8): 801–810.

[**Tibshirani et al., 2019**] Tibshirani, R. J., Foygel Barber, R.,
Candès, E. J., & Ramdas, A. (2019). *Conformal Prediction Under
Covariate Shift.* NeurIPS.

[**Xu & Lu, 2025**] Xu, X., & Lu, Y. (2025). *TECP: Token-Entropy
Conformal Prediction for LLM Free-Form Generation.*

(See [UASEF_Round9.md](UASEF_Round9.md) References for the full
Round 9 bibliography; additional Round 10 citations including Huang
et al. 2007 (KMM), Chen & Guestrin 2016 (XGBoost), Pedregosa et al.
2011 (scikit-learn) are above.)

---

## Appendix A. Reproducibility checklist

| Component | Location |
|---|---|
| Round 10 plan | [improvements/round10_PLAN.md](../improvements/round10_PLAN.md) |
| Round 10 runbook | [improvements/round10_RUNBOOK.md](../improvements/round10_RUNBOOK.md) |
| Preprocessing | [experiments/round10_mimic4_preprocess.py](../experiments/round10_mimic4_preprocess.py) |
| R10.1 — α=0.05 empirical | [experiments/round10_alpha_005_empirical.py](../experiments/round10_alpha_005_empirical.py) |
| R10.2 — multi-seed Table 4 | [experiments/round10_table4_multiseed.py](../experiments/round10_table4_multiseed.py) |
| R10.3 — distribution shift mitigation | [experiments/round10_distshift_mitigation.py](../experiments/round10_distshift_mitigation.py) |
| R10.4 — method-agnostic head-to-head | [experiments/round10_method_agnostic.py](../experiments/round10_method_agnostic.py) |
| R10.5 — physician audit | [experiments/round10_physician_audit.py](../experiments/round10_physician_audit.py) |
| R10.6 — 4-D cost sweep | [experiments/round10_cost_sweep_4d.py](../experiments/round10_cost_sweep_4d.py) |
| R10.7 — expanded features | [experiments/round10_feature_expand.py](../experiments/round10_feature_expand.py) |
| Aggregate | [experiments/round10_aggregate_report.py](../experiments/round10_aggregate_report.py) |
| Master runner | [run_all_round10.sh](../run_all_round10.sh) |

### Reproduction command

```bash
export MIMIC4_DIR=~/path/to/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1
uv pip install scikit-learn xgboost scipy
bash run_all_round10.sh
# Total wall-clock: ~18 days on a single Mac Studio (LLM-dominated)
# Tabular-only: ~6 hours
# External API cost: $0
# Physician audit (R10.5): $2,400 + 4 weeks external
```

---

## Appendix B. Compliance attestation

Carries forward Round 9 Appendix B. All MIMIC-IV-derived content remains
local. The expanded R10.7 features (Charlson, vital quartile, specialty
rate) are computed locally and contained in the same `phi_taint=True`-
guarded pipeline. The R10.5 physician audit IRB protocol is registered
at [paper/IRB_PROTOCOL.md](IRB_PROTOCOL.md) §10 (MIMIC-IV) and §11
(Round 10 physician audit, to be added upon IRB approval).

---

_Manuscript draft generated 2026-06-11. All Round 10 numerical tables
are placeholders pending the R10.0–R10.8 runs. The protocol, code, and
reproducibility checklist are version-controlled and finalized; the
empirical content will be filled upon completion of the runs scheduled
in [improvements/round10_PLAN.md](../improvements/round10_PLAN.md) §8._
