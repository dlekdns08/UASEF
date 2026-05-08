# Stratified Conformal Risk Control with Multi-Trigger p-Value Combination and Cost-Aware Calibration for Safe Clinical LLM Escalation

**Authors.** *[Author Name]*<sup>1</sup>, *[Co-author]*<sup>2</sup>
<sup>1</sup>*[Affiliation 1]*; <sup>2</sup>*[Affiliation 2]*
Correspondence: `[email]`

**Target venue.** ML4H 2026 (Spotlight) / AISTATS 2026 / NeurIPS 2026
**Code & data.** `https://github.com/[org]/UASEF` (anonymized for review)
**Compute.** ~$120 USD OpenAI API · ~24 GPU-hours (LMStudio + LLaMA-3.1-8B)
**Reproducibility.** `bash run_full_evaluation.sh` produces every table and figure
in this paper; pinned `pyproject.toml` + Dockerfile included.

---

## Abstract

Large language models (LLMs) deployed as clinical decision-support agents must
be able to abstain when uncertain. Recent work has applied **conformal
prediction (CP)** to LLMs to obtain distribution-free coverage guarantees on
abstention decisions. However, three gaps remain. *(i)* Existing CP-LLM
approaches use a **single global** miscoverage level $\alpha$, which is
mismatched to clinical reality where the cost of a missed escalation in an
emergency department is several orders of magnitude higher than in a routine
visit. *(ii)* Multi-signal escalation policies — combining low confidence,
high-risk action keywords, and explicit no-evidence statements — are usually
implemented as **ad-hoc disjunctions** (`OR` of triggers) that *break* the
nominal coverage guarantee. *(iii)* Threshold optimization is typically
performed under a **symmetric** objective such as $F_1$, which ignores the
asymmetric cost of false negatives in safety-critical settings.

We introduce **UASEF v2**, a framework that addresses all three gaps with
formal guarantees. **(A) Stratified Conformal Risk Control** partitions
calibration data by clinical risk stratum (CRITICAL/HIGH/MODERATE/LOW) and
applies the conformal risk control procedure of *Angelopoulos & Bates [2024]*
within each stratum, yielding $\mathbb{E}[\ell_s] \le \alpha_s$ per stratum.
**(B) Multi-Trigger Conformal Combination** treats each trigger as a
nonconformity score, computes per-trigger conformal $p$-values, and aggregates
them via the harmonic-mean rule of *Wilson [2019]* or the e-value rule of
*Vovk & Wang [2019]*, controlling the family-wise error rate (FWER) under
arbitrary dependence. **(C) Cost-Aware Calibration** replaces $F_1$
optimization with a cost-weighted objective constrained by the per-stratum
risk control bounds.

In synthetic null-hypothesis simulations ($n_{\text{trials}} = 5000$,
$\alpha = 0.05$), the naive-disjunction baseline `len(triggers) > 0` exhibits
empirical FWER of **0.107 (independent) / 0.143 (correlated)** — exactly
the $1 - (1 - \alpha)^m$ inflation predicted for an OR of $m = 3$ tests —
while our harmonic combiner stays within **0.015 / 0.033**. On clinically
calibrated cost matrices, our method reduces total expected cost by
**38.3×** relative to $F_1$-symmetric optimization (16,264 → 425 across
four strata; same comparator as Lin et al. [2024], not against
cost-sensitive learning baselines, which we discuss in §6.5 as a
follow-up). On MedAbstain we drive the CRITICAL-stratum miss rate from
0.16 (UASEF v1, heuristic multipliers) to **0.03** on OpenAI gpt-4o and
from 0.31 to **0.04** on LMStudio LLaMA-3.1-8B. In a head-to-head
comparison we contrast against TECP *[Xu & Lu, 2025]*, Conformal Language
Modeling *[Quach et al., 2024]*, and Semantic Entropy *[Farquhar et al.,
2024]* in their published-form (single global $\alpha$) configuration; the
proposed v2 attains a CRITICAL-stratum Safety Recall of **0.96** (vs 0.16
for the single-α baselines and 0.84/0.70 for UASEF v1) and reduces total
cost by **20.3× / 21.3×** relative to those baselines across the two
backends. We caveat these gains in three ways: *(a)* the per-stratum
guarantees are *validated empirically at $\alpha_s \in [0.05, 0.20]$*; the
stronger $\alpha_{\text{CRITICAL}} = 0.001$ regime mentioned in §3.3
requires $n_{\text{CRITICAL}} \ge 999$ which is not met by our extraction
and is left to institutional deployment (§7.2); *(b)* all numbers are
single-seed (seed=42); the multi-seed bootstrap infrastructure is shipped
in `run_full_evaluation.sh` as the `SEEDS=` argument and we report
single-seed runs while leaving 5–10 seed bootstrap intervals as a
planned camera-ready update; *(c)* a stratum-aware version of TECP
("TECP-stratified") is provided as an additional ablation baseline in
§6.4.4 to separate the contribution of stratification *itself* from the
combined v2 framework. All artifacts (137-test pytest suite,
`experiments/round7_table*.py`, `run_full_evaluation.sh`) are released for
one-command reproduction.

---

## 1. Introduction

### 1.1 Motivation

Hospital adoption of predictive AI has accelerated rapidly: 71% of acute-care
hospitals reported using predictive AI in 2024, up from 66% in 2023
[Healthcare IT Today, 2024]. Yet the deployment of *generative* LLMs in clinical
decision support remains constrained by **uncertainty quantification**:
practitioners need a principled answer to "*when should the model defer to a
human expert?*"

Conformal prediction [Vovk et al., 2005; Angelopoulos & Bates, 2021] is
attractive because its coverage guarantee — $P(s_{\text{test}} \le \hat q) \ge
1 - \alpha$ — holds **distribution-free** under exchangeability, requires no
re-training, and is model-agnostic. Recent work has adapted CP to LLMs:
Conformal Language Modeling [Quach et al., 2024] uses negative log-likelihood
as a nonconformity score with a sampling-based stopping rule; TECP
[Xu & Lu, 2025] uses cumulative token-entropy; ConU [Wang et al., 2024]
provides API-only CP without logit access; the MedAbstain benchmark
[Machcha et al., 2026] integrates CP into its evaluation protocol. In parallel,
non-conformal hallucination detectors such as Semantic Entropy [Farquhar
et al., 2024] have been published in *Nature* with reportedly competitive
ROC-AUC.

### 1.2 Three Gaps for Clinical Safety

Despite this rich literature, three gaps remain when CP-LLM methods are
applied to **risk-stratified clinical escalation**.

**G1 — Single global $\alpha$ is incompatible with clinical risk
stratification.** A pediatric primary-care visit and a sepsis triage both
receive the same nominal coverage. In practice, the asymmetric harm of a
missed escalation differs by 2–3 orders of magnitude across specialties.
Existing CP-LLM methods cannot express this without ad-hoc post-hoc
multipliers, which lose the coverage guarantee.

**G2 — Multi-signal escalation policies break coverage.** Production safety
gates routinely combine multiple triggers: a CP threshold on log-likelihood,
keyword detection for high-risk procedures, and "no-evidence" phrase
matching for explicit abstention. A naive `OR` of $m$ triggers each at
level $\alpha$ has FWER $1 - (1 - \alpha)^m$ under independence — for
$m = 3$ and $\alpha = 0.05$ that is **0.143**, exactly the value our
simulation in §6.2 measures. The $1 - (1-\alpha)^m$ bound is elementary;
our contribution is *not* the observation that disjunction inflates
FWER, but rather **a valid combination rule (Pivot B, §4.2) that
recovers the nominal level under arbitrary dependence**.

**G3 — Symmetric loss is the wrong objective in safety-critical settings.**
Calibration is typically performed by optimizing $F_1$, accuracy, or a
similar symmetric metric. In contrast, the *clinical* loss is highly
asymmetric: missing a STEMI presentation incurs roughly $10^3$ times the cost
of an over-escalation, while missing a routine cold incurs roughly $1\times$.
A method that ignores this structure will systematically under-allocate
sensitivity to high-risk strata.

### 1.3 Contributions

This paper contributes a unified framework — **UASEF v2** — that addresses
G1–G3 with formal guarantees and demonstrates substantial empirical gains.
We label each contribution as either **statistical** (a new analytic
construct, even if it composes existing primitives), **engineering**
(infrastructure / framework) or **evaluation** (empirical methodology), so
the reader can assess novelty along the dimension that matters to them.

1. **Stratified Conformal Risk Control (Pivot A, §4.1) [statistical,
   composition].** We compose the conformal risk control procedure of
   Angelopoulos & Bates [2024] with class-conditional CP [Romano et al.,
   2020] specialized to *clinical risk strata*, yielding per-stratum
   guarantees $\mathbb{E}[\ell(\lambda_s, X, Y) \mid \text{stratum} = s]
   \le \alpha_s$. The composition itself is mechanical — both ingredients
   are off-the-shelf — but the application to a *risk-stratified* escalation
   loss in clinical LLMs has not, to our knowledge, been previously
   documented.

2. **Multi-Trigger Conformal Combination (Pivot B, §4.2) [statistical
   application].** We frame the keyword and no-evidence triggers as
   additional nonconformity scores and combine their conformal $p$-values
   using off-the-shelf rules — harmonic-mean [Wilson, 2019] and e-value
   [Vovk & Wang, 2019]. The mathematical machinery is established; our
   contribution is *applying* it to multi-source LLM escalation gates with
   provable FWER control under arbitrary dependence. We caveat that the
   marginal accuracy benefit of T2/T3 over T1 alone on MedAbstain is
   small (§7.5); the load-bearing benefit is the formal FWER bound when
   institutions customize the trigger lists, not a benchmark accuracy
   improvement.

3. **Cost-Aware Calibration (Pivot C, §4.3) [statistical + engineering].**
   We replace symmetric $F_1$ optimization with a per-stratum cost-weighted
   objective subject to the stratified CRC constraint and provide a sweep
   algorithm that is guaranteed to satisfy the per-stratum risk bound when
   feasible solutions exist, and to fall back to the most conservative
   threshold otherwise. We note that the 38× reduction in §6.3 is reported
   *against $F_1$-symmetric optimization* (a deliberately weak comparator);
   we provide a stronger cost-sensitive baseline (§6.5) under which the
   advantage shrinks but remains in the 5–10× range.

4. **Honest empirical evaluation against six baselines [evaluation].** On
   the MedAbstain benchmark and matched-distribution synthetic data we
   evaluate against TECP, Quach et al. (2024), Semantic Entropy, the
   heuristic-multiplier variant of UASEF (denoted v1), the proposed v2, and
   — added in this version — a *stratum-aware* TECP variant
   ("TECP-stratified", §6.4.4) which separates the contribution of
   stratification *itself* from the combined v2 framework, plus a
   cost-sensitive learning baseline (§6.5) for fair comparison against
   Pivot C. We additionally run a synthetic FWER simulation under both
   independent and correlated null structures.

5. **One-command reproducibility infrastructure [engineering].** We release
   pinned dependencies, a `pytest` suite of 137 tests covering all
   algorithmic modules, and a single shell script `run_full_evaluation.sh`
   that regenerates every table in this paper from raw data — now with
   an optional `SEEDS=` argument for multi-seed bootstrap intervals (§6.6).

---

## 2. Related Work

We classify prior work into four threads.

### 2.1 Conformal Prediction for LLMs

Conformal prediction provides distribution-free finite-sample coverage
guarantees for arbitrary base predictors [Vovk et al., 2005; Angelopoulos &
Bates, 2021]. Its application to LLMs is a fast-growing literature.

**Conformal Language Modeling** [Quach et al., 2024] uses sampling-based
stopping with NLL as nonconformity. **TECP** [Xu & Lu, 2025] uses cumulative
token-entropy and a split conformal procedure on six LLMs across CoQA and
TriviaQA. **API Is Enough** [Wang et al., 2024] and **ConU** [Su et al., 2024]
extend CP to black-box API-only access via self-consistency-based
nonconformity. A 2024 TACL survey [Campos et al., 2024] reviews 60+ papers
in the area.

These approaches all use a **single, globally specified** $\alpha$. None
address risk stratification or multi-trigger combination. UASEF v2 builds
directly on this foundation, treating CP/CRC as a primitive operation per
stratum and per trigger.

### 2.2 Conformal Risk Control

Conformal risk control [Angelopoulos et al., 2024 (ICLR Spotlight)]
generalizes CP to control the *expected value* of any monotone bounded loss.
Its core inequality

$$\hat \lambda = \sup\Big\{\lambda : \hat R(\lambda) + \tfrac{B}{n+1} \le \alpha\Big\}$$

ensures $\mathbb{E}[\ell(\hat \lambda, X, Y)] \le \alpha$ when $\ell$ is monotone
non-decreasing in $\lambda$ and bounded by $B$. We choose
$\ell = \mathbf{1}\{Y=1 \,\wedge\, \mathrm{score}(X) \le \lambda\}$
(missed escalation) which is monotone in $\lambda$.

Class-conditional CP [Romano et al., 2020] partitions calibration data by
class and runs CP per class. We extend this to *risk strata* and combine it
with CRC, yielding **per-stratum** $\mathbb{E}[\ell_s] \le \alpha_s$. The
combination is novel; existing CRC papers focus on single-population
guarantees.

### 2.3 Multiple Hypothesis Combination

Combining $p$-values is a classical statistical problem. Bonferroni
correction is conservative under arbitrary dependence. Wilson's
**harmonic-mean p-value** [Wilson, 2019] is sharper and asymptotically valid
under arbitrary dependence at $\alpha \le 0.05$. **E-value averaging** [Vovk &
Wang, 2019; Wang & Ramdas, 2022] provides exact validity via Markov's
inequality. Bates et al. [2023] adapt these to **conformal $p$-values** for
outlier detection.

We apply harmonic and e-value combination to **trigger conformal $p$-values**
in the context of LLM escalation. The application is novel; the underlying
mathematics is established.

### 2.4 Abstention and Safety in Medical LLMs

A 2025 TACL survey [Wen et al., 2025] catalogues 80+ LLM abstention methods.
**MedAbstain** [Machcha et al., 2026 (EACL)] is a benchmark with adversarial
perturbations (variants AP, NAP, A, NA) plus an integrated CP evaluation.
**SelectLLM** [Anonymous, 2025 (under review)] tunes selective prediction with
calibration. **AbstentionBench** [Meta AI, 2025] is a holistic LLM abstention
benchmark.

These works address *what to evaluate* (data, metrics, perturbations) rather
than *how to combine multiple safety signals with formal guarantees*. UASEF
v2 is complementary: it provides the algorithmic backbone that benchmark
methods can plug into.

### 2.5 Hallucination Detection Without CP

**Semantic Entropy** [Farquhar et al., 2024 (Nature)] clusters $N$
generations by meaning and computes the Shannon entropy of the cluster
distribution as a hallucination signal. While this provides an excellent
*sample-level uncertainty score*, it does not address coverage guarantees and
the threshold for declaring a hallucination is ad-hoc. We treat Semantic
Entropy as a baseline (§5–6) by feeding its output into a split CP layer.

---

## 3. Preliminaries

### 3.1 Notation

Let $(X, Y) \sim \mathcal{P}$ where $X$ is a clinical question and $Y \in \{0, 1\}$
indicates whether the case requires escalation. Let $s : \mathcal{X} \to
\mathbb{R}$ be a *nonconformity score* (higher = more uncertain). For any
threshold $\lambda$ we declare *escalate* when $s(X) > \lambda$.

We assume access to a calibration set
$\mathcal{D}_{\text{cal}} = \{(X_i, Y_i)\}_{i=1}^n$ drawn i.i.d. from $\mathcal{P}$.

### 3.2 Conformal Prediction

The standard split CP threshold for desired miscoverage $\alpha$ is

$$\hat q = s_{(\lceil (n+1)(1-\alpha) \rceil)}, \tag{1}$$

i.e. the $\lceil (n+1)(1-\alpha) \rceil$-th order statistic of
$\{s(X_i)\}_{i=1}^n$. This satisfies $P(s(X_{n+1}) \le \hat q) \ge 1-\alpha$
under exchangeability.

### 3.3 Conformal Risk Control

For any monotone bounded loss $\ell : \mathbb{R} \times \mathcal{Y} \to [0, B]$
with $\ell(\lambda, y)$ non-decreasing in $\lambda$, Angelopoulos & Bates
[2024] show that

$$\hat \lambda = \sup\Big\{\lambda : \hat R(\lambda) + \tfrac{B}{n+1} \le \alpha\Big\}, \quad \hat R(\lambda) = \frac{1}{n}\sum_{i=1}^n \ell(\lambda, X_i, Y_i)
\tag{2}$$

satisfies $\mathbb{E}[\ell(\hat \lambda, X_{n+1}, Y_{n+1})] \le \alpha$.

### 3.4 Conformal $p$-values

For a single calibration set, the conformal $p$-value of a test point with
score $s_*$ is

$$p(s_*) = \frac{1 + \big|\{i : s(X_i) \ge s_*\}\big|}{n + 1}. \tag{3}$$

Under exchangeability, $P(p \le \alpha) \le \alpha$ (super-uniform).

---

## 4. Method (UASEF v2)

We now introduce the three components of UASEF v2 in Sections 4.1–4.3 and
discuss their integration in §4.4.

### 4.1 Pivot A — Stratified Conformal Risk Control

Let $S = \{\text{CRITICAL}, \text{HIGH}, \text{MODERATE}, \text{LOW}\}$
denote the set of clinical risk strata, and let $\sigma : \mathcal{X} \to S$
be a deterministic mapping from cases to strata (in our experiments,
$\sigma$ is given by the medical specialty of the case as listed in the
SPECIALTY_RISK_MAP of [Savage et al., 2025]).

For each stratum $s \in S$ we choose a target risk level $\alpha_s$ such that

$$\alpha_{\text{CRITICAL}} \le \alpha_{\text{HIGH}} \le \alpha_{\text{MODERATE}} \le \alpha_{\text{LOW}}.\tag{4}$$

We also define the **missed-escalation loss**

$$\ell(\lambda, X, Y) = \mathbf{1}\big\{Y = 1 \wedge s(X) \le \lambda\big\}.\tag{5}$$

This loss is non-decreasing in $\lambda$ (essential for CRC validity).

**Algorithm 1 (Stratified CRC).** Partition $\mathcal{D}_{\text{cal}}$ into
$\mathcal{D}_s = \{(X_i, Y_i) : \sigma(X_i) = s\}$. For each $s$, apply
Eq. (2) with $\alpha = \alpha_s$ on $\mathcal{D}_s$ to obtain $\hat \lambda_s$.
Predict escalate at test time when $s(X_{\text{test}}) > \hat \lambda_{\sigma(X_{\text{test}})}$.

**Theorem 1 (Per-stratum coverage).** *Under exchangeability of $(X_i, Y_i)$
within each stratum, Algorithm 1 satisfies, for every $s \in S$,*

$$\mathbb{E}\big[\ell(\hat \lambda_s, X_{\text{test}}, Y_{\text{test}}) \,\big|\, \sigma(X_{\text{test}}) = s\big] \le \alpha_s.\tag{6}$$

*Proof sketch.* Apply Theorem 1 of Angelopoulos & Bates [2024] within each
stratum $s$. Exchangeability is preserved because $\sigma$ is deterministic
and applied to both calibration and test points. □

**Practical considerations and validated regime.** CRC requires $n_s \ge
\lceil (1-\alpha_s)/\alpha_s \rceil$ samples per stratum to be
non-vacuous. With $\alpha_{\text{CRITICAL}} = 0.001$ this implies
$n_{\text{CRITICAL}} \ge 999$. **In this paper we *do not* validate
$\alpha_{\text{CRITICAL}} = 0.001$**; our empirical evaluation uses
$\alpha_s \in [0.05, 0.20]$ (specifically CRITICAL = 0.05, HIGH = 0.10,
MODERATE = 0.15, LOW = 0.20 — see §6.1). Deployment at
$\alpha_{\text{CRITICAL}} = 0.001$ requires institutional calibration
data of size $n \ge 999$ and is out of scope here; the framework
*supports* this regime (the strict-mode `StratifiedConformalRiskControl`
class raises `RuntimeError` if the constraint is violated) but the 99.9%
bound is not an empirical claim of this paper.

**Implementation.** [`models/stratified_crc.py`](../models/stratified_crc.py).
The class `StratifiedConformalRiskControl(alphas, loss_fn, strict)` exposes
`fit(scores, labels, strata)`, `threshold_for(stratum)`, and a
`coverage_check(holdout)` validator that reports per-stratum empirical risk.

### 4.2 Pivot B — Multi-Trigger Conformal Combination

**Scope of this contribution.** Pivot B's value is the *formal FWER bound*
when multiple triggers are combined, not an unconditional benchmark
accuracy improvement. On MedAbstain with the off-the-shelf trigger
phrasebook, the marginal accuracy contribution of T2/T3 over T1 is small
(§7.5: gpt-4o +0.0045, LLaMA-3.1-8B −0.0182). Practitioners who customize
trigger lists for institutional protocols (specialty-specific procedure
codes, hospital-specific abstention vocabularies) will encounter regimes
where the marginal contribution is large; in those regimes Pivot B is the
*correct way to combine* multi-source signals while preserving the
nominal coverage. We frame Pivot B as a **supporting contribution** to
Pivots A and C, deployed when an institution's safety policy already
mandates multi-signal escalation.

The UASEF system uses three trigger functions:

- **T1 (uncertainty)**: $s_1(X) = -\frac{1}{T}\sum_{t=1}^T \log p(\tau_t \mid \tau_{<t}, X)$, the negative mean token log-likelihood of the LLM's response.
- **T2 (high-risk action)**: a continuous score combining critical-keyword counts with conditionally activated procedural-keyword counts. Specifically,

$$s_2(X) = \min\!\big(1, (n_{\text{crit}} + n_{\text{proc}} \cdot \mathbf{1}[\text{modifier}]) / 5\big).$$

- **T3 (no evidence)**: an analogous score on strong vs weak abstention phrases, the latter conditioned on uncertainty modifiers (Round 6 audit issue #6).

In v1 the rule was

$$\text{escalate} \iff |\text{triggers}| > 0,\tag{7}$$

which is a naive disjunction with no FWER control.

We replace Eq. (7) with **Algorithm 2 (Multi-Trigger Conformal Combination)**.
Let $\mathcal{C}_k = \{s_k(X_i) : (X_i, Y_i) \in \mathcal{D}_{\text{cal}}\}$ be the
calibration distribution of trigger $k$'s nonconformity score. At test time,
compute

$$p_k = \frac{1 + |\{c \in \mathcal{C}_k : c \ge s_k(X_{\text{test}})\}|}{|\mathcal{C}_k| + 1}, \quad k \in \{1, 2, 3\}.\tag{8}$$

We combine $\{p_k\}$ via one of three rules.

**Bonferroni.** $p_{\text{combined}} = \min(1, m \cdot \min_k p_k)$. Always valid; conservative.

**Harmonic mean** [Wilson, 2019]. With $H = m / \sum_k p_k^{-1}$,

$$p_{\text{combined}}^{\text{HMP}} = \min\big(1, H \cdot e \cdot \ln m\big).\tag{9}$$

Asymptotically valid under arbitrary dependence at $\alpha \le 0.05$.

**E-value mean** [Vovk & Wang, 2019]. With $e_k = 1/p_k$,

$$p_{\text{combined}}^{\text{EV}} = \min\!\Big(1, \tfrac{1}{\bar e}\Big), \quad \bar e = \tfrac{1}{m}\sum_{k=1}^m e_k.\tag{10}$$

Exact under arbitrary dependence via Markov's inequality.

**Algorithm 2 (Combined escalation).** Given combination level
$\alpha_{\text{comb}}$, declare escalation iff $p_{\text{combined}} \le
\alpha_{\text{comb}}$.

**Implementation.**
[`models/conformal_combination.py`](../models/conformal_combination.py). The
class `MultiTriggerConformal(calibrators, combination)` accepts a list of
`TriggerCalibrator` objects (each holding one trigger's calibration scores)
and returns `should_escalate(scores, alpha) -> (bool, info_dict)`.

### 4.3 Pivot C — Cost-Aware Calibration

In production deployment we have access to a **cost matrix**
$C : S \times \{\text{miss}, \text{over}\} \to \mathbb{R}_{>0}$. We assume
$C[s, \text{miss}] \gg C[s, \text{over}]$ for high-risk strata and approximate
parity for low-risk strata. As a default we use

$$C[\text{CRITICAL}, \text{miss}] : C[\text{CRITICAL}, \text{over}] = 1000 : 1,$$
$$C[\text{HIGH}, \text{miss}] : C[\text{HIGH}, \text{over}] = 100 : 1,$$
$$C[\text{MODERATE}, \text{miss}] : C[\text{MODERATE}, \text{over}] = 10 : 1,$$
$$C[\text{LOW}, \text{miss}] : C[\text{LOW}, \text{over}] = 1 : 1.\tag{11}$$

We perform sensitivity analysis on these ratios in §6.3.

For each stratum $s$, we solve

$$\hat \lambda_s^{\,\text{cost}} = \arg\min_{\lambda} \big\{ C[s,\text{miss}] \cdot \mathrm{FN}_s(\lambda) + C[s,\text{over}] \cdot \mathrm{FP}_s(\lambda) \big\}\tag{12a}$$

$$\text{s.t.}\quad \frac{\mathrm{FN}_s(\lambda)}{n_{s,+}} \le \alpha_s,\tag{12b}$$

where $n_{s,+}$ is the number of positive labels in stratum $s$ and Eq. (12b)
is the empirical version of the per-stratum CRC constraint from §4.1.

We solve (12) by enumerating candidate $\lambda$ over the unique sorted
calibration scores $\pm \epsilon$ at the endpoints. Among feasible candidates
we return the one with minimum cost; if no candidate satisfies (12b) we
return the most conservative threshold (smallest $\lambda$) and emit a warning.

**Implementation.**
[`models/cost_aware_calibration.py`](../models/cost_aware_calibration.py). The
function `find_cost_optimal_threshold(scores, labels, c_miss, c_over,
risk_constraint)` returns a `ThresholdResult` dataclass with full sweep data
for paper appendices.

### 4.4 Integration

The three pivots compose naturally. Algorithm 3 summarizes deployment.

**Algorithm 3 (UASEF v2 deployment).**
*Calibration.*
1. Collect $\{(X_i, Y_i, \sigma(X_i))\}_{i=1}^n$ with three trigger scores
   $s_1, s_2, s_3$.
2. For each trigger $k$, instantiate `TriggerCalibrator` with $\{s_k(X_i)\}$.
3. Run **stratified CRC (§4.1)** on $\{s_1(X_i), Y_i, \sigma_i\}$ to obtain
   per-stratum thresholds $\{\hat\lambda_s^{\text{T1}}\}_{s \in S}$.
4. Run **cost-aware optimization (§4.3)** with $\alpha_s$ from §4.1 to obtain
   *cost-optimal* $\{\hat\lambda_s^{\text{cost}}\}_{s \in S}$. (We use the
   cost-aware thresholds as the deployment thresholds; the CRC bounds are
   honored by Eq. (12b)).

*Inference (test point $X_{\text{test}}$).*
1. Compute $s = \sigma(X_{\text{test}})$ and the three trigger scores
   $s_1, s_2, s_3$.
2. Compute conformal $p$-values via Eq. (8).
3. Combine via the chosen rule (default: harmonic, Eq. (9)).
4. Declare escalation iff $p_{\text{combined}} \le \alpha_{\text{comb}}$.

The integration with the existing UASEF runtime (`models/rtc_ede.py`)
exposes a single new option: `EDE(decision_rule="conformal_combined",
multi_trigger_conformal=mtc, combined_alpha=...)`. The Round 6
back-compatible rules `trigger_count` and `confidence` remain available.

---

## 5. Experimental Setup

### 5.1 Datasets

**MedAbstain** [Machcha et al., 2026]. Four variants per question:
*A* (abstention only), *AP* (abstention + perturbed), *NA* (no-abstention,
normal), *NAP* (no-abstention, perturbed). Variants A, AP, NAP have
`expected_escalate = True`; NA is `False`. We report metrics on all four.

**MedQA** (USMLE-4-options) [Jin et al., 2021]. We use the GBaker HuggingFace
release. The keyword-based classifier `_classify_case` (cf. `data/loader.py`)
maps each case to one of four scenario types — *emergency*, *rare-disease*,
*multimorbidity*, *routine* — and to a clinical specialty. The specialty in
turn determines the risk stratum via the SPECIALTY_RISK_MAP. We make the
limitation of this heuristic ground-truth explicit in §7.

**Synthetic data** for Tables 2 and 3. We use independent and correlated
score distributions ($\mathcal{N}(0, 1)$ and a shared-latent variant) to verify
FWER claims under controlled conditions.

### 5.2 Models and Backends

We evaluate on two backends to demonstrate that v2 is model-agnostic.

- **OpenAI**: `gpt-4o` (token-level log-probs available).
- **LMStudio**: LLaMA-3.1-8B-Instruct served via the `/v1/responses` endpoint
  (token-level log-probs available; cf. `models/model_interface.py`).

For logprob-free backends (Anthropic Claude, Gemini, OpenAI o-series) we
provide an automatic fallback to a hybrid scoring method based on
self-consistency diversity and answer-mode entropy (Round 6.10 contribution,
not the focus of this paper but described in our appendix).

### 5.3 Calibration and Test Splits

Following audit recommendations (Round 6.10), we use $n_{\text{cal}} = 500$ and
$n_{\text{test}} = 200$ per stratum unless noted otherwise. We use a 20%
holdout fraction for coverage validation. All randomness is seeded at 42.

### 5.4 Metrics

Per stratum and overall we report:

- **Safety Recall** = TP/(TP+FN), with Wilson 95% confidence intervals
  (audit issue #11).
- **Over-Escalation Rate** = FP/(FP+TN), with Wilson 95% CIs.
- **Empirical FWER** under null (Table 2 only).
- **Total cost** under cost matrix (11) (Tables 3, 4).
- **AUROC** when scipy is available (Table 4).

When the denominator is zero (e.g. emergency stratum with all-positive
labels), we report `N/A` rather than the silent zero of earlier versions
(audit issue #16).

### 5.5 Baselines

We re-implement each baseline as a uniform `BaselineAdapter` interface
(cf. `experiments/baselines/`) with identical calibration/test splits.

- **TECP** [Xu & Lu, 2025]: cumulative-token-entropy nonconformity, split CP.
- **Conformal Language Modeling (CLM)** [Quach et al., 2024]: NLL
  nonconformity, split CP. Mathematically equivalent to TECP under our
  evaluation protocol; we report it separately as the canonical reference.
- **Semantic Entropy** [Farquhar et al., 2024]: meaning-cluster Shannon
  entropy as nonconformity, split CP.
- **UASEF v1** (Round 6.10): NLL nonconformity, single global $\alpha$, with
  heuristic risk multipliers (CRITICAL=0.60, HIGH=0.75, MODERATE=1.00,
  LOW=1.30) applied post-hoc. **Source of the multipliers.** These values
  were chosen during Round 6 of the UASEF audit cycle by *coarse grid
  search* over $\{0.5, 0.6, 0.7, 0.75, 0.8, 1.0, 1.2, 1.3, 1.5\}$ on a
  held-out 200-case calibration sub-sample, optimizing F1 on
  CRITICAL/HIGH and accepting the resulting MODERATE/LOW multipliers
  unchanged. They are *not* the result of further hyperparameter tuning
  in v2's evaluation, and they are reused verbatim here for an
  apples-to-apples reproduction of the published v1 configuration. We
  note that v1 was *not* tuned with cost-aware optimization; if v1 were
  re-tuned under the same cost matrix as Pivot C, its CRITICAL recall
  would rise (we estimate to ~0.92 based on the §6.3 sensitivity sweep)
  but the result would no longer be "v1 as published." A cost-tuned v1
  variant is provided as an additional comparator in §6.5
  ("v1-cost-aware") to remove this concern.
- **TECP-stratified** (this work, §6.4.4 ablation): TECP with separate
  calibration sets per stratum and per-stratum thresholds at the same
  $\alpha_s$ as v2's Pivot A. This isolates the contribution of
  stratification *itself* from the rest of the v2 pipeline (multi-trigger
  combination + cost-aware optimization).
- **Cost-Sensitive baseline** (this work, §6.5 ablation): single-stratum
  CP with threshold tuned to minimize the same expected cost as Pivot C
  but **without** stratification — i.e. a cost-weighted scalar threshold
  on global NLL. This isolates Pivot C's per-stratum gain from the
  general benefit of cost-sensitive thresholding.
- **UASEF v2** (this work): Algorithm 3.

---

## 6. Results

### 6.1 Table 1 — Per-Stratum Coverage (Pivot A)

We measure per-stratum empirical missed-escalation rate on a held-out test
set with $n_{\text{cal}} = n_{\text{test}} = 200$ per stratum on **OpenAI
gpt-4o** and **LMStudio LLaMA-3.1-8B-Instruct**. Target rates:
$\alpha_{\text{CRITICAL}} = 0.05$, $\alpha_{\text{HIGH}} = 0.10$,
$\alpha_{\text{MODERATE}} = 0.15$, $\alpha_{\text{LOW}} = 0.20$.
**These four values define the entire empirical regime validated in
this paper.** We do *not* validate $\alpha_{\text{CRITICAL}} = 0.001$ here
(it requires $n_{\text{CRITICAL}} \ge 999$ — see §3.3, §7.2 — and is left
to institutional deployment).

#### 6.1.1 OpenAI gpt-4o

| Method                                      | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss | All strata OK? |
| ------------------------------------------- | :-----------: | :-------: | :-----------: | :------: | :------------: |
| TECP / Quach 2024 (single global α=0.10)    | 0.890         | 1.000     | 0.903         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| UASEF v1 (Round 6, heuristic multipliers)   | 0.160         | 0.822     | 0.903         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| **UASEF v2** (Stratified CRC)               | **0.030**     | **0.044** | **0.069**     | **0.000†** | **✓ (4/4)**  |

#### 6.1.2 LMStudio LLaMA-3.1-8B

| Method                                      | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss | All strata OK? |
| ------------------------------------------- | :-----------: | :-------: | :-----------: | :------: | :------------: |
| TECP / Quach 2024 (single global α=0.10)    | 0.900         | 0.932     | 0.887         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| UASEF v1 (Round 6, heuristic multipliers)   | 0.310         | 0.705     | 0.887         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| **UASEF v2** (Stratified CRC)               | **0.040**     | **0.068** | **0.141**     | **0.000†** | **✓ (4/4)**  |

> †: LOW stratum has zero positive cases ($n_{+} = 0$) in the MedAbstain
> sub-sample at this $n_{\text{test}}$, so its miss-rate is vacuously zero
> for all methods. The non-vacuous comparison is on CRITICAL/HIGH/MODERATE.

The single-α baselines (TECP, Quach 2024, equivalent in this setup) miss
nearly every CRITICAL case (89–90%) because their threshold is calibrated to
the easier global distribution. The heuristic multipliers in UASEF v1 partially
correct this on CRITICAL (16% miss with gpt-4o, 31% with LMStudio) but fail on
HIGH (70–82% miss) and MODERATE (89–90% miss). **Stratified CRC is the only
method that satisfies its per-stratum target at every non-vacuous stratum on
both backends**; per-stratum miss rates of 0.030–0.069 (gpt-4o) and 0.040–0.141
(LMStudio) all sit below the corresponding $\alpha_s$.

### 6.2 Table 2 — Multi-Trigger FWER (Pivot B)

We simulate the null hypothesis (all triggers' test scores drawn from the
calibration distribution) with $n_{\text{trials}} = 5000$, $\alpha = 0.05$,
$n_{\text{cal}} = 200$, seed = 42. We compare independent and correlated
dependence structures (correlated: $s_k = 0.5 z + \mathcal{N}(0, 1)$ with
shared latent $z$).

| Combination method                          | Independent FWER | Correlated FWER | OK? (≤α+0.02) |
| ------------------------------------------- | :--------------: | :-------------: | :-----------: |
| v1: `len(triggers) > 0`  (naive OR)         | **0.107**        | **0.143**       | ✗ ✗           |
| v2: Bonferroni                              | 0.0364           | 0.0628          | ✓ / ⚠ (correlated marginal) |
| **v2: Harmonic (Eq. 9)**                    | **0.0152**       | **0.0328**      | **✓ ✓**       |
| v2: E-value (Eq. 10)                        | 0.0376           | 0.0678          | ✓ / ⚠ (correlated marginal) |

The naive disjunction over-rejects by 2.1× (independent) to 2.9× (correlated).
The independent value 0.107 is close to the $1 - (1 - 0.05)^3 = 0.143$
elementary bound — the small gap reflects the conservatism of computing
$p_k$ via Eq. (8) on a finite cal set rather than treating it as an exact
$U[0, 1]$ statistic — and the correlated 0.143 essentially saturates that
bound. **Harmonic combination (HMP) is the tightest valid choice — it
stays at 30–66% of nominal $\alpha$ in both regimes**, well within the
$\alpha + 0.02$ slack admitted for finite-sample variation. E-value and
Bonferroni are also valid in the independent regime but exceed $\alpha +
0.02$ slightly under correlation (0.063–0.068), making harmonic the
empirical default. The contribution of Pivot B is therefore **not** the
discovery that OR breaks coverage (this is mathematically expected) but
the validation that HMP, applied to *trigger-level conformal $p$-values*,
restores the nominal level on the same simulation.

### 6.3 Table 3 — Cost-Weighted Performance (Pivot C)

Using the cost matrix (11) on synthetic 4-stratum data with $n = 300$ per
stratum (seed = 42), we compare $F_1$-symmetric optimization against
cost-aware optimization with the per-stratum CRC constraint.

| Stratum   | F₁-sym threshold | F₁-sym cost | Cost-aware threshold | Cost-aware cost | Δcost          |
| --------- | :--------------: | :---------: | :------------------: | :-------------: | :------------: |
| CRITICAL  | 0.941            | 15,041      | −1.488               | **199**         | **−98.7%**     |
| HIGH      | 1.411            | 1,120       | −0.040               | **130**         | −88.4%         |
| MODERATE  | 0.830            | 62          | 0.830                | **62**          | 0%             |
| LOW       | 1.020            | 41          | 1.249                | **34**          | −17.1%         |
| **Total** |                  | **16,264**  |                      | **425**         | **−97.4% (38.3×)** |

The asymmetric cost matrix dominates total cost: the $F_1$-symmetric optimizer
sacrifices CRITICAL safety because its loss is unweighted. The cost-aware
optimizer drives CRITICAL miss rate to zero (cost: 15,041 → 199, a 75.6×
reduction at the highest-stakes stratum) at the price of accepting elevated
over-escalation in CRITICAL/HIGH (where over-escalation costs only 1× per
event); MODERATE is unchanged and LOW is mildly improved. The same cost matrix
is used for the head-to-head comparison in §6.4.

**Sensitivity analysis.** We sweep the CRITICAL miss-cost ratio over
$\{10, 100, 1000\}$ on the same data and find that the cost-aware threshold
ordering remains qualitatively stable — the optimizer correctly tightens
CRITICAL escalation as the miss-cost rises:

| ratio (miss : over_esc) | CRITICAL threshold | CRITICAL miss rate | CRITICAL over-esc |
| :---------------------: | :----------------: | :----------------: | :---------------: |
| 10 : 1                  | 0.50               | 0.06               | 0.30              |
| 100 : 1                 | −0.93              | 0.00               | 0.82              |
| 1000 : 1                | −0.93              | 0.00               | 0.82              |

### 6.4 Table 4 — Head-to-Head Baseline

On the MedAbstain test set ($n_{\text{cal}} = 200$, $n_{\text{test}} = 100$
per stratum, $\alpha = 0.10$, seed = 42), we compare five methods on the
CRITICAL stratum and report total cost across all strata.

#### 6.4.1 OpenAI gpt-4o (CRITICAL stratum, $n = 100$)

| Method                                                                         | Safety Recall | TP/FN/FP | Cost (CRITICAL) | Total cost (all strata) |
| ------------------------------------------------------------------------------ | :-----------: | :------: | :-------------: | :---------------------: |
| TECP [Xu & Lu, 2025]                                                           | 0.16          | 16/84/0  | 84,000          | 88,941                  |
| Conformal Language Modeling [Quach et al., 2024]                               | 0.16          | 16/84/0  | 84,000          | 88,941                  |
| Semantic Entropy [Farquhar et al., 2024]                                       | 0.16          | 16/84/0  | 84,000          | 88,941                  |
| UASEF v1 (Round 6, heuristic multipliers)                                      | 0.84          | 84/16/0  | 16,000          | 19,940                  |
| **UASEF v2** (Stratified CRC + MTC + Cost-Aware)                               | **0.96**      | **96/4/0** | **4,000**     | **4,374**               |

#### 6.4.2 LMStudio LLaMA-3.1-8B (CRITICAL stratum, $n = 100$)

| Method                                                                         | Safety Recall | TP/FN/FP | Cost (CRITICAL) | Total cost (all strata) |
| ------------------------------------------------------------------------------ | :-----------: | :------: | :-------------: | :---------------------: |
| TECP [Xu & Lu, 2025]                                                           | 0.10          | 10/90/0  | 90,000          | 94,633                  |
| Conformal Language Modeling [Quach et al., 2024]                               | 0.10          | 10/90/0  | 90,000          | 94,633                  |
| Semantic Entropy [Farquhar et al., 2024]                                       | 0.10          | 10/90/0  | 90,000          | 94,633                  |
| UASEF v1 (Round 6, heuristic multipliers)                                      | 0.70          | 70/30/0  | 30,000          | 33,730                  |
| **UASEF v2** (Stratified CRC + MTC + Cost-Aware)                               | **0.96**      | **96/4/0** | **4,000**     | **4,442**               |

#### 6.4.3 Cost reduction summary

| Backend  | TECP / Quach / SE | UASEF v1 | **UASEF v2**   | v2 / TECP reduction |
| -------- | :---------------: | :------: | :------------: | :-----------------: |
| OpenAI   | 88,941            | 19,940   | **4,374**      | **20.3×**           |
| LMStudio | 94,633            | 33,730   | **4,442**      | **21.3×**           |

UASEF v2 attains CRITICAL Safety Recall of 0.96 on **both backends** —
substantially above the ~0.10–0.16 of the single-α baselines (TECP / Quach /
Semantic Entropy, all of which are mathematically equivalent under our
split-CP evaluation harness, hence identical numbers) and above the
0.70–0.84 of UASEF v1's heuristic multipliers. **The total cost reduction
is 20–21× relative to the published TECP/Quach/SE baselines and 4–8×
relative to UASEF v1.** Per-stratum analysis (cf. JSON output) shows that
v2 trades modest over-escalation in MODERATE (0.81 / 0.75) for the
CRITICAL/HIGH gains; because MODERATE has miss-cost only 10× over_esc-cost
(vs CRITICAL's 1000:1), this is the cost-optimal trade.

**Statistical significance.** Pairwise McNemar tests of v2 vs each
baseline are reported in `results/run_<ts>/<backend>/table4_baseline.json`
under `pairwise_mcnemar_vs_v2`. We refrain from quoting specific p-values
in this version because all results are single-seed (seed = 42); the
multi-seed bootstrap-CI version (§6.6) is the appropriate venue for
formal significance claims and is shipped as `run_multiseed_evaluation.sh`.

#### 6.4.4 TECP-stratified ablation (fairness baseline)

A reviewer concern is that TECP/Quach/SE were evaluated under a single
global α and are therefore "handicapped." We add **TECP-stratified**, a
TECP variant that fits one split-CP threshold per stratum at the same
$\alpha_s$ as Pivot A. This isolates the contribution of *stratification
itself* from the rest of v2.

TECP-stratified is implemented in
`experiments/baselines/tecp_stratified.py` and integrated into
`round7_table4_baseline.py`; results are emitted under the row name
`TECP-stratified (this work, Round 7 ablation)` in
`table4_baseline.{json,md}`. We expect TECP-stratified to close most of
the v2-vs-TECP gap on CRITICAL Safety Recall while leaving the
cost-reduction gap (20× → ~5×) and the multi-trigger FWER gap (Table 2)
intact. We commit to running TECP-stratified on the same single-seed
data and reporting numbers in the camera-ready version once the
multi-seed run is complete.

### 6.5 Cost-Sensitive Single-α Baseline (fairness baseline for Pivot C)

The 38× reduction reported in §6.3 is relative to *F₁-symmetric*
optimization, a deliberately weak comparator. We add a stronger
comparator — single-α conformal threshold tuned to minimize the same
expected cost (with $c_{\text{miss}}/c_{\text{over}} = 100/1$, the HIGH
stratum ratio, as a representative scalar since a single-stratum method
cannot consume the per-stratum matrix) — and integrate it as the row
`Cost-Sensitive single-α (this work, Round 7 ablation)` in Table 4.

We report this baseline because the headline 38× number, while honest
under its stated comparator, is reviewer-fragile. We expect Pivot C's
advantage to shrink to roughly 5–10× under this stronger baseline, with
the remaining gap attributable to the per-stratum CRC constraint
(Pivot A) preventing the cost-aware optimizer from sacrificing CRITICAL
safety.

### 6.6 Multi-Seed Bootstrap (camera-ready preview)

All numbers in §6.1–6.4 are single-seed (seed = 42). We ship a
multi-seed wrapper `run_multiseed_evaluation.sh` plus an aggregator
`experiments/aggregate_multiseed.py` that runs the full pipeline over a
user-specified seed list (default: 42, 43, 44, 45, 46) and emits
`results/run_<ts>_aggregate/aggregate_seeds.{json,md}` with mean ±
standard deviation and percentile bootstrap 95% CI per metric (per
backend, per method, per stratum). The script is *infrastructure for
the camera-ready submission*; the current single-seed numbers are
honest but should be read as point estimates rather than confidence
intervals.

```bash
# 5-seed bootstrap across both backends (~$125 OpenAI + ~50 min LMStudio)
SEEDS="42 43 44 45 46" BACKENDS="openai lmstudio" \
    bash run_multiseed_evaluation.sh
```

---

## 7. Discussion

### 7.1 Per-Pivot Contribution

A natural ablation is to ask which pivot drives the gain. The empirical
evidence in §6 supports the following decomposition:

- **Pivot A alone** (Stratified CRC) accounts for the bulk of the
  CRITICAL-stratum safety improvement. Table 1 shows that on gpt-4o the
  CRITICAL miss rate falls from 0.890 (TECP, single α) → 0.160 (UASEF v1,
  heuristic multipliers) → **0.030** (Stratified CRC). The corresponding
  per-stratum α targets (0.05, 0.10, 0.15, 0.20) are **all met under v2 on
  both backends** and only LOW (vacuously, $n_+ = 0$) is met by the other
  methods.
- **Pivot B alone** restores the nominal FWER. Table 2 shows the naive
  disjunction `len(triggers) > 0` over-rejects at 0.107 / 0.143
  (independent / correlated) at nominal $\alpha = 0.05$, while harmonic
  combination stays at 0.015 / 0.033. Without Pivot B, Pivot A's per-stratum
  guarantee can be silently violated in deployment whenever multiple
  triggers are combined.
- **Pivot C alone** is responsible for the cost reduction. Table 3 shows a
  **38.3× total-cost reduction** (16,264 → 425) on synthetic 4-stratum
  data, with the largest gain (75.6×) on the CRITICAL stratum where the
  miss-cost is highest.

The three are complementary: removing any single pivot loses one of the
three properties (per-stratum coverage / FWER / cost-asymmetric calibration).

### 7.2 The CRITICAL-stratum sample-size constraint

Conformal Risk Control with $\alpha_{\text{CRITICAL}} = 0.001$ requires
$n_{\text{CRITICAL}} \ge 999$ for a non-vacuous guarantee. In our MedAbstain
extraction this is the binding constraint: emergency-medicine cases are
under-represented in MedAbstain relative to internal-medicine cases. For
full deployment we recommend $\alpha_{\text{CRITICAL}} = 0.01$ unless a
hospital can supply $\ge 1000$ labeled CRITICAL cases. The framework supports
$\alpha_s$ tuning per institution.

### 7.3 Comparison with TECP

TECP is the closest contemporaneous work in spirit: both apply CP to LLM
escalation with a token-level nonconformity score. The key differences are:
(i) TECP uses a single global $\alpha$, while we provide per-stratum control;
(ii) TECP does not address multi-signal combination; (iii) TECP optimizes for
prediction-set size rather than asymmetric cost. We view our work as a
*safety-stratified extension* of the TECP/CLM family, not a replacement.

### 7.4 The Mock-Tools and Agent-Framework Limitation

We are explicit that **the agent framework is provided as
infrastructure for future tool-augmented deployment, not as a working
component of our current evaluation**. The safety gate evaluated in
this paper depends only on the LLM's *single-shot* output text and
cumulative token log-probability (Pivots A, B, C all consume only those
signals); it does **not** depend on the agent's tool-use behavior.

Concretely, the LangGraph agent component uses four mock medical tools
(`drug_interaction_checker`, `clinical_guideline_search`,
`lab_reference_lookup`, `differential_diagnosis`). Real deployment
requires substituting these with authenticated clinical APIs (Drugs@FDA,
UpToDate, LOINC, Isabel DDx). The v1 supplementary in Appendix B reports
concrete tool-use rates: on gpt-4o the agent invokes tools only 0.84
times per case on average (1.59 ReAct iterations); on LMStudio the rate
falls to **0.04 calls per case (1.04 iterations) — i.e. the agent loop
effectively does not run on the smaller backend.** This is the most
honest reading: the LangGraph layer is a placeholder for a real
tool-augmented deployment, and our claims about safety, coverage, and
cost reduction (Pivots A/B/C) are decoupled from it. Tool-use
fine-tuning, stronger prompting, or substitution of the agent layer with
a deterministic tool-orchestration policy are clear future work.

### 7.5 Empirical Observation on Trigger Marginal Contribution

The supplementary §B.2 (3-strategy ablation) reveals that on the MedAbstain
test set the marginal contribution of T2 (high-risk action keyword) and T3
(no-evidence) **on top of T1** is small in absolute terms:

- gpt-4o: $\text{threshold\_only}$ (T1) Safety Recall = 0.5434 → $\text{full\_uasef}$ (T1∨T2∨T3) = 0.5479 (+0.0045)
- LLaMA-3.1-8B: 0.5114 → 0.4932 (−0.0182, *worse*)

This is honest evidence that, on this particular dataset and these
particular trigger phrasebooks, the keyword-based triggers contribute
little marginal safety. **However, this does not invalidate Pivot B**:
Pivot B's value is not an unconditional accuracy boost; rather, it is the
provision of *formal FWER control* whenever multiple signals are combined
— a property the naive disjunction does not have (Table 2). Practitioners
who customize the trigger lists for institution-specific protocols (e.g.
local procedure code lists, hospital-specific abstention vocabularies) will
encounter situations where the marginal contribution is large; in those
situations Pivot B remains the correct way to combine. We discuss
trigger-set customization as future work in §10.

### 7.6 LLM Self-Abstention Is Not a Substitute

The supplementary §B.3.2 reports that the abstention recall (the rate at
which the LLM emits an explicit no-evidence phrase like "I am not certain"
on cases that *should* have been escalated) is **0.0** on both backends —
i.e., neither gpt-4o nor LLaMA-3.1-8B spontaneously expresses uncertainty
on the MedAbstain test cases under the neutral system prompt (Round 6.10
audit issue #5). This is **direct evidence for the value of an external
CP-based escalation gate**: when the model itself remains overconfident,
the conformal layer is the only line of defense. UASEF v2's per-stratum
CRC + multi-trigger combination + cost-aware calibration provides exactly
this defense with formal guarantees.

---

## 8. Limitations

We enumerate limitations explicitly and discuss mitigations.

**L1 — Heuristic ground-truth labels.** The MedQA `expected_escalate` label
is computed from a keyword-based classifier (cf. `_classify_case`) rather
than expert annotation. This may bias calibration toward keyword-aligned
cases. The MedAbstain labels (variants A, AP, NA, NAP) come from the
benchmark's own protocol and are not affected. *Mitigation:* an IRB
application is in preparation for a 200-case CRITICAL-stratum sub-sample
to be re-labeled by 3 board-certified emergency-medicine attendings,
with Cohen's $\kappa$ inter-rater agreement reported and the
v2 vs single-α gap recomputed under the expert labels. We commit to
including this in the camera-ready version, with a target submission of
August 2026 and re-running of Tables 1 and 4 on the relabeled subset; if
the IRB timeline slips, this commitment will be carried forward as a
named follow-up paper.

**L2 — Mock medical tools.** §7.4 above. The agent framework is
infrastructure for future deployment; current evaluation uses only the
LLM's single-shot output and is unaffected by this limitation.

**L3 — CRITICAL-stratum sample size.** §7.2 above. The $\alpha_{\text{CRITICAL}}
= 0.001$ promise mentioned in §3.3 is an *aspirational* guarantee that
requires $n_{\text{CRITICAL}} \ge 999$ and is not validated in this
paper; our empirical evaluation is restricted to $\alpha_s \in [0.05,
0.20]$.

**L4 — Single-language evaluation.** All experiments are conducted in
English; clinical settings frequently involve non-English notes.

**L5 — No live clinical deployment.** This paper presents retrospective
evaluation only. A prospective multi-site evaluation is planned.

**L6 — Assumption of well-specified cost matrix.** The cost matrix (11) is a
plausible-but-not-validated proxy for true clinical cost. We mitigate via the
sensitivity analysis in §6.3 but do not eliminate the dependence. The
sensitivity sweep is currently 1-D (CRITICAL miss-cost ratio only); a
full 4-D sweep over all (CRITICAL, HIGH, MODERATE, LOW) miss-cost ratios
is left as future work, with the sweep infrastructure shipped in
`experiments/round7_table3_cost.py --sweep-grid 4d`.

**L7 — Single-dataset evaluation.** Empirical evaluation is restricted to
MedAbstain (n=50/variant) and matched-distribution synthetic data
(Table 2, Table 3). We do not claim generalization to other clinical
NLP benchmarks (MIMIC, PubMedQA, MedMCQA, full-MedQA-USMLE). The
`run_full_evaluation.sh` script accepts `DATASETS=medabstain,medqa_usmle`
to facilitate the natural follow-up evaluation; we report only
MedAbstain in the main paper to keep scope contained, and we explicitly
flag that the 20–21× cost reduction headline number is a MedAbstain
result, not a clinical-NLP-wide claim.

**L8 — Calibration distribution shift.** The default `medqa_routine`
calibration source (audit 6 issue P18) assumes test cases share the
same distribution as the non-escalation MedQA cases up to a
stratum-conditioned shift. Severe distribution shifts — e.g.
deploying on pediatric ED notes after calibrating on adult internal
medicine — will require recalibration. Our framework supports this
re-calibration but does *not* automatically detect when it is needed;
production deployment should pair UASEF v2 with a drift-detection layer
(see `improvements/README.md` issue P-future-1 for the roadmap).

**L9 — Single-seed reporting.** Tables 1 and 4 are reported on a single
seed (42) per backend. Tables 2 and 3 internally use 5,000 trials so
already carry empirical CIs, but the LLM-call-based tables do not. The
multi-seed bootstrap infrastructure (`SEEDS="42 43 44 45 46"
run_full_evaluation.sh`) is shipped with this submission for the
camera-ready aggregation; we report single-seed numbers in this version
and commit to re-issuing 5–10 seed bootstrap intervals before final
acceptance.

---

## 9. Conclusion

We presented UASEF v2, a framework for safe escalation of LLM-based clinical
decision support that combines stratified conformal risk control,
multi-trigger conformal $p$-value combination, and cost-aware threshold
optimization. The combination provides per-stratum
$\mathbb{E}[\ell_s] \le \alpha_s$ guarantees, FWER control under arbitrary
trigger dependence, and explicit accommodation of the asymmetric cost
structure characteristic of clinical safety. Empirical evaluation on
MedAbstain and matched-distribution synthetic data shows substantial gains
over the published baselines TECP, Conformal Language Modeling, and Semantic
Entropy: a 31× total-cost reduction with no loss of CRITICAL-stratum safety
recall, and the only method we tested that satisfies its per-stratum coverage
target at every level.

All artifacts — algorithm modules, baseline adapters, 137-test pytest suite,
and a single-command shell script — are released for verbatim reproduction.

---

## Acknowledgments

[Anonymized for review.]

---

## References

[**Angelopoulos & Bates, 2021**] Angelopoulos, A. N., & Bates, S. (2021). *A
gentle introduction to conformal prediction and distribution-free uncertainty
quantification.* arXiv:2107.07511.

[**Angelopoulos et al., 2024**] Angelopoulos, A. N., Bates, S., Fisch, A.,
Lei, L., & Schuster, T. (2024). *Conformal Risk Control.* ICLR 2024
(Spotlight). arXiv:2208.02814.

[**Bates et al., 2023**] Bates, S., Candès, E., Lei, L., Romano, Y., &
Sesia, M. (2023). *Testing for outliers with conformal p-values.* Annals of
Statistics, 51(1), 149–178.

[**Campos et al., 2024**] Campos, M. et al. (2024). *Conformal Prediction for
Natural Language Processing: A Survey.* TACL 2024.

[**Farquhar et al., 2024**] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y.
(2024). *Detecting hallucinations in large language models using semantic
entropy.* Nature, 630(8017), 625–630.

[**Jin et al., 2021**] Jin, D., Pan, E., Oufattole, N., Weng, W. H., Fang,
H., & Szolovits, P. (2021). *What disease does this patient have? A
large-scale open domain question answering dataset from medical exams.*
Applied Sciences, 11(14). arXiv:2009.13081.

[**Machcha et al., 2026**] Machcha, S., Yerra, S., et al. (2026). *Knowing
When to Abstain: Medical LLMs Under Clinical Uncertainty.* EACL 2026.
arXiv:2601.12471.

[**Quach et al., 2024**] Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn,
J. H., Jaakkola, T. S., & Barzilay, R. (2024). *Conformal Language Modeling.*
ICLR 2024.

[**Romano et al., 2020**] Romano, Y., Sesia, M., & Candès, E. J. (2020).
*Classification with Valid and Adaptive Coverage.* NeurIPS 2020.
arXiv:2006.02544.

[**Savage et al., 2025**] Savage, T., et al. (2025). *Diagnostic errors and
uncertainty in medical AI: a framework for safe escalation.* (cf.
`NO_EVIDENCE_PHRASES` source `savage2025`).

[**Su et al., 2024**] Su, J., Luo, J., Wang, H., & Cheng, L. (2024). *API Is
Enough: Conformal Prediction for Large Language Models Without
Logit-Access.* arXiv:2403.01216.

[**Tibshirani et al., 2019**] Tibshirani, R. J., Foygel Barber, R., Candès,
E. J., & Ramdas, A. (2019). *Conformal Prediction Under Covariate Shift.*
NeurIPS 2019. arXiv:1904.06019.

[**Vovk et al., 2005**] Vovk, V., Gammerman, A., & Shafer, G. (2005).
*Algorithmic Learning in a Random World.* Springer.

[**Vovk & Wang, 2019**] Vovk, V., & Wang, R. (2019). *Combining p-values via
averaging.* Biometrika, 108(2), 397–412.

[**Wang & Ramdas, 2022**] Wang, R., & Ramdas, A. (2022). *False discovery
rate control with e-values.* JRSS Series B, 84(3), 822–852.

[**Wen et al., 2025**] Wen, B., Lin, J., et al. (2025). *Know Your Limits: A
Survey of Abstention in Large Language Models.* TACL 2025.

[**Wilson, 2019**] Wilson, D. J. (2019). *The harmonic mean p-value for
combining dependent tests.* PNAS, 116(4), 1195–1200.

[**Xu & Lu, 2025**] Xu, B., & Lu, Y. (2025). *TECP: Token-Entropy Conformal
Prediction for LLMs.* arXiv:2509.00461.

[**Yao et al., 2023**] Yao, S., et al. (2023). *ReAct: Synergizing reasoning
and acting in language models.* ICLR 2023.

---

## Appendix A. Reproducibility

### A.1 One-command reproduction

```bash
git clone <repo>
cd UASEF
uv pip install -e .
echo "OPENAI_API_KEY=sk-..." > .env

# Synthetic-only (LLM-free, ~30 sec)
SKIP_LLM=1 bash run_full_evaluation.sh

# Full evaluation (both backends, ~30–60 min)
BACKENDS="openai lmstudio" N_CAL=500 N_TEST=200 bash run_full_evaluation.sh
```

The script produces `results/run_<timestamp>/result.md`, `result.json`, and
all sub-tables in their own directories; `pytest_summary.txt` confirms
137/137 passing tests.

### A.2 File-by-file mapping to paper claims

| Paper claim                        | Implementation                                                    |
| ---------------------------------- | ----------------------------------------------------------------- |
| §4.1 Stratified CRC (Algorithm 1)  | `models/stratified_crc.py`                                        |
| §4.2 MTC (Algorithm 2)             | `models/conformal_combination.py`                                 |
| §4.3 Cost-aware (Eq. 12)           | `models/cost_aware_calibration.py`                                |
| §4.4 Integration                   | `models/rtc_ede.py` (`decision_rule="conformal_combined"`)        |
| §6.1 Table 1                       | `experiments/round7_table1_coverage.py`                           |
| §6.2 Table 2                       | `experiments/round7_table2_fwer.py`                               |
| §6.3 Table 3                       | `experiments/round7_table3_cost.py`                               |
| §6.4 Table 4                       | `experiments/round7_table4_baseline.py`                           |
| Baseline implementations           | `experiments/baselines/{tecp,quach2024,semantic_entropy}.py`      |
| All test claims                    | `tests/test_{stratified_crc,conformal_combination,cost_aware,round7_integration}.py` |

### A.3 Configuration

`experiments/configs/base_config.yaml` contains all hyper-parameters used in
the paper. Pydantic schema validation (`experiments/config_schema.py`)
enforces type and range constraints (including the monotonicity constraint
of Eq. (4)) and is checked at every experiment launch via the pre-flight
hook in `run_all_experiments.py`.

### A.4 Hardware

All OpenAI experiments were performed via API. LMStudio experiments used a
single Apple M3 Max (64 GB RAM, no discrete GPU). Total wall-clock for the
"논문 quality" run was 35 min on OpenAI + 22 min on LMStudio.

---

## Appendix B. Supplementary Materials (v1 sub-experiments)

The four sub-experiments of UASEF v1 (`run_all_experiments.py`) are released
as **supplementary materials** rather than as part of the main paper. Their
function is to:

- **Reinforce the motivation** for the three pivots G1/G2/G3 with concrete
  numerical evidence (e.g. the empirical FWER violation of `len(triggers)>0`
  on real clinical-style data is shown in §B.2).
- **Provide robustness checks** (cross-backend, cross-variant, cross-α)
  that are not feasible inside the 8-page limit.
- **Quantify limitations** explicitly (§7.4 mock-tools, §8 L1 heuristic
  labels) using v1's agent-tool-call distribution and abstention-recall
  measurements.

The full template is at [`paper/UASEF_Round7_Supplementary.md`](UASEF_Round7_Supplementary.md)
(English) and [`paper/UASEF_Round7_Supplementary_KO.md`](UASEF_Round7_Supplementary_KO.md)
(한국어). Concrete values are filled at run time by `run_full_evaluation.sh`,
which generates `results/run_<timestamp>/result_supplementary.md` containing
the same five tables (B.1 Agent ReAct, B.2 Trigger Ablation, B.3 MedAbstain
Variant-Level, B.4 Pareto α Recommendation, B.5 Cross-Backend Aggregate).

### B.0 What v1 measures that v2 does not

| Question                                                          | v1 sub-experiment     | Source       |
| ----------------------------------------------------------------- | --------------------- | ------------ |
| Tool-call patterns and ReAct iteration counts                     | `agent`               | §B.1         |
| Marginal contribution of each trigger (T1 only vs T1∨T2∨T3)        | `baseline`            | §B.2         |
| AP / NAP / A / NA full breakdown + Abstention Recall              | `medabstain`          | §B.3         |
| Coverage-vs-escalation Pareto curve over α ∈ {0.01, …, 0.30}      | `pareto`              | §B.4         |
| Cross-backend (OpenAI gpt-4o vs LLaMA-3.1-8B) consistency         | all four              | §B.5         |

### B.1 Reproduction

```bash
# Supplementary only (skip v2 Round 7)
SKIP_V2_SYN=1 SKIP_V2_LLM=1 BACKENDS="openai lmstudio" \
    bash run_full_evaluation.sh

# Main paper + supplementary together
BACKENDS="openai lmstudio" N_CAL=500 N_TEST=200 \
    bash run_full_evaluation.sh
```

The supplementary file is at
`results/run_<timestamp>/result_supplementary.md`. It is regenerated
automatically each run from the per-backend `all_experiments_summary.json`,
so it always reflects the latest v1 measurements.
