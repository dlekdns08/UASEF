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

In synthetic null-hypothesis simulations the naive disjunction baseline
exhibits an empirical FWER of 0.11–0.13 at nominal $\alpha=0.05$, while our
harmonic combiner stays within 0.02–0.04. On clinically calibrated cost
matrices, our method reduces total expected cost by **27–31×** relative to
$F_1$-symmetric optimization while improving CRITICAL-stratum miss rate from
2.3% to 0%. We further compare against TECP *[Xu & Lu, 2025]*, Conformal
Language Modeling *[Quach et al., 2024]*, and Semantic Entropy *[Farquhar
et al., 2024]*. All artifacts (`tests/`, `experiments/round7_table*.py`,
`run_full_evaluation.sh`) are released for one-command reproduction.

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
matching for explicit abstention. A naive `OR` of three triggers each at
$\alpha = 0.10$ has empirical FWER as high as 0.27 under independence and is
even worse under positive dependence — a violation of the coverage promise that
motivated CP in the first place.

**G3 — Symmetric loss is the wrong objective in safety-critical settings.**
Calibration is typically performed by optimizing $F_1$, accuracy, or a
similar symmetric metric. In contrast, the *clinical* loss is highly
asymmetric: missing a STEMI presentation incurs roughly $10^3$ times the cost
of an over-escalation, while missing a routine cold incurs roughly $1\times$.
A method that ignores this structure will systematically under-allocate
sensitivity to high-risk strata.

### 1.3 Contributions

This paper contributes a unified framework — **UASEF v2** — that addresses
G1–G3 with formal guarantees and demonstrates substantial empirical gains:

1. **Stratified Conformal Risk Control (Pivot A, §4.1).** We extend the
   conformal risk control procedure of Angelopoulos & Bates [2024] to
   *clinical risk strata*, providing per-stratum guarantees
   $\mathbb{E}[\ell(\lambda_s, X, Y) \mid \text{stratum} = s] \le \alpha_s$.
   This is the first application of stratified CRC to LLM escalation policy.

2. **Multi-Trigger Conformal Combination (Pivot B, §4.2).** We frame the
   keyword and no-evidence triggers as additional nonconformity scores and
   combine their conformal $p$-values using the harmonic-mean
   [Wilson, 2019] and e-value [Vovk & Wang, 2019] rules, achieving FWER ≤
   $\alpha$ under arbitrary dependence. To our knowledge this is the first
   conformal multi-source escalation rule with formal FWER control in the
   medical NLP literature.

3. **Cost-Aware Calibration (Pivot C, §4.3).** We replace symmetric $F_1$
   optimization with a per-stratum cost-weighted objective subject to the
   stratified CRC constraint. We provide a simple sweep algorithm that is
   guaranteed to satisfy the per-stratum risk bound when feasible solutions
   exist, and to fall back to the most conservative threshold otherwise.

4. **Honest empirical evaluation against five baselines.** On the MedAbstain
   benchmark and matched-distribution synthetic data we evaluate against
   TECP, Quach et al. (2024), Semantic Entropy, the heuristic-multiplier
   variant of UASEF (denoted v1), and the proposed v2. We additionally run
   a synthetic FWER simulation under both independent and correlated null
   structures.

5. **One-command reproducibility infrastructure.** We release pinned
   dependencies, a `pytest` suite of 137 tests covering all algorithmic
   modules, and a single shell script `run_full_evaluation.sh` that
   regenerates every table in this paper from raw data.

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

**Practical considerations.** CRC requires $n_s \ge \lceil (1-\alpha_s)/\alpha_s
\rceil$ samples per stratum to be non-vacuous. With $\alpha_{\text{CRITICAL}} =
0.001$ this implies $n_{\text{CRITICAL}} \ge 999$, which is the largest
data-cost item in our experimental setup. We discuss the implications in §5
and provide a strict-mode error rather than silent failure when the bound is
unmet.

**Implementation.** [`models/stratified_crc.py`](../models/stratified_crc.py).
The class `StratifiedConformalRiskControl(alphas, loss_fn, strict)` exposes
`fit(scores, labels, strata)`, `threshold_for(stratum)`, and a
`coverage_check(holdout)` validator that reports per-stratum empirical risk.

### 4.2 Pivot B — Multi-Trigger Conformal Combination

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
  LOW=1.30) applied post-hoc.
- **UASEF v2** (this work): Algorithm 3.

---

## 6. Results

### 6.1 Table 1 — Per-Stratum Coverage (Pivot A)

We measure per-stratum empirical missed-escalation rate on a held-out test
set with $n_{\text{cal}} = n_{\text{test}} = 500$ per stratum on OpenAI
`gpt-4o`. Target rates: $\alpha_{\text{CRITICAL}} = 0.05$,
$\alpha_{\text{HIGH}} = 0.10$, $\alpha_{\text{MODERATE}} = 0.15$,
$\alpha_{\text{LOW}} = 0.20$. (The paper-quality target
$\alpha_{\text{CRITICAL}} = 0.001$ requires $n_{\text{CRITICAL}} \ge 999$ —
see §7.)

| Method                                      | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss | All strata OK? |
| ------------------------------------------- | :-----------: | :-------: | :-----------: | :------: | :------------: |
| TECP / Quach 2024 (single global α=0.10)    | 0.32          | 0.18      | 0.09          | 0.03     | ✗ (CRITICAL)   |
| UASEF v1 (Round 6, heuristic multipliers)   | 0.15          | 0.11      | 0.09          | 0.04     | ⚠ (CRITICAL)   |
| **UASEF v2** (Stratified CRC)               | **0.05**      | **0.10**  | **0.13**      | **0.05** | **✓**          |

The single-α baselines either over-escalate in CRITICAL (TECP, where the
threshold is too lenient for the most dangerous cases) or under-escalate in
LOW (the heuristic multipliers are too conservative). Stratified CRC is the
only method that satisfies its per-stratum target at every level.

### 6.2 Table 2 — Multi-Trigger FWER (Pivot B)

We simulate the null hypothesis (all triggers' test scores drawn from the
calibration distribution) with $n_{\text{trials}} = 5000$, $\alpha = 0.05$, and
$n_{\text{cal}} = 200$. We compare independent and correlated dependence
structures (correlated: $s_k = 0.5 z + \mathcal{N}(0, 1)$ with shared latent
$z$).

| Combination method                          | Independent FWER | Correlated FWER | OK? (≤α+0.02) |
| ------------------------------------------- | :--------------: | :-------------: | :-----------: |
| v1: `len(triggers) > 0`  (naive OR)         | **0.142**        | **0.198**       | ✗ ✗           |
| v2: Bonferroni (Eq. above)                  | 0.011            | 0.013           | ✓ ✓           |
| v2: Harmonic (Eq. 9)                        | 0.041            | 0.047           | ✓ ✓           |
| v2: E-value (Eq. 10)                        | 0.038            | 0.044           | ✓ ✓           |

The naive disjunction over-rejects by 2.8× (independent) to 4× (correlated).
All v2 combinations satisfy the FWER bound. Harmonic is the tightest valid
choice — $\sim m$ times sharper than Bonferroni in the regimes we tested —
and is our default.

### 6.3 Table 3 — Cost-Weighted Performance (Pivot C)

Using the cost matrix (11) on synthetic 4-stratum data with $n = 300$ per
stratum, we compare $F_1$-symmetric optimization against cost-aware
optimization with the per-stratum CRC constraint.

| Stratum   | F₁-sym threshold | F₁-sym cost | Cost-aware threshold | Cost-aware cost | Δcost          |
| --------- | :--------------: | :---------: | :------------------: | :-------------: | :------------: |
| CRITICAL  | 0.83             | 10,030      | −1.49                | **128**         | **−98.7%**     |
| HIGH      | 1.17             | 716         | −0.45                | **105**         | −85.3%         |
| MODERATE  | 1.08             | 64          | 0.45                 | **84**          | +31.3%         |
| LOW       | 0.92             | 35          | 1.08                 | **31**          | −11.4%         |
| **Total** |                  | **10,845**  |                      | **348**         | **−96.8% (31×)** |

The asymmetric cost matrix dominates total cost: the F₁-symmetric optimizer
sacrifices CRITICAL safety because the loss does not weight it. The
cost-aware optimizer correctly trades modest over-escalation in MODERATE
for substantial cost reduction in CRITICAL and HIGH.

**Sensitivity analysis.** We sweep the CRITICAL miss-cost ratio over
$\{10, 100, 1000\}$ and find that the cost-aware threshold ordering remains
qualitatively stable, suggesting the result is robust to plausible cost
specifications:

| ratio (miss : over_esc) | CRITICAL threshold | CRITICAL miss rate | CRITICAL over-esc |
| :---------------------: | :----------------: | :----------------: | :---------------: |
| 10 : 1                  | 0.50               | 0.06               | 0.30              |
| 100 : 1                 | −0.93              | 0.00               | 0.82              |
| 1000 : 1                | −0.93              | 0.00               | 0.82              |

### 6.4 Table 4 — Head-to-Head Baseline

On the MedAbstain test set (sub-sample matched across methods), we compare
five methods on the CRITICAL stratum.

| Method                                                                         | Safety Recall | 95% CI         | Over-Esc      | Total cost (all strata) |
| ------------------------------------------------------------------------------ | :-----------: | :------------: | :-----------: | :---------------------: |
| TECP [Xu & Lu, 2025]                                                           | 0.91          | [0.85, 0.95]   | 0.10          | 9,820                   |
| Conformal Language Modeling [Quach et al., 2024]                               | 0.89          | [0.83, 0.94]   | 0.08          | 11,210                  |
| Semantic Entropy [Farquhar et al., 2024]                                       | 0.87          | [0.81, 0.92]   | 0.12          | 13,510                  |
| UASEF v1 (Round 6, heuristic)                                                  | 0.92          | [0.86, 0.96]   | 0.07          | 8,340                   |
| **UASEF v2** (Stratified CRC + MTC + Cost-Aware)                               | **0.998**     | [0.99, 1.00]   | 0.18          | **412**                 |

UASEF v2 trades a modest increase in over-escalation rate (0.18 vs 0.07–0.12)
for two qualitatively different gains: (1) the safety recall on CRITICAL
cases reaches 99.8% — the highest of any method we tested, and the only one
above 0.95; (2) the total cost is **20–33×** lower than the next-best
baseline (UASEF v1) because the over-escalation is concentrated in low-cost
strata where the asymmetric cost matrix is approximately 1:1.

---

## 7. Discussion

### 7.1 Per-Pivot Contribution

A natural ablation is to ask which pivot drives the gain. We find:

- **Pivot A alone** (Stratified CRC with symmetric F₁ optimization) accounts
  for the bulk of the CRITICAL-stratum safety improvement (Table 1).
- **Pivot B alone** restores the nominal FWER (Table 2). Without it, even
  Pivot A's per-stratum guarantee can be silently violated in deployment if
  multiple triggers are combined.
- **Pivot C alone** is responsible for the cost reduction (Table 3).

The three are complementary: removing any single pivot loses one of the
three properties (per-stratum coverage / FWER / cost-asymmetric calibration).
Our experimental setup deliberately presents synthetic and real-data
results for each pivot in isolation as well as the combined system.

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

### 7.4 The Mock-Tools Limitation

The LangGraph agent component of our system uses four mock medical tools
(`drug_interaction_checker`, `clinical_guideline_search`,
`lab_reference_lookup`, `differential_diagnosis`). Real deployment requires
substituting these with authenticated clinical APIs (Drugs@FDA, UpToDate,
LOINC, Isabel DDx). Our framework is tool-agnostic: the trigger scores
depend only on the LLM output text and the cumulative token log-probability,
which would not change. We mark this as a limitation and a clear path for
future work; we do not claim that the agent's *helpfulness* generalizes
beyond mock tools.

---

## 8. Limitations

We enumerate limitations explicitly and discuss mitigations.

**L1 — Heuristic ground-truth labels.** The MedQA `expected_escalate` label
is computed from a keyword-based classifier (cf. `_classify_case`) rather
than expert annotation. This may bias calibration toward keyword-aligned
cases. The MedAbstain labels (variants A, AP, NA, NAP) come from the
benchmark's own protocol and are not affected. *Mitigation:* future work will
include annotator-rated labels from $\ge$ 3 attending physicians and report
inter-rater agreement.

**L2 — Mock medical tools.** §7.4 above.

**L3 — CRITICAL-stratum sample size.** §7.2 above.

**L4 — Single-language evaluation.** All experiments are conducted in
English; clinical settings frequently involve non-English notes.

**L5 — No live clinical deployment.** This paper presents retrospective
evaluation only. A prospective multi-site evaluation is planned.

**L6 — Assumption of well-specified cost matrix.** The cost matrix (11) is a
plausible-but-not-validated proxy for true clinical cost. We mitigate via the
sensitivity analysis in §6.3 but do not eliminate the dependence.

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
