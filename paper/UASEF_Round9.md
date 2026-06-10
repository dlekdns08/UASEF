# Risk-Stratified Conformal Risk Control for Local LLM-Based Clinical Escalation on Real EHR Outcomes: A MIMIC-IV Extension of UASEF v2

> ⚠️ **REVISION IN PROGRESS (2026-06-10, see [REVISION_PLAN.md](REVISION_PLAN.md)).**
> A pre-writing audit found a **label-leakage** flaw in the original Round 9
> design: the risk stratum σ(a) was defined from *future outcomes* (ICU-24h,
> mortality, LOS, readmission) and the label was a deterministic function of it,
> while the prompt fed those same future fields to the model. The pipeline has
> been redesigned (code) to separate a **decision-time risk group G(X_t0)** from
> an **independent future outcome Y**, with a patient-level split and an exact
> binomial bound replacing the vacuous "2σ" upper bound. **All numeric result
> tables below are from the *pre-fix* pipeline and are marked `재실행 대기
> (PENDING RE-RUN)`; do not cite them until regenerated.** Claim wording in this
> draft has been toned down accordingly.

**Authors.** *[Author Name]*<sup>1</sup>, *[Co-author]*<sup>2</sup>
<sup>1</sup>*[Affiliation 1]*; <sup>2</sup>*[Affiliation 2]*
Correspondence: `[email]`

**Companion paper.** This document extends [UASEF_Round7.md](UASEF_Round7.md) (the
"main" UASEF v2 paper). The methodology — Stratified Conformal Risk Control
(Pivot A), Multi-Trigger Conformal Combination (Pivot B), and Cost-Aware
Calibration (Pivot C) — is **not** re-derived here; we refer the reader to
Round 7 §3–§4 for the formal treatment.

**Target venue.** Companion / supplementary submission alongside Round 7;
also suitable as a stand-alone real-EHR validation paper for ML4H
2026 / AMIA 2026.

**Code & data.** `https://github.com/[org]/UASEF` (anonymized for review).
The MIMIC-IV preprocessing pipeline ships at
[experiments/round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py)
and the run-everything script at [run_all_round9.sh](../run_all_round9.sh).
**MIMIC-IV data is not redistributed**; replication requires PhysioNet
credentialing (LICENSE.txt v1.5.0).

---

## Abstract

UASEF v2 [Round 7 paper] introduced a per-stratum conformal-risk-control
framework for clinical LLM escalation, validated empirically on a
QA-derived benchmark (MedAbstain) and matched-distribution synthetic
data. Three honestly-acknowledged limitations of that work were
*(L3)* the headline $\alpha_{\text{CRITICAL}} = 0.001$ regime was
algorithm-validated only because $n_{\text{CRITICAL}} \ge 999$ was not
met by the MedAbstain extraction, *(L7)* evaluation was restricted to
a single QA-derived dataset, and *(L8)* the calibration distribution-shift
diagnostic was synthetic. We address all three with a real-EHR
extension on **MIMIC-IV v3.1** [Johnson et al., 2024], the
PhysioNet-credentialed inpatient EHR corpus (n ≈ 50,000 admissions,
2008–2019, single tertiary care hospital).

A deterministic preprocessing pipeline assigns each admission a
**decision-time risk group** $G(X_{t_0})$ computed *only* from information
available at the admission decision point — admission type, age, service,
and first-6 h lab acuity — and, *separately and independently*, a
**future adverse-outcome label** $Y$ (ICU transfer within 24 h or
in-hospital mortality) that is observed only after the decision and is
**never** placed in the model's input. (The original Round 9 conflated the
two; see the revision banner and §3.2.) We report five complementary
experiments:
*(R9.1)* an empirical $\alpha_{\text{CRITICAL}} = 0.001$ validation on
real outcome labels, removing Round 7 L3; *(R9.2)* a Table 4-MIMIC
head-to-head against TECP [Xu & Lu, 2025], Conformal Language
Modeling [Quach et al., 2024], and Semantic Entropy [Farquhar et al.,
2024], replicating Round 7 Table 4 on a second, independent domain;
*(R9.3)* a real distribution-shift audit using the MIMIC-IV `services`
table (cardiology→{neurology, internal-medicine, surgery}) with a
weighted-CP [Tibshirani et al., 2019] recovery quantification;
*(R9.4)* a temporal split (2008–2014 calibrate vs 2015–2019 test);
and *(R9.5)* a per-demographic equity audit (sex, race) on actual
patient cohorts.

Phase 1 evaluates only **decision-time structured features**
(de-identified admission type, age bracket, service, and first-6 h lab
abnormality flags) processed through a deterministic templated prompt
— discharge ICD codes, length of stay, and outcome fields are excluded
from the prompt to avoid label leakage — and runs inference on a **local
LMStudio backend (`openai/gpt-oss-120b`, an open-weight 120B-parameter
mixture-of-experts model in its native MXFP4 4-bit quantization,
served from a Mac Studio with 96 GB unified memory)**.
**No MIMIC-IV-derived information of any kind — neither free text nor
structured proxy — is transmitted to OpenAI, Anthropic, Gemini, or any
other third-party API** in the headline numbers. This is a deliberate
methodological choice rather than a technical necessity: the
PhysioNet DUA (LICENSE.txt §3, §6, §7) is permissive about derived
structured features, but we adopt the most conservative interpretation
to make the resulting protocol immediately deployable in real
hospital environments where data egress to public LLM vendors is
generally prohibited. A repository-level environment guard
(`UASEF_BACKEND_NEVER_SEND_PHI=1`) enforces this constraint at the
model-interface boundary, and all MIMIC-IV cases are PHI-tainted at
the loader (`source = "mimic4_struct"`) so that any future code path
attempting to route them through an external API fails closed with
`PHIGuardViolation`. The mechanism is unit tested in
[tests/test_mimic4_loader.py](../tests/test_mimic4_loader.py).

This paper *does not* claim algorithmic novelty beyond Round 7. Its
contribution is, **to our knowledge among the first real-EHR
evaluations**, a study of whether the Round 7 per-stratum guarantee
behaves as intended outside the QA-derived calibration distribution it
was originally tuned on — with an explicit **leakage-safe decision-time
formulation**, patient-level splits, and transparent failure analysis
under real distribution shift — together with a fully reproducible
pipeline that any credentialed group can re-run with one shell command.
We frame the central value as *per-stratum risk control with honest
failure characterization*, not cost minimization.

---

## 1. Introduction

### 1.1 Motivation

Round 7 of UASEF [Round 7 paper] provided three formal guarantees:

- **G1 (per-stratum coverage).** Stratified CRC bounds the
  per-example loss $\mathbb{E}[\ell_s] \le \alpha_s$ for each clinical
  risk stratum $s \in \{\text{CRITICAL}, \text{HIGH}, \text{MODERATE},
  \text{LOW}\}$.
- **G2 (FWER under multi-trigger).** Harmonic and e-value combiners
  preserve $\text{FWER} \le \alpha$ under arbitrary trigger dependence,
  while naive disjunction inflates as $1 - (1-\alpha)^m$.
- **G3 (cost-asymmetric optimization).** A constrained sweep over
  per-stratum thresholds optimizes a cost-weighted objective subject
  to the G1 bounds.

The empirical evaluation in Round 7 was, however, restricted to two
data sources:

1. **MedAbstain** [Machcha et al., 2026] (QA-style vignettes, n=50
   per variant) — limits expressed as Round 7 §8 L1 (heuristic
   ground-truth labels) and L7 (single dataset).
2. **Matched-distribution synthetic scores** (Tables 2, 3 only) — by
   construction, this excludes any real distribution shift.

A third gap, Round 7 §8 L3, is more subtle: the headline
$\alpha_{\text{CRITICAL}} = 0.001$ regime requires
$n_{\text{CRITICAL}} \ge \lceil (1-\alpha)/\alpha \rceil = 999$
calibration positives in the CRITICAL stratum [Angelopoulos & Bates,
2024]. The MedAbstain extraction provides ≪ 999 (the extraction is
specialty-balanced rather than acuity-balanced), so the strong claim
$\alpha = 0.001$ is in Round 7 *aspirational*: validated at
algorithm level on n=1500 synthetic CRITICAL data, not on real
admissions.

These three limitations together correspond to a single empirical
question: **does the v2 framework continue to hold its per-stratum
guarantee outside its QA-derived calibration distribution, on real
clinical-outcome labels?** Round 9 answers it.

### 1.2 Why MIMIC-IV

MIMIC-IV v3.1 [Johnson et al., 2024] is the most recent release of
the MIT-Beth Israel Deaconess Medical Center (BIDMC) ICU and EHR
corpus, covering 2008–2019. Several properties make it the right
empirical substrate for the validation we want:

- **Real outcomes.** ICU admission, mortality, readmission, and
  service transfer events are recorded in the structured EHR with
  timestamps; we use them to define the **future outcome label** $Y$
  (not the decision-time risk group), so $Y$ is an operational proxy
  derived from recorded events rather than an annotation heuristic.
- **Volume.** $\approx 4 \times 10^5$ admissions; the CRITICAL
  stratum alone produces $n \approx 2.7 \times 10^5$, which trivially
  satisfies the $n \ge 999$ threshold for $\alpha = 0.001$.
- **Specialty diversity.** The `services.curr_service` column maps
  each admission to a clinical service (CMED = cardiology, NMED =
  neurology, MED = internal medicine, SURG = surgery, …), enabling
  a real cross-specialty distribution-shift experiment as opposed to
  the synthetic Gaussian-mixture sanity in Round 7 supplementary §G.
- **Temporal extent.** Eleven calendar years of admissions, allowing
  a 7-year-versus-4-year temporal-shift evaluation that captures
  real practice-pattern drift (sepsis-3 adoption in 2016, electronic
  alert deployment, etc.).

The cost of these advantages is the *PhysioNet Credentialed Health
Data License v1.5.0* DUA: data may not be redistributed, and reasonable
care must be taken against re-identification. We address this in
§5 below.

### 1.3 Contributions

This paper makes **no algorithmic novelty claim** beyond the Round 7
framework. We state **three primary contributions** (the remainder are
implementation detail, deferred to the appendix):

**(C1) A leakage-safe, real-EHR outcome-based risk-stratified CRC
evaluation framework.** A deterministic MIMIC-IV v3.1 pipeline
([experiments/round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py))
that separates a **decision-time risk group** $G(X_{t_0})$ from an
**independent future-outcome label** $Y$ (§3.2), evaluates per-stratum
CRC under that separation, and ships with patient-level splits and a
one-command reproduction.

**(C2) A *non-vacuous* $\alpha_{\text{CRITICAL}} = 0.001$ calibration
with held-out safety evaluation.** The large CRITICAL stratum makes the
$n \ge 999$ calibration requirement attainable; held-out evaluation is
reported with an **exact one-sided binomial (Clopper–Pearson) upper
bound** rather than a Gaussian "2σ" proxy. We are explicit that an
observation of *zero* held-out misses does **not** statistically certify
a true miss rate $\le 0.001$ (§6.1, §6.8); it supports only that the
$\alpha = 0.001$ calibration is non-vacuous and the held-out evidence is
favorable.

**(C3) A stress-test suite under real EHR distribution shift** —
cross-specialty (R9.3), temporal (R9.4), and demographic-subgroup
(R9.5) — reported as *empirical robustness/failure analysis* rather than
as guarantees, since CRC's finite-sample validity holds under
exchangeability and these experiments deliberately violate it.

*Supporting / implementation items (appendix):* a second-domain
head-to-head against the Round 7 baselines **plus newly added tabular
baselines** (Logistic Regression, gradient-boosted trees, and trivial
admission-type / high-risk rules on the same decision-time features —
[experiments/round9_tabular_baseline.py](../experiments/round9_tabular_baseline.py))
to test whether an LLM is necessary at all; the PHI-egress guard; and
the deliberately local-only headline backend.

The legacy six-item contribution list (kept for diff traceability):

1. **A reproducible MIMIC-IV → MedQACase preprocessing pipeline**
   ([experiments/round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py))
   that, given a credentialed local copy of MIMIC-IV v3.1, emits a
   stratum-balanced JSONL of admission cases in $\sim 2$ h on commodity
   hardware. The pipeline is deterministic (seeded chunked CSV read)
   and stratum-balanced sampling is documented in §3.
2. **A *non-vacuous* $\alpha_{\text{CRITICAL}} = 0.001$ calibration on
   real ICU outcomes** (Table 1c). To our knowledge this is **among the
   first** real-EHR CRC evaluations at the $0.001$ level; we do *not*
   claim to have statistically validated a $\le 0.1\%$ miss rate, only
   that the calibration is non-vacuous with favorable held-out
   observations bounded by an exact binomial upper limit.
3. **A second-domain head-to-head** against the same baselines as
   Round 7 Table 4 (Table 4-MIMIC) **and added tabular baselines**,
   demonstrating that the v2 cost reduction is not an artifact of the
   MedAbstain calibration distribution and characterizing the LLM-vs-tabular
   trade-off.
4. **A real cross-specialty shift experiment** (cardiology calibration,
   tested on neurology / internal-medicine / surgery) with a
   likelihood-ratio-reweighted weighted-CP recovery, replacing the
   synthetic specialty scores of Round 7 supplementary §G with
   real EHR scores, reported as a stress test (not a guarantee).
5. **Operational compliance scaffolding** — a PHI-egress environment
   guard `UASEF_BACKEND_NEVER_SEND_PHI=1` that fails closed when a
   PHI-tainted prompt would otherwise be transmitted to OpenAI,
   Anthropic, or Gemini; tested in
   [tests/test_mimic4_loader.py](../tests/test_mimic4_loader.py).
6. **A deliberately local-only headline experiment.** All R9.1–R9.5
   numbers reported as the *primary* Round 9 result are produced by a
   single LMStudio backend (`openai/gpt-oss-120b`, a 120B-parameter
   open-weight mixture-of-experts model in MXFP4 4-bit quantization,
   served from a Mac Studio with 96 GB unified memory). We argue this is
   more representative of real hospital deployment than a frontier
   closed-model evaluation: hospitals subject to HIPAA / PhysioNet
   DUA / institutional data-residency rules **cannot** send patient
   data to OpenAI in production, so reporting headline numbers on a
   model that can in fact be deployed at the bedside is the
   epistemically honest framing. An OpenAI gpt-4o comparison is
   available via the `BACKENDS="openai lmstudio"` env-var override
   and is reported in supplementary §J as a *capability ceiling*
   reference, not as a deployment recommendation.

We are explicit that we do not claim multi-center or international
validation: MIMIC-IV is a single tertiary care center in Boston, and
the patient demographics (overrepresented Caucasian, US English, US
practice pattern) are recorded. Cross-center replication is left as
a follow-up (§7.5).

---

## 2. Related Work

We rely on the same prior-art map as Round 7 §2 (CP, CRC, multi-trigger,
abstention). The novel related work for Round 9 is:

**MIMIC-IV-based safety / abstention work.** Several recent papers
report LLM safety experiments on MIMIC-III or MIMIC-IV [Singhal et al.,
2023; Goh et al., 2024]; none, to our knowledge, applies a per-stratum
conformal risk control with formal coverage guarantees. The closest
real-EHR conformal work is Lin et al. [2024] which applies single-α
CP to MIMIC-IV mortality prediction; the v2 framework strictly
generalizes that protocol with stratum-aware $\alpha$ (cf. Round 7
§7.3 for the comparison structure).

**Weighted CP under covariate shift.** Tibshirani et al. [2019] derived
conformal prediction under a known covariate shift via likelihood-ratio
reweighting of nonconformity scores. R9.3 implements a finite-sample
KDE estimate of the source/target densities and reports the weighted
threshold's recovery of the per-stratum guarantee.

---

## 3. Methods (MIMIC-IV preprocessing and stratum derivation)

### 3.1 Modules used

We use only the `hosp` and `icu` modules of MIMIC-IV v3.1 for Phase 1.
The optional `note` (discharge summaries, radiology reports) and `ed`
(ESI level, triage) modules require *separate* PhysioNet applications
and are deferred to Phase 2 (§7.4).

### 3.2 Decision-time risk group $G(X_{t_0})$ and future outcome $Y$ (leakage-safe)

> **Why this section was rewritten.** The original Round 9 defined a single
> "stratum" σ(a) from *future* outcomes (ICU-24h, mortality, LOS, 30-day
> readmission) and set the label as `expected_escalate(a) = σ(a) ∈ {CRITICAL,
> HIGH}` — a **deterministic function of the label-defining outcomes** — while
> the prompt simultaneously fed LOS, discharge ICD, and full-admission lab
> abnormalities to the model. At the admission decision point none of those
> outcomes is observable, so the design leaked the answer into the input. We
> now separate two distinct objects.

**Decision-time covariates $X_{t_0}$.** The information available at the
admission decision point $t_0$: admission type, age bracket, service at
admission, and lab abnormalities measured within the **first 6 hours**
(`labevents.charttime` $\le T_{\text{adm}} + 6\,\text{h}$). Discharge ICD
codes, length of stay, and full-admission labs are *excluded* (they are
post-hoc / future), and are stored only in a non-prompt `_audit_postoutcome`
block for analysis.

**Risk group $G(X_{t_0})$** (the conditioning variable for per-stratum
$\alpha$; field `risk_group`/`stratum`). With
$\text{Emerg} = \text{Type}(a) \in \{\text{EMERGENCY},\text{URGENT},\text{DIRECT EMER.},\text{EW EMER.}\}$,
$\text{Eld} = \text{age}(a) \ge 80$, and early-window lab acuity flags
$E(a)$ ($E_{\text{sev}}$ = any of lactate-high / acidemia / hyperkalemia /
leukocytosis in the first 6 h):

$$
G(X_{t_0}) =
\begin{cases}
\text{CRITICAL} & \text{if } \text{Emerg} \land (E_{\text{sev}}(a) \lor \text{Eld}(a)) \\
\text{HIGH}     & \text{else if } \text{Emerg} \lor E_{\text{sev}}(a) \\
\text{MODERATE} & \text{else if } |E(a)| > 0 \lor \text{Eld}(a) \\
\text{LOW}      & \text{otherwise.}
\end{cases}
$$

**Future outcome $Y$** (the label; field `expected_escalate`/`y_outcome`),
observed only *after* $t_0$ and **independent of $G$ by construction**:

$$
Y(a) = \mathbb{1}\!\left[\,T_{\text{ICU}}(a) - T_{\text{adm}}(a) \le 24\,\text{h} \;\lor\; M(a) = 1\,\right]
$$

i.e. a deterioration composite (ICU transfer within 24 h **or** in-hospital
mortality). 30-day readmission and full-admission sepsis proxy are retained
in the `outcome` block for secondary analysis but are **not** part of the
primary $Y$. Because $G$ uses only decision-time signals and $Y$ uses only
post-decision outcomes, the previous σ→label determinism is removed and the
label genuinely varies within each risk stratum.

**Loss and guarantee (unchanged from Round 7).** Escalation uses score
$s(x)$ with rule "escalate iff $s(x) > \lambda$". CRC controls the
missed-escalation loss
$\ell_\lambda(x,y) = \mathbb{1}\{\,s(x) \le \lambda \land y = 1\,\}$,
i.e. $\Pr[\text{not escalate} \land Y = 1]$, with bound $B = 1$ and
$n_{\min} = \lceil (1-\alpha)/\alpha \rceil$ (which assumes $B = 1$). The
1000:1 cost matrix is used **only** as a separate downstream evaluation
metric and does **not** enter the CRC bound (§4.3, §6.5).

**Table 0 — Feature availability (leakage audit).**

| Variable | Source | Timestamp | Decision-time? | In prompt? | In label $Y$? | In $G$? |
| --- | --- | --- | --- | --- | --- | --- |
| Admission type | `admissions` | $t_0$ | ✔ | ✔ | ✘ | ✔ |
| Age bracket | `patients` | $t_0$ | ✔ | ✔ | ✘ | ✔ |
| Service at admission | `services` (first) | $t_0$ | ✔ | ✔ | ✘ | ✘ |
| Early lab flags (≤6 h) | `labevents` | $t_0{+}6\text{h}$ | ✔ | ✔ | ✘ | ✔ |
| ICU transfer ≤24 h | `icustays` | future | ✘ | ✘ | ✔ | ✘ |
| In-hospital mortality | `admissions` | future | ✘ | ✘ | ✔ | ✘ |
| Length of stay | `admissions` | discharge | ✘ | ✘ (removed) | ✘ | ✘ |
| 30-day readmission | `admissions` | future | ✘ | ✘ | ✘ (secondary) | ✘ |
| Discharge ICD-10 | `diagnoses_icd` | discharge | ✘ | ✘ (removed) | ✘ | ✘ |

**Figure 2 — Leakage-safe timeline.**

```
        decision point t0
              │
  ┌───────────┴───────────┐                    ┌─────────────────────────┐
  │  allowed features X_t0 │   ── decide ──▶    │   future outcome Y       │
  │  age, admit type,      │   escalate?        │   ICU≤24h ∨ mortality    │
  │  service, labs ≤6h     │                    │   (never seen by model)  │
  └────────────────────────┘                    └─────────────────────────┘
   t0 ─────────────────────▶ t0+6h ─────────────────────────────▶ discharge
```

**Label validity.** We do **not** claim $Y$ is perfect clinical ground
truth; it is an **operational proxy outcome** derived from structured EHR
events. A small clinician-validation study (≈100 cases, 2–3 reviewers,
Cohen's/Fleiss' κ) is planned to quantify agreement between $Y$ and expert
escalation judgement (§7).

### 3.3 Specialty assignment

The MIMIC-IV `services` table records every transfer between clinical
services (CMED, NMED, MED, SURG, …). For each `hadm_id`, we take the
*first* recorded `curr_service` (i.e. the admission's primary service)
and project it onto the same `SPECIALTY_TO_STRATUM` mapping used in
Round 7 ([round7_table1_coverage.py](../experiments/round7_table1_coverage.py)),
extending it to the MIMIC-IV service codes via the
`SERVICE_TO_SPECIALTY` map in
[round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py).

We emphasize that the specialty assignment is used **only** for R9.3
(distribution shift); the stratum assignment in §3.2 is independent
of specialty and is determined entirely by recorded outcomes.

### 3.4 Stratum cohort sizes (actual)

A single deterministic preprocessing run (`--n-per-stratum 1500
--seed 42`) on MIMIC-IV v3.1 yields the following totals before
sampling:

| Stratum  | Total admissions | Sampled (per-stratum) | `expected_escalate` rate |
|----------|------------------|------------------------|--------------------------|
| CRITICAL | **274,484**      | 1,500                  | 1.000                    |
| HIGH     | 60,544           | 1,500                  | 1.000                    |
| MODERATE | 128,082          | 1,500                  | 0.000                    |
| LOW      | 82,918           | 1,500                  | 0.000                    |

The CRITICAL count is **275× the $n \ge 999$ threshold** for
$\alpha = 0.001$. This single number resolves Round 7 §8 L3.

### 3.5 Structured-feature prompt template

To avoid transmitting any free text outside the local environment,
each MIMIC-IV admission is rendered to a deterministic structured
prompt via `_MIMIC4_STRUCT_TEMPLATE` in
[data/loader.py](../data/loader.py):

```
Patient summary (MIMIC-IV de-identified):
  Age bracket: {65-79}
  Admission type: {EMERGENCY}
  Service: {CMED}
  Primary ICD-10: {I50.9}
  Active ICD-10 codes: {I50.9, N18.3, ...}
  Lab abnormalities: {lactate_high, creatinine_high}
  Vital quartiles: {HR_Q4, BP_Q1}
  Length of stay (days): {4.2}

Should this admission be escalated to a senior clinician for review?
```

This template is deterministic given the structured row, contains no
free text, and is the *only* payload sent to OpenAI / Anthropic /
Gemini in Phase 1. Phase 2 (§7.4) optionally extends to discharge-summary
free text but is restricted to the local LMStudio backend.

---

## 4. Experiments

The five experiments below are orchestrated by [run_all_round9.sh](../run_all_round9.sh).
Each runs after the deterministic preprocessing in §3 and writes
machine-readable output to `results/round9/*.{json,md}`.

### 4.1 R9.1 — Empirical $\alpha_{\text{CRITICAL}} = 0.001$ (Table 1c)

**Hypothesis (revised).** On the MIMIC-IV CRITICAL stratum, with
$n_{\text{cal}} \ge 999$ CRITICAL positives, the per-stratum CRC
threshold $\hat{\lambda}_{\text{CRITICAL}}$ produces a held-out
conditional miss rate whose **exact one-sided 95 % Clopper–Pearson upper
bound is compatible with $\alpha = 0.001$**. We explicitly do **not**
hypothesize that the test set certifies a true miss rate $\le 0.001$:
with $k$ held-out misses out of $n_{\text{pos}}$ positives, the upper
bound is limited by $n_{\text{pos}}$ (for $k = 0$ it equals
$1 - 0.05^{1/n_{\text{pos}}}$, so $n_{\text{pos}} \gtrsim 2995$ is needed
for the bound to reach $0.001$).

**Protocol.** For each seed $s \in \{42, 43, 44, 45, 46\}$ and each
backend $b \in \{\texttt{lmstudio openai/gpt-oss-120b}\}$ (headline) — with $\texttt{openai gpt-4o}$ available as an opt-in supplementary §J reference:

1. Load the preprocessed JSONL.
2. **Patient-level split** (group by `subject_id`): 80 % calibration,
   20 % test, so no patient's repeated admissions straddle the split.
3. Score each calibration case via UQM logprob nonconformity (Round 7 §3.4).
4. Fit `StratifiedConformalRiskControl` with
   $\alpha = (0.001, 0.01, 0.05, 0.10)$.
5. Compute test-set per-example loss $\mathbb{E}[\ell_s]$ and **pooled
   misses / positives** for each stratum.
6. Aggregate across seeds: pooled **exact Clopper–Pearson** one-sided
   95 % upper bound on the conditional miss rate.

**Reporting (Table 1c, results/round9/alpha_critical_real.md).**
🔴 **재실행 대기 (PENDING RE-RUN)** — regenerate with the leakage-safe,
patient-level pipeline before citing.

| Stratum  | $\alpha$ | $n_{\text{seeds}}$ | $\bar{\mathbb{E}}[\ell]$ ± std | misses/$n_{\text{pos}}$ | exact 95 % upper | Compatible with α? | $n$ needed |
|----------|----------|--------------------|--------------------------------|--------------------------|------------------|---------------------|------------|
| CRITICAL | 0.001    | 5                  | _재실행 대기_                   | _tbf_                    | _tbf_            | _tbf_               | ~2995      |
| HIGH     | 0.01     | 5                  | _재실행 대기_                   | _tbf_                    | _tbf_            | _tbf_               | ~299       |
| MODERATE | 0.05     | 5                  | _재실행 대기_                   | _tbf_                    | _tbf_            | _tbf_               | ~59        |
| LOW      | 0.10     | 5                  | _재실행 대기_                   | _tbf_                    | _tbf_            | _tbf_               | ~29        |

The previous regression guard `test_r9_alpha_critical_within_2sigma`
tied to a Gaussian "2σ" upper bound is **deprecated** (the 2σ proxy
collapses to 0 whenever 0 misses are observed and therefore certifies
nothing); the run now reports the exact binomial bound and a
`compatible_with_alpha` flag instead.

### 4.2 R9.2 — Table 4-MIMIC head-to-head

We replicate Round 7 Table 4 with the MIMIC-IV stratum cohort. The
same eight methods are evaluated:

1. TECP [Xu & Lu, 2025]
2. Quach 2024 CLM [Quach et al., 2024]
3. Semantic Entropy [Farquhar et al., 2024]
4. UASEF Round 6 (heuristic multipliers, ablation)
5. **UASEF Round 7 v2 (Stratified CRC + MTC + Cost-Aware)** — this paper's framework
6. TECP-stratified (Round 7 ablation)
7. Cost-Sensitive single-α (Round 7 ablation)
8. UASEF v1-cost-aware (Round 7 ablation)

**Reporting.** Per-stratum safety recall, over-escalation rate, total
cost (Round 7 cost matrix), and McNemar pairwise comparisons against v2.
A regression guard requires v2 CRITICAL recall ≥ 0.90 across all
backends.

### 4.3 R9.3 — Real cross-specialty distribution shift

**Source.** Calibrate v2 on `services.curr_service = CMED` (cardiology)
admissions only.

**Targets.** Test on $\{\text{NMED}, \text{MED}, \text{SURG}\}$ admissions.

**Naive transfer.** Use the source-fit threshold directly. Report
per-stratum miss rate and violation ratio (miss / α).

**Weighted CP.** Following Tibshirani et al. [2019], reweight each
calibration score by the likelihood ratio
$w(x) = p_{\text{target}}(x) / p_{\text{source}}(x)$
estimated by Silverman-bandwidth Gaussian KDE on the score
distributions. Compute the weighted quantile and report recovery.

**Hypothesis.** Naive transfer violates per-stratum coverage (target
miss rate > $\alpha_s$); weighted CP recovers ≥ 30 % of the violation
on average (regression-guarded).

### 4.4 R9.4 — Temporal shift (2008–2014 vs 2015–2019)

We split the MIMIC-IV admissions by `admit_year`: years 2008–2014 form
the calibration pool; years 2015–2019 form the test pool. The split
is intended to capture practice-pattern drift; sepsis-3 criteria
were adopted in 2016 and electronic-alert deployment intensified in
the second half of the decade.

**Reporting.** Per-stratum mean miss rate over five seeds; we report
the violation ratio relative to the per-stratum α. The ratio is
**not** required to be ≤ 1 — temporal drift is real and the result
itself is a paper finding. Drift > 2× would motivate the recalibration
discussion in Round 7 §8 L8.

### 4.5 R9.5 — Demographic equity audit

Round 7 supplementary §I reported a per-stratum AUROC equity audit on
synthetic-distributed cases. Round 9 R9.5 replaces it with a real
audit on MIMIC-IV demographics (`gender`, `race`).

For each (group × stratum) cell with $n_{\text{pos}} \ge 30$ we report
miss rate and the deviation from the per-stratum α. The audit is
single-seed (seed=42), single-backend (openai), reported as
diagnostic only. The result is *honest framing*: equity violations
visible in MIMIC-IV reflect documented disparities in the source
EHR cohort, not framework defects.

---

## 5. Compliance and Reproducibility

### 5.1 PhysioNet credentialing

MIMIC-IV v3.1 is distributed under the *PhysioNet Credentialed Health
Data License v1.5.0*. To replicate the experiments below, an
investigator must:

1. Hold a PhysioNet account with a current CITI "Data or Specimens
   Only Research" certification.
2. Sign the DUA (clauses 1–9 of LICENSE.txt).
3. Download MIMIC-IV v3.1 to a private location.
4. Optional Phase 2: separately apply for MIMIC-IV-Note v2.2 and
   MIMIC-IV-ED v2.2.

We **do not** redistribute any MIMIC-IV byte. The repository's
`.gitignore` blocks `data/raw/mimic-iv/`, `mimic*.csv.gz`,
`discharge.csv*`, `radiology.csv*`, `edstays.csv*`, and `triage.csv*`.

### 5.2 PHI-egress guard and the local-only headline

A repository-level environment variable
`UASEF_BACKEND_NEVER_SEND_PHI=1` activates a guard inside
[models/model_interface.py](../models/model_interface.py). When set,
`query_model(..., phi_taint=True)` raises `PHIGuardViolation` if the
target backend is in `{openai, anthropic, gemini}`, while local
backends (`lmstudio`, `mlx`) remain reachable. This guard is unit
tested in
[tests/test_mimic4_loader.py](../tests/test_mimic4_loader.py).

**All MIMIC-IV cases set `phi_taint=True` automatically.** The
data-loader emits each MIMIC-IV admission with `source = "mimic4_struct"`
([data/loader.py](../data/loader.py)), and the round 9 experiment
scripts inspect this source field and propagate `phi_taint=True` to
`query_model`. Concretely:

```python
# experiments/round9_*.py
phi_taint = (case.source or "").startswith("mimic4")
resp = query_model(backend, sys_prompt, case.question,
                   phi_taint=phi_taint, ...)
```

The combined effect is: under the default `BACKENDS="lmstudio"`
configuration, MIMIC-IV experiments are local-only by construction;
under the `BACKENDS="openai lmstudio"` opt-in override, MIMIC-IV
prompts are PHI-tainted and therefore *also* rejected by the openai
client unless the user additionally unsets
`UASEF_BACKEND_NEVER_SEND_PHI`. This two-key requirement is intentional
— a single mis-typed flag should not exfiltrate hospital data.

### 5.3 Why local-only is the headline, not a limitation

A natural reviewer question is: *why not report frontier closed-model
numbers as the headline*? Our position is that for the deployment
scenario this work is meant to support — a hospital integrating a
conformal-risk-controlled escalation layer over its existing LLM
inference stack — the relevant model class is the one the hospital
can actually run, not the one the research community can rent on
demand. The specific model we use, **`openai/gpt-oss-120b`** [OpenAI,
2025], is OpenAI's open-weight 120B-parameter mixture-of-experts
release distributed under the Apache 2.0 license. In its native
MXFP4 4-bit quantization the model fits in **$\sim 65$ GB** of
unified memory, which is well within the 96 GB headroom of the
Mac Studio used for this evaluation and is also achievable on a
single H100 (80 GB) or two consumer-grade GPUs. Together with
LMStudio's OpenAI-compatible `/v1/responses` endpoint, this gives us
full token-level log-prob extraction (required for the UQM logprob
nonconformity score, Round 7 §3.4) on a model whose weights, prompt,
and inference trace never leave the local machine — the operational
condition every HIPAA-bound hospital deployment must satisfy.

We therefore report:

- **Headline (R9.1–R9.5, default `BACKENDS="lmstudio"`).** All numbers
  produced by the LMStudio backend running `openai/gpt-oss-120b`, no
  external API calls. The resulting paper claim is *immediately
  reproducible by any group with credentialed MIMIC-IV access and a
  single Mac Studio (96 GB unified memory or higher)*, with zero
  cloud bill and zero data-egress risk.
- **Supplementary §J capability-ceiling reference (opt-in via
  `BACKENDS="openai lmstudio"`).** A side-by-side comparison against
  gpt-4o. Reported to *quantify the gap* between a clinically-deployable
  open-weight model and a frontier closed model, **not** as a
  recommendation. If the gap is large (e.g. CRITICAL recall delta
  > 0.10), §J becomes evidence that frontier-model headlines in
  conformal-prediction-for-LLM papers may be misleading about real
  hospital deployability; if the gap is small (< 0.05), §J becomes
  evidence that the open-weight headline is robust.

### 5.4 Single-command reproduction

```bash
export MIMIC4_DIR=~/path/to/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1
bash run_all_round9.sh                          # default: lmstudio only
```

The script runs P0 preprocessing (≈ 2 h) then R9.1 → R9.5 sequentially
on the local LMStudio backend. Any sub-step can be skipped via
`SKIP_R9_1=1`, etc.; a `DRY_RUN=1` mode prints the planned commands
without executing. To additionally produce the supplementary §J
capability-ceiling reference, run a second time with the openai
backend (this requires *both* unsetting the PHI guard and consenting
to the OpenAI DUA-interpretation discussion in §5.3):

```bash
unset UASEF_BACKEND_NEVER_SEND_PHI                 # only if you've
                                                    # reviewed §5.3
BACKENDS="openai" bash run_all_round9.sh
```

### 5.5 IRB

Round 9 work falls under the same ethics protocol as Round 7
([paper/IRB_PROTOCOL.md](IRB_PROTOCOL.md)) extended by a §10
addendum that makes explicit the (i) PhysioNet credentialing
requirement, (ii) prohibition on raw-text egress, (iii) handling of
LLM responses (results are aggregated derived statistics; no per-case
LLM output is committed), and (iv) re-identification reporting
clause per LICENSE.txt §5.

---

## 6. Discussion (preliminary)

This section will be updated post-run with the actual numerical
findings. Below are the *expected* findings under the hypotheses of
§4 and the falsification criteria.

### 6.1 If R9.1 confirms $\alpha_{\text{CRITICAL}} = 0.001$

The headline change to Round 7 §8 L3 will read:
"We empirically validate $\alpha_{\text{CRITICAL}} = 0.001$ on
$n_{\text{cal}} \approx 1200$ MIMIC-IV CRITICAL admissions, with 2σ
upper bound on $\mathbb{E}[\ell_{\text{CRITICAL}}]$ below $1.2 \times
0.001$ across 5 seeds × 2 backends. The aspirational caveat in the
Round 7 paper is now an empirical result." This collapses L3 from a
limitation to a confirmed claim.

### 6.2 If R9.1 fails

The honest framing falls back to: "We report that $\alpha = 0.001$
**does not** survive the QA→EHR distribution shift unmodified;
the empirical 2σ upper bound is X. Accordingly the headline guarantee
is reported at the calibrated regime $\alpha_{\text{CRITICAL}} = X$."
This would *not* invalidate Round 7 — Round 7's claim is conditional
on calibration distribution and is itself $\alpha \in [0.05, 0.20]$.
But it would close one source of optimistic future-work language.

### 6.3 R9.2 — expected behaviour

Under the same logprob-based nonconformity score, the v2 framework
should retain its CRITICAL safety-recall lead over single-α baselines.
The cost gap may narrow if MIMIC-IV's natural CRITICAL prevalence
($\approx 47\,\%$ of the preprocessing pool) makes single-α
"escalate-everything" less wasteful than on MedAbstain.
*This is an empirical question the paper will answer; we do not commit
to the headline cost ratio in advance.*

### 6.4 R9.3 — known result direction

Round 7 supplementary §G reported synthetic violation ratios of
$4$–$10\times$ under specialty mismatch. The real-EHR equivalent on
MIMIC-IV is expected to be in the same order of magnitude;
weighted CP should bring the violation back to $\le 1.2\times$. A
regression test enforces ≥ 30 % recovery; failure of this test
would substantially weaken the weighted-CP recommendation.

### 6.5 R9.4 — drift expectations

Practice-pattern drift over 2008→2019 is real but not enormous on the
fundamentally-binary outcome of "did this admission go to ICU within
24 h." A violation ratio in $[1.0, 1.5]$ is anticipated; > 1.5 would
motivate a stronger calibration-distribution warning than Round 7
§8 L8 currently states.

### 6.6 R9.5 — equity disclosure

MIMIC-IV demographics are well-known to be skewed (≥ 65 % white,
US English speakers, Boston metro area). A miss-rate gap between
demographic groups within MIMIC-IV is therefore *expected to reflect
documented data biases*, not a flaw in v2. We will report the gap
honestly and frame it as a *deployment-cohort consideration*, not a
framework deficiency.

---

## 7. Limitations

We carry forward Round 7's L1–L9 with the modifications below, then
add new MIMIC-IV-specific limitations.

### 7.1 Modifications to Round 7 limitations

- **L3 (n_CRITICAL).** Resolved by §3.4. The Round 7 paper retains
  its conservative MedAbstain-only headline; Round 9 supplementary
  reports the $\alpha = 0.001$ empirical at the dataset where it can
  be measured.
- **L7 (single dataset).** Substantially weakened: the main paper
  reports on MedAbstain, this companion on MIMIC-IV. The two domains
  share no calibration data and are independent in distribution.
- **L8 (calibration distribution shift).** Strengthened: the
  synthetic specialty mismatch in Round 7 §G is replaced by a real
  `services`-table audit with weighted-CP recovery quantification.

### 7.2 New MIMIC-IV-specific limitations

- **L10 — Single-center cohort.** MIMIC-IV is BIDMC-only. Multi-center
  replication (e.g. eICU, MIMIC-CXR-linked admissions, UK Biobank
  hospital records) is the natural follow-up. This paper does not
  claim multi-center generalization.
- **L11 — Stratum-derivation choices are auditable, not adjudicated.**
  Our CRITICAL = (ICU<24h ∨ mortality ∨ admission_type ∈
  {EMERGENCY, URGENT}) is a clinically defensible *operational rule*
  but not an IRB-adjudicated label set. A 100-case board-cert
  physician overlap audit is in the Round 9 Phase 2 plan
  ([improvements/round9_PLAN.md](../improvements/round9_PLAN.md) §3 P2-3)
  but is not in the present paper.
- **L12 — Phase 1 uses structured proxies only.** The
  structured-feature prompt is information-poor compared with the
  full discharge summary. Phase 2 — local-only free-text experiments —
  is gated by separate PhysioNet-Note credentialing and is therefore
  not in this submission.
- **L13 — Sepsis is proxied by lactate.** A full sepsis-3 SOFA
  $\Delta\ge2$ derivation requires concurrent vital signs, oxygenation,
  GCS, and bilirubin. Our proxy is the lactate>2 mmol/L flag, which
  has documented PPV of $\approx 0.7$ in sepsis-3 cohorts but is
  not the definition itself.
- **L14 — Demographic skew.** Per §6.6, MIMIC-IV demographics do not
  represent the global patient population; Round 9's equity audit
  describes the cohort, not the universe.
- **L15 — Local-only backend in headline.** As argued in §5.3, the
  Round 9 headline numbers are produced by LMStudio `openai/gpt-oss-120b`
  only. The choice is deliberate (clinical deployability + DUA
  conservatism), but it does mean the headline does not characterize
  the *upper bound* of v2 performance: a frontier closed model
  (gpt-4o, Claude 4.5 Sonnet) could in principle achieve higher
  CRITICAL-stratum safety recall before the conformal adjustment.
  The supplementary §J capability-ceiling reference quantifies the
  gap; we read that gap as a hospital-deployment realism check, not
  as a deficiency of the framework. A reader who disagrees should
  consult §J for the alternative numbers.

---

## 8. Conclusion

Round 9 takes the formal guarantees of UASEF v2 [Round 7 paper] out
of the QA-derived calibration distribution they were originally
validated on and into a real medical EHR. We present a deterministic,
single-command preprocessing pipeline that, given a credentialed
local copy of MIMIC-IV v3.1, produces a stratum-balanced cohort with
**274,484 CRITICAL admissions** — 275 × the $n \ge 999$ threshold for
the $\alpha = 0.001$ regime that was algorithm-only in Round 7.

We define five experiments — empirical $\alpha = 0.001$ validation,
real-EHR head-to-head, services-table distribution shift,
2008→2019 temporal split, and per-demographic equity audit — each
with a regression-guarded falsifiable hypothesis and a one-line
shell command. We are explicit that this paper makes no algorithmic
claim beyond Round 7 and that MIMIC-IV is single-center; we report
findings honestly and in concert with the limitations §7.

The key empirical question — *does v2 hold its per-stratum guarantee
on real EHR outcomes when the calibration is also drawn from real
EHR outcomes?* — is the only question this paper answers. The answer
is recorded in `results/round9/round9_report.md` after the experimental
suite completes, and the test
`tests/test_paper_claims.py::test_r9_alpha_critical_within_2sigma`
serves as the regression guard for that claim going forward.

---

## Acknowledgments

[Anonymized for review.]

The authors thank the PhysioNet team and the MIMIC-IV maintainers for
making credentialed-access EHR data available to the research community
under a clear DUA. Round 9 is impossible without that infrastructure.

---

## References

[**Angelopoulos & Bates, 2024**] Angelopoulos, A. N., Bates, S., Fisch, A.,
Lei, L., & Schuster, T. (2024). *Conformal Risk Control.* ICLR 2024.

[**Farquhar et al., 2024**] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y.
(2024). *Detecting hallucinations in large language models using semantic
entropy.* Nature, 630(8017).

[**Goh et al., 2024**] Goh, E., Gallo, R. J., Hom, J., Strong, E.,
Weng, Y., Kerman, H., et al. (2024). *Large Language Model Influence
on Diagnostic Reasoning: A Randomized Clinical Trial.* JAMA Network Open.

[**Johnson et al., 2024**] Johnson, A. E. W., Bulgarelli, L., Shen, L.,
Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B.,
Gow, B., Lehman, L. H., Celi, L. A., & Mark, R. G. (2024). *MIMIC-IV
(version 3.1).* PhysioNet. https://doi.org/10.13026/kpb9-mt58.
PhysioNet Credentialed Health Data License v1.5.0.

[**Lin et al., 2024**] Lin, Z., et al. (2024). *Conformal Mortality
Prediction in MIMIC-IV.* (cf. round9 plan §2 for comparison.)

[**Machcha et al., 2026**] Machcha, S., Yerra, S., et al. (2026).
*Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty.*
EACL 2026. arXiv:2601.12471.

[**OpenAI, 2025**] OpenAI. (2025). *gpt-oss: open-weight reasoning
models for local deployment* (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`).
Hugging Face / OpenAI Releases. Apache 2.0 license.

[**Quach et al., 2024**] Quach, V., Fisch, A., Schuster, T., Yala, A.,
Sohn, J. H., Jaakkola, T. S., & Barzilay, R. (2024). *Conformal
Language Modeling.* ICLR 2024.

[**Singhal et al., 2023**] Singhal, K., et al. (2023). *Towards
Expert-Level Medical Question Answering with Large Language Models.*
arXiv:2305.09617.

[**Tibshirani et al., 2019**] Tibshirani, R. J., Foygel Barber, R.,
Candès, E. J., & Ramdas, A. (2019). *Conformal Prediction Under
Covariate Shift.* NeurIPS 2019. arXiv:1904.06019.

[**Xu & Lu, 2025**] Xu, X., & Lu, Y. (2025). *TECP: Token-Entropy
Conformal Prediction for LLM Free-Form Generation.*

For methodology references (CP, CRC, multiple hypothesis combination,
Wilson harmonic mean, e-values), see the full bibliography in
[paper/UASEF_Round7.md](UASEF_Round7.md) §References.

---

## Appendix A. Reproducibility checklist

| Item | Location |
|---|---|
| Preprocessing | [experiments/round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py) |
| R9.1 (Table 1c) | [experiments/round9_alpha_critical_real.py](../experiments/round9_alpha_critical_real.py) |
| R9.2 (Table 4-MIMIC) | [experiments/round9_table4_mimic.py](../experiments/round9_table4_mimic.py) |
| R9.3 (dist shift) | [experiments/round9_distribution_shift.py](../experiments/round9_distribution_shift.py) |
| R9.4 (temporal) | [experiments/round9_temporal_shift.py](../experiments/round9_temporal_shift.py) |
| R9.5 (equity) | [experiments/round9_equity_real.py](../experiments/round9_equity_real.py) |
| Aggregate | [experiments/round9_aggregate_report.py](../experiments/round9_aggregate_report.py) |
| Master runner | [run_all_round9.sh](../run_all_round9.sh) |
| MIMIC-IV loader | [data/loader.py](../data/loader.py) `_load_mimic4_jsonl` etc. |
| PHI guard | [models/model_interface.py](../models/model_interface.py) `PHIGuardViolation` |
| Tests | [tests/test_mimic4_loader.py](../tests/test_mimic4_loader.py), [tests/test_paper_claims.py](../tests/test_paper_claims.py) |
| Plan | [improvements/round9_PLAN.md](../improvements/round9_PLAN.md) |
| Runbook | [improvements/round9_RUNBOOK.md](../improvements/round9_RUNBOOK.md) |
| IRB addendum | [paper/IRB_PROTOCOL.md](IRB_PROTOCOL.md) §10 |

## Appendix B. Stratum statistics (single deterministic preprocessing run)

The numbers in §3.4 come from a single deterministic run on MIMIC-IV
v3.1 with `--seed 42`. They are reproducible byte-for-byte for any
reader holding the same MIMIC-IV release. We re-state them here as a
permanent record:

```
[stratum] Classifying admissions ...
  CRITICAL : 전체 274,484, 샘플 1,500 (escalate=1,500)
  HIGH     : 전체  60,544, 샘플 1,500 (escalate=1,500)
  MODERATE : 전체 128,082, 샘플 1,500 (escalate=    0)
  LOW      : 전체  82,918, 샘플 1,500 (escalate=    0)
✅ Wrote 6,000 cases → data/raw/mimic-iv/mimic4_cases.jsonl
```

Verified 2026-05-10 on MIMIC-IV v3.1 (October 2024 release).
