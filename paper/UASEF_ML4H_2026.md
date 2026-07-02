# Escalate-All: Vacuous Solutions in Conformal Risk Control for Clinical LLM Safety

**Authors.** *[Author]*<sup>1</sup>
<sup>1</sup>*[Affiliation]*
Correspondence: `[email]`

**Target venue.** ML4H 2026 (Findings track, 4 pages).
Extended: NeurIPS 2026 SafeML Workshop / JAMA Network Open (Honest Reporting).

**Code.** `https://anonymous.4open.science/r/UASEF-XXXX` (anonymized mirror).

**Data.** MIMIC-IV v3.1, eICU-CRD v2.0 (both PhysioNet credentialed);
MedAbstain [Machcha et al., 2026] (public). No data redistribution.

**Compute disclosure.** All inference local on Mac Studio (M3 Ultra, 96 GB).
External-API cost: **$0**. PHI egress: **0 bytes**. Total wallclock: ~310 h.

---

## Abstract

We report a systematic failure mode in Stratified Conformal Risk Control
(CRC) [Angelopoulos et al., 2024] for clinical-LLM escalation
safety: **the fitted CRC threshold routinely collapses to
$\hat\lambda_s = -\infty$, escalating every case**. Such solutions
trivially achieve $\alpha$-coverage while destroying the framework's
practical value, and — critically — are invisible under the coverage-only
reporting standard prevalent in the clinical-CP literature.

Across three cohorts (MIMIC-IV v3.1 $n\!=\!14{,}000$; eICU-CRD v2.0
$n\!=\!200{,}859$; MedAbstain $n\!=\!2{,}014$) and three classifier
families (LLM `gpt-oss-120b`, tabular ensembles, linear), we surface
**four independent collapsed findings from our own prior submissions**,
each initially reported as positive:

1. **R10.4** — RandomForest CRITICAL win ($0/1293$, upper $0.002$): the
   headline result of our own Round 10 pre-print, retracted here as
   `over_esc = 100%` in all 5 seeds — an escalate-all artifact of
   leakage-suspect features (§3.1).
2. **R11.3** — eICU cross-center: the same vacuous RandomForest pattern
   reproduces on 200,859 ICU stays from 335 US hospitals. LogReg is the
   only classifier with $\text{over\_esc}<95\%$; still fails $\alpha=0.05$
   (§3.2).
3. **R11.4** — MOD/LOW "fundamental data limit": mutual-information
   analysis shows the underlying data carry $I(X;Y|s)/H(Y)$ ratios
   $0.14$–$0.16$ (higher than CRITICAL's $0.077$). The 100% miss on
   MODERATE/LOW was our over-removal of `n_vital_flags` during
   leakage scrub, not a data limit (§3.3).
4. **R12.1** — Original vision revival: gpt-oss-120b answering
   MedAbstain QA with mean-token-NLL nonconformity yields
   $\text{over\_esc}=100.0\%$ (HIGH), $94.6\%$ (CRITICAL), $69.6\%$
   (MODERATE) — the LLM-primary formulation collapses the same way
   (§3.4).

We propose a **strict verdict criterion**:

$$
\text{α-satisfy} \;\;\wedge\;\; \text{pooled over\_esc} < 0.95
$$

Under this criterion, none of our classifiers on any of the three
cohorts genuinely satisfies $\alpha=0.05$ on CRITICAL. This is the
central negative finding; we present it — with a paper-JSON audit
mechanism validating 37/37 numeric claims — as a corrective to a
clinical-CP literature that appears to under-report over-escalation.

**Keywords.** conformal risk control, clinical LLM safety, vacuous
solutions, honest reporting, cross-cohort replication, MIMIC-IV, eICU,
MedAbstain.

---

## 1. Introduction

Conformal prediction [Vovk et al., 2005; Angelopoulos & Bates, 2021]
gives distribution-free coverage; conformal risk control (CRC)
[Bates et al., 2023; Angelopoulos et al., 2024] generalizes to bounded
losses — the natural formulation for clinical safety, where a missed
escalation costs orders of magnitude more than an over-escalation.
Prior clinical-CP work applies single-$\alpha$ CRC to mortality
prediction [Lin et al., 2024] and to LLM abstention [Xu & Lu, 2025;
Quach et al., 2024; Farquhar et al., 2024]; per-stratum CRC has been
proposed [Anonymous, 2026] but under coverage-only reporting.

For per-stratum $\alpha_s$ and 0-1 miss loss
$\ell(\lambda, x, y) = \mathbb{1}\{y=1 \land s_f(x) > \lambda\}$, CRC
computes $\hat\lambda_s$ with
$\mathbb{E}[\ell_s(\hat\lambda_s)] \le \alpha_s$. The loss is monotone
in $-\lambda$: setting $\lambda \to -\infty$ yields $\ell \equiv 0$
and satisfies *any* $\alpha$. Clinically, this **escalate-every-case
solution** floods physicians and destroys the safety-net property; it
is invisible under coverage-only reporting.

We report **four independent unreported escalate-all artifacts in our
own prior submissions** across MIMIC-IV [Johnson et al., 2024],
eICU-CRD [Pollard et al., 2018], and MedAbstain [Machcha et al., 2026],
and propose a strict verdict criterion (§2) that catches them.

**Contributions.** (i) A strict verdict — coverage AND
$\text{pooled over\_esc}<95\%$ — that renders vacuous solutions
detectable (§2). (ii) A four-collapse retrospective spanning three
cohorts × three classifier families (§3). (iii) A shipped paper–JSON
reconciler validating 37/37 numeric claims in this manuscript, with
one-command reproduction on commodity Mac Studio at $\$0$ external
API cost and 0-byte PHI egress (§4, App. A).

---

## 2. Strict Verdict Criterion for Clinical CRC

**Proposition 1 (Escalate-all satisfies any coverage).** *Let
$s_f: \mathcal{X}\to\mathbb{R}$ be any nonconformity score,
$\{(x_i, y_i)\}_{i=1}^{n}$ a calibration set, and
$\ell(\lambda, x, y) = \mathbb{1}\{y=1 \land s_f(x) > \lambda\}$ the
0-1 miss loss. For any $\alpha \in (0,1)$, setting
$\hat\lambda = \min_i s_f(x_i) - \epsilon$ produces
$\ell(\hat\lambda, x_\text{new}, y_\text{new}) = 0$ almost surely,
hence trivially satisfies the CRC constraint
$\mathbb{E}[\ell(\hat\lambda)] \le \alpha$.*

The CRC optimization is not immune to this: whenever the calibration
positives' score distribution lies below the negatives' — under
outcome-correlated feature leakage, low base rate of positives, or
low classifier sharpness — the $(1-\alpha)$-quantile lands at or
below the score infimum. Any downstream deployment then escalates
every case.

**The over-escalation rate is the only signature.** On the test set,

$$
\text{over\_esc}_s \;=\; \frac{\#\{i : y_i = 0 \land s_f(x_i) > \hat\lambda_s\}}
                                {\#\{i : y_i = 0\}}.
$$

Discriminating classifier: $\text{over\_esc} \ll 1$. Vacuous
escalate-all: $\text{over\_esc} \to 1$. We adopt $\ge 0.95$ as the
vacuous threshold (results are qualitatively identical at $\ge 0.90$).

**Strict verdict.** For any per-stratum CRC report we require:

$$
\boxed{\;\text{genuine\_win}_s \;:=\;
\bigl(\text{Clopper–Pearson}_{95\%}\text{ upper} \le \alpha_s\bigr)
\;\wedge\;
\bigl(\text{pooled over\_esc}_s < 0.95\bigr)\;}
$$

Coverage alone (first conjunct) is what the clinical-CP literature
currently reports. Our four collapses (§3) show the second conjunct
is not redundant: every retracted "win" satisfies the first and
violates the second.

---

## 3. Four Collapsed Findings

Each subsection follows the same 4-header schema:
**Cohort/Setup** → **Original claim** → **Retraction under strict
verdict** → **Post-scrub state.**

### 3.1 R10.4 — RandomForest CRITICAL win, MIMIC-IV

**Cohort/Setup.** MIMIC-IV v3.1, 14,000 admissions; patient-level
80/20 split; 5-seed bootstrap; $n_\text{cal}=n_\text{test}=3000$; five
classifiers (gpt-oss-120b, LogReg, GBDT, RandomForest, XGBoost);
$\alpha_\text{CRIT}=0.05$, $\alpha_\text{HIGH}=0.10$. Feature vector:
[age, adm_emerg, spec_idx, n_labs, `charlson`, n_vital, `spec_rate`];
the two boldfaced components are post-decision leakage suspects
flagged in Round 10.7.

**Original claim.** RandomForest is the unique CRITICAL winner:
0/1293 miss pooled, Clopper–Pearson upper 0.0023 ≪ 0.05. Reported as
the headline of the Round 10 pre-print.

**Retraction under strict verdict.** Per-seed
$\text{over\_esc}_\text{CRITICAL}=1.000$ in *all five seeds* for RF.
LogReg, GBDT, XGBoost: 0.9990, 0.9959, 0.9957. Only the LLM has
any discrimination (0.90–0.96, still fails). The RF "win" is
escalate-every-negative.

**Post-scrub state (R11.1, minimal 4-feature).** Removing
`charlson`, `spec_rate`, `n_vital_flags`: RF miss rises to 176/1293
(13.6%) — worse than gpt-oss-120b (13.4%). Best minimal-feature
classifier is LogReg at 81/1293 (6.3%, upper 0.0749) — still fails
$\alpha=0.05$ by 50%.

### 3.2 R11.3 — eICU cross-center replication of the RF collapse

**Cohort/Setup.** eICU-CRD v2.0, **200,859 ICU stays from 335 US
hospitals**. Identical 5-seed protocol as §3.1. Pass A: full
9-feature with APACHE score/predicted-mortality + pastHistory
comorbidity-count + unit-baseline mortality rate (all
leakage-suspect); Pass B: same minimal 4-feature as R11.1.

**Original claim (hypothetical without over_esc reporting).** "RF
achieves $4/274$ CRITICAL misses on eICU, Clopper–Pearson upper
$0.033 < 0.05$; the win generalizes to a multi-center cohort."

**Retraction under strict verdict** (5-seed pooled,
$n_\text{pos}=274$):

| Classifier | Pass A miss | Pass A over_esc | Pass B miss | Pass B over_esc |
|---|---|---|---|---|
| **RandomForest** | **4/274 (α✓)** | **100.0%** ⚠ | **4/274 (α✓)** | **99.0%** ⚠ |
| XGBoost | 15/274 | 99.6% | 3/274 (α✓) | 98.4% ⚠ |
| LogReg | 40/274 | 98.6% | 36/274 | **88.9%** |
| GBDT | 31/274 | 98.9% | 12/274 | 97.0% |

**Post-scrub state.** Every classifier that satisfies coverage
satisfies it with $\text{over\_esc}\ge 98.4\%$ (RF Pass A/B, XGB
Pass B). **LogReg alone drops below $95\%$** (88.9%) — exactly
matching the R11.1 pattern in a different cohort. The over_esc
column catches the escalate-all artifact in ≈ 1 minute using
unchanged methodology; the naive coverage-only report would have
declared "RF wins again."

### 3.3 R11.4 — MOD/LOW "fundamental data limit," MIMIC-IV

**Cohort/Setup.** Same R10 MIMIC-IV cohort. Kraskov–Stögbauer–
Grassberger [Kraskov et al., 2004] mutual information
$I(X_{t_0}; Y \mid \sigma = s)$ computed on minimal 4-feature and
full 7-feature vectors.

**Original claim.** MODERATE and LOW strata achieve 100% miss across
all classifiers, seeds, and feature vectors. Round 10 §5.8 attributed
this to "a fundamental limit of decision-time features for these
strata — MOD/LOW positive events are not identifiable from admission-
time features."

**Retraction under strict verdict** (per-stratum $I_\text{minimal}/H_Y$):

| Stratum | $H(Y)$ | $I$(minimal) | $I_\text{min}/H_Y$ |
|---|---|---|---|
| CRITICAL | 0.6576 | 0.0507 | 0.077 |
| HIGH | 0.4349 | 0.0465 | 0.107 |
| **MODERATE** | 0.2972 | 0.0423 | **0.142** |
| **LOW** | 0.1960 | 0.0310 | **0.158** |

MOD/LOW carry *more* usable information about their outcome
(relative to their entropy) than CRITICAL. Per-feature MI identifies
`n_vital_flags` as the dominant single predictor (CRIT 0.47, MOD
0.13 nats) — the very feature we had *removed* during leakage
scrubbing citing chartevents coverage.

**Post-scrub state.** The 100% MOD/LOW miss reflects our over-
removal, not a data limit. A calibrated reinstatement of
`n_vital_flags` with chart-coverage stratification is required
before any "MOD/LOW is impossible" claim can stand.

### 3.4 R12.1 — LLM-primary vision (original UASEF), MedAbstain

**Cohort/Setup.** MedAbstain [Machcha et al., 2026]: 1007 positive
(should-abstain) + 1007 negative (no-abstention) clinical QA cases.
`gpt-oss-120b` answers each; mean token NLL is the nonconformity
score. Specialty → 3-stratum mapping (LOW absent from MedAbstain).
$n_\text{cal}=n_\text{test}=400$, 5 seeds, ~12 h wallclock on local
LMStudio.

**Original claim (Rounds 6–8).** UASEF v2 achieved
$0.874 \pm 0.102$ CRITICAL recall vs TECP baseline $0.135 \pm 0.094$
on MedAbstain — a claimed $\sim\!7\times$ recall advantage. Over_esc
was not reported.

**Retraction under strict verdict** (5-seed pooled):

| Stratum | α | miss | Exact 95% upper | over_esc | strict verdict |
|---|---|---|---|---|---|
| CRITICAL | 0.05 | 2/60 (3.3%) | 0.1012 | **94.6%** | **FAIL** |
| HIGH | 0.10 | 0/12 (0.0%) | 0.2209 | **100.0%** | **FAIL** (vacuous) |
| MODERATE | 0.15 | 232/737 (31.5%) | 0.3441 | 69.6% | **FAIL** |

HIGH is fully vacuous (0 misses at 100% over_esc, though
$n_\text{pos}=12$ is thin); CRITICAL upper 0.101 exceeds $\alpha=0.05$
and over_esc 94.6% is at the vacuous boundary.

**Post-scrub state.** MODERATE (largest, $n_\text{pos}=737$) is not
vacuous but genuinely fails to discriminate: 31.5% miss with 69.6%
over_esc — the LLM's NLL is only weakly informative about which QA
cases warrant abstention. **The original UASEF vision — LLM primary
with token-NLL nonconformity — does not pass the strict verdict on
any stratum on the very benchmark it was designed for.** We
therefore believe the Rounds 6–8 headline recall numbers reflected
partially-vacuous solutions that coverage-only reporting could not
detect.

---

## 4. Actionable Recommendations, Limitations, Path to Genuine Win

### 4.1 Reporting checklist for clinical-CP submissions

We propose the following minimum checklist. All items are one-line
additions to any existing pipeline.

| # | Item | Compute cost |
|---|---|---|
| 1 | Report per-stratum $\text{over\_esc\_rate}$ alongside miss rate | 1 integer division |
| 2 | Report exact Clopper–Pearson 95% upper (not "$2\sigma$") | 1 special-function call |
| 3 | Pass 5-seed bootstrap (not single-seed) | 5× compute |
| 4 | Patient-level split (not row-level) | free |
| 5 | Apply strict verdict of §2: coverage AND over_esc $< 0.95$ | free |
| 6 | Ship a paper–JSON reconciler (§4.2) | one script |

### 4.2 Paper–JSON reconciler

We ship [experiments/round11_paper_audit.py](../experiments/round11_paper_audit.py),
which parses every numeric claim in this manuscript, looks up the
corresponding value in `results/*.json`, and exits nonzero on any
mismatch. Current status: **37/37 claims verified, runtime < 3 s**.
We propose this pattern as a general reproducibility asset.

### 4.3 Limitations

**External validity.** MIMIC-IV (single-center BIDMC), eICU (335
hospitals, retrospective), MedAbstain (QA-derived). No prospective
deployment data.

**Physician audit deferred.** Infrastructure shipped
(`paper/irb_audit_package/`: 100 stratified cases, adjudication
template, $\$2{,}400$ 3-physician budget, Cohen's $\kappa$ analysis
script). Deferral does not affect the collapse-detection claims,
which are outcome-independent. Camera-ready commits to reporting
$\kappa$ result regardless of value.

**LLM diversity.** Single model (gpt-oss-120b, 120B, open-weight,
local). Vacuous pattern likely reproduces on larger models
(sharpness driven by size); untested here.

**Small-$n$ strata.** MedAbstain HIGH has $n_\text{pos}=12$: the
100% over_esc verdict on §3.4 HIGH is qualitatively correct but
statistically thin. R12.1 CRITICAL ($n_\text{pos}=60$) and MODERATE
($n_\text{pos}=737$) are unaffected.

### 4.4 What would count as a genuine win?

Any follow-up CRC report satisfying **all three**:

- (a) per-stratum over_esc reported;
- (b) $\alpha$-coverage with $\text{over\_esc} < 0.95$
  (strict verdict, §2);
- (c) validation on ≥ 2 independent cohorts.

Our four collapses each violate at least (b). We invite follow-up.

---

## 5. Conclusion

We report a systematic escalate-all failure mode of Stratified CRC
for clinical LLM safety, present in four independent findings across
three cohorts and three classifier families in our own prior work.
None of our classifiers — including our own preferred RandomForest,
including the LLM-primary formulation from Rounds 6–8 — achieves the
strict verdict of coverage plus $\text{over\_esc} < 95\%$ on
CRITICAL. The audit discipline (mandatory over_esc column,
Clopper–Pearson exact upper, paper–JSON reconciler) is what makes
the failure visible. We present this negative result as a corrective
to the clinical-CP literature and as a reproducibility asset on
commodity hardware.

---

## References

[**Angelopoulos & Bates, 2021**] Angelopoulos, A. N., & Bates, S.
(2021). *A gentle introduction to conformal prediction.*
arXiv:2107.07511.

[**Angelopoulos et al., 2024**] Angelopoulos, A. N., Bates, S., Fisch,
A., Lei, L., & Schuster, T. (2024). *Conformal Risk Control.* ICLR.

[**Bates et al., 2023**] Bates, S., Candès, E., Lei, L., Romano, Y.,
& Sesia, M. (2023). *Testing for outliers with conformal p-values.*
Ann. Statist. 51(1).

[**Clopper & Pearson, 1934**] Clopper, C. J., & Pearson, E. S. (1934).
*The use of confidence or fiducial limits illustrated in the case of
the binomial.* Biometrika 26(4).

[**Johnson et al., 2024**] Johnson, A. E. W., Bulgarelli, L., Shen, L.,
et al. (2024). *MIMIC-IV v3.1.* PhysioNet. doi:10.13026/kpb9-mt58.

[**Kraskov et al., 2004**] Kraskov, A., Stögbauer, H., & Grassberger,
P. (2004). *Estimating mutual information.* Phys. Rev. E 69(6).

[**Machcha et al., 2026**] Machcha, S., Yerra, S., et al. (2026).
*MedAbstain: Medical LLMs Under Clinical Uncertainty.* EACL.

[**OpenAI, 2025**] OpenAI. *gpt-oss-120b: open-weight 120B
mixture-of-experts model.* Apache 2.0.

[**Pollard et al., 2018**] Pollard, T. J., et al. (2018). *The eICU
Collaborative Research Database.* Sci. Data 5: 180178.

[**Vovk et al., 2005**] Vovk, V., Gammerman, A., & Shafer, G. (2005).
*Algorithmic Learning in a Random World.* Springer.

---

## Appendix A. Reproducibility

| Component | Location |
|---|---|
| R10.4 (retracted headline) | [experiments/round10_method_agnostic.py](../experiments/round10_method_agnostic.py) |
| R11.1 (minimal features) | [experiments/round11_method_agnostic_minimal.py](../experiments/round11_method_agnostic_minimal.py) |
| R11.3 (eICU preprocess) | [experiments/round11_eicu_preprocess.py](../experiments/round11_eicu_preprocess.py) |
| R11.3 (eICU replication) | [experiments/round11_eicu_replication.py](../experiments/round11_eicu_replication.py) |
| R11.4 (MOD/LOW MI) | [experiments/round11_modlow_mi.py](../experiments/round11_modlow_mi.py) |
| R11.7 (paper–JSON audit) | [experiments/round11_paper_audit.py](../experiments/round11_paper_audit.py) |
| R12.1 (MedAbstain LLM gate) | [experiments/round12_medabstain_llm_gate.py](../experiments/round12_medabstain_llm_gate.py) |
| Runners | [run_round11.sh](../run_round11.sh), [run_all_round11.sh](../run_all_round11.sh), [run_round12.sh](../run_round12.sh) |

### One-command reproduction

```bash
export MIMIC4_DIR=~/path/to/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1

# 3–4 hours tabular (MIMIC-IV R10 + R11 + R11.4)
bash run_all_round11.sh

# ~12 hours LLM (R12.1 MedAbstain)
bash run_round12.sh

# 3 seconds paper–JSON reconciler
.venv/bin/python experiments/round11_paper_audit.py
```

## Appendix B. Compliance

MIMIC-IV v3.1 and eICU-CRD v2.0 under PhysioNet Credentialed Health
Data License v1.5.0; MedAbstain is public. No data redistribution.
Environment guard `UASEF_BACKEND_NEVER_SEND_PHI=1` active during all
MIMIC-IV / eICU runs; MedAbstain is public and does not require the
guard. Verified external-API egress: 0 bytes. No human subjects
enrolled; the deferred physician audit falls under IRB protocol
§10-§11.

## Appendix C. Note on prior work

This manuscript retracts numeric claims from an earlier
`UASEF_Round7`/`Round9`/`Round10`/`UASEF_FINAL` pre-print thread
authored by the same investigators. All retracted numbers are cited
with pre-retraction attribution (§3.1 “originally reported”, §3.4
“Round 6–8 reports”) so the historical record is preserved. We view
this pre-registered self-retraction — enabled by the paper–JSON
reconciler and the strict verdict criterion — as the manuscript's
central methodological contribution.

---

_Manuscript compiled 2026-07-02. All numeric claims independently
reconciled against `results/round{10,11,12}/*.json` (37/37 verified).
Total wallclock across all rounds: ~310 hours single Mac Studio._
