# Five Ways to Fake a Clinical Escalation Gate: A Verified-Negative Case Study in Conformal Risk Control on MIMIC-IV

**Authors.** *[Author]*<sup>1</sup>
<sup>1</sup>*[Affiliation]*
Correspondence: `[email]`

**Target venue.** ML4H 2026 (Findings track) / reproducibility & pitfalls
track. Extended: NeurIPS 2026 SafeML Workshop.

**Code.** `https://anonymous.4open.science/r/UASEF-XXXX` (anonymized).

**Data.** MIMIC-IV v3.1 (PhysioNet credentialed). No redistribution.

**Compute disclosure.** All local (Mac Studio M3 Ultra, 96 GB). External-API
cost: **$0**. PHI egress: **0 bytes**.

---

## Abstract

We set out to build a per-stratum Conformal Risk Control (CRC) escalation
gate for clinical deterioration on MIMIC-IV: when a decision-support model is
uncertain, escalate to a clinician rather than risk a missed deterioration.
Over an extended investigation we produced **five separate results that each
looked like a working escalation gate and each dissolved under audit into a
distinct artifact.** We report all five, the audit that caught each, and the
verified conclusion.

The five artifact classes, each independently reproduced in our own pipeline:

1. **Score-sign inversion.** A tabular nonconformity score defined as
   $-\hat p(y{=}1)$ (instead of $+\hat p$) silently collapses the CRC
   threshold to escalate-all: 0 misses at 100% over-escalation, which passes
   any coverage-only review. This produced a spurious "RandomForest is the
   unique winner (0/1293 CRITICAL misses)" headline.
2. **Feature-parsing loss.** Re-parsing features from a serialized string
   dropped the one informative feature, artificially depressing AUROC to
   ~0.55 (near-chance) and motivating a false "fundamental framework limit"
   claim.
3. **Definitional label leakage.** Decision-time features that encode the
   outcome (a comorbidity index and a cohort-level base rate) inflated
   discrimination to AUROC 0.83–0.94.
4. **Temporal leakage.** A one-sided "first-6h lab" window admitted labs
   drawn *after* ICU transfer: 88.6% of CRITICAL positives had their ICU
   transfer inside the window, and 65% of CRITICAL positives had their
   feature value charted post-transfer — inflating CRITICAL AUROC to 0.835.
5. **Informative missingness.** *Which* early labs a clinician ordered is a
   strong severity proxy; the missing-flags alone recover ~99% of the
   apparent signal, so an apparent "lab-value early-warning gate" is really a
   test-ordering-behavior gate.

After correcting all five, the honest decision-time ceiling on the full
cohort is **CRITICAL AUROC 0.796** (leakage-guarded), of which the numeric
lab-value contribution is negligible (value-only 0.795 vs flag-only 0.791),
on only 27% feature coverage, yielding a gate that still escalates 53% of
negatives. **We could not build a clinically deployable, leakage-safe
escalation gate from MIMIC-IV decision-time features, and every apparent
success was an artifact.** Our contributions are (a) the five reproduced
failure modes, (b) a unit-tested conformal core with orientation, temporal,
and missingness guards that catch them, and (c) the honest negative result.

**Keywords.** conformal risk control, clinical escalation, leakage, informative
missingness, negative results, reproducibility, MIMIC-IV.

---

## 1. Introduction

Conformal prediction [Vovk et al., 2005; Angelopoulos & Bates, 2021] and its
risk-control extension [Angelopoulos et al., 2024] promise distribution-free
guarantees for selective prediction, an attractive foundation for clinical
escalation: escalate a case to a human when a model is uncertain. We
attempted to instantiate a per-stratum CRC escalation gate on MIMIC-IV v3.1
[Johnson et al., 2024] with decision-time features (available at admission,
before outcomes).

We failed — but instructively. Every configuration that appeared to succeed
turned out to be one of five distinct artifacts, each invisible to the
coverage-only reporting standard common in the clinical-CP literature. This
paper is the audit trail. We believe it is more useful than a sixth
apparent success would have been, because the five failure modes are
general traps, not idiosyncrasies of our pipeline, and because the audit
tooling that caught them is reusable.

**Contributions.**
1. **Five reproduced artifact classes** (§3), each of which produced a
   plausible "working gate" in our own pipeline.
2. **A unit-tested conformal core** (§2, `models/conformal_escalation.py`,
   10/10 synthetic tests with known answers) plus guards for score
   orientation, temporal windows, and informative missingness.
3. **A verified honest negative** (§4): the leakage-safe decision-time
   ceiling is AUROC ~0.80 on CRITICAL, driven by test-ordering behavior not
   lab chemistry, at 27% coverage — not a deployable gate.

---

## 2. Setup and the Verified Core

**Task.** For each admission and clinical risk stratum $s \in
\{\text{CRITICAL, HIGH, MODERATE, LOW}\}$, decide whether to escalate.
Label (clean, non-circular): $Y = \mathbb{1}\{\text{ICU within 24h}\} \lor
\mathbb{1}\{\text{in-hospital mortality}\}$ — neither component is
lab-defined, avoiding definitional circularity.

**Convention (single source of truth).** Nonconformity score $s(x)$: higher
= riskier. Escalate iff $s(x) > \lambda$. Miss $= \mathbb{1}\{y{=}1 \land
s{\le}\lambda\}$; over-escalation $= \mathbb{1}\{y{=}0 \land s{>}\lambda\}$.
Standard CRC selects the *efficient* threshold $\hat\lambda = \sup\{\lambda:
\hat R_\text{miss}(\lambda) + B/(n{+}1) \le \alpha\}$ — the largest $\lambda$
meeting the miss bound, i.e. escalate the fewest cases. **Escalate-all is the
$\lambda\to-\infty$ endpoint; correct CRC takes the opposite ($\sup$) end and
does *not* collapse when the score discriminates** (verified on separable
synthetic data: over-escalation 0.011 at AUROC ~1.0).

**Verified core.** `models/conformal_escalation.py` implements StandardCRC and
a two-sided BoundedCRC, with (i) an orientation guard that warns when
calibration positives score below negatives (the sign-bug fingerprint), and
(ii) explicit `infeasible` reporting instead of a silent escalate-all
fallback. A synthetic test suite (`tests/test_conformal_escalation.py`, 10
tests, all passing) pins the behavior on data with known answers: separable →
non-vacuous gate; pure noise → high over-escalation; inverted sign →
reproduced escalate-all collapse.

**Reporting standard.** Every result reports per-stratum miss rate AND
over-escalation rate AND direction-robust AUROC; a coverage claim without
over-escalation is uninterpretable.

---

## 3. Five Artifacts That Looked Like a Working Gate

Each subsection: the apparent result → the audit → the corrected reading.

### 3.1 Score-sign inversion (escalate-all collapse)

**Apparent result.** A method-agnostic comparison of five classifiers on
MIMIC-IV reported RandomForest as the unique CRITICAL winner: 0/1293 pooled
misses, exact 95% upper 0.002, "the LLM is the worst candidate."

**Audit.** The tabular nonconformity score was $s = -\hat p(y{=}1)$, so
positives (high $\hat p$) received the *lowest* scores; under escalate-iff-
$s{>}\lambda$, catching positives forces $\lambda\to-\infty$ =
escalate-everything. Direct signature: over-escalation $= 1.000$ in all five
seeds (0 misses is trivially achieved by escalating all), and AUROC computed
on the raw score is 0.06 (the anti-ranked fingerprint; $1-0.06=0.94$). On
correcting the sign ($+\hat p$), RandomForest CRITICAL miss jumps from 0 to
13.6%.

**Corrected reading.** The "unique winner" was an escalate-all artifact of a
one-character sign error. The orientation guard (§2) now warns on it.

### 3.2 Feature-parsing loss (false "framework limit")

**Apparent result.** With "leakage-safe minimal features," no classifier
satisfied $\alpha$; AUROC ~0.55 (near-chance). We had written this up as a
"fundamental limit of the CRC framework at minimal features."

**Audit.** The minimal feature vector was re-parsed from a serialized
`meta_info` string that dropped the count of abnormal early labs — the one
feature carrying signal. Extracting the same four features directly from the
source raised CRITICAL AUROC from 0.556 to 0.761.

**Corrected reading.** Not a framework limit — a parsing bug. The signal
existed; the string round-trip lost it.

### 3.3 Definitional label leakage

**Apparent result.** A richer feature set gave AUROC 0.83–0.94 and a gate
with low over-escalation.

**Audit.** The features included a Charlson comorbidity index computed from
*current-admission* ICD codes and a *cohort-level* specialty base rate — both
encode the outcome. Per-feature checks confirmed these two carried the
discrimination.

**Corrected reading.** Leakage-driven; illegitimate for deployment. Removing
them drops AUROC to the 0.55–0.72 range (§3.2's honest features).

### 3.4 Temporal leakage (labs after ICU transfer)

**Apparent result.** Extracting *actual first-6h lab values* from raw
MIMIC-IV (`labevents`) lifted CRITICAL AUROC to 0.835 — apparently a genuine
lab-value early-warning gate.

**Audit.** The window was one-sided (`charttime ≤ admittime+6h`), with no
guard against `charttime > icu_intime`. Recovering `icu_intime` from
`icustays`: **88.6% of CRITICAL positives transfer to ICU within the 6h
window** (median ~1h after admit), and for **65% of CRITICAL positives the
feature value was charted after ICU transfer** — i.e., after the escalation
it is meant to predict. The AUROC gradient tracks contamination exactly:
CRITICAL (88.6% in-window ICU) scores 0.835 vs HIGH/MODERATE/LOW
(12%/5%/2%) at 0.74/0.68/0.73.

**Corrected reading.** A decision-time guard (`charttime ≤
min(admittime+6h, icu_intime)`) drops CRITICAL AUROC 0.835 → **0.796**.

### 3.5 Informative missingness (ordering, not chemistry)

**Apparent result.** Even after the temporal guard, CRITICAL AUROC 0.796
looked like a lab-value gate.

**Audit.** Decomposing the feature vector: **flag-only (which labs were
ordered, zero numeric values) = 0.791; value-only = 0.795; full = 0.796.**
The missing-flags recover essentially the whole signal; the numeric chemistry
adds nothing beyond noise. Mechanism: ordering an early blood-gas / lactate
panel is itself a near-deterministic marker of clinician concern for an
ICU-bound patient (per-lab present-rate gap positives−negatives up to +0.46).
Compounded by selection: only 27% of admissions have any guarded in-window
lab; label prevalence in that subset is 28% vs 8% in the no-lab group.

**Corrected reading.** The apparent "lab-value early-warning gate" is a
test-ordering-behavior proxy, institution-specific and coverage-limited.

---

## 4. The Verified Honest Negative

After correcting all five artifacts, the definitive decision-time ceiling
(leakage-guarded, full 14,000-admission cohort, 5-seed patient-level split,
clean ICU/mortality label, verified core):

| Stratum | FULL AUROC | value-only | flag-only | StdCRC miss | over_esc |
|---|---|---|---|---|---|
| **CRITICAL** | **0.796** | 0.795 | 0.791 | 0.101 | **0.534** |
| HIGH | 0.688 | 0.685 | 0.669 | 0.153 | 0.591 |
| MODERATE | 0.675 | 0.669 | 0.669 | 0.310 | 0.555 |
| LOW | 0.707 | 0.703 | 0.703 | 0.270 | 0.471 |

Three honest conclusions:

1. **No deployable gate.** At the strongest stratum (CRITICAL), the
   leakage-safe gate escalates 53% of negatives to reach a 10% miss rate; it
   does not satisfy $\alpha=0.05$ on held-out test. The value $\approx$ flag
   equality shows the signal is ordering behavior, not chemistry, on 27%
   coverage — not a clinically deployable, generalizable early-warning gate.
2. **Prior "wins" were artifacts.** Every apparent success in our own
   investigation (§3) was one of five failure modes. We report them so the
   field does not repeat them.
3. **AUROC 0.80 in context.** The number sits within the published
   NEWS/MEWS/LAPS2 range (0.70–0.87), but that literature reports it as a
   *risk score*, not a coverage-guaranteed escalation gate; our contribution
   is to show that wrapping such a score in CRC does not, by itself, yield a
   safe gate, and that the over-escalation column is what reveals this.

---

## 5. Discussion

**Why report a negative?** The clinical-CP literature under-reports
over-escalation and rarely audits for the five failure modes above. A
plausible-looking coverage result (0 misses, tight CP bound) is exactly what
each of our artifacts produced. If our own pipeline generated five such
mirages, others likely exist in print. The reusable defenses are cheap:
report over-escalation; guard score orientation; guard the temporal window
against the outcome time; decompose value vs missingness; and never re-parse
features through a lossy serialization.

**Limitations.** Single database (MIMIC-IV); single-center for the ICU
outcome timing; the LLM path was investigated separately and is a genuine
near-chance null on MedAbstain, not developed here. A prospective,
multi-center replication of the negative would strengthen it.

**On b-CRC.** We explored a two-sided (miss + over-escalation) bounded-CRC
loss that provably excludes the escalate-all endpoint. Once the score-sign
bug (§3.1) is fixed, standard CRC no longer collapses to escalate-all, so
b-CRC's necessity is reduced; we retain it in the core as a guard, not a
headline. The propositions and their corrected statements are in Appendix A.

---

## 6. Conclusion

We tried to build a conformal clinical escalation gate on MIMIC-IV and
produced five results that each looked like success and each was an artifact:
a score-sign inversion, a feature-parsing loss, definitional leakage,
temporal leakage, and informative missingness. Corrected end to end, the
honest decision-time ceiling is AUROC ~0.80 on CRITICAL, driven by
test-ordering behavior rather than lab chemistry, at 27% coverage, escalating
53% of negatives — not a deployable gate. The contribution is the audit: five
reproduced failure modes, a unit-tested core with orientation/temporal/
missingness guards, and an honest, verified negative result. All artifacts
are reproducible on commodity hardware at $0 external cost and zero PHI
egress.

---

## References

[Angelopoulos & Bates, 2021] *A gentle introduction to conformal prediction.*
arXiv:2107.07511.
[Angelopoulos et al., 2024] *Conformal Risk Control.* ICLR.
[Johnson et al., 2024] *MIMIC-IV v3.1.* PhysioNet. doi:10.13026/kpb9-mt58.
[Vovk et al., 2005] *Algorithmic Learning in a Random World.* Springer.
[NEWS/MEWS/LAPS2 references — to finalize.]

---

## Appendix A. Reproducibility

| Artifact | Experiment | Location |
|---|---|---|
| Verified core + tests | — | [models/conformal_escalation.py](../models/conformal_escalation.py), [tests/test_conformal_escalation.py](../tests/test_conformal_escalation.py) |
| §3.1 sign bug | R13/R15 | [experiments/round15_verified_core.py](../experiments/round15_verified_core.py) |
| §3.2 parsing loss | R16 | [experiments/round16_richer_features.py](../experiments/round16_richer_features.py) |
| §3.3 leakage features | R15 | [results/round15/](../results/round15/) |
| §3.4 temporal leakage | R17 | [experiments/round17_raw_lab_values.py](../experiments/round17_raw_lab_values.py) |
| §3.5 missingness + §4 floor | R18 | [experiments/round18_leakage_safe_floor.py](../experiments/round18_leakage_safe_floor.py) |

```bash
export UASEF_BACKEND_NEVER_SEND_PHI=1
.venv/bin/python -m pytest tests/test_conformal_escalation.py -q   # 10/10
.venv/bin/python experiments/round18_leakage_safe_floor.py --extract  # ~25 min
```

## Appendix B. Compliance

MIMIC-IV v3.1 under PhysioNet Credentialed Health Data License v1.5.0. No
redistribution; `.gitignore` excludes all raw data and derived lab caches.
All processing local; verified external egress 0 bytes; no re-identification
attempted.

---

_Final manuscript 2026-07-02. All §3–§4 numbers reproduce against
`results/round{15,16,17,18}/*.json`. Five artifacts, one honest negative._
