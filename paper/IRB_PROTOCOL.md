# UASEF — IRB Protocol for Heuristic-Label Validation (L1 mitigation)

**Status:** Draft v1, 2026-05-08. Pre-submission to host institution IRB.

This document describes the protocol that operationalizes the §8 L1
limitation commitment in the main paper (paper/UASEF_Round7.md). Its
goal is to replace the keyword-based `expected_escalate` ground truth
on a sub-sample of MedQA/MedAbstain CRITICAL cases with attending-
physician adjudicated labels, then re-run Tables 1 and 4 on the
relabeled subset for the camera-ready version.

The protocol is published here so that reviewers can verify the
mitigation plan is concrete (not a vague "future work" gesture) and so
that other groups working in this area can adopt the same labeling
scheme.

---

## 1. Subjects and data

- **No human subjects are recruited.** The protocol covers
  retrospective re-labeling of *publicly released benchmark data*
  (MedQA-USMLE-4-options [Jin et al., 2021]; MedAbstain
  [Machcha et al., 2026]).
- No PHI / PII is involved. All cases are exam-style vignettes already
  in the public domain via HuggingFace.
- The annotation work is on the labelers (attending physicians); their
  identities are pseudonymized in the released agreement file.

## 2. Labelers

- **N = 3** board-certified physicians, all with active hospital
  privileges in the target specialty:
  - 1 emergency medicine attending (target: CRITICAL stratum)
  - 1 internal medicine attending (target: HIGH/MODERATE strata)
  - 1 family medicine attending (target: LOW stratum + cross-check)
- Each labeler signs a NDA/conflict-of-interest disclosure prior to
  receiving cases.
- Compensation: USD 80/hour, ~10 hours expected.

## 3. Sample

- **n = 200 cases** drawn stratified-randomly from the MedAbstain test
  set (n=50 per stratum × 4 strata). Random seed 42, stratification by
  `SPECIALTY_TO_STRATUM` mapping in `experiments/round7_table1_coverage.py`.
- This is a CRITICAL-stratum-balanced sub-sample chosen so the
  re-labeled CRITICAL recall comparison (v2 vs single-α baselines) has
  ≥ 50 positives per arm and Wilson 95% CI half-width ≤ 0.10.
- Power calculation: at v2 = 0.96 vs TECP = 0.16 on the original labels
  (Table 4), even a 50% reduction in effect size (i.e. true v2 = 0.78
  vs TECP = 0.40) is detectable at α = 0.05 / (1-β) = 0.80 with n = 50.

## 4. Labeling task

For each case, the labeler answers:

> "Should this case be escalated to a human specialist before the AI
> system commits to an answer? **YES / NO**, and a 1-sentence
> justification."

The "YES" criterion is operationalized as:

- **Time-critical**: missing the diagnosis within hours risks death or
  permanent disability (e.g. STEMI, pulmonary embolism, sepsis,
  meningitis, ectopic pregnancy).
- **High specialist dependence**: the answer requires multi-modal
  evidence (imaging, labs, prior history) that a text-only LLM cannot
  reliably weight.
- **Rare disease/atypical**: the prevalence is low enough that an
  LLM trained on broad corpora is unlikely to have well-calibrated
  uncertainty.

The labeler additionally records *which* of the three criteria
applies. This metadata is not used in the main metric but enables a
secondary analysis (does the current keyword classifier over-trigger
on time-critical and under-trigger on rare/atypical?).

## 5. Inter-rater agreement

- Each case is labeled by ≥ 2 physicians (labelers 1+2 for
  CRITICAL/HIGH; 1+3 for MODERATE/LOW; 2+3 for cross-checks).
- Disagreements are resolved by a third labeler's adjudication.
- We report **Cohen's κ** per stratum and overall, plus a confusion
  table of (heuristic_label × physician_label).

## 6. Analysis plan (camera-ready)

Once labels are received:

1. **Replication of Table 1** with physician labels — recompute
   per-stratum miss rate for TECP / Round 6 / Round 7 / TECP-stratified
   / v1-cost-aware (`run_full_evaluation.sh` already produces all of
   these via the shipped baselines).
2. **Replication of Table 4** with physician labels — same set of
   methods, same metrics.
3. **Sensitivity analysis** — report the Table 4 cost-reduction ratio
   computed twice: once with heuristic labels and once with physician
   labels. The expected outcome is a small (~10-30%) shrinkage of the
   reduction ratio because the physician labels will identify some
   keyword-misclassified CRITICAL cases that v2 was not actually
   catching either; v2's advantage over single-α is robust to this.
4. **Disagreement pattern analysis** — which case features predict
   physician-vs-heuristic disagreement? This informs L1 mitigation in
   future deployments.

The analysis plan is registered before unblinding the labelers'
results to avoid p-hacking.

## 7. Timeline

| Phase | Target |
| --- | --- |
| IRB submission | 2026-05-22 |
| IRB approval (expected exempt status) | 2026-06-15 |
| Labeler recruitment | 2026-06-30 |
| Labeling complete | 2026-07-31 |
| Re-analysis + camera-ready | 2026-08-15 |

If the IRB timeline slips beyond 2026-08-30, the camera-ready will
note the result will appear in a *named follow-up paper* rather than
being held in the conference proceedings revision.

### 7.1 LLM-Judge Self-Consistency Fallback (round 8)

In parallel to the IRB process, we run a **non-clinical**, *partial-validation*
fallback to surface any catastrophic L1 mismatch *before* IRB results arrive:

- **Procedure.** $n = 200$ stratified-random MedAbstain CRITICAL/HIGH cases
  are independently judged by two LLMs (OpenAI gpt-5.5, Anthropic
  claude-opus-4-7) for ESCALATE / NO ESCALATE.
- **Reporting.** We publish Cohen's $\kappa$ between the two judges and
  the agreement rate against the heuristic-classifier label
  (`results/round8/llm_judge_relabel.{json,md}`).
- **Decision rule.** Auxiliary "consensus label" is used in supplementary
  §C **only if $\kappa \ge 0.7$**.
- **Hard limit.** LLM-judge agreement measures *reliability*, not *validity*.
  Two models trained on overlapping distributions can systematically agree
  on biased labels. **The IRB-driven physician relabeling remains the
  load-bearing L1 mitigation** in the camera-ready; the LLM-judge is a
  pre-IRB sanity probe.

## 8. Pre-registration

This protocol is the pre-registration document. It is committed to the
public repository at `paper/IRB_PROTOCOL.md` before the labeling
begins. Any deviation from this protocol will be noted in the
camera-ready version with a clear "deviation log" appendix.

## 9. Data release

- Re-labeled cases will be released as a JSONL at
  `data/medabstain_relabeled_v1.jsonl` (200 rows; `case_id`,
  `physician_label`, `heuristic_label`, `criterion`, `kappa_pair`).
- The aggregate Table 1/4 numbers (with both label sets) will be
  reported in the camera-ready paper Appendix.
- Individual labeler identities remain pseudonymized.

---

## 10. MIMIC-IV addendum (Round 9; PhysioNet credentialed)

The Round 9 plan
([improvements/round9_PLAN.md](../improvements/round9_PLAN.md))
extends the evaluation to MIMIC-IV v3.1
[Johnson et al., 2024], which is **not** in the public domain and
requires a separate compliance regime in addition to this IRB protocol.

**Credentialing prerequisites for any team replicating Round 9.**

1. PhysioNet account + CITI "Data or Specimens Only Research"
   certification (or institutional equivalent).
2. Signed *PhysioNet Credentialed Health Data License v1.5.0* DUA
   (see `LICENSE.txt` shipped with the MIMIC-IV download).
3. For the optional Phase-2 free-text experiments, additional
   credentialing for **MIMIC-IV-Note v2.2** is required.

**Operational obligations enforced by this repository.**

- MIMIC-IV raw data is **never** committed; `.gitignore` blocks
  `data/raw/mimic-iv/`, `mimic*.csv.gz`, `discharge.csv*`,
  `radiology.csv*`, `edstays.csv*`, and `triage.csv*` globally.
- Phase 1 (`hosp` + `icu` only) transmits only **structured features**
  (de-identified ICD-10 codes, lab abnormality flags, vital quartiles,
  `admission_type`) through a deterministic template
  `_MIMIC_NOTE_TEMPLATE`. Free-text discharge summaries are excluded
  from any external API call.
- Phase 2 (free-text experiments) is restricted to the local LMStudio
  backend. The `UASEF_BACKEND_NEVER_SEND_PHI=1` environment guard
  rejects OpenAI / Anthropic backend calls when the case `source` field
  is `mimic4_note*`. The guard is unit-tested in
  `tests/test_mimic4_loader.py`.
- Re-identification attempts on MIMIC-IV data are forbidden by the
  DUA. Any suspected re-identification will be reported to
  `PHI-report@physionet.org` per LICENSE clause 5.

**Cross-reference with §1–§9.** The MedAbstain re-labeling protocol in
§§1–9 above is *independent* of the MIMIC-IV addendum. The MIMIC-IV
ground truth is the patient's recorded clinical outcome
(ICU admission, mortality, sepsis flag, readmission), not a physician
re-label of an exam vignette. A 100-case overlap audit
(board-certified physician judgment vs MIMIC-IV outcome label) is
included in the Round 9 Phase-2 plan as an additional sanity check.

---

**Contact (for IRB review).** [redacted for double-blind]
