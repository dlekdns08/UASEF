# R10.5 — IRB Physician Adjudication Package

**Compensation**: USD 80/hr × ~10hr × 3 physicians = **$2,400** total budget.
**Timeline**: 4-week external process.
**IRB**: paper/IRB_PROTOCOL.md §11 (Round 10 addendum).

## Task

You are 1 of 3 board-certified physicians (target: emergency, internal
medicine, family medicine) adjudicating 100 MIMIC-IV admissions.

For each case, decide whether the admission should be **escalated** to a
senior clinician for review **at the time of admission decision** (i.e., no
future outcomes available).

## Important — leakage-safe protocol

Each case summary contains **only decision-time features** (admission type,
age, service, primary ICD code as known at admission, decision-time lab
flags, vital sign quartile flags, Charlson comorbidity index). It does
**not** contain future outcomes (ICU transfer, mortality, length of stay,
discharge diagnosis).

## Format

For each case in `r10_5_audit_blinded.jsonl`, append a line to your
response file (e.g., `physician_1_responses.jsonl`):

```jsonl
{"case_id": "20000123", "physician_id": "p1", "label": "YES", "rationale": "high comorbidity, lactate elevation"}
{"case_id": "20000124", "physician_id": "p1", "label": "NO", "rationale": "stable vitals, low Charlson"}
```

`label`: must be exactly "YES" or "NO" (no other strings).
`rationale`: 1 short clinical sentence (≤ 25 words).

## Submission

Send your response JSONL to the IRB coordinator. All three responses are
then merged and analyzed by `experiments/round10_physician_audit.py` for:

- Inter-rater Cohen's κ (pairwise + mean)
- Majority vote vs outcome-derived label confusion matrix
- Per-stratum agreement rate

## Compliance

- All cases are MIMIC-IV PhysioNet-credentialed data.
- Re-identification attempts are prohibited by DUA.
- Do not redistribute case summaries.
- Compensation processed upon completion.

## Files in this package

- `r10_5_audit_blinded.jsonl` — 100 case summaries (no labels)
- `case_template.jsonl` — empty response template
- `example_filled.jsonl` — sample response (1 case) for format reference
- `instructions.md` — this file
