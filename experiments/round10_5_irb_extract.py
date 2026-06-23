"""
Round 10 R10.5 — IRB physician audit: case extraction infrastructure.

본 스크립트는 physician 작업 *전에* 필요한 case extraction 만 처리.
실제 physician adjudication 은 4-week 외부 process — `round10_physician_audit.py`
가 결과 JSONL 받은 후 Cohen's κ 계산.

산출 (commit 금지 — DUA):
  data/raw/r10_5_audit_cases.jsonl       — 100 cases (50 CRITICAL + 50 HIGH)
  data/raw/r10_5_audit_blinded.jsonl     — physician 에게 줄 blinded version (label 제거)
  data/raw/r10_5_audit_ground_truth.csv  — case_id → outcome-derived label (audit 종료 후만 사용)

Physician package 도 함께 생성:
  paper/irb_audit_package/instructions.md           — adjudication 지침
  paper/irb_audit_package/case_template.jsonl       — 빈 label 채울 template
  paper/irb_audit_package/example_filled.jsonl      — sample filled response

Compensation 명시 ($80/hr × ~10hr × 3 physicians = $2,400).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import _load_mimic4_jsonl

DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"


def _format_case_for_physician(row: dict) -> dict:
    """JSONL row → physician-facing record (PHI-free where possible)."""
    struct = row.get("structured", {}) or {}
    demo = row.get("demographics", {}) or {}
    summary_lines = [
        f"Patient case summary (MIMIC-IV de-identified):",
        f"  Age bracket: {demo.get('age_bucket', 'unknown')}",
        f"  Sex: {demo.get('sex', '?')}",
        f"  Admission type: {row.get('admission_type', 'unknown')}",
        f"  Primary service: {struct.get('service', 'unknown')}",
        f"  Primary ICD-10: {struct.get('icd_primary', 'unknown')}",
        f"  Active ICD-10 codes: {', '.join(struct.get('icd_codes', [])[:8])}",
        f"  Lab abnormalities (decision-time): {', '.join(struct.get('lab_flags', []))}",
        f"  Vital quartile flags: {', '.join(row.get('vital_flags', []))}",
        f"  Charlson comorbidity index: {row.get('charlson_index', 0)}",
        "",
        "Question: As a board-certified physician, would you escalate this admission "
        "to a senior clinician for review at the time of decision (no future outcomes "
        "available)?",
        "",
        "Please answer with:",
        "  label: YES or NO",
        "  rationale: 1 short sentence",
    ]
    return {
        "case_id": row.get("hadm_id", ""),
        "stratum": row.get("stratum", "MODERATE"),
        "blinded_summary": "\n".join(summary_lines),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--n-critical", type=int, default=50)
    ap.add_argument("--n-high", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-dir", type=Path,
                    default=ROOT / "data" / "raw" / "audit_r10_5")
    ap.add_argument("--package-dir", type=Path,
                    default=ROOT / "paper" / "irb_audit_package")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"JSONL missing: {args.jsonl}")

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    args.package_dir.mkdir(parents=True, exist_ok=True)

    # Load JSONL rows directly (loader 의 MedQACase 가 아니라 raw dict)
    rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} candidate rows")

    by_stratum = {"CRITICAL": [], "HIGH": []}
    for r in rows:
        s = r.get("stratum", "")
        if s in by_stratum:
            by_stratum[s].append(r)

    rng = random.Random(args.seed)
    selected = []
    for stratum, target_n in [("CRITICAL", args.n_critical), ("HIGH", args.n_high)]:
        pool = by_stratum[stratum]
        rng.shuffle(pool)
        selected.extend(pool[:target_n])
    rng.shuffle(selected)  # 최종 ordering 도 random
    print(f"selected {len(selected)} cases (CRITICAL={args.n_critical}, HIGH={args.n_high})")

    # 1. Full audit cases (commit 금지)
    full_path = args.audit_dir / "r10_5_audit_cases.jsonl"
    with open(full_path, "w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {full_path}")

    # 2. Blinded version for physicians (label 제거, case_id + summary 만)
    blinded_path = args.audit_dir / "r10_5_audit_blinded.jsonl"
    with open(blinded_path, "w") as f:
        for r in selected:
            f.write(json.dumps(_format_case_for_physician(r), ensure_ascii=False) + "\n")
    print(f"  wrote {blinded_path}")

    # 3. Ground truth (separate, audit 종료 후만 비교 — physician 에게 송신 금지)
    gt_path = args.audit_dir / "r10_5_audit_ground_truth.csv"
    with open(gt_path, "w") as f:
        f.write("case_id,stratum,outcome_derived_label,specialty\n")
        for r in selected:
            f.write(f"{r.get('hadm_id', '')},{r.get('stratum', '')},"
                    f"{'YES' if r.get('expected_escalate') else 'NO'},"
                    f"{r.get('specialty', '')}\n")
    print(f"  wrote {gt_path}")

    # 4. Physician package — instructions
    instructions = f"""# R10.5 — IRB Physician Adjudication Package

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
{{"case_id": "20000123", "physician_id": "p1", "label": "YES", "rationale": "high comorbidity, lactate elevation"}}
{{"case_id": "20000124", "physician_id": "p1", "label": "NO", "rationale": "stable vitals, low Charlson"}}
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
"""
    (args.package_dir / "instructions.md").write_text(instructions)

    # 5. Template + example  (raw row 는 hadm_id 만 있음 — case_id 로 alias)
    template = "\n".join([
        json.dumps({"case_id": r.get("hadm_id", ""), "physician_id": "p1",
                    "label": "", "rationale": ""})
        for r in selected[:5]
    ]) + "\n"
    (args.package_dir / "case_template.jsonl").write_text(template)
    example = json.dumps({
        "case_id": selected[0]["hadm_id"] if selected else "20000001",
        "physician_id": "p1",
        "label": "YES",
        "rationale": "Elevated lactate + cardiology service + emergency admit → consider sepsis."
    }) + "\n"
    (args.package_dir / "example_filled.jsonl").write_text(example)

    # 6. Symlink blinded JSONL into package
    pkg_blinded = args.package_dir / "r10_5_audit_blinded.jsonl"
    if pkg_blinded.exists() or pkg_blinded.is_symlink():
        pkg_blinded.unlink()
    pkg_blinded.symlink_to(blinded_path.absolute())

    print(f"\n✅ IRB audit package ready:")
    print(f"   internal data (commit 금지): {args.audit_dir}")
    print(f"   physician package: {args.package_dir}")
    print(f"   - instructions.md")
    print(f"   - r10_5_audit_blinded.jsonl ({len(selected)} cases)")
    print(f"   - case_template.jsonl + example_filled.jsonl")
    print(f"\nNext: send package to 3 board-certified physicians (4-week process, $2400 budget)")


if __name__ == "__main__":
    main()
