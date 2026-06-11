"""
Round 10 R10.5 — IRB physician adjudication post-processing.

3 physician 의 escalation YES/NO 라벨이 들어오면 (사람-주도 4주 process)
Cohen's κ + outcome-derived label 과의 confusion matrix 산출.

본 스크립트는 외부 process 의 결과 JSONL 을 받아 처리만 함 — LLM 호출 없음.

산출: results/round10/r10_5_physician_audit.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _cohen_kappa(a: list, b: list) -> float:
    """Binary Cohen's κ."""
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a = sum(1 for x in a if x) / n
    p_b = sum(1 for x in b if x) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_e >= 1:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physician-labels", type=Path,
                    help="JSONL: {case_id, physician_id, label, rationale}")
    ap.add_argument("--outcome-labels", type=Path,
                    help="JSONL: {hadm_id, expected_escalate}")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "r10_5_physician_audit")
    args = ap.parse_args()

    if not args.physician_labels or not args.physician_labels.exists():
        print(f"[R10.5] physician labels 미존재: {args.physician_labels}")
        print("  Round 10 plan §3 R10.5 의 4주 IRB process 가 필요. SKIP.")
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "pending_physician_audit",
            "message": "physician labels 아직 미준비 — R10 plan 의 4주 process 가 필요",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2))
        Path(str(args.out) + ".md").write_text(
            "# R10.5 — PENDING\n\nPhysician audit JSONL 미준비. R10.5 의 외부 4주 process 가 진행되어야 함."
        )
        return

    if not args.outcome_labels or not args.outcome_labels.exists():
        sys.exit(f"[R10.5] outcome labels 미존재: {args.outcome_labels}")

    # Load physician labels
    by_case: dict = {}
    with open(args.physician_labels) as f:
        for line in f:
            r = json.loads(line)
            cid = str(r["case_id"])
            by_case.setdefault(cid, {})[r["physician_id"]] = bool(r["label"])

    # Load outcome labels
    outcome_map: dict = {}
    with open(args.outcome_labels) as f:
        for line in f:
            r = json.loads(line)
            outcome_map[str(r.get("hadm_id"))] = bool(r.get("expected_escalate", False))

    # Per-pair κ + vs outcome
    physician_ids = sorted({pid for d in by_case.values() for pid in d})
    pair_kappas = {}
    for i, p1 in enumerate(physician_ids):
        for p2 in physician_ids[i+1:]:
            a, b = [], []
            for cid, labels in by_case.items():
                if p1 in labels and p2 in labels:
                    a.append(labels[p1]); b.append(labels[p2])
            pair_kappas[f"{p1}-{p2}"] = _cohen_kappa(a, b)

    # Majority vote vs outcome
    maj_vs_outcome_a, maj_vs_outcome_b = [], []
    tp = fn = fp = tn = 0
    for cid, labels in by_case.items():
        if cid not in outcome_map: continue
        votes = list(labels.values())
        if not votes: continue
        majority = sum(votes) > len(votes) / 2
        outcome = outcome_map[cid]
        maj_vs_outcome_a.append(majority)
        maj_vs_outcome_b.append(outcome)
        if majority and outcome: tp += 1
        elif (not majority) and outcome: fn += 1
        elif majority and (not outcome): fp += 1
        else: tn += 1
    maj_kappa = _cohen_kappa(maj_vs_outcome_a, maj_vs_outcome_b)

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_cases": len(by_case),
        "n_physicians": len(physician_ids),
        "pair_kappa": pair_kappas,
        "mean_pair_kappa": (sum(pair_kappas.values()) / len(pair_kappas)
                             if pair_kappas else None),
        "majority_vs_outcome_kappa": maj_kappa,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2))
    lines = ["# Round 10 R10.5 — Physician audit\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- n_cases: {report['n_cases']}, n_physicians: {report['n_physicians']}\n")
    lines.append(f"**Mean pairwise κ: {report['mean_pair_kappa']:.3f}** (threshold for substantial agreement: 0.6)")
    lines.append(f"**Majority vs outcome-derived label κ: {report['majority_vs_outcome_kappa']:.3f}**\n")
    lines.append("## Confusion matrix (majority physician vote vs outcome-derived label)\n")
    lines.append(f"- TP={tp} FN={fn} FP={fp} TN={tn}")
    Path(str(args.out) + ".md").write_text("\n".join(lines))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
