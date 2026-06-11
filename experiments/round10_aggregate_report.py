"""
Round 10 — aggregate report.

R10.1–R10.7 의 산출 .md 를 인덱싱하여 ROUND10_FINAL_REPORT.md 생성.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(p: Path) -> str:
    if p.exists():
        return p.read_text()
    return f"_MISSING: {p.name}_\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=ROOT / "results" / "round10")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "ROUND10_FINAL_REPORT.md")
    args = ap.parse_args()

    files = [
        ("R10.1 — Powered α=0.05 empirical",     "r10_1_alpha_005_empirical.md"),
        ("R10.2 — Multi-seed Table 4-MIMIC",     "r10_2_table4_multiseed.md"),
        ("R10.3 — Distribution shift mitigation","r10_3_mitigation.md"),
        ("R10.4 — Method-agnostic CRC (HEADLINE)","r10_4_method_agnostic.md"),
        ("R10.5 — Physician audit",              "r10_5_physician_audit.md"),
        ("R10.6 — 4-D cost matrix sweep",        "r10_6_cost_sweep_4d.md"),
        ("R10.7 — Expanded-feature validation",  "r10_7_feature_expansion.md"),
    ]

    out = ["# Round 10 — Aggregate Report (MIMIC-IV, Method-Agnostic CRC)\n"]
    out.append(f"- Generated: {datetime.now().isoformat()}")
    out.append("- Plan: [improvements/round10_PLAN.md](../../improvements/round10_PLAN.md)")
    out.append("- Runbook: [improvements/round10_RUNBOOK.md](../../improvements/round10_RUNBOOK.md)")
    out.append("- Paper (EN): [paper/UASEF_Round10.md](../../paper/UASEF_Round10.md)\n")
    for title, fname in files:
        out.append(f"\n---\n\n## {title}\n")
        out.append(_read(args.in_dir / fname))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))
    print(f"✅ {args.out}")


if __name__ == "__main__":
    main()
