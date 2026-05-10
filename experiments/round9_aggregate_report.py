"""
Round 9 — aggregate Round 9 산출물을 한 MD 로 묶어 round9_report.md 생성.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read_md(p: Path) -> str:
    if p.exists():
        return p.read_text()
    return f"_MISSING: {p.name}_\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=ROOT / "results" / "round9")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round9" / "round9_report.md")
    args = ap.parse_args()

    files = [
        ("R9.1 — α=0.001 empirical (Table 1c)",     "alpha_critical_real.md"),
        ("R9.2 — Table 4-MIMIC head-to-head",       "table4_mimic.md"),
        ("R9.3 — Real-EHR distribution shift",       "dist_shift_real.md"),
        ("R9.4 — Temporal shift",                    "temporal_shift.md"),
        ("R9.5 — Demographic equity audit",          "equity_audit_real.md"),
    ]

    out = ["# Round 9 — Aggregate Report (MIMIC-IV)\n"]
    out.append(f"- Generated: {datetime.now().isoformat()}\n")
    out.append("이 보고서는 round9 단계별 .md 를 합쳐서 한 눈에 보기 위한 인덱스입니다.\n")
    for title, fname in files:
        out.append(f"\n---\n\n## {title}\n")
        out.append(_read_md(args.in_dir / fname))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))
    print(f"✅ {args.out}")


if __name__ == "__main__":
    main()
