"""
Round 10 R10.2 — Multi-seed Table 4-MIMIC.

R9.2 의 8-method head-to-head 를 5 시드로 반복하여 McNemar pooled
significance 확보. 모든 라벨링/CRC 호출은 R9.2 의 검증된 구현 재활용.

산출: results/round10/r10_2_table4_multiseed.{json,md}
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# R9.2 의 검증된 single-seed runner 직접 호출.
from experiments.round9_table4_mimic import run_one_seed, aggregate as r9_aggregate


def write_md(report: dict, out_md: Path):
    lines = ["# Round 10 R10.2 — Multi-seed Table 4-MIMIC\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Seeds: {report['seeds']}")
    lines.append(f"- Backends: {', '.join(report['backends'])}")
    lines.append(f"- α (Table 4): {report['alpha']}\n")
    for backend, agg in report["per_backend"].items():
        lines.append(f"## backend = {backend}\n")
        lines.append("| Method | CRITICAL Recall (mean ± std) | Total Cost (mean ± std) |")
        lines.append("| --- | --- | --- |")
        for name, m in agg.items():
            cr = (f"{m['critical_recall_mean']:.4f} ± {m['critical_recall_std']:.4f}"
                  if m.get('critical_recall_mean') is not None else "—")
            tc = (f"{m['total_cost_mean']:.1f} ± {m['total_cost_std']:.1f}"
                  if m.get('total_cost_mean') is not None else "—")
            lines.append(f"| {name} | {cr} | {tc} |")
        lines.append("")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal-per-stratum", type=int, default=200)
    ap.add_argument("--n-test-per-stratum", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--backends", nargs="+", default=["lmstudio"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "r10_2_table4_multiseed")
    args = ap.parse_args()

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_cal_per_stratum": args.n_cal_per_stratum,
        "n_test_per_stratum": args.n_test_per_stratum,
        "alpha": args.alpha,
        "seeds": args.seeds,
        "backends": args.backends,
        "per_seed": {}, "per_backend": {},
    }
    for backend in args.backends:
        per_seed = []
        for seed in args.seeds:
            try:
                r = run_one_seed(backend, seed,
                                 args.n_cal_per_stratum, args.n_test_per_stratum,
                                 args.alpha, verbose=True)
                per_seed.append(r)
            except FileNotFoundError as e:
                print(f"[R10.2] preprocessed MIMIC-IV 미존재: {e}"); sys.exit(2)
        report["per_seed"][backend] = per_seed
        report["per_backend"][backend] = r9_aggregate(per_seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
