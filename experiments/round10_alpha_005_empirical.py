"""
Round 10 R10.1 — Properly-powered α=0.05 empirical validation (Table 1d).

R9.1 의 정직한 후속: n_pos≥3000 으로 cohort 확장하여 exact Clopper-
Pearson 95% upper bound 가 진짜로 α=0.05 만족하는지 검증. R9.1 의
n_pos=99 / 0.030 upper 한계 극복.

산출: results/round10/r10_1_alpha_005_empirical.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# round10_method_agnostic 의 LLM 평가 함수를 재활용 — alphas 만 다름.
from experiments.round10_method_agnostic import (
    evaluate_one_seed, aggregate_across_seeds, _attach_r10_features,
    write_md as _write_md_base,
)
from data.loader import _load_mimic4_jsonl

DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--backends", nargs="+", default=["lmstudio"])
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "r10_1_alpha_005_empirical")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"[R10.1] preprocessed JSONL 미존재: {args.jsonl}")

    print(f"[R10.1] loading {args.jsonl} ...")
    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=42)
    _attach_r10_features(cases, args.jsonl)
    print(f"[R10.1] loaded {len(cases)} cases — running LLM (lmstudio gpt-oss-120b)")

    from datetime import datetime
    report = {
        "timestamp": datetime.now().isoformat(),
        "classifiers": ["gpt_oss_120b"],
        "n_cal": args.n_cal, "n_test": args.n_test,
        "seeds": args.seeds,
        "per_seed": {}, "per_classifier": {},
    }
    per_seed = []
    for seed in args.seeds:
        r = evaluate_one_seed("gpt_oss_120b", cases, seed,
                               args.n_cal, args.n_test)
        per_seed.append(r)
    report["per_seed"]["gpt_oss_120b"] = per_seed
    report["per_classifier"]["gpt_oss_120b"] = aggregate_across_seeds(per_seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(
        json.dumps(report, indent=2, default=str))
    _write_md_base(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
