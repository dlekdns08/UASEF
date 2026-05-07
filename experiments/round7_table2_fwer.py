"""
Round 7 Table 2 — Multi-Trigger Combination FWER 검증 (Pivot B).

Null hypothesis (no trigger should fire) 하에서 각 combination 방법의
empirical FWER (false escalation rate) 측정. v1의 `len(triggers)>0`은 위반,
Round 7 (harmonic / e_value)는 보존을 검증.

실행:
    python experiments/round7_table2_fwer.py --n-trials 5000 --alpha 0.05

이 스크립트는 LLM 호출 없이 합성 데이터로 동작 — 즉시 실행 가능.

산출:
    results/round7/table2_fwer.json
    results/round7/table2_fwer.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.conformal_combination import (
    TriggerCalibrator, MultiTriggerConformal, COMBINATION_METHODS,
)


def measure_fwer(
    method: str, n_trials: int, alpha: float,
    n_cal: int, dependence: str = "independent", seed: int = 42,
) -> float:
    """
    Null hypothesis 하의 empirical FWER 측정.

    dependence:
      "independent": 3개 trigger 모두 독립 N(0,1) score
      "correlated":  공통 latent factor 0.5 × global + N(0,1) noise (mid-level dep)
    """
    rng = random.Random(seed)
    cals = []
    for name in ("T1", "T2", "T3"):
        cal = TriggerCalibrator(name)
        cal.fit([rng.gauss(0, 1) for _ in range(n_cal)])
        cals.append(cal)
    mtc = MultiTriggerConformal(cals, combination=method)

    rejections = 0
    for _ in range(n_trials):
        if dependence == "independent":
            scores = [rng.gauss(0, 1) for _ in range(3)]
        else:
            shared = rng.gauss(0, 1)
            scores = [0.5 * shared + rng.gauss(0, 1) for _ in range(3)]
        if mtc.should_escalate(scores, alpha=alpha)[0]:
            rejections += 1
    return rejections / n_trials


def measure_naive_or_fwer(
    n_trials: int, alpha: float, n_cal: int,
    dependence: str = "independent", seed: int = 42,
) -> float:
    """v1의 len(triggers)>0 식 (Bonferroni 미적용 OR) 시뮬레이션."""
    rng = random.Random(seed)
    cals = []
    for _ in range(3):
        cals.append(sorted([rng.gauss(0, 1) for _ in range(n_cal)]))

    def conformal_p(score, cal):
        n_geq = sum(1 for s in cal if s >= score)
        return (1 + n_geq) / (len(cal) + 1)

    rejections = 0
    for _ in range(n_trials):
        if dependence == "independent":
            scores = [rng.gauss(0, 1) for _ in range(3)]
        else:
            shared = rng.gauss(0, 1)
            scores = [0.5 * shared + rng.gauss(0, 1) for _ in range(3)]
        # naive OR: 어느 한 trigger라도 p_i ≤ α이면 fire
        if any(conformal_p(s, cals[i]) <= alpha for i, s in enumerate(scores)):
            rejections += 1
    return rejections / n_trials


def main():
    parser = argparse.ArgumentParser(description="Round 7 Table 2 — FWER under null")
    parser.add_argument("--n-trials", type=int, default=5000)
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "round7"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== FWER 검증 (n_trials={args.n_trials}, α={args.alpha}) ===\n")
    rows = []
    for dep in ("independent", "correlated"):
        # naive (v1)
        naive_fwer = measure_naive_or_fwer(
            n_trials=args.n_trials, alpha=args.alpha,
            n_cal=args.n_cal, dependence=dep, seed=args.seed,
        )
        rows.append({
            "method": "v1: len(triggers) > 0 (naive OR)",
            "dependence": dep,
            "empirical_fwer": round(naive_fwer, 4),
            "target": args.alpha,
            "ok": naive_fwer <= args.alpha + 0.02,
        })
        # 3 combination methods
        for method in COMBINATION_METHODS:
            fwer = measure_fwer(
                method=method, n_trials=args.n_trials,
                alpha=args.alpha, n_cal=args.n_cal,
                dependence=dep, seed=args.seed,
            )
            rows.append({
                "method": f"v2: {method}",
                "dependence": dep,
                "empirical_fwer": round(fwer, 4),
                "target": args.alpha,
                "ok": fwer <= args.alpha + 0.02,
            })

    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_trials": args.n_trials, "n_cal": args.n_cal,
                    "alpha": args.alpha, "seed": args.seed},
        "rows": rows,
    }
    (out_dir / "table2_fwer.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Round 7 Table 2 — Multi-Trigger Combination FWER\n",
        f"- n_trials={args.n_trials}, n_cal={args.n_cal}, target α={args.alpha}\n",
        "| Method | Dependence | Empirical FWER | OK (≤α+0.02)? |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['dependence']} "
            f"| {r['empirical_fwer']} | {'✓' if r['ok'] else '✗'} |"
        )
    (out_dir / "table2_fwer.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✅ saved: {out_dir}/table2_fwer.{{json,md}}")


if __name__ == "__main__":
    main()
