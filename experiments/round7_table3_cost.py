"""
Round 7 Table 3 — Cost-Weighted Performance (Pivot C).

비대칭 cost matrix를 적용했을 때 total cost / Safety Recall / Over-Esc rate가
어떻게 바뀌는지 확인. F1-symmetric (Round 6) vs cost-aware (Round 7) 비교.

실행:
    python experiments/round7_table3_cost.py

LLM 호출 없이 합성 데이터로 동작.

산출:
    results/round7/table3_cost.json
    results/round7/table3_cost.md
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

from models.cost_aware_calibration import (
    find_cost_optimal_threshold, sweep_cost_aware_per_stratum,
    cost_ratio_sweep, DEFAULT_COST_MATRIX,
)


def f1_symmetric_threshold(scores, labels) -> float:
    """Round 6 baseline: F1-safety = harmonic_mean(recall, 1-over_esc) 최대화."""
    candidates = sorted(set(scores))
    best_thr, best_f1 = candidates[0], -1.0
    for thr in candidates:
        preds = [s > thr for s in scores]
        tp = sum(p and l for p, l in zip(preds, labels))
        fn = sum(not p and l for p, l in zip(preds, labels))
        fp = sum(p and not l for p, l in zip(preds, labels))
        tn = sum(not p and not l for p, l in zip(preds, labels))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        over = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * recall * (1 - over) / (recall + (1 - over) + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def total_cost(scores, labels, threshold, c_miss, c_over):
    fn = sum(1 for s, l in zip(scores, labels) if l and s <= threshold)
    fp = sum(1 for s, l in zip(scores, labels) if (not l) and s > threshold)
    return c_miss * fn + c_over * fp


def _generate_synthetic(n_per_stratum: int, seed: int) -> tuple[dict, dict]:
    """Synthetic 4-stratum (score, label) generator used by both 1-D and 4-D sweep."""
    random.seed(seed)
    sb, lb = {}, {}
    for stratum, base in [("CRITICAL", 0.30), ("HIGH", 0.20),
                          ("MODERATE", 0.10), ("LOW", 0.05)]:
        sb[stratum], lb[stratum] = [], []
        for _ in range(n_per_stratum):
            l = random.random() < base
            s = random.gauss(2.0 if l else 0.0, 1.0)
            sb[stratum].append(s); lb[stratum].append(l)
    return sb, lb


def run_4d_sweep(n_per_stratum: int, seed: int, ratios: list[int]) -> dict:
    """
    4-D Cartesian sweep over (CRITICAL, HIGH, MODERATE, LOW) miss:over cost
    ratios. For each combination of ratios, fit the cost-aware optimizer +
    a F1-symmetric Round-6 baseline, and record total cost / cost-reduction
    ratio.

    Args:
        n_per_stratum: synthetic n per stratum.
        seed: RNG seed.
        ratios: candidate miss-cost values (over_esc cost fixed at 1).
                e.g. [10, 100, 1000] → 3^4 = 81 combinations.

    Returns:
        dict with `combinations`: list of result rows + summary stats.
    """
    sb, lb = _generate_synthetic(n_per_stratum, seed)
    alphas = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}

    rows = []
    n_total = len(ratios) ** 4
    print(f"[Phase 4D] sweeping {n_total} combinations ...")
    counter = 0
    for r_crit in ratios:
        for r_high in ratios:
            for r_mod in ratios:
                for r_low in ratios:
                    counter += 1
                    cost_matrix = {
                        "CRITICAL":  {"miss": r_crit, "over_esc": 1.0},
                        "HIGH":      {"miss": r_high, "over_esc": 1.0},
                        "MODERATE":  {"miss": r_mod,  "over_esc": 1.0},
                        "LOW":       {"miss": r_low,  "over_esc": 1.0},
                    }
                    # Round 6: F1-symmetric per stratum (cost-agnostic)
                    r6_total = 0.0
                    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
                        thr = f1_symmetric_threshold(sb[stratum], lb[stratum])
                        cm = cost_matrix[stratum]
                        r6_total += total_cost(sb[stratum], lb[stratum], thr, cm["miss"], cm["over_esc"])
                    # Round 7: cost-aware
                    out = sweep_cost_aware_per_stratum(sb, lb, cost_matrix, alphas)
                    r7_total = sum(r.cost for r in out.values())
                    ratio = (r6_total / r7_total) if r7_total > 0 else None
                    rows.append({
                        "ratios": {"CRITICAL": r_crit, "HIGH": r_high,
                                   "MODERATE": r_mod, "LOW": r_low},
                        "round6_total_cost": r6_total,
                        "round7_total_cost": r7_total,
                        "reduction_ratio": round(ratio, 2) if ratio else None,
                    })

    # summary stats
    valid = [r["reduction_ratio"] for r in rows if r["reduction_ratio"] is not None]
    summary = {
        "n_combinations": len(rows),
        "ratio_grid": ratios,
        "min_reduction": min(valid) if valid else None,
        "max_reduction": max(valid) if valid else None,
        "median_reduction": sorted(valid)[len(valid)//2] if valid else None,
        "mean_reduction": round(sum(valid)/len(valid), 2) if valid else None,
    }
    return {"combinations": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Round 7 Table 3 — Cost-Weighted Performance")
    parser.add_argument("--n-per-stratum", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep-grid",
        choices=["1d", "4d"],
        default="1d",
        help="1d: original CRITICAL ratio sweep (Table 3 main). "
             "4d: Cartesian sweep over all 4 strata (3^4=81 by default).",
    )
    parser.add_argument(
        "--ratios",
        type=int,
        nargs="+",
        default=[10, 100, 1000],
        help="Miss-cost candidate values for 4-D sweep (over_esc cost fixed at 1).",
    )
    args = parser.parse_args()

    out_dir = ROOT / "results" / "round7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 4-D Cartesian sweep mode ──
    if args.sweep_grid == "4d":
        result = run_4d_sweep(args.n_per_stratum, args.seed, args.ratios)
        payload_4d = {
            "timestamp": datetime.now().isoformat(),
            "mode": "4d",
            "n_per_stratum": args.n_per_stratum,
            "seed": args.seed,
            "ratios": args.ratios,
            **result,
        }
        (out_dir / "table3_cost_4d.json").write_text(
            json.dumps(payload_4d, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        s = result["summary"]
        md = [
            "# Round 7 Table 3 — 4-D Cost-Matrix Sweep\n",
            f"- n_per_stratum={args.n_per_stratum}, seed={args.seed}, "
            f"ratio grid={args.ratios} → {s['n_combinations']} combinations\n",
            "## Reduction-ratio statistics\n",
            f"- min: **{s['min_reduction']}×**",
            f"- median: **{s['median_reduction']}×**",
            f"- mean: **{s['mean_reduction']}×**",
            f"- max: **{s['max_reduction']}×**",
            "",
            "## Per-combination (top 10 by reduction)",
            "| CRIT:over | HIGH:over | MOD:over | LOW:over | R6 cost | R7 cost | reduction |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        rows_sorted = sorted(
            (r for r in result["combinations"] if r["reduction_ratio"] is not None),
            key=lambda r: -r["reduction_ratio"],
        )
        for row in rows_sorted[:10]:
            r = row["ratios"]
            md.append(
                f"| {r['CRITICAL']}:1 | {r['HIGH']}:1 | {r['MODERATE']}:1 | {r['LOW']}:1 | "
                f"{row['round6_total_cost']:.0f} | {row['round7_total_cost']:.0f} | "
                f"{row['reduction_ratio']}× |"
            )
        (out_dir / "table3_cost_4d.md").write_text("\n".join(md), encoding="utf-8")
        print("\n".join(md))
        print(f"\n✅ saved: {out_dir}/table3_cost_4d.{{json,md}}")
        return

    # ── 1-D mode (original Table 3) ──
    # 합성 데이터: 4 stratum × n_per
    sb, lb = _generate_synthetic(args.n_per_stratum, args.seed)

    # ── Round 6: F1-symmetric per stratum ────────────────────────────────
    round6_rows = []
    round6_total_cost = 0.0
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        thr = f1_symmetric_threshold(sb[stratum], lb[stratum])
        cm = DEFAULT_COST_MATRIX[stratum]
        c = total_cost(sb[stratum], lb[stratum], thr, cm["miss"], cm["over_esc"])
        round6_total_cost += c
        round6_rows.append({"stratum": stratum, "threshold": round(thr, 3), "cost": c})

    # ── Round 7: cost-aware per stratum ──────────────────────────────────
    alphas = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    out = sweep_cost_aware_per_stratum(sb, lb, DEFAULT_COST_MATRIX, alphas)
    round7_rows = []
    round7_total_cost = 0.0
    for stratum, r in out.items():
        round7_total_cost += r.cost
        round7_rows.append({
            "stratum": stratum, "threshold": round(r.threshold, 3),
            "cost": r.cost, "miss_rate": r.miss_rate,
            "over_esc_rate": r.over_esc_rate,
        })

    payload = {
        "timestamp": datetime.now().isoformat(),
        "n_per_stratum": args.n_per_stratum,
        "cost_matrix": DEFAULT_COST_MATRIX,
        "round6_total_cost": round6_total_cost,
        "round7_total_cost": round7_total_cost,
        "cost_reduction_ratio": round(round6_total_cost / round7_total_cost, 2)
                                 if round7_total_cost > 0 else None,
        "round6_per_stratum": round6_rows,
        "round7_per_stratum": round7_rows,
    }
    (out_dir / "table3_cost.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Round 7 Table 3 — Cost-Weighted Performance\n",
        f"- n_per_stratum={args.n_per_stratum}, default cost matrix\n",
        f"- **Round 6 total cost**: {round6_total_cost:.1f}",
        f"- **Round 7 total cost**: {round7_total_cost:.1f}",
        f"- **Reduction**: {payload['cost_reduction_ratio']}× (Round 6 / Round 7)\n",
        "## Per-stratum",
        "| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r6, r7 in zip(round6_rows, round7_rows):
        md.append(
            f"| {r6['stratum']} | {r6['threshold']} | {r6['cost']:.1f} | "
            f"{r7['threshold']} | {r7['cost']:.1f} | "
            f"{r7['miss_rate']} | {r7['over_esc_rate']} |"
        )
    (out_dir / "table3_cost.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✅ saved: {out_dir}/table3_cost.{{json,md}}")


if __name__ == "__main__":
    main()
