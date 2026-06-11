"""
Round 10 R10.6 — 4-D cost matrix sensitivity sweep.

Round 7 §6.3 의 합성 4-D sweep 을 실 MIMIC-IV CRITICAL stratum 에 적용.
v2 vs `v1-cost-aware` ablation 의 cost 비교가 어떤 cost-ratio regime
에서 성립/실패 하는지 정량화.

LLM 호출 없음 — LogReg + CRC 로 빠르게 (R9.6 finding: LogReg ≈ LLM).
산출: results/round10/r10_6_cost_sweep_4d.{json,md}
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import _load_mimic4_jsonl
from experiments.round10_method_agnostic import (
    _make_classifier, _feature_vector, _attach_r10_features, _subject_id,
)
from experiments.metrics_utils import patient_level_split
from models.stratified_crc import StratifiedConformalRiskControl

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"


def _compute_cost(scores, labels, strata, thresholds: dict, cost_matrix: dict):
    total = 0.0
    per_stratum = {}
    for s in ALPHAS:
        idx = [i for i, x in enumerate(strata) if x == s]
        if not idx:
            per_stratum[s] = {"cost": 0, "n": 0}; continue
        lam = thresholds.get(s, 0.0)
        fn = sum(1 for i in idx if labels[i] and scores[i] <= lam)
        fp = sum(1 for i in idx if (not labels[i]) and scores[i] > lam)
        cm = cost_matrix.get(s, {"miss": 1, "over_esc": 1})
        cost = cm["miss"] * fn + cm["over_esc"] * fp
        total += cost
        per_stratum[s] = {"cost": cost, "fn": fn, "fp": fp, "n": len(idx)}
    return total, per_stratum


def _v1_cost_aware_thresholds(cal_scores, cal_labels, cal_strata, cost_matrix):
    """v1-cost-aware: stratum 별 cost-minimizing threshold."""
    from experiments.baselines.uasef_v1_cost import UASEFv1CostAwareBaseline
    v1c = UASEFv1CostAwareBaseline(alpha=0.10)
    v1c.fit(cal_scores, cal_labels, cal_strata)
    return {s: v1c.q_hat * v1c.multipliers.get(s, 1.0) for s in ALPHAS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="+", default=[10, 100, 1000],
                    help="miss:over_esc cost ratios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "r10_6_cost_sweep_4d")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"[R10.6] preprocessed JSONL 미존재: {args.jsonl}")

    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=args.seed)
    _attach_r10_features(cases, args.jsonl)
    print(f"[R10.6] loaded {len(cases)} cases — LogReg scoring (no LLM)")

    cal, test = patient_level_split(cases, group_of=_subject_id,
                                     cal_frac=0.8, seed=args.seed)
    clf = _make_classifier("logreg")
    X_cal = [_feature_vector(c) for c in cal]
    y_cal = [bool(c.expected_escalate) for c in cal]
    if not clf.fit(X_cal, y_cal):
        sys.exit("[R10.6] LogReg fit 실패 — calibration data 가 single-class")
    cal_scores = [clf.score(x) for x in X_cal]
    cal_labels = [bool(c.expected_escalate) for c in cal]
    cal_strata = [(c.scenario_type or "").upper() for c in cal]
    test_scores = [clf.score(_feature_vector(c)) for c in test]
    test_labels = [bool(c.expected_escalate) for c in test]
    test_strata = [(c.scenario_type or "").upper() for c in test]

    # v2 (StratifiedCRC) thresholds — cost matrix 무관
    crc = StratifiedConformalRiskControl(alphas=ALPHAS)
    crc.fit(cal_scores, cal_labels, cal_strata)
    v2_thr = {s: crc.threshold_for(s) for s in ALPHAS}

    rows = []
    for combo in itertools.product(args.grid, repeat=4):
        cm = {
            "CRITICAL": {"miss": combo[0], "over_esc": 1},
            "HIGH":     {"miss": combo[1], "over_esc": 1},
            "MODERATE": {"miss": combo[2], "over_esc": 1},
            "LOW":      {"miss": combo[3], "over_esc": 1},
        }
        v1_thr = _v1_cost_aware_thresholds(
            cal_scores, cal_labels, cal_strata, cm)
        v2_cost, _ = _compute_cost(test_scores, test_labels, test_strata,
                                    v2_thr, cm)
        v1_cost, _ = _compute_cost(test_scores, test_labels, test_strata,
                                    v1_thr, cm)
        winner = "v2" if v2_cost < v1_cost else ("v1" if v1_cost < v2_cost else "tie")
        rows.append({
            "ratios": {"CRITICAL": combo[0], "HIGH": combo[1],
                       "MODERATE": combo[2], "LOW": combo[3]},
            "v2_cost": v2_cost, "v1_cost": v1_cost,
            "winner": winner,
            "ratio_v1_v2": (v2_cost / v1_cost) if v1_cost > 0 else float("inf"),
        })

    n_v2_wins = sum(1 for r in rows if r["winner"] == "v2")
    n_v1_wins = sum(1 for r in rows if r["winner"] == "v1")
    n_tie = sum(1 for r in rows if r["winner"] == "tie")

    report = {
        "timestamp": datetime.now().isoformat(),
        "grid": args.grid,
        "n_combinations": len(rows),
        "v2_wins": n_v2_wins, "v1_wins": n_v1_wins, "ties": n_tie,
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))

    # md
    lines = ["# Round 10 R10.6 — 4-D cost matrix sweep\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Cost ratio grid: {args.grid}")
    lines.append(f"- 81 조합. v2 wins: {n_v2_wins}, v1 wins: {n_v1_wins}, ties: {n_tie}\n")
    lines.append("## v2 가 win 한 조합 top 10\n")
    lines.append("| CRIT:over | HIGH:over | MOD:over | LOW:over | v2 cost | v1 cost | ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    v2_winners = sorted([r for r in rows if r["winner"] == "v2"],
                         key=lambda r: -r["ratio_v1_v2"])[:10]
    for r in v2_winners:
        rs = r["ratios"]
        lines.append(f"| {rs['CRITICAL']}:1 | {rs['HIGH']}:1 | {rs['MODERATE']}:1 | "
                     f"{rs['LOW']}:1 | {r['v2_cost']:.0f} | {r['v1_cost']:.0f} | "
                     f"{r['ratio_v1_v2']:.2f}× |")
    lines.append("\n## v1-cost-aware 가 win 한 조합 top 10\n")
    lines.append("| CRIT:over | HIGH:over | MOD:over | LOW:over | v2 cost | v1 cost | ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    v1_winners = sorted([r for r in rows if r["winner"] == "v1"],
                         key=lambda r: r["ratio_v1_v2"])[:10]
    for r in v1_winners:
        rs = r["ratios"]
        lines.append(f"| {rs['CRITICAL']}:1 | {rs['HIGH']}:1 | {rs['MODERATE']}:1 | "
                     f"{rs['LOW']}:1 | {r['v2_cost']:.0f} | {r['v1_cost']:.0f} | "
                     f"{r['ratio_v1_v2']:.2f}× |")
    Path(str(args.out) + ".md").write_text("\n".join(lines))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
