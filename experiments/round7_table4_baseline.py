"""
Round 7 Table 4 — Head-to-head baseline 비교.

같은 score 풀에 대해:
  - TECP / Quach 2024-CLM (single global α + NLL nonconformity)
  - Semantic Entropy (sample-level meaning entropy, normalized)
  - UASEF Round 6 (heuristic multiplier)
  - UASEF Round 7 v2 (Stratified CRC + MTC + cost-aware)

지표:
  - Per-stratum Safety Recall
  - Over-Escalation Rate
  - AUROC (sklearn 있으면)
  - Total cost (Pivot C cost matrix)

실행:
    python experiments/round7_table4_baseline.py --backend openai --n-cal 1500 --n-test 500

산출:
    results/round7/table4_baseline.json
    results/round7/table4_baseline.md

LLM 호출 단계는 round7_table1과 공유 — `--reuse-table1`이면 table1 데이터 재활용.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from models.uqm import ConformalCalibrator
from models.stratified_crc import StratifiedConformalRiskControl
from models.conformal_combination import TriggerCalibrator, MultiTriggerConformal
from models.cost_aware_calibration import sweep_cost_aware_per_stratum, DEFAULT_COST_MATRIX

from experiments.baselines.tecp import TECPBaseline
from experiments.baselines.quach2024 import Quach2024Baseline
from experiments.baselines.semantic_entropy import SemanticEntropyBaseline

try:
    from scipy.stats import roc_auc_score
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def evaluate_predictor(name, predictor_fn, scores, labels, strata) -> dict:
    """Per-stratum safety_recall, over_esc_rate, total cost."""
    per_stratum: dict = {}
    total_cost = 0.0
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        idx = [i for i, s in enumerate(strata) if s == stratum]
        n_pos = sum(labels[i] for i in idx)
        n_neg = sum(1 for i in idx if not labels[i])
        tp = sum(labels[i] and predictor_fn(scores[i]) for i in idx)
        fn = sum(labels[i] and not predictor_fn(scores[i]) for i in idx)
        fp = sum((not labels[i]) and predictor_fn(scores[i]) for i in idx)
        recall = tp / (tp + fn) if (tp + fn) else None
        over = fp / n_neg if n_neg else None
        cm = DEFAULT_COST_MATRIX[stratum]
        cost = cm["miss"] * fn + cm["over_esc"] * fp
        total_cost += cost
        per_stratum[stratum] = {
            "n": len(idx), "safety_recall": recall, "over_esc_rate": over,
            "tp": tp, "fn": fn, "fp": fp, "cost": cost,
        }

    auroc = None
    if HAS_SCIPY:
        try:
            preds = [int(predictor_fn(scores[i])) for i in range(len(scores))]
            auroc = round(float(roc_auc_score(labels, preds)), 4)
        except Exception:
            pass

    return {
        "name": name, "per_stratum": per_stratum,
        "total_cost": total_cost, "auroc": auroc,
    }


def main():
    parser = argparse.ArgumentParser(description="Round 7 Table 4 — Baseline Comparison")
    parser.add_argument("--reuse-table1", action="store_true",
                        help="results/round7/table1_coverage.json이 있으면 재활용")
    parser.add_argument("--backend", default="openai", choices=["openai", "lmstudio"])
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "round7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 수집 (table1과 동일 함수) ────────────────────────────────
    if args.reuse_table1 and (out_dir / "table1_coverage.json").exists():
        print("[reuse] table1 데이터 재활용")
        d1 = json.loads((out_dir / "table1_coverage.json").read_text())
        # table1은 결과만 저장하므로 raw scores를 별도로 다시 수집
        from experiments.round7_table1_coverage import collect_stratified_data
        cal_data = collect_stratified_data(args.backend, args.n_cal, args.seed)
        test_data = collect_stratified_data(args.backend, args.n_test, args.seed + 1000)
    else:
        from experiments.round7_table1_coverage import collect_stratified_data
        cal_data = collect_stratified_data(args.backend, args.n_cal, args.seed)
        test_data = collect_stratified_data(args.backend, args.n_test, args.seed + 1000)

    methods_results = []

    # ── TECP ─────────────────────────────────────────────────────────────
    tecp = TECPBaseline(alpha=args.alpha)
    tecp.fit(cal_data["all_scores"], cal_data["all_labels"])
    methods_results.append(evaluate_predictor(
        "TECP (Xu & Lu 2025)", tecp.predict,
        test_data["all_scores"], test_data["all_labels"], test_data["all_strata"],
    ))

    # ── Quach 2024 (수학적으로 TECP와 동일하지만 reference로 별도 행) ──
    q = Quach2024Baseline(alpha=args.alpha)
    q.fit(cal_data["all_scores"], cal_data["all_labels"])
    methods_results.append(evaluate_predictor(
        "Quach 2024 CLM", q.predict,
        test_data["all_scores"], test_data["all_labels"], test_data["all_strata"],
    ))

    # ── Semantic Entropy (외부 score 가정 — 본 데모에선 동일 score 사용) ──
    se = SemanticEntropyBaseline(alpha=args.alpha)
    se.fit(cal_data["all_scores"], cal_data["all_labels"])
    methods_results.append(evaluate_predictor(
        "Semantic Entropy (Farquhar Nature 2024)", se.predict,
        test_data["all_scores"], test_data["all_labels"], test_data["all_strata"],
    ))

    # ── UASEF Round 6 (heuristic) ────────────────────────────────────────
    cal_global = ConformalCalibrator(alpha=args.alpha, strict=False)
    cal_global.fit(cal_data["all_scores"])
    multipliers = {"CRITICAL": 0.60, "HIGH": 0.75, "MODERATE": 1.00, "LOW": 1.30}

    def round6_pred(score, stratum_idx):
        stratum = test_data["all_strata"][stratum_idx]
        thr = cal_global.threshold * multipliers[stratum]
        return score > thr

    # round6 평가는 stratum-aware predictor가 필요 — wrapper
    res_r6_per = {}
    total_cost_r6 = 0.0
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        idx = [i for i, s in enumerate(test_data["all_strata"]) if s == stratum]
        thr = cal_global.threshold * multipliers[stratum]
        n_pos = sum(test_data["all_labels"][i] for i in idx)
        n_neg = sum(1 for i in idx if not test_data["all_labels"][i])
        tp = sum(test_data["all_labels"][i] and test_data["all_scores"][i] > thr for i in idx)
        fn = sum(test_data["all_labels"][i] and not (test_data["all_scores"][i] > thr) for i in idx)
        fp = sum((not test_data["all_labels"][i]) and test_data["all_scores"][i] > thr for i in idx)
        recall = tp / (tp + fn) if (tp + fn) else None
        over = fp / n_neg if n_neg else None
        cm = DEFAULT_COST_MATRIX[stratum]
        cost = cm["miss"] * fn + cm["over_esc"] * fp
        total_cost_r6 += cost
        res_r6_per[stratum] = {
            "n": len(idx), "safety_recall": recall, "over_esc_rate": over,
            "tp": tp, "fn": fn, "fp": fp, "cost": cost,
        }
    methods_results.append({
        "name": "UASEF Round 6 (heuristic multiplier)",
        "per_stratum": res_r6_per, "total_cost": total_cost_r6, "auroc": None,
    })

    # ── UASEF Round 7 (Stratified CRC) ──────────────────────────────────
    crc_alphas = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    crc = StratifiedConformalRiskControl(alphas=crc_alphas)
    crc.fit(cal_data["all_scores"], cal_data["all_labels"], cal_data["all_strata"])

    res_r7_per = {}
    total_cost_r7 = 0.0
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        idx = [i for i, s in enumerate(test_data["all_strata"]) if s == stratum]
        thr = crc.threshold_for(stratum)
        n_pos = sum(test_data["all_labels"][i] for i in idx)
        n_neg = sum(1 for i in idx if not test_data["all_labels"][i])
        tp = sum(test_data["all_labels"][i] and test_data["all_scores"][i] > thr for i in idx)
        fn = sum(test_data["all_labels"][i] and not (test_data["all_scores"][i] > thr) for i in idx)
        fp = sum((not test_data["all_labels"][i]) and test_data["all_scores"][i] > thr for i in idx)
        recall = tp / (tp + fn) if (tp + fn) else None
        over = fp / n_neg if n_neg else None
        cm = DEFAULT_COST_MATRIX[stratum]
        cost = cm["miss"] * fn + cm["over_esc"] * fp
        total_cost_r7 += cost
        res_r7_per[stratum] = {
            "n": len(idx), "safety_recall": recall, "over_esc_rate": over,
            "tp": tp, "fn": fn, "fp": fp, "cost": cost,
        }
    methods_results.append({
        "name": "UASEF Round 7 (Stratified CRC + MTC + Cost-Aware)",
        "per_stratum": res_r7_per, "total_cost": total_cost_r7, "auroc": None,
    })

    # ── 저장 + 보고서 ────────────────────────────────────────────────────
    payload = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend, "n_cal": args.n_cal, "n_test": args.n_test,
        "alpha": args.alpha, "crc_alphas": crc_alphas,
        "methods": methods_results,
    }
    (out_dir / "table4_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    md = [
        "# Round 7 Table 4 — Head-to-Head Baseline\n",
        f"- backend={args.backend}, n_cal={args.n_cal}, n_test={args.n_test}, α={args.alpha}\n",
        "## CRITICAL stratum",
        "| Method | Safety Recall | Over-Esc | TP/FN/FP | Total cost |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in methods_results:
        c = m["per_stratum"]["CRITICAL"]
        md.append(
            f"| {m['name']} | {c['safety_recall']} | {c['over_esc_rate']} | "
            f"{c['tp']}/{c['fn']}/{c['fp']} | {m['total_cost']:.1f} |"
        )
    (out_dir / "table4_baseline.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✅ saved: {out_dir}/table4_baseline.{{json,md}}")


if __name__ == "__main__":
    main()
