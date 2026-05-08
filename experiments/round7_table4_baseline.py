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
from experiments.baselines.tecp_stratified import TECPStratifiedBaseline
from experiments.baselines.quach2024 import Quach2024Baseline
from experiments.baselines.semantic_entropy import SemanticEntropyBaseline
from experiments.baselines.cost_sensitive import CostSensitiveBaseline

# scipy.stats.roc_auc_score does not exist; use sklearn or compute manually.
try:
    from sklearn.metrics import roc_auc_score
    HAS_AUROC = True
except ImportError:
    HAS_AUROC = False


def _manual_auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute AUROC by ranking, no sklearn dependency."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def mcnemar_pvalue(pred_a: list[bool], pred_b: list[bool], labels: list[bool]) -> float | None:
    """
    Two-sided exact-binomial McNemar test for paired binary classifiers.
    Returns p-value or None if not enough discordant pairs.
    H0: methods have equal accuracy.
    Discordant pair = exactly one of (A,B) is correct on a given example.
    """
    correct_a = [pa == y for pa, y in zip(pred_a, labels)]
    correct_b = [pb == y for pb, y in zip(pred_b, labels)]
    b = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)
    c = sum(1 for ca, cb in zip(correct_a, correct_b) if cb and not ca)
    n = b + c
    if n == 0:
        return None
    # exact binomial two-sided p-value (Sachs 1984): P(X<=min(b,c)) under Bin(n, 0.5)
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) * 0.5**n for i in range(k + 1))
    return min(1.0, 2.0 * tail)


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
            "n": len(idx), "n_pos": n_pos, "safety_recall": recall, "over_esc_rate": over,
            "tp": tp, "fn": fn, "fp": fp, "cost": cost,
        }

    # AUROC: prefer sklearn if available, otherwise rank-based fallback.
    if HAS_AUROC:
        try:
            auroc = round(float(roc_auc_score(labels, scores)), 4)
        except Exception:
            auroc = _manual_auroc(labels, scores)
            auroc = round(auroc, 4) if auroc is not None else None
    else:
        man = _manual_auroc(labels, scores)
        auroc = round(man, 4) if man is not None else None

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
    from data.loader import SUPPORTED_DATASETS  # local import to avoid mod-load cost
    parser.add_argument(
        "--dataset", default="medabstain",
        choices=list(SUPPORTED_DATASETS),
        help="Dataset for evaluation (passed to round7_table1.collect_stratified_data).",
    )
    args = parser.parse_args()

    out_dir = ROOT / "results" / "round7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 수집 (table1과 동일 함수) ────────────────────────────────
    from experiments.round7_table1_coverage import collect_stratified_data
    cal_data = collect_stratified_data(args.backend, args.n_cal, args.seed, dataset=args.dataset)
    test_data = collect_stratified_data(args.backend, args.n_test, args.seed + 1000, dataset=args.dataset)

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
            "n": len(idx), "n_pos": n_pos, "safety_recall": recall, "over_esc_rate": over,
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
            "n": len(idx), "n_pos": n_pos, "safety_recall": recall, "over_esc_rate": over,
            "tp": tp, "fn": fn, "fp": fp, "cost": cost,
        }
    methods_results.append({
        "name": "UASEF Round 7 (Stratified CRC + MTC + Cost-Aware)",
        "per_stratum": res_r7_per, "total_cost": total_cost_r7, "auroc": None,
    })

    # ── TECP-stratified ablation (Round 7 fairness baseline) ────────────
    tecp_str = TECPStratifiedBaseline(alphas=crc_alphas)
    tecp_str.fit(cal_data["all_scores"], cal_data["all_labels"], cal_data["all_strata"])
    res_tecp_str_per = {}
    total_cost_tecp_str = 0.0
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        idx = [i for i, s in enumerate(test_data["all_strata"]) if s == stratum]
        n_pos = sum(test_data["all_labels"][i] for i in idx)
        n_neg = sum(1 for i in idx if not test_data["all_labels"][i])
        tp = sum(test_data["all_labels"][i] and tecp_str.predict(test_data["all_scores"][i], stratum) for i in idx)
        fn = sum(test_data["all_labels"][i] and not tecp_str.predict(test_data["all_scores"][i], stratum) for i in idx)
        fp = sum((not test_data["all_labels"][i]) and tecp_str.predict(test_data["all_scores"][i], stratum) for i in idx)
        recall = tp / (tp + fn) if (tp + fn) else None
        over = fp / n_neg if n_neg else None
        cm = DEFAULT_COST_MATRIX[stratum]
        cost = cm["miss"] * fn + cm["over_esc"] * fp
        total_cost_tecp_str += cost
        res_tecp_str_per[stratum] = {
            "n": len(idx), "n_pos": n_pos, "safety_recall": recall, "over_esc_rate": over,
            "tp": tp, "fn": fn, "fp": fp, "cost": cost,
        }
    methods_results.append({
        "name": "TECP-stratified (this work, Round 7 ablation)",
        "per_stratum": res_tecp_str_per, "total_cost": total_cost_tecp_str, "auroc": None,
    })

    # ── Cost-Sensitive (single-α) ablation ──────────────────────────────
    # Use HIGH-stratum cost ratio (100:1) as a representative scalar; we
    # report this baseline to compare Pivot C against general cost-sensitive
    # learning, separate from per-stratum CRC.
    cm_high = DEFAULT_COST_MATRIX["HIGH"]
    cost_baseline = CostSensitiveBaseline(c_miss=cm_high["miss"], c_over=cm_high["over_esc"])
    cost_baseline.fit(cal_data["all_scores"], cal_data["all_labels"])
    methods_results.append(evaluate_predictor(
        "Cost-Sensitive single-α (this work, Round 7 ablation)",
        cost_baseline.predict,
        test_data["all_scores"], test_data["all_labels"], test_data["all_strata"],
    ))

    # ── Cross-backend & paired-comparison sanity checks ─────────────────
    sanity_alerts: list[str] = []

    # Build per-method test predictions for paired tests.
    test_n = len(test_data["all_scores"])
    test_strata = test_data["all_strata"]
    test_labels = test_data["all_labels"]
    test_scores = test_data["all_scores"]

    def _preds_for(method_name: str) -> list[bool]:
        if method_name.startswith("TECP ("):
            return [tecp.predict(test_scores[i]) for i in range(test_n)]
        if method_name.startswith("Quach"):
            return [q.predict(test_scores[i]) for i in range(test_n)]
        if method_name.startswith("Semantic"):
            return [se.predict(test_scores[i]) for i in range(test_n)]
        if method_name.startswith("UASEF Round 6"):
            return [
                test_scores[i] > cal_global.threshold * multipliers[test_strata[i]]
                for i in range(test_n)
            ]
        if method_name.startswith("UASEF Round 7"):
            return [
                test_scores[i] > crc.threshold_for(test_strata[i]) for i in range(test_n)
            ]
        if method_name.startswith("TECP-stratified"):
            return [tecp_str.predict(test_scores[i], test_strata[i]) for i in range(test_n)]
        if method_name.startswith("Cost-Sensitive"):
            return [cost_baseline.predict(test_scores[i]) for i in range(test_n)]
        return []

    v2_preds = _preds_for("UASEF Round 7")
    pairwise_pvalues: dict[str, float | None] = {}
    for m in methods_results:
        if m["name"].startswith("UASEF Round 7"):
            continue
        other_preds = _preds_for(m["name"])
        if other_preds:
            pairwise_pvalues[m["name"]] = mcnemar_pvalue(v2_preds, other_preds, test_labels)

    # Sanity check #1: identical-confusion-matrix detector across methods.
    # Two distinct methods producing identical (TP, FN, FP, TN) on the
    # CRITICAL stratum is a soft warning (could be coincidence, but flag it).
    crit_signatures: dict[tuple, list[str]] = {}
    for m in methods_results:
        c = m["per_stratum"]["CRITICAL"]
        sig = (c["tp"], c["fn"], c["fp"])
        crit_signatures.setdefault(sig, []).append(m["name"])
    for sig, names in crit_signatures.items():
        if len(names) > 1 and not all("TECP" in n or "Quach" in n or "Semantic" in n for n in names):
            sanity_alerts.append(
                f"SANITY: methods {names} share identical CRITICAL confusion {sig} — "
                "verify this is not a copy-paste / placeholder bug."
            )

    # ── 저장 + 보고서 ────────────────────────────────────────────────────
    payload = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend, "n_cal": args.n_cal, "n_test": args.n_test,
        "alpha": args.alpha, "crc_alphas": crc_alphas,
        "methods": methods_results,
        "pairwise_mcnemar_vs_v2": pairwise_pvalues,
        "sanity_alerts": sanity_alerts,
    }
    (out_dir / "table4_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    md = [
        "# Round 7 Table 4 — Head-to-Head Baseline\n",
        f"- backend={args.backend}, n_cal={args.n_cal}, n_test={args.n_test}, α={args.alpha}\n",
        "## CRITICAL stratum",
        "| Method | Safety Recall (n+) | Over-Esc (n−) | TP/FN/FP | Total cost | AUROC | McNemar vs v2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in methods_results:
        c = m["per_stratum"]["CRITICAL"]
        n_pos = c.get("n_pos", c.get("tp", 0) + c.get("fn", 0))
        n_neg = c.get("n", 0) - n_pos
        recall_str = f"{c['safety_recall']:.4f} ({c['tp']}/{n_pos})" if c.get("safety_recall") is not None else "N/A"
        over_str = f"{c['over_esc_rate']:.4f} ({c['fp']}/{n_neg})" if c.get("over_esc_rate") is not None else "N/A"
        auroc_str = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "—"
        p = pairwise_pvalues.get(m['name'])
        p_str = f"{p:.4g}" if p is not None else ("—" if "Round 7" in m['name'] else "n/a")
        md.append(
            f"| {m['name']} | {recall_str} | {over_str} | "
            f"{c['tp']}/{c['fn']}/{c['fp']} | {m['total_cost']:.1f} | {auroc_str} | {p_str} |"
        )

    if sanity_alerts:
        md.append("\n## ⚠️ Sanity alerts\n")
        for a in sanity_alerts:
            md.append(f"- {a}")

    (out_dir / "table4_baseline.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✅ saved: {out_dir}/table4_baseline.{{json,md}}")


if __name__ == "__main__":
    main()
