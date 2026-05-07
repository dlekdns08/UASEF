"""
Round 7 Table 1 — Per-Stratum Coverage Validity (Pivot A 검증).

UASEF Round 7의 stratified CRC가 실제로 stratum별 보장을 충족하는지 검증.
Round 6 (heuristic multiplier)와 TECP (single global α)와 비교.

실행:
    python experiments/round7_table1_coverage.py --backend openai --n-cal 1500 --n-test 500

산출:
    results/round7/table1_coverage.json
    results/round7/table1_coverage.md
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

from models.uqm import UQM, ConformalCalibrator, compute_nonconformity_score
from models.stratified_crc import StratifiedConformalRiskControl
from models.model_interface import query_model
from data.loader import load_calibration_questions, load_scenarios
from experiments.baselines.tecp import TECPBaseline


SPECIALTY_TO_STRATUM = {
    "emergency_medicine": "CRITICAL", "intensive_care": "CRITICAL", "trauma_surgery": "CRITICAL",
    "cardiology": "HIGH", "neurology": "HIGH", "oncology": "HIGH", "cardiothoracic_surgery": "HIGH",
    "internal_medicine": "MODERATE", "surgery": "MODERATE",
    "pediatrics": "MODERATE", "obstetrics": "MODERATE",
    "general_practice": "LOW", "preventive_medicine": "LOW",
    "dermatology": "LOW", "psychiatry": "LOW",
}


def collect_stratified_data(backend: str, n_per_stratum: int, seed: int) -> dict:
    """LLM 호출로 stratum별 (score, label) 수집."""
    print(f"\n[Phase 1] Calibration scores 수집 ...")
    sys_prompt = UQM.SYSTEM_PROMPT
    scenario_map = load_scenarios(n_per_scenario=n_per_stratum, split="test", seed=seed)

    scores_by_stratum: dict[str, list[float]] = {s: [] for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]}
    labels_by_stratum: dict[str, list[bool]] = {s: [] for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]}

    all_scores, all_labels, all_strata = [], [], []
    for scenario_type, cases in scenario_map.items():
        for case in cases:
            stratum = SPECIALTY_TO_STRATUM.get(case.specialty, "MODERATE")
            try:
                resp = query_model(backend, sys_prompt, case.question, temperature=0.0)
                score = compute_nonconformity_score(resp)
            except Exception as e:
                print(f"  [skip] {e}")
                continue
            scores_by_stratum[stratum].append(score)
            labels_by_stratum[stratum].append(case.expected_escalate)
            all_scores.append(score)
            all_labels.append(case.expected_escalate)
            all_strata.append(stratum)

    return {
        "scores_by_stratum": scores_by_stratum,
        "labels_by_stratum": labels_by_stratum,
        "all_scores": all_scores,
        "all_labels": all_labels,
        "all_strata": all_strata,
    }


def evaluate_method(name: str, predictor_fn, scores: list, labels: list, strata: list) -> dict:
    """주어진 prediction function의 stratum별 empirical miss rate 측정."""
    per_stratum: dict = {}
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        idx = [i for i, s in enumerate(strata) if s == stratum]
        n = len(idx)
        if n == 0:
            per_stratum[stratum] = {"n": 0, "miss_rate": None}
            continue
        misses = 0
        n_pos = 0
        for i in idx:
            if labels[i]:
                n_pos += 1
                if not predictor_fn(scores[i]):    # 실제 양성인데 미트리거
                    misses += 1
        miss_rate = misses / n_pos if n_pos > 0 else 0.0
        per_stratum[stratum] = {"n": n, "n_pos": n_pos,
                                 "miss_rate": round(miss_rate, 4)}
    return {"name": name, "per_stratum": per_stratum}


def main():
    parser = argparse.ArgumentParser(description="Round 7 Table 1 — Coverage Validity")
    parser.add_argument("--backend", default="openai", choices=["openai", "lmstudio"])
    parser.add_argument("--n-cal", type=int, default=200,
                        help="Calibration n per stratum (논문 권장 ≥1000 for α=0.001)")
    parser.add_argument("--n-test", type=int, default=100, help="Test n per stratum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-global", type=float, default=0.10,
                        help="TECP / Round 6 single α")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "round7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 수집 ──────────────────────────────────────────────────────
    cal_data = collect_stratified_data(args.backend, args.n_cal, seed=args.seed)
    test_data = collect_stratified_data(args.backend, args.n_test, seed=args.seed + 1000)

    # ── 1. TECP / Quach (single global α) ────────────────────────────────
    tecp = TECPBaseline(alpha=args.alpha_global)
    tecp.fit(cal_data["all_scores"], cal_data["all_labels"])
    res_tecp = evaluate_method(
        "TECP / Quach 2024 (global α)", tecp.predict,
        test_data["all_scores"], test_data["all_labels"], test_data["all_strata"],
    )

    # ── 2. UASEF Round 6 (heuristic multiplier) ──────────────────────────
    cal_global = ConformalCalibrator(alpha=args.alpha_global, strict=False)
    cal_global.fit(cal_data["all_scores"])
    multipliers = {"CRITICAL": 0.60, "HIGH": 0.75, "MODERATE": 1.00, "LOW": 1.30}

    def round6_predict_factory(stratum):
        thr = cal_global.threshold * multipliers[stratum]
        return lambda s: s > thr

    res_round6 = {"name": "UASEF Round 6 (heuristic multiplier)", "per_stratum": {}}
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        pred = round6_predict_factory(stratum)
        idx = [i for i, s in enumerate(test_data["all_strata"]) if s == stratum]
        n_pos = sum(test_data["all_labels"][i] for i in idx)
        misses = sum(test_data["all_labels"][i] and not pred(test_data["all_scores"][i]) for i in idx)
        miss = misses / n_pos if n_pos else 0.0
        res_round6["per_stratum"][stratum] = {
            "n": len(idx), "n_pos": n_pos,
            "miss_rate": round(miss, 4),
        }

    # ── 3. UASEF Round 7 (Stratified CRC) ────────────────────────────────
    crc_alphas = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    crc = StratifiedConformalRiskControl(alphas=crc_alphas)
    crc.fit(cal_data["all_scores"], cal_data["all_labels"], cal_data["all_strata"])
    res_round7 = {"name": "UASEF Round 7 (Stratified CRC)", "per_stratum": {}}
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        thr = crc.threshold_for(stratum)
        pred = lambda s, t=thr: s > t
        idx = [i for i, s in enumerate(test_data["all_strata"]) if s == stratum]
        n_pos = sum(test_data["all_labels"][i] for i in idx)
        misses = sum(test_data["all_labels"][i] and not pred(test_data["all_scores"][i]) for i in idx)
        miss = misses / n_pos if n_pos else 0.0
        res_round7["per_stratum"][stratum] = {
            "n": len(idx), "n_pos": n_pos,
            "miss_rate": round(miss, 4),
            "target_alpha": crc_alphas[stratum],
            "ok": miss <= crc_alphas[stratum] * 1.1,
        }

    # ── 저장 + 보고서 ────────────────────────────────────────────────────
    payload = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend,
        "n_cal": args.n_cal, "n_test": args.n_test,
        "alpha_global": args.alpha_global,
        "crc_alphas": crc_alphas,
        "results": [res_tecp, res_round6, res_round7],
    }
    (out_dir / "table1_coverage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # markdown report
    md = ["# Round 7 Table 1 — Per-Stratum Coverage Validity\n",
          f"- backend: `{args.backend}`, n_cal={args.n_cal}/stratum, n_test={args.n_test}/stratum\n",
          "| Method | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss |",
          "| --- | --- | --- | --- | --- |"]
    for r in [res_tecp, res_round6, res_round7]:
        row = f"| {r['name']} |"
        for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            v = r["per_stratum"][s].get("miss_rate")
            row += f" {v if v is not None else 'N/A'} |"
        md.append(row)
    (out_dir / "table1_coverage.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "\n".join(md))
    print(f"\n✅ saved: {out_dir}/table1_coverage.{{json,md}}")


if __name__ == "__main__":
    main()
