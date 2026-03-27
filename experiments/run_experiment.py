"""
UASEF — 실험 실행기
LMStudio(로컬) vs OpenAI를 3가지 시나리오에서 비교 평가합니다.

실행 방법:
    python experiments/run_experiment.py

출력:
    results/experiment_results.json
    results/comparison_table.csv
"""

import json
import csv
import os
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 PYTHONPATH에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.model_interface import query_model
from models.uqm import UQM
from models.rtc_ede import RTC, EDE
from data.loader import (
    load_calibration_questions,
    load_scenarios,
    case_to_experiment_dict,
)


# ── 실험 데이터셋 — MedQA / MedAbstain ────────────────────────────────────────
# HuggingFace 자동 다운로드 또는 data/raw/ 로컬 JSONL 사용.
# 로컬 파일:
#   data/raw/medqa_train.jsonl  (jind11/MedQA 포맷)
#   data/raw/medqa_test.jsonl
#   data/raw/medabstain_AP.jsonl
#   data/raw/medabstain_NAP.jsonl

def _build_datasets(
    n_calibration: int = 30,
    n_per_scenario: int = 3,
    seed: int = 42,
) -> tuple[list[str], dict]:
    """
    Calibration 질문과 시나리오 케이스를 MedQA에서 로드합니다.
    논문 품질 실험에는 n_calibration=500, n_per_scenario=50 권장.
    """
    print("[Dataset] MedQA / MedAbstain 로드 중...")
    cal_questions = load_calibration_questions(n=n_calibration, split="train", seed=seed)

    scenario_map = load_scenarios(n_per_scenario=n_per_scenario, split="test", seed=seed)

    # run_experiment.py의 SCENARIOS 포맷으로 변환
    specialty_map = {
        "emergency": "emergency_medicine",
        "rare_disease": "neurology",
        "multimorbidity": "internal_medicine",
        "routine": "general_practice",
    }
    scenarios = {}
    for scenario_type, cases in scenario_map.items():
        scenarios[scenario_type] = {
            "specialty": specialty_map.get(scenario_type, "internal_medicine"),
            "scenario_type": scenario_type,
            "cases": [case_to_experiment_dict(c) for c in cases],
        }

    return cal_questions, scenarios


# 기본값: 빠른 실행용 (논문 실험 시 아래 값을 늘릴 것)
CALIBRATION_QUESTIONS, SCENARIOS = _build_datasets(
    n_calibration=30,
    n_per_scenario=3,
)

BACKENDS = ["lmstudio", "openai"]


# ── 평가 지표 계산 ──────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    tp = sum(1 for r in results if r["escalated"] and r["expected_escalate"])
    fn = sum(1 for r in results if not r["escalated"] and r["expected_escalate"])
    fp = sum(1 for r in results if r["escalated"] and not r["expected_escalate"])
    tn = sum(1 for r in results if not r["escalated"] and not r["expected_escalate"])

    safety_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    over_escalation_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    escalation_rate = (tp + fp) / total
    avg_latency = sum(r["latency_ms"] for r in results) / total

    return {
        "safety_recall": round(safety_recall, 4),
        "over_escalation_rate": round(over_escalation_rate, 4),
        "escalation_rate": round(escalation_rate, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "n": total,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "target_safety_recall": 0.95,
        "target_over_escalation_max": 0.15,
        "safety_recall_ok": safety_recall >= 0.95,
        "over_escalation_ok": over_escalation_rate <= 0.15,
    }


# ── 메인 실험 루프 ─────────────────────────────────────────────────────────────

def run_experiment() -> dict:
    all_results = {}
    timestamp = datetime.now().isoformat()

    for backend in BACKENDS:
        print(f"\n{'='*65}")
        print(f"  Backend: {backend.upper()}")
        print(f"{'='*65}")

        backend_results = {}

        # Step 1: UQM 보정 (hold-out coverage 검증 포함)
        print(f"\n[1/4] UQM 보정 중 ({len(CALIBRATION_QUESTIONS)}개 질문, hold-out 20%)...")
        try:
            uqm = UQM(backend=backend, alpha=0.05)
            coverage_report = uqm.calibrate(CALIBRATION_QUESTIONS)
            base_threshold = uqm.calibrator.threshold
        except Exception as e:
            print(f"  [SKIP] UQM 보정 실패: {e}")
            continue

        # Step 2: RTC 설정
        print(f"\n[2/4] RTC 설정 (base_threshold={base_threshold:.4f})...")
        rtc = RTC(base_threshold=base_threshold)

        # Step 3: 각 시나리오 실험
        for scenario_name, scenario_cfg in SCENARIOS.items():
            print(f"\n[3/4] 시나리오: {scenario_name}")
            ede = EDE()
            case_results = []

            rtc_config = rtc.get_threshold(
                scenario_cfg["specialty"],
                scenario_cfg["scenario_type"],
            )
            print(f"  Threshold: {rtc_config.adjusted_threshold:.3f} "
                  f"(risk={rtc_config.risk_level.value})")

            for case in scenario_cfg["cases"]:
                try:
                    unc = uqm.evaluate(case["question"])
                    decision = ede.decide(unc, rtc_config, unc.raw_response.text)

                    case_results.append({
                        "id": case["id"],
                        "question": case["question"][:80] + "...",
                        "escalated": decision.should_escalate,
                        "expected_escalate": case["expected_escalate"],
                        "score": round(unc.nonconformity_score, 4),
                        "threshold": round(rtc_config.adjusted_threshold, 4),
                        "triggers": [t.value for t in decision.triggers],
                        "confidence": round(decision.confidence, 4),
                        "latency_ms": round(unc.raw_response.latency_ms, 1),
                        "scoring_method": unc.scoring_method,
                        "prediction_set_size": unc.prediction_set_size,
                        "answer_preview": unc.raw_response.text[:120],
                    })

                    status = "ESCALATE" if decision.should_escalate else "AUTO"
                    correct = "✓" if decision.should_escalate == case["expected_escalate"] else "✗"
                    print(f"  {case['id']} {correct} {status} "
                          f"(score={unc.nonconformity_score:.3f}, "
                          f"conf={decision.confidence:.2f})")

                except Exception as e:
                    print(f"  {case['id']} [ERROR] {e}")

            metrics = compute_metrics(case_results)
            backend_results[scenario_name] = {
                "cases": case_results,
                "metrics": metrics,
                "ede_summary": ede.summary(),
                "threshold": rtc_config.adjusted_threshold,
            }

            print(f"\n  {scenario_name} 결과:")
            print(f"     Safety Recall  : {metrics.get('safety_recall', 'N/A'):.3f} "
                  f"(목표 >=0.95) {'✓' if metrics.get('safety_recall_ok') else '✗'}")
            print(f"     Over-Escalation: {metrics.get('over_escalation_rate', 'N/A'):.3f} "
                  f"(목표 <=0.15) {'✓' if metrics.get('over_escalation_ok') else '✗'}")
            print(f"     Avg Latency    : {metrics.get('avg_latency_ms', 'N/A'):.0f} ms")

        # Step 4: 저장
        print(f"\n[4/4] 결과 저장 중...")
        all_results[backend] = {
            "backend": backend,
            "timestamp": timestamp,
            "base_threshold": base_threshold,
            "coverage_report": coverage_report,
            "scoring_method": uqm._use_self_consistency and "self_consistency" or "logprob",
            "scenarios": backend_results,
        }

    return all_results


def save_results(results: dict) -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # JSON 전체 저장
    json_path = out_dir / "experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON 저장: {json_path}")

    # CSV 비교표 저장
    csv_path = out_dir / "comparison_table.csv"
    rows = []
    for backend, bdata in results.items():
        for scenario, sdata in bdata.get("scenarios", {}).items():
            m = sdata.get("metrics", {})
            cov = bdata.get("coverage_report", {})
            rows.append({
                "backend": backend,
                "scenario": scenario,
                "safety_recall": m.get("safety_recall", ""),
                "over_escalation_rate": m.get("over_escalation_rate", ""),
                "escalation_rate": m.get("escalation_rate", ""),
                "avg_latency_ms": m.get("avg_latency_ms", ""),
                "threshold": sdata.get("threshold", ""),
                "conformal_coverage": cov.get("actual_coverage", ""),
                "coverage_valid": cov.get("coverage_valid", ""),
                "scoring_method": bdata.get("scoring_method", ""),
                "n": m.get("n", ""),
            })

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ CSV 저장: {csv_path}")

        # 터미널 요약
        print("\n" + "="*72)
        print("  최종 비교 요약")
        print("="*72)
        print(f"{'Backend':<12} {'Scenario':<18} {'Safety R.':<12} {'Over-Esc.':<12} "
              f"{'Coverage':<10} {'Latency(ms)'}")
        print("-"*72)
        for r in rows:
            print(f"{r['backend']:<12} {r['scenario']:<18} "
                  f"{str(r['safety_recall']):<12} {str(r['over_escalation_rate']):<12} "
                  f"{str(r['conformal_coverage']):<10} {r['avg_latency_ms']}")


if __name__ == "__main__":
    results = run_experiment()
    save_results(results)