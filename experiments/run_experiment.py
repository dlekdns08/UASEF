"""
UASEF — 순차 파이프라인 실험 실행기

실험 구조:
  [Primary]  OpenAI (GPT-4o-mini) — logprob-based CP
             token-level logprobs로 비적합 점수 계산. 논문 주요 결과.
  [Ablation] 로컬 (LMStudio)      — self_consistency-based CP
             N회 쿼리의 Jaccard 다양성으로 비적합 점수 계산.
             "블랙박스 LLM에도 UASEF 적용 가능함"을 검증하는 ablation study.

  두 방식 모두 CP coverage 보장(P(s≤q̂)≥1-α)은 수학적으로 성립하지만,
  비적합 함수가 달라 결과를 직접 수치 비교하지 않습니다.

실행 방법:
    # Primary + Ablation 전체 실험 (권장)
    python experiments/run_experiment.py --n-cal 500 --n-test 50

    # Primary만 (OpenAI logprob)
    python experiments/run_experiment.py --backend openai --n-cal 500 --n-test 50

    # Ablation만 (로컬 self_consistency)
    python experiments/run_experiment.py --backend lmstudio --n-cal 500 --n-test 50

    # 시나리오별 config 적용
    python experiments/run_experiment.py --config experiments/configs/scenario_emergency.yaml

출력:
    results/experiment_results.json
    results/comparison_table.csv
"""

import argparse
import json
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

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
#
# ── 시나리오별 데이터 소스 ─────────────────────────────────────────────────────
# emergency:      MedQA 응급 키워드 필터 + MedAbstain AP/NAP
# rare_disease:   MedQA 희귀질환 키워드 필터 + MedAbstain AP/NAP
#                 ⚠ 계획서의 'PubMedQA 기반'은 현재 미구현 상태.
#                    PubMedQA를 추가하려면 data/loader.py의 load_pubmedqa() 구현 후
#                    load_scenarios()의 rare_disease 버킷에 병합 필요.
# multimorbidity: MedQA 3개+ 만성질환 키워드 동시 언급 필터 + MedAbstain AP/NAP
# routine:        MedQA 기타 (expected_escalate=False)

_SPECIALTY_MAP = {
    "emergency":      "emergency_medicine",
    "rare_disease":   "neurology",
    "multimorbidity": "internal_medicine",
    "routine":        "general_practice",
}


# ── Config 로딩 ───────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """override가 base의 nested 값을 재귀적으로 덮어씁니다."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    base_config.yaml을 로드하고 추가 config로 오버라이드합니다.

    Args:
        config_path: 시나리오별 config 파일 (예: configs/scenario_emergency.yaml).
                     None이면 base_config.yaml만 사용.
    """
    base_path = ROOT / "experiments" / "configs" / "base_config.yaml"
    with open(base_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)
        print(f"[Config] {config_path.name} 적용됨")

    return config


def _build_datasets(cfg: dict) -> tuple[list[str], dict]:
    """Config에 따라 MedQA / MedAbstain 데이터를 로드합니다."""
    d = cfg["data"]
    uqm_cfg = cfg["uqm"]
    dist = d.get("distribution_source", "medqa")

    print("[Dataset] MedQA / MedAbstain 로드 중...")
    cal_questions = load_calibration_questions(
        n=d["n_calibration"], split=d["calibration_split"], seed=d["seed"]
    )

    include_pubmedqa = d.get("include_pubmedqa", False)

    scenario_cfg = cfg.get("scenario")
    if scenario_cfg:
        # 단일 시나리오 config
        stype = scenario_cfg["scenario_type"]
        scenario_map = load_scenarios(
            n_per_scenario=d["n_test_per_scenario"],
            split=d["test_split"],
            seed=d["seed"],
            include_pubmedqa=include_pubmedqa,
        )
        scenarios = {
            stype: {
                "specialty": scenario_cfg["specialty"],
                "scenario_type": stype,
                "distribution_source": dist,
                "cases": [case_to_experiment_dict(c) for c in scenario_map.get(stype, [])],
            }
        }
    else:
        # 전체 시나리오
        scenario_map = load_scenarios(
            n_per_scenario=d["n_test_per_scenario"],
            split=d["test_split"],
            seed=d["seed"],
            include_pubmedqa=include_pubmedqa,
        )
        scenarios = {
            st: {
                "specialty": _SPECIALTY_MAP.get(st, "internal_medicine"),
                "scenario_type": st,
                "distribution_source": dist,
                "cases": [case_to_experiment_dict(c) for c in cases],
            }
            for st, cases in scenario_map.items()
        }

    return cal_questions, scenarios


BACKENDS = ["openai", "lmstudio"]  # Primary(openai) → Ablation(lmstudio) 순


def _scoring_method_for(backend: str) -> str:
    """백엔드별 기본 scoring method 선택.

    Primary:  openai   → logprob          (token-level logprobs 활용)
    Ablation: 로컬     → self_consistency  (logprobs 불필요)
    """
    return "logprob" if backend == "openai" else "self_consistency"


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

def run_experiment(cfg: dict) -> dict:
    all_results = {}
    timestamp = datetime.now().isoformat()

    calibration_questions, scenarios = _build_datasets(cfg)
    backends = cfg.get("backends", BACKENDS)
    uqm_cfg = cfg["uqm"]
    dist_source = cfg["data"].get("distribution_source", "medqa")

    for backend in backends:
        print(f"\n{'='*65}")
        print(f"  Backend: {backend.upper()}")
        print(f"{'='*65}")

        backend_results = {}

        # Step 1: UQM 보정
        print(f"\n[1/4] UQM 보정 중 ({len(calibration_questions)}개 질문, hold-out 20%)...")
        try:
            uqm = UQM(
                backend=backend,
                alpha=uqm_cfg["alpha"],
                scoring_method=uqm_cfg.get("scoring_method", "logprob"),
                consistency_n=uqm_cfg.get("consistency_n", 5),
            )
            coverage_report = uqm.calibrate(
                calibration_questions,
                holdout_fraction=uqm_cfg.get("holdout_fraction", 0.2),
                distribution_source=dist_source,
            )
            base_threshold = uqm.calibrator.threshold
        except Exception as e:
            print(f"  [SKIP] UQM 보정 실패: {e}")
            continue

        # Step 2: RTC 설정
        print(f"\n[2/4] RTC 설정 (base_threshold={base_threshold:.4f})...")
        rtc = RTC(base_threshold=base_threshold)

        # Step 3: 각 시나리오 실험
        for scenario_name, scenario_cfg in scenarios.items():
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
                    unc = uqm.evaluate(case["question"], distribution_source=dist_source)
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
            "scoring_method": uqm.active_scoring_method,
            "distribution_source": dist_source,
            "uqm_config": uqm_cfg,
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
    parser = argparse.ArgumentParser(description="UASEF 실험 실행기")
    parser.add_argument(
        "--config", type=str, default=None,
        help="시나리오별 YAML config 경로 (예: experiments/configs/scenario_emergency.yaml)",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=["emergency", "rare_disease", "multimorbidity"],
        help="실행할 단일 시나리오 (config 파일 대신 사용 가능)",
    )
    parser.add_argument(
        "--n-cal", type=int, default=None,
        help="Calibration 질문 수 (논문 권장: 500)",
    )
    parser.add_argument(
        "--n-test", type=int, default=None,
        help="시나리오별 테스트 케이스 수 (논문 권장: 50)",
    )
    parser.add_argument(
        "--scoring-method", type=str, default=None,
        choices=["logprob", "self_consistency", "auto"],
        help="비적합 점수 산출 방식 (logprob=primary, self_consistency=ablation)",
    )
    args = parser.parse_args()

    # Config 결정
    cfg_path = None
    if args.config:
        cfg_path = Path(args.config)
    elif args.scenario:
        cfg_path = ROOT / "experiments" / "configs" / f"scenario_{args.scenario}.yaml"

    cfg = load_config(cfg_path)

    # CLI 인자로 오버라이드
    if args.n_cal:
        cfg["data"]["n_calibration"] = args.n_cal
    if args.n_test:
        cfg["data"]["n_test_per_scenario"] = args.n_test
    if args.scoring_method:
        cfg["uqm"]["scoring_method"] = args.scoring_method

    print(f"\n[Config] scoring_method={cfg['uqm']['scoring_method']}, "
          f"n_cal={cfg['data']['n_calibration']}, "
          f"n_test={cfg['data']['n_test_per_scenario']}")

    results = run_experiment(cfg)
    save_results(results)