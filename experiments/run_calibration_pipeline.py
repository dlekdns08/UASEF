"""
UASEF — 캘리브레이션 파이프라인

하드코딩된 RTC 배율·엔트로피 임계값·EDE 계수를 데이터에서 역산하고
결과를 base_config.yaml에 저장합니다.

실행 순서:
    1. CP calibration  → base threshold q̂ 산출
    2. entropy_calibration  → Youden's J로 H 임계값 결정
    3. rtc_calibration      → 위험도별 배율 Pareto sweep
    4. ede_coefficient_search → (t1_weight, entropy_boost) grid search
    5. base_config.yaml 업데이트 → 재현 가능한 실험 환경 보장

실행 방법:
    python experiments/run_calibration_pipeline.py --n-cal 500 --n-labeled 50

출력:
    experiments/configs/base_config.yaml (rtc / entropy_threshold / ede 섹션 갱신)
    results/calibration_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from models.uqm import UQM
from models.rtc_ede import RTC, EDE, SPECIALTY_RISK_MAP, RiskLevel
from models.rtc_calibration import sweep_all_risk_levels
from models.entropy_calibration import find_entropy_threshold
from models.ede_coefficient_search import grid_search_ede_coefficients
from data.loader import load_calibration_questions, load_scenarios, case_to_experiment_dict


# 시나리오 → 위험도 레이블 매핑
_SCENARIO_TO_RISK: dict[str, str] = {
    "emergency":      "CRITICAL",
    "rare_disease":   "HIGH",
    "multimorbidity": "MODERATE",
    "routine":        "LOW",
}

_SPECIALTY_MAP: dict[str, str] = {
    "emergency":      "emergency_medicine",
    "rare_disease":   "neurology",
    "multimorbidity": "internal_medicine",
    "routine":        "general_practice",
}


def _load_base_config() -> dict:
    path = ROOT / "experiments" / "configs" / "base_config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_base_config(cfg: dict) -> None:
    path = ROOT / "experiments" / "configs" / "base_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [Config] base_config.yaml 갱신됨: {path}")


def run_calibration_pipeline(
    backend: str = "openai",
    n_calibration: int = 30,
    n_labeled: int = 10,
    seed: int = 42,
) -> dict:
    """
    전체 캘리브레이션 파이프라인을 실행합니다.

    Args:
        backend:       UQM에 사용할 백엔드 ("openai" | "lmstudio")
        n_calibration: CP calibration용 비레이블 질문 수 (권장: ≥500)
        n_labeled:     RTC/EDE calibration용 레이블 케이스 수 (권장: ≥50/시나리오)
        seed:          재현성을 위한 랜덤 시드

    Returns:
        캘리브레이션 결과 요약 dict (calibration_report.json에도 저장)
    """
    timestamp = datetime.now().isoformat()
    report: dict = {"timestamp": timestamp, "backend": backend}

    # ── Step 1: CP Calibration → base threshold q̂ ────────────────────────────
    print("\n" + "="*65)
    print("  Step 1/4: CP Calibration → base threshold q̂")
    print("="*65)

    cal_questions = load_calibration_questions(n=n_calibration, split="train", seed=seed)
    uqm = UQM(backend=backend, alpha=0.05, scoring_method="logprob")

    try:
        coverage_report = uqm.calibrate(
            cal_questions,
            holdout_fraction=0.2,
            distribution_source="medqa",
        )
        base_threshold = uqm.calibrator.threshold
        print(f"  base threshold q̂ = {base_threshold:.4f}")
        print(f"  holdout coverage  = {coverage_report.get('actual_coverage', 'N/A'):.3f}")
        report["base_threshold"] = base_threshold
        report["coverage_report"] = coverage_report
    except Exception as e:
        print(f"  [ERROR] CP calibration 실패: {e}")
        raise

    # ── Step 2: 레이블 데이터 수집 (UQM 평가 포함) ───────────────────────────
    print("\n" + "="*65)
    print("  Step 2/4: 레이블 데이터 수집 및 UQM 평가")
    print("="*65)

    scenario_map = load_scenarios(n_per_scenario=n_labeled, split="test", seed=seed)

    # 위험도 레벨별 (scores, labels, entropy_values) 수집
    scores_by_level: dict[str, list[float]] = {k: [] for k in _SCENARIO_TO_RISK.values()}
    labels_by_level: dict[str, list[bool]] = {k: [] for k in _SCENARIO_TO_RISK.values()}
    all_entropy: list[float] = []
    all_entropy_labels: list[bool] = []

    # EDE grid search용 데이터
    all_t1_flags: list[bool] = []
    all_trigger_counts: list[int] = []
    all_entropy_flags: list[bool] = []
    all_labels: list[bool] = []

    rtc = RTC(base_threshold=base_threshold)
    ede_default = EDE()

    for stype, cases in scenario_map.items():
        risk_level = _SCENARIO_TO_RISK.get(stype, "MODERATE")
        specialty = _SPECIALTY_MAP.get(stype, "internal_medicine")
        rtc_config = rtc.get_threshold(specialty, stype)

        print(f"\n  [{stype}] {len(cases)}건 평가 중 (risk={risk_level})...")

        for case in cases:
            case_dict = case_to_experiment_dict(case)
            try:
                unc = uqm.evaluate(case_dict["question"], distribution_source="medqa")
                decision = ede_default.decide(unc, rtc_config, unc.raw_response.text)

                score = unc.nonconformity_score
                label = case_dict["expected_escalate"]
                entropy = unc.confidence_entropy

                scores_by_level[risk_level].append(score)
                labels_by_level[risk_level].append(label)
                all_entropy.append(entropy)
                all_entropy_labels.append(label)

                # EDE grid search 입력 준비
                from models.rtc_ede import EscalationTrigger
                t1_flag = EscalationTrigger.UNCERTAINTY_EXCEEDED in decision.triggers
                trigger_count = len(decision.triggers)
                import math
                entropy_flag = (
                    not math.isnan(entropy) and entropy > ede_default.entropy_threshold
                )

                all_t1_flags.append(t1_flag)
                all_trigger_counts.append(trigger_count)
                all_entropy_flags.append(entropy_flag)
                all_labels.append(label)

            except Exception as e:
                print(f"    [SKIP] {case_dict['id']}: {e}")

    # ── Step 3: Entropy Threshold (Youden's J) ───────────────────────────────
    print("\n" + "="*65)
    print("  Step 3/4: Entropy Threshold → Youden's J")
    print("="*65)

    entropy_result = find_entropy_threshold(all_entropy, all_entropy_labels)
    entropy_threshold = entropy_result["threshold"]

    if entropy_result["fallback_used"]:
        print(f"  [FALLBACK] 유효 샘플 부족 → entropy_threshold = {entropy_threshold}")
    else:
        print(f"  entropy_threshold = {entropy_threshold:.4f} nats/token")
        print(f"  Youden's J        = {entropy_result['youdens_j']:.4f}")
        print(f"  Sensitivity       = {entropy_result['sensitivity']:.4f}")
        print(f"  Specificity       = {entropy_result['specificity']:.4f}")
        print(f"  유효 샘플 수       = {entropy_result['n_valid']}")

    report["entropy_calibration"] = entropy_result

    # ── Step 4: RTC 배율 Pareto Sweep ────────────────────────────────────────
    print("\n" + "="*65)
    print("  Step 4a/4: RTC 배율 Pareto Sweep")
    print("="*65)

    rtc_multipliers = sweep_all_risk_levels(
        scores_by_level=scores_by_level,
        labels_by_level=labels_by_level,
        base_threshold=base_threshold,
    )
    report["rtc_multipliers"] = rtc_multipliers

    # ── Step 4b: EDE Coefficient Grid Search ─────────────────────────────────
    print("\n" + "="*65)
    print("  Step 4b/4: EDE Coefficient Grid Search")
    print("="*65)

    if len(all_labels) >= 5:
        ede_result = grid_search_ede_coefficients(
            t1_flags=all_t1_flags,
            trigger_counts=all_trigger_counts,
            entropy_flags=all_entropy_flags,
            labels=all_labels,
        )
        t1_weight = ede_result["best_t1_weight"]
        entropy_boost = ede_result["best_entropy_boost"]
        print(f"  t1_weight     = {t1_weight}")
        print(f"  entropy_boost = {entropy_boost}")
        print(f"  F1-safety     = {ede_result['best_f1_safety']:.4f}")
        print(f"  Safety Recall = {ede_result['best_safety_recall']:.4f}")
        print(f"  Over-Esc Rate = {ede_result['best_over_escalation']:.4f}")
        report["ede_grid_search"] = {
            "best_t1_weight": t1_weight,
            "best_entropy_boost": entropy_boost,
            "best_f1_safety": ede_result["best_f1_safety"],
            "best_safety_recall": ede_result["best_safety_recall"],
            "best_over_escalation": ede_result["best_over_escalation"],
        }
    else:
        t1_weight = 0.4
        entropy_boost = 0.15
        print(f"  [FALLBACK] 데이터 부족 → t1_weight={t1_weight}, entropy_boost={entropy_boost}")
        report["ede_grid_search"] = {
            "best_t1_weight": t1_weight,
            "best_entropy_boost": entropy_boost,
            "fallback_used": True,
        }

    # ── Step 5: base_config.yaml 갱신 ────────────────────────────────────────
    print("\n" + "="*65)
    print("  Step 5/4: base_config.yaml 갱신")
    print("="*65)

    base_cfg = _load_base_config()
    base_cfg["rtc"] = rtc_multipliers
    base_cfg["entropy_threshold"] = float(entropy_threshold)
    base_cfg["ede"] = {
        "t1_weight": float(t1_weight),
        "entropy_boost": float(entropy_boost),
    }
    _save_base_config(base_cfg)

    # ── 캘리브레이션 리포트 저장 ──────────────────────────────────────────────
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [Report] 저장됨: {report_path}")

    print("\n" + "="*65)
    print("  캘리브레이션 완료")
    print(f"    entropy_threshold : {entropy_threshold}")
    print(f"    rtc multipliers   : {rtc_multipliers}")
    print(f"    ede t1_weight     : {t1_weight}")
    print(f"    ede entropy_boost : {entropy_boost}")
    print("="*65)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UASEF 캘리브레이션 파이프라인")
    parser.add_argument(
        "--backend", type=str, default="openai",
        choices=["openai", "lmstudio"],
        help="사용할 LLM 백엔드",
    )
    parser.add_argument(
        "--n-cal", type=int, default=30,
        help="CP calibration용 비레이블 질문 수 (권장: ≥500)",
    )
    parser.add_argument(
        "--n-labeled", type=int, default=10,
        help="RTC/EDE calibration용 레이블 케이스 수/시나리오 (권장: ≥50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="재현성을 위한 랜덤 시드",
    )
    args = parser.parse_args()

    print(f"\n[Pipeline] backend={args.backend}, n_cal={args.n_cal}, "
          f"n_labeled={args.n_labeled}, seed={args.seed}")

    run_calibration_pipeline(
        backend=args.backend,
        n_calibration=args.n_cal,
        n_labeled=args.n_labeled,
        seed=args.seed,
    )
