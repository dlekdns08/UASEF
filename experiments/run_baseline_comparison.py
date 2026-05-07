"""
UASEF 베이스라인 비교 실험.

세 가지 에스컬레이션 전략을 동일한 케이스에서 비교하여 각 구성 요소의 기여를 정량화합니다.

비교 대상:
  1. no_escalation:   항상 자율 행동 (에스컬레이션 없음)
  2. threshold_only:  CP Trigger 1만 사용 (T2 키워드, T3 근거 부재, 엔트로피 boost 없음)
  3. full_uasef:      T1 + T2 + T3 + 엔트로피 가중치 전체 사용

실험 구조:
  [Primary]  OpenAI (GPT-4o-mini) — logprob-based CP
  [Ablation] 로컬 (LMStudio GGUF) — logprob-based CP

⚠ 계획서의 Temperature Scaling / MC Dropout은 현재 미구현.
  향후 추가 시 아래 BaselineScorer 인터페이스를 준수하면 됩니다:

    class BaselineScorer(Protocol):
        def score(self, question: str) -> float: ...
        def threshold(self) -> float: ...

실행:
    # Primary (OpenAI logprob)
    python experiments/run_baseline_comparison.py --backend openai --n-cal 500 --n-test 50

    # Ablation (로컬 logprob)
    python experiments/run_baseline_comparison.py --backend lmstudio --n-cal 500 --n-test 50

    # Primary + Ablation 모두
    python experiments/run_baseline_comparison.py --n-cal 500 --n-test 50

출력:
    results/baseline_comparison.json
    results/baseline_comparison.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from models.uqm import UQM
from models.rtc_ede import RTC, EDE
from data.loader import load_calibration_questions, load_scenarios, case_to_experiment_dict
from experiments.config_utils import load_calibration_config, load_config
from experiments.metrics_utils import compute_binary_metrics, fmt_rate, fmt_ci


# ── 베이스라인 에스컬레이션 함수 ──────────────────────────────────────────────────

def run_threshold_only(
    uqm: UQM,
    rtc_config,
    question: str,
    distribution_source: str,
) -> tuple[bool, float]:
    """
    Threshold-only 베이스라인: CP Trigger 1만 사용.
    EDE의 T2(키워드), T3(근거 부재), 엔트로피 boost를 제외합니다.
    순수 Conformal Prediction의 효과를 분리 측정하는 데 활용합니다.
    """
    unc = uqm.evaluate(question, distribution_source=distribution_source)
    escalated = unc.nonconformity_score > rtc_config.adjusted_threshold
    return escalated, unc.nonconformity_score


def run_full_uasef(
    uqm: UQM,
    rtc: RTC,
    ede: EDE,
    question: str,
    specialty: str,
    scenario_type: str,
    distribution_source: str,
) -> tuple[bool, float]:
    """
    Full UASEF: T1 + T2 + T3 + 엔트로피 가중치 전체 사용.
    """
    unc = uqm.evaluate(question, distribution_source=distribution_source)
    rtc_config = rtc.get_threshold(specialty, scenario_type)
    decision = ede.decide(unc, rtc_config, unc.raw_response.text)
    return decision.should_escalate, unc.nonconformity_score


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """audit issue #16/#11: silent zero 제거 + Wilson CI."""
    if not results:
        return {"error": "케이스 없음"}
    return compute_binary_metrics(results, pred_key="escalated", label_key="expected")


# ── 메인 비교 루프 ────────────────────────────────────────────────────────────

_SPECIALTY_MAP = {
    "emergency":      "emergency_medicine",
    "rare_disease":   "neurology",
    "multimorbidity": "internal_medicine",
    "routine":        "general_practice",
}


def _scoring_method_for(backend: str) -> str:
    """Primary: openai → logprob / Ablation: lmstudio GGUF → logprob"""
    return "logprob"


def run_baseline_comparison(
    backend: str,
    n_cal: int = 500,
    n_test: int = 200,
    scoring_method: str = "auto",
    alpha: Optional[float] = None,
    seed: int = 42,
    prompt_mode: str = "neutral",      # audit #5
    strict: bool = False,              # audit #19
    decision_rule: Optional[str] = None,  # audit #2 — None이면 base_config 값
) -> dict:
    effective_method = (
        _scoring_method_for(backend) if scoring_method == "auto" else scoring_method
    )
    role = "[Primary]" if effective_method == "logprob" else "[Ablation]"

    cfg = load_config()
    effective_alpha = alpha if alpha is not None else cfg.get("uqm", {}).get("alpha", 0.10)

    print(f"\n{'='*65}")
    print(f"  베이스라인 비교 — Backend: {backend.upper()}  {role}")
    print(f"  scoring={effective_method}, n_cal={n_cal}, n_test={n_test}, α={effective_alpha}")
    print(f"  prompt_mode={prompt_mode}, strict={strict}, decision_rule={decision_rule or '(config)'}")
    print(f"{'='*65}")

    # UQM 보정
    print(f"\n[1/3] UQM 보정 중 (MedQA, n={n_cal})...")
    from experiments.config_utils import load_hybrid_weights   # audit 6.10
    _hyb_dw, _hyb_ew = load_hybrid_weights()
    uqm = UQM(
        backend=backend, alpha=effective_alpha, scoring_method=effective_method,
        prompt_mode=prompt_mode, strict=strict,
        hybrid_diversity_weight=_hyb_dw, hybrid_entropy_weight=_hyb_ew,
    )
    try:
        cal_questions = load_calibration_questions(n=n_cal, split="train", seed=seed)
        uqm.calibrate(cal_questions, distribution_source="medqa")
    except Exception as e:
        print(f"  [SKIP] UQM 보정 실패: {e}")
        return {}

    rtc_multipliers, ede_kwargs = load_calibration_config()
    from experiments.config_utils import load_scenario_multipliers
    scenario_multipliers = load_scenario_multipliers()
    # CLI override 우선 (audit #2)
    if decision_rule is not None:
        ede_kwargs["decision_rule"] = decision_rule
    rtc = RTC(
        base_threshold=uqm.calibrator.threshold,
        multipliers=rtc_multipliers,
        scenario_multipliers=scenario_multipliers,
    )
    ede = EDE(**ede_kwargs)

    # 테스트 케이스 로드
    print(f"\n[2/3] 테스트 케이스 로드 중 (n_per_scenario={n_test})...")
    scenario_map = load_scenarios(n_per_scenario=n_test, split="test", seed=seed,
                                  include_pubmedqa=False)

    # 세 가지 전략에 대한 결과 수집
    strategy_results: dict[str, list[dict]] = {
        "no_escalation": [],
        "threshold_only": [],
        "full_uasef":     [],
    }

    print(f"\n[3/3] 케이스별 평가 중...")
    total_cases = sum(len(c) for c in scenario_map.values())
    processed = 0

    for stype, cases in scenario_map.items():
        specialty = _SPECIALTY_MAP.get(stype, "internal_medicine")
        rtc_config = rtc.get_threshold(specialty, stype)

        for case in cases:
            q = case.question
            exp = case.expected_escalate
            processed += 1

            # 케이스 출처에 따라 distribution_source 결정
            # MedAbstain 케이스는 MedQA와 다른 분포 → WeightedCP 보정 활성화
            dist_source = (
                "medabstain" if getattr(case, "source", "").startswith("medabstain")
                else "medqa"
            )

            # Baseline 1: 절대 에스컬레이션 안 함
            strategy_results["no_escalation"].append({
                "escalated": False,
                "expected": exp,
                "scenario_type": stype,
            })

            # Baseline 2: CP Trigger 1만
            try:
                esc_t1, score = run_threshold_only(uqm, rtc_config, q, dist_source)
            except Exception as e:
                print(f"  [{processed}/{total_cases}] threshold_only 오류: {e}")
                esc_t1, score = False, 0.0
            strategy_results["threshold_only"].append({
                "escalated": esc_t1,
                "expected": exp,
                "score": score,
                "scenario_type": stype,
            })

            # Baseline 3: Full UASEF
            try:
                esc_full, score_full = run_full_uasef(
                    uqm, rtc, ede, q, specialty, stype, dist_source
                )
            except Exception as e:
                print(f"  [{processed}/{total_cases}] full_uasef 오류: {e}")
                esc_full, score_full = False, 0.0
            strategy_results["full_uasef"].append({
                "escalated": esc_full,
                "expected": exp,
                "score": score_full,
                "scenario_type": stype,
            })

            if processed % 10 == 0:
                print(f"  [{processed}/{total_cases}] 완료")

    # 전략별 메트릭 집계
    metrics = {
        name: compute_metrics(results)
        for name, results in strategy_results.items()
    }

    return {
        "backend": backend,
        "timestamp": datetime.now().isoformat(),
        "scoring_method": effective_method,
        "n_calibration": n_cal,
        "n_test_per_scenario": n_test,
        "metrics": metrics,
        "raw": strategy_results,
    }


# ── 결과 출력 및 저장 ─────────────────────────────────────────────────────────

def _print_comparison_table(result: dict) -> None:
    print("\n" + "="*65)
    print("  베이스라인 비교 결과")
    print("="*65)
    print(f"  Backend: {result['backend']} | Method: {result['scoring_method']}")
    print("-"*65)
    print(f"  {'전략':<22} {'Safety Recall':>14} {'Over-Esc Rate':>14} {'OK(≥0.95)':>10}")
    print("  " + "-"*60)
    for name, m in result.get("metrics", {}).items():
        if "error" in m:
            print(f"  {name:<22} — 오류")
            continue
        ok = "✓" if m.get("safety_recall_ok") else "✗"
        print(
            f"  {name:<22} "
            f"{fmt_rate(m.get('safety_recall')):>14} "
            f"{fmt_rate(m.get('over_escalation_rate')):>14} "
            f"{ok:>10}"
        )
    print()


def save_results(all_results: dict) -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "baseline_comparison.json"
    # raw 데이터는 파일 크기가 클 수 있으므로 메트릭만 저장
    save_data = {
        backend: {k: v for k, v in data.items() if k != "raw"}
        for backend, data in all_results.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON 저장: {json_path}")

    csv_path = out_dir / "baseline_comparison.csv"
    rows = []
    for backend, data in all_results.items():
        for strategy, m in data.get("metrics", {}).items():
            if "error" in m:
                continue
            rows.append({
                "backend": backend,
                "scoring_method": data.get("scoring_method"),
                "strategy": strategy,
                "safety_recall": m.get("safety_recall"),
                "over_escalation_rate": m.get("over_escalation_rate"),
                "safety_recall_ok": m.get("safety_recall_ok"),
                "tp": m.get("tp"), "fn": m.get("fn"),
                "fp": m.get("fp"), "tn": m.get("tn"),
            })
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ CSV 저장: {csv_path}")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UASEF 베이스라인 비교 실험")
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["openai", "lmstudio", "mlx", "anthropic", "gemini"],
        help="단일 백엔드만 실행 (기본: openai[Primary] + lmstudio[Ablation] 모두)",
    )
    parser.add_argument("--n-cal", type=int, default=500,
                        help="calibration 질문 수 (권장: 500)")
    parser.add_argument("--n-test", type=int, default=50,
                        help="시나리오별 테스트 케이스 수 (권장: 50)")
    parser.add_argument(
        "--scoring-method", type=str, default="auto",
        choices=["logprob", "self_consistency", "hybrid", "auto"],
        help="비적합 점수 방식 강제 지정. 기본: auto (openai=logprob, lmstudio=logprob)",
    )
    parser.add_argument("--alpha", type=float, default=None,
                        help="Conformal prediction α (기본: base_config.yaml uqm.alpha = 0.10)")
    parser.add_argument("--seed", type=int, default=42)
    # audit 6라운드 신규 옵션
    parser.add_argument(
        "--prompt-mode", type=str, default="neutral",
        choices=["neutral", "instructed"],
        help="UQM SYSTEM_PROMPT (audit #5)",
    )
    parser.add_argument(
        "--decision-rule", type=str, default=None,
        choices=["trigger_count", "confidence"],
        help="EDE 결정 규칙 (audit #2). 미지정 시 base_config.yaml 사용.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="CP 최소 n 미달 시 RuntimeError로 중단 (audit #19)",
    )
    args = parser.parse_args()

    backends = [args.backend] if args.backend else ["openai", "lmstudio"]

    all_results = {}
    for backend in backends:
        try:
            result = run_baseline_comparison(
                backend=backend,
                n_cal=args.n_cal,
                n_test=args.n_test,
                scoring_method=args.scoring_method,
                alpha=args.alpha,
                seed=args.seed,
                prompt_mode=args.prompt_mode,
                strict=args.strict,
                decision_rule=args.decision_rule,
            )
            if result:
                all_results[backend] = result
                _print_comparison_table(result)
        except Exception as e:
            print(f"\n[SKIP] {backend}: {e}")

    if all_results:
        save_results(all_results)
