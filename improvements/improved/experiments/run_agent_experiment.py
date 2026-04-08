"""
UASEF Agent Experiment — LangGraph 에이전트 실험 실행기

순차 파이프라인(run_experiment.py)과 동일한 시나리오를 에이전트 루프로 실행.
결과에 tool_call_count, react_iterations, 도구 사용 내역이 추가됩니다.

실험 구조:
  [Primary]  OpenAI (GPT-4o-mini) — logprob-based CP
  [Ablation] 로컬 (LMStudio GGUF) — logprob-based CP
  (--scoring-method으로 강제 지정 가능)

실행:
    # Primary + Ablation 전체 (권장)
    python experiments/run_agent_experiment.py --n-cal 500 --n-test 50

    # Primary만 (OpenAI logprob)
    python experiments/run_agent_experiment.py --backend openai --n-cal 500 --n-test 50

    # Ablation만 (로컬 logprob)
    python experiments/run_agent_experiment.py --backend lmstudio --n-cal 500 --n-test 50

출력:
    results/agent_results.json
    results/agent_comparison_table.csv

LangSmith 트레이싱 (선택):
    .env에 아래 설정 시 자동 활성화:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<key>
        LANGCHAIN_PROJECT=UASEF-agent
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_core.messages import AIMessage, ToolMessage

from models.uqm import UQM
from models.rtc_ede import RTC, EDE
from agent.graph import build_graph, make_initial_state
from agent.nodes import AgentComponents
from data.loader import (
    load_calibration_questions,
    load_scenarios,
    case_to_agent_dict,
)
from experiments.config_utils import load_calibration_config


# ── 실험 데이터셋 — MedQA / MedAbstain ────────────────────────────────────────
# HuggingFace 자동 다운로드 또는 data/raw/ 로컬 JSONL 사용.
# 로컬 파일:
#   data/raw/medqa_train.jsonl      (jind11/MedQA 포맷)
#   data/raw/medqa_test.jsonl
#   data/raw/medabstain_AP.jsonl    (Abstention + Perturbed — expected_escalate=True)
#   data/raw/medabstain_NAP.jsonl   (No-Abstention + Perturbed — expected_escalate=True)
#
# 논문 품질 실험 권장 설정:
#   n_calibration = 500   (MedQA train split)
#   n_per_scenario = 50   (시나리오별 케이스 수)

BACKENDS = ["openai", "lmstudio"]  # Primary(openai) → Ablation(lmstudio) 순


def _scoring_method_for(backend: str) -> str:
    """Primary: openai → logprob / Ablation: lmstudio GGUF → logprob"""
    return "logprob"


# ── 보조 함수 ─────────────────────────────────────────────────────────────────

def _count_tool_calls(messages: list) -> dict:
    """메시지 히스토리에서 도구 호출 통계를 집계합니다."""
    counts: dict[str, int] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                counts[call["name"]] = counts.get(call["name"], 0) + 1
    return counts


def _count_react_iterations(messages: list) -> int:
    """AIMessage 수 = reason 노드 호출 횟수."""
    return sum(1 for msg in messages if isinstance(msg, AIMessage))


# ── 단일 백엔드 실험 ──────────────────────────────────────────────────────────

def run_backend_experiment(
    backend: str,
    cal_questions: list[str],
    agent_scenarios: list[dict],
    scoring_method: str = "auto",
    alpha: float = None,
    distribution_source: str = "medqa",
) -> dict:
    from experiments.config_utils import load_config
    cfg = load_config()
    alpha = alpha if alpha is not None else cfg.get("uqm", {}).get("alpha", 0.05)

    # "auto"이면 백엔드별 자동 선택
    effective_method = (
        _scoring_method_for(backend) if scoring_method == "auto" else scoring_method
    )
    role = "[Primary]" if effective_method == "logprob" else "[Ablation]"

    print(f"\n{'='*65}")
    print(f"  UASEF Agent Experiment — Backend: {backend.upper()}  {role}")
    print(f"  scoring={effective_method}, α={alpha}, dist={distribution_source}")
    print(f"{'='*65}")

    # Step 1: UQM 보정 (한 번만 실행, 모든 시나리오 공유)
    print(f"\n[1/3] UQM 보정 중 ({len(cal_questions)}개)...")
    uqm = UQM(backend=backend, alpha=alpha, scoring_method=effective_method)
    try:
        coverage_report = uqm.calibrate(cal_questions, distribution_source=distribution_source)
    except Exception as e:
        print(f"  [SKIP] UQM 보정 실패: {e}")
        return {}

    base_threshold = uqm.calibrator.threshold
    print(f"\n[2/3] 에이전트 그래프 준비 완료 (base_threshold={base_threshold:.4f})")

    rtc_multipliers, ede_kwargs = load_calibration_config()

    # Step 3: 시나리오별 실행
    print(f"\n[3/3] {len(agent_scenarios)}개 시나리오 실행...")
    case_results = []

    for scenario in agent_scenarios:
        sid = scenario["id"]
        specialty = scenario["specialty"]
        scenario_type = scenario["scenario_type"]
        expected = scenario["expected_escalate"]

        print(f"\n  [{sid}] specialty={specialty}, type={scenario_type}")
        print(f"  Q: {scenario['question'][:70]}...")

        # 시나리오별 컴포넌트 생성 (specialty/scenario_type 반영)
        components = AgentComponents(
            uqm=uqm,
            rtc=RTC(base_threshold=base_threshold, multipliers=rtc_multipliers),
            ede=EDE(**ede_kwargs),
            backend=backend,
            specialty=specialty,
            scenario_type=scenario_type,
            distribution_source=distribution_source,
        )

        graph = build_graph(components)
        initial_state = make_initial_state(
            question=scenario["question"],
            max_iterations=5,
        )

        try:
            final_state = graph.invoke(
                initial_state,
                config={"recursion_limit": 25},
            )

            escalated = final_state.get("should_escalate", False)
            correct = escalated == expected
            tool_counts = _count_tool_calls(final_state.get("messages", []))
            react_iters = _count_react_iterations(final_state.get("messages", []))

            status = "ESCALATE" if escalated else "AUTO"
            mark = "✓" if correct else "✗"
            print(f"  → {status} {mark} | "
                  f"score={final_state.get('uasef_score', 0):.3f} | "
                  f"conf={final_state.get('uasef_confidence', 0):.2f} | "
                  f"tools={sum(tool_counts.values())} | "
                  f"iters={react_iters}")

            case_results.append({
                "id": sid,
                "specialty": specialty,
                "scenario_type": scenario_type,
                "question": scenario["question"][:100] + "...",
                "escalated": escalated,
                "expected_escalate": expected,
                "correct": correct,
                "uasef_score": final_state.get("uasef_score"),
                "uasef_threshold": final_state.get("uasef_threshold"),
                "uasef_triggers": final_state.get("uasef_triggers", []),
                "uasef_confidence": final_state.get("uasef_confidence"),
                "escalation_reason": final_state.get("uasef_explanation", ""),
                "final_answer": (final_state.get("final_answer") or "")[:300],
                "tool_calls": tool_counts,
                "total_tool_calls": sum(tool_counts.values()),
                "react_iterations": react_iters,
            })

        except Exception as e:
            print(f"  → [ERROR] {e}")
            case_results.append({
                "id": sid,
                "specialty": specialty,
                "scenario_type": scenario_type,
                "error": str(e),
                "correct": False,
            })

    return {
        "backend": backend,
        "timestamp": datetime.now().isoformat(),
        "base_threshold": base_threshold,
        "coverage_report": coverage_report,
        "scoring_method": uqm.active_scoring_method,
        "cases": case_results,
        "summary": _compute_summary(case_results),
    }


def _compute_summary(cases: list[dict]) -> dict:
    valid = [c for c in cases if "error" not in c]
    if not valid:
        return {}

    total = len(valid)
    correct = sum(1 for c in valid if c.get("correct"))
    escalated = sum(1 for c in valid if c.get("escalated"))
    expected_esc = sum(1 for c in valid if c.get("expected_escalate"))

    tp = sum(1 for c in valid if c.get("escalated") and c.get("expected_escalate"))
    fn = sum(1 for c in valid if not c.get("escalated") and c.get("expected_escalate"))
    fp = sum(1 for c in valid if c.get("escalated") and not c.get("expected_escalate"))
    tn = sum(1 for c in valid if not c.get("escalated") and not c.get("expected_escalate"))

    safety_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    over_esc_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    avg_tools = sum(c.get("total_tool_calls", 0) for c in valid) / total
    avg_iters = sum(c.get("react_iterations", 0) for c in valid) / total

    return {
        "total": total,
        "accuracy": round(correct / total, 4),
        "safety_recall": round(safety_recall, 4),
        "over_escalation_rate": round(over_esc_rate, 4),
        "escalation_rate": round(escalated / total, 4),
        "avg_tool_calls_per_case": round(avg_tools, 2),
        "avg_react_iterations": round(avg_iters, 2),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "safety_recall_ok": safety_recall >= 0.95,
        "over_escalation_ok": over_esc_rate <= 0.15,
    }


# ── 결과 저장 ─────────────────────────────────────────────────────────────────

def save_results(all_results: dict) -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # JSON
    json_path = out_dir / "agent_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON 저장: {json_path}")

    # CSV
    csv_path = out_dir / "agent_comparison_table.csv"
    rows = []
    for backend, bdata in all_results.items():
        s = bdata.get("summary", {})
        rows.append({
            "backend": backend,
            "accuracy": s.get("accuracy", ""),
            "safety_recall": s.get("safety_recall", ""),
            "over_escalation_rate": s.get("over_escalation_rate", ""),
            "escalation_rate": s.get("escalation_rate", ""),
            "avg_tool_calls": s.get("avg_tool_calls_per_case", ""),
            "avg_react_iterations": s.get("avg_react_iterations", ""),
            "conformal_coverage": bdata.get("coverage_report", {}).get("actual_coverage", ""),
            "scoring_method": bdata.get("scoring_method", ""),
            "n_cases": s.get("total", ""),
        })

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ CSV 저장: {csv_path}")

    # 터미널 요약
    print("\n" + "="*65)
    print("UASEF Agent 실험 요약")
    print("="*65)
    print(f"{'Backend':<12} {'Acc':<8} {'Safety R.':<12} {'Over-Esc.':<12} "
          f"{'Avg Tools':<12} {'Avg Iters'}")
    print("-"*65)
    for r in rows:
        print(f"{r['backend']:<12} {str(r['accuracy']):<8} "
              f"{str(r['safety_recall']):<12} {str(r['over_escalation_rate']):<12} "
              f"{str(r['avg_tool_calls']):<12} {r['avg_react_iterations']}")


# ── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="UASEF Agent Experiment (LangGraph)")
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["lmstudio", "openai"],
        help="단일 백엔드만 실행 (기본: openai[Primary] + lmstudio[Ablation] 모두)",
    )
    parser.add_argument("--n-cal", type=int, default=500, help="Calibration 질문 수 (권장: 500)")
    parser.add_argument("--n-test", type=int, default=3, help="시나리오별 테스트 케이스 수 (권장: 50)")
    parser.add_argument(
        "--scoring-method", type=str, default="auto",
        choices=["logprob", "self_consistency", "auto"],
        help="비적합 점수 방식 강제 지정. 기본: auto (openai=logprob, lmstudio=logprob)",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Conformal prediction α (기본: base_config.yaml uqm.alpha)")
    parser.add_argument("--seed", type=int, default=42, help="데이터 샘플링 시드")
    parser.add_argument(
        "--include-pubmedqa", action="store_true",
        help="PubMedQA 'maybe' 케이스를 rare_disease 버킷에 추가",
    )
    args = parser.parse_args()

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project = os.getenv("LANGCHAIN_PROJECT", "UASEF-agent")
    if tracing == "true":
        print(f"[LangSmith] 트레이싱 활성화 — 프로젝트: {project}")
    else:
        print("[LangSmith] 트레이싱 비활성화 (.env에 LANGCHAIN_TRACING_V2=true 설정 시 활성화)")

    print(f"\n[Config] scoring={args.scoring_method}, α={args.alpha}, "
          f"n_cal={args.n_cal}, n_test={args.n_test}, seed={args.seed}")

    print("\n[Dataset] MedQA / MedAbstain 로드 중...")
    cal_questions = load_calibration_questions(n=args.n_cal, split="train", seed=args.seed)
    scenario_map = load_scenarios(n_per_scenario=args.n_test, split="test", seed=args.seed,
                                  include_pubmedqa=args.include_pubmedqa)
    agent_scenarios = [
        case_to_agent_dict(case)
        for cases in scenario_map.values()
        for case in cases
    ]
    print(f"  Calibration: {len(cal_questions)}개 | 시나리오: {len(agent_scenarios)}개")

    backends = [args.backend] if args.backend else BACKENDS

    all_results = {}
    for backend in backends:
        try:
            result = run_backend_experiment(
                backend=backend,
                cal_questions=cal_questions,
                agent_scenarios=agent_scenarios,
                scoring_method=args.scoring_method,
                alpha=args.alpha,
                distribution_source="medqa",
            )
            if result:
                all_results[backend] = result
        except Exception as e:
            print(f"\n[SKIP] {backend}: {e}")

    if all_results:
        save_results(all_results)