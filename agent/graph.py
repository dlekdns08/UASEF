"""
UASEF Agent — LangGraph StateGraph 조립

그래프 흐름:
    START → reason
    reason → (tool_calls & iter < max) → act → reason  [ReAct 루프]
           → (no tool_calls | 한계) → uasef_check
    uasef_check → (should_escalate) → escalate → END
                → (safe)           → finalize  → END

LangSmith 트레이싱:
    .env에 LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY 설정 시 자동 활성화.
    별도 코드 불필요.
"""

from __future__ import annotations
import functools
import os

from langgraph.graph import StateGraph, END

from models.uqm import UQM
from models.rtc_ede import RTC, EDE
from agent.state import MedicalAgentState
from agent.nodes import (
    AgentComponents,
    reason, act, uasef_check, escalate, finalize,
    route_after_reason, route_after_uasef,
)


def build_graph(components: AgentComponents):
    """
    UASEF 통합 의료 에이전트 그래프를 빌드하고 컴파일합니다.

    Args:
        components: 사전 보정된 UQM, RTC, EDE, backend 정보를 포함한 컴포넌트 묶음.

    Returns:
        CompiledGraph: LangGraph 컴파일 그래프. .invoke() / .stream()으로 실행.
    """
    # 컴포넌트를 각 노드 함수에 부분 적용 (functools.partial)
    # — State에 비직렬화 객체를 넣지 않고 클로저로 전달
    _reason      = functools.partial(reason,      components=components)
    _act         = functools.partial(act,          components=components)
    _uasef_check = functools.partial(uasef_check,  components=components)
    _escalate    = functools.partial(escalate,     components=components)
    _finalize    = functools.partial(finalize,     components=components)

    builder = StateGraph(MedicalAgentState)

    # 노드 등록
    builder.add_node("reason",      _reason)
    builder.add_node("act",         _act)
    builder.add_node("uasef_check", _uasef_check)
    builder.add_node("escalate",    _escalate)
    builder.add_node("finalize",    _finalize)

    # 엣지 정의
    builder.set_entry_point("reason")

    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {"act": "act", "uasef_check": "uasef_check"},
    )
    builder.add_edge("act", "reason")       # ReAct 루프: 도구 실행 후 재추론

    builder.add_conditional_edges(
        "uasef_check",
        route_after_uasef,
        {"escalate": "escalate", "finalize": "finalize"},
    )
    builder.add_edge("escalate", END)
    builder.add_edge("finalize",  END)

    return builder.compile()


def make_initial_state(
    question: str,
    max_iterations: int = 5,
) -> MedicalAgentState:
    """빈 초기 상태를 생성합니다."""
    return MedicalAgentState(
        messages=[],
        question=question,
        iteration=0,
        max_iterations=max_iterations,
        uasef_score=None,
        uasef_threshold=None,
        uasef_triggers=None,
        uasef_confidence=None,
        uasef_explanation=None,
        should_escalate=None,
        final_answer=None,
        escalation_reason=None,
    )