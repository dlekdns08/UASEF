"""
UASEF Agent — LangGraph 노드 함수 + 라우팅 함수

노드 구조:
    reason      — ChatOpenAI + 도구 바인딩 → ReAct 추론
    act         — 도구 실행, ToolMessage 반환
    uasef_check — UQM + RTC + EDE로 불확실성 판정
    escalate    — Human-in-the-loop 에스컬레이션 보고서 생성 (LLM 미사용)
    finalize    — 최종 답변 정리

AgentComponents를 functools.partial로 바인딩하므로 State에 비직렬화 객체 불포함.
"""

from __future__ import annotations
import os
import json
import functools
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from models.model_interface import ModelResponse

from models.uqm import UQM
from models.rtc_ede import RTC, EDE
from agent.state import MedicalAgentState
from agent.tools import MEDICAL_TOOLS


# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical decision support AI helping analyze complex medical cases.

Your approach:
1. Analyze the patient case carefully
2. Use tools to gather evidence — drug interactions, guidelines, lab references, differentials
3. After gathering enough evidence, synthesize into a clear clinical recommendation
4. State confidence explicitly — if uncertain, say "I am not certain"

Tool use guidelines:
- Use drug_interaction_checker for any multi-drug regimen
- Use clinical_guideline_search for management questions
- Use lab_reference_lookup when lab values need interpretation
- Use differential_diagnosis for diagnostic uncertainty

When you have enough information, provide your final recommendation WITHOUT calling additional tools.
Patient safety is paramount."""


# ── AgentComponents ───────────────────────────────────────────────────────────

@dataclass
class AgentComponents:
    """
    노드에 주입되는 UASEF 컴포넌트 묶음.
    graph.py에서 functools.partial로 각 노드 함수에 바인딩.
    """
    uqm: UQM
    rtc: RTC
    ede: EDE
    backend: str
    specialty: str = "internal_medicine"
    scenario_type: str = "routine"
    distribution_source: str = "medqa"   # calibration과 동일한 소스를 유지해야 CP 보장


# ── LLM 초기화 헬퍼 ──────────────────────────────────────────────────────────

def _make_llm(backend: str, bind_tools: bool = True) -> ChatOpenAI:
    # logprobs=True: uasef_check에서 LLM 재호출 없이 응답을 재사용하기 위해 활성화.
    # LMStudio가 지원하지 않으면 response_metadata에서 조용히 None이 됩니다.
    logprobs_kwargs = {"logprobs": True, "top_logprobs": 5}
    if backend == "lmstudio":
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model=os.getenv("LMSTUDIO_MODEL", "meta-llama-3.1-8b-instruct"),
            temperature=0.0,
            model_kwargs=logprobs_kwargs,
        )
    else:
        llm = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.0,
            model_kwargs=logprobs_kwargs,
        )
    return llm.bind_tools(MEDICAL_TOOLS) if bind_tools else llm


def _extract_model_response(msg: AIMessage, backend: str) -> Optional[ModelResponse]:
    """
    AIMessage의 response_metadata에서 ModelResponse를 재구성합니다.
    uasef_check에서 LLM 재호출 없이 logprobs를 재사용하기 위해 사용됩니다.
    logprobs가 없으면 None을 반환하고 호출부는 LLM을 재호출합니다.
    """
    try:
        meta = msg.response_metadata or {}
        logprobs_data = (meta.get("logprobs") or {}).get("content") or []
        if not logprobs_data:
            return None
        lp_list = [tok["logprob"] for tok in logprobs_data]
        top_lp_list = [
            [alt["logprob"] for alt in (tok.get("top_logprobs") or [])]
            for tok in logprobs_data
        ]
        top_lp_list = [tlp for tlp in top_lp_list if tlp] or None
        usage = meta.get("token_usage") or {}
        return ModelResponse(
            text=msg.content if isinstance(msg.content, str) else "",
            logprobs=lp_list,
            top_logprobs=top_lp_list,
            latency_ms=0.0,
            model_name=meta.get("model_name", backend),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
    except Exception:
        return None


# ── 노드 함수들 ────────────────────────────────────────────────────────────────

def reason(state: MedicalAgentState, components: AgentComponents) -> dict:
    """
    [reason 노드] ReAct 추론 단계.
    messages가 비어있으면 SystemMessage + HumanMessage로 초기화.
    LLM 응답이 tool_calls를 포함하면 → route_after_reason에서 'act'로 분기.
    """
    messages = state["messages"]

    # 첫 호출: 시스템 프롬프트 + 질문 주입
    if not messages:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]

    llm = _make_llm(components.backend)
    response: AIMessage = llm.invoke(messages)

    return {
        "messages": messages + [response] if not state["messages"] else [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def act(state: MedicalAgentState, components: AgentComponents) -> dict:
    """
    [act 노드] 도구 실행 단계.
    마지막 AIMessage의 tool_calls를 순회하며 MEDICAL_TOOLS에서 찾아 실행.
    결과를 ToolMessage로 변환 — reason 노드로 다시 라우팅됨.
    """
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {"messages": []}

    # 도구 이름 → 함수 매핑
    tool_map = {t.name: t for t in MEDICAL_TOOLS}

    tool_messages: list[ToolMessage] = []
    for call in last_msg.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        call_id = call["id"]

        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"[도구 오류] {tool_name}: {e}"
        else:
            result = f"[알 수 없는 도구] '{tool_name}'"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call_id)
        )

    return {"messages": tool_messages}


def uasef_check(state: MedicalAgentState, components: AgentComponents) -> dict:
    """
    [uasef_check 노드] UASEF 안전 게이트.
    원본 질문(state['question'])을 UQM.evaluate()에 전달 — 독립적 재판 역할.
    최신 AIMessage 텍스트도 EDE trigger 분석에 사용.

    UQM은 내부적으로 query_model()을 재호출하므로 LangGraph 히스토리와 독립적.
    이 설계가 의도적: UASEF는 에이전트 출력을 외부에서 감사(audit)하는 구조.
    """
    # 최신 AIMessage 텍스트 + 가능하면 logprobs 추출 (LLM 재호출 방지)
    response_text = ""
    pre_resp = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            response_text = msg.content if isinstance(msg.content, str) else ""
            pre_resp = _extract_model_response(msg, components.backend)
            break

    # UQM 평가 (calibration과 동일한 distribution_source 유지 — CP exchangeability)
    # pre_resp가 있으면 logprob 모드에서 LLM 재호출 생략
    unc = components.uqm.evaluate(
        state["question"],
        distribution_source=components.distribution_source,
        pre_computed_response=pre_resp,
    )

    # RTC 임계값 조정
    rtc_config = components.rtc.get_threshold(
        components.specialty,
        components.scenario_type,
    )

    # EDE 에스컬레이션 결정
    decision = components.ede.decide(unc, rtc_config, response_text)

    return {
        "uasef_score": round(unc.nonconformity_score, 4),
        "uasef_threshold": round(rtc_config.adjusted_threshold, 4),
        "uasef_triggers": [t.value for t in decision.triggers],
        "uasef_confidence": round(decision.confidence, 4),
        "uasef_explanation": decision.explanation,
        "should_escalate": decision.should_escalate,
    }


def escalate(state: MedicalAgentState, components: AgentComponents) -> dict:
    """
    [escalate 노드] Human-in-the-loop 에스컬레이션.
    LLM을 추가 호출하지 않고 UASEF 결과로 구조화된 보고서 생성.
    실제 시스템에서는 여기서 EHR 알림, Slack, 페이징 시스템 호출.
    """
    score = state.get("uasef_score", 0)
    threshold = state.get("uasef_threshold", 0)
    margin = threshold - score  # 음수일수록 임계값을 많이 초과한 것
    triggers = state.get("uasef_triggers") or []
    confidence = state.get("uasef_confidence", 0)
    explanation = state.get("uasef_explanation", "")
    question = state.get("question", "")

    report = (
        f"[UASEF ESCALATION — 전문의 확인 필요]\n"
        f"{'─'*50}\n"
        f"질문: {question[:120]}\n"
        f"{'─'*50}\n"
        f"불확실성 점수: {score:.4f} (임계값: {threshold:.4f})\n"
        f"임계값 초과 마진: {abs(margin):.4f} "
        f"({'높은' if abs(margin) > 0.5 else '낮은'} 초과)\n"
        f"에스컬레이션 확신도: {confidence:.2f}\n"
        f"활성 트리거:\n"
        + "\n".join(f"  • {t}" for t in triggers)
        + f"\n판정 근거: {explanation}\n"
        f"{'─'*50}\n"
        f"권고: 담당 전문의에게 즉시 인계하고 자율 행동을 중단합니다."
    )

    print(f"\n{'='*55}")
    print(report)
    print(f"{'='*55}")

    return {
        "final_answer": report,
        "escalation_reason": explanation,
    }


def finalize(state: MedicalAgentState, components: AgentComponents) -> dict:
    """
    [finalize 노드] 에스컬레이션 불필요 → 최종 답변 확정.
    메시지 히스토리에서 마지막 실질적 AIMessage를 final_answer로 저장.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            return {"final_answer": answer}
    return {"final_answer": "답변을 생성하지 못했습니다."}


# ── 라우팅 함수 ────────────────────────────────────────────────────────────────

def route_after_reason(
    state: MedicalAgentState,
) -> Literal["act", "uasef_check"]:
    """
    reason 노드 이후 분기.
    - 마지막 AIMessage에 tool_calls 존재 AND 반복 횟수 미달 → 'act'
    - 그 외 (최종 답변 생성 or 반복 한계) → 'uasef_check'
    """
    last_msg = state["messages"][-1]
    has_tool_calls = (
        isinstance(last_msg, AIMessage)
        and bool(last_msg.tool_calls)
    )
    under_limit = state.get("iteration", 0) < state.get("max_iterations", 5)

    if has_tool_calls and under_limit:
        return "act"
    return "uasef_check"


def route_after_uasef(
    state: MedicalAgentState,
) -> Literal["escalate", "finalize"]:
    """
    uasef_check 노드 이후 분기.
    - should_escalate=True  → 'escalate'
    - should_escalate=False → 'finalize'
    """
    return "escalate" if state.get("should_escalate") else "finalize"