"""
UASEF Agent — State 정의

MedicalAgentState: LangGraph StateGraph의 공유 상태 TypedDict.
messages 필드만 operator.add 리듀서로 누적, 나머지는 덮어쓰기.
"""

from __future__ import annotations
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class MedicalAgentState(TypedDict):
    # 대화 히스토리 — 각 노드가 append, LangGraph가 operator.add로 병합
    messages: Annotated[list[BaseMessage], operator.add]

    # 원본 질문 (변경 없음, 노드 참조용)
    question: str

    # ReAct 루프 제어
    iteration: int        # 현재 reason 호출 횟수
    max_iterations: int   # 안전 탈출 임계값 (기본 5)

    # UASEF 판정 결과 (uasef_check 노드가 채움)
    uasef_score: Optional[float]
    uasef_threshold: Optional[float]
    uasef_triggers: Optional[list[str]]    # EscalationTrigger.value 목록
    uasef_confidence: Optional[float]
    uasef_explanation: Optional[str]
    should_escalate: Optional[bool]

    # 최종 출력
    final_answer: Optional[str]
    escalation_reason: Optional[str]