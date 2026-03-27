"""
UASEF — Module 2: Risk-Threshold Calibrator (RTC)
         Module 3: Escalation Decision Engine (EDE)

RTC: 전문과목·시나리오별 동적 임계값 조정
EDE: 3가지 트리거 기반 Human-in-the-Loop 에스컬레이션 결정
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from modules.uqm import UncertaintyResult


# ═══════════════════════════════════════════════════════════════════════════════
# RTC — Risk-Threshold Calibrator
# ═══════════════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    CRITICAL = "critical"     # 즉각 처치 필요 (응급의학, 중환자의학)
    HIGH = "high"             # 고위험 (심장외과, 신경외과)
    MODERATE = "moderate"     # 중등도 (내과, 외과)
    LOW = "low"               # 저위험 (일반 외래, 예방의학)


# 전문과목별 위험 온톨로지
SPECIALTY_RISK_MAP: dict[str, RiskLevel] = {
    # Critical
    "emergency_medicine": RiskLevel.CRITICAL,
    "intensive_care": RiskLevel.CRITICAL,
    "trauma_surgery": RiskLevel.CRITICAL,
    # High
    "cardiology": RiskLevel.HIGH,
    "neurology": RiskLevel.HIGH,
    "oncology": RiskLevel.HIGH,
    "cardiothoracic_surgery": RiskLevel.HIGH,
    # Moderate
    "internal_medicine": RiskLevel.MODERATE,
    "surgery": RiskLevel.MODERATE,
    "pediatrics": RiskLevel.MODERATE,
    "obstetrics": RiskLevel.MODERATE,
    # Low
    "general_practice": RiskLevel.LOW,
    "preventive_medicine": RiskLevel.LOW,
    "dermatology": RiskLevel.LOW,
    "psychiatry": RiskLevel.LOW,
}

# 위험도별 기본 임계값 배율 (base_threshold × multiplier)
RISK_THRESHOLD_MULTIPLIER: dict[RiskLevel, float] = {
    RiskLevel.CRITICAL: 0.60,   # 더 보수적 (낮은 threshold → 더 많이 에스컬레이션)
    RiskLevel.HIGH:     0.75,
    RiskLevel.MODERATE: 1.00,   # 기준값
    RiskLevel.LOW:      1.30,   # 덜 보수적
}


@dataclass
class RTCConfig:
    specialty: str
    scenario_type: str          # "emergency" | "rare_disease" | "multimorbidity" | "routine"
    base_threshold: float       # UQM calibration으로부터 온 q̂
    adjusted_threshold: float = 0.0
    risk_level: RiskLevel = RiskLevel.MODERATE

    def __post_init__(self):
        self.risk_level = SPECIALTY_RISK_MAP.get(self.specialty, RiskLevel.MODERATE)
        multiplier = RISK_THRESHOLD_MULTIPLIER[self.risk_level]
        # 시나리오 보정: 응급·희귀질환은 추가로 낮춤
        if self.scenario_type in ("emergency", "rare_disease"):
            multiplier *= 0.85
        self.adjusted_threshold = self.base_threshold * multiplier


class RTC:
    """
    Risk-Threshold Calibrator.

    사용법:
        rtc = RTC(base_threshold=2.31)  # UQM.calibrator.threshold
        threshold = rtc.get_threshold("emergency_medicine", "emergency")
    """

    def __init__(self, base_threshold: float):
        self.base_threshold = base_threshold

    def get_threshold(self, specialty: str, scenario_type: str) -> RTCConfig:
        return RTCConfig(
            specialty=specialty,
            scenario_type=scenario_type,
            base_threshold=self.base_threshold,
        )

    def pareto_frontier(
        self,
        alphas: list[float] = None,
        specialties: list[str] = None,
    ) -> list[dict]:
        """
        Coverage guarantee ↔ Escalation rate Pareto frontier 포인트를 반환합니다.
        실제 연구에서는 calibration set을 순회하며 계산해야 합니다.
        여기서는 시뮬레이션 값을 반환합니다.
        """
        alphas = alphas or [0.01, 0.05, 0.10, 0.15, 0.20]
        specialties = specialties or ["emergency_medicine", "internal_medicine", "general_practice"]
        results = []
        for alpha in alphas:
            for spec in specialties:
                cfg = RTCConfig(spec, "routine", self.base_threshold)
                results.append({
                    "alpha": alpha,
                    "specialty": spec,
                    "risk_level": cfg.risk_level.value,
                    "adjusted_threshold": cfg.adjusted_threshold,
                    "estimated_coverage": 1 - alpha,
                    # escalation_rate는 실험 후 채워야 함
                })
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# EDE — Escalation Decision Engine
# ═══════════════════════════════════════════════════════════════════════════════

class EscalationTrigger(Enum):
    UNCERTAINTY_EXCEEDED = "uncertainty_threshold_exceeded"
    HIGH_RISK_ACTION = "high_risk_action_category"
    NO_EVIDENCE = "no_evidence_available"


@dataclass
class EscalationDecision:
    should_escalate: bool
    triggers: list[EscalationTrigger]
    confidence: float              # 에스컬레이션 확신도 (0~1)
    explanation: str
    rtc_config: RTCConfig
    uncertainty_result: UncertaintyResult
    log: dict = field(default_factory=dict)


# 고위험 키워드 목록 (실제 연구에서는 의료 온톨로지로 확장)
HIGH_RISK_KEYWORDS = {
    "intubate", "intubation", "code blue", "emergency surgery",
    "thrombolysis", "tpa", "vasopressor", "epinephrine", "defibrillate",
    "do not resuscitate", "dnr", "withdraw care", "comfort measures only",
}

NO_EVIDENCE_PHRASES = {
    "i am not certain", "i don't know", "insufficient evidence",
    "no clear guideline", "limited data", "unknown etiology",
    "case report only", "experimental",
}


class EDE:
    """
    Escalation Decision Engine.

    사용법:
        ede = EDE()
        decision = ede.decide(uncertainty_result, rtc_config, response_text)
        if decision.should_escalate:
            hand_off_to_clinician(decision)
    """

    def __init__(self):
        self.escalation_log: list[EscalationDecision] = []

    def decide(
        self,
        uncertainty_result: UncertaintyResult,
        rtc_config: RTCConfig,
        response_text: str,
    ) -> EscalationDecision:
        triggers = []
        text_lower = response_text.lower()

        # Trigger 1: 불확실성 임계 초과
        if uncertainty_result.nonconformity_score > rtc_config.adjusted_threshold:
            triggers.append(EscalationTrigger.UNCERTAINTY_EXCEEDED)

        # Trigger 2: 고위험 행동 키워드 감지
        if any(kw in text_lower for kw in HIGH_RISK_KEYWORDS):
            triggers.append(EscalationTrigger.HIGH_RISK_ACTION)

        # Trigger 3: 근거 부재 표현 감지
        if any(ph in text_lower for ph in NO_EVIDENCE_PHRASES):
            triggers.append(EscalationTrigger.NO_EVIDENCE)

        should_escalate = len(triggers) > 0
        confidence = min(1.0, len(triggers) / 3 + (
            0.4 if EscalationTrigger.UNCERTAINTY_EXCEEDED in triggers else 0.0
        ))

        explanation = self._build_explanation(triggers, rtc_config, uncertainty_result)

        decision = EscalationDecision(
            should_escalate=should_escalate,
            triggers=triggers,
            confidence=confidence,
            explanation=explanation,
            rtc_config=rtc_config,
            uncertainty_result=uncertainty_result,
            log={
                "score": uncertainty_result.nonconformity_score,
                "threshold": rtc_config.adjusted_threshold,
                "specialty": rtc_config.specialty,
                "risk_level": rtc_config.risk_level.value,
                "trigger_count": len(triggers),
            },
        )
        self.escalation_log.append(decision)
        return decision

    @staticmethod
    def _build_explanation(
        triggers: list[EscalationTrigger],
        rtc_config: RTCConfig,
        unc: UncertaintyResult,
    ) -> str:
        parts = []
        if EscalationTrigger.UNCERTAINTY_EXCEEDED in triggers:
            parts.append(
                f"불확실성 점수({unc.nonconformity_score:.3f}) > "
                f"임계값({rtc_config.adjusted_threshold:.3f})"
            )
        if EscalationTrigger.HIGH_RISK_ACTION in triggers:
            parts.append("고위험 임상 행동 키워드 감지")
        if EscalationTrigger.NO_EVIDENCE in triggers:
            parts.append("근거 부재 표현 감지")
        if not parts:
            return "에스컬레이션 불필요 — 자율 행동 가능"
        return " | ".join(parts)

    def summary(self) -> dict:
        total = len(self.escalation_log)
        escalated = sum(1 for d in self.escalation_log if d.should_escalate)
        return {
            "total_queries": total,
            "escalated": escalated,
            "escalation_rate": escalated / total if total else 0,
            "trigger_breakdown": {
                t.value: sum(
                    1 for d in self.escalation_log if t in d.triggers
                )
                for t in EscalationTrigger
            },
        }


# ── 빠른 확인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.uqm import UncertaintyResult, ModelResponse

    # 가상 UncertaintyResult
    fake_resp = ModelResponse("Unknown rare disease.", None, 300, "test", 20, 10)
    unc = UncertaintyResult(
        nonconformity_score=3.5,
        prediction_set_size=5,
        confidence_entropy=float("nan"),
        should_escalate=True,
        threshold_used=2.31,
        raw_response=fake_resp,
    )

    rtc = RTC(base_threshold=2.31)
    cfg = rtc.get_threshold("emergency_medicine", "emergency")
    print(f"Adjusted threshold: {cfg.adjusted_threshold:.3f} (risk={cfg.risk_level.value})")

    ede = EDE()
    decision = ede.decide(unc, cfg, "I am not certain. Consider intubation.")
    print(f"Escalate: {decision.should_escalate}")
    print(f"Triggers: {[t.value for t in decision.triggers]}")
    print(f"Explanation: {decision.explanation}")
    print(json.dumps(ede.summary(), indent=2, ensure_ascii=False))
