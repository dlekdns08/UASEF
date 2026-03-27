"""
UASEF — Module 2: Risk-Threshold Calibrator (RTC)
         Module 3: Escalation Decision Engine (EDE)

RTC: 전문과목·시나리오별 동적 임계값 조정
EDE: 3가지 트리거 + 엔트로피 가중치 기반 Human-in-the-Loop 에스컬레이션 결정
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from models.uqm import UncertaintyResult


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
        # 시나리오 보정: 응급·희귀질환은 추가로 낮춤 (더 보수적)
        if self.scenario_type in ("emergency", "rare_disease"):
            multiplier *= 0.85
        self.adjusted_threshold = self.base_threshold * multiplier


class RTC:
    """
    Risk-Threshold Calibrator.

    사용법:
        rtc = RTC(base_threshold=2.31)  # UQM.calibrator.threshold
        config = rtc.get_threshold("emergency_medicine", "emergency")
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
        experiment_results: Optional[dict] = None,
    ) -> list[dict]:
        """
        Coverage guarantee ↔ Escalation rate Pareto frontier 포인트를 반환합니다.

        experiment_results가 제공되면 실험 데이터에서 escalation_rate를 추출합니다.
        없으면 α와 specialty 배율만으로 추정합니다 (시뮬레이션).
        """
        alphas = alphas or [0.01, 0.05, 0.10, 0.15, 0.20]
        specialties = specialties or ["emergency_medicine", "internal_medicine", "general_practice"]
        results = []

        for alpha in alphas:
            for spec in specialties:
                cfg = RTCConfig(spec, "routine", self.base_threshold)
                # 실험 데이터에서 escalation_rate 추출 시도
                esc_rate = None
                if experiment_results:
                    for backend_data in experiment_results.values():
                        for sc_data in backend_data.get("scenarios", {}).values():
                            m = sc_data.get("metrics", {})
                            if m:
                                esc_rate = m.get("escalation_rate")
                                break
                        if esc_rate is not None:
                            break

                results.append({
                    "alpha": alpha,
                    "specialty": spec,
                    "risk_level": cfg.risk_level.value,
                    "adjusted_threshold": cfg.adjusted_threshold,
                    "estimated_coverage": 1 - alpha,
                    "escalation_rate": esc_rate,   # None이면 실험 전
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


# 고위험 키워드 목록 (실제 연구에서는 SNOMED CT 등 의료 온톨로지로 확장)
HIGH_RISK_KEYWORDS = {
    "intubate", "intubation", "mechanical ventilation",
    "code blue", "emergency surgery",
    "thrombolysis", "tpa", "alteplase",
    "vasopressor", "norepinephrine", "epinephrine",
    "defibrillate", "defibrillation",
    "do not resuscitate", "dnr", "withdraw care", "comfort measures only",
}

NO_EVIDENCE_PHRASES = {
    "i am not certain", "i'm not certain",
    "i don't know", "i do not know",
    "insufficient evidence", "no clear guideline",
    "limited data", "unknown etiology",
    "case report only", "experimental", "off-label",
}

# 엔트로피 기반 불확실성 판단 임계값 (nats/token)
# GPT-4 수준 모델에서 통상 0.5~1.5 nats/token; 2.0 이상은 높은 불확실성
ENTROPY_HIGH_THRESHOLD = 2.0


class EDE:
    """
    Escalation Decision Engine.

    트리거 구조:
        Trigger 1 — UNCERTAINTY_EXCEEDED: nonconformity_score > adjusted_threshold
        Trigger 2 — HIGH_RISK_ACTION:     고위험 임상 키워드 감지
        Trigger 3 — NO_EVIDENCE:          근거 부재 표현 감지
        Entropy 가중치 (Trigger 아님): logprobs 기반 entropy가 높으면 confidence +0.15

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

        # Confidence 계산:
        #   - 기본: trigger 수 / 3
        #   - UNCERTAINTY_EXCEEDED 포함 시 +0.4 (주 트리거이므로 가중치)
        #   - logprobs 기반 entropy가 높으면 +0.15 (2차 신호)
        entropy = uncertainty_result.confidence_entropy
        entropy_boost = (
            0.15
            if not math.isnan(entropy) and entropy > ENTROPY_HIGH_THRESHOLD
            else 0.0
        )
        confidence = min(1.0,
            len(triggers) / 3
            + (0.4 if EscalationTrigger.UNCERTAINTY_EXCEEDED in triggers else 0.0)
            + entropy_boost
        )

        explanation = self._build_explanation(triggers, rtc_config, uncertainty_result, entropy)

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
                "entropy": entropy if not math.isnan(entropy) else None,
                "entropy_boost": entropy_boost,
                "scoring_method": uncertainty_result.scoring_method,
            },
        )
        self.escalation_log.append(decision)
        return decision

    @staticmethod
    def _build_explanation(
        triggers: list[EscalationTrigger],
        rtc_config: RTCConfig,
        unc: UncertaintyResult,
        entropy: float,
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
        if not math.isnan(entropy) and entropy > ENTROPY_HIGH_THRESHOLD:
            parts.append(f"높은 토큰 엔트로피({entropy:.2f} nats/token)")
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
    from models.model_interface import ModelResponse
    from models.uqm import UncertaintyResult

    # 가상 UncertaintyResult
    fake_resp = ModelResponse("Unknown rare disease.", None, 300, "test", 20, 10)
    unc = UncertaintyResult(
        nonconformity_score=3.5,
        prediction_set_size=5,
        confidence_entropy=2.5,   # 높은 엔트로피
        should_escalate=True,
        threshold_used=2.31,
        raw_response=fake_resp,
        scoring_method="logprob",
    )

    rtc = RTC(base_threshold=2.31)
    cfg = rtc.get_threshold("emergency_medicine", "emergency")
    print(f"Adjusted threshold: {cfg.adjusted_threshold:.3f} (risk={cfg.risk_level.value})")

    ede = EDE()
    decision = ede.decide(unc, cfg, "I am not certain. Consider intubation.")
    print(f"Escalate: {decision.should_escalate}")
    print(f"Triggers: {[t.value for t in decision.triggers]}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Explanation: {decision.explanation}")
    print(json.dumps(ede.summary(), indent=2, ensure_ascii=False))