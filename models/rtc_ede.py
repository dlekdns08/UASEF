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

# 위험도별 기본 배율 (데이터 부족 시 사용; run_calibration_pipeline.py로 재산출 가능)
_DEFAULT_MULTIPLIERS: dict = {}  # RiskLevel 키는 아래 enum 정의 후 채워짐


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
# ⚠ 이 값은 하드코딩 기본값입니다.
#   run_calibration_pipeline.py 실행 후 base_config.yaml의 rtc 섹션에 저장된
#   데이터 기반 배율이 RTC(base_threshold, multipliers=cfg["rtc"]) 형태로 주입됩니다.
RISK_THRESHOLD_MULTIPLIER: dict[RiskLevel, float] = {
    RiskLevel.CRITICAL: 0.60,   # 더 보수적 (낮은 threshold → 더 많이 에스컬레이션)
    RiskLevel.HIGH:     0.75,
    RiskLevel.MODERATE: 1.00,   # 기준값
    RiskLevel.LOW:      1.30,   # 덜 보수적
}

# _DEFAULT_MULTIPLIERS를 RiskLevel enum 정의 후 채움 (모듈 초기화용)
_DEFAULT_MULTIPLIERS.update(RISK_THRESHOLD_MULTIPLIER)


@dataclass
class RTCConfig:
    specialty: str
    scenario_type: str          # "emergency" | "rare_disease" | "multimorbidity" | "routine"
    base_threshold: float       # UQM calibration으로부터 온 q̂
    adjusted_threshold: float = 0.0
    risk_level: RiskLevel = RiskLevel.MODERATE
    # run_calibration_pipeline.py 또는 base_config.yaml에서 주입되는 데이터 기반 배율
    # None이면 모듈 기본값 RISK_THRESHOLD_MULTIPLIER 사용
    multipliers: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        self.risk_level = SPECIALTY_RISK_MAP.get(self.specialty, RiskLevel.MODERATE)
        mmap = self.multipliers if self.multipliers is not None else RISK_THRESHOLD_MULTIPLIER
        multiplier = mmap.get(self.risk_level, RISK_THRESHOLD_MULTIPLIER[self.risk_level])
        # 시나리오 보정: 응급·희귀질환은 추가로 낮춤 (더 보수적)
        if self.scenario_type in ("emergency", "rare_disease"):
            multiplier *= 0.85
        self.adjusted_threshold = self.base_threshold * multiplier


class RTC:
    """
    Risk-Threshold Calibrator.

    base_threshold(UQM calibration의 q̂)에 전문과목·시나리오별 배율을 적용합니다.

    배율 주입 흐름:
        1. run_calibration_pipeline.py → Pareto sweep으로 위험도별 최적 배율 산출
        2. base_config.yaml rtc 섹션에 저장
        3. run_experiment.py → cfg["rtc"]를 RTC(base_threshold, multipliers=...) 에 전달
        4. RTCConfig.__post_init__()에서 데이터 기반 배율로 adjusted_threshold 계산

    multipliers 인자 없이 생성하면 RISK_THRESHOLD_MULTIPLIER 기본값을 사용합니다.

    사용법:
        rtc = RTC(base_threshold=2.31)                          # 기본 배율
        rtc = RTC(base_threshold=2.31, multipliers=cfg["rtc"]) # 데이터 기반 배율
        config = rtc.get_threshold("emergency_medicine", "emergency")
    """

    def __init__(
        self,
        base_threshold: float,
        multipliers: Optional[dict[str, float]] = None,
    ):
        self.base_threshold = base_threshold
        # config에서 전달되는 배율 (대문자 문자열 키 "CRITICAL" 등을 RiskLevel enum으로 변환)
        self._multipliers: dict[RiskLevel, float] = dict(RISK_THRESHOLD_MULTIPLIER)
        if multipliers:
            for k, v in multipliers.items():
                try:
                    self._multipliers[RiskLevel(k.lower())] = float(v)
                except ValueError:
                    pass  # 알 수 없는 키 무시

    def get_threshold(self, specialty: str, scenario_type: str) -> RTCConfig:
        return RTCConfig(
            specialty=specialty,
            scenario_type=scenario_type,
            base_threshold=self.base_threshold,
            multipliers=self._multipliers,
        )

    def pareto_frontier(
        self,
        sweep_results: Optional[list[dict]] = None,
        alphas: list[float] = None,
        specialties: list[str] = None,
    ) -> list[dict]:
        """
        Coverage guarantee ↔ Escalation rate Pareto frontier 포인트를 반환합니다.

        sweep_results가 제공되면 pareto_sweep.py의 실측 데이터를 사용합니다.
        없으면 α와 specialty 배율로 이론값(시뮬레이션)만 반환합니다.

        Args:
            sweep_results: pareto_sweep.py run_pareto_sweep()의 반환값.
                           각 항목: {alpha, specialty, scenario_type,
                                     actual_coverage, escalation_rate, ...}
            alphas:        sweep_results 없을 때 사용할 α 값 목록.
            specialties:   sweep_results 없을 때 사용할 전문과목 목록.
        """
        # ── 실측 데이터 경로 ────────────────────────────────────────────────────
        if sweep_results:
            enriched = []
            for pt in sweep_results:
                spec = pt.get("specialty", "internal_medicine")
                st = pt.get("scenario_type", "routine")
                cfg = RTCConfig(spec, st, self.base_threshold)
                enriched.append({
                    **pt,
                    "risk_level": cfg.risk_level.value,
                    # base_threshold가 달라도 배율 정보는 유지
                    "adjusted_threshold": round(cfg.adjusted_threshold, 4),
                })
            return enriched

        # ── 이론값 추정 (실측 없을 때) ──────────────────────────────────────────
        # actual_coverage / escalation_rate = None 으로 표시:
        # pareto_sweep.py를 실행하면 채워집니다.
        alphas = alphas or [0.01, 0.05, 0.10, 0.15, 0.20]
        specialties = specialties or [
            "emergency_medicine", "internal_medicine", "general_practice"
        ]
        results = []
        for alpha in alphas:
            for spec in specialties:
                cfg = RTCConfig(spec, "routine", self.base_threshold)
                results.append({
                    "alpha": alpha,
                    "specialty": spec,
                    "risk_level": cfg.risk_level.value,
                    "adjusted_threshold": round(cfg.adjusted_threshold, 4),
                    "target_coverage": round(1 - alpha, 4),
                    "actual_coverage": None,    # 미측정 — pareto_sweep.py 실행 필요
                    "escalation_rate": None,    # 미측정 — pareto_sweep.py 실행 필요
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


# ── Trigger 2: HIGH_RISK_ACTION — 2단계 맥락 인식 구조 ──────────────────────
#
# [Stage 1] CRITICAL_KEYWORDS: EOL·위기 결정 → 문맥 무관 무조건 트리거
#   이 범주는 AI가 단독으로 판단해서는 안 되는 결정 영역.
#   응급 소생 여부(DNR), 연명치료 중단(withdraw care) 등이 해당.
#   키워드가 발견되면 나머지 조건 없이 즉시 트리거.
#
# [Stage 2] PROCEDURAL_KEYWORDS + UNCERTAINTY_MODIFIERS: 맥락 조건부 트리거
#   처치 자체(intubation, epinephrine 등)는 지침 권고일 수 있음.
#   → "give epinephrine for anaphylaxis"   → 트리거 안 함 (명확한 처치)
#   → "consider intubation if deteriorates" → 트리거   (불확실 맥락)
#   불확실 표현(UNCERTAINTY_MODIFIERS)이 동반될 때만 HIGH_RISK_ACTION 발동.
#
# 이 설계는 과-에스컬레이션(over-escalation)을 줄이기 위한 핵심 구조임.
# 논문 방법론 섹션에 반드시 수식 또는 의사코드로 명시해야 재현 가능.
CRITICAL_KEYWORDS = {
    "code blue",
    "do not resuscitate", "dnr",
    "withdraw care", "comfort measures only",
}

# PROCEDURAL_KEYWORDS: 시술/처치 자체는 지침 권고일 수 있으므로 맥락 필요.
#   UNCERTAINTY_MODIFIERS와 함께 나타날 때만 에스컬레이션.
#   예) "give epinephrine for anaphylaxis" → 미트리거
#       "consider intubation if patient deteriorates" → 트리거
PROCEDURAL_KEYWORDS = {
    "intubate", "intubation", "mechanical ventilation",
    "emergency surgery",
    "thrombolysis", "tpa", "alteplase",
    "vasopressor", "norepinephrine", "epinephrine",
    "defibrillate", "defibrillation",
}

# PROCEDURAL_KEYWORDS 트리거를 활성화하는 불확실 표현
UNCERTAINTY_MODIFIERS = {
    "consider", "may need", "might require", "possibly",
    "if deteriorates", "if worsens", "borderline", "unclear",
    "not certain", "uncertain", "limited evidence", "off-label",
}

# 출처: ① MedAbstain AP 샘플 빈도 상위, ② Savage et al. 2025, ③ 수동 코딩 (GPT-4o 500건)
NO_EVIDENCE_PHRASES: list[dict] = [
    # --- MedAbstain AP 샘플 출처 ---
    {"phrase": "i am not certain",          "source": "medabstain"},
    {"phrase": "i'm not certain",           "source": "medabstain"},
    {"phrase": "i'm not sure",              "source": "medabstain"},
    {"phrase": "insufficient evidence",     "source": "medabstain"},
    {"phrase": "limited information",       "source": "medabstain"},
    {"phrase": "limited data",              "source": "medabstain"},
    {"phrase": "cannot determine",          "source": "medabstain"},
    # --- Savage et al. 2025 출처 ---
    {"phrase": "this is unclear",           "source": "savage2025"},
    {"phrase": "evidence is mixed",         "source": "savage2025"},
    {"phrase": "conflicting data",          "source": "savage2025"},
    {"phrase": "requires further workup",   "source": "savage2025"},
    {"phrase": "beyond my knowledge",       "source": "savage2025"},
    {"phrase": "no clear guideline",        "source": "savage2025"},
    # --- 수동 코딩 (GPT-4o 500건) 출처 ---
    {"phrase": "i don't know",              "source": "manual"},
    {"phrase": "i do not know",             "source": "manual"},
    {"phrase": "unknown etiology",          "source": "manual"},
    {"phrase": "case report only",          "source": "manual"},
    {"phrase": "experimental",              "source": "manual"},
    {"phrase": "off-label",                 "source": "manual"},
    {"phrase": "i would recommend consulting", "source": "manual"},
    {"phrase": "not enough context",        "source": "manual"},
    {"phrase": "differential is broad",     "source": "manual"},
    {"phrase": "clinical judgment needed",  "source": "manual"},
]

# 실제 탐지에 사용할 문자열 집합 (빠른 멤버십 검사용)
NO_EVIDENCE_STRINGS: set[str] = {p["phrase"] for p in NO_EVIDENCE_PHRASES}


def detect_no_evidence(text: str) -> tuple[bool, list[str]]:
    """
    텍스트에서 근거 부재 표현을 탐지합니다.

    Returns:
        (triggered: bool, matched_phrases: list[str])
        논문 재현을 위해 매칭된 문구도 함께 반환합니다.
    """
    text_lower = text.lower()
    matched = [p for p in NO_EVIDENCE_STRINGS if p in text_lower]
    return len(matched) > 0, matched


# 엔트로피 기반 불확실성 판단 기본 임계값 (nats/token)
# ⚠ 이 값은 하드코딩 기본값입니다.
#   run_calibration_pipeline.py → Youden's J로 자동 결정 → base_config.yaml에 저장
#   EDE(entropy_threshold=cfg["entropy_threshold"]) 형태로 주입됩니다.
ENTROPY_HIGH_THRESHOLD = 2.0


class EDE:
    """
    Escalation Decision Engine.

    트리거 구조:
        Trigger 1 — UNCERTAINTY_EXCEEDED: nonconformity_score > adjusted_threshold
        Trigger 2 — HIGH_RISK_ACTION:     고위험 임상 키워드 감지
        Trigger 3 — NO_EVIDENCE:          근거 부재 표현 감지
        Entropy 가중치 (Trigger 아님): logprobs 기반 entropy가 높으면 confidence += entropy_boost

    계수 주입 흐름:
        1. run_calibration_pipeline.py → grid_search_ede_coefficients()로 최적 계수 산출
        2. base_config.yaml ede 섹션에 저장
        3. run_experiment.py → EDE(t1_weight=..., entropy_boost=..., entropy_threshold=...) 전달

    사용법:
        ede = EDE()                                          # 기본 계수
        ede = EDE(t1_weight=0.4, entropy_boost=0.15,        # 데이터 기반 계수
                  entropy_threshold=2.07)
        decision = ede.decide(uncertainty_result, rtc_config, response_text)
        if decision.should_escalate:
            hand_off_to_clinician(decision)
    """

    def __init__(
        self,
        t1_weight: float = 0.4,
        entropy_boost: float = 0.15,
        entropy_threshold: float = ENTROPY_HIGH_THRESHOLD,
    ):
        self.t1_weight = t1_weight
        self.entropy_boost = entropy_boost
        self.entropy_threshold = entropy_threshold
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

        # Trigger 2: 고위험 행동 키워드 감지 (맥락 인식)
        #   - CRITICAL_KEYWORDS: EOL/위기 결정 → 문맥 무관 트리거
        #   - PROCEDURAL_KEYWORDS: 시술 키워드 → 불확실 표현 동반 시만 트리거
        has_critical = any(kw in text_lower for kw in CRITICAL_KEYWORDS)
        has_uncertain_procedure = (
            any(kw in text_lower for kw in PROCEDURAL_KEYWORDS)
            and any(mod in text_lower for mod in UNCERTAINTY_MODIFIERS)
        )
        if has_critical or has_uncertain_procedure:
            triggers.append(EscalationTrigger.HIGH_RISK_ACTION)

        # Trigger 3: 근거 부재 표현 감지
        triggered, _ = detect_no_evidence(response_text)
        if triggered:
            triggers.append(EscalationTrigger.NO_EVIDENCE)

        should_escalate = len(triggers) > 0

        # Confidence 계산:
        #   - 기본: trigger 수 / 3
        #   - UNCERTAINTY_EXCEEDED 포함 시 +t1_weight (주 트리거이므로 가중치)
        #   - logprobs 기반 entropy가 높으면 +entropy_boost (2차 신호)
        entropy = uncertainty_result.confidence_entropy
        entropy_boost = (
            self.entropy_boost
            if not math.isnan(entropy) and entropy > self.entropy_threshold
            else 0.0
        )
        confidence = min(1.0,
            len(triggers) / 3
            + (self.t1_weight if EscalationTrigger.UNCERTAINTY_EXCEEDED in triggers else 0.0)
            + entropy_boost
        )

        explanation = self._build_explanation(
            triggers, rtc_config, uncertainty_result, entropy, self.entropy_threshold
        )

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
                "entropy_threshold_used": self.entropy_threshold,
                "t1_weight_used": self.t1_weight,
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
        entropy_threshold: float = ENTROPY_HIGH_THRESHOLD,
    ) -> str:
        parts = []
        if EscalationTrigger.UNCERTAINTY_EXCEEDED in triggers:
            parts.append(
                f"불확실성 점수({unc.nonconformity_score:.3f}) > "
                f"임계값({rtc_config.adjusted_threshold:.3f})"
            )
        if EscalationTrigger.HIGH_RISK_ACTION in triggers:
            parts.append("고위험 임상 행동 감지 (EOL 결정 또는 불확실 처치 언급)")
        if EscalationTrigger.NO_EVIDENCE in triggers:
            parts.append("근거 부재 표현 감지")
        if not math.isnan(entropy) and entropy > entropy_threshold:
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