"""RTC + EDE tests (audit 6 issues #1, #2, #6, #15, #20)."""
from __future__ import annotations

import pytest

from models.rtc_ede import (
    RTC, EDE, RTCConfig, EscalationTrigger,
    detect_no_evidence, NO_EVIDENCE_STRONG, NO_EVIDENCE_WEAK,
    DEFAULT_SCENARIO_MULTIPLIERS, ENTROPY_HIGH_THRESHOLD,
)
from models.uqm import UncertaintyResult
from models.model_interface import ModelResponse


def _fake_unc(score=0.5, weighted=False, threshold_used=2.0):
    return UncertaintyResult(
        nonconformity_score=score,
        margin=0.0,
        confidence_entropy=0.1,
        should_escalate=False,
        threshold_used=threshold_used,
        raw_response=ModelResponse("fake", None, 0, "test", 0, 0),
        scoring_method="logprob",
        weighted_cp_used=weighted,
    )


# ── audit #1: WeightedCP threshold passthrough ────────────────────────────────


def test_rtc_config_effective_threshold_standard():
    cfg = RTCConfig(specialty="emergency_medicine", scenario_type="emergency",
                    base_threshold=2.0)
    # CRITICAL(0.60) × emergency(0.90) = 0.54 → adjusted=1.08
    assert abs(cfg.adjusted_threshold - 1.08) < 1e-6
    assert abs(cfg.multiplier_value - 0.54) < 1e-6


def test_rtc_config_effective_threshold_with_weighted():
    cfg = RTCConfig(specialty="emergency_medicine", scenario_type="emergency",
                    base_threshold=2.0)
    # weighted q̂_w=2.5, multiplier=0.54 → effective=1.35
    eff = cfg.effective_threshold(uncertainty_threshold=2.5)
    assert abs(eff - 1.35) < 1e-6


def test_ede_uses_weighted_threshold_when_weighted_cp_used():
    rtc = RTC(base_threshold=2.0)
    cfg = rtc.get_threshold("emergency_medicine", "emergency")
    # standard adjusted = 1.08, weighted_eff = 1.35 (q̂_w=2.5)
    unc = _fake_unc(score=1.20, weighted=True, threshold_used=2.5)
    ede = EDE(decision_rule="trigger_count")
    decision = ede.decide(unc, cfg, "no triggers in this answer")
    # score=1.20 > standard 1.08 → 트리거 OK
    # 그러나 weighted_eff=1.35 > 1.20 → 미트리거 (audit #1 정상 동작)
    assert not decision.should_escalate
    assert abs(decision.log["threshold"] - 1.35) < 1e-6


# ── audit #2: decision_rule ──────────────────────────────────────────────────


def test_ede_trigger_count_rule():
    cfg = RTCConfig(specialty="general_practice", scenario_type="routine",
                    base_threshold=2.0)
    unc = _fake_unc(score=0.1)  # 임계 미만 → no T1
    ede = EDE(decision_rule="trigger_count")
    decision = ede.decide(unc, cfg, "safe routine answer")
    assert not decision.should_escalate


def test_ede_confidence_rule():
    """EDE(decision_rule='confidence') — 트리거 없어도 confidence > τ면 escalate."""
    cfg = RTCConfig(specialty="general_practice", scenario_type="routine",
                    base_threshold=2.0)
    unc = _fake_unc(score=0.1)
    ede = EDE(decision_rule="confidence", confidence_threshold=0.0)  # 임의 0
    decision = ede.decide(unc, cfg, "safe answer")
    # confidence=0 > 0 false → not escalate
    assert not decision.should_escalate


# ── audit #6: NO_EVIDENCE strong/weak 2단계 ───────────────────────────────────


def test_detect_no_evidence_weak_only_no_modifier():
    """audit #6: 'may vary' 단독은 트리거 안 됨."""
    triggered, _ = detect_no_evidence("Treatment may vary by region.")
    assert not triggered


def test_detect_no_evidence_strong():
    triggered, _ = detect_no_evidence("I am not certain about this.")
    assert triggered


def test_detect_no_evidence_weak_with_modifier():
    """weak + UNCERTAINTY_MODIFIERS → 트리거."""
    triggered, _ = detect_no_evidence(
        "Treatment may vary; consider escalation if borderline."
    )
    assert triggered


def test_no_evidence_strong_weak_split():
    """audit #6: 30 strong + 14 weak."""
    assert len(NO_EVIDENCE_STRONG) >= 25
    assert len(NO_EVIDENCE_WEAK) >= 10
    assert NO_EVIDENCE_STRONG.isdisjoint(NO_EVIDENCE_WEAK)


# ── audit #7: ENTROPY_HIGH_THRESHOLD 도달 가능 ────────────────────────────────


def test_entropy_threshold_reachable():
    """audit #7: top_logprobs=5 entropy 상한 ln(5)≈1.609 미만이어야 fallback이 의미있음."""
    import math
    assert ENTROPY_HIGH_THRESHOLD < math.log(5)


# ── audit #15: code blue 강등 ─────────────────────────────────────────────────


def test_code_blue_no_longer_critical():
    """audit #15: 'code blue'는 CRITICAL → PROCEDURAL로 이동."""
    from models.rtc_ede import CRITICAL_KEYWORDS, PROCEDURAL_KEYWORDS
    assert "code blue" not in CRITICAL_KEYWORDS
    assert "code blue" in PROCEDURAL_KEYWORDS


# ── audit #20: scenario_multipliers data-driven ──────────────────────────────


def test_scenario_multipliers_default():
    assert DEFAULT_SCENARIO_MULTIPLIERS["emergency"] == 0.90
    assert DEFAULT_SCENARIO_MULTIPLIERS["rare_disease"] == 0.90
    assert DEFAULT_SCENARIO_MULTIPLIERS["multimorbidity"] == 1.00
    assert DEFAULT_SCENARIO_MULTIPLIERS["routine"] == 1.00


def test_rtc_accepts_scenario_multipliers_override():
    rtc = RTC(base_threshold=2.0, scenario_multipliers={"emergency": 0.5})
    cfg = rtc.get_threshold("emergency_medicine", "emergency")
    # CRITICAL(0.60) × emergency(0.50) = 0.30 → adjusted=0.60
    assert abs(cfg.adjusted_threshold - 0.60) < 1e-6
