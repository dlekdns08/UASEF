"""Round 7 integration: RTC+EDE+config가 stratified CRC + MTC + cost-aware와 결합."""
from __future__ import annotations

import random

import pytest

from models.stratified_crc import StratifiedConformalRiskControl
from models.conformal_combination import (
    TriggerCalibrator, MultiTriggerConformal,
)
from models.cost_aware_calibration import sweep_cost_aware_per_stratum, DEFAULT_COST_MATRIX
from models.rtc_ede import RTC, EDE, RTCConfig
from models.uqm import UncertaintyResult
from models.model_interface import ModelResponse


def _fake_unc(score: float = 0.5, weighted: bool = False, threshold_used: float = 2.0):
    return UncertaintyResult(
        nonconformity_score=score, margin=0.0, confidence_entropy=0.1,
        should_escalate=False, threshold_used=threshold_used,
        raw_response=ModelResponse("fake answer", None, 0, "test", 0, 0),
        scoring_method="logprob", weighted_cp_used=weighted,
    )


def _fit_strat_crc():
    random.seed(0)
    scores, labels, strata = [], [], []
    for stratum, base in [("CRITICAL", 0.30), ("HIGH", 0.20),
                          ("MODERATE", 0.10), ("LOW", 0.05)]:
        for _ in range(200):
            l = random.random() < base
            s = random.gauss(2.0 if l else 0.0, 1.0)
            scores.append(s); labels.append(l); strata.append(stratum)
    crc = StratifiedConformalRiskControl(
        alphas={"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    )
    crc.fit(scores, labels, strata)
    return crc


def _fit_mtc():
    random.seed(1)
    cals = []
    for name in ("T1", "T2", "T3"):
        c = TriggerCalibrator(name)
        c.fit([random.gauss(0, 1) for _ in range(100)])
        cals.append(c)
    return MultiTriggerConformal(cals, combination="harmonic")


# ── RTC + StratifiedCRC ──────────────────────────────────────────────────────


def test_rtc_uses_stratified_calibrator_when_provided():
    """RTC(stratified_calibrator=...)가 multiplier 경로 무시하고 stratum λ 직접 반환."""
    crc = _fit_strat_crc()
    rtc = RTC(base_threshold=99.9, stratified_calibrator=crc)   # base_threshold는 ignored
    cfg = rtc.get_threshold("emergency_medicine", "emergency")
    assert cfg.adjusted_threshold == crc.threshold_for("CRITICAL")
    # multiplier_value=1.0 (이미 stratum-aware)
    assert cfg.multiplier_value == 1.0


def test_rtc_falls_back_to_v1_when_no_stratified():
    """stratified_calibrator=None이면 기존 multiplier 경로."""
    rtc = RTC(base_threshold=2.0)
    cfg = rtc.get_threshold("emergency_medicine", "emergency")
    # CRITICAL multiplier(0.60) × emergency scenario(0.90) = 0.54
    assert abs(cfg.adjusted_threshold - 1.08) < 1e-6


# ── EDE + MultiTriggerConformal ──────────────────────────────────────────────


def test_ede_conformal_combined_requires_mtc():
    with pytest.raises(ValueError, match="multi_trigger_conformal"):
        EDE(decision_rule="conformal_combined", multi_trigger_conformal=None)


def test_ede_conformal_combined_smoke():
    mtc = _fit_mtc()
    ede = EDE(
        decision_rule="conformal_combined",
        multi_trigger_conformal=mtc,
        combined_alpha=0.05,
    )
    cfg = RTCConfig(specialty="general_practice", scenario_type="routine",
                    base_threshold=2.0)
    unc = _fake_unc(score=2.5)
    decision = ede.decide(unc, cfg, "Take aspirin 81mg.")
    info = decision.log["conformal_combined_info"]
    assert "p_combined" in info
    assert "p_per_trigger" in info
    assert info["combination"] == "harmonic"
    assert len(info["p_per_trigger"]) == 3


def test_ede_t2_t3_nonconformity_scores():
    """T2/T3 nonconformity score가 [0, 1] 범위."""
    safe = "Take aspirin 81mg daily."
    uncertain = "Consider intubation if patient deteriorates. I am not certain."

    assert EDE.t2_nonconformity_score(safe) == 0.0
    assert 0 < EDE.t2_nonconformity_score(uncertain) <= 1.0
    assert EDE.t3_nonconformity_score(safe) == 0.0
    assert 0 < EDE.t3_nonconformity_score(uncertain) <= 1.0


def test_ede_t2_critical_keyword_unconditional():
    """CRITICAL_KEYWORDS는 modifier 없이도 단독 트리거."""
    text = "Patient has DNR order in chart."
    assert EDE.t2_nonconformity_score(text) > 0.0


def test_ede_t2_procedural_requires_modifier():
    """PROCEDURAL_KEYWORDS는 modifier 없으면 0."""
    no_mod = "Give epinephrine for anaphylaxis."   # 명확한 처치, 미트리거
    with_mod = "Consider epinephrine if patient deteriorates."
    assert EDE.t2_nonconformity_score(no_mod) == 0.0
    assert EDE.t2_nonconformity_score(with_mod) > 0.0


# ── End-to-end: stratified CRC + MTC 결합 ────────────────────────────────────


def test_e2e_rtc_strat_crc_with_ede_conformal_combined():
    """RTC + StratifiedCRC + EDE(conformal_combined) 전체 파이프라인."""
    crc = _fit_strat_crc()
    mtc = _fit_mtc()
    rtc = RTC(base_threshold=0.0, stratified_calibrator=crc)
    ede = EDE(
        decision_rule="conformal_combined",
        multi_trigger_conformal=mtc,
        combined_alpha=0.05,
    )

    # 두 specialty (다른 stratum) 비교
    cfg_crit = rtc.get_threshold("emergency_medicine", "emergency")    # CRITICAL
    cfg_low  = rtc.get_threshold("general_practice", "routine")        # LOW

    assert cfg_crit.adjusted_threshold == crc.threshold_for("CRITICAL")
    assert cfg_low.adjusted_threshold  == crc.threshold_for("LOW")

    # decide 호출 (둘 다 같은 응답이지만 stratum이 다름)
    unc = _fake_unc(score=2.5)
    text = "Routine answer."
    d_crit = ede.decide(unc, cfg_crit, text)
    d_low  = ede.decide(unc, cfg_low,  text)

    # 둘 다 conformal_combined info 포함
    for d in (d_crit, d_low):
        assert d.log["decision_rule"] == "conformal_combined"
        assert d.log["conformal_combined_info"] is not None


# ── cost-aware sweep + stratified 결합 (Pivot A + C) ────────────────────────


def test_cost_aware_per_stratum_with_stratified_alphas():
    """Pivot A의 alphas를 Pivot C의 risk_constraint로 직접 주입."""
    random.seed(0)
    sb, lb = {}, {}
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        sb[stratum] = [random.gauss(0, 1) for _ in range(200)]
        lb[stratum] = [s > 0.5 for s in sb[stratum]]

    alphas = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    out = sweep_cost_aware_per_stratum(
        sb, lb,
        cost_matrix=DEFAULT_COST_MATRIX,
        alpha_constraints=alphas,
    )
    # 각 stratum은 자신의 risk_constraint를 충족하거나 fallback 표시
    for stratum, r in out.items():
        if not r.constraint_violated:
            assert r.miss_rate <= alphas[stratum]
