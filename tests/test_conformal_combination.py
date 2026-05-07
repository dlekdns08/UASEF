"""Multi-Trigger Conformal Combination tests (Round 7 Pivot B)."""
from __future__ import annotations

import math
import random

import pytest

from models.conformal_combination import (
    conformal_pvalue,
    combine_p_bonferroni, combine_p_harmonic, combine_e_value,
    COMBINATION_METHODS,
    TriggerCalibrator, MultiTriggerConformal,
)


# ── conformal p-value primitive ──────────────────────────────────────────────


def test_conformal_pvalue_extreme_score():
    """가장 큰 score는 가장 작은 p (1/(n+1))."""
    cal = list(range(10))   # 0..9
    p = conformal_pvalue(score=100.0, calibration_scores=cal)
    assert p == 1 / (10 + 1)


def test_conformal_pvalue_smallest_score():
    """가장 작은 score는 가장 큰 p ((n+1)/(n+1) = 1)."""
    cal = list(range(10))
    p = conformal_pvalue(score=-100.0, calibration_scores=cal)
    assert p == (10 + 1) / (10 + 1)
    assert p == 1.0


def test_conformal_pvalue_empty_cal_returns_one():
    assert conformal_pvalue(0.5, []) == 1.0


def test_conformal_pvalue_super_uniform():
    """exchangeability하에서 P(p_test ≤ α) ≤ α + 1/(n+1)."""
    random.seed(0)
    cal = [random.gauss(0, 1) for _ in range(200)]
    n_trials = 5000
    rejections_alpha_05 = 0
    for _ in range(n_trials):
        test = random.gauss(0, 1)   # same dist
        p = conformal_pvalue(test, cal)
        if p <= 0.05:
            rejections_alpha_05 += 1
    rate = rejections_alpha_05 / n_trials
    # 이론: ≤ 0.05 + 1/201 ≈ 0.055
    assert rate <= 0.06, f"Type I error rate {rate} > 0.06"


# ── combination methods ─────────────────────────────────────────────────────


def test_bonferroni_basic():
    assert combine_p_bonferroni([0.01, 0.5, 0.5]) == 0.03   # 3 × 0.01
    assert combine_p_bonferroni([0.5]) == 0.5
    assert combine_p_bonferroni([]) == 1.0
    assert combine_p_bonferroni([0.5, 0.5, 0.5]) == 1.0     # capped


def test_harmonic_basic():
    p = combine_p_harmonic([0.01, 0.01, 0.01])
    # 모두 같으면 H = 0.01, p_HMP = 0.01 × e × ln(3) ≈ 0.0299
    assert 0.025 < p < 0.035


def test_harmonic_single_pvalue():
    """m=1이면 p_HMP = p_1."""
    assert combine_p_harmonic([0.42]) == 0.42


def test_harmonic_zero_returns_zero():
    assert combine_p_harmonic([0.0, 0.5, 0.5]) == 0.0


def test_evalue_basic():
    """e_i = 1/p_i, e_combined = mean. 모두 0.01이면 mean = 100, p = 0.01."""
    p = combine_e_value([0.01, 0.01, 0.01])
    assert abs(p - 0.01) < 1e-9


def test_evalue_capped_at_one():
    p = combine_e_value([0.9, 0.9, 0.9])
    assert p == 1.0


def test_combination_methods_dict():
    assert set(COMBINATION_METHODS) == {"bonferroni", "harmonic", "e_value"}


# ── TriggerCalibrator ────────────────────────────────────────────────────────


def test_trigger_calibrator_fit_and_pvalue():
    cal = TriggerCalibrator("T1")
    cal.fit([0.0, 1.0, 2.0, 3.0, 4.0])
    assert cal.pvalue(2.5) == (1 + 2) / (5 + 1)   # 3, 4 are ≥ 2.5


def test_trigger_calibrator_empty_fit_raises():
    cal = TriggerCalibrator("T1")
    with pytest.raises(ValueError, match="빈"):
        cal.fit([])


# ── MultiTriggerConformal end-to-end ─────────────────────────────────────────


def _make_3_calibrators(seed: int = 0):
    random.seed(seed)
    cals = []
    for name in ("T1", "T2", "T3"):
        c = TriggerCalibrator(name)
        c.fit([random.gauss(0, 1) for _ in range(100)])
        cals.append(c)
    return cals


def test_mtc_construction_validates():
    with pytest.raises(ValueError, match="최소"):
        MultiTriggerConformal([], combination="harmonic")
    cals = _make_3_calibrators()
    with pytest.raises(ValueError, match="Unknown combination"):
        MultiTriggerConformal(cals, combination="bogus")


def test_mtc_score_length_mismatch_raises():
    cals = _make_3_calibrators()
    mtc = MultiTriggerConformal(cals, combination="harmonic")
    with pytest.raises(ValueError, match="길이"):
        mtc.per_trigger_pvalues([0.5, 0.5])   # 2 ≠ 3


def test_mtc_should_escalate_returns_info_dict():
    cals = _make_3_calibrators()
    mtc = MultiTriggerConformal(cals, combination="harmonic")
    escalate, info = mtc.should_escalate([3.0, 3.0, 3.0], alpha=0.05)
    assert isinstance(escalate, bool)
    assert "p_per_trigger" in info
    assert "p_combined" in info
    assert "combination" in info
    assert info["combination"] == "harmonic"
    assert len(info["p_per_trigger"]) == 3


@pytest.mark.parametrize("method", ["bonferroni", "harmonic", "e_value"])
def test_fwer_under_null(method):
    """
    Pivot B 핵심 검증: 모든 trigger가 null (calibration과 같은 분포)이면
    FWER ≤ α 충족. v1의 `len(triggers)>0` 식은 이 보장이 없다.
    """
    cals = _make_3_calibrators(seed=42)
    mtc = MultiTriggerConformal(cals, combination=method)
    n_trials = 2000
    alpha = 0.05
    rng = random.Random(123)
    rejections = sum(
        mtc.should_escalate([rng.gauss(0, 1) for _ in range(3)], alpha=alpha)[0]
        for _ in range(n_trials)
    )
    rate = rejections / n_trials
    # finite sample slack: ≤ α + 0.02
    assert rate <= alpha + 0.02, f"{method}: empirical FWER {rate} > {alpha + 0.02}"


def test_mtc_anomaly_detection():
    """모든 trigger가 anomalous이면 적어도 한 method는 reject해야 함."""
    cals = _make_3_calibrators()
    rejected_count = 0
    for method in ["bonferroni", "harmonic", "e_value"]:
        mtc = MultiTriggerConformal(cals, combination=method)
        if mtc.should_escalate([5.0, 5.0, 5.0], alpha=0.05)[0]:
            rejected_count += 1
    assert rejected_count >= 1, "extreme anomaly에서 어떤 combination도 reject 안함"
