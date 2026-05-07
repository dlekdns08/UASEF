"""Stratified CRC tests (Round 7 Pivot A)."""
from __future__ import annotations

import math
import random

import pytest

from models.stratified_crc import (
    StratifiedConformalRiskControl,
    missed_escalation_loss,
    DEFAULT_ALPHAS, DEFAULT_STRATA,
    min_n_for_alpha,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _synthetic_4strata(n_per: int, seed: int = 0):
    """4 stratum × n_per. positive label은 score 평균 2, negative 0."""
    random.seed(seed)
    scores, labels, strata = [], [], []
    for stratum, base_rate in [
        ("CRITICAL", 0.30), ("HIGH", 0.20),
        ("MODERATE", 0.10), ("LOW", 0.05),
    ]:
        for _ in range(n_per):
            l = random.random() < base_rate
            s = random.gauss(2.0 if l else 0.0, 1.0)
            scores.append(s); labels.append(l); strata.append(stratum)
    return scores, labels, strata


# ── basic loss + helpers ─────────────────────────────────────────────────────


def test_missed_escalation_loss_correct_cases():
    # label=True, score ≤ λ → loss=1 (miss)
    assert missed_escalation_loss(lam=1.0, score=0.5, label=True) == 1.0
    # label=True, score > λ → loss=0 (correctly escalated)
    assert missed_escalation_loss(lam=1.0, score=2.0, label=True) == 0.0
    # label=False → loss=0 always
    assert missed_escalation_loss(lam=1.0, score=0.5, label=False) == 0.0


def test_loss_monotone_in_lambda():
    """ℓ(λ, score, label)가 λ에 대해 monotone non-decreasing — CRC validity 핵심."""
    score, label = 1.0, True
    losses = [missed_escalation_loss(lam, score, label) for lam in [0.5, 0.9, 1.0, 1.1, 2.0]]
    # λ가 score를 넘어가는 시점에 loss가 1로 jump
    for i in range(len(losses) - 1):
        assert losses[i] <= losses[i + 1]


def test_min_n_for_alpha():
    assert min_n_for_alpha(0.10) == 20      # max(20, 9)
    assert min_n_for_alpha(0.05) == 20      # max(20, 19)
    assert min_n_for_alpha(0.01) == 99
    assert min_n_for_alpha(0.001) == 999


# ── core CRC behavior ────────────────────────────────────────────────────────


def test_crc_fit_returns_per_stratum_lambdas():
    scores, labels, strata = _synthetic_4strata(n_per=200)
    crc = StratifiedConformalRiskControl(
        alphas={"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    )
    report = crc.fit(scores, labels, strata)
    assert set(report.per_stratum) == {"CRITICAL", "HIGH", "MODERATE", "LOW"}
    for st, info in report.per_stratum.items():
        assert info.is_data_sufficient
        assert info.empirical_risk_at_lambda <= info.alpha + 0.01    # 약간 slack


def test_crc_threshold_for_returns_per_stratum_lambda():
    scores, labels, strata = _synthetic_4strata(n_per=200)
    crc = StratifiedConformalRiskControl(
        alphas={"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    )
    crc.fit(scores, labels, strata)
    crit_lam = crc.threshold_for("CRITICAL")
    low_lam = crc.threshold_for("LOW")
    # 더 엄격한 stratum (CRITICAL)이 작은 λ → 더 많이 escalate
    # (label True 비율이 더 높은 분포에서 R̂가 더 빨리 α 초과 → 더 작은 λ에서 멈춤)
    assert crit_lam <= low_lam


def test_crc_holdout_coverage_validates():
    scores, labels, strata = _synthetic_4strata(n_per=300, seed=1)
    crc = StratifiedConformalRiskControl(
        alphas={"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    )
    crc.fit(scores, labels, strata)
    # 같은 분포로 holdout
    h_scores, h_labels, h_strata = _synthetic_4strata(n_per=300, seed=2)
    cov = crc.coverage_check(h_scores, h_labels, h_strata, slack=0.30)
    for st, c in cov.items():
        assert c["ok"] is True, f"{st}: R={c['empirical_risk']} > α={c['target_alpha']}"


# ── strict mode + edge cases ─────────────────────────────────────────────────


def test_strict_raises_when_n_too_small():
    crc = StratifiedConformalRiskControl(
        alphas={"CRITICAL": 0.001},  # min_n=999
        strict=True,
    )
    with pytest.raises(RuntimeError, match="min_n"):
        crc.fit(scores=[1.0]*50, labels=[True]*50, strata=["CRITICAL"]*50)


def test_nonstrict_warns_and_uses_conservative_lambda():
    import warnings
    crc = StratifiedConformalRiskControl(
        alphas={"CRITICAL": 0.001},
        strict=False,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        crc.fit(scores=[1.0]*50, labels=[True]*50, strata=["CRITICAL"]*50)
    assert any("min_n" in str(m.message) for m in w)
    assert not crc.report.per_stratum["CRITICAL"].is_data_sufficient


def test_empty_input_raises():
    crc = StratifiedConformalRiskControl(alphas={"CRITICAL": 0.10})
    with pytest.raises(ValueError, match="빈 calibration"):
        crc.fit(scores=[], labels=[], strata=[])


def test_length_mismatch_raises():
    crc = StratifiedConformalRiskControl(alphas={"CRITICAL": 0.10})
    with pytest.raises(ValueError, match="length mismatch"):
        crc.fit(scores=[1.0, 2.0], labels=[True], strata=["CRITICAL"])


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alphas"):
        StratifiedConformalRiskControl(alphas={"X": 1.5})


def test_threshold_for_unknown_stratum_returns_zero():
    """없는 stratum 요청 시 0.0 (가장 보수적 — 모든 것 escalate)."""
    crc = StratifiedConformalRiskControl(alphas={"CRITICAL": 0.10})
    crc.fit([1.0]*30, [True]*15 + [False]*15, ["CRITICAL"]*30)
    assert crc.threshold_for("NONEXISTENT") == 0.0


def test_loss_upper_bound_validation():
    with pytest.raises(ValueError, match="loss_upper_bound"):
        StratifiedConformalRiskControl(alphas=DEFAULT_ALPHAS, loss_upper_bound=0)
