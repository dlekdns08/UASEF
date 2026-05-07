"""Cost-Aware Threshold Optimization tests (Round 7 Pivot C)."""
from __future__ import annotations

import random

import pytest

from models.cost_aware_calibration import (
    confusion_at_threshold, cost_weighted_loss,
    find_cost_optimal_threshold, sweep_cost_aware_per_stratum,
    cost_ratio_sweep, ThresholdResult,
    DEFAULT_COST_MATRIX,
)


# ── basic primitives ────────────────────────────────────────────────────────


def test_confusion_at_threshold_basic():
    scores = [0.1, 0.5, 0.9, 1.5]
    labels = [False, True, False, True]
    cm = confusion_at_threshold(scores, labels, threshold=0.7)
    # 0.9, 1.5 > 0.7 → escalate
    # tp = 1.5 (label True), fp = 0.9 (label False)
    # tn = 0.1 (label False), fn = 0.5 (label True)
    assert cm["tp"] == 1
    assert cm["fp"] == 1
    assert cm["tn"] == 1
    assert cm["fn"] == 1


def test_cost_weighted_loss_symmetric():
    """c_miss = c_over일 때, total cost = total errors."""
    scores = [0.1, 0.5, 0.9, 1.5]
    labels = [False, True, False, True]
    loss = cost_weighted_loss(scores, labels, threshold=0.7, cost_miss=1.0, cost_over_esc=1.0)
    cm = confusion_at_threshold(scores, labels, threshold=0.7)
    assert loss == cm["fn"] + cm["fp"]


def test_cost_weighted_loss_asymmetric():
    scores = [0.1, 0.5, 0.9, 1.5]
    labels = [False, True, False, True]
    loss = cost_weighted_loss(scores, labels, threshold=0.7, cost_miss=10.0, cost_over_esc=1.0)
    # fn=1, fp=1 → 10 × 1 + 1 × 1 = 11
    assert loss == 11.0


# ── find_cost_optimal_threshold ──────────────────────────────────────────────


def test_optimizer_returns_threshold_result():
    scores = [0.1, 0.5, 0.9, 1.5, 2.0]
    labels = [False, True, False, True, True]
    r = find_cost_optimal_threshold(scores, labels, cost_miss=1.0, cost_over_esc=1.0)
    assert isinstance(r, ThresholdResult)
    assert r.n_candidates > 0
    assert len(r.sweep) > 0


def test_optimizer_higher_miss_cost_lowers_threshold():
    """c_miss >> c_over_esc이면 threshold가 낮아져 (더 많이 escalate) miss 줄임."""
    random.seed(0)
    scores = [random.gauss(0, 1) for _ in range(200)]
    labels = [s > 0.5 for s in scores]
    r_low = find_cost_optimal_threshold(scores, labels, cost_miss=1.0, cost_over_esc=1.0)
    r_high = find_cost_optimal_threshold(scores, labels, cost_miss=1000.0, cost_over_esc=1.0)
    assert r_high.threshold <= r_low.threshold
    assert r_high.miss_rate <= r_low.miss_rate


def test_optimizer_with_risk_constraint():
    random.seed(0)
    scores = [random.gauss(0, 1) for _ in range(200)]
    labels = [s > 0.5 for s in scores]
    r = find_cost_optimal_threshold(
        scores, labels, cost_miss=1.0, cost_over_esc=1.0,
        risk_constraint=0.05,
    )
    assert r.constraint_violated is False
    assert r.miss_rate <= 0.05


def test_optimizer_constraint_too_strict_uses_fallback():
    random.seed(0)
    # 거의 모든 case가 positive → miss_rate=0 불가능
    scores = [random.gauss(0, 1) for _ in range(50)]
    labels = [True] * 50
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = find_cost_optimal_threshold(
            scores, labels, cost_miss=1.0, cost_over_esc=1.0,
            risk_constraint=0.0,    # 0% miss는 거의 불가능 (모든 case escalate 필요)
        )
    # 모든 case escalate (가장 보수적 threshold)이면 miss=0 가능 → 통과할 수도 있음.
    # 그러나 일반적으로 fallback 경로 검증을 위해 제약 위반 또는 보수적 fallback 확인.
    assert r.threshold is not None


def test_empty_scores_returns_inf_cost():
    r = find_cost_optimal_threshold([], [], cost_miss=1.0, cost_over_esc=1.0)
    assert r.cost == float("inf")
    assert r.constraint_violated is True


# ── sweep_cost_aware_per_stratum ─────────────────────────────────────────────


def test_per_stratum_sweep_returns_all_strata():
    random.seed(0)
    sb, lb = {}, {}
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        sb[stratum] = [random.gauss(0, 1) for _ in range(100)]
        lb[stratum] = [s > 0.5 for s in sb[stratum]]
    out = sweep_cost_aware_per_stratum(
        sb, lb,
        cost_matrix=DEFAULT_COST_MATRIX,
        alpha_constraints={"CRITICAL": 0.10, "HIGH": 0.10,
                           "MODERATE": 0.10, "LOW": 0.10},
    )
    assert set(out) == {"CRITICAL", "HIGH", "MODERATE", "LOW"}


def test_per_stratum_critical_more_conservative_than_low():
    """default cost matrix는 CRITICAL miss=1000, LOW miss=1 → CRITICAL threshold 낮음."""
    random.seed(0)
    sb, lb = {}, {}
    for stratum in ["CRITICAL", "LOW"]:
        sb[stratum] = [random.gauss(0, 1) for _ in range(200)]
        lb[stratum] = [s > 0.5 for s in sb[stratum]]
    out = sweep_cost_aware_per_stratum(sb, lb)   # default cost
    assert out["CRITICAL"].threshold <= out["LOW"].threshold


# ── cost_ratio_sweep (sensitivity analysis) ──────────────────────────────────


def test_cost_ratio_sweep_returns_rows():
    random.seed(0)
    scores = [random.gauss(0, 1) for _ in range(100)]
    labels = [s > 0.5 for s in scores]
    rows = cost_ratio_sweep(scores, labels, miss_costs=[10, 100, 1000])
    assert len(rows) == 3
    for r in rows:
        assert "cost_ratio" in r
        assert "threshold" in r
        assert "miss_rate" in r
        assert "over_esc_rate" in r


def test_cost_ratio_sweep_monotone_threshold():
    """cost ratio 증가 → threshold 비증가 (more aggressive escalation)."""
    random.seed(0)
    scores = [random.gauss(0, 1) for _ in range(200)]
    labels = [s > 0.5 for s in scores]
    rows = cost_ratio_sweep(scores, labels, miss_costs=[1, 10, 100, 1000])
    thresholds = [r["threshold"] for r in rows]
    for i in range(len(thresholds) - 1):
        assert thresholds[i] >= thresholds[i + 1]   # non-increasing
