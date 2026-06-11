"""
Round 10 R10.0 — Multi-seed infrastructure validation.

bootstrap CI / Clopper-Pearson upper / patient-level split helpers 가
통계적으로 정확한지 검증.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.metrics_utils import (
    clopper_pearson_upper,
    n_for_zero_miss_upper,
    patient_level_split,
    bootstrap_ci,
    wilson_ci,
)


# ── Clopper-Pearson exact upper ─────────────────────────────────────────────

def test_clopper_pearson_zero_misses_n300():
    """0/300 → upper ≈ 0.00995 (단측 95%) — 'rule of three' 의 정확형."""
    upper = clopper_pearson_upper(k=0, n=300, conf=0.95)
    assert abs(upper - 0.00995) < 1e-4, f"expected ~0.00995, got {upper}"


def test_clopper_pearson_zero_misses_n2995():
    """0/2995 → upper ≈ 0.001 — α=0.001 의 minimum n 이 맞음을 검증."""
    upper = clopper_pearson_upper(k=0, n=2995, conf=0.95)
    assert upper <= 0.001 + 1e-4, f"expected ≤0.001, got {upper}"


def test_clopper_pearson_zero_misses_n3000():
    """R10.1 의 target — 0/3000 → upper < 0.001."""
    upper = clopper_pearson_upper(k=0, n=3000, conf=0.95)
    assert upper < 0.001, f"expected <0.001, got {upper}"


def test_clopper_pearson_one_miss_n1000():
    """1/1000 → upper > 1/1000 이지만 합리적으로 작아야."""
    upper = clopper_pearson_upper(k=1, n=1000, conf=0.95)
    assert 0.001 < upper < 0.01, f"unexpected: {upper}"


def test_clopper_pearson_full_miss():
    """k=n → upper = 1.0."""
    assert clopper_pearson_upper(k=5, n=5, conf=0.95) == 1.0


# ── n_for_zero_miss_upper (inverse function) ────────────────────────────────

def test_n_for_alpha_0_001():
    """α=0.001 의 0 miss upper bound 를 얻으려면 n ≈ 2995."""
    n = n_for_zero_miss_upper(target_alpha=0.001, conf=0.95)
    assert 2990 <= n <= 3000


def test_n_for_alpha_0_05():
    """α=0.05 의 0 miss upper bound — n ≈ 59."""
    n = n_for_zero_miss_upper(target_alpha=0.05, conf=0.95)
    assert 55 <= n <= 65


def test_n_for_alpha_0_10():
    """α=0.10 → n ≈ 29."""
    n = n_for_zero_miss_upper(target_alpha=0.10, conf=0.95)
    assert 27 <= n <= 32


# ── Patient-level split ─────────────────────────────────────────────────────

def test_patient_level_split_no_leakage():
    """같은 환자의 여러 admission 이 cal/test 양쪽에 안 들어감을 보장."""
    # 100 환자 × 평균 2.5 admission
    items = []
    for subj in range(100):
        for adm in range(2 + (subj % 3)):
            items.append({"subject_id": subj, "hadm_id": subj * 10 + adm})
    cal, test = patient_level_split(items, group_of=lambda x: x["subject_id"],
                                     cal_frac=0.8, seed=42)
    cal_subj = {x["subject_id"] for x in cal}
    test_subj = {x["subject_id"] for x in test}
    assert cal_subj.isdisjoint(test_subj), \
        f"leakage detected: {cal_subj & test_subj}"
    assert len(cal_subj) + len(test_subj) == 100


def test_patient_level_split_ratio():
    """대략 cal_frac 비율 유지."""
    items = [{"subject_id": i} for i in range(1000)]
    cal, test = patient_level_split(items, group_of=lambda x: x["subject_id"],
                                     cal_frac=0.8, seed=42)
    cal_ratio = len(cal) / len(items)
    assert 0.75 < cal_ratio < 0.85


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def test_bootstrap_ci_basic():
    """간단한 mean 추정의 CI 가 합리적 범위."""
    import random
    rng = random.Random(42)
    samples = [rng.gauss(0.5, 0.1) for _ in range(200)]
    ci = bootstrap_ci(samples, statistic_fn=lambda s: sum(s) / len(s),
                       n_iter=1000, confidence=0.95, seed=42)
    assert ci is not None
    lo, hi = ci
    assert lo < 0.50 < hi, f"CI {ci} should contain mean 0.5"
    assert hi - lo < 0.10, f"CI too wide: {ci}"


def test_bootstrap_ci_small_n():
    """n < 5 면 None 반환."""
    assert bootstrap_ci([1.0, 2.0], statistic_fn=lambda s: sum(s) / len(s)) is None


# ── Wilson CI sanity ────────────────────────────────────────────────────────

def test_wilson_ci_50_percent():
    """p=0.5, n=100 → 약 [0.40, 0.60]."""
    lo, hi = wilson_ci(0.5, 100)
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61


def test_wilson_ci_zero():
    """p=0 → lo=0, hi 가 1/n 근처."""
    lo, hi = wilson_ci(0.0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.05
