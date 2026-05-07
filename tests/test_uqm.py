"""UQM core tests (audit 6.10).

audit 6 / 6.9 / 6.10에서 도입된 동작들을 회귀 방지로 잠근다.
"""
from __future__ import annotations

import os
import warnings

import pytest

from models.uqm import (
    UQM,
    ScoringMethod,
    _answer_diversity,
    _answer_mode_entropy,
    _collect_sc_samples,
    compute_self_consistency_score,
    compute_hybrid_score,
)
from models.model_interface import (
    backend_supports_logprobs,
    LOGPROB_INCOMPATIBLE_BACKENDS,
    LOGPROB_INCOMPATIBLE_MODEL_PATTERNS,
)


# ── audit 6.9: backend_supports_logprobs ──────────────────────────────────────


@pytest.mark.parametrize("backend,model,expected", [
    ("openai", "gpt-4o", True),
    ("openai", "gpt-4o", True),
    ("openai", "gpt-4.1", True),
    ("openai", "o1-preview", False),
    ("openai", "o1-mini", False),
    ("openai", "o3-mini", False),
    ("openai", "o4-mini", False),
    ("openai", "gpt-5", False),
    ("openai", "gpt-5-mini", False),
    ("lmstudio", None, True),
    ("mlx", None, True),
    ("anthropic", None, False),
    ("gemini", None, False),
])
def test_backend_supports_logprobs(backend, model, expected):
    assert backend_supports_logprobs(backend, model) is expected


def test_logprob_incompatible_constants():
    assert "anthropic" in LOGPROB_INCOMPATIBLE_BACKENDS
    assert "gemini" in LOGPROB_INCOMPATIBLE_BACKENDS
    # 패턴이 정규식 문자열이어야 함
    assert all(isinstance(p, str) for p in LOGPROB_INCOMPATIBLE_MODEL_PATTERNS)


# ── audit 6.9: hybrid 컴포넌트 ────────────────────────────────────────────────


def test_answer_diversity_same_text():
    assert _answer_diversity(["aspirin"] * 5) == 0.0


def test_answer_diversity_completely_different():
    div = _answer_diversity(["A", "B", "C", "D", "E"])
    assert 0.99 < div <= 1.0


def test_answer_mode_entropy_same():
    assert _answer_mode_entropy(["x"] * 5) == 0.0


def test_answer_mode_entropy_max():
    H = _answer_mode_entropy(["a", "b", "c", "d", "e"])
    assert abs(H - 1.0) < 1e-9


def test_answer_mode_entropy_bimodal():
    H = _answer_mode_entropy(["A", "A", "A", "B", "B"])  # 3:2
    assert 0.3 < H < 0.6   # 약 0.42


# ── audit 6.10: 모델 사전 점검 + fallback 통일 ───────────────────────────────


def test_strict_anthropic_logprob_raises():
    with pytest.raises(RuntimeError, match="anthropic"):
        UQM(backend="anthropic", alpha=0.10, scoring_method="logprob", strict=True)


def test_nonstrict_anthropic_auto_switch_to_hybrid():
    """audit 6.10: anthropic도 hybrid로 전환 (기존 SC에서 변경)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        uqm = UQM(backend="anthropic", alpha=0.10, scoring_method="logprob", strict=False)
    assert uqm.active_scoring_method == "hybrid"
    assert any("자동 전환" in str(m.message) for m in w)


def test_strict_openai_reasoning_model_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "o3-mini")
    with pytest.raises(RuntimeError, match="o3-mini"):
        UQM(backend="openai", alpha=0.10, scoring_method="logprob", strict=True)


def test_nonstrict_openai_reasoning_auto_switch(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")
    uqm = UQM(backend="openai", alpha=0.10, scoring_method="logprob", strict=False)
    assert uqm.active_scoring_method == "hybrid"


def test_hybrid_directly():
    uqm = UQM(backend="openai", alpha=0.10, scoring_method="hybrid")
    assert uqm.active_scoring_method == "hybrid"


# ── audit 6.10: hybrid weights 인자 ──────────────────────────────────────────


def test_uqm_accepts_hybrid_weights():
    uqm = UQM(
        backend="openai", alpha=0.10, scoring_method="hybrid",
        hybrid_diversity_weight=0.3, hybrid_entropy_weight=0.7,
    )
    assert uqm.hybrid_diversity_weight == 0.3
    assert uqm.hybrid_entropy_weight == 0.7


# ── audit 6 issue #19: strict mode for n<min_n ─────────────────────────────────


def test_strict_n_below_min_raises():
    """α=0.05 → n_min=19. n=10이면 strict=True에서 RuntimeError."""
    from models.uqm import ConformalCalibrator
    cal = ConformalCalibrator(alpha=0.05, strict=True)
    with pytest.raises(RuntimeError, match="최소값"):
        cal.fit([0.1] * 10)


def test_nonstrict_n_below_min_warns():
    from models.uqm import ConformalCalibrator
    cal = ConformalCalibrator(alpha=0.05, strict=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal.fit([0.1] * 10)
    assert any("최소값" in str(m.message) for m in w)
