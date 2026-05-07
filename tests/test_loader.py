"""data/loader tests (audit #3 fallback gate, stable hash)."""
from __future__ import annotations

import os

import pytest

from data.loader import (
    _stable_id,
    _distribution_source_for,
    case_to_experiment_dict,
    case_to_agent_dict,
    MedQACase,
    ALLOW_FALLBACK_ENV,
)


# ── audit ID hashing: stable across processes ────────────────────────────────


def test_stable_id_deterministic():
    """Python builtin hash()는 PYTHONHASHSEED 영향 → md5로 안정 ID."""
    a = _stable_id("X", "hello world")
    b = _stable_id("X", "hello world")
    assert a == b


def test_stable_id_different_for_different_text():
    assert _stable_id("X", "a") != _stable_id("X", "b")


# ── audit #13: distribution_source 변형별 세분화 ─────────────────────────────


def test_distribution_source_medqa():
    case = MedQACase(question="q", options={}, answer_idx="A", answer="x", source="medqa_hf")
    assert _distribution_source_for(case) == "medqa"


def test_distribution_source_medabstain_variants():
    for variant in ("AP", "NAP", "A", "NA"):
        case = MedQACase(question="q", options={}, answer_idx="A", answer="x",
                         source=f"medabstain_{variant}")
        assert _distribution_source_for(case) == f"medabstain_{variant}"


def test_distribution_source_pubmedqa_mimic():
    for src, expected in [("pubmedqa", "pubmedqa"), ("mimic3", "mimic3")]:
        case = MedQACase(question="q", options={}, answer_idx="A", answer="x", source=src)
        assert _distribution_source_for(case) == expected


# ── case_to_*_dict ID는 안정적 ───────────────────────────────────────────────


def test_case_to_experiment_dict_stable_id():
    case = MedQACase(question="What is aspirin?", options={}, answer_idx="A",
                     answer="x", source="medqa")
    d1 = case_to_experiment_dict(case)
    d2 = case_to_experiment_dict(case)
    assert d1["id"] == d2["id"]
    assert d1["distribution_source"] == "medqa"


def test_case_to_agent_dict_distribution():
    case = MedQACase(question="q", options={}, answer_idx="A", answer="x",
                     source="medabstain_AP", expected_escalate=True)
    d = case_to_agent_dict(case)
    assert d["distribution_source"] == "medabstain_AP"
    assert d["expected_escalate"] is True


# ── audit #3: fallback gate 환경변수 ──────────────────────────────────────────


def test_allow_fallback_env_constant():
    assert ALLOW_FALLBACK_ENV == "UASEF_ALLOW_FALLBACK"
