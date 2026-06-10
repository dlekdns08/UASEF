"""
Tests for Round 9 MIMIC-IV loader + PHI guard.

Real MIMIC-IV data 가 없는 CI 환경에서도 graceful skip 되도록 작성.
JSONL 이 존재할 때는 schema 검증.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import (
    MedQACase,
    _MIMIC4_DEFAULT_PATH,
    _load_mimic4_jsonl,
    load_mimic4_cases,
    load_mimic4_by_stratum,
    load_mimic4_by_specialty,
    load_dataset_for_stratification,
    SUPPORTED_DATASETS,
)
from models.model_interface import (
    PHI_GUARD_ENV,
    PHIGuardViolation,
    _phi_guard_active,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema-only tests (no data file required)
# ─────────────────────────────────────────────────────────────────────────────

def test_mimic4_in_supported_datasets():
    assert "mimic4" in SUPPORTED_DATASETS


def test_dispatcher_returns_empty_when_data_missing(tmp_path, monkeypatch):
    """When MIMIC-IV JSONL is absent, dispatcher returns []."""
    # ensure default path is missing
    if _MIMIC4_DEFAULT_PATH.exists():
        pytest.skip("MIMIC-IV data is present; skipping graceful-skip test")
    cases = load_dataset_for_stratification("mimic4", n=10, verbose=False)
    assert cases == []


def test_load_mimic4_cases_raises_when_data_missing():
    if _MIMIC4_DEFAULT_PATH.exists():
        pytest.skip("MIMIC-IV data is present; skipping graceful-skip test")
    with pytest.raises(FileNotFoundError):
        load_mimic4_cases(n=10, verbose=False)


def test_load_mimic4_with_synthetic_jsonl(tmp_path):
    """Schema validation: synthetic JSONL → MedQACase objects with required fields."""
    p = tmp_path / "mimic4_cases.jsonl"
    sample = {
        "hadm_id": "20000001",
        "subject_id": "10000001",
        "stratum": "CRITICAL",            # = risk_group G (decision-time)
        "risk_group": "CRITICAL",
        "expected_escalate": True,        # = Y (future outcome, independent of G)
        "y_outcome": True,
        "specialty": "cardiology",
        "admission_type": "EMERGENCY",
        "admit_year": 2015,
        "demographics": {"sex": "F", "race": "WHITE", "age_bucket": "65-79"},
        "outcome": {
            "icu_within_24h": True, "in_hospital_mortality": False,
            "deterioration_composite": True,
            "sepsis": False, "readmit_30d": False, "transfusion_24h": False,
        },
        "structured": {
            "early_lab_flags": ["lactate_high"],
            "early_vital_quartiles": [],
            "service": "CMED",
        },
        "_audit_postoutcome": {
            "icd_primary": "I50.9", "icd_codes": ["I50.9", "N18.3"],
            "lab_flags_full": ["lactate_high", "creatinine_high"], "los_days": 4.2,
        },
        "note_text": None,
    }
    with open(p, "w") as f:
        for i in range(10):
            row = dict(sample)
            row["hadm_id"] = f"2000000{i}"
            f.write(json.dumps(row) + "\n")

    cases = _load_mimic4_jsonl(p, n=10, seed=42)
    assert len(cases) == 10
    for c in cases:
        assert isinstance(c, MedQACase)
        assert c.source == "mimic4_struct"
        assert c.specialty == "cardiology"
        assert c.expected_escalate is True
        assert "Patient summary" in c.question
        assert "subject_id=" in c.meta_info
        assert "hadm_id=" in c.meta_info
        # leakage-safe: 미래/사후 정보가 prompt 에 들어가면 안 된다.
        assert "I50.9" not in c.question          # discharge ICD 제외
        assert "Length of stay" not in c.question  # los 제외
        assert "lactate_high" in c.question        # early lab 는 decision-time 가용


def test_load_mimic4_by_stratum_with_synthetic(tmp_path):
    p = tmp_path / "mimic4_cases.jsonl"
    rows = []
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        for i in range(20):
            rows.append({
                "hadm_id": f"{stratum}_{i}",
                "subject_id": f"S{i}",
                "stratum": stratum,            # = risk_group G (decision-time)
                "risk_group": stratum,
                "expected_escalate": (i % 3 == 0),  # = Y, G 와 독립적으로 변동
                "y_outcome": (i % 3 == 0),
                "specialty": "cardiology",
                "admission_type": "EMERGENCY",
                "admit_year": 2015,
                "demographics": {"sex": "F", "race": "WHITE", "age_bucket": "65-79"},
                "outcome": {"icu_within_24h": (i % 3 == 0), "in_hospital_mortality": False},
                "structured": {"early_lab_flags": [], "early_vital_quartiles": [],
                               "service": "CMED"},
                "note_text": None,
            })
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = load_mimic4_by_stratum(n_per_stratum=15, path=p, verbose=False)
    assert set(out.keys()) == {"CRITICAL", "HIGH", "MODERATE", "LOW"}
    for s, cs in out.items():
        assert 0 < len(cs) <= 15


# ─────────────────────────────────────────────────────────────────────────────
# PHI guard tests
# ─────────────────────────────────────────────────────────────────────────────

def test_phi_guard_off_by_default(monkeypatch):
    monkeypatch.delenv(PHI_GUARD_ENV, raising=False)
    assert _phi_guard_active() is False


def test_phi_guard_active_when_env_set(monkeypatch):
    monkeypatch.setenv(PHI_GUARD_ENV, "1")
    assert _phi_guard_active() is True


def test_phi_guard_blocks_external_backend_when_tainted(monkeypatch):
    """phi_taint=True 인 prompt 는 openai 백엔드로 가지 않아야 한다."""
    monkeypatch.setenv(PHI_GUARD_ENV, "1")
    from models.model_interface import query_model

    with pytest.raises(PHIGuardViolation):
        query_model(
            backend="openai",
            system_prompt="sys",
            user_prompt="raw note",
            phi_taint=True,
        )


def test_phi_guard_allows_local_backend_when_tainted(monkeypatch):
    """phi_taint=True 더라도 lmstudio (local) 는 차단되지 않아야 한다 (실제 호출 안 함)."""
    monkeypatch.setenv(PHI_GUARD_ENV, "1")
    from models.model_interface import query_model, EXTERNAL_BACKENDS, PHIGuardViolation

    # lmstudio 가 EXTERNAL_BACKENDS 에 포함되지 않음을 확인 (실제 connect 는 시도하지 않음)
    assert "lmstudio" not in EXTERNAL_BACKENDS
    # 실제 호출은 lmstudio 서버가 없으면 connection error — guard 만 검증.
    # PHIGuardViolation 이 발생하지 않아야 한다.
    try:
        query_model(
            backend="lmstudio",
            system_prompt="sys",
            user_prompt="raw note",
            phi_taint=True,
        )
    except PHIGuardViolation:
        pytest.fail("PHI guard 가 lmstudio 도 차단함 — 의도와 다름")
    except Exception:
        pass  # connection error 등은 허용 (lmstudio 미가동 시)


def test_phi_guard_inactive_when_env_not_set(monkeypatch):
    """env 미설정이면 phi_taint=True 라도 차단하지 않아야 한다."""
    monkeypatch.delenv(PHI_GUARD_ENV, raising=False)
    from models.model_interface import query_model, PHIGuardViolation

    # PHIGuardViolation 이 발생하지 않아야. 실제 OpenAI 호출은 시도되지만 그건 별개.
    try:
        query_model(
            backend="openai",
            system_prompt="sys",
            user_prompt="x",
            phi_taint=True,
            max_completion_tokens=1,
        )
    except PHIGuardViolation:
        pytest.fail("PHI guard env 미설정 시 차단되면 안 됨")
    except Exception:
        pass  # API 호출 자체의 실패는 허용 (API key 없을 수도 있음)
