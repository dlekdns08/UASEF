"""Pydantic schema validation tests (audit 6.10)."""
from __future__ import annotations

import pytest

from experiments.config_schema import validate_config_dict, BaseConfig


def _valid_cfg() -> dict:
    return {
        "uqm": {
            "alpha": 0.10, "scoring_method": "logprob",
            "consistency_n": 5, "holdout_fraction": 0.2,
            "prompt_mode": "neutral", "strict": False,
        },
        "data": {
            "n_calibration": 500, "n_test_per_scenario": 200,
            "calibration_split": "train", "test_split": "test",
            "seed": 42, "distribution_source": "medqa", "include_pubmedqa": False,
        },
        "backends": ["openai", "lmstudio"],
        "output": {"results_dir": "results"},
        "rtc": {"CRITICAL": 0.6, "HIGH": 0.75, "MODERATE": 1.0, "LOW": 1.3},
        "scenario_multipliers": {
            "emergency": 0.9, "rare_disease": 0.9,
            "multimorbidity": 1.0, "routine": 1.0,
        },
        "entropy_threshold": 0.6,
        "ede": {
            "t1_weight": 0.4, "entropy_boost": 0.15,
            "decision_rule": "trigger_count", "confidence_threshold": 0.5,
        },
        "hybrid": {"diversity_weight": 0.5, "entropy_weight": 0.5},
    }


def test_valid_config_passes():
    cfg = validate_config_dict(_valid_cfg())
    assert cfg.uqm.alpha == 0.10
    assert cfg.hybrid.diversity_weight == 0.5


def test_invalid_alpha_range():
    bad = _valid_cfg()
    bad["uqm"]["alpha"] = 1.5
    with pytest.raises(Exception):  # pydantic ValidationError
        validate_config_dict(bad)


def test_invalid_scoring_method():
    bad = _valid_cfg()
    bad["uqm"]["scoring_method"] = "magic"
    with pytest.raises(Exception):
        validate_config_dict(bad)


def test_invalid_decision_rule():
    bad = _valid_cfg()
    bad["ede"]["decision_rule"] = "vote"
    with pytest.raises(Exception):
        validate_config_dict(bad)


def test_string_in_scenario_multiplier_rejected():
    """audit 6.10: 'emergency: "0.9"' (str) 같은 silent 오류 차단."""
    bad = _valid_cfg()
    bad["scenario_multipliers"]["emergency"] = "0.9"   # str
    # pydantic은 자동 coercion이 가능하면 통과 — 그러나 잘못된 값(예: "abc")은 거부
    bad["scenario_multipliers"]["emergency"] = "abc"
    with pytest.raises(Exception):
        validate_config_dict(bad)


def test_unknown_backend_rejected():
    bad = _valid_cfg()
    bad["backends"] = ["openai", "unknown_provider"]
    with pytest.raises(Exception):
        validate_config_dict(bad)


def test_empty_backends_rejected():
    bad = _valid_cfg()
    bad["backends"] = []
    with pytest.raises(Exception):
        validate_config_dict(bad)


def test_hybrid_weight_sum_warns():
    """합 ≠ 1이면 경고만 (실패는 아님)."""
    import warnings
    cfg = _valid_cfg()
    cfg["hybrid"] = {"diversity_weight": 0.3, "entropy_weight": 0.4}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_config_dict(cfg)
    assert any("hybrid" in str(m.message).lower() for m in w)
