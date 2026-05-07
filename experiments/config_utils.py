"""
UASEF — 실험 공통 Config 유틸리티

base_config.yaml의 캘리브레이션 결과(rtc, entropy_threshold, ede)를 로드하여
RTC/EDE 생성자에 전달할 인자를 반환합니다.

사용법:
    from experiments.config_utils import load_calibration_config, make_ede_kwargs

    rtc_multipliers, ede_kwargs = load_calibration_config()
    rtc = RTC(base_threshold=q_hat, multipliers=rtc_multipliers)
    ede = EDE(**ede_kwargs)
"""

from __future__ import annotations

from pathlib import Path
import yaml

_BASE_CONFIG_PATH = Path(__file__).parent / "configs" / "base_config.yaml"


def load_config(config_path: Path = _BASE_CONFIG_PATH) -> dict:
    """base_config.yaml 전체를 dict로 반환합니다."""
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def load_calibration_config(config_path: Path = _BASE_CONFIG_PATH) -> tuple[dict | None, dict]:
    """
    base_config.yaml에서 캘리브레이션 결과를 읽어 반환합니다.

    Returns:
        rtc_multipliers: {"CRITICAL": 0.60, ...} 또는 None (미산출 시)
        ede_kwargs:      EDE() 생성 인자
                         {t1_weight, entropy_boost, entropy_threshold,
                          decision_rule, confidence_threshold}

    audit (2026-05-07):
        - entropy_threshold fallback: 2.0 → 0.6 (top_logprobs=5 도달 가능 범위, issue #7)
        - decision_rule, confidence_threshold 신규 (issue #2)
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None, {
            "t1_weight": 0.4,
            "entropy_boost": 0.15,
            "entropy_threshold": 0.6,
            "decision_rule": "trigger_count",
            "confidence_threshold": 0.5,
        }

    rtc_multipliers = cfg.get("rtc") or None
    ede_cfg = cfg.get("ede") or {}
    entropy_threshold = cfg.get("entropy_threshold", 0.6)

    ede_kwargs = {
        "t1_weight": float(ede_cfg.get("t1_weight", 0.4)),
        "entropy_boost": float(ede_cfg.get("entropy_boost", 0.15)),
        "entropy_threshold": float(entropy_threshold),
        "decision_rule": str(ede_cfg.get("decision_rule", "trigger_count")),
        "confidence_threshold": float(ede_cfg.get("confidence_threshold", 0.5)),
    }

    return rtc_multipliers, ede_kwargs


def load_scenario_multipliers(config_path: Path = _BASE_CONFIG_PATH) -> dict[str, float] | None:
    """audit issue #20: 시나리오별 추가 배율을 base_config에서 로드 (없으면 None)."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    return cfg.get("scenario_multipliers") or None


def load_hybrid_weights(config_path: Path = _BASE_CONFIG_PATH) -> tuple[float, float]:
    """
    audit 6.10: hybrid scoring의 (diversity_weight, entropy_weight)를 로드.
    base_config.yaml의 `hybrid` 섹션 또는 fallback 0.5/0.5.
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return (0.5, 0.5)
    h = cfg.get("hybrid") or {}
    return (
        float(h.get("diversity_weight", 0.5)),
        float(h.get("entropy_weight", 0.5)),
    )


# ── Round 7 (audit 7) loaders ────────────────────────────────────────────────


def load_stratified_alphas(config_path: Path = _BASE_CONFIG_PATH) -> dict[str, float] | None:
    """audit 7 Pivot A: per-stratum CRC alphas. None이면 default 사용."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    sa = cfg.get("stratified_alphas")
    if not sa:
        return None
    return {k: float(v) for k, v in sa.items()}


def load_cost_matrix(config_path: Path = _BASE_CONFIG_PATH) -> dict[str, dict[str, float]] | None:
    """audit 7 Pivot C: per-stratum 비대칭 cost matrix. None이면 default 사용."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    cm = cfg.get("costs")
    if not cm:
        return None
    return {
        k: {"miss": float(v["miss"]), "over_esc": float(v["over_esc"])}
        for k, v in cm.items()
    }


def load_multi_trigger_config(config_path: Path = _BASE_CONFIG_PATH) -> dict | None:
    """
    audit 7 Pivot B: multi_trigger 섹션 로드.
    Returns: {"enabled", "combination", "combined_alpha"} 또는 None.
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    mt = cfg.get("multi_trigger")
    if not mt:
        return None
    return {
        "enabled":        bool(mt.get("enabled", False)),
        "combination":    str(mt.get("combination", "harmonic")),
        "combined_alpha": float(mt.get("combined_alpha", 0.05)),
    }
