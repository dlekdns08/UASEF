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
        ede_kwargs:      {"t1_weight": 0.4, "entropy_boost": 0.15, "entropy_threshold": 2.0}
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None, {"t1_weight": 0.4, "entropy_boost": 0.15, "entropy_threshold": 2.0}

    rtc_multipliers = cfg.get("rtc") or None
    ede_cfg = cfg.get("ede") or {}
    entropy_threshold = cfg.get("entropy_threshold", 2.0)

    ede_kwargs = {
        "t1_weight": float(ede_cfg.get("t1_weight", 0.4)),
        "entropy_boost": float(ede_cfg.get("entropy_boost", 0.15)),
        "entropy_threshold": float(entropy_threshold),
    }

    return rtc_multipliers, ede_kwargs
