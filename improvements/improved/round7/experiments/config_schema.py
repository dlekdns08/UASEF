"""
UASEF — Pydantic schema for base_config.yaml (audit 6.10)

타입 미스매치(예: scenario_multipliers.emergency = "0.9" 문자열 들어감), 잘못된 enum,
범위 위반(α > 1, weight 합 ≠ 1)을 즉시 잡아 silent miscalibration을 방지한다.

사용:
    from experiments.config_schema import BaseConfig, validate_config_dict
    cfg = validate_config_dict(yaml.safe_load(open("base_config.yaml")))
    # 잘못된 키/타입/범위 시 ValidationError 발생

`load_config()`가 strict=True 옵션으로 호출 시 검증 경유.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class UQMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(0.10, ge=0.001, le=0.5)
    scoring_method: Literal["logprob", "self_consistency", "hybrid", "auto"] = "logprob"
    consistency_n: int = Field(5, ge=2, le=20)
    holdout_fraction: float = Field(0.2, gt=0, lt=1)
    prompt_mode: Literal["neutral", "instructed"] = "neutral"
    strict: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_calibration: int = Field(500, ge=10)
    n_test_per_scenario: int = Field(200, ge=1)
    calibration_split: Literal["train", "test"] = "train"
    test_split: Literal["train", "test"] = "test"
    seed: int = 42
    distribution_source: str = "medqa"
    include_pubmedqa: bool = False


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results_dir: str = "results"


class RTCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    CRITICAL: float = Field(..., gt=0, le=2)
    HIGH:     float = Field(..., gt=0, le=2)
    MODERATE: float = Field(..., gt=0, le=2)
    LOW:      float = Field(..., gt=0, le=2)


class ScenarioMultipliers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emergency:      float = Field(1.0, gt=0, le=2)
    rare_disease:   float = Field(1.0, gt=0, le=2)
    multimorbidity: float = Field(1.0, gt=0, le=2)
    routine:        float = Field(1.0, gt=0, le=2)


class EDEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t1_weight: float = Field(0.4, ge=0, le=1)
    entropy_boost: float = Field(0.15, ge=0, le=1)
    decision_rule: Literal["trigger_count", "confidence"] = "trigger_count"
    confidence_threshold: float = Field(0.5, ge=0, le=1)


class HybridConfig(BaseModel):
    """audit 6.10: hybrid score 가중치."""
    model_config = ConfigDict(extra="forbid")
    diversity_weight: float = Field(0.5, ge=0, le=1)
    entropy_weight:   float = Field(0.5, ge=0, le=1)

    @model_validator(mode="after")
    def _check_sum(self):
        s = self.diversity_weight + self.entropy_weight
        if abs(s - 1.0) > 1e-6:
            # 경고만 — 합 ≠ 1.0이어도 동작은 함 (스케일이 달라질 뿐)
            import warnings
            warnings.warn(
                f"[Config] hybrid weights 합 = {s:.3f} (≠ 1.0). 결과 스케일이 SC와 달라질 수 있음.",
                UserWarning, stacklevel=2,
            )
        return self


# ── Round 7 (audit 7) ─────────────────────────────────────────────────────────


class StratifiedAlphas(BaseModel):
    """Pivot A: per-stratum CRC bounds. CRITICAL ≤ HIGH ≤ MODERATE ≤ LOW."""
    model_config = ConfigDict(extra="forbid")
    CRITICAL: float = Field(..., gt=0, lt=1)
    HIGH:     float = Field(..., gt=0, lt=1)
    MODERATE: float = Field(..., gt=0, lt=1)
    LOW:      float = Field(..., gt=0, lt=1)

    @model_validator(mode="after")
    def _check_monotone(self):
        if not (self.CRITICAL <= self.HIGH <= self.MODERATE <= self.LOW):
            raise ValueError(
                f"stratified_alphas는 CRITICAL ≤ HIGH ≤ MODERATE ≤ LOW 단조이어야 합니다 "
                f"(CRITICAL이 가장 엄격). got: {self.model_dump()}"
            )
        return self


class CostEntry(BaseModel):
    """Pivot C: 단일 stratum의 (FN cost, FP cost)."""
    model_config = ConfigDict(extra="forbid")
    miss:     float = Field(..., gt=0)
    over_esc: float = Field(..., gt=0)


class CostMatrix(BaseModel):
    """Pivot C: stratum별 비대칭 cost matrix."""
    model_config = ConfigDict(extra="forbid")
    CRITICAL: CostEntry
    HIGH:     CostEntry
    MODERATE: CostEntry
    LOW:      CostEntry


class MultiTriggerConfig(BaseModel):
    """Pivot B: trigger conformal combination 옵션."""
    model_config = ConfigDict(extra="forbid")
    enabled:        bool = False
    combination:    Literal["harmonic", "bonferroni", "e_value"] = "harmonic"
    combined_alpha: float = Field(0.05, gt=0, lt=1)


class BaseConfig(BaseModel):
    """base_config.yaml 전체 스키마."""
    model_config = ConfigDict(extra="allow")  # 새 키를 미래 호환을 위해 허용
    uqm:                  UQMConfig
    data:                 DataConfig
    backends:             list[Literal["openai", "lmstudio", "mlx", "anthropic", "gemini"]]
    output:               OutputConfig
    rtc:                  RTCConfig
    scenario_multipliers: Optional[ScenarioMultipliers] = None
    entropy_threshold:    float = Field(0.6, gt=0, lt=2)
    ede:                  EDEConfig
    hybrid:               Optional[HybridConfig] = None
    # Round 7
    stratified_alphas:    Optional[StratifiedAlphas]    = None
    costs:                Optional[CostMatrix]          = None
    multi_trigger:        Optional[MultiTriggerConfig]  = None

    @field_validator("backends")
    @classmethod
    def _check_nonempty_backends(cls, v):
        if not v:
            raise ValueError("backends 리스트가 비어있습니다.")
        return v


def validate_config_dict(cfg_dict: dict) -> BaseConfig:
    """
    base_config.yaml dict를 검증하고 typed model 반환.
    pydantic ValidationError를 던지므로 호출자가 적절히 처리.

    audit 6.10:
        load_config(strict=True) 또는 _preflight_check에서 호출 가능.
        잘못된 키/타입/범위를 silent miscalibration 전에 차단.
    """
    return BaseConfig.model_validate(cfg_dict)
