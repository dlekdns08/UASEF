"""
UASEF — Baseline adapters for Round 7 head-to-head comparison.

각 baseline은 동일 인터페이스 (same_calibration_data → score → predict)를 따른다.
실제 LLM 호출 없이도 score 분포만 있으면 calibration / threshold 결정 가능.

지원 baseline:
  - tecp.py             — Token-Entropy Conformal Prediction (Xu & Lu 2025)
  - quach2024.py        — Conformal Language Modeling (ICLR 2024 단순화)
  - semantic_entropy.py — Semantic Entropy (Farquhar Nature 2024 단순화)

각 어댑터의 fit/predict 인터페이스는 BaselineAdapter abstract class 따름.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaselineAdapter(ABC):
    """모든 Round 7 비교 baseline이 구현해야 할 인터페이스."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, scores: list[float], labels: list[bool]) -> None:
        """Calibration: score 분포 학습."""
        ...

    @abstractmethod
    def predict(self, score: float) -> bool:
        """Test point: True = escalate."""
        ...

    @abstractmethod
    def info(self) -> dict:
        """재현성 메타데이터."""
        ...
