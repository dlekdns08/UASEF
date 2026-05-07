"""
Conformal Language Modeling (Quach et al., ICLR 2024) — 단순화 baseline.

원 논문은 sampling 기반 dynamic stopping rule + rejection을 다룬다. Round 7
비교 목적상 핵심 아이디어인 NLL nonconformity + split CP만 단순화 재구현.

UASEF v1과 거의 동일하지만:
  - Quach: NLL = -mean(per-token logprobs)
  - v1:    동일 (UQM의 compute_nonconformity_score)

따라서 본 어댑터는 v1과 수학적으로 동등 — UASEF v1을 baseline으로 비교할 때
"Quach 2024 family"의 reference로 사용 가능.

Round 7 paper 표 4에서:
  Quach 2024 = UASEF v1 (single global α, NLL nonconformity)
  UASEF v2 (Round 7) = stratified CRC + MTC + cost-aware

Limitation:
  - Quach 원 논문의 dynamic sampling stopping rule은 미구현
  - Sub-component conformity (semantic compare)도 미구현
"""
from __future__ import annotations

import math
from . import BaselineAdapter


class Quach2024Baseline(BaselineAdapter):
    name = "Quach2024-CLM"

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.threshold: float = float("inf")
        self.n: int = 0

    def fit(self, scores: list[float], labels: list[bool]) -> None:
        """
        Split CP, same as TECP but conceptually represents the CLM family.
        labels는 미사용.
        """
        if not scores:
            raise ValueError("빈 calibration set")
        self.n = len(scores)
        sorted_s = sorted(scores)
        rank = max(1, min(self.n, math.ceil((self.n + 1) * (1 - self.alpha))))
        self.threshold = sorted_s[rank - 1]

    def predict(self, score: float) -> bool:
        return score > self.threshold

    def info(self) -> dict:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_calibration": self.n,
            "reference": "Quach et al. (2024). Conformal Language Modeling. ICLR 2024.",
            "limitation": "Dynamic stopping rule + sub-component conformity는 미구현",
        }
