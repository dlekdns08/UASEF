"""
TECP — Token-Entropy Conformal Prediction (Xu & Lu, arXiv 2509.00461, Sept 2025).

Round 7 비교 baseline.

≡ UASEF v1과의 관계:
  UASEF v1: nonconformity = -mean(token logprobs)
  TECP:     nonconformity = sum(token-level entropy)  ≈  -sum(logprobs) of greedy token
            (entropy ≈ -log p when peaked dist; mean vs sum 차이만)

Round 7 비교에서 본 어댑터는 TECP를 정확히 재현하기보다는, 그 핵심 아이디어
(누적 token entropy + split CP)를 단일 global α로 표준화해 v1처럼 동작시킴.

Args:
  alpha: split conformal level (default 0.10)

Limitation:
  - TECP 원 논문은 sequence-level entropy 누적을 사용. 본 어댑터는 score를
    외부에서 받아 그대로 split CP의 nonconformity로 사용 → 호출자가 "TECP-style"
    score(예: cumulative -logprob)를 미리 계산했다고 가정.
  - LLM 직접 호출은 별도 코드(experiments/round7_table*.py)에서 수행.
"""
from __future__ import annotations

import math
from . import BaselineAdapter


class TECPBaseline(BaselineAdapter):
    name = "TECP"

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.threshold: float = float("inf")
        self.n: int = 0

    def fit(self, scores: list[float], labels: list[bool]) -> None:
        """
        Split CP threshold:
            q̂ = scores_sorted[ ceil((n+1)(1-α)) - 1 ]
        labels은 사용 안함 (TECP는 unsupervised CP).
        """
        if not scores:
            raise ValueError("빈 calibration set")
        self.n = len(scores)
        sorted_s = sorted(scores)
        # split CP quantile (Vovk 2005)
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
            "reference": "Xu & Lu (2025). TECP: Token-Entropy Conformal Prediction for LLMs. arXiv:2509.00461",
        }
