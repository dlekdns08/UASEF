"""
Semantic Entropy (Farquhar et al., Nature 2024) — 단순화 baseline.

원 논문은 N개의 sampled generation을 의미적으로 클러스터링한 후 클러스터
distribution의 Shannon entropy를 hallucination signal로 사용.

본 어댑터는 score(이미 외부에서 계산된 semantic entropy)를 받아 split CP의
nonconformity로 사용. UASEF audit 6.10의 `_embedding_utils.compute_semantic_entropy`
와 결합 가능.

Round 7 비교에서 UASEF v2 hybrid mode (`UASEF_HYBRID_SEMANTIC=1`)와 같은 데이터 풀로
동작 — 차이는 v2가 stratified + MTC + cost-aware를 추가한다는 점.

Limitation:
  - Farquhar 원 논문은 GPT-3 / LLaMA 2개 모델만 평가. 본 어댑터는 score를 외부에서
    받으므로 임의 LLM 적용 가능.
  - Semantic clustering의 similarity_threshold는 외부 caller가 결정.
"""
from __future__ import annotations

import math
from . import BaselineAdapter


class SemanticEntropyBaseline(BaselineAdapter):
    name = "SemanticEntropy"

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.threshold: float = float("inf")
        self.n: int = 0

    def fit(self, scores: list[float], labels: list[bool]) -> None:
        """
        scores는 sample-level semantic entropy [0, 1] (정규화된 cluster Shannon).

        labels는 미사용. 별도로 score-vs-label AUROC를 측정하려면 caller가 처리.
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
            "reference": "Farquhar et al. (2024). Detecting hallucinations in LLMs using semantic entropy. Nature 630(8017), 625-630.",
            "limitation": "Semantic clustering similarity_threshold는 caller가 외부에서 결정",
        }
