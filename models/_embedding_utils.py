"""
UASEF — Embedding utilities for WeightedCP & semantic entropy (audit 6.10)

sentence-transformers 미설치 환경에서도 import 안전. 호출 함수가 None을 반환하면
호출자가 fallback을 사용해야 한다.

설치:
    pip install sentence-transformers

활성화:
    export UASEF_WCP_EMBEDDING=1   # WeightedCP에 embedding 사용
    export UASEF_EMBEDDING_MODEL=all-MiniLM-L6-v2   # 기본 (가벼움)
    # 또는 의료 도메인 특화: pritamdeka/S-PubMedBert-MS-MARCO

audit 6.10:
  - WeightedConformalCalibrator._compute_weights에서 사용
  - compute_semantic_entropy에서 사용 (응답 클러스터링)
"""

from __future__ import annotations

import math
import os
from typing import Optional

# 모델 캐싱 (반복 로드 방지)
_model_cache = {}


def get_embedding_model():
    """
    sentence-transformers 모델 로드 (캐시).
    미설치 시 None 반환 — 호출자가 fallback 처리.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return None

    name = os.environ.get("UASEF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    if name not in _model_cache:
        try:
            _model_cache[name] = SentenceTransformer(name)
        except Exception:
            return None
    return _model_cache[name]


def cosine_similarities(query: "list[float]", corpus: "list[list[float]]") -> list[float]:
    """
    L2-정규화 가정. dot-product = cosine. numpy 의존성 회피용 순수 Python 구현.
    """
    sims = []
    for c in corpus:
        s = sum(q_i * c_i for q_i, c_i in zip(query, c))
        sims.append(max(-1.0, min(1.0, float(s))))
    return sims


def compute_semantic_entropy(
    texts: list[str],
    similarity_threshold: float = 0.85,
) -> Optional[float]:
    """
    Kuhn et al. (2023) "Semantic Entropy" 의 단순 구현.

    같은 의미를 지닌 응답을 같은 클러스터로 묶고, 클러스터 분포의 Shannon entropy를
    [0,1] 정규화. _answer_mode_entropy(정확 매칭)는 표현이 다르면 모두 다른 모드로
    분류하지만, semantic entropy는 의미가 같으면 같은 모드로 묶는다.

    Args:
        texts:                 N개의 응답 텍스트
        similarity_threshold:  embedding 코사인 유사도가 이 값 이상이면 같은 클러스터

    Returns:
        Shannon entropy / log(N) ∈ [0,1]
        embedding 모델 미사용 가능 시 None.

    audit 6.10:
        UQM에서 ScoringMethod.HYBRID_SEMANTIC으로 노출 가능 (현재는 helper만).
    """
    if len(texts) < 2:
        return 0.0
    model = get_embedding_model()
    if model is None:
        return None

    embs = model.encode(texts, normalize_embeddings=True)

    # Greedy clustering: 새 텍스트가 기존 어느 클러스터의 대표 임베딩과
    # similarity_threshold 이상이면 그 클러스터에 합류, 아니면 새 클러스터 생성.
    clusters: list[list[int]] = []
    representatives: list[list[float]] = []
    for i, emb in enumerate(embs):
        emb_list = list(emb)
        joined = False
        for c_idx, rep in enumerate(representatives):
            sim = sum(a * b for a, b in zip(emb_list, rep))
            if sim >= similarity_threshold:
                clusters[c_idx].append(i)
                joined = True
                break
        if not joined:
            clusters.append([i])
            representatives.append(emb_list)

    n = len(texts)
    probs = [len(c) / n for c in clusters]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    H_max = math.log(n) if n > 1 else 1.0
    return min(1.0, max(0.0, H / H_max))
