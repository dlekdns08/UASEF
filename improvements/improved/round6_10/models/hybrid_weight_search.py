"""
UASEF — HYBRID Score Weight Grid Search (audit 6.10)

`compute_hybrid_score`의 두 가중치 (diversity_weight, entropy_weight)를 calibration
데이터에서 F1-safety 최대화 기준으로 결정합니다.

audit 6.9에서 hybrid scoring이 도입되었으나 가중치는 0.5/0.5로 하드코딩이었음.
audit 6.10: t1_weight, entropy_boost와 동일하게 데이터 기반 calibration 대상으로 승격.

탐색 격자: (div_w, ent_w) ∈ {(0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3)}
   - 두 신호가 합이 1.0인 convex combination만 탐색
   - SC-only(=1.0,0.0)와 mode-entropy-only(=0.0,1.0)도 비교 가능

사용법:
    from models.hybrid_weight_search import grid_search_hybrid_weights
    result = grid_search_hybrid_weights(diversities, mode_entropies, labels)
    print(result["best_diversity_weight"], result["best_entropy_weight"])
    # → UQM에 적용은 calibration_pipeline에서 base_config["hybrid"]에 저장 후 자동 사용
"""

from __future__ import annotations


# (div_w, ent_w) 후보. 합이 1.0인 단순 convex combination + 양 끝점.
HYBRID_WEIGHT_CANDIDATES: list[tuple[float, float]] = [
    (1.0, 0.0),    # SC-only (Jaccard diversity 단독)
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),    # audit 6.9 default
    (0.4, 0.6),
    (0.3, 0.7),
    (0.0, 1.0),    # mode-entropy-only
]


def grid_search_hybrid_weights(
    diversities: list[float],
    mode_entropies: list[float],
    labels: list[bool],
    weight_candidates: list[tuple[float, float]] | None = None,
    threshold_quantile: float = 0.85,
) -> dict:
    """
    F1-safety = harmonic_mean(Safety Recall, 1 - Over-Escalation Rate) 최대화.

    각 (div_w, ent_w) 조합에 대해:
        score_i = (div_w · diversity_i + ent_w · mode_entropy_i)
        threshold = score 분포의 threshold_quantile-분위수
        pred_i = score_i > threshold
    Safety Recall / Over-Escalation Rate를 계산하여 F1-safety 최대 조합 선택.

    Args:
        diversities:       각 샘플의 Jaccard diversity (0~1) — _answer_diversity 출력
        mode_entropies:    각 샘플의 normalized mode entropy (0~1) — _answer_mode_entropy 출력
        labels:            True = should escalate (ground truth)
        weight_candidates: 탐색 격자. None이면 HYBRID_WEIGHT_CANDIDATES.
        threshold_quantile: per-grid threshold를 설정할 score 분위수 (기본 0.85
                            = 상위 15% 케이스를 escalate로 분류 가정).

    Returns:
        {
            "best_diversity_weight": float,
            "best_entropy_weight": float,
            "best_f1_safety": float,
            "best_safety_recall": float,
            "best_over_escalation": float,
            "grid_results": list[dict]   # 전체 grid 결과 (appendix용)
        }
    """
    if not (len(diversities) == len(mode_entropies) == len(labels)):
        raise ValueError(
            f"length mismatch: diversities={len(diversities)}, "
            f"mode_entropies={len(mode_entropies)}, labels={len(labels)}"
        )
    if not diversities:
        return {
            "error": "빈 입력",
            "best_diversity_weight": 0.5,
            "best_entropy_weight": 0.5,
        }

    candidates = weight_candidates or HYBRID_WEIGHT_CANDIDATES
    grid_results: list[dict] = []
    best: dict = {"best_f1_safety": -1.0}

    n = len(diversities)
    for w_d, w_e in candidates:
        scores = [w_d * d + w_e * h for d, h in zip(diversities, mode_entropies)]
        # threshold_quantile 분위수 (sorted)
        scores_sorted = sorted(scores)
        idx = max(0, min(n - 1, int(threshold_quantile * n)))
        threshold = scores_sorted[idx]
        preds = [s > threshold for s in scores]

        tp = sum(p and l for p, l in zip(preds, labels))
        fn = sum(not p and l for p, l in zip(preds, labels))
        fp = sum(p and not l for p, l in zip(preds, labels))
        tn = sum(not p and not l for p, l in zip(preds, labels))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        over_esc = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # F1-safety: recall과 (1 - over_escalation)의 조화평균
        f1 = (2.0 * recall * (1.0 - over_esc)) / (recall + (1.0 - over_esc) + 1e-9)

        row = {
            "diversity_weight": w_d,
            "entropy_weight": w_e,
            "threshold": round(threshold, 4),
            "safety_recall": round(recall, 4),
            "over_escalation": round(over_esc, 4),
            "f1_safety": round(f1, 4),
        }
        grid_results.append(row)

        if f1 > best["best_f1_safety"]:
            best = {
                "best_diversity_weight": w_d,
                "best_entropy_weight": w_e,
                "best_threshold": round(threshold, 4),
                "best_f1_safety": round(f1, 4),
                "best_safety_recall": round(recall, 4),
                "best_over_escalation": round(over_esc, 4),
            }

    best["grid_results"] = grid_results
    return best
