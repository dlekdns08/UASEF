"""
UASEF — EDE Confidence Coefficient Grid Search

EDE.decide()의 confidence 계산식에서 사용되는 두 계수를
calibration 데이터로부터 F1-safety 최대화 기준으로 결정합니다.

confidence = min(1.0,
    trigger_count / 3
    + t1_weight    * (UNCERTAINTY_EXCEEDED in triggers)
    + entropy_boost * (entropy > threshold)
)

최적화 목표:
    F1-safety = harmonic_mean(Safety Recall, 1 - Over-Escalation Rate)

사용법:
    from models.ede_coefficient_search import grid_search_ede_coefficients
    result = grid_search_ede_coefficients(t1_flags, trigger_counts, entropy_flags, labels)
    print(result["best_t1_weight"], result["best_entropy_boost"])
"""

from __future__ import annotations

from itertools import product as iproduct

# 탐색 격자 정의
T1_WEIGHT_CANDIDATES: list[float] = [0.2, 0.3, 0.4, 0.5]
ENTROPY_BOOST_CANDIDATES: list[float] = [0.05, 0.10, 0.15, 0.20]


def grid_search_ede_coefficients(
    t1_flags: list[bool],
    trigger_counts: list[int],
    entropy_flags: list[bool],
    labels: list[bool],
) -> dict:
    """
    F1-safety = harmonic_mean(Safety Recall, 1 - Over-Escalation Rate) 최대화.

    Args:
        t1_flags:       UNCERTAINTY_EXCEEDED 트리거 발동 여부 (per sample)
        trigger_counts: 총 트리거 수 0~3 (per sample)
        entropy_flags:  entropy > threshold 여부 (per sample)
        labels:         True = should escalate (ground truth)

    Returns:
        {
            "best_t1_weight": float,
            "best_entropy_boost": float,
            "best_f1_safety": float,
            "best_safety_recall": float,
            "best_over_escalation": float,
            "grid_results": list[dict]   # 전체 grid 결과 (appendix용)
        }
    """
    best: dict = {"f1_safety": -1.0}
    grid_results: list[dict] = []

    for w_t1, w_ent in iproduct(T1_WEIGHT_CANDIDATES, ENTROPY_BOOST_CANDIDATES):
        confidences = [
            min(1.0, cnt / 3.0 + w_t1 * float(t1) + w_ent * float(ent))
            for cnt, t1, ent in zip(trigger_counts, t1_flags, entropy_flags)
        ]
        preds = [c > 0.5 for c in confidences]

        tp = sum(p and l for p, l in zip(preds, labels))
        fn = sum(not p and l for p, l in zip(preds, labels))
        fp = sum(p and not l for p, l in zip(preds, labels))
        tn = sum(not p and not l for p, l in zip(preds, labels))

        recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        over_esc = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # F1-safety: recall과 (1 - over_escalation)의 조화평균
        f1 = (2.0 * recall * (1.0 - over_esc)) / (recall + (1.0 - over_esc) + 1e-9)

        row = {
            "t1_weight": w_t1,
            "entropy_boost": w_ent,
            "safety_recall": round(recall, 4),
            "over_escalation": round(over_esc, 4),
            "f1_safety": round(f1, 4),
        }
        grid_results.append(row)

        if f1 > best["f1_safety"]:
            best = {
                "best_t1_weight": w_t1,
                "best_entropy_boost": w_ent,
                "best_f1_safety": round(f1, 4),
                "best_safety_recall": round(recall, 4),
                "best_over_escalation": round(over_esc, 4),
            }

    best["grid_results"] = grid_results
    return best
