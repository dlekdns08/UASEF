"""
UASEF — Entropy Threshold Calibration (Youden's J)

엔트로피 임계값(ENTROPY_HIGH_THRESHOLD)을 하드코딩 대신
calibration 데이터에서 Youden's J 통계량으로 자동 결정합니다.

Youden's J = Sensitivity + Specificity - 1 (최대화 지점 선택)

NaN 처리: logprobs 미지원 백엔드(self_consistency 모드)에서는 entropy=nan이 발생합니다.
         유효 샘플이 10개 미만이면 fallback threshold=2.0을 반환합니다.

사용법:
    from models.entropy_calibration import find_entropy_threshold
    result = find_entropy_threshold(entropy_values, labels)
    print(result["threshold"])   # e.g., 2.07
    print(result["youdens_j"])   # e.g., 0.61
"""

from __future__ import annotations

import numpy as np


def find_entropy_threshold(
    entropy_values: list[float],
    labels: list[bool],
    n_thresholds: int = 200,
) -> dict:
    """
    Youden's J = Sensitivity + Specificity - 1 최대화 지점을 임계값으로 선택.

    Args:
        entropy_values: 각 샘플의 평균 엔트로피 (nats/token). NaN 허용.
        labels:         True = should escalate
        n_thresholds:   ROC sweep 해상도 (기본 200)

    Returns:
        {
            "threshold": float,       # 채택된 임계값 (논문에 보고)
            "youdens_j": float,
            "sensitivity": float,
            "specificity": float,
            "n_valid": int,           # NaN 제거 후 유효 샘플 수
            "fallback_used": bool,
            "roc_data": dict          # ROC curve 재현용
        }
    """
    # NaN 필터링
    valid = [
        (e, l)
        for e, l in zip(entropy_values, labels)
        if not (isinstance(e, float) and np.isnan(e))
    ]

    if len(valid) < 10:
        return {
            "threshold": 2.0,
            "youdens_j": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "n_valid": len(valid),
            "fallback_used": True,
            "roc_data": {"fpr": [], "tpr": [], "thresholds": []},
        }

    ents, labs = zip(*valid)
    ents = list(ents)
    labs = list(labs)

    thresholds = np.linspace(min(ents), max(ents), n_thresholds)
    best: dict = {
        "threshold": 2.0,
        "youdens_j": -1.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
    }

    tpr_list: list[float] = []
    fpr_list: list[float] = []

    for t in thresholds:
        preds = [e > t for e in ents]
        tp = sum(p and l for p, l in zip(preds, labs))
        fn = sum(not p and l for p, l in zip(preds, labs))
        fp = sum(p and not l for p, l in zip(preds, labs))
        tn = sum(not p and not l for p, l in zip(preds, labs))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1.0

        tpr_list.append(sensitivity)
        fpr_list.append(1.0 - specificity)

        if j > best["youdens_j"]:
            best = {
                "threshold": float(t),
                "youdens_j": float(j),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
            }

    return {
        "threshold": round(best["threshold"], 4),
        "youdens_j": round(best["youdens_j"], 4),
        "sensitivity": round(best["sensitivity"], 4),
        "specificity": round(best["specificity"], 4),
        "n_valid": len(valid),
        "fallback_used": False,
        "roc_data": {
            "fpr": [round(x, 4) for x in fpr_list],
            "tpr": [round(x, 4) for x in tpr_list],
            "thresholds": [round(float(t), 4) for t in thresholds],
        },
    }
