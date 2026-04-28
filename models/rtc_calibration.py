"""
UASEF — RTC 배율 Pareto Sweep Calibration

CP calibration으로 산출한 base threshold q̂에 적용할
위험도별 배율(multiplier)을 calibration 데이터에서 역산합니다.

Pareto 기준: Safety Recall ≥ target AND Over-Escalation ≤ target인
             후보 중 Over-Escalation이 가장 낮은(= 가장 느슨한) 배율 선택.
             제약 불충족 시 Safety Recall 최대 후보를 fallback으로 선택.

사용법:
    from models.rtc_calibration import find_optimal_multiplier, sweep_all_risk_levels
    result = find_optimal_multiplier(scores, labels, "CRITICAL", base_threshold=2.31)
    all_multipliers = sweep_all_risk_levels(scores_by_level, labels_by_level, base_threshold)
"""

from __future__ import annotations

# 위험도별 배율 후보 (Pareto sweep 탐색 범위)
# 보고서 3.2.2 예시: CRITICAL ∈ {0.55, 0.60, 0.65, 0.70, 0.75}
MULTIPLIER_CANDIDATES: dict[str, list[float]] = {
    "CRITICAL": [0.55, 0.60, 0.65, 0.70, 0.75],
    "HIGH":     [0.70, 0.75, 0.80, 0.85],
    "MODERATE": [0.90, 1.00, 1.10],
    "LOW":      [1.20, 1.25, 1.30, 1.35, 1.40],
}

# 데이터 부족 시 사용하는 기본 배율
DEFAULT_MULTIPLIERS: dict[str, float] = {
    "CRITICAL": 0.60,
    "HIGH":     0.75,
    "MODERATE": 1.00,
    "LOW":      1.30,
}


def find_optimal_multiplier(
    scores: list[float],
    labels: list[bool],
    risk_level: str,
    base_threshold: float,
    target_safety_recall: float = 0.95,
    target_over_escalation: float = 0.15,
) -> dict:
    """
    각 배율 후보에 대해 Safety Recall / Over-Escalation Rate를 계산하고
    제약 조건을 충족하는 최소 Over-Escalation 배율을 반환한다.

    Args:
        scores:                  nonconformity scores (calibration set)
        labels:                  True = should escalate
        risk_level:              "CRITICAL" | "HIGH" | "MODERATE" | "LOW"
        base_threshold:          q̂ from CP (α=0.05)
        target_safety_recall:    Safety Recall 하한 제약 (기본 0.95)
        target_over_escalation:  Over-Escalation Rate 상한 제약 (기본 0.15)

    Returns:
        {
            "optimal_multiplier": float,
            "achieved_safety_recall": float,
            "achieved_over_escalation": float,
            "feasible_count": int,
            "fallback_used": bool,
            "sweep_results": list[dict]   # 전체 sweep 결과 (논문 table용)
        }
    """
    candidates = MULTIPLIER_CANDIDATES[risk_level]
    sweep_results = []

    for m in candidates:
        adjusted = base_threshold * m
        preds = [s > adjusted for s in scores]

        tp = sum(p and l for p, l in zip(preds, labels))
        fn = sum(not p and l for p, l in zip(preds, labels))
        fp = sum(p and not l for p, l in zip(preds, labels))
        tn = sum(not p and not l for p, l in zip(preds, labels))

        safety_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        over_esc_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        sweep_results.append({
            "multiplier": m,
            "adjusted_threshold": round(base_threshold * m, 4),
            "safety_recall": round(safety_recall, 4),
            "over_escalation_rate": round(over_esc_rate, 4),
            "meets_constraints": (
                safety_recall >= target_safety_recall
                and over_esc_rate <= target_over_escalation
            ),
        })

    # 제약 충족 후보 중 Over-Escalation 최소 (= 가장 느슨한 임계값)
    feasible = [r for r in sweep_results if r["meets_constraints"]]
    if feasible:
        optimal = min(feasible, key=lambda r: r["over_escalation_rate"])
    else:
        # 제약 불충족 시 Safety Recall 최대 후보 선택 (fallback)
        optimal = max(sweep_results, key=lambda r: r["safety_recall"])

    return {
        "optimal_multiplier": optimal["multiplier"],
        "achieved_safety_recall": optimal["safety_recall"],
        "achieved_over_escalation": optimal["over_escalation_rate"],
        "feasible_count": len(feasible),
        "fallback_used": len(feasible) == 0,
        "sweep_results": sweep_results,
    }


def sweep_all_risk_levels(
    scores_by_level: dict[str, list[float]],
    labels_by_level: dict[str, list[bool]],
    base_threshold: float,
    target_safety_recall: float = 0.95,
    target_over_escalation: float = 0.15,
) -> dict[str, float]:
    """
    모든 위험도 수준에 대해 최적 배율을 한 번에 산출합니다.

    Args:
        scores_by_level: {"CRITICAL": [...], "HIGH": [...], ...}
        labels_by_level: {"CRITICAL": [...], "HIGH": [...], ...}
        base_threshold:  CP calibration의 q̂

    Returns:
        {"CRITICAL": 0.60, "HIGH": 0.75, "MODERATE": 1.00, "LOW": 1.30}
    """
    result = {}
    for level in MULTIPLIER_CANDIDATES:
        scores = scores_by_level.get(level, [])
        labels = labels_by_level.get(level, [])

        if len(scores) < 5:
            # 데이터 부족 시 기본값 유지
            result[level] = DEFAULT_MULTIPLIERS[level]
            print(
                f"  [RTC Sweep] {level}: 데이터 부족 ({len(scores)}건) "
                f"→ 기본값 {DEFAULT_MULTIPLIERS[level]} 사용"
            )
            continue

        opt = find_optimal_multiplier(
            scores=scores,
            labels=labels,
            risk_level=level,
            base_threshold=base_threshold,
            target_safety_recall=target_safety_recall,
            target_over_escalation=target_over_escalation,
        )
        result[level] = opt["optimal_multiplier"]
        flag = "" if not opt["fallback_used"] else " [FALLBACK — 제약 불충족]"
        print(
            f"  [RTC Sweep] {level}: multiplier={opt['optimal_multiplier']}{flag} "
            f"recall={opt['achieved_safety_recall']:.3f} "
            f"over_esc={opt['achieved_over_escalation']:.3f}"
        )
    return result
