"""
UASEF — 실험 공통 메트릭 유틸 (audit 2026-05-07 issue #16, #11)

issue #16: 단일 클래스 시나리오(emergency=positives only / routine=negatives only)에서
   safety_recall 또는 over_escalation_rate를 0.0으로 silent-zero 보고하던 문제를 수정.
   분모가 0이면 None을 반환해 "N/A"로 출력되도록 한다.

issue #11: Wilson score interval로 신뢰구간을 함께 반환해 Safety Recall=0.95 vs 0.96처럼
   통계적으로 구분 불가능한 결과를 표면화한다.
"""

from __future__ import annotations

import math
from typing import Optional


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval (95% by default).
    n=0이면 (0.0, 1.0) 반환.
    """
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    delta = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    lo = max(0.0, centre - delta)
    hi = min(1.0, centre + delta)
    return (round(lo, 4), round(hi, 4))


def safe_rate(numer: int, denom: int) -> Optional[float]:
    """
    분모가 0이면 None 반환 (silent zero 방지).
    """
    if denom <= 0:
        return None
    return round(numer / denom, 4)


def compute_binary_metrics(
    results: list[dict],
    pred_key: str = "escalated",
    label_key: str = "expected_escalate",
) -> dict:
    """
    이진 분류 메트릭 + Wilson 95% CI.

    Returns dict with keys:
        n, tp, fn, fp, tn,
        safety_recall, safety_recall_ci, safety_recall_ok,    # None if no positives
        over_escalation_rate, over_esc_ci, over_escalation_ok, # None if no negatives
        precision, f1, escalation_rate
    """
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "평가 가능한 케이스 없음"}

    tp = sum(1 for r in valid if r.get(pred_key) and r.get(label_key))
    fn = sum(1 for r in valid if not r.get(pred_key) and r.get(label_key))
    fp = sum(1 for r in valid if r.get(pred_key) and not r.get(label_key))
    tn = sum(1 for r in valid if not r.get(pred_key) and not r.get(label_key))
    total = len(valid)

    safety_recall = safe_rate(tp, tp + fn)
    over_escalation_rate = safe_rate(fp, fp + tn)
    precision = safe_rate(tp, tp + fp)
    escalation_rate = round((tp + fp) / total, 4) if total else 0.0

    f1: Optional[float]
    if precision is not None and safety_recall is not None and (precision + safety_recall) > 0:
        f1 = round(2 * precision * safety_recall / (precision + safety_recall), 4)
    else:
        f1 = None

    safety_ci = wilson_ci(safety_recall, tp + fn) if safety_recall is not None else None
    over_ci = wilson_ci(over_escalation_rate, fp + tn) if over_escalation_rate is not None else None

    return {
        "n": total,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "safety_recall": safety_recall,
        "safety_recall_ci": safety_ci,
        "safety_recall_ok": (safety_recall is not None and safety_recall >= 0.95),
        "over_escalation_rate": over_escalation_rate,
        "over_escalation_ci": over_ci,
        "over_escalation_ok": (over_escalation_rate is not None and over_escalation_rate <= 0.15),
        "precision": precision,
        "f1": f1,
        "escalation_rate": escalation_rate,
    }


def fmt_rate(value: Optional[float], decimals: int = 4) -> str:
    """N/A 처리 안전 포맷."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_ci(ci: Optional[tuple[float, float]]) -> str:
    if ci is None:
        return ""
    return f" [{ci[0]:.3f},{ci[1]:.3f}]"
