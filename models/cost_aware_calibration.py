"""
UASEF — Cost-Aware Threshold Optimization (Round 7, Pivot C)

═══════════════════════════════════════════════════════════════════════════════
이론 배경
═══════════════════════════════════════════════════════════════════════════════

기존 (v1, Round 6.10 — `models/rtc_calibration.py`):
    F1-safety = harmonic_mean(Safety Recall, 1 - Over-Escalation Rate)
    → 가정: FN과 FP의 cost가 동등 (대칭)

임상 현실:
    CRITICAL stratum (응급)에서 missed escalation = 환자 사망 → cost ≈ ∞
    LOW stratum (일반외래)에서 over-escalation = 임상의 시간 30분
    비대칭 비율: CRITICAL FN : FP ≈ 1000:1, LOW FN : FP ≈ 1:1

v2 (Round 7, 본 모듈):
    Cost(λ, stratum) = c_FN(s) × FN_count(λ, s) + c_FP × FP_count(λ, s)
    λ_s* = argmin_λ Cost(λ, s)
           s.t. FN_rate(λ, s) ≤ α_s    (Conformal Risk Control 제약 — Pivot A)

세 가지 효과:
  1) 임상 의미의 cost matrix 명시화 (sensitivity analysis 가능)
  2) Per-stratum optimization (Pivot A의 stratification과 정합)
  3) Symmetric F1보다 reviewer-friendly (JAMA류가 즉각 이해)

═══════════════════════════════════════════════════════════════════════════════
참고문헌
═══════════════════════════════════════════════════════════════════════════════
- El-Yaniv, R., & Wiener, Y. (2010). On the Foundations of Noise-free
  Selective Classification. JMLR, 11, 1605-1641.
- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep
  Neural Networks. NeurIPS 2017.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional


# ── default cost matrix (임상 비용 비율 추정) ────────────────────────────────
#
# 정당화:
#   CRITICAL: 응급의학과 missed STEMI/sepsis ≈ 사망 (배상금 / DALY 손실 ~$M)
#             vs over-escalation ≈ 응급의 30분 ($50). ratio ~1000:1.
#   LOW:      일반외래 missed routine 감기 → 큰 위험 없음
#             vs over-escalation = 일반의 시간 5분 ≈ 동등.
#   MODERATE/HIGH: 보간.
#
# Sensitivity analysis: paper에서 ratio 100/500/1000 모두 sweep해 robustness 입증.

DEFAULT_COST_MATRIX: dict[str, dict[str, float]] = {
    "CRITICAL": {"miss": 1000.0, "over_esc": 1.0},
    "HIGH":     {"miss":  100.0, "over_esc": 1.0},
    "MODERATE": {"miss":   10.0, "over_esc": 1.0},
    "LOW":      {"miss":    1.0, "over_esc": 1.0},
}


# ── core utilities ──────────────────────────────────────────────────────────


def confusion_at_threshold(
    scores: list[float],
    labels: list[bool],
    threshold: float,
) -> dict:
    """
    Score > threshold이면 escalate. 실제 라벨과 비교한 confusion matrix.
    """
    tp = fp = tn = fn = 0
    for s, l in zip(scores, labels):
        pred = s > threshold
        if pred and l:
            tp += 1
        elif pred and not l:
            fp += 1
        elif (not pred) and (not l):
            tn += 1
        else:  # not pred and l
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n": tp + fp + tn + fn,
            "n_pos": tp + fn, "n_neg": fp + tn}


def cost_weighted_loss(
    scores: list[float],
    labels: list[bool],
    threshold: float,
    cost_miss: float,
    cost_over_esc: float,
) -> float:
    """
    Loss(λ) = c_miss × FN(λ) + c_over × FP(λ)
    """
    cm = confusion_at_threshold(scores, labels, threshold)
    return cost_miss * cm["fn"] + cost_over_esc * cm["fp"]


# ── per-stratum cost-aware threshold search ──────────────────────────────────


@dataclass
class ThresholdResult:
    """단일 stratum의 cost-aware optimization 결과."""
    threshold: float
    cost: float
    miss_rate: float
    over_esc_rate: float
    constraint_violated: bool      # risk_constraint 충족 여부
    fallback_used: bool             # 충족 후보 없어서 가장 보수적 선택
    n_candidates: int
    sweep: list[dict] = field(default_factory=list)   # 전체 결과 (논문 appendix)


def find_cost_optimal_threshold(
    scores: list[float],
    labels: list[bool],
    cost_miss: float,
    cost_over_esc: float,
    risk_constraint: Optional[float] = None,
) -> ThresholdResult:
    """
    단일 stratum에서 cost-weighted minimum threshold 탐색.

    Args:
        scores, labels:     calibration data
        cost_miss:          c_FN
        cost_over_esc:      c_FP
        risk_constraint:    α_s — empirical miss rate (FN/n_pos)의 상한.
                            None이면 제약 없이 cost만 최소화.

    Algorithm:
        candidates = sorted(unique(scores)) ∪ {min(scores)-ε, max(scores)+ε}
        for each thr:
            compute cost, miss_rate
            if risk_constraint이 None or miss_rate ≤ risk_constraint:
                feasible.append((thr, cost))
        if feasible:
            return argmin cost
        else:
            fallback: argmin miss_rate (가장 보수적, 제약 위반 명시)

    Returns:
        ThresholdResult.
    """
    if not scores:
        return ThresholdResult(
            threshold=0.0, cost=float("inf"),
            miss_rate=0.0, over_esc_rate=0.0,
            constraint_violated=True, fallback_used=True,
            n_candidates=0,
        )

    # 후보: unique scores ± epsilon (양 끝 외삽)
    eps = 1e-6
    uniq = sorted(set(scores))
    candidates = [uniq[0] - eps] + uniq + [uniq[-1] + eps]

    sweep_rows: list[dict] = []
    feasible: list[dict] = []
    for thr in candidates:
        cm = confusion_at_threshold(scores, labels, thr)
        cost = cost_miss * cm["fn"] + cost_over_esc * cm["fp"]
        miss_rate = cm["fn"] / cm["n_pos"] if cm["n_pos"] > 0 else 0.0
        over_rate = cm["fp"] / cm["n_neg"] if cm["n_neg"] > 0 else 0.0
        row = {
            "threshold": float(thr),
            "cost": float(cost),
            "miss_rate": round(miss_rate, 4),
            "over_esc_rate": round(over_rate, 4),
        }
        sweep_rows.append(row)
        if risk_constraint is None or miss_rate <= risk_constraint:
            feasible.append(row)

    if feasible:
        best = min(feasible, key=lambda r: r["cost"])
        return ThresholdResult(
            threshold=best["threshold"], cost=best["cost"],
            miss_rate=best["miss_rate"], over_esc_rate=best["over_esc_rate"],
            constraint_violated=False, fallback_used=False,
            n_candidates=len(candidates), sweep=sweep_rows,
        )

    # fallback: 가장 보수적 (miss_rate 최소)
    best = min(sweep_rows, key=lambda r: r["miss_rate"])
    if risk_constraint is not None:
        warnings.warn(
            f"[CostAware] risk_constraint={risk_constraint} 충족 후보 없음. "
            f"가장 보수적 threshold로 fallback (miss_rate={best['miss_rate']}). "
            f"데이터를 늘리거나 constraint를 완화하세요.",
            UserWarning, stacklevel=2,
        )
    return ThresholdResult(
        threshold=best["threshold"], cost=best["cost"],
        miss_rate=best["miss_rate"], over_esc_rate=best["over_esc_rate"],
        constraint_violated=True, fallback_used=True,
        n_candidates=len(candidates), sweep=sweep_rows,
    )


def sweep_cost_aware_per_stratum(
    scores_by_stratum: dict[str, list[float]],
    labels_by_stratum: dict[str, list[bool]],
    cost_matrix: Optional[dict[str, dict[str, float]]] = None,
    alpha_constraints: Optional[dict[str, float]] = None,
) -> dict[str, ThresholdResult]:
    """
    Pivot A의 stratification + Pivot C의 cost-aware threshold 결합.

    각 stratum 독립적으로 최적 threshold 산출. cost_matrix가 stratum별로
    다른 비대칭을 인코딩하므로 결과 threshold도 stratum별로 다르게 나옴.

    Args:
        scores_by_stratum:   {"CRITICAL": [...], "HIGH": [...], ...}
        labels_by_stratum:   같은 키.
        cost_matrix:         {stratum: {"miss": float, "over_esc": float}}.
                              None이면 DEFAULT_COST_MATRIX.
        alpha_constraints:   {stratum: α_s} risk constraint per stratum.
                              None이면 제약 없음 (순수 cost 최소화).

    Returns:
        {stratum: ThresholdResult}.
    """
    cm = cost_matrix or DEFAULT_COST_MATRIX
    out: dict[str, ThresholdResult] = {}
    for stratum in cm:
        scores = scores_by_stratum.get(stratum, [])
        labels = labels_by_stratum.get(stratum, [])
        out[stratum] = find_cost_optimal_threshold(
            scores=scores, labels=labels,
            cost_miss=cm[stratum]["miss"],
            cost_over_esc=cm[stratum]["over_esc"],
            risk_constraint=(alpha_constraints or {}).get(stratum),
        )
    return out


# ── sensitivity analysis 유틸 (논문 Table 3 직접 사용) ──────────────────────


def cost_ratio_sweep(
    scores: list[float],
    labels: list[bool],
    miss_costs: list[float] = (10.0, 100.0, 1000.0),
    over_cost: float = 1.0,
    risk_constraint: Optional[float] = None,
) -> list[dict]:
    """
    동일 데이터에 대해 miss:over_esc cost ratio를 sweep.

    paper에서 cost matrix의 임의성 우려에 대한 답변 — robustness 입증.

    Returns:
        [{cost_ratio, threshold, miss_rate, over_esc_rate}, ...]
    """
    out = []
    for c_miss in miss_costs:
        result = find_cost_optimal_threshold(
            scores, labels, c_miss, over_cost, risk_constraint
        )
        out.append({
            "cost_ratio": f"{c_miss:.0f}:{over_cost:.0f}",
            "threshold": round(result.threshold, 4),
            "cost": round(result.cost, 2),
            "miss_rate": result.miss_rate,
            "over_esc_rate": result.over_esc_rate,
            "constraint_violated": result.constraint_violated,
        })
    return out


# ── 빠른 동작 확인 ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import random
    random.seed(0)

    # 합성: positive(양성, 라벨=True)는 score 높고, negative는 낮음
    scores, labels = [], []
    for _ in range(500):
        l = random.random() < 0.3
        s = random.gauss(2.0 if l else 0.0, 1.0)
        scores.append(s); labels.append(l)

    print("=== Cost ratio sweep (single stratum) ===")
    rows = cost_ratio_sweep(scores, labels)
    for r in rows:
        print(f"  ratio={r['cost_ratio']:>8s}  threshold={r['threshold']:>7.3f}  "
              f"miss={r['miss_rate']:.3f}  over={r['over_esc_rate']:.3f}")

    print("\n=== With CRC constraint α=0.05 ===")
    rows = cost_ratio_sweep(scores, labels, risk_constraint=0.05)
    for r in rows:
        viol = "✗" if r["constraint_violated"] else "✓"
        print(f"  ratio={r['cost_ratio']:>8s}  threshold={r['threshold']:>7.3f}  "
              f"miss={r['miss_rate']:.3f}  over={r['over_esc_rate']:.3f}  {viol}")

    print("\n=== Per-stratum sweep (synthetic 4 strata) ===")
    sb, lb = {}, {}
    for stratum, base in [("CRITICAL", 0.30), ("HIGH", 0.20),
                          ("MODERATE", 0.10), ("LOW", 0.05)]:
        s_scores, s_labels = [], []
        for _ in range(200):
            l = random.random() < base
            s = random.gauss(2.0 if l else 0.0, 1.0)
            s_scores.append(s); s_labels.append(l)
        sb[stratum] = s_scores; lb[stratum] = s_labels

    results = sweep_cost_aware_per_stratum(
        sb, lb,
        alpha_constraints={"CRITICAL": 0.05, "HIGH": 0.10,
                           "MODERATE": 0.15, "LOW": 0.20},
    )
    for s, r in results.items():
        viol = "✗" if r.constraint_violated else "✓"
        print(f"  {s:9s}: thr={r.threshold:>7.3f}  cost={r.cost:>8.1f}  "
              f"miss={r.miss_rate:.3f}  over={r.over_esc_rate:.3f}  {viol}")
