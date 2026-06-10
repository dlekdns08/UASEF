"""
UASEF — 실험 공통 메트릭 유틸 (audit 2026-05-07 issue #16, #11)

issue #16: 단일 클래스 시나리오(emergency=positives only / routine=negatives only)에서
   safety_recall 또는 over_escalation_rate를 0.0으로 silent-zero 보고하던 문제를 수정.
   분모가 0이면 None을 반환해 "N/A"로 출력되도록 한다.

issue #11: Wilson score interval로 신뢰구간을 함께 반환해 Safety Recall=0.95 vs 0.96처럼
   통계적으로 구분 불가능한 결과를 표면화한다.

audit 6.10:
  - bootstrap_ci(): AUROC/F1 등 비-비율 통계량의 Bootstrap 95% CI
  - bonferroni_adjust(): 다중 비교 보정 (시나리오별 OK 표시)
  - results_dir(): --run-tag 환경변수 인식 (UASEF_RESULTS_DIR)
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Optional, Callable


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


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """
    단측 Clopper-Pearson(exact binomial) 상한.

    관측 miss k/n 에 대한 (1-α) 신뢰수준의 true miss rate 상한을 반환.
    k=0 의 닫힌형: upper = 1 - (1-conf)**(1/n)  ('rule of three' 의 정확형).
      예) 0/300, conf=0.95 → 0.00995.  0/2995 → ≈0.001.
    REVISION_PLAN P0-4: 0/300 에서 'α=0.001 실증' 주장을 막기 위한 정직한 상한.
    """
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    if k >= n:
        return 1.0
    # 일반 케이스: Beta 역함수 = scipy 없으면 이분탐색으로 regularized inc. beta.
    try:
        from scipy.stats import beta  # type: ignore
        return float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        # betainc 이분탐색 (scipy 미설치 fallback)
        lo, hi = k / n, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            # P(X<=k | n, mid) via regularized incomplete beta 보수적 근사
            # binomial 누적은 정규근사로 — fallback 정확도는 제한적.
            from math import erf, sqrt
            mean, var = n * mid, n * mid * (1 - mid)
            z = (k + 0.5 - mean) / sqrt(var) if var > 0 else 0.0
            cdf = 0.5 * (1 + erf(z / sqrt(2)))
            if cdf > 1 - conf:
                lo = mid
            else:
                hi = mid
        return round(hi, 6)


def n_for_zero_miss_upper(target_alpha: float, conf: float = 0.95) -> int:
    """
    0 miss 관측 시 단측 상한을 target_alpha 이하로 만드는 최소 n.
    1 - (1-conf)**(1/n) ≤ alpha  ⇒  n ≥ ln(1-conf)/ln(1-alpha).
      예) alpha=0.001, conf=0.95 → n ≈ 2995.
    """
    if not (0 < target_alpha < 1):
        raise ValueError("target_alpha must be in (0,1)")
    return math.ceil(math.log(1.0 - conf) / math.log(1.0 - target_alpha))


def patient_level_split(
    items: list,
    group_of: Callable,
    cal_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list, list]:
    """
    환자(그룹) 단위 split — 같은 환자의 여러 admission 이 cal/test 양쪽에
    들어가는 repeated-admission leakage 를 방지 (REVISION_PLAN P0-3).

    group_of(item) -> 그룹 키(예: subject_id). 그룹을 셔플해 cal_frac 비율로
    그룹 자체를 나눈 뒤, 그룹에 속한 모든 item 을 해당 분할로 보낸다.
    """
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for it in items:
        buckets[group_of(it)].append(it)
    groups = list(buckets.keys())
    random.Random(seed).shuffle(groups)
    cut = int(len(groups) * cal_frac)
    cal_groups = set(groups[:cut])
    cal, test = [], []
    for g, its in buckets.items():
        (cal if g in cal_groups else test).extend(its)
    return cal, test


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


# ── audit 6.10: Bootstrap CI for AUROC/F1/etc (Phase B2) ───────────────────────


def bootstrap_ci(
    samples: list,
    statistic_fn: Callable,
    n_iter: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Optional[tuple[float, float]]:
    """
    Percentile bootstrap 95% CI.

    Args:
        samples:        리스트 (예: [(score, label), ...])
        statistic_fn:   samples → 단일 통계량(float)
        n_iter:         리샘플링 횟수 (기본 1000)
        confidence:     신뢰수준 (기본 0.95)
        seed:           재현성

    Returns:
        (lo, hi) 또는 통계량이 None일 때 None.

    예:
        # AUROC CI
        ci = bootstrap_ci(
            samples=list(zip(scores, labels)),
            statistic_fn=lambda s: roc_auc_score([l for _,l in s], [sc for sc,_ in s]),
        )
    """
    n = len(samples)
    if n < 5:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_iter):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        try:
            v = statistic_fn(resample)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                stats.append(float(v))
        except Exception:
            continue
    if len(stats) < 10:
        return None
    stats.sort()
    alpha = (1 - confidence) / 2
    lo_idx = int(alpha * len(stats))
    hi_idx = int((1 - alpha) * len(stats)) - 1
    return (round(stats[lo_idx], 4), round(stats[hi_idx], 4))


# ── audit 6.10: Multiple-comparison correction (Phase B3) ──────────────────────


def bonferroni_adjust(p_values: list[float]) -> list[float]:
    """단순 Bonferroni: p_adj = min(1, p × m)."""
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Holm-Bonferroni step-down: 더 강력한 제어.
    Returns: 각 p-value의 reject 여부 (True = significant after correction).
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * m
    for k, (orig_idx, p) in enumerate(indexed):
        if p < alpha / (m - k):
            reject[orig_idx] = True
        else:
            break  # step-down: 한 번 멈추면 이후 모두 fail
    return reject


# ── audit 6.10: --run-tag 결과 디렉토리 (Phase A5) ────────────────────────────


def results_dir(default: Path) -> Path:
    """
    UASEF_RESULTS_DIR 환경변수가 있으면 그 경로, 없으면 default 반환.
    --run-tag로 결과를 분리할 때 모든 sub-runner의 save 함수가 사용.
    """
    custom = os.environ.get("UASEF_RESULTS_DIR")
    if custom:
        p = Path(custom)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return default
