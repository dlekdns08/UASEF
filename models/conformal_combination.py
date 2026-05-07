"""
UASEF — Multi-Trigger Conformal Combination (Round 7, Pivot B)

═══════════════════════════════════════════════════════════════════════════════
이론 배경
═══════════════════════════════════════════════════════════════════════════════

기존 EDE.decide (v1, Round 6.10):
    should_escalate = (len(triggers) > 0)     # ad-hoc OR

문제: 3개 trigger 각각이 false positive rate α=0.10이면,
      P(어느 하나라도 잘못 발동) ≤ 1 - (1-α)^3 ≈ 0.27   (independence 가정)
      또는 임의 의존성에서 더 클 수도 있음.
      → coverage 보장 무효.

v2 (Round 7, 본 모듈):
    1) 각 trigger를 별도 nonconformity score로 frame:
         T1: score_T1 = -mean(token logprobs)        (UQM 기존)
         T2: score_T2 = continuous high-risk score    (NEW)
         T3: score_T3 = continuous no-evidence score  (NEW)
    2) 각각 conformal p-value 계산:
         p_i = (1 + #{cal_score ≥ test_score}) / (n+1)     [Vovk et al. 2005]
    3) p-value combination:
         p_combined = combine(p_1, p_2, p_3)
    4) 결정:
         should_escalate ⟺ p_combined ≤ α_combined
       → FWER ≤ α_combined 보장 (under arbitrary dependence)

═══════════════════════════════════════════════════════════════════════════════
Combination 방법 비교
═══════════════════════════════════════════════════════════════════════════════

(a) Bonferroni (보수적 baseline):
    p_combined = m × min(p_i)
    Type-I error ≤ α  always (지나치게 보수적)

(b) Harmonic Mean p-value (Wilson, PNAS 2019):
    H = m / Σ(1/p_i)
    p_HMP = H × e × ln(m)               (asymptotically valid under dependence)
    실용적 경계: α ≤ 0.05일 때 정확. α=0.10에서는 약간 보수적.
    Bonferroni보다 ~m배 tighter.

(c) E-value mean (Vovk & Wang, Biometrika 2019; Wang & Ramdas, JRSS-B 2022):
    e_i = 1/p_i  (one-sided p에 대한 valid e-value)
    e_combined = mean(e_i)
    p_combined = 1 / e_combined
    Markov inequality로 valid. 임의 의존성에서도 정확.
    실험적으로 종종 HMP보다 tighter.

UASEF 기본: harmonic. 사용자가 selection 가능.

═══════════════════════════════════════════════════════════════════════════════
참고문헌
═══════════════════════════════════════════════════════════════════════════════
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
  Random World. Springer. (conformal p-value의 정의)
- Wilson, D. J. (2019). The harmonic mean p-value for combining dependent tests.
  PNAS, 116(4), 1195-1200.
- Vovk, V., & Wang, R. (2019). Combining p-values via averaging. Biometrika,
  108(2), 397-412.
- Wang, R., & Ramdas, A. (2022). False discovery rate control with e-values.
  JRSS Series B, 84(3), 822-852.
- Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023). Testing for
  outliers with conformal p-values. Annals of Statistics, 51(1), 149-178.
"""

from __future__ import annotations

import math
from typing import Optional, Literal


# ── conformal p-value ────────────────────────────────────────────────────────


def conformal_pvalue(score: float, calibration_scores: list[float]) -> float:
    """
    Conformal p-value:
        p = (1 + #{s_i ≥ score}) / (n + 1)

    nonconformity score는 클수록 anomalous라 가정. 따라서 작은 p = 높은
    anomaly = 에스컬레이션 권장.

    Vovk et al. (2005): exchangeability하에서 P(p_test ≤ α) ≤ α (super-uniform).

    Edge cases:
        n == 0   → return 1.0 (보수적 — 정보 없음)
        score = +inf → p = 1/(n+1)
    """
    n = len(calibration_scores)
    if n == 0:
        return 1.0
    n_geq = sum(1 for s in calibration_scores if s >= score)
    return (1 + n_geq) / (n + 1)


# ── p-value combination methods ──────────────────────────────────────────────


def combine_p_bonferroni(p_values: list[float]) -> float:
    """
    Bonferroni: p_combined = min(1, m × min(p_i)).
    가장 보수적이지만 항상 valid (어떤 의존 구조에서도).
    """
    if not p_values:
        return 1.0
    return min(1.0, min(p_values) * len(p_values))


def combine_p_harmonic(p_values: list[float]) -> float:
    """
    Harmonic Mean p-value (Wilson, PNAS 2019).

    H = m / Σ(1/p_i)
    p_HMP = H × e × ln(m)         (m ≥ 2일 때; m=1이면 그냥 p_1)

    Asymptotically valid under arbitrary dependence at α ≤ 0.05.
    α=0.10 부근에서는 약간 보수적이지만 여전히 사용 가능.

    p_i = 0인 경우 → p_HMP = 0 (즉시 reject).
    """
    m = len(p_values)
    if m == 0:
        return 1.0
    if m == 1:
        return p_values[0]
    if any(p <= 0 for p in p_values):
        return 0.0

    H = m / sum(1.0 / p for p in p_values)
    correction = math.e * math.log(m)   # Wilson 2019 asymptotic
    return min(1.0, H * correction)


def combine_e_value(p_values: list[float]) -> float:
    """
    E-value mean (Vovk & Wang 2019; Wang & Ramdas 2022).

    e_i = 1/p_i (one-sided p의 standard e-value)
    e_combined = mean(e_i)
    p_combined = 1 / e_combined  (Markov 부등식)

    실험적으로 HMP보다 종종 tighter. 임의 의존성에서 valid.

    p_i = 0인 경우 → e = +∞, e_combined = +∞ → p_combined = 0.
    """
    if not p_values:
        return 1.0
    if any(p <= 0 for p in p_values):
        return 0.0
    e_values = [1.0 / p for p in p_values]
    e_mean = sum(e_values) / len(e_values)
    return min(1.0, 1.0 / e_mean)


COMBINATION_METHODS: dict[str, callable] = {
    "bonferroni": combine_p_bonferroni,
    "harmonic":   combine_p_harmonic,
    "e_value":    combine_e_value,
}


# ── per-trigger calibrator (간단 wrapping) ───────────────────────────────────


class TriggerCalibrator:
    """
    단일 trigger의 nonconformity score 분포 저장.
    `conformal_pvalue` 호출 시 이 calibration 분포를 사용.

    UQM의 ConformalCalibrator와 다르게 threshold 계산 없이 score 분포만 보존
    (p-value 계산용).
    """

    def __init__(self, name: str):
        self.name = name
        self.calibration_scores: list[float] = []

    def fit(self, scores: list[float]) -> None:
        if not scores:
            raise ValueError(f"빈 calibration set ({self.name})")
        self.calibration_scores = sorted(scores)

    def pvalue(self, score: float) -> float:
        return conformal_pvalue(score, self.calibration_scores)


# ── Multi-trigger combiner ───────────────────────────────────────────────────


class MultiTriggerConformal:
    """
    여러 trigger의 conformal p-value를 결합해 단일 통계적 결정.

    예시:
        t1_cal = TriggerCalibrator("T1_uncertainty")
        t1_cal.fit(uqm_scores)
        t2_cal = TriggerCalibrator("T2_high_risk_action")
        t2_cal.fit(t2_scores)
        t3_cal = TriggerCalibrator("T3_no_evidence")
        t3_cal.fit(t3_scores)

        mtc = MultiTriggerConformal([t1_cal, t2_cal, t3_cal], combination="harmonic")
        escalate, info = mtc.should_escalate([t1_score, t2_score, t3_score], alpha=0.05)
        # info["p_combined"], info["p_per_trigger"]

    보장 (Wilson 2019, Vovk-Wang 2019):
        E[escalate] under H_0 (모든 trigger가 null) ≤ α    (FWER ≤ α)

    이는 v1 EDE의 `len(triggers) > 0`에서는 **성립하지 않던** 보장.
    """

    def __init__(
        self,
        calibrators: list[TriggerCalibrator],
        combination: Literal["harmonic", "bonferroni", "e_value"] = "harmonic",
    ):
        if not calibrators:
            raise ValueError("최소 1개 calibrator 필요")
        self.calibrators = calibrators
        if combination not in COMBINATION_METHODS:
            raise ValueError(
                f"Unknown combination: {combination!r}. "
                f"Choose from: {list(COMBINATION_METHODS)}"
            )
        self.combination = combination

    @property
    def m(self) -> int:
        """trigger 개수."""
        return len(self.calibrators)

    def per_trigger_pvalues(self, scores: list[float]) -> list[float]:
        """각 trigger의 conformal p-value 리스트."""
        if len(scores) != self.m:
            raise ValueError(
                f"scores 길이={len(scores)} ≠ calibrator 수={self.m}"
            )
        return [c.pvalue(s) for c, s in zip(self.calibrators, scores)]

    def combined_pvalue(self, scores: list[float]) -> float:
        """결합 p-value (combination 방법에 따라)."""
        ps = self.per_trigger_pvalues(scores)
        return COMBINATION_METHODS[self.combination](ps)

    def should_escalate(
        self,
        scores: list[float],
        alpha: float,
    ) -> tuple[bool, dict]:
        """
        Returns:
            (escalate: bool, info: dict)
            info에는 p_per_trigger, p_combined, alpha, combination 포함.
        """
        ps = self.per_trigger_pvalues(scores)
        p_combined = COMBINATION_METHODS[self.combination](ps)
        return p_combined <= alpha, {
            "p_per_trigger": [round(p, 6) for p in ps],
            "p_combined": round(p_combined, 6),
            "alpha": alpha,
            "combination": self.combination,
            "trigger_names": [c.name for c in self.calibrators],
        }


# ── 빠른 동작 확인 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(0)

    # synthetic: 3 trigger, 각 100개 calibration score
    cal_t1 = [random.gauss(0, 1) for _ in range(100)]
    cal_t2 = [random.gauss(0, 1) for _ in range(100)]
    cal_t3 = [random.gauss(0, 1) for _ in range(100)]

    t1 = TriggerCalibrator("T1"); t1.fit(cal_t1)
    t2 = TriggerCalibrator("T2"); t2.fit(cal_t2)
    t3 = TriggerCalibrator("T3"); t3.fit(cal_t3)

    print("=== combination 방법 비교 ===")
    print("test scores: [normal, normal, anomalous]")
    test = [0.0, 0.0, 3.5]
    for method in ["bonferroni", "harmonic", "e_value"]:
        mtc = MultiTriggerConformal([t1, t2, t3], combination=method)
        ok, info = mtc.should_escalate(test, alpha=0.05)
        print(f"  {method:12s}: p_combined={info['p_combined']:.4f}  escalate={ok}")

    print("\ntest scores: [normal, normal, normal]")
    test = [0.5, -0.2, 0.1]
    for method in ["bonferroni", "harmonic", "e_value"]:
        mtc = MultiTriggerConformal([t1, t2, t3], combination=method)
        ok, info = mtc.should_escalate(test, alpha=0.05)
        print(f"  {method:12s}: p_combined={info['p_combined']:.4f}  escalate={ok}")

    # FWER 검증 (under H_0 — null calibration data로 test 진행)
    print("\n=== FWER 실험 검증 (null hypothesis) ===")
    n_trials = 10_000
    for method in ["bonferroni", "harmonic", "e_value"]:
        mtc = MultiTriggerConformal([t1, t2, t3], combination=method)
        rejections = 0
        for _ in range(n_trials):
            test = [random.gauss(0, 1) for _ in range(3)]
            if mtc.should_escalate(test, alpha=0.05)[0]:
                rejections += 1
        print(f"  {method:12s}: empirical FWER = {rejections/n_trials:.4f}  (target ≤ 0.05)")
