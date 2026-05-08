"""
UASEF — Stratified Conformal Risk Control (Round 7, Pivot A)

═══════════════════════════════════════════════════════════════════════════════
이론 배경
═══════════════════════════════════════════════════════════════════════════════

Conformal Risk Control (Angelopoulos & Bates, ICLR 2024 Spotlight)는 표준 CP를
일반화하여 임의의 monotone bounded loss function의 expected risk를 제어한다.

  λ̂ = sup{ λ : R̂(λ) + B/(n+1) ≤ α }     (B = upper bound on loss)

여기서 R̂(λ) = (1/n) Σ_i ℓ(λ, x_i, y_i)는 calibration set의 empirical risk이고
ℓ는 λ에 대해 monotone (보통 non-decreasing).

UASEF의 missed-escalation loss:
    ℓ(λ, score, label) = 𝟙{label = True AND score ≤ λ}

해석: 라벨이 "에스컬레이션 필요"인데 score가 threshold 이하라 미트리거 →
λ가 클수록 더 많은 case가 threshold 이하로 떨어짐 → loss 증가 (monotone).

Stratification (Romano, Sesia, Candès, NeurIPS 2020 — class-conditional CP의 결합):
  각 stratum (CRITICAL/HIGH/MODERATE/LOW)에 대해 별도 calibration 진행 →
  stratum별 marginal 보장:  E[ℓ(λ_s, X, Y) | stratum=s] ≤ α_s

이는 단순 global conformal보다 강한 보장 — 임상 안전이 중요한 stratum에서
α를 매우 작게 설정 가능 (예: CRITICAL α=0.001).

═══════════════════════════════════════════════════════════════════════════════
v1과의 차이
═══════════════════════════════════════════════════════════════════════════════

v1 (Round 6.10):
    q̂ = quantile(scores, 1-α)                # global α=0.10
    threshold_specialty = q̂ × multiplier_s    # heuristic 0.60/0.75/1.00/1.30

v2 (Round 7, 본 모듈):
    For each stratum s:
        λ_s = sup{λ : R̂_s(λ) + 1/(n_s+1) ≤ α_s}
    threshold_specialty = λ_{stratum(specialty)}    # 보장 동반

═══════════════════════════════════════════════════════════════════════════════
참고문헌
═══════════════════════════════════════════════════════════════════════════════
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024).
  Conformal Risk Control. ICLR 2024 (Spotlight). arXiv:2208.02814
- Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with Valid and
  Adaptive Coverage. NeurIPS 2020. arXiv:2006.02544
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional


# 기본 stratum 목록 (RTC.RiskLevel과 정합).
# alphas dict는 사용자가 명시적으로 전달; 기본값은 임상 직관:
#   CRITICAL이 가장 엄격, LOW가 가장 관대.
DEFAULT_STRATA: tuple[str, ...] = ("CRITICAL", "HIGH", "MODERATE", "LOW")
DEFAULT_ALPHAS: dict[str, float] = {
    "CRITICAL": 0.001,    # 0.1% missed escalation 한계
    "HIGH":     0.010,    # 1%
    "MODERATE": 0.050,    # 5%
    "LOW":      0.100,    # 10%
}


def missed_escalation_loss(lam: float, score: float, label: bool) -> float:
    """
    0/1 missed-escalation loss.

    ℓ(λ, score, label) = 𝟙{label AND score ≤ λ}

    monotone non-decreasing in λ (essential for CRC validity).
    """
    return 1.0 if (label and score <= lam) else 0.0


@dataclass
class StratumStats:
    """단일 stratum의 calibration 결과."""
    stratum: str
    n: int
    alpha: float
    lambda_hat: float
    empirical_risk_at_lambda: float
    is_data_sufficient: bool


@dataclass
class CRCFitReport:
    """fit() 결과. 보고서 / 재현성 추적용."""
    per_stratum: dict[str, StratumStats] = field(default_factory=dict)
    n_total: int = 0


def min_n_for_alpha(alpha: float) -> int:
    """
    CRC 보장이 의미를 가지려면 n ≥ ceil((1-α)/α).
    α=0.001 → n ≥ 999, α=0.01 → 99, α=0.05 → 19, α=0.10 → 9.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    return max(20, math.ceil((1 - alpha) / alpha))


class StratifiedConformalRiskControl:
    """
    Per-stratum CRC.

    예시:
        crc = StratifiedConformalRiskControl(
            alphas={"CRITICAL": 0.001, "HIGH": 0.01, "MODERATE": 0.05, "LOW": 0.10}
        )
        crc.fit(scores=[...], labels=[...], strata=["CRITICAL", "LOW", ...])
        threshold = crc.threshold_for("CRITICAL")
        report = crc.coverage_check(holdout_scores, holdout_labels, holdout_strata)

    각 stratum에 대해 보장:
        E[ℓ(λ_s, X, Y) | stratum=s] ≤ α_s    (Angelopoulos & Bates 2024 Theorem 1)
    """

    def __init__(
        self,
        alphas: Optional[dict[str, float]] = None,
        loss_fn: Optional[Callable[[float, float, bool], float]] = None,
        loss_upper_bound: float = 1.0,
        strict: bool = False,
    ):
        """
        Args:
            alphas:           {stratum: ε}. None이면 DEFAULT_ALPHAS 사용.
            loss_fn:          ℓ(λ, score, label) → [0, B]. 기본 missed_escalation_loss.
            loss_upper_bound: ℓ의 상한 B (CRC 공식의 분자). 0/1 loss는 B=1.
            strict:           True이면 stratum 데이터가 min_n 미만일 때 RuntimeError.
        """
        self.alphas = dict(alphas) if alphas is not None else dict(DEFAULT_ALPHAS)
        self.loss_fn = loss_fn or missed_escalation_loss
        self.B = float(loss_upper_bound)
        self.strict = strict

        # validation
        for s, a in self.alphas.items():
            if not (0 < a < 1):
                raise ValueError(f"alphas[{s}]={a} must be in (0,1)")
        if self.B <= 0:
            raise ValueError(f"loss_upper_bound must be > 0, got {self.B}")

        self.lambdas: dict[str, float] = {}
        self.report: CRCFitReport = CRCFitReport()
        self._fitted: bool = False

    # ── public API ─────────────────────────────────────────────────────────

    def fit(
        self,
        scores: list[float],
        labels: list[bool],
        strata: list[str],
    ) -> CRCFitReport:
        """
        Per-stratum CRC fit.

        Args:
            scores: nonconformity scores (클수록 위험 — UQM logprob NLL과 같은 부호)
            labels: True = 실제 에스컬레이션이 필요했던 케이스
            strata: 각 sample의 stratum 문자열 (CRITICAL/HIGH/MODERATE/LOW)

        Returns:
            CRCFitReport — stratum별 (n, λ̂, R̂(λ̂), 데이터 충분성).
        """
        if not (len(scores) == len(labels) == len(strata)):
            raise ValueError(
                f"length mismatch: scores={len(scores)}, "
                f"labels={len(labels)}, strata={len(strata)}"
            )
        if not scores:
            raise ValueError("빈 calibration set")

        self.lambdas.clear()
        self.report = CRCFitReport(n_total=len(scores))

        for stratum, alpha in self.alphas.items():
            mask = [s == stratum for s in strata]
            s_scores = [scores[i] for i, m in enumerate(mask) if m]
            s_labels = [labels[i]  for i, m in enumerate(mask) if m]
            n = len(s_scores)

            min_n = min_n_for_alpha(alpha)
            sufficient = n >= min_n
            if not sufficient:
                msg = (
                    f"[StratifiedCRC] stratum={stratum} n={n} < min_n={min_n} "
                    f"(α={alpha}). CRC 보장이 무효 — 본 stratum에 대한 출력은 "
                    f"vacuous fallback (모든 case escalate)이며, paper-quality "
                    f"보고에 사용해서는 안 됨. 해결: (a) 더 많은 calibration "
                    f"데이터 수집, (b) 더 큰 α 사용, 또는 (c) strict=True로 즉시 fail."
                )
                if self.strict:
                    raise RuntimeError(msg + " (strict=True)")
                warnings.warn(msg, UserWarning, stacklevel=2)
                # fallback: 가장 보수적 — 모든 score보다 작은 λ → 항상 escalate
                # 즉 λ = -inf이면 R̂(-inf)=0. 실용적으로는 min_score - 1.
                lam = (min(s_scores) - 1.0) if s_scores else 0.0
                self.lambdas[stratum] = lam
                self.report.per_stratum[stratum] = StratumStats(
                    stratum=stratum, n=n, alpha=alpha,
                    lambda_hat=lam, empirical_risk_at_lambda=0.0,
                    is_data_sufficient=False,
                )
                continue

            lam = self._solve_lambda(s_scores, s_labels, alpha)
            self.lambdas[stratum] = lam
            R_at_lam = sum(
                self.loss_fn(lam, sc, lb)
                for sc, lb in zip(s_scores, s_labels)
            ) / n
            self.report.per_stratum[stratum] = StratumStats(
                stratum=stratum, n=n, alpha=alpha,
                lambda_hat=lam,
                empirical_risk_at_lambda=R_at_lam,
                is_data_sufficient=True,
            )

        self._fitted = True
        return self.report

    def threshold_for(self, stratum: str) -> float:
        """
        해당 stratum의 λ̂ 반환. 미학습 또는 미존재 stratum 시 가장 보수적 0.0.
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        return self.lambdas.get(stratum, 0.0)

    def coverage_check(
        self,
        holdout_scores: list[float],
        holdout_labels: list[bool],
        holdout_strata: list[str],
        slack: float = 0.10,
    ) -> dict:
        """
        Holdout set에서 stratum별 empirical risk가 α_stratum × (1+slack) 이하인지 검증.

        slack: 유한 표본 변동 허용치 (기본 10%).

        Returns:
            {
                stratum: {
                    "n": int, "empirical_risk": float | None,
                    "target_alpha": float, "ok": bool | None
                }
            }
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        out: dict = {}
        for stratum, alpha in self.alphas.items():
            mask = [s == stratum for s in holdout_strata]
            ss = [holdout_scores[i] for i, m in enumerate(mask) if m]
            sl = [holdout_labels[i]  for i, m in enumerate(mask) if m]
            if not ss:
                out[stratum] = {
                    "n": 0, "empirical_risk": None,
                    "target_alpha": alpha, "ok": None,
                }
                continue
            lam = self.lambdas[stratum]
            R = sum(self.loss_fn(lam, sc, lb) for sc, lb in zip(ss, sl)) / len(ss)
            out[stratum] = {
                "n": len(ss),
                "empirical_risk": round(R, 4),
                "target_alpha": alpha,
                "ok": R <= alpha * (1 + slack),
            }
        return out

    # ── internal ───────────────────────────────────────────────────────────

    def _solve_lambda(self, scores: list[float], labels: list[bool], alpha: float) -> float:
        """
        CRC 알고리즘:
            λ̂ = sup{ λ ∈ candidates : R̂(λ) + B/(n+1) ≤ α }

        R̂(λ) = (1/n) Σ ℓ(λ, score_i, label_i) — λ에 대해 monotone non-decreasing.

        후보: 모든 unique scores + (-∞ proxy as min(scores)-eps). λ가 score 사이에서
        변하면 R̂이 step-function이므로 unique scores를 후보로 충분.

        Returns: 가장 큰 λ 중 risk bound 만족. 모두 위반 시 가장 작은 score 반환
                  (가장 보수적 — 거의 모든 case 에스컬레이션).
        """
        n = len(scores)
        if n == 0:
            return 0.0

        # 후보 정렬 (오름차순). 가장 큰 λ부터 검사.
        candidates = sorted(set(scores))
        threshold_bound = alpha - self.B / (n + 1)

        # threshold_bound < 0이면 어떤 λ도 만족 불가 (n이 너무 작거나 α가 너무 작음).
        # 이 경우 가장 보수적 λ를 반환 (R̂=0 만족하는 λ — score보다 작은 값).
        if threshold_bound < 0:
            return candidates[0] - 1.0

        # 가장 큰 λ부터 검사 (sup 정의에 충실).
        best_lam = candidates[0] - 1.0   # fallback: 가장 보수적
        for lam in reversed(candidates):
            R = sum(self.loss_fn(lam, sc, lb) for sc, lb in zip(scores, labels)) / n
            if R <= threshold_bound:
                best_lam = lam
                break
            # else: 더 작은 λ 시도
        return float(best_lam)


# ── 빠른 동작 확인 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(0)

    # 합성 데이터: 4 stratum, 각 200개. CRITICAL 라벨 분포가 더 위험.
    scores, labels, strata = [], [], []
    n_per = 200
    for stratum, base_rate in [
        ("CRITICAL", 0.30), ("HIGH", 0.20),
        ("MODERATE", 0.10), ("LOW", 0.05),
    ]:
        for _ in range(n_per):
            label = random.random() < base_rate
            # 양성은 score ~ N(2, 1), 음성은 N(0, 1)
            score = random.gauss(2.0 if label else 0.0, 1.0)
            scores.append(score)
            labels.append(label)
            strata.append(stratum)

    crc = StratifiedConformalRiskControl(
        alphas={"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
    )
    report = crc.fit(scores, labels, strata)
    for s, st in report.per_stratum.items():
        print(f"  {s}: n={st.n}, α={st.alpha}, λ̂={st.lambda_hat:.4f}, "
              f"R̂(λ̂)={st.empirical_risk_at_lambda:.4f}")

    # holdout 검증 (같은 분포 다시 생성)
    h_scores, h_labels, h_strata = [], [], []
    for stratum, base_rate in [
        ("CRITICAL", 0.30), ("HIGH", 0.20),
        ("MODERATE", 0.10), ("LOW", 0.05),
    ]:
        for _ in range(n_per):
            label = random.random() < base_rate
            score = random.gauss(2.0 if label else 0.0, 1.0)
            h_scores.append(score); h_labels.append(label); h_strata.append(stratum)

    cov = crc.coverage_check(h_scores, h_labels, h_strata)
    print("\nCoverage check (holdout):")
    for s, c in cov.items():
        ok = "✓" if c["ok"] else "✗"
        print(f"  {s}: R={c['empirical_risk']}, target α={c['target_alpha']} {ok}")
