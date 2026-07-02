"""
Bounded Conformal Risk Control (b-CRC).

Reference: paper/BOUNDED_CRC_ALGORITHM.md (this repo).

Standard CRC uses the loss

    ℓ_CRC(λ, x, y) = 𝟙{y=1 ∧ s(x) > λ}

which is monotone non-increasing in λ; the vacuous solution λ → -∞
(escalate every case) yields ℓ ≡ 0 and satisfies any α. We label this
the "escalate-all collapse."

b-CRC replaces this with a symmetric two-sided loss

    ℓ(λ, x, y) = c_m · 𝟙{y=1 ∧ s(x) ≤ λ}   (miss)
                + c_o · 𝟙{y=0 ∧ s(x) > λ}   (over_esc)

with c_m + c_o = 1, both positive. The over_esc term penalises the
escalate-all solution: as λ → -∞, ℓ → c_o · Pr(Y=0) > 0. Whenever
α < c_o · Pr(Y=0), the vacuous solution is excluded by the CRC
constraint (Proposition 2 of the algorithm note).

The finite-sample coverage guarantee of Angelopoulos et al. (2024)
inherits directly: the loss is bounded in [0, max(c_m, c_o)].

Usage
─────
    from models.bounded_crc import BoundedCRC, StratifiedBoundedCRC

    # single-α
    crc = BoundedCRC(alpha=0.05, c_miss=0.95, c_over=0.05)
    crc.fit(cal_scores, cal_labels)
    escalate = crc.predict(test_scores)   # 1 = escalate
    lam = crc.threshold_                  # fitted threshold

    # per-stratum
    scrc = StratifiedBoundedCRC(
        alphas={"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15},
        c_miss=0.95, c_over=0.05,
    )
    scrc.fit(cal_scores, cal_labels, cal_strata)
    for s in scrc.strata:
        print(s, scrc.threshold_for(s))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core b-CRC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundedCRC:
    """Single-α bounded conformal risk control.

    Parameters
    ----------
    alpha : float
        Target expected loss (in (0, 1)).
    c_miss : float, default 0.95
        Cost weight for missed escalations (positives not escalated).
    c_over : float, default 0.05
        Cost weight for over-escalations (negatives escalated).
        c_miss + c_over must equal 1.

    Attributes (populated after ``fit``)
    -----------------------------------
    threshold_ : float | None
        Selected threshold. ``None`` when the target risk is infeasible
        given the calibration data (b-CRC reports FAIL explicitly rather
        than silently returning the vacuous solution).
    n_cal_ : int
        Number of calibration points used.
    empirical_risk_ : float | None
        Empirical b-CRC loss at ``threshold_``.
    infeasible_ : bool
        True iff no threshold satisfies the target α on the calibration
        set. When True, ``predict`` raises ``BoundedCRCInfeasible``.
    """

    alpha: float
    c_miss: float = 0.95
    c_over: float = 0.05

    threshold_: float | None = field(default=None, init=False)
    n_cal_: int = field(default=0, init=False)
    empirical_risk_: float | None = field(default=None, init=False)
    infeasible_: bool = field(default=False, init=False)

    def __post_init__(self):
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if not (self.c_miss > 0 and self.c_over > 0):
            raise ValueError("c_miss and c_over must both be positive")
        if abs((self.c_miss + self.c_over) - 1.0) > 1e-9:
            raise ValueError(f"c_miss + c_over must equal 1, got "
                              f"{self.c_miss + self.c_over}")

    # ── loss + empirical risk ────────────────────────────────────────────
    def _empirical_risk(self, scores: np.ndarray, labels: np.ndarray,
                          lam: float) -> tuple[float, float, float]:
        """Return (risk, miss_rate, over_esc_rate) at threshold lam."""
        pos_mask = labels == 1
        neg_mask = labels == 0
        n = len(scores)

        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        miss_indicator = pos_mask & (scores <= lam)
        over_indicator = neg_mask & (scores > lam)

        # Per-case losses (averaged over n, not per-class)
        miss_avg = miss_indicator.sum() / n
        over_avg = over_indicator.sum() / n
        risk = self.c_miss * miss_avg + self.c_over * over_avg

        # Rates (per-class, for reporting)
        miss_rate = (miss_indicator.sum() / n_pos) if n_pos else float("nan")
        over_rate = (over_indicator.sum() / n_neg) if n_neg else float("nan")

        return float(risk), float(miss_rate), float(over_rate)

    def fit(self, scores, labels) -> "BoundedCRC":
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels shape mismatch")

        # NaN 제거 (LLM 실패 등)
        finite = np.isfinite(scores)
        scores = scores[finite]
        labels = labels[finite]

        n = len(scores)
        self.n_cal_ = n
        if n == 0:
            self.infeasible_ = True
            self.threshold_ = None
            self.empirical_risk_ = None
            return self

        B = max(self.c_miss, self.c_over)

        # Candidate thresholds: sorted unique scores + boundary sentinels
        eps = 1e-9
        # Sorted low → high; also allow just-below-min (accept everything)
        # and just-above-max (accept nothing)
        candidates = np.unique(scores)
        candidates = np.concatenate([
            [candidates.min() - eps],
            candidates,
            [candidates.max() + eps],
        ])

        # b-CRC constraint: (n/(n+1)) R̂ + B/(n+1) ≤ α
        # We want the *smallest* λ that satisfies the constraint (equivalently
        # the least-conservative escalation rule that still meets the risk
        # target). For b-CRC the risk is non-monotone in λ, so we scan.
        best_lam = None
        best_risk = None
        for lam in candidates:
            risk, _, _ = self._empirical_risk(scores, labels, lam)
            constraint = (n / (n + 1)) * risk + B / (n + 1)
            if constraint <= self.alpha:
                # Prefer the smallest λ (most escalation) among feasible;
                # but many λ can tie, take the smallest.
                if best_lam is None or lam < best_lam:
                    best_lam = float(lam)
                    best_risk = float(risk)

        if best_lam is None:
            self.infeasible_ = True
            self.threshold_ = None
            self.empirical_risk_ = None
        else:
            self.threshold_ = best_lam
            self.empirical_risk_ = best_risk
            self.infeasible_ = False
        return self

    def predict(self, scores) -> np.ndarray:
        if self.infeasible_ or self.threshold_ is None:
            raise BoundedCRCInfeasible(
                f"b-CRC infeasible at α={self.alpha} with n_cal={self.n_cal_}")
        scores = np.asarray(scores, dtype=float)
        return (scores > self.threshold_).astype(int)

    def evaluate(self, scores, labels) -> dict:
        """Empirical risk + rates on test set at the fitted threshold."""
        if self.infeasible_ or self.threshold_ is None:
            return {"infeasible": True, "threshold": None}
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        finite = np.isfinite(scores)
        scores = scores[finite]; labels = labels[finite]
        risk, miss_rate, over_rate = self._empirical_risk(
            scores, labels, self.threshold_)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        misses = int(((labels == 1) & (scores <= self.threshold_)).sum())
        over_esc = int(((labels == 0) & (scores > self.threshold_)).sum())
        return {
            "infeasible": False,
            "threshold": self.threshold_,
            "n": len(scores), "n_pos": n_pos, "n_neg": n_neg,
            "misses": misses, "miss_rate": miss_rate,
            "over_esc": over_esc, "over_esc_rate": over_rate,
            "empirical_risk": risk,
            "alpha": self.alpha,
            "c_miss": self.c_miss, "c_over": self.c_over,
        }


class BoundedCRCInfeasible(RuntimeError):
    """b-CRC could not find a threshold satisfying α on the calibration set.

    This is not a silent-vacuous fallback (as vanilla CRC would give);
    b-CRC declares infeasibility explicitly so downstream code cannot
    accidentally deploy an escalate-all solution.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Stratified b-CRC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StratifiedBoundedCRC:
    """Per-stratum bounded CRC.

    Independent b-CRC per stratum with (optionally per-stratum) cost
    weights.
    """

    alphas: dict[str, float]
    c_miss: float | dict[str, float] = 0.95
    c_over: float | dict[str, float] = 0.05

    per_stratum_: dict[str, BoundedCRC] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.strata = list(self.alphas.keys())

    def _cost_for(self, s: str) -> tuple[float, float]:
        cm = self.c_miss[s] if isinstance(self.c_miss, dict) else self.c_miss
        co = self.c_over[s] if isinstance(self.c_over, dict) else self.c_over
        return cm, co

    def fit(self, scores, labels, strata) -> "StratifiedBoundedCRC":
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        strata = np.asarray(strata)
        for s in self.strata:
            mask = strata == s
            cm, co = self._cost_for(s)
            crc = BoundedCRC(alpha=self.alphas[s], c_miss=cm, c_over=co)
            if mask.sum() > 0:
                crc.fit(scores[mask], labels[mask])
            else:
                crc.infeasible_ = True
            self.per_stratum_[s] = crc
        return self

    def threshold_for(self, s: str) -> float | None:
        return self.per_stratum_[s].threshold_

    def infeasible_for(self, s: str) -> bool:
        return self.per_stratum_[s].infeasible_

    def evaluate(self, scores, labels, strata) -> dict[str, dict]:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        strata = np.asarray(strata)
        out = {}
        for s in self.strata:
            mask = strata == s
            crc = self.per_stratum_[s]
            if crc.infeasible_ or mask.sum() == 0:
                out[s] = {"infeasible": True, "n": int(mask.sum())}
                continue
            out[s] = crc.evaluate(scores[mask], labels[mask])
        return out
