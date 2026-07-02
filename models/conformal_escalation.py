"""
Clean, unit-tested conformal escalation core (Round 15 rebuild).

Motivation: the R10-R13 pipeline accumulated three interacting convention
bugs (score sign, threshold-selection direction, and vacuous fallbacks)
that produced spurious escalate-all "collapses." This module rebuilds the
core from scratch with ONE explicit convention, an orientation guard, and
a companion test suite (tests/test_conformal_escalation.py) that pins the
behavior on synthetic data with KNOWN answers.

CONVENTION (single source of truth)
-----------------------------------
- score s(x): higher = riskier = more likely to need escalation.
- escalate(x) := 1{ s(x) > lambda }.
- miss  := positive (y=1) NOT escalated  = 1{ y=1 and s <= lambda }.
- over  := negative (y=0) escalated       = 1{ y=0 and s >  lambda }.

Standard CRC (miss-only loss) picks the *efficient* threshold
    lambda_hat = sup { lambda : R_hat_miss(lambda) + B/(n+1) <= alpha }
i.e. the LARGEST lambda still meeting the miss bound (escalate the FEWEST
cases). Crucially, with a discriminating score this is NOT escalate-all:
escalate-all is the lambda -> -inf endpoint; CRC takes the sup end.

Bounded CRC (two-sided loss c_m*miss + c_o*over) minimizes the risk over
the feasible set, and reports INFEASIBLE (never a silent escalate-all
fallback) when no lambda meets the bound.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


class OrientationError(RuntimeError):
    """Raised/warned when calibration positives do not score higher than
    negatives, indicating the score sign is likely inverted."""


def check_orientation(scores: np.ndarray, labels: np.ndarray,
                       strict: bool = False) -> float:
    """Return AUROC(score, label). Warn (or raise) if < 0.5 (inverted)."""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U AUROC with tie handling
    order_vals = np.concatenate([pos, neg])
    ranks = _avg_ranks(order_vals)
    pos_ranks = ranks[:len(pos)]
    u = pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0
    au = float(u / (len(pos) * len(neg)))
    if au < 0.5:
        msg = (f"score orientation looks INVERTED: AUROC={au:.3f} < 0.5 "
               f"(positives score lower than negatives). The convention is "
               f"'higher score = riskier'. Flip the score sign upstream.")
        if strict:
            raise OrientationError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return au


def _avg_ranks(vals: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties averaged."""
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals), dtype=float)
    sorted_vals = vals[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _rates(scores, labels, lam):
    """Return (miss_rate, over_rate, n_pos, n_neg) at threshold lam."""
    pos = labels == 1; neg = labels == 0
    n_pos = int(pos.sum()); n_neg = int(neg.sum())
    miss = int((pos & (scores <= lam)).sum())
    over = int((neg & (scores > lam)).sum())
    miss_rate = miss / n_pos if n_pos else float("nan")
    over_rate = over / n_neg if n_neg else float("nan")
    return miss, over, miss_rate, over_rate, n_pos, n_neg


# ─────────────────────────────────────────────────────────────────────────────
# Standard CRC (miss-only) — efficient threshold (sup lambda)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StandardCRC:
    """Miss-only conformal risk control with the EFFICIENT (sup lambda)
    threshold. B = 1 (0-1 loss). No silent escalate-all fallback:
    infeasible target -> infeasible_=True."""
    alpha: float
    threshold_: float | None = field(default=None, init=False)
    infeasible_: bool = field(default=False, init=False)
    n_cal_: int = field(default=0, init=False)

    def fit(self, scores, labels, check_orient: bool = True) -> "StandardCRC":
        scores = np.asarray(scores, float); labels = np.asarray(labels, int)
        m = np.isfinite(scores); scores, labels = scores[m], labels[m]
        n = len(scores); self.n_cal_ = n
        if n == 0:
            self.infeasible_ = True; self.threshold_ = None; return self
        if check_orient:
            check_orientation(scores, labels)
        B = 1.0
        bound = self.alpha - B / (n + 1)
        if bound < 0:
            # even 0 empirical miss cannot meet the finite-sample bound
            self.infeasible_ = True; self.threshold_ = None; return self
        # sup lambda with miss_rate(lambda) <= bound.  miss_rate is
        # non-decreasing in lambda, so scan candidates high->low and take the
        # LARGEST lambda still feasible.
        cand = np.concatenate([[scores.min() - 1e-9], np.unique(scores),
                                [scores.max() + 1e-9]])
        feasible = None
        for lam in np.sort(cand)[::-1]:      # high -> low
            _, _, miss_rate, _, n_pos, _ = _rates(scores, labels, lam)
            if n_pos == 0:
                continue
            if miss_rate <= bound:
                feasible = float(lam); break   # first (largest) feasible
        if feasible is None:
            self.infeasible_ = True; self.threshold_ = None
        else:
            self.threshold_ = feasible; self.infeasible_ = False
        return self

    def evaluate(self, scores, labels) -> dict:
        return _evaluate(self.threshold_, self.infeasible_, scores, labels,
                          alpha=self.alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Bounded CRC (two-sided) — risk-minimizing threshold, explicit infeasible
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundedCRC:
    """Two-sided CRC: loss = c_m*miss + c_o*over (per-case averaged).
    Picks argmin_lambda R_hat over the feasible set; INFEASIBLE if none."""
    alpha: float
    c_miss: float = 0.9
    c_over: float = 0.1
    threshold_: float | None = field(default=None, init=False)
    infeasible_: bool = field(default=False, init=False)
    n_cal_: int = field(default=0, init=False)
    empirical_risk_: float | None = field(default=None, init=False)

    def __post_init__(self):
        if self.c_miss <= 0: raise ValueError("c_miss>0")
        if self.c_over < 0: raise ValueError("c_over>=0")
        if abs(self.c_miss + self.c_over - 1) > 1e-9:
            raise ValueError("c_miss + c_over must == 1")

    def _risk(self, scores, labels, lam):
        pos = labels == 1; neg = labels == 0
        n = len(scores)
        miss = (pos & (scores <= lam)).sum()
        over = (neg & (scores > lam)).sum()
        return (self.c_miss * miss + self.c_over * over) / n

    def fit(self, scores, labels, check_orient: bool = True) -> "BoundedCRC":
        scores = np.asarray(scores, float); labels = np.asarray(labels, int)
        m = np.isfinite(scores); scores, labels = scores[m], labels[m]
        n = len(scores); self.n_cal_ = n
        if n == 0:
            self.infeasible_ = True; self.threshold_ = None; return self
        if check_orient:
            check_orientation(scores, labels)
        B = max(self.c_miss, self.c_over)
        cand = np.concatenate([[scores.min() - 1e-9], np.unique(scores),
                                [scores.max() + 1e-9]])
        best_lam = None; best_risk = None
        for lam in cand:
            r = self._risk(scores, labels, lam)
            if (n / (n + 1)) * r + B / (n + 1) <= self.alpha:
                if best_risk is None or r < best_risk:
                    best_risk = float(r); best_lam = float(lam)
        if best_lam is None:
            self.infeasible_ = True; self.threshold_ = None
            self.empirical_risk_ = None
        else:
            self.threshold_ = best_lam; self.infeasible_ = False
            self.empirical_risk_ = best_risk
        return self

    def evaluate(self, scores, labels) -> dict:
        d = _evaluate(self.threshold_, self.infeasible_, scores, labels,
                       alpha=self.alpha)
        d["c_miss"] = self.c_miss; d["c_over"] = self.c_over
        return d


def _evaluate(threshold, infeasible, scores, labels, alpha) -> dict:
    if infeasible or threshold is None:
        return {"infeasible": True, "threshold": None, "alpha": alpha}
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    m = np.isfinite(scores); scores, labels = scores[m], labels[m]
    miss, over, miss_rate, over_rate, n_pos, n_neg = _rates(scores, labels, threshold)
    from experiments.metrics_utils import clopper_pearson_upper
    upper = clopper_pearson_upper(miss, n_pos, 0.95) if n_pos else None
    # Two distinct notions of "coverage":
    #  - CRC guarantee (in expectation): miss_rate <= alpha. This is what CRC
    #    actually controls. Note CP-upper(miss_rate=alpha) is ALWAYS > alpha,
    #    so requiring CP-upper<=alpha (as the earlier UASEF paper did) is a
    #    strictly stronger, often-unattainable bar and was over-strict.
    #  - High-confidence coverage: CP 95% upper <= alpha (stronger).
    satisfies_crc = (miss_rate is not None and miss_rate <= alpha)
    high_conf = (upper is not None and upper <= alpha)
    vacuous = (over_rate is not None and over_rate >= 0.95)
    return {
        "infeasible": False, "threshold": threshold, "alpha": alpha,
        "n_pos": n_pos, "n_neg": n_neg, "misses": miss, "over_esc": over,
        "miss_rate": miss_rate, "over_esc_rate": over_rate,
        "exact_upper95": upper,
        "satisfies_crc": satisfies_crc,        # miss_rate <= alpha (CRC guarantee)
        "high_conf_coverage": high_conf,        # CP upper <= alpha (stronger)
        "vacuous": vacuous,
        # genuine win = meets the CRC guarantee AND is not escalate-all
        "genuine_win": satisfies_crc and not vacuous,
    }
