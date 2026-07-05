"""
Label-conditional (Mondrian) split-conformal escalation for pre-release LLM QA.

This is the mathematical core of the Phase-0-3 redesign
(improvements/phase0_3_redesign.md): a *pre-release* gate that decides whether to
RELEASE an LLM's answer or ESCALATE it to a human, with the finite-sample
guarantee

        P(release | incorrect) <= alpha .

Convention (single source of truth): a risk score `r(x, a)` with **higher =
more likely to be an error / should escalate**. The gate

        release  <=>  r(x, a) < tau
        escalate <=>  r(x, a) >= tau

The threshold `tau` is calibrated on the ERROR cases only (label-conditional /
Mondrian conformal; Vovk et al. 2003): with the k-th smallest error-case risk

        k    = floor(alpha * (n_err + 1))
        tau  = r_err_sorted[k-1]          (1-indexed k-th smallest)

a fresh exchangeable error case satisfies  P(r_new < tau | incorrect)
= k/(n_err+1) <= alpha  (continuous scores; ties handled conservatively by the
strict `<`). This differs from the endpoint-CRC core
(models/conformal_escalation.py, which controls the miss rate over a nested
threshold family); here the guarantee is conditional on the (unknown at test
time) error label, which is exactly what "don't release wrong answers" needs.

Feasibility: k >= 1 requires  n_err >= ceil((1-alpha)/alpha)
(alpha=0.10 -> 9, alpha=0.05 -> 19; ~30 / ~60 recommended so tau is not swung by
extremes). If n_err is below this, the guarantee can only be met by releasing
nothing (tau = -inf, escalate-all): we return feasible=False and say so
explicitly rather than silently collapsing — the same discipline as the endpoint
core's INFEASIBLE handling.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC(score, label) with label 1 = error. NaN if a class is empty."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def check_orientation(risk: np.ndarray, error: np.ndarray, *, warn: bool = True) -> float:
    """Return AUROC(risk, error); warn if it looks inverted (< 0.5), i.e. errors
    score LOWER than correct answers, which would make `release <=> r < tau`
    release the wrong answers preferentially."""
    au = _auroc(risk, error)
    if warn and np.isfinite(au) and au < 0.5:
        warnings.warn(
            f"risk orientation looks INVERTED: AUROC(risk,error)={au:.3f} < 0.5 "
            "(errors score lower than correct answers). Convention is "
            "'higher risk = more likely error'. Flip the risk sign upstream.",
            RuntimeWarning, stacklevel=2,
        )
    return au


@dataclass
class LCCResult:
    alpha: float
    tau: float                       # release iff risk < tau
    n_err_cal: int
    n_cal: int
    min_n_err: int
    feasible: bool                   # n_err >= ceil((1-alpha)/alpha)
    orientation_auroc: float
    reason: str = ""
    # ── test-set metrics (filled by evaluate) ────────────────────────────────
    released_given_incorrect: float | None = None   # PRIMARY: must be <= alpha
    release_rate: float | None = None
    escalation_rate: float | None = None
    over_escalation: float | None = None            # P(escalate | correct)
    incorrect_given_release: float | None = None    # secondary: accepted-answer error rate
    n_test: int | None = None
    extra: dict = field(default_factory=dict)


class LabelConditionalConformal:
    """Pre-release conformal escalation with P(release | incorrect) <= alpha.

    Usage:
        gate = LabelConditionalConformal(alpha=0.10).fit(risk_cal, error_cal)
        released = gate.predict(risk_test)              # bool array
        res = gate.evaluate(risk_test, error_test)      # LCCResult with metrics
    """

    def __init__(self, alpha: float = 0.10):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._fit: LCCResult | None = None

    def min_n_err(self) -> int:
        return math.ceil((1.0 - self.alpha) / self.alpha)

    def fit(self, risk_cal, error_cal, *, check_orient: bool = True) -> "LabelConditionalConformal":
        risk = np.asarray(risk_cal, float)
        err = np.asarray(error_cal, int)
        if risk.shape != err.shape:
            raise ValueError("risk_cal and error_cal must be the same shape")
        au = check_orientation(risk, err, warn=check_orient)
        err_risk = np.sort(risk[err == 1])
        n_err = int(err_risk.size)
        min_ne = self.min_n_err()
        k = math.floor(self.alpha * (n_err + 1))
        if k >= 1 and n_err >= min_ne:
            tau = float(err_risk[k - 1])          # k-th smallest error risk
            feasible = True
            reason = (f"tau=r_err[{k}] of {n_err} error cases; "
                      f"guarantee P(release|incorrect) <= {k}/{n_err + 1} "
                      f"= {k / (n_err + 1):.3f} <= {self.alpha}")
        else:
            tau = float("-inf")                   # release nothing => escalate-all
            feasible = False
            reason = (f"INFEASIBLE: n_err={n_err} < min {min_ne} for alpha={self.alpha} "
                      f"(k={k}); only escalate-all meets the bound. Collect more "
                      f"error cases or raise alpha.")
        self._fit = LCCResult(alpha=self.alpha, tau=tau, n_err_cal=n_err,
                              n_cal=int(risk.size), min_n_err=min_ne, feasible=feasible,
                              orientation_auroc=au, reason=reason)
        return self

    @property
    def tau(self) -> float:
        if self._fit is None:
            raise RuntimeError("call fit() first")
        return self._fit.tau

    def predict(self, risk_test) -> np.ndarray:
        """Return a boolean array: True = RELEASE (risk < tau), False = escalate."""
        return np.asarray(risk_test, float) < self.tau

    def evaluate(self, risk_test, error_test) -> LCCResult:
        """Compute locked-test metrics. error_test: 1 = incorrect, 0 = correct."""
        if self._fit is None:
            raise RuntimeError("call fit() first")
        risk = np.asarray(risk_test, float)
        err = np.asarray(error_test, int)
        released = risk < self._fit.tau
        n = int(risk.size)
        n_inc = int((err == 1).sum())
        n_cor = int((err == 0).sum())
        n_rel = int(released.sum())
        r = self._fit
        r.n_test = n
        r.released_given_incorrect = (float((released & (err == 1)).sum() / n_inc)
                                      if n_inc else float("nan"))
        r.release_rate = float(n_rel / n) if n else float("nan")
        r.escalation_rate = float((~released).sum() / n) if n else float("nan")
        r.over_escalation = (float(((~released) & (err == 0)).sum() / n_cor)
                             if n_cor else float("nan"))
        r.incorrect_given_release = (float((released & (err == 1)).sum() / n_rel)
                                     if n_rel else float("nan"))
        r.extra = {"n_incorrect": n_inc, "n_correct": n_cor, "n_released": n_rel,
                   "alpha_satisfied": bool(np.isnan(r.released_given_incorrect)
                                           or r.released_given_incorrect <= self.alpha)}
        return r
