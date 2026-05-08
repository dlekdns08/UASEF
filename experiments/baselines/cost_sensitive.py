"""
Cost-Sensitive baseline — Single-stratum CP with cost-tuned threshold (Round 7 ablation).

This is the strong-fairness counterpart to TECP-stratified: where
TECP-stratified isolates stratification, this baseline isolates the
*cost-aware threshold optimization* without per-stratum CRC.

Mechanism:
  1. Compute global NLL nonconformity scores on calibration set.
  2. Sweep candidate thresholds along the sorted unique scores.
  3. At each candidate, compute total expected cost on calibration:
        cost = c_miss * FN + c_over * FP
     using a *single* miss / over_esc cost (e.g., the average of the
     stratum-specific values, or the CRITICAL row of the cost matrix
     to give the strongest single-threshold cost-aware signal).
  4. Pick the threshold minimizing expected cost.

Why this baseline exists:
  §6.3 reports a 38× cost reduction relative to F1-symmetric
  optimization. This is a deliberately weak comparator. With a
  cost-sensitive baseline, the headline 38× shrinks toward 5–10× — a
  more honest characterization of Pivot C's marginal contribution above
  general cost-sensitive learning. Pivot C remains preferred because of
  its *per-stratum* CRC constraint, but reviewers may reject 38× as a
  "stacked deck" without this comparator.

Reference:
  Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI.
"""
from __future__ import annotations

from . import BaselineAdapter


class CostSensitiveBaseline(BaselineAdapter):
    name = "Cost-Sensitive (single-α)"

    def __init__(self, c_miss: float = 100.0, c_over: float = 1.0):
        """
        c_miss: penalty for a missed escalation (FN).
        c_over: penalty for an over-escalation (FP).
        Default ratio 100:1 ≈ HIGH-stratum default; user can pass
        CRITICAL ratio (1000:1) for a stronger cost signal.
        """
        if c_miss <= 0 or c_over <= 0:
            raise ValueError("c_miss and c_over must be positive.")
        self.c_miss = c_miss
        self.c_over = c_over
        self.threshold: float = float("inf")
        self.n: int = 0
        self.fit_cost: float | None = None

    def fit(self, scores: list[float], labels: list[bool]) -> None:
        """Sweep unique scores; pick min-cost threshold."""
        if not scores:
            raise ValueError("empty calibration set")
        if len(scores) != len(labels):
            raise ValueError("scores and labels mismatch")
        self.n = len(scores)

        # Threshold candidates: sorted unique scores plus ±ε at boundaries.
        unique = sorted(set(scores))
        eps = 1e-9
        candidates = [unique[0] - eps] + unique + [unique[-1] + eps]

        best_cost = float("inf")
        best_thr = float("inf")
        for thr in candidates:
            fn = sum(1 for s, y in zip(scores, labels) if y and s <= thr)
            fp = sum(1 for s, y in zip(scores, labels) if (not y) and s > thr)
            cost = self.c_miss * fn + self.c_over * fp
            if cost < best_cost:
                best_cost = cost
                best_thr = thr

        self.threshold = best_thr
        self.fit_cost = best_cost

    def predict(self, score: float) -> bool:
        return score > self.threshold

    def info(self) -> dict:
        return {
            "name": self.name,
            "c_miss": self.c_miss,
            "c_over": self.c_over,
            "threshold": self.threshold,
            "n_calibration": self.n,
            "fit_cost": self.fit_cost,
            "reference": "Elkan (2001) IJCAI cost-sensitive learning.",
        }
