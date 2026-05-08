"""
UASEF v1-cost-aware — v1 heuristic-multiplier baseline tuned via cost grid (Round 7 ablation).

Why this exists:
  §5.5 of the main paper notes that v1's published multipliers
  {CRITICAL=0.60, HIGH=0.75, MODERATE=1.00, LOW=1.30} were chosen by an
  audit-time grid search optimizing F1 — *not* under the cost matrix used
  by Pivot C. A reviewer can object: "of course v2 wins on cost — v1
  wasn't tuned for cost." This adapter retunes the v1 multipliers under
  the **same** stratum-specific cost matrix as Pivot C, then evaluates.

Mechanism:
  1. Fit a single global CP threshold q̂ at α=0.10 (same as v1).
  2. For each stratum, sweep multiplier m ∈ MULTIPLIER_GRID and pick the
     m that minimizes expected cost on the calibration set:
        cost = c_miss * FN(q̂·m) + c_over * FP(q̂·m)
     subject to no constraint other than m > 0.
  3. At test time, predict via score > q̂·m[stratum].

Difference from Pivot C:
  - This is *post-hoc* multiplier tuning, not stratum-conditional CRC.
  - There is no per-stratum coverage guarantee — only point-wise cost
    optimization on calibration. So if v1-cost-aware comes close to v2
    on cost but fails the CRC bound, that's the value of Pivot A.

Reference:
  Round 6 audit cycle (improvements/README.md §6) — original v1 multipliers.
  Elkan (2001) IJCAI — cost-sensitive learning.
"""
from __future__ import annotations

import math
from . import BaselineAdapter
from models.cost_aware_calibration import DEFAULT_COST_MATRIX


MULTIPLIER_GRID = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90,
                   1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00]


class UASEFv1CostAwareBaseline(BaselineAdapter):
    name = "UASEF v1-cost-aware"

    def __init__(
        self,
        alpha: float = 0.10,
        cost_matrix: dict | None = None,
        multiplier_grid: list[float] | None = None,
    ):
        self.alpha = alpha
        self.cost_matrix = cost_matrix or DEFAULT_COST_MATRIX
        self.multiplier_grid = multiplier_grid or MULTIPLIER_GRID
        self.q_hat: float = float("inf")
        self.n: int = 0
        self.multipliers: dict[str, float] = {}
        self.fit_costs: dict[str, float] = {}

    def fit(
        self,
        scores: list[float],
        labels: list[bool],
        strata: list[str] | None = None,
    ) -> None:
        """
        Two-stage fit:
          (1) Global CP threshold from sorted scores.
          (2) Per-stratum cost-tuned multiplier.
        Requires `strata` (extends the BaselineAdapter signature).
        """
        if strata is None:
            raise ValueError("UASEFv1CostAwareBaseline.fit requires `strata`.")
        if not (len(scores) == len(strata) == len(labels)):
            raise ValueError("scores/labels/strata length mismatch.")
        if not scores:
            raise ValueError("empty calibration set")

        self.n = len(scores)
        sorted_s = sorted(scores)
        rank = max(1, min(self.n, math.ceil((self.n + 1) * (1 - self.alpha))))
        self.q_hat = sorted_s[rank - 1]

        # Per-stratum multiplier tuning
        for s in self.cost_matrix:
            idx = [i for i, st in enumerate(strata) if st == s]
            if not idx:
                self.multipliers[s] = 1.0
                self.fit_costs[s] = 0.0
                continue
            cm = self.cost_matrix[s]
            best_m, best_cost = 1.0, float("inf")
            for m in self.multiplier_grid:
                thr = self.q_hat * m
                fn = sum(1 for i in idx if labels[i] and scores[i] <= thr)
                fp = sum(1 for i in idx if (not labels[i]) and scores[i] > thr)
                cost = cm["miss"] * fn + cm["over_esc"] * fp
                if cost < best_cost:
                    best_cost, best_m = cost, m
            self.multipliers[s] = best_m
            self.fit_costs[s] = best_cost

    def predict(self, score: float, stratum: str | None = None) -> bool:
        """Stratum-aware prediction via tuned multiplier."""
        if stratum is None:
            raise ValueError("UASEFv1CostAwareBaseline.predict requires `stratum`.")
        m = self.multipliers.get(stratum, 1.0)
        return score > self.q_hat * m

    def info(self) -> dict:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "n_calibration": self.n,
            "tuned_multipliers": self.multipliers,
            "calibration_cost_per_stratum": self.fit_costs,
            "reference": (
                "v1 (UASEF Round 6, heuristic multipliers) re-tuned per Elkan (2001) "
                "IJCAI cost-sensitive learning under DEFAULT_COST_MATRIX."
            ),
        }
