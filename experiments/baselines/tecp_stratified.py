r"""
TECP-stratified — Per-stratum split-CP version of TECP (Round 7 ablation).

This baseline isolates the contribution of *stratification itself* from the
combined v2 framework (Stratified CRC + Multi-Trigger + Cost-Aware). It
fits one TECP threshold per stratum at the same per-stratum α as Pivot A,
then predicts using a stratum-aware predictor.

Why this baseline exists:
  Reviewers may object that the published TECP/Quach/SE baselines were
  evaluated under a single global α and are therefore "handicapped" by
  not being given access to stratum information. This adapter gives TECP
  the same stratification handle that Pivot A uses, so the remaining gap
  is attributable purely to (a) CRC's expected-loss control vs split-CP's
  rank-based control, (b) the cost-aware threshold optimization, and
  (c) the multi-trigger combination.

Difference from Stratified CRC (Pivot A):
  - This adapter uses split-CP rank quantile (vacuous for the
    miss-rate loss when n is small)
  - Pivot A uses CRC's $\hat{R}(\lambda) + B/(n+1) \le \alpha$ bound,
    which is tighter for binary loss and has a finite-sample correction.

Predictor signature differs from non-stratified baselines: predict(score, stratum).
"""
from __future__ import annotations

import math
from . import BaselineAdapter


class TECPStratifiedBaseline(BaselineAdapter):
    name = "TECP-stratified"

    def __init__(self, alphas: dict[str, float] | None = None):
        if alphas is None:
            alphas = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
        self.alphas = alphas
        self.thresholds: dict[str, float] = {}
        self.n_per_stratum: dict[str, int] = {}

    def fit(self, scores: list[float], labels: list[bool], strata: list[str] | None = None) -> None:
        """
        Per-stratum split-CP fit. labels is unused (CP is label-free).
        Requires `strata` parameter (extends the BaselineAdapter signature).
        """
        if strata is None:
            raise ValueError("TECPStratifiedBaseline.fit requires `strata` argument.")
        if not (len(scores) == len(strata) == len(labels)):
            raise ValueError("scores, labels, strata must be same length.")

        for s in self.alphas:
            idx = [i for i, st in enumerate(strata) if st == s]
            if not idx:
                self.thresholds[s] = float("inf")
                self.n_per_stratum[s] = 0
                continue
            stratum_scores = sorted(scores[i] for i in idx)
            n = len(stratum_scores)
            alpha_s = self.alphas[s]
            rank = max(1, min(n, math.ceil((n + 1) * (1 - alpha_s))))
            self.thresholds[s] = stratum_scores[rank - 1]
            self.n_per_stratum[s] = n

    def predict(self, score: float, stratum: str | None = None) -> bool:
        """Stratum-aware prediction. score > threshold[stratum] ⇒ escalate."""
        if stratum is None:
            raise ValueError("TECPStratifiedBaseline.predict requires `stratum`.")
        thr = self.thresholds.get(stratum, float("inf"))
        return score > thr

    def info(self) -> dict:
        return {
            "name": self.name,
            "alphas": self.alphas,
            "thresholds": self.thresholds,
            "n_per_stratum": self.n_per_stratum,
            "reference": (
                "Xu & Lu (2025) TECP, arXiv:2509.00461 + Romano et al. (2020) "
                "class-conditional CP, NeurIPS — composed for ablation."
            ),
        }
