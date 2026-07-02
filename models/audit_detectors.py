"""
Formal audit detectors for conformal clinical-escalation pipelines.

Each of the five failure modes documented in the accompanying paper is turned
here into an explicit DETECTOR with a well-defined (input, statistic, decision
rule). The companion benchmark (experiments/round20_detector_benchmark.py)
measures each detector's sensitivity/specificity on data with KNOWN answers via
controlled contamination injection.

A detector returns an AuditFlag(flagged, statistic, threshold, detail).

Detectors
---------
1. OrientationDetector          — score sign inverted (positives score below negatives)
2. EscalateAllDetector          — CRC threshold collapsed to escalate-everything
3. TemporalLeakageDetector      — feature timestamps fall after the outcome time
4. InformativeMissingnessDetector — signal is which-labs-ordered, not the values
5. DefinitionalLeakageDetector  — a single feature is near-perfectly outcome-predictive
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AuditFlag:
    name: str
    flagged: bool
    statistic: float
    threshold: float
    detail: str = ""


# ── shared ────────────────────────────────────────────────────────────────────

def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float); sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


# ── 1. Orientation ────────────────────────────────────────────────────────────

class OrientationDetector:
    """Flag when calibration positives score BELOW negatives (AUROC < 0.5),
    the fingerprint of a score-sign inversion that collapses CRC to
    escalate-all under escalate-iff-score>lambda."""
    def __init__(self, margin: float = 0.0):
        self.margin = margin
    def detect(self, scores, labels) -> AuditFlag:
        au = _auroc(scores, labels)
        thr = 0.5 - self.margin
        return AuditFlag("orientation", au < thr, au, thr,
                         f"AUROC(score,label)={au:.3f}; <{thr} => inverted sign")


# ── 2. Escalate-all (vacuous) ─────────────────────────────────────────────────

class EscalateAllDetector:
    """Flag when the fitted CRC threshold escalates >= vacuous_thr of negatives.
    Input: over-escalation rate at the fitted threshold on the test set."""
    def __init__(self, vacuous_thr: float = 0.95):
        self.vacuous_thr = vacuous_thr
    def detect(self, over_esc_rate: float) -> AuditFlag:
        return AuditFlag("escalate_all", over_esc_rate >= self.vacuous_thr,
                         float(over_esc_rate), self.vacuous_thr,
                         f"over_esc={over_esc_rate:.3f}")


# ── 3. Temporal leakage ───────────────────────────────────────────────────────

class TemporalLeakageDetector:
    """Flag when a non-trivial fraction of feature timestamps (for positive
    cases) fall AFTER the outcome time — i.e. the feature was measured after
    the event it is meant to predict.

    Input: arrays feature_time, outcome_time, label (aligned per case). Only
    positives with an observed feature contribute (outcome_time is NaN/inf for
    negatives)."""
    def __init__(self, frac_thr: float = 0.05):
        self.frac_thr = frac_thr
    def detect(self, feature_time, outcome_time, label) -> AuditFlag:
        ft = np.asarray(feature_time, float); ot = np.asarray(outcome_time, float)
        y = np.asarray(label, int)
        mask = (y == 1) & np.isfinite(ft) & np.isfinite(ot)
        if mask.sum() == 0:
            return AuditFlag("temporal_leakage", False, 0.0, self.frac_thr,
                             "no positive with both feature and outcome time")
        after = (ft[mask] > ot[mask]).mean()
        return AuditFlag("temporal_leakage", after > self.frac_thr, float(after),
                         self.frac_thr,
                         f"{after:.3f} of positive features charted after outcome")


# ── 4. Informative missingness ────────────────────────────────────────────────

class InformativeMissingnessDetector:
    """Flag when the signal is dominated by WHICH features are present (the
    missingness pattern) rather than their values.

    Input: fitted-model AUROCs from the three feature encodings
    (full = value+flag, value_only, flag_only). Rule: if flag_only recovers
    >= recover_thr of full's above-chance AUROC, the pipeline is
    ordering-driven, not value-driven."""
    def __init__(self, recover_thr: float = 0.85):
        self.recover_thr = recover_thr
    def detect(self, auroc_full, auroc_value, auroc_flag) -> AuditFlag:
        base = 0.5
        num = max(auroc_flag - base, 0.0)
        den = max(auroc_full - base, 1e-9)
        recover = num / den
        return AuditFlag("informative_missingness", recover >= self.recover_thr,
                         float(recover), self.recover_thr,
                         f"flag-only recovers {recover:.2%} of full above-chance "
                         f"AUROC (full={auroc_full:.3f}, value={auroc_value:.3f}, "
                         f"flag={auroc_flag:.3f})")


# ── 5. Definitional leakage ───────────────────────────────────────────────────

class DefinitionalLeakageDetector:
    """Flag individual features whose UNIVARIATE AUROC against the label is
    near-perfect (>= auroc_thr): a decision-time feature should not, alone,
    almost-determine the outcome.

    Input: feature matrix X (n, d), labels y, optional feature names.
    Returns a flag if ANY feature exceeds the threshold; detail lists them."""
    def __init__(self, auroc_thr: float = 0.90):
        self.auroc_thr = auroc_thr
    def detect(self, X, y, names=None) -> AuditFlag:
        X = np.asarray(X, float); y = np.asarray(y, int)
        d = X.shape[1]
        names = names or [f"f{j}" for j in range(d)]
        flagged = []
        max_au = 0.5
        for j in range(d):
            au = _auroc(X[:, j], y)
            if not np.isfinite(au):
                continue
            au = max(au, 1 - au)   # direction-robust
            max_au = max(max_au, au)
            if au >= self.auroc_thr:
                flagged.append((names[j], round(au, 3)))
        return AuditFlag("definitional_leakage", len(flagged) > 0, float(max_au),
                         self.auroc_thr,
                         f"leaky features: {flagged}" if flagged
                         else f"max univariate AUROC={max_au:.3f}")
