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


# ── functional API (thin, importable) ─────────────────────────────────────────
# Each detector is exposed as a plain function so callers can use the suite
# without instantiating classes:  from models.audit_detectors import detect_orientation

def detect_orientation(scores, labels, *, margin: float = 0.0) -> AuditFlag:
    return OrientationDetector(margin).detect(scores, labels)


def detect_escalate_all(over_esc_rate: float, *, vacuous_thr: float = 0.95) -> AuditFlag:
    return EscalateAllDetector(vacuous_thr).detect(over_esc_rate)


def detect_temporal_leakage(feature_time, outcome_time, label, *, frac_thr: float = 0.05) -> AuditFlag:
    return TemporalLeakageDetector(frac_thr).detect(feature_time, outcome_time, label)


def detect_informative_missingness(auroc_full, auroc_value, auroc_flag, *,
                                   recover_thr: float = 0.85) -> AuditFlag:
    return InformativeMissingnessDetector(recover_thr).detect(auroc_full, auroc_value, auroc_flag)


def detect_definitional_leakage(X, y, *, names=None, auroc_thr: float = 0.90) -> AuditFlag:
    return DefinitionalLeakageDetector(auroc_thr).detect(X, y, names=names)


#: name -> detector class, for programmatic iteration / documentation
DETECTORS = {
    "orientation": OrientationDetector,
    "escalate_all": EscalateAllDetector,
    "temporal_leakage": TemporalLeakageDetector,
    "informative_missingness": InformativeMissingnessDetector,
    "definitional_leakage": DefinitionalLeakageDetector,
}

__all__ = [
    "AuditFlag", "DETECTORS",
    "OrientationDetector", "EscalateAllDetector", "TemporalLeakageDetector",
    "InformativeMissingnessDetector", "DefinitionalLeakageDetector",
    "detect_orientation", "detect_escalate_all", "detect_temporal_leakage",
    "detect_informative_missingness", "detect_definitional_leakage",
    "run_all_detectors",
]


def run_all_detectors(*, scores=None, labels=None, over_esc_rate=None,
                      feature_time=None, outcome_time=None,
                      auroc_full=None, auroc_value=None, auroc_flag=None,
                      X=None, y=None, names=None) -> dict[str, AuditFlag]:
    """Run whichever detectors have their inputs supplied; return name -> AuditFlag.

    Every argument is optional — a detector runs only if its inputs are present,
    so the same call site works whether you hold scores, a feature matrix, both,
    or precomputed AUROCs."""
    out: dict[str, AuditFlag] = {}
    if scores is not None and labels is not None:
        out["orientation"] = detect_orientation(scores, labels)
    if over_esc_rate is not None:
        out["escalate_all"] = detect_escalate_all(over_esc_rate)
    if feature_time is not None and outcome_time is not None and labels is not None:
        out["temporal_leakage"] = detect_temporal_leakage(feature_time, outcome_time, labels)
    if auroc_full is not None and auroc_flag is not None:
        av = auroc_value if auroc_value is not None else auroc_full
        out["informative_missingness"] = detect_informative_missingness(auroc_full, av, auroc_flag)
    if X is not None and y is not None:
        out["definitional_leakage"] = detect_definitional_leakage(X, y, names=names)
    return out


# ── smoke test:  python -m models.audit_detectors ─────────────────────────────
def _smoke() -> int:
    """Synthetic known-answer smoke test (no data files, deterministic).

    Each detector is exercised on a clean case (must NOT flag) and a
    contaminated case (must flag). Returns process exit code (0 = all pass)."""
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.3).astype(int)
    # correctly-signed score: positives higher; inverted: positives lower
    good = rng.normal(y * 1.2, 1.0)
    checks = []  # (name, expected_flag, AuditFlag)

    checks.append(("orientation/clean", False, detect_orientation(good, y)))
    checks.append(("orientation/inverted", True, detect_orientation(-good, y)))
    checks.append(("escalate_all/clean", False, detect_escalate_all(0.30)))
    checks.append(("escalate_all/vacuous", True, detect_escalate_all(0.99)))

    ft = rng.random(n); ot = np.where(y == 1, rng.random(n), np.inf)
    ft_clean = np.where(y == 1, ot - 0.3, ft)          # features before outcome
    ft_leak = np.where(y == 1, ot + 0.3, ft)           # features after outcome
    checks.append(("temporal/clean", False, detect_temporal_leakage(ft_clean, ot, y)))
    checks.append(("temporal/leak", True, detect_temporal_leakage(ft_leak, ot, y)))

    checks.append(("missingness/value-driven", False,
                   detect_informative_missingness(0.80, 0.79, 0.52)))
    checks.append(("missingness/ordering-driven", True,
                   detect_informative_missingness(0.80, 0.55, 0.79)))

    Xr = rng.normal(size=(n, 4))                        # random, non-leaky
    Xl = np.column_stack([Xr, y + rng.normal(0, 0.01, n)])  # last col = label
    checks.append(("definitional/clean", False, detect_definitional_leakage(Xr, y)))
    checks.append(("definitional/leak", True,
                   detect_definitional_leakage(Xl, y, names=["a", "b", "c", "d", "LEAK"])))

    print(f"UASEF diagnostic detectors — smoke test (n={n}, seed=0)\n")
    print(f"{'case':32s} {'expect':7s} {'flagged':8s} {'stat':>7s}  result")
    print("-" * 72)
    ok = True
    for name, expected, flag in checks:
        passed = flag.flagged == expected
        ok &= passed
        print(f"{name:32s} {str(expected):7s} {str(flag.flagged):8s} "
              f"{flag.statistic:7.3f}  {'PASS' if passed else 'FAIL'}")
    print("-" * 72)
    print(f"{'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(f.flagged == e for _, e, f in checks)}/{len(checks)})")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
