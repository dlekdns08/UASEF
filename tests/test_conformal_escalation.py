"""
Unit tests pinning the conformal-escalation conventions on SYNTHETIC data
with KNOWN answers. These are the guardrails that would have caught the
R10-R13 sign / threshold-direction / vacuous-fallback bugs.

Run: .venv/bin/python -m pytest tests/test_conformal_escalation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.conformal_escalation import (
    StandardCRC, BoundedCRC, check_orientation, OrientationError,
)


def _make(sep: float, n=4000, pos_rate=0.35, seed=0):
    """Synthetic scores: positives centered at +sep/2, negatives at -sep/2.
    sep=0 -> pure noise (AUROC 0.5); large sep -> separable (AUROC ~1).
    Convention: higher score = positive/risky."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < pos_rate).astype(int)
    scores = np.where(labels == 1,
                      rng.normal(+sep / 2, 1.0, n),
                      rng.normal(-sep / 2, 1.0, n))
    return scores, labels


# ─────────────────────────────────────────────────────────────────────────────
# Orientation guard
# ─────────────────────────────────────────────────────────────────────────────

def test_orientation_correct_direction():
    s, y = _make(sep=3.0, seed=1)
    au = check_orientation(s, y)
    assert au > 0.9, f"separable positives should give high AUROC, got {au}"


def test_orientation_detects_flip():
    s, y = _make(sep=3.0, seed=1)
    with pytest.warns(RuntimeWarning):
        au = check_orientation(-s, y)   # inverted sign
    assert au < 0.1


def test_orientation_strict_raises_on_flip():
    s, y = _make(sep=3.0, seed=1)
    with pytest.raises(OrientationError):
        check_orientation(-s, y, strict=True)


# ─────────────────────────────────────────────────────────────────────────────
# KEY TEST: separable data => Standard CRC is NON-vacuous (NOT escalate-all)
# This is the test that falsifies "vanilla CRC always collapses to escalate-all".
# ─────────────────────────────────────────────────────────────────────────────

def test_standard_crc_separable_is_not_escalate_all():
    s, y = _make(sep=4.0, seed=2)   # AUROC ~0.997
    crc = StandardCRC(alpha=0.05).fit(s, y)
    assert not crc.infeasible_
    r = crc.evaluate(s, y)
    # coverage met
    assert r["miss_rate"] <= 0.08, r
    # and CRUCIALLY not escalate-all: over_esc must be low
    assert r["over_esc_rate"] < 0.15, (
        f"separable data must yield a non-vacuous threshold, "
        f"got over_esc={r['over_esc_rate']}")
    assert r["genuine_win"], r


def test_standard_crc_pure_noise_is_vacuous_or_infeasible():
    s, y = _make(sep=0.0, seed=3)   # AUROC ~0.5
    crc = StandardCRC(alpha=0.05).fit(s, y)
    # With no signal, the only way to bound miss<=0.05 is to escalate ~all.
    r = crc.evaluate(s, y)
    if not r["infeasible"]:
        assert r["over_esc_rate"] > 0.85, (
            f"pure-noise CRC at alpha=0.05 must be near escalate-all, "
            f"got over_esc={r['over_esc_rate']}")
        assert not r["genuine_win"]


def test_standard_crc_weak_signal_high_over_esc():
    """The substantive property: CRC always meets miss<=alpha, but over_esc
    tracks discrimination. Weak signal (AUROC~0.55) => miss<=alpha met BUT
    over_esc very high (clinically near-useless), and NOT high-confidence."""
    s, y = _make(sep=0.4, seed=4)   # AUROC ~0.55 (like leakage-safe MIMIC)
    crc = StandardCRC(alpha=0.05).fit(s, y)
    r = crc.evaluate(s, y)
    if not r["infeasible"]:
        assert r["satisfies_crc"], "CRC guarantee (miss<=alpha) should hold"
        # to catch 95% of positives from near-random scores you escalate ~all
        assert r["over_esc_rate"] > 0.8, r
        # clinically marginal: high-confidence coverage NOT achieved
        assert not r["high_conf_coverage"], r


# ─────────────────────────────────────────────────────────────────────────────
# Bounded CRC
# ─────────────────────────────────────────────────────────────────────────────

def test_bounded_crc_separable_genuine_win():
    s, y = _make(sep=4.0, seed=5)
    b = BoundedCRC(alpha=0.05, c_miss=0.9, c_over=0.1).fit(s, y)
    assert not b.infeasible_
    r = b.evaluate(s, y)
    assert r["over_esc_rate"] < 0.2, r
    assert r["genuine_win"], r


def test_bounded_crc_pure_noise_infeasible_not_escalate_all():
    s, y = _make(sep=0.0, seed=6)
    b = BoundedCRC(alpha=0.05, c_miss=0.9, c_over=0.1).fit(s, y)
    # b-CRC must NOT silently return escalate-all; it reports INFEASIBLE
    # (Proposition 2: alpha < c_over*Pr(Y=0) excludes the escalate-all endpoint)
    if not b.infeasible_:
        r = b.evaluate(s, y)
        assert r["over_esc_rate"] < 0.99, (
            "b-CRC must never return the escalate-all endpoint")


def test_bounded_crc_prop2_excludes_escalate_all():
    """Directly verify Proposition 2: with alpha < c_over*Pr(Y=0), the
    escalate-all endpoint (over_esc=1) is infeasible."""
    s, y = _make(sep=0.0, pos_rate=0.35, seed=7)  # Pr(Y=0)=0.65
    # c_over=0.1 => c_over*Pr(Y=0)=0.065 > alpha=0.05 => escalate-all excluded
    b = BoundedCRC(alpha=0.05, c_miss=0.9, c_over=0.1).fit(s, y)
    if not b.infeasible_:
        r = b.evaluate(s, y)
        assert r["over_esc_rate"] < 0.95


# ─────────────────────────────────────────────────────────────────────────────
# The bug reproduction: inverted sign forces escalate-all even on separable data
# ─────────────────────────────────────────────────────────────────────────────

def test_inverted_sign_reproduces_escalate_all_bug():
    """With the OLD buggy convention (score = -P), separable data collapses
    to escalate-all — reproducing the R10.4 artifact."""
    s, y = _make(sep=4.0, seed=8)   # separable, AUROC ~1 in correct orientation
    # feed the INVERTED score (the bug): positives now score LOW
    crc = StandardCRC(alpha=0.05).fit(-s, y, check_orient=False)
    r = crc.evaluate(-s, y)
    if not r["infeasible"]:
        # the bug: even perfectly separable data yields escalate-all
        assert r["over_esc_rate"] > 0.9, (
            f"inverted sign should force escalate-all, got {r['over_esc_rate']}")
        assert not r["genuine_win"]
