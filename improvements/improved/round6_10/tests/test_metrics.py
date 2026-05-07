"""metrics_utils tests (audit #11, #16, audit 6.10 bootstrap/Bonferroni/results_dir)."""
from __future__ import annotations

import os
import statistics
from pathlib import Path

import pytest

from experiments.metrics_utils import (
    compute_binary_metrics,
    wilson_ci,
    safe_rate,
    fmt_rate,
    fmt_ci,
    bootstrap_ci,
    bonferroni_adjust,
    holm_bonferroni,
    results_dir,
)


# ── audit #16: silent zero → None ─────────────────────────────────────────────


def test_compute_binary_metrics_all_positives_returns_none_for_over_esc():
    cases = [{"escalated": True, "expected_escalate": True}] * 50
    m = compute_binary_metrics(cases)
    assert m["safety_recall"] == 1.0
    assert m["over_escalation_rate"] is None       # 분모 0
    assert m["over_escalation_ok"] is False         # None은 OK 아님


def test_compute_binary_metrics_all_negatives():
    cases = [{"escalated": False, "expected_escalate": False}] * 30
    m = compute_binary_metrics(cases)
    assert m["safety_recall"] is None
    assert m["over_escalation_rate"] == 0.0


def test_safe_rate_zero_denom():
    assert safe_rate(5, 0) is None
    assert safe_rate(5, 10) == 0.5


# ── audit #11: Wilson CI ──────────────────────────────────────────────────────


def test_wilson_ci_basic():
    lo, hi = wilson_ci(0.95, 50)
    assert 0 < lo < 0.95 < hi <= 1.0


def test_wilson_ci_zero_n():
    lo, hi = wilson_ci(0.5, 0)
    assert lo == 0.0 and hi == 1.0


# ── audit 6.10: bootstrap CI ─────────────────────────────────────────────────


def test_bootstrap_ci_mean():
    samples = [(0.9,), (0.8,), (0.3,), (0.2,), (0.85,), (0.15,), (0.5,), (0.6,)]
    ci = bootstrap_ci(samples, lambda s: statistics.mean(x[0] for x in s), n_iter=200)
    assert ci is not None
    lo, hi = ci
    assert lo < hi


def test_bootstrap_ci_too_small_returns_none():
    assert bootstrap_ci([(1,)] * 3, lambda s: 0.5) is None


# ── audit 6.10: multiple-comparison correction ───────────────────────────────


def test_bonferroni_adjust():
    assert bonferroni_adjust([0.01, 0.04, 0.5]) == [0.03, 0.12, 1.0]


def test_holm_bonferroni_step_down():
    rej = holm_bonferroni([0.001, 0.04, 0.5], alpha=0.05)
    assert rej == [True, False, False]


# ── audit 6.10: --run-tag results_dir ────────────────────────────────────────


def test_results_dir_default(tmp_path):
    d = tmp_path / "default"
    d.mkdir()
    # 환경변수 미설정 시
    os.environ.pop("UASEF_RESULTS_DIR", None)
    assert results_dir(d) == d


def test_results_dir_env_override(tmp_path):
    custom = tmp_path / "tagged"
    os.environ["UASEF_RESULTS_DIR"] = str(custom)
    try:
        out = results_dir(tmp_path / "default")
        assert out == custom
        assert custom.exists()
    finally:
        del os.environ["UASEF_RESULTS_DIR"]


# ── 포맷팅 ────────────────────────────────────────────────────────────────────


def test_fmt_rate_none():
    assert fmt_rate(None) == "N/A"
    assert fmt_rate(0.95) == "0.9500"


def test_fmt_ci_none():
    assert fmt_ci(None) == ""
    assert "0.900" in fmt_ci((0.9, 0.95))
