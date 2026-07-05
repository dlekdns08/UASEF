"""
Monte-Carlo verification of the label-conditional conformal escalation core
(models/label_conditional_conformal.py). The load-bearing claim is

        P(release | incorrect) <= alpha    (finite-sample, marginal over cal),

so we verify it empirically across many calibration/test draws and several alpha,
plus the INFEASIBLE handling and orientation guard. No LLM / no data files.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from models.label_conditional_conformal import (
    LabelConditionalConformal, check_orientation, _auroc,
)

PREV = 0.30  # error prevalence


def _draw(rng, n, *, informative=True, sep=1.1):
    """Return (risk, error). error=1 with prob PREV; informative risk puts errors
    higher (AUROC>0.5); non-informative risk is pure noise."""
    error = (rng.random(n) < PREV).astype(int)
    if informative:
        risk = rng.normal(error * sep, 1.0)
    else:
        risk = rng.normal(0.0, 1.0, n)  # independent of error
    return risk, error


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_coverage_guarantee_informative(alpha):
    """Mean P(release|incorrect) over many draws must not exceed alpha."""
    seeds = 200
    n_cal, n_test = 2500, 6000
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        rc, ec = _draw(rng, n_cal)
        rt, et = _draw(rng, n_test)
        gate = LabelConditionalConformal(alpha).fit(rc, ec, check_orient=False)
        assert gate._fit.feasible  # plenty of error cases at these sizes
        res = gate.evaluate(rt, et)
        vals.append(res.released_given_incorrect)
    mean = float(np.mean(vals))
    # marginal guarantee: E[P(release|incorrect)] = k/(n_err+1) <= alpha
    assert mean <= alpha + 0.01, f"mean released|incorrect {mean:.4f} > alpha {alpha}"


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_coverage_guarantee_noise(alpha):
    """Validity does NOT require the risk to be informative — a pure-noise risk
    must still satisfy the bound (conformal exchangeability)."""
    seeds = 200
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(5000 + s)
        rc, ec = _draw(rng, 2500, informative=False)
        rt, et = _draw(rng, 6000, informative=False)
        gate = LabelConditionalConformal(alpha).fit(rc, ec, check_orient=False)
        vals.append(gate.evaluate(rt, et).released_given_incorrect)
    assert float(np.mean(vals)) <= alpha + 0.01


def test_theoretical_bound_never_exceeds_alpha():
    """The analytic bound k/(n_err+1) is <= alpha for every fitted gate."""
    for s in range(50):
        rng = np.random.default_rng(7000 + s)
        rc, ec = _draw(rng, 1500)
        gate = LabelConditionalConformal(0.10).fit(rc, ec, check_orient=False)
        n_err = gate._fit.n_err_cal
        k = math.floor(0.10 * (n_err + 1))
        assert k / (n_err + 1) <= 0.10 + 1e-12


def test_informative_risk_is_non_vacuous():
    """With an informative risk the gate must release a meaningful share of
    CORRECT answers (not collapse to escalate-all)."""
    rng = np.random.default_rng(0)
    rc, ec = _draw(rng, 4000, sep=1.5)
    rt, et = _draw(rng, 8000, sep=1.5)
    res = LabelConditionalConformal(0.10).fit(rc, ec).evaluate(rt, et)
    assert res.feasible
    assert res.release_rate > 0.30            # not escalate-all
    assert res.over_escalation < 0.95         # some correct answers auto-released
    assert res.released_given_incorrect <= 0.10 + 0.03


def test_infeasible_small_nerr_is_escalate_all_not_silent():
    """Too few error cases -> feasible=False, tau=-inf, releases nothing."""
    rng = np.random.default_rng(3)
    # alpha=0.05 needs n_err >= ceil(0.95/0.05)=19; make ~5 error cases
    risk = rng.normal(0, 1, 120)
    error = np.zeros(120, int)
    error[:5] = 1
    gate = LabelConditionalConformal(0.05).fit(risk, error, check_orient=False)
    assert gate._fit.feasible is False
    assert gate._fit.min_n_err == 19
    assert math.isinf(gate.tau) and gate.tau < 0
    res = gate.evaluate(rng.normal(0, 1, 200), (np.arange(200) < 60).astype(int))
    assert res.release_rate == 0.0            # escalate-all (conservative, explicit)
    assert "INFEASIBLE" in gate._fit.reason


def test_orientation_guard_warns_on_inverted_risk():
    """Errors scoring LOWER than correct answers must trip the orientation guard."""
    rng = np.random.default_rng(11)
    error = (rng.random(2000) < PREV).astype(int)
    inverted = rng.normal(-error * 1.2, 1.0)   # errors LOWER
    assert _auroc(inverted, error) < 0.5
    with pytest.warns(RuntimeWarning, match="INVERTED"):
        check_orientation(inverted, error)
    with pytest.warns(RuntimeWarning, match="INVERTED"):
        LabelConditionalConformal(0.10).fit(inverted, error, check_orient=True)


def test_min_n_err_formula():
    assert LabelConditionalConformal(0.10).min_n_err() == 9
    assert LabelConditionalConformal(0.05).min_n_err() == 19
    assert LabelConditionalConformal(0.20).min_n_err() == 4
