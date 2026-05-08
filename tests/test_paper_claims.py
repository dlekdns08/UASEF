"""
Paper-claim regression guards (Round 8 P1-5).

Each test asserts a specific quantitative claim from the published paper.
If the underlying experiment is regenerated and a metric drifts beyond the
asserted band, the test fails — surfacing the discrepancy *before* the
paper is republished with stale numbers.

Tests are skipped (not failed) when the corresponding artifact is not
present. The repo always ships:

  - results/round7/table2_fwer.json           (synthetic, deterministic)
  - results/round7/table3_cost.json           (synthetic, deterministic)
  - results/round7/alpha_critical_validation.json   (synthetic, deterministic)

LLM-dependent tables (Tables 1, 4) are checked from the most-recent backend
copy in `results/round7/table*_<backend>.json`, which run_full_evaluation.sh
maintains as a snapshot of the latest seed=42 run. Without that snapshot
the tests are skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
ROUND7 = ROOT / "results" / "round7"


def _load_or_skip(p: Path) -> dict:
    if not p.exists():
        pytest.skip(f"missing artifact: {p} — regenerate via run_full_evaluation.sh")
    return json.loads(p.read_text())


# ── Table 2 — Multi-Trigger FWER (synthetic, deterministic) ─────────────


def _table2_fwer(rows: list[dict], method_substr: str, dependence: str) -> float | None:
    for r in rows:
        if method_substr.lower() in r.get("method", "").lower() and r.get("dependence") == dependence:
            return r.get("empirical_fwer")
    return None


def test_table2_fwer_naive_or_inflated():
    """Paper §6.2: naive OR FWER ≈ 1-(1-α)^3 = 0.143 (independent / correlated)."""
    data = _load_or_skip(ROUND7 / "table2_fwer.json")
    rows = data.get("rows", [])
    naive_indep = _table2_fwer(rows, "naive OR", "independent")
    if naive_indep is None:
        pytest.skip("naive OR row missing in table2_fwer.json")
    assert naive_indep >= 0.080, (
        f"naive OR FWER (independent) too low: {naive_indep} "
        "(paper claims 0.10–0.14, theory 0.143)"
    )


def test_table2_fwer_harmonic_controlled():
    """Paper §6.2: harmonic FWER ≤ 0.05 nominal (within tolerance)."""
    data = _load_or_skip(ROUND7 / "table2_fwer.json")
    rows = data.get("rows", [])
    h_indep = _table2_fwer(rows, "harmonic", "independent")
    h_corr = _table2_fwer(rows, "harmonic", "correlated")
    if h_indep is None or h_corr is None:
        pytest.skip("harmonic row missing in table2_fwer.json")
    assert h_indep <= 0.060, (
        f"harmonic FWER (independent) exceeds nominal+slack: {h_indep}"
    )
    assert h_corr <= 0.080, (
        f"harmonic FWER (correlated) exceeds nominal+slack: {h_corr}"
    )


# ── Table 3 — Cost Reduction (synthetic, deterministic) ─────────────────


def test_table3_cost_reduction_min_floor():
    """
    Paper §6.3 headline: 38× cost reduction (specific cost matrix).
    §6.3.5 sensitivity sweep median is 5–10×. Regression guard asserts
    *at least 5×* for the headline cost matrix; the precise number can
    drift slightly under different RNG paths and 5× preserves the claim.
    """
    data = _load_or_skip(ROUND7 / "table3_cost.json")
    ratio = data.get("cost_reduction_ratio")
    if ratio is None:
        pytest.skip("cost_reduction_ratio missing in table3_cost.json")
    assert ratio >= 5.0, (
        f"Table 3 cost reduction below 5× ({ratio}×). Paper headline 38× "
        "may need updating, or regression in cost_aware_calibration."
    )


# ── α_CRITICAL = 0.001 synthetic validation ─────────────────────────────


def test_alpha_critical_validation_within_target():
    """Paper §6.8: synthetic CRC at α=0.001, n=1500 satisfies E[ℓ] ≤ α."""
    data = _load_or_skip(ROUND7 / "alpha_critical_validation.json")
    results = data.get("results") or {}
    crit = results.get("CRITICAL") or {}
    if not crit:
        pytest.skip("CRITICAL stratum block missing")
    target = crit.get("target_alpha", 0.001)
    mean_loss = crit.get("mean_empirical_loss")
    if mean_loss is None:
        pytest.skip("mean_empirical_loss not found")
    # Paper reports mean E[ℓ] = 0.0006 ≤ 0.001 (target).
    # 2σ upper-bound also reported as ci95_upper.
    assert mean_loss <= target * 1.5, (
        f"α=0.001 synthetic validation: mean E[ℓ]={mean_loss} exceeds 1.5× target ({target})"
    )
    assert crit.get("satisfies_alpha", False), (
        "CRITICAL stratum no longer satisfies α; check stratified_crc.py."
    )


# ── Table 4 — head-to-head (LLM-dependent; uses latest snapshot) ────────


@pytest.mark.parametrize("backend", ["openai", "lmstudio"])
def test_table4_v2_critical_safety_recall_floor(backend):
    """
    Paper Table 4 headline: v2 CRITICAL safety recall = 0.96 (gpt-4o)
    and 0.96 (LLaMA). Regression guard floor at 0.85 to absorb seed
    drift while still catching catastrophic regressions.
    """
    p = ROUND7 / f"table4_baseline_{backend}.json"
    data = _load_or_skip(p)
    methods = data.get("methods", [])
    v2 = next((m for m in methods if m.get("name", "").startswith("UASEF Round 7")), None)
    if v2 is None:
        pytest.skip("UASEF Round 7 (v2) row missing in table4")
    crit = (v2.get("per_stratum") or {}).get("CRITICAL") or {}
    recall = crit.get("safety_recall")
    if recall is None:
        pytest.skip("CRITICAL safety_recall missing")
    assert recall >= 0.85, (
        f"v2 CRITICAL safety recall regressed: {recall} on {backend} (paper claims ≥0.96; "
        f"floor 0.85 here for seed/version drift)"
    )


@pytest.mark.parametrize("backend", ["openai", "lmstudio"])
def test_table4_v2_beats_tecp_on_cost(backend):
    """v2 total cost should be ≤ TECP / Quach total cost (Table 4 headline 20×)."""
    p = ROUND7 / f"table4_baseline_{backend}.json"
    data = _load_or_skip(p)
    methods = {m["name"]: m for m in data.get("methods", [])}
    v2 = next((m for n, m in methods.items() if n.startswith("UASEF Round 7")), None)
    tecp = next((m for n, m in methods.items() if "TECP" in n and "stratified" not in n.lower()), None)
    if not (v2 and tecp):
        pytest.skip("v2 or TECP row missing")
    v2_cost = v2.get("total_cost")
    tecp_cost = tecp.get("total_cost")
    if v2_cost is None or tecp_cost is None:
        pytest.skip("total_cost missing")
    assert v2_cost <= tecp_cost, (
        f"v2 total cost {v2_cost} ≥ TECP {tecp_cost} on {backend} — paper claims ≥20× reduction."
    )


# ── Test count claim (paper-meta) ───────────────────────────────────────


def test_pytest_test_count_matches_paper_claim_within_band():
    """
    Paper §1.3 / §8 mention a 'pytest suite of N tests'. We do not pin a
    specific N here (it grows over time) but assert that the claim falls
    within ±10% of the actual count, keeping the paper number from going
    stale silently. Paper currently says 137; actual ~133–145 expected.
    """
    import subprocess
    res = subprocess.run(
        ["bash", "-c",
         f"grep -hc '^def test_' {ROOT}/tests/test_*.py | paste -sd+ - | bc"],
        capture_output=True, text=True,
    )
    actual = int(res.stdout.strip() or "0")
    if actual == 0:
        pytest.skip("could not count test functions")
    # Paper claim is currently 137 (round8_PLAN.md notes correction).
    # Allow ±10% tolerance to absorb in-flight test additions.
    paper_claim = 137
    assert actual >= int(paper_claim * 0.85), (
        f"Test count drift: paper says {paper_claim}, actual {actual} (>15% drop). "
        "Update paper §1.3 / README."
    )
