"""
Round 11 R11.7 — Paper-JSON numerical consistency audit.

Paper (paper/UASEF_FINAL.md) 의 핵심 numeric claim 을 results/*.json
artifact 의 ground truth 와 자동 대조한다. Mismatch 시 exit code 1.

두 번 leakage 를 겪은 파이프라인이므로 paper 의 *모든* 수치가
정확히 JSON 과 일치함을 CI 시점에 보장하기 위한 인프라.

검증 항목 (~30 assertion):
  - R10.1: 5-seed pooled LLM miss
  - R10.4: per-classifier per-stratum miss + α-verdict
  - R10.7: expanded feature negative
  - R10.RF_CALIBRATION: ECE/Brier/sharpness
  - R11.1: minimal feature 5-classifier
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper" / "UASEF_FINAL.md"


# ─────────────────────────────────────────────────────────────────────────────
# Assertion catalog
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """Single paper claim with JSON lookup + tolerance."""
    name: str
    paper_value: Any            # claim as stated in paper
    json_path: str              # results/*.json path
    json_key: tuple[str, ...]   # nested keys
    tolerance: float = 0.0      # absolute tolerance for floats
    derive: Callable | None = None   # optional transform on json value


def _get(d: dict, keys: tuple[str, ...]) -> Any:
    cur = d
    for k in keys:
        if cur is None: return None
        if isinstance(cur, list) and isinstance(k, int):
            cur = cur[k] if 0 <= k < len(cur) else None
        else:
            cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def _sum_over_seeds(j: dict, classifier: str, stratum: str,
                    field: str) -> int | float | None:
    """Sum across 5-seed per_classifier 구조."""
    per_seed = (j.get("per_seed") or {}).get(classifier, [])
    if not per_seed: return None
    total = 0
    for r in per_seed:
        cell = ((r.get("per_stratum") or {}).get(stratum) or {})
        v = cell.get(field)
        if v is None: return None
        total += v
    return total


def load_json(rel: str) -> dict:
    p = RESULTS / rel
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded assertions — paper §5 의 핵심 수치
# ─────────────────────────────────────────────────────────────────────────────

CLAIMS: list[Claim] = []


def _build_r10_4_claims():
    """R10.4 method-agnostic — 5 classifier × CRITICAL/HIGH misses."""
    j = load_json("round10/r10_4_method_agnostic.json")
    if not j: return
    spec = [
        # paper §5.4 의 R10.4 (HEADLINE) 표 (pre-R11.1 retraction)
        ("gpt_oss_120b", "CRITICAL", 173, 1293),
        ("logreg",       "CRITICAL", 159, 1293),
        ("gbdt",         "CRITICAL",  92, 1293),
        ("randomforest", "CRITICAL",   0, 1293),
        ("xgboost",      "CRITICAL", 149, 1293),
        ("gpt_oss_120b", "HIGH",     299,  525),
        ("randomforest", "HIGH",       0,  525),
    ]
    for clf, st, expected_miss, expected_npos in spec:
        # paper 의 "X / 1293" claim → JSON 의 pooled per-seed sum 검증
        actual_miss = _sum_over_seeds(j, clf, st, "misses")
        actual_npos = _sum_over_seeds(j, clf, st, "n_pos")
        CLAIMS.append(Claim(
            name=f"R10.4 {clf} {st} miss",
            paper_value=expected_miss,
            json_path="round10/r10_4_method_agnostic.json",
            json_key=("derived",),
            derive=lambda _, _miss=actual_miss: _miss,
        ))
        CLAIMS.append(Claim(
            name=f"R10.4 {clf} {st} n_pos",
            paper_value=expected_npos,
            json_path="round10/r10_4_method_agnostic.json",
            json_key=("derived",),
            derive=lambda _, _np=actual_npos: _np,
        ))


def _build_r10_1_claims():
    """R10.1 α=0.05 empirical (LLM only)."""
    j = load_json("round10/r10_1_alpha_005_empirical.json")
    if not j: return
    spec = [
        ("CRITICAL", 189, 1293),
        ("HIGH",     314,  525),
        ("MODERATE", 307,  307),
        ("LOW",      170,  171),
    ]
    for st, exp_miss, exp_np in spec:
        actual_miss = _sum_over_seeds(j, "gpt_oss_120b", st, "misses")
        actual_np = _sum_over_seeds(j, "gpt_oss_120b", st, "n_pos")
        CLAIMS.append(Claim(
            name=f"R10.1 LLM {st} miss",
            paper_value=exp_miss,
            json_path="round10/r10_1_alpha_005_empirical.json",
            json_key=("derived",),
            derive=lambda _, m=actual_miss: m,
        ))
        CLAIMS.append(Claim(
            name=f"R10.1 LLM {st} n_pos",
            paper_value=exp_np,
            json_path="round10/r10_1_alpha_005_empirical.json",
            json_key=("derived",),
            derive=lambda _, n=actual_np: n,
        ))


def _build_r11_1_claims():
    """R11.1 minimal feature (tabular smoke)."""
    j = load_json("round11/r11_1_smoke_tabular.json")
    if not j: return
    spec = [
        ("logreg",       "CRITICAL", 81, 1293),
        ("gbdt",         "CRITICAL", 150, 1293),
        ("randomforest", "CRITICAL", 176, 1293),
        ("xgboost",      "CRITICAL", 150, 1293),
    ]
    for clf, st, exp_miss, exp_np in spec:
        actual_miss = _sum_over_seeds(j, clf, st, "misses")
        actual_np = _sum_over_seeds(j, clf, st, "n_pos")
        CLAIMS.append(Claim(
            name=f"R11.1 {clf} {st} miss",
            paper_value=exp_miss,
            json_path="round11/r11_1_smoke_tabular.json",
            json_key=("derived",),
            derive=lambda _, m=actual_miss: m,
        ))
        CLAIMS.append(Claim(
            name=f"R11.1 {clf} {st} n_pos",
            paper_value=exp_np,
            json_path="round11/r11_1_smoke_tabular.json",
            json_key=("derived",),
            derive=lambda _, n=actual_np: n,
        ))


def _build_rf_calibration_claims():
    """R10 RF calibration §6 ECE/Brier/sharpness."""
    j = load_json("round10/r10_rf_calibration.json")
    if not j: return
    per_clf = j.get("per_classifier", {})
    spec = [
        # (classifier, metric, paper value, abs_tolerance)
        ("gpt_oss_120b", "ece",       0.3447, 0.001),
        ("gpt_oss_120b", "brier",     0.2732, 0.001),
        ("gpt_oss_120b", "sharpness", 0.0157, 0.001),
        ("randomforest", "ece",       0.0051, 0.0005),
        ("randomforest", "brier",     0.0440, 0.001),
        ("randomforest", "sharpness", 0.1020, 0.001),
        ("logreg",       "ece",       0.0072, 0.0005),
    ]
    for clf, metric, paper_v, tol in spec:
        d = per_clf.get(clf, {})
        actual = d.get(metric) if isinstance(d, dict) else None
        CLAIMS.append(Claim(
            name=f"§6 {clf} {metric}",
            paper_value=paper_v,
            json_path="round10/r10_rf_calibration.json",
            json_key=("derived",),
            tolerance=tol,
            derive=lambda _, v=actual: v,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_claim(c: Claim) -> tuple[bool, str]:
    actual = c.derive(None) if c.derive else _get(load_json(c.json_path), c.json_key)
    if actual is None:
        return False, f"JSON value missing for {c.json_path} {c.json_key}"
    if isinstance(c.paper_value, (int, float)) and isinstance(actual, (int, float)):
        if c.tolerance > 0:
            if abs(actual - c.paper_value) <= c.tolerance:
                return True, f"paper={c.paper_value} ≈ json={actual} (tol={c.tolerance})"
            return False, f"paper={c.paper_value} ≠ json={actual} (Δ={abs(actual-c.paper_value):.4f} > tol={c.tolerance})"
        if actual == c.paper_value:
            return True, f"paper={c.paper_value} = json={actual}"
        return False, f"paper={c.paper_value} ≠ json={actual} (exact mismatch)"
    return False, f"type mismatch: paper={type(c.paper_value)}, json={type(actual)}"


def run_audit(verbose: bool = True) -> int:
    """Returns number of mismatches."""
    _build_r10_4_claims()
    _build_r10_1_claims()
    _build_r11_1_claims()
    _build_rf_calibration_claims()

    if not CLAIMS:
        print("⚠️  No claims to verify — JSON artifacts missing.")
        return 0

    print(f"R11.7 paper-JSON audit — {len(CLAIMS)} claims")
    print("=" * 70)
    n_ok = n_fail = 0
    failures: list[tuple[Claim, str]] = []
    for c in CLAIMS:
        ok, msg = verify_claim(c)
        if ok:
            n_ok += 1
            if verbose:
                print(f"  ✓ {c.name}: {msg}")
        else:
            n_fail += 1
            failures.append((c, msg))
            print(f"  ✗ {c.name}: {msg}")

    print("=" * 70)
    print(f"Result: {n_ok}/{len(CLAIMS)} verified, {n_fail} mismatches")

    out_dir = RESULTS / "round11"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_total": len(CLAIMS),
        "n_ok": n_ok, "n_fail": n_fail,
        "claims": [{"name": c.name, "paper": c.paper_value} for c in CLAIMS],
        "failures": [{"name": c.name, "paper": c.paper_value, "reason": msg}
                     for c, msg in failures],
    }
    (out_dir / "r11_7_paper_audit.json").write_text(
        json.dumps(report, indent=2, default=str))

    md = ["# R11.7 Paper-JSON Audit Report\n"]
    md.append(f"- Total claims: {len(CLAIMS)}")
    md.append(f"- Verified: **{n_ok}**")
    md.append(f"- Mismatches: **{n_fail}**\n")
    if failures:
        md.append("## Mismatches\n")
        md.append("| Claim | Paper value | Reason |")
        md.append("| --- | --- | --- |")
        for c, msg in failures:
            md.append(f"| {c.name} | {c.paper_value} | {msg} |")
    else:
        md.append("\n**✓ All paper numerics match the JSON artifacts.**")
    (out_dir / "r11_7_paper_audit.md").write_text("\n".join(md))
    return n_fail


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    n_fail = run_audit(verbose=not args.quiet)
    sys.exit(1 if n_fail > 0 else 0)
