"""
Round 10 R10.3 — Distribution shift mitigation strategies.

Round 9 R9.3/R9.4 가 violation 을 *검출* 했다면 R10.3 는 그것을 *완화*.

세 strategy:
  A. online_recal — temporal rolling-window recalibration
  B. kmm — Kernel Mean Matching weighted CP
  C. group_conditional — (stratum × specialty) joint partition CRC

각 strategy 의 Round 9 naive baseline 대비 violation × 감소 보고.

산출: results/round10/r10_3_mitigation.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import _load_mimic4_jsonl
from experiments.metrics_utils import clopper_pearson_upper
from experiments.round10_method_agnostic import (
    _llm_scores, _make_classifier, _feature_vector, _attach_r10_features,
    _subject_id,
)
from models.stratified_crc import StratifiedConformalRiskControl

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"


def _parse_year_group(meta: str):
    for tok in (meta or "").split():
        if tok.startswith("anchor_year_group="):
            return tok.split("=", 1)[1].replace("_", " ")
    return None


def _score_cases_logreg(cases):
    """LogReg 로 빠르게 score 산출 (LLM 호출 부담 회피)."""
    clf = _make_classifier("logreg")
    X = [_feature_vector(c) for c in cases]
    y = [bool(c.expected_escalate) for c in cases]
    if not clf.fit(X, y):
        return [0.0] * len(cases)
    return [clf.score(x) for x in X]


def _eval_threshold(scores, labels, strata, thresholds: dict, alphas: dict):
    """Per-stratum miss / over-esc / violation × 계산."""
    per_stratum = {}
    for s, alpha in alphas.items():
        idx = [i for i, x in enumerate(strata) if x == s]
        if not idx:
            per_stratum[s] = None; continue
        lam = thresholds.get(s, 0.0)
        n_pos = sum(labels[i] for i in idx)
        misses = sum(1 for i in idx if labels[i] and scores[i] <= lam)
        miss_rate = (misses / n_pos) if n_pos else None
        per_stratum[s] = {
            "n": len(idx), "n_pos": n_pos, "misses": misses,
            "miss_rate": miss_rate,
            "violation_x": (miss_rate / alpha) if miss_rate is not None else None,
        }
    return per_stratum


# ── Strategy A: Online rolling-window recalibration ────────────────────────

def _strategy_online_recal(cases, seed: int):
    """anchor_year_group 으로 시간 split, 각 test era 마다 직전 cal era 로 재학습."""
    # Group by era
    by_era = defaultdict(list)
    for c in cases:
        era = _parse_year_group(c.meta_info)
        if era:
            by_era[era].append(c)
    eras_sorted = sorted(by_era.keys())
    if len(eras_sorted) < 3:
        return {"error": "insufficient eras for rolling recal"}

    # cal: era[-2], test: era[-1]
    cal_cases = by_era[eras_sorted[-2]]
    test_cases = by_era[eras_sorted[-1]]
    rng = random.Random(seed)
    rng.shuffle(cal_cases); rng.shuffle(test_cases)

    cal_scores = _score_cases_logreg(cal_cases)
    test_scores = _score_cases_logreg(test_cases)
    cal_labels = [bool(c.expected_escalate) for c in cal_cases]
    cal_strata = [(c.scenario_type or "").upper() for c in cal_cases]
    test_labels = [bool(c.expected_escalate) for c in test_cases]
    test_strata = [(c.scenario_type or "").upper() for c in test_cases]

    crc = StratifiedConformalRiskControl(alphas=ALPHAS)
    crc.fit(cal_scores, cal_labels, cal_strata)
    return _eval_threshold(test_scores, test_labels, test_strata,
                            {s: crc.threshold_for(s) for s in ALPHAS}, ALPHAS)


# ── Strategy B: Kernel Mean Matching weighted CP ───────────────────────────

def _kmm_weights(source_scores, target_scores, lam: float = 1.0):
    """
    KMM (Huang 2007) 의 단순 implementation — Gaussian kernel.
    Full QP 대신 row-stochastic approximation 사용 (no cvxopt 의존성).
    """
    n_s = len(source_scores)
    n_t = len(target_scores)
    if n_s == 0 or n_t == 0:
        return [1.0] * n_s
    # bandwidth: median heuristic
    all_s = source_scores + target_scores
    sd = st.stdev(all_s) if len(all_s) > 1 else 1.0
    h = max(1.06 * sd * (len(all_s) ** (-1/5)), 1e-3)

    weights = []
    for s_i in source_scores:
        # ratio ≈ density at target / density at source
        kt = sum(math.exp(-((s_i - t) ** 2) / (2 * h * h)) for t in target_scores) / n_t
        ks = sum(math.exp(-((s_i - s_j) ** 2) / (2 * h * h)) for s_j in source_scores) / n_s
        w = (kt / (ks + 1e-9)) if ks > 0 else 1.0
        weights.append(max(min(w, 100.0), 1e-3))
    return weights


def _weighted_quantile(values, weights, q):
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs) or 1.0
    target = q * total
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return v
    return pairs[-1][0]


def _strategy_kmm(cases, seed: int):
    """cardiology source vs internal_medicine target — KMM weighted CP."""
    src = [c for c in cases if c.specialty == "cardiology"]
    tgt = [c for c in cases if c.specialty == "internal_medicine"]
    rng = random.Random(seed)
    rng.shuffle(src); rng.shuffle(tgt)
    src = src[:300]
    tgt = tgt[:300]
    if not src or not tgt:
        return {"error": "no specialty data"}

    src_scores = _score_cases_logreg(src)
    src_labels = [bool(c.expected_escalate) for c in src]
    src_strata = [(c.scenario_type or "").upper() for c in src]
    tgt_scores = _score_cases_logreg(tgt)
    tgt_labels = [bool(c.expected_escalate) for c in tgt]
    tgt_strata = [(c.scenario_type or "").upper() for c in tgt]

    # KMM weights for cardiology source
    weights = _kmm_weights(src_scores, tgt_scores)

    per_stratum = {}
    for s, alpha in ALPHAS.items():
        s_idx_pos = [i for i, x in enumerate(src_strata)
                     if x == s and src_labels[i]]
        if len(s_idx_pos) < 5:
            per_stratum[s] = None; continue
        stratum_scores = [src_scores[i] for i in s_idx_pos]
        stratum_weights = [weights[i] for i in s_idx_pos]
        lam_w = _weighted_quantile(stratum_scores, stratum_weights, alpha)
        t_idx = [i for i, x in enumerate(tgt_strata) if x == s]
        n_pos = sum(tgt_labels[i] for i in t_idx)
        misses = sum(1 for i in t_idx if tgt_labels[i] and tgt_scores[i] <= lam_w)
        miss_rate = (misses / n_pos) if n_pos else None
        per_stratum[s] = {
            "n": len(t_idx), "n_pos": n_pos, "misses": misses,
            "miss_rate": miss_rate,
            "violation_x": (miss_rate / alpha) if miss_rate is not None else None,
        }
    return per_stratum


# ── Strategy C: Group-conditional CRC ──────────────────────────────────────

def _strategy_group_conditional(cases, seed: int):
    """(stratum × specialty) joint partition — fallback to stratum-marginal."""
    rng = random.Random(seed)
    rng.shuffle(cases)
    cut = int(len(cases) * 0.8)
    cal = cases[:cut]
    test = cases[cut:]
    cal_scores = _score_cases_logreg(cal)
    cal_labels = [bool(c.expected_escalate) for c in cal]
    cal_strata = [(c.scenario_type or "").upper() for c in cal]
    cal_specs = [c.specialty or "internal_medicine" for c in cal]
    test_scores = _score_cases_logreg(test)
    test_labels = [bool(c.expected_escalate) for c in test]
    test_strata = [(c.scenario_type or "").upper() for c in test]
    test_specs = [c.specialty or "internal_medicine" for c in test]

    # cell-level threshold
    cell_thresholds: dict = {}
    fallback_crc = StratifiedConformalRiskControl(alphas=ALPHAS)
    fallback_crc.fit(cal_scores, cal_labels, cal_strata)

    for s in ALPHAS:
        cell_thresholds[s] = {}
        for spec in set(cal_specs):
            idx = [i for i, (cs, sp) in enumerate(zip(cal_strata, cal_specs))
                   if cs == s and sp == spec]
            if len(idx) < 20:  # n_min not met → fallback
                cell_thresholds[s][spec] = fallback_crc.threshold_for(s)
                continue
            # local CRC quantile on cell
            cell_scores = sorted([cal_scores[i] for i in idx])
            q_idx = max(0, math.ceil((1 - ALPHAS[s]) * len(cell_scores)) - 1)
            cell_thresholds[s][spec] = cell_scores[q_idx]

    per_stratum = {}
    for s, alpha in ALPHAS.items():
        idx = [i for i, x in enumerate(test_strata) if x == s]
        if not idx:
            per_stratum[s] = None; continue
        n_pos = sum(test_labels[i] for i in idx)
        misses = 0
        for i in idx:
            if test_labels[i]:
                spec = test_specs[i]
                lam = cell_thresholds[s].get(spec, fallback_crc.threshold_for(s))
                if test_scores[i] <= lam:
                    misses += 1
        miss_rate = (misses / n_pos) if n_pos else None
        per_stratum[s] = {
            "n": len(idx), "n_pos": n_pos, "misses": misses,
            "miss_rate": miss_rate,
            "violation_x": (miss_rate / alpha) if miss_rate is not None else None,
        }
    return per_stratum


# ── Main ─────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "online_recal": _strategy_online_recal,
    "kmm": _strategy_kmm,
    "group_conditional": _strategy_group_conditional,
}


def write_md(report: dict, out_md: Path):
    lines = ["# Round 10 R10.3 — Distribution shift mitigation\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Strategies: {', '.join(report['strategies'])}")
    lines.append(f"- Seeds: {report['seeds']}\n")
    for strat, agg in report["per_strategy"].items():
        lines.append(f"## strategy = {strat}\n")
        if isinstance(agg, dict) and agg.get("error"):
            lines.append(f"⚠️ {agg['error']}\n"); continue
        lines.append("| stratum | α | mean miss | mean violation × |")
        lines.append("| --- | --- | --- | --- |")
        for s, a in agg.items():
            if a is None or a.get("error"):
                lines.append(f"| {s} | {ALPHAS[s]} | — | — |"); continue
            mr = a.get("miss_rate_mean", "—")
            vx = a.get("violation_x_mean", "—")
            mrs = f"{mr:.4f}" if isinstance(mr, float) else "—"
            vxs = f"{vx:.2f}×" if isinstance(vx, float) else "—"
            lines.append(f"| {s} | {ALPHAS[s]} | {mrs} | {vxs} |")
        lines.append("")
    out_md.write_text("\n".join(lines))


def _aggregate_strategy(per_seed: list) -> dict:
    out = {}
    for s in ALPHAS:
        vals_miss = []
        vals_viol = []
        for r in per_seed:
            if isinstance(r, dict) and r.get("error"):
                continue
            cell = (r or {}).get(s)
            if cell is None:
                continue
            if cell.get("miss_rate") is not None:
                vals_miss.append(cell["miss_rate"])
            if cell.get("violation_x") is not None:
                vals_viol.append(cell["violation_x"])
        if not vals_miss:
            out[s] = None; continue
        out[s] = {
            "n_seeds": len(vals_miss),
            "miss_rate_mean": st.mean(vals_miss),
            "miss_rate_std": st.stdev(vals_miss) if len(vals_miss) > 1 else 0.0,
            "violation_x_mean": st.mean(vals_viol) if vals_viol else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+",
                    default=["online_recal", "kmm", "group_conditional"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--backends", nargs="+", default=["lmstudio"])  # for runner CLI parity
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "r10_3_mitigation")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"[R10.3] preprocessed JSONL 미존재: {args.jsonl}")

    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=42)
    _attach_r10_features(cases, args.jsonl)
    print(f"[R10.3] loaded {len(cases)} cases")

    report = {
        "timestamp": datetime.now().isoformat(),
        "strategies": args.strategies,
        "seeds": args.seeds,
        "per_seed": {}, "per_strategy": {},
    }
    for strat in args.strategies:
        fn = STRATEGIES.get(strat)
        if fn is None:
            report["per_strategy"][strat] = {"error": f"unknown strategy {strat}"}
            continue
        per_seed = []
        for seed in args.seeds:
            print(f"\n── strategy={strat} seed={seed} ──", flush=True)
            try:
                r = fn(cases, seed)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"  [error] {r['error']}")
            per_seed.append(r)
        report["per_seed"][strat] = per_seed
        report["per_strategy"][strat] = _aggregate_strategy(per_seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
