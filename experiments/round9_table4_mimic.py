"""
Round 9 R9.2 — Table 4-MIMIC head-to-head on real EHR data
══════════════════════════════════════════════════════════════════════════════

MedAbstain 기반 round7 Table 4 와 동일한 8 method 비교를 MIMIC-IV CRITICAL/
HIGH/MODERATE/LOW stratum (real outcome 라벨) 에서 반복.

stratum 결정은 SPECIALTY_TO_STRATUM 가 아니라 **case.scenario_type 의 값
(preprocessing 산출 라벨)** 을 직접 사용한다 — MIMIC-IV 의 stratum 은
임상 outcome 으로부터 derive 되었기 때문.

산출: results/round9/table4_mimic.{json,md}
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import load_mimic4_by_stratum
from models.uqm import UQM, ConformalCalibrator, compute_nonconformity_score
from models.stratified_crc import StratifiedConformalRiskControl
from models.cost_aware_calibration import DEFAULT_COST_MATRIX
from models.model_interface import query_model

from experiments.baselines.tecp import TECPBaseline
from experiments.baselines.tecp_stratified import TECPStratifiedBaseline
from experiments.baselines.quach2024 import Quach2024Baseline
from experiments.baselines.semantic_entropy import SemanticEntropyBaseline
from experiments.baselines.cost_sensitive import CostSensitiveBaseline
from experiments.baselines.uasef_v1_cost import UASEFv1CostAwareBaseline

ALPHAS_TABLE4 = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}


def _collect_scores(backend: str, cases: list, verbose: bool = False):
    import time as _time
    sys_prompt = UQM.SYSTEM_PROMPT
    scores, labels, strata = [], [], []
    skipped = 0
    t_start = _time.perf_counter()
    log_every = max(1, len(cases) // 40)
    for i, case in enumerate(cases):
        # MIMIC-IV case 는 PHI taint — env guard 가 외부 API 송신 차단.
        phi_taint = (case.source or "").startswith("mimic4")
        try:
            resp = query_model(backend, sys_prompt, case.question, temperature=0.0,
                                phi_taint=phi_taint)
            sc = compute_nonconformity_score(resp)
        except Exception as e:
            skipped += 1
            if verbose and skipped <= 3:
                print(f"  [skip {i}] {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        scores.append(sc)
        labels.append(case.expected_escalate)
        strata.append(case.scenario_type.upper())
        if verbose and ((i + 1) % log_every == 0 or i + 1 == len(cases)):
            elapsed = _time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(cases) - i - 1) / rate if rate > 0 else 0
            print(f"  [{backend}] {i+1}/{len(cases)} ({rate:.2f}/s, ETA {eta/60:.1f}min, skipped={skipped})", flush=True)
    if verbose and skipped:
        print(f"  total skipped: {skipped}/{len(cases)} cases", flush=True)
    return scores, labels, strata


def _split_cal_test(buckets, n_cal_per: int, n_test_per: int, seed: int):
    import random
    rng = random.Random(seed)
    cal, test = [], []
    for s, cs in buckets.items():
        rng.shuffle(cs)
        cal.extend(cs[:n_cal_per])
        test.extend(cs[n_cal_per:n_cal_per + n_test_per])
    return cal, test


def evaluate_predictor(name, predictor_fn, scores, labels, strata) -> dict:
    per_stratum = {}
    total_cost = 0.0
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        idx = [i for i, s in enumerate(strata) if s == stratum]
        if not idx:
            per_stratum[stratum] = {"n": 0, "skipped": True}; continue
        n_pos = sum(labels[i] for i in idx)
        n_neg = sum(1 for i in idx if not labels[i])
        tp = sum(labels[i] and predictor_fn(scores[i]) for i in idx)
        fn = sum(labels[i] and not predictor_fn(scores[i]) for i in idx)
        fp = sum((not labels[i]) and predictor_fn(scores[i]) for i in idx)
        recall = tp / (tp + fn) if (tp + fn) else None
        over = fp / n_neg if n_neg else None
        cm = DEFAULT_COST_MATRIX[stratum]
        cost = cm["miss"] * fn + cm["over_esc"] * fp
        total_cost += cost
        per_stratum[stratum] = {"n": len(idx), "n_pos": n_pos,
                                 "tp": tp, "fn": fn, "fp": fp,
                                 "safety_recall": recall, "over_esc_rate": over,
                                 "cost": cost}
    return {"name": name, "per_stratum": per_stratum, "total_cost": total_cost}


def evaluate_stratum_aware(name, threshold_for_stratum, scores, labels, strata) -> dict:
    """For predictors that need a per-stratum threshold."""
    per_stratum = {}
    total_cost = 0.0
    for stratum in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        idx = [i for i, s in enumerate(strata) if s == stratum]
        if not idx:
            per_stratum[stratum] = {"n": 0, "skipped": True}; continue
        thr = threshold_for_stratum(stratum)
        n_pos = sum(labels[i] for i in idx)
        n_neg = sum(1 for i in idx if not labels[i])
        tp = sum(labels[i] and scores[i] > thr for i in idx)
        fn = sum(labels[i] and not (scores[i] > thr) for i in idx)
        fp = sum((not labels[i]) and scores[i] > thr for i in idx)
        recall = tp / (tp + fn) if (tp + fn) else None
        over = fp / n_neg if n_neg else None
        cm = DEFAULT_COST_MATRIX[stratum]
        cost = cm["miss"] * fn + cm["over_esc"] * fp
        total_cost += cost
        per_stratum[stratum] = {"n": len(idx), "n_pos": n_pos,
                                 "tp": tp, "fn": fn, "fp": fp,
                                 "safety_recall": recall, "over_esc_rate": over,
                                 "cost": cost, "threshold": thr}
    return {"name": name, "per_stratum": per_stratum, "total_cost": total_cost}


def run_one_seed(backend: str, seed: int, n_cal_per: int, n_test_per: int, alpha: float, verbose: bool):
    if verbose:
        print(f"\n── {backend} seed={seed} ──")
    buckets = load_mimic4_by_stratum(n_per_stratum=n_cal_per + n_test_per, seed=seed, verbose=False)
    cal_cases, test_cases = _split_cal_test(buckets, n_cal_per, n_test_per, seed)
    if verbose:
        print(f"  cal={len(cal_cases)} test={len(test_cases)}")

    cal_scores, cal_labels, cal_strata = _collect_scores(backend, cal_cases, verbose)
    test_scores, test_labels, test_strata = _collect_scores(backend, test_cases, verbose)

    methods = []

    # --- single-α global baselines ---
    tecp = TECPBaseline(alpha=alpha)
    tecp.fit(cal_scores, cal_labels)
    methods.append(evaluate_predictor("TECP (Xu & Lu 2025)", tecp.predict,
                                       test_scores, test_labels, test_strata))

    q = Quach2024Baseline(alpha=alpha)
    q.fit(cal_scores, cal_labels)
    methods.append(evaluate_predictor("Quach 2024 CLM", q.predict,
                                       test_scores, test_labels, test_strata))

    se = SemanticEntropyBaseline(alpha=alpha)
    se.fit(cal_scores, cal_labels)
    methods.append(evaluate_predictor("Semantic Entropy (Farquhar Nature 2024)", se.predict,
                                       test_scores, test_labels, test_strata))

    # --- UASEF Round 6 (heuristic multipliers) ---
    cal_global = ConformalCalibrator(alpha=alpha, strict=False)
    cal_global.fit(cal_scores)
    multipliers = {"CRITICAL": 0.60, "HIGH": 0.75, "MODERATE": 1.00, "LOW": 1.30}
    def r6_thr(s): return cal_global.threshold * multipliers[s]
    methods.append(evaluate_stratum_aware("UASEF Round 6 (heuristic multiplier)", r6_thr,
                                           test_scores, test_labels, test_strata))

    # --- UASEF Round 7 v2 (Stratified CRC, threshold-only) ---
    # round7_table4_baseline.py 의 reference 구현과 동일: cost-aware sweep 은
    # Round 7 paper 의 §6.4 (Table 4) 컨텍스트에서 별도로 보고되며, 본 R9.2
    # 헤드라인 비교는 순수 stratified-CRC threshold 만 사용.
    crc = StratifiedConformalRiskControl(alphas=ALPHAS_TABLE4)
    crc.fit(cal_scores, cal_labels, cal_strata)
    def r7_thr(s): return crc.threshold_for(s)
    methods.append(evaluate_stratum_aware(
        "UASEF Round 7 (Stratified CRC + MTC + Cost-Aware)", r7_thr,
        test_scores, test_labels, test_strata))

    # --- TECP-stratified ablation ---
    tecp_s = TECPStratifiedBaseline(alphas=ALPHAS_TABLE4)
    tecp_s.fit(cal_scores, cal_labels, cal_strata)
    methods.append(evaluate_stratum_aware(
        "TECP-stratified (this work, Round 9 ablation)",
        lambda s: tecp_s.threshold_for(s),
        test_scores, test_labels, test_strata))

    # --- Cost-Sensitive single-α ablation ---
    cs = CostSensitiveBaseline(alpha=alpha, miss_cost=1000, fp_cost=1.0)
    cs.fit(cal_scores, cal_labels)
    methods.append(evaluate_predictor("Cost-Sensitive single-α (this work, Round 9 ablation)",
                                       cs.predict, test_scores, test_labels, test_strata))

    # --- UASEF v1-cost-aware ablation ---
    v1c = UASEFv1CostAwareBaseline(alpha=alpha)
    v1c.fit(cal_scores, cal_labels, cal_strata)
    methods.append(evaluate_stratum_aware("UASEF v1-cost-aware (this work, Round 9 ablation)",
                                           lambda s: v1c.threshold_for(s),
                                           test_scores, test_labels, test_strata))

    return {"seed": seed, "n_cal": len(cal_cases), "n_test": len(test_cases),
            "methods": methods}


def aggregate(per_seed: list[dict]) -> dict:
    """Mean ± std across seeds for total_cost and CRITICAL recall."""
    by_method: dict = {}
    for r in per_seed:
        for m in r["methods"]:
            name = m["name"]
            by_method.setdefault(name, {"total_cost": [], "critical_recall": []})
            by_method[name]["total_cost"].append(m["total_cost"])
            cr = m["per_stratum"].get("CRITICAL", {}).get("safety_recall")
            if cr is not None:
                by_method[name]["critical_recall"].append(cr)
    summary = {}
    for name, d in by_method.items():
        c = d["total_cost"]; r = d["critical_recall"]
        summary[name] = {
            "total_cost_mean": round(st.mean(c), 1) if c else None,
            "total_cost_std":  round(st.stdev(c), 1) if len(c) > 1 else 0.0,
            "critical_recall_mean": round(st.mean(r), 4) if r else None,
            "critical_recall_std":  round(st.stdev(r), 4) if len(r) > 1 else 0.0,
        }
    return summary


def write_md(report: dict, out_md: Path):
    lines = ["# Round 9 R9.2 — Table 4-MIMIC (real-EHR head-to-head)\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Seeds: {report['seeds']}")
    lines.append(f"- Backends: {', '.join(report['backends'])}")
    lines.append(f"- α (Table 4-MIMIC): {report['alpha']}\n")
    for backend, agg in report["per_backend"].items():
        lines.append(f"## backend = {backend}\n")
        lines.append("| Method | CRITICAL Recall (mean ± std) | Total Cost (mean ± std) |")
        lines.append("| --- | --- | --- |")
        for name, m in agg.items():
            cr = (f"{m['critical_recall_mean']:.4f} ± {m['critical_recall_std']:.4f}"
                  if m['critical_recall_mean'] is not None else "—")
            tc = (f"{m['total_cost_mean']:.1f} ± {m['total_cost_std']:.1f}"
                  if m['total_cost_mean'] is not None else "—")
            lines.append(f"| {name} | {cr} | {tc} |")
        lines.append("")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal-per-stratum", type=int, default=200)
    ap.add_argument("--n-test-per-stratum", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--backends", nargs="+", default=["lmstudio"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round9" / "table4_mimic")
    args = ap.parse_args()

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_cal_per_stratum": args.n_cal_per_stratum,
        "n_test_per_stratum": args.n_test_per_stratum,
        "alpha": args.alpha,
        "seeds": args.seeds,
        "backends": args.backends,
        "per_seed": {},
        "per_backend": {},
    }
    for backend in args.backends:
        per_seed = []
        for seed in args.seeds:
            try:
                r = run_one_seed(backend, seed,
                                 args.n_cal_per_stratum, args.n_test_per_stratum,
                                 args.alpha, verbose=True)
                per_seed.append(r)
            except FileNotFoundError as e:
                print(f"[R9.2] preprocessed MIMIC-IV missing: {e}")
                sys.exit(2)
        report["per_seed"][backend] = per_seed
        report["per_backend"][backend] = aggregate(per_seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
