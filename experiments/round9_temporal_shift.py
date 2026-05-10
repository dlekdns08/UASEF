"""
Round 9 R9.4 — Temporal distribution shift on MIMIC-IV
══════════════════════════════════════════════════════════════════════════════

Calibrate v2 on admissions from 2008-2014, evaluate on 2015-2019. Healthcare
practice (e.g., sepsis-3 criteria adopted 2016, electronic alerts deployment
mid-decade) creates a real temporal drift; the question is whether v2 holds
its per-stratum coverage guarantee under that drift.

산출: results/round9/temporal_shift.{json,md}
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import _MIMIC4_DEFAULT_PATH, _load_mimic4_jsonl
from models.uqm import UQM, compute_nonconformity_score
from models.stratified_crc import StratifiedConformalRiskControl
from models.model_interface import query_model

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}


def _split_by_year(cases, cal_range, test_range):
    cal, test = [], []
    for c in cases:
        try:
            year = int(c.meta_info.split("admit_year=")[1].split()[0])
        except Exception:
            year = 0
        if cal_range[0] <= year <= cal_range[1]:
            cal.append(c)
        elif test_range[0] <= year <= test_range[1]:
            test.append(c)
    return cal, test


def _scores(backend: str, cases):
    sys_prompt = UQM.SYSTEM_PROMPT
    s_, l_, st_ = [], [], []
    for c in cases:
        try:
            resp = query_model(backend, sys_prompt, c.question, temperature=0.0)
            sc = compute_nonconformity_score(resp)
        except Exception:
            continue
        s_.append(sc); l_.append(c.expected_escalate); st_.append(c.scenario_type.upper())
    return s_, l_, st_


def evaluate(backend: str, seed: int, cal_range, test_range,
             n_cal: int, n_test: int, verbose: bool):
    if verbose: print(f"\n── {backend} seed={seed} cal={cal_range} test={test_range} ──")
    if not _MIMIC4_DEFAULT_PATH.exists():
        raise FileNotFoundError(_MIMIC4_DEFAULT_PATH)
    all_cases = _load_mimic4_jsonl(_MIMIC4_DEFAULT_PATH, n=10**9, seed=seed)
    cal_pool, test_pool = _split_by_year(all_cases, cal_range, test_range)
    rng = random.Random(seed)
    rng.shuffle(cal_pool); rng.shuffle(test_pool)
    cal_cases = cal_pool[:n_cal]
    test_cases = test_pool[:n_test]
    if verbose:
        print(f"  cal pool {len(cal_pool)}, test pool {len(test_pool)} → cal {len(cal_cases)} test {len(test_cases)}")

    cs, cl, cst = _scores(backend, cal_cases)
    ts, tl, tst = _scores(backend, test_cases)

    crc = StratifiedConformalRiskControl(alphas=ALPHAS)
    crc.fit(cs, cl, cst)

    per_stratum = {}
    for s, alpha in ALPHAS.items():
        idx = [i for i, x in enumerate(tst) if x == s]
        if not idx:
            per_stratum[s] = None; continue
        lam = crc.threshold_for(s)
        n_pos = sum(tl[i] for i in idx)
        misses = sum(1 for i in idx if tl[i] and ts[i] <= lam)
        miss_rate = (misses / n_pos) if n_pos else None
        per_stratum[s] = {
            "n": len(idx), "n_pos": n_pos, "lambda": lam,
            "miss_rate": miss_rate,
            "violation_ratio": (miss_rate / alpha) if (miss_rate and alpha) else None,
        }
    return per_stratum


def write_md(report: dict, out_md: Path):
    lines = ["# Round 9 R9.4 — Temporal distribution shift on MIMIC-IV\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Calibration years: {report['cal_range']}, test years: {report['test_range']}")
    lines.append(f"- Backends: {', '.join(report['backends'])}\n")
    for backend, agg in report["per_backend"].items():
        lines.append(f"## backend = {backend}\n")
        lines.append("| stratum | α | mean miss | std | violation× |")
        lines.append("| --- | --- | --- | --- | --- |")
        for s, a in agg.items():
            if a is None:
                lines.append(f"| {s} | {ALPHAS[s]} | — | — | — |"); continue
            lines.append(
                f"| {s} | {ALPHAS[s]} | {a['miss_mean']:.4f} | {a['miss_std']:.4f} | {a['violation_ratio']:.2f}× |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-years", nargs=2, type=int, default=[2008, 2014])
    ap.add_argument("--test-years", nargs=2, type=int, default=[2015, 2019])
    ap.add_argument("--n-cal", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--backends", nargs="+", default=["openai", "lmstudio"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round9" / "temporal_shift")
    args = ap.parse_args()

    report = {
        "timestamp": datetime.now().isoformat(),
        "cal_range": args.cal_years, "test_range": args.test_years,
        "n_cal": args.n_cal, "n_test": args.n_test,
        "seeds": args.seeds, "backends": args.backends,
        "per_seed": {}, "per_backend": {},
    }
    for backend in args.backends:
        per_seed = []
        for seed in args.seeds:
            try:
                r = evaluate(backend, seed, args.cal_years, args.test_years,
                             args.n_cal, args.n_test, True)
            except FileNotFoundError as e:
                print(f"[R9.4] preprocessed MIMIC-IV missing: {e}"); sys.exit(2)
            per_seed.append(r)
        report["per_seed"][backend] = per_seed

        # aggregate per stratum across seeds
        agg = {}
        for s in ALPHAS:
            misses = [r[s]["miss_rate"] for r in per_seed if r.get(s) and r[s]["miss_rate"] is not None]
            if not misses:
                agg[s] = None; continue
            mean = st.mean(misses); std_ = st.stdev(misses) if len(misses) > 1 else 0.0
            agg[s] = {
                "n_seeds": len(misses), "miss_mean": mean, "miss_std": std_,
                "violation_ratio": mean / ALPHAS[s],
            }
        report["per_backend"][backend] = agg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
