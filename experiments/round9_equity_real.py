"""
Round 9 R9.5 — Demographic equity audit on MIMIC-IV
══════════════════════════════════════════════════════════════════════════════

paper supplementary §I 의 합성 equity audit 을 실 데이터 버전으로 대체.
race / sex / age_bucket × stratum × miss_rate 를 보고하여 v2 의 stratified
CRC 가 demographic 그룹간 큰 격차를 만들지 않는지 진단.

산출: results/round9/equity_audit_real.{json,md}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
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


def _parse_demo(meta_info: str):
    out = {"sex": "?", "race": "?", "age_bucket": "?"}
    parts = meta_info.split()
    for p in parts:
        if p.startswith("sex="): out["sex"] = p.split("=", 1)[1]
        elif p.startswith("race="): out["race"] = p.split("=", 1)[1]
    return out


def _scores(backend: str, cases, verbose: bool = True):
    import time as _time
    sys_prompt = UQM.SYSTEM_PROMPT
    s_, l_, st_, demos = [], [], [], []
    skipped = 0
    t_start = _time.perf_counter()
    log_every = max(1, len(cases) // 20)
    for i, c in enumerate(cases):
        try:
            resp = query_model(backend, sys_prompt, c.question, temperature=0.0)
            sc = compute_nonconformity_score(resp)
        except Exception as e:
            skipped += 1
            if verbose and skipped <= 2:
                print(f"  [skip {i}] {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        s_.append(sc); l_.append(c.expected_escalate); st_.append(c.scenario_type.upper())
        demos.append(_parse_demo(c.meta_info))
        if verbose and ((i + 1) % log_every == 0 or i + 1 == len(cases)):
            elapsed = _time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    [{backend}] {i+1}/{len(cases)} ({rate:.2f}/s, skipped={skipped})", flush=True)
    return s_, l_, st_, demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="lmstudio")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round9" / "equity_audit_real")
    args = ap.parse_args()

    if not _MIMIC4_DEFAULT_PATH.exists():
        sys.exit(f"[R9.5] MIMIC-IV preprocessed JSONL missing: {_MIMIC4_DEFAULT_PATH}")

    all_cases = _load_mimic4_jsonl(_MIMIC4_DEFAULT_PATH, n=10**9, seed=args.seed)
    rng = random.Random(args.seed); rng.shuffle(all_cases)
    cal = all_cases[:args.n_cal]
    test = all_cases[args.n_cal:args.n_cal + args.n_test]

    cs, cl, cst, _ = _scores(args.backend, cal)
    ts, tl, tst, td = _scores(args.backend, test)

    crc = StratifiedConformalRiskControl(alphas=ALPHAS)
    crc.fit(cs, cl, cst)

    # group × stratum miss rate
    audit = defaultdict(lambda: defaultdict(lambda: {"n": 0, "n_pos": 0, "miss": 0}))
    for i, demo in enumerate(td):
        for grp_key in ["sex", "race"]:
            grp = demo.get(grp_key, "?")
            stratum = tst[i]
            lam = crc.threshold_for(stratum)
            audit[(grp_key, grp)][stratum]["n"] += 1
            if tl[i]:
                audit[(grp_key, grp)][stratum]["n_pos"] += 1
                if ts[i] <= lam:
                    audit[(grp_key, grp)][stratum]["miss"] += 1

    # to plain dict for JSON
    out_audit = {}
    for (gk, gv), per_stratum in audit.items():
        key = f"{gk}={gv}"
        out_audit[key] = {}
        for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            d = per_stratum.get(s, {"n": 0, "n_pos": 0, "miss": 0})
            mr = (d["miss"] / d["n_pos"]) if d["n_pos"] else None
            out_audit[key][s] = {
                "n": d["n"], "n_pos": d["n_pos"], "miss": d["miss"],
                "miss_rate": mr,
                "alpha": ALPHAS[s],
                "violation": (mr - ALPHAS[s]) if mr is not None else None,
            }

    report = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend, "seed": args.seed,
        "n_cal": len(cal), "n_test": len(test),
        "audit": out_audit,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))

    # markdown
    lines = ["# Round 9 R9.5 — Demographic equity audit (MIMIC-IV)\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Backend: {args.backend}, seed: {args.seed}\n")
    lines.append("| group | stratum | n_pos | miss_rate | α target | violation |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for grp, ps in out_audit.items():
        for s, d in ps.items():
            mr = f"{d['miss_rate']:.4f}" if d['miss_rate'] is not None else "—"
            v = f"{d['violation']:+.4f}" if d['violation'] is not None else "—"
            lines.append(f"| {grp} | {s} | {d['n_pos']} | {mr} | {d['alpha']} | {v} |")
    Path(str(args.out) + ".md").write_text("\n".join(lines))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
