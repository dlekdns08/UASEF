"""
Round 9 R9.3 — Real-EHR distribution shift (services-table specialty transfer)
══════════════════════════════════════════════════════════════════════════════

Round 8 의 round8_distribution_shift.py 는 **synthetic** specialty score 분포로
shift 를 시뮬레이션했다. R9.3 은 MIMIC-IV `services` 테이블의 ground-truth
specialty 지정 (`curr_service` → cardiology / neurology / general-medicine /
surgery) 으로 **실 EHR transfer** 를 측정한다.

방법:
  1. cardiology specialty 만으로 calibrate (StratifiedConformalRiskControl)
  2. {neurology, general_medicine, surgery} 각각에서 v2 threshold 적용 → miss 측정
  3. weighted CP (Tibshirani 2019) 의 likelihood-ratio reweighting 으로 회복 정량화
     - source distribution: cardiology 의 score 분포 (KDE)
     - target distribution: 각 test specialty 의 score 분포 (KDE)
     - threshold = quantile of source-weighted nonconformity scores

산출: results/round9/dist_shift_real.{json,md}  — paper supplementary §G 강화 입력
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import load_mimic4_by_specialty
from models.uqm import UQM, compute_nonconformity_score
from models.stratified_crc import StratifiedConformalRiskControl
from models.model_interface import query_model

ALPHAS_R9 = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}


def _scores(backend: str, cases, verbose: bool = True):
    import time as _time
    sys_prompt = UQM.SYSTEM_PROMPT
    out_s, out_l, out_st = [], [], []
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
        out_s.append(sc); out_l.append(c.expected_escalate); out_st.append(c.scenario_type.upper())
        if verbose and ((i + 1) % log_every == 0 or i + 1 == len(cases)):
            elapsed = _time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    [{backend}] {i+1}/{len(cases)} ({rate:.2f}/s, skipped={skipped})", flush=True)
    return out_s, out_l, out_st


def _gaussian_kde_pdf(values: list[float], bandwidth: float | None = None):
    """Tiny gaussian KDE without scipy. Returns f(x) callable."""
    if not values:
        return lambda x: 1e-9
    n = len(values)
    if bandwidth is None:
        # Silverman's rule
        sd = st.stdev(values) if n > 1 else 1.0
        bandwidth = 1.06 * (sd or 1.0) * (n ** (-1 / 5))
        bandwidth = max(bandwidth, 1e-3)
    coeff = 1.0 / (n * bandwidth * math.sqrt(2 * math.pi))
    def pdf(x: float) -> float:
        s = 0.0
        for v in values:
            z = (x - v) / bandwidth
            s += math.exp(-0.5 * z * z)
        return coeff * s + 1e-12
    return pdf


def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    """Weighted q-quantile (Tibshirani 2019 likelihood-ratio reweighted)."""
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    target = q * total
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return v
    return pairs[-1][0]


def evaluate_specialty_transfer(backend: str, source: str, targets: list[str],
                                 n_per_spec: int, seed: int, alpha: float,
                                 verbose: bool):
    if verbose: print(f"\n── source={source} backend={backend} seed={seed} ──")
    src_cases = load_mimic4_by_specialty(source, n=n_per_spec, seed=seed, verbose=False)
    if not src_cases:
        raise RuntimeError(f"specialty={source!r} 케이스 0개 — preprocessing 결과 확인")
    src_s, src_l, src_st = _scores(backend, src_cases)

    crc = StratifiedConformalRiskControl(alphas=ALPHAS_R9)
    crc.fit(src_s, src_l, src_st)

    src_pdf = _gaussian_kde_pdf(src_s)

    rows = []
    for tgt in targets:
        tgt_cases = load_mimic4_by_specialty(tgt, n=n_per_spec, seed=seed + 1000, verbose=False)
        if not tgt_cases:
            print(f"  [skip] target={tgt} 데이터 없음")
            continue
        tgt_s, tgt_l, tgt_st = _scores(backend, tgt_cases)
        if not tgt_s:
            continue
        tgt_pdf = _gaussian_kde_pdf(tgt_s)

        # naive transfer: source threshold 직접 적용
        # 해당 stratum 별 miss rate 측정
        per_stratum_naive = {}
        for s, alpha_s in ALPHAS_R9.items():
            idx = [i for i, x in enumerate(tgt_st) if x == s]
            if not idx: continue
            lam = crc.threshold_for(s)
            n_pos = sum(tgt_l[i] for i in idx)
            misses = sum(1 for i in idx if tgt_l[i] and tgt_s[i] <= lam)
            per_stratum_naive[s] = {
                "n": len(idx), "n_pos": n_pos,
                "lambda_naive": lam,
                "miss_rate": (misses / n_pos) if n_pos else None,
                "violation": (misses / n_pos - alpha_s) if n_pos else None,
                "violation_ratio": (misses / n_pos / alpha_s) if (n_pos and alpha_s) else None,
            }

        # weighted CP (Tibshirani 2019): likelihood ratio w(x) = p_target(x) / p_source(x)
        weights_src = []
        for v in src_s:
            w = tgt_pdf(v) / src_pdf(v)
            weights_src.append(max(min(w, 100.0), 1e-3))   # clip to avoid blow-up

        per_stratum_weighted = {}
        for s, alpha_s in ALPHAS_R9.items():
            mask = [i for i, x in enumerate(src_st) if x == s and src_l[i]]   # positive cases of stratum s in source
            if len(mask) < 5:
                per_stratum_weighted[s] = None
                continue
            stratum_scores = [src_s[i] for i in mask]
            stratum_weights = [weights_src[i] for i in mask]
            lam_w = _weighted_quantile(stratum_scores, stratum_weights, alpha_s)
            idx_t = [i for i, x in enumerate(tgt_st) if x == s]
            n_pos = sum(tgt_l[i] for i in idx_t)
            misses = sum(1 for i in idx_t if tgt_l[i] and tgt_s[i] <= lam_w)
            per_stratum_weighted[s] = {
                "n": len(idx_t), "n_pos": n_pos,
                "lambda_weighted": lam_w,
                "miss_rate": (misses / n_pos) if n_pos else None,
                "violation_ratio": (misses / n_pos / alpha_s) if (n_pos and alpha_s) else None,
            }

        rows.append({
            "target_specialty": tgt,
            "naive": per_stratum_naive,
            "weighted_cp": per_stratum_weighted,
        })
    return {"source": source, "rows": rows}


def write_md(report: dict, out_md: Path):
    lines = ["# Round 9 R9.3 — Real-EHR distribution shift (MIMIC-IV services)\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Source specialty: {report['source']}")
    lines.append(f"- Targets: {', '.join(report['targets'])}\n")
    for backend, runs in report["per_backend"].items():
        lines.append(f"## backend = {backend}\n")
        for run in runs:
            lines.append(f"### seed = {run['seed']}\n")
            lines.append("| target specialty | stratum | n_pos | naive miss | violation× | weighted miss | weighted violation× |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for row in run["transfer"]:
                tgt = row["target_specialty"]
                for s in ["CRITICAL","HIGH","MODERATE","LOW"]:
                    n = row["naive"].get(s); w = (row["weighted_cp"] or {}).get(s)
                    if not n: continue
                    naive_miss = f"{n['miss_rate']:.4f}" if n.get('miss_rate') is not None else "—"
                    naive_v = f"{n['violation_ratio']:.2f}" if n.get('violation_ratio') is not None else "—"
                    w_miss = f"{w['miss_rate']:.4f}" if w and w.get('miss_rate') is not None else "—"
                    w_v = f"{w['violation_ratio']:.2f}" if w and w.get('violation_ratio') is not None else "—"
                    lines.append(f"| {tgt} | {s} | {n.get('n_pos',0)} | {naive_miss} | {naive_v}× | {w_miss} | {w_v}× |")
            lines.append("")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cardiology")
    ap.add_argument("--targets", nargs="+", default=["neurology", "internal_medicine", "surgery"])
    ap.add_argument("--n-per-spec", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--backends", nargs="+", default=["lmstudio"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round9" / "dist_shift_real")
    args = ap.parse_args()

    report = {
        "timestamp": datetime.now().isoformat(),
        "source": args.source, "targets": args.targets,
        "n_per_spec": args.n_per_spec, "seeds": args.seeds, "backends": args.backends,
        "per_backend": {},
    }
    for backend in args.backends:
        runs = []
        for seed in args.seeds:
            try:
                r = evaluate_specialty_transfer(
                    backend, args.source, args.targets, args.n_per_spec, seed, args.alpha, True,
                )
            except FileNotFoundError as e:
                print(f"[R9.3] preprocessed MIMIC-IV missing: {e}"); sys.exit(2)
            runs.append({"seed": seed, "transfer": r["rows"]})
        report["per_backend"][backend] = runs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
