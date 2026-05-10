"""
Round 9 R9.1 — α_CRITICAL = 0.001 empirical validation on MIMIC-IV
══════════════════════════════════════════════════════════════════════════════

paper §7.2 (L3) 의 'n_CRITICAL ≥ 999 unmet → α=0.001 aspirational' 한계를
MIMIC-IV CRITICAL stratum (n≈40k from real outcomes) 으로 직접 empirical
검증한다.

방법:
  1. preprocessed JSONL 에서 stratum-balanced 샘플 (CRITICAL n=1500 default)
  2. 각 case → query_model(backend) → compute_nonconformity_score
  3. StratifiedConformalRiskControl(alphas={'CRITICAL': 0.001, ...}).fit
  4. holdout 으로 empirical risk E[ℓ] 측정 → 2σ upper bound 가 α=0.001 이하인지
  5. seed × backend 조합으로 멀티시드 → bootstrap 95% CI

산출: results/round9/alpha_critical_real.{json,md}  (Table 1c)
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

from data.loader import load_mimic4_by_stratum
from models.uqm import UQM, compute_nonconformity_score
from models.model_interface import query_model
from models.stratified_crc import StratifiedConformalRiskControl


ALPHAS_R9 = {"CRITICAL": 0.001, "HIGH": 0.01, "MODERATE": 0.05, "LOW": 0.10}


def collect_scores(backend: str, cases: list, verbose: bool = False) -> tuple[list[float], list[bool], list[str]]:
    sys_prompt = UQM.SYSTEM_PROMPT
    scores, labels, strata = [], [], []
    skipped = 0
    for i, case in enumerate(cases):
        try:
            resp = query_model(backend, sys_prompt, case.question, temperature=0.0)
            sc = compute_nonconformity_score(resp)
        except Exception as e:
            skipped += 1
            if verbose and skipped <= 3:
                print(f"  [skip] {e}")
            continue
        scores.append(sc)
        labels.append(case.expected_escalate)
        strata.append(case.scenario_type.upper())
        if verbose and (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(cases)}")
    if skipped:
        print(f"  skipped {skipped}/{len(cases)} cases")
    return scores, labels, strata


def evaluate_one_seed(backend: str, seed: int, n_critical: int, alphas: dict, verbose: bool):
    if verbose:
        print(f"\n── backend={backend} seed={seed} ──")
    bucket = load_mimic4_by_stratum(n_per_stratum=n_critical, seed=seed, verbose=False)
    cal_frac = 0.8
    cal: list = []
    test: list = []
    rng = random.Random(seed)
    for stratum, cases in bucket.items():
        rng.shuffle(cases)
        cut = int(len(cases) * cal_frac)
        cal.extend(cases[:cut])
        test.extend(cases[cut:])
    if verbose:
        print(f"  cal={len(cal)} test={len(test)}")

    cal_scores, cal_labels, cal_strata = collect_scores(backend, cal, verbose)
    test_scores, test_labels, test_strata = collect_scores(backend, test, verbose)

    crc = StratifiedConformalRiskControl(alphas=alphas)
    crc.fit(cal_scores, cal_labels, cal_strata)

    per_stratum_result = {}
    for s in alphas:
        s_idx = [i for i, x in enumerate(test_strata) if x == s]
        if not s_idx:
            per_stratum_result[s] = None
            continue
        lam = crc.threshold_for(s)
        misses = 0
        positives = 0
        per_example_loss_sum = 0.0
        for i in s_idx:
            sc, lbl = test_scores[i], test_labels[i]
            # missed_escalation_loss: 1 if (label True and sc <= lam) else 0
            loss = 1.0 if (lbl and sc <= lam) else 0.0
            per_example_loss_sum += loss
            if lbl:
                positives += 1
                if sc <= lam:
                    misses += 1
        n = len(s_idx)
        per_stratum_result[s] = {
            "n": n,
            "n_pos": positives,
            "lambda_hat": lam,
            "alpha_target": alphas[s],
            "empirical_E_loss": per_example_loss_sum / n,
            "miss_rate_cond": (misses / positives) if positives else None,
            "n_satisfies_min": n >= int(round((1 - alphas[s]) / alphas[s])),
        }
    return per_stratum_result


def aggregate(results_per_seed: list[dict], alphas: dict) -> dict:
    """Across seeds: mean ± std + 2σ upper for E[ℓ] + bootstrap 95% CI."""
    out = {}
    for s in alphas:
        E_vals = [r[s]["empirical_E_loss"] for r in results_per_seed if r.get(s)]
        if not E_vals:
            out[s] = None; continue
        mean = st.mean(E_vals)
        std = st.stdev(E_vals) if len(E_vals) > 1 else 0.0
        upper_2s = mean + 2 * std
        # percentile bootstrap 95% CI on the mean
        rng = random.Random(42)
        means = []
        n = len(E_vals)
        for _ in range(2000):
            sample = [E_vals[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        ci_lo = means[int(0.025 * len(means))]
        ci_hi = means[int(0.975 * len(means)) - 1]
        out[s] = {
            "alpha_target": alphas[s],
            "n_seeds": len(E_vals),
            "values": [round(v, 5) for v in E_vals],
            "mean":   round(mean, 5),
            "std":    round(std, 5),
            "two_sigma_upper":  round(upper_2s, 5),
            "ci95":  [round(ci_lo, 5), round(ci_hi, 5)],
            "satisfies_alpha": upper_2s <= alphas[s],
        }
    return out


def write_md(report: dict, out_md: Path) -> None:
    lines = ["# Round 9 R9.1 — α=0.001 empirical (MIMIC-IV CRITICAL)\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Backends: {', '.join(report['backends'])}")
    lines.append(f"- Seeds: {report['seeds']}\n")
    lines.append("Each cell: `mean ± std  [2σ upper]  (95% CI)`. ✓ if 2σ upper ≤ α.\n")
    for backend, agg in report["per_backend"].items():
        lines.append(f"## backend={backend}\n")
        lines.append("| stratum | α | n_seeds | E[ℓ] mean ± std | 2σ upper | 95% CI | satisfies? |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for s, r in agg.items():
            if r is None:
                lines.append(f"| {s} | {ALPHAS_R9[s]} | 0 | — | — | — | — |")
                continue
            ok = "✓" if r["satisfies_alpha"] else "✗"
            lines.append(
                f"| {s} | {r['alpha_target']} | {r['n_seeds']} | "
                f"{r['mean']:.5f} ± {r['std']:.5f} | {r['two_sigma_upper']:.5f} | "
                f"[{r['ci95'][0]:.5f}, {r['ci95'][1]:.5f}] | {ok} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-critical", type=int, default=1500)
    ap.add_argument("--alpha-critical", type=float, default=0.001)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--backends", nargs="+", default=["openai", "lmstudio"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round9" / "alpha_critical_real")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    alphas = dict(ALPHAS_R9)
    alphas["CRITICAL"] = args.alpha_critical

    report: dict = {
        "timestamp": datetime.now().isoformat(),
        "n_critical": args.n_critical,
        "alphas": alphas,
        "seeds": args.seeds,
        "backends": args.backends,
        "per_backend": {},
        "per_seed": {},
    }
    for backend in args.backends:
        per_seed = []
        for seed in args.seeds:
            try:
                r = evaluate_one_seed(backend, seed, args.n_critical, alphas, args.verbose)
            except FileNotFoundError as e:
                print(f"[R9.1] preprocessed MIMIC-IV missing: {e}")
                sys.exit(2)
            per_seed.append(r)
        report["per_backend"][backend] = aggregate(per_seed, alphas)
        report["per_seed"][backend] = per_seed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(args.out) + ".json")
    md_path   = Path(str(args.out) + ".md")
    json_path.write_text(json.dumps(report, indent=2, default=str))
    write_md(report, md_path)
    print(f"\n✅ {json_path}")
    print(f"✅ {md_path}")


if __name__ == "__main__":
    main()
