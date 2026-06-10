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
from experiments.metrics_utils import (
    patient_level_split, clopper_pearson_upper, n_for_zero_miss_upper,
)


ALPHAS_R9 = {"CRITICAL": 0.001, "HIGH": 0.01, "MODERATE": 0.05, "LOW": 0.10}


def _subject_id(case) -> str:
    """meta_info 의 `subject_id=...` 토큰 추출 (patient-level split key)."""
    for tok in (case.meta_info or "").split():
        if tok.startswith("subject_id="):
            return tok.split("=", 1)[1]
    # fallback: hadm_id 단위(=admission 단위). 구버전 JSONL 호환.
    for tok in (case.meta_info or "").split():
        if tok.startswith("hadm_id="):
            return "h:" + tok.split("=", 1)[1]
    return id(case).__str__()


def collect_scores(backend: str, cases: list, verbose: bool = False) -> tuple[list[float], list[bool], list[str]]:
    import sys as _sys
    import time as _time
    sys_prompt = UQM.SYSTEM_PROMPT
    scores, labels, strata = [], [], []
    skipped = 0
    t_start = _time.perf_counter()
    log_every = max(1, len(cases) // 40)  # ~40 log lines per pass
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
            print(f"  [{backend}] {i+1}/{len(cases)} cases ({rate:.2f}/s, ETA {eta/60:.1f}min, skipped={skipped})", flush=True)
            _sys.stdout.flush()
    if skipped:
        print(f"  total skipped: {skipped}/{len(cases)} cases", flush=True)
    return scores, labels, strata


def evaluate_one_seed(backend: str, seed: int, n_critical: int, alphas: dict, verbose: bool):
    if verbose:
        print(f"\n── backend={backend} seed={seed} ──")
    bucket = load_mimic4_by_stratum(n_per_stratum=n_critical, seed=seed, verbose=False)
    # ── PATIENT-LEVEL split (REVISION_PLAN P0-3) ──
    # 같은 subject_id 의 여러 admission 이 cal/test 양쪽에 들어가지 않도록 환자
    # 단위로 분할. stratum 별로 분할해 stratum 균형을 유지하되 그룹 키는 subject_id.
    cal: list = []
    test: list = []
    for stratum, cases in bucket.items():
        c_cal, c_test = patient_level_split(
            cases, group_of=_subject_id, cal_frac=0.8, seed=seed,
        )
        cal.extend(c_cal)
        test.extend(c_test)
    if verbose:
        cal_subj = {_subject_id(c) for c in cal}
        test_subj = {_subject_id(c) for c in test}
        overlap = cal_subj & test_subj
        print(f"  cal={len(cal)} test={len(test)} "
              f"(patient-level; subject overlap={len(overlap)})")

    cal_scores, cal_labels, cal_strata = collect_scores(backend, cal, verbose)
    test_scores, test_labels, test_strata = collect_scores(backend, test, verbose)

    if not cal_scores:
        raise RuntimeError(
            f"backend={backend!r} 로 cal score 가 0개 수집되었습니다. "
            f"API 키 / 백엔드 가용성을 확인하세요 (UASEF/.env)."
        )

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
            "misses": misses,
            "lambda_hat": lam,
            "alpha_target": alphas[s],
            "empirical_E_loss": per_example_loss_sum / n,
            "miss_rate_cond": (misses / positives) if positives else None,
            "n_satisfies_min": n >= int(round((1 - alphas[s]) / alphas[s])),
        }
    return per_stratum_result


def aggregate(results_per_seed: list[dict], alphas: dict) -> dict:
    """
    Across seeds: pooled exact binomial(Clopper-Pearson) upper bound 가 핵심 통계.

    ⚠️ 이전 'mean + 2σ' 상한은 0 miss 관측 시 std=0 → upper=0 으로 vacuous 였음
    (REVISION_PLAN P0-4). 0/N 관측은 true rate≤α 를 *증명하지 않으며*, 정직한
    상한은 1-(1-conf)**(1/N). 따라서 satisfies_alpha 는 단측 exact 상한 기준.
    """
    out = {}
    for s in alphas:
        rows = [r[s] for r in results_per_seed if r.get(s)]
        if not rows:
            out[s] = None; continue
        E_vals = [r["empirical_E_loss"] for r in rows]
        # seed 전체에 걸쳐 conditional miss 를 pool (independence 가정 하 보수적).
        pooled_misses = sum(r.get("misses", 0) for r in rows)
        pooled_pos = sum(r.get("n_pos", 0) for r in rows)
        pooled_n = sum(r.get("n", 0) for r in rows)
        cond_upper = clopper_pearson_upper(pooled_misses, pooled_pos, conf=0.95) \
            if pooled_pos > 0 else 1.0
        mean = st.mean(E_vals)
        std = st.stdev(E_vals) if len(E_vals) > 1 else 0.0
        out[s] = {
            "alpha_target": alphas[s],
            "n_seeds": len(E_vals),
            "values": [round(v, 5) for v in E_vals],
            "mean":   round(mean, 5),
            "std":    round(std, 5),
            "pooled_n": pooled_n,
            "pooled_n_pos": pooled_pos,
            "pooled_misses": pooled_misses,
            "cond_miss_rate": round(pooled_misses / pooled_pos, 6) if pooled_pos else None,
            "exact_upper95": round(cond_upper, 6),
            "n_needed_for_alpha": (n_for_zero_miss_upper(alphas[s]) if pooled_misses == 0 else None),
            # 정직한 판정: 0/N 은 'α 실증'이 아니라 'α 이하와 양립(non-vacuous)'.
            "compatible_with_alpha": cond_upper <= alphas[s],
            "note": (
                "0 miss observed; upper bound limited by test n — "
                "NOT a proof that true miss rate ≤ alpha"
                if pooled_misses == 0 else "miss observed"
            ),
        }
    return out


def write_md(report: dict, out_md: Path) -> None:
    lines = ["# Round 9 R9.1 — α=0.001 empirical (MIMIC-IV CRITICAL)\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Backends: {', '.join(report['backends'])}")
    lines.append(f"- Seeds: {report['seeds']}\n")
    lines.append(
        "각 셀: pooled `misses/n_pos`, 단측 exact(Clopper-Pearson) 95% 상한.\n"
        "✓ = 관측이 α 와 **양립(non-vacuous)**. ⚠️ 0 miss 는 true rate ≤ α 의 *증명이 아님* "
        "— 상한은 test n 으로 제한됨.\n")
    for backend, agg in report["per_backend"].items():
        lines.append(f"## backend={backend}\n")
        lines.append("| stratum | α | n_seeds | E[ℓ] mean±std | misses/n_pos | exact 95% upper | compat? | n needed (α) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for s, r in agg.items():
            if r is None:
                lines.append(f"| {s} | {ALPHAS_R9[s]} | 0 | — | — | — | — | — |")
                continue
            ok = "✓" if r["compatible_with_alpha"] else "✗"
            need = r.get("n_needed_for_alpha")
            need_s = str(need) if need else "—"
            lines.append(
                f"| {s} | {r['alpha_target']} | {r['n_seeds']} | "
                f"{r['mean']:.5f} ± {r['std']:.5f} | {r['pooled_misses']}/{r['pooled_n_pos']} | "
                f"{r['exact_upper95']:.5f} | {ok} | {need_s} |"
            )
        lines.append("")
        lines.append(
            "> 해석: CRITICAL 에서 0 miss 가 관측되어도 95% 단측 상한이 α=0.001 이하가 "
            "되려면 n_pos ≥ ~2995 필요. 현재 표본에서의 0 miss 는 'α=0.001 실증'이 아니라 "
            "'α=0.001 calibration 이 non-vacuous 하고 held-out 관측이 우호적' 이라는 주장만 "
            "지지함 (REVISION_PLAN P0-4).\n")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-critical", type=int, default=1500)
    ap.add_argument("--alpha-critical", type=float, default=0.001)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--backends", nargs="+", default=["lmstudio"])
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
