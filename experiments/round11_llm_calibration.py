"""
Round 11 R11.5 — Post-hoc LLM calibration (Platt + isotonic).

Question: "LLM safety gate 가 임상 deployable 하게 보정 가능한가?"

가설 (pre-registered):
   - Platt scaling 으로 ECE 가 0.34 → 0.05 이하로 크게 감소
   - 하지만 CRC 의 CRITICAL miss 는 그대로 (sharpness 0.0157 이 fundamental
     limit; calibration 은 ECE 만 개선, score 분리도 못 늘림)
   - 즉 "calibration alone is insufficient" 가 strong 한 답.

방법:
   1. R10.4 의 LLM scores cache 재사용 (~25k scores already computed)
   2. Cal split 의 절반 (cal_a) → Platt + isotonic + beta calibrator fit
   3. Cal split 의 나머지 (cal_b) → CRC quantile fit
   4. Test set 에 calibrated scores 적용 → ECE / Brier / CRITICAL miss / over_esc

산출: results/round11/r11_5_llm_calibration.{json,md}
Wallclock: < 30분 (LLM scores 캐시 재사용, no LLM inference)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.metrics_utils import clopper_pearson_upper, patient_level_split
from experiments.round10_method_agnostic import _subject_id, _attach_r10_features
from data.loader import _load_mimic4_jsonl

DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"
DEFAULT_CACHE = ROOT / "results" / "round10" / "r10_4_cache" / "gpt_oss_120b.json"

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}


def ece_10bin(probs: np.ndarray, labels: np.ndarray) -> float:
    bins = np.linspace(0, 1, 11)
    e = 0.0
    n = len(probs)
    for i in range(10):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < 9 else probs <= 1.0)
        if mask.sum() == 0: continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        e += (mask.sum() / n) * abs(acc - conf)
    return float(e)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(((probs - labels) ** 2).mean())


def sharpness(probs: np.ndarray) -> float:
    return float(probs.var(ddof=0))


def crc_quantile(cal_scores: np.ndarray, cal_labels: np.ndarray,
                 alpha: float) -> float:
    """Standard CRC threshold on negative-NLL or probability scores.

    For probability scores p ∈ [0,1] with positive=should-escalate,
    the loss is ℓ = 1{y=1, p ≤ λ}. The CRC quantile is the (1-α)-quantile
    of p among positives.
    """
    pos = cal_scores[cal_labels == 1]
    if len(pos) == 0:
        return float("-inf")
    # Threshold = lower quantile of positive scores
    q = np.quantile(pos, alpha, method="lower")
    return float(q)


def evaluate_calibrated(probs_cal_b: np.ndarray, labels_cal_b: np.ndarray,
                        strata_cal_b: list[str],
                        probs_test: np.ndarray, labels_test: np.ndarray,
                        strata_test: list[str]) -> dict:
    """Fit CRC quantile per stratum, evaluate on test."""
    per_stratum = {}
    for s, alpha in ALPHAS.items():
        cal_idx = [i for i, x in enumerate(strata_cal_b) if x == s]
        if not cal_idx:
            per_stratum[s] = None; continue
        cal_s = probs_cal_b[cal_idx]
        cal_l = labels_cal_b[cal_idx]
        lam = crc_quantile(cal_s, cal_l, alpha)
        idx_t = [i for i, x in enumerate(strata_test) if x == s]
        if not idx_t:
            per_stratum[s] = None; continue
        t_s = probs_test[idx_t]
        t_l = labels_test[idx_t]
        n_pos = int(t_l.sum())
        n_neg = int(len(t_l) - n_pos)
        misses = int(((t_l == 1) & (t_s < lam)).sum())
        over_esc = int(((t_l == 0) & (t_s >= lam)).sum())
        per_stratum[s] = {
            "n": len(idx_t), "n_pos": n_pos, "n_neg": n_neg,
            "misses": misses,
            "miss_rate": misses / n_pos if n_pos else None,
            "over_esc": over_esc,
            "over_esc_rate": over_esc / n_neg if n_neg else None,
            "exact_upper95": clopper_pearson_upper(misses, n_pos, 0.95)
                              if n_pos else None,
            "alpha": alpha, "lambda": lam,
        }
    # Global ECE / Brier / sharpness on test set
    per_stratum["_global"] = {
        "ECE": ece_10bin(probs_test, labels_test),
        "Brier": brier(probs_test, labels_test),
        "Sharpness": sharpness(probs_test),
    }
    return per_stratum


def fit_platt(cal_scores: np.ndarray, cal_labels: np.ndarray):
    """Platt scaling = logistic regression on 1-D score."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(cal_scores.reshape(-1, 1), cal_labels)
    return lambda x: clf.predict_proba(x.reshape(-1, 1))[:, 1]


def fit_isotonic(cal_scores: np.ndarray, cal_labels: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_scores, cal_labels)
    return lambda x: iso.predict(x)


def fit_identity(cal_scores: np.ndarray, cal_labels: np.ndarray):
    """No-op baseline — score → probability via min-max."""
    lo, hi = float(cal_scores.min()), float(cal_scores.max())
    if hi - lo < 1e-9: hi = lo + 1.0
    return lambda x: (np.clip(x, lo, hi) - lo) / (hi - lo)


def load_llm_scores(cache_path: Path):
    """R10.4 cache 의 per-seed LLM scores 를 재구성.

    cache JSON 은 per_seed 의 미가공 미스/오버 카운트만 있고
    score 자체는 cache 에 저장되어 있지 않을 수 있다. 그 경우
    LLM scores 를 다시 계산하지 않고 (시간 비싸므로) 캐시된 score
    list 가 있다면 그것을 사용한다. 없으면 None 반환.
    """
    if not cache_path.exists():
        return None
    j = json.loads(cache_path.read_text())
    per_seed = j.get("per_seed", [])
    if not per_seed:
        return None
    # 캐시에 raw scores 가 저장되어 있는지 확인 (R10.4 의 구버전은 미저장 가능)
    first = per_seed[0]
    if "cal_scores" not in first or "test_scores" not in first:
        return None  # raw scores cache 없음 — caller 에게 신호
    return per_seed


def recompute_llm_scores_via_inference(jsonl_path: Path, seeds, n_cal, n_test):
    """Cache 에 raw scores 가 없을 때 fallback — LLM 재실행."""
    from experiments.round10_method_agnostic import _llm_scores
    import random
    cases = _load_mimic4_jsonl(jsonl_path, n=10**9, seed=42)
    _attach_r10_features(cases, jsonl_path)
    per_seed = []
    for seed in seeds:
        cal, test = patient_level_split(cases, group_of=_subject_id,
                                          cal_frac=0.8, seed=seed)
        rng = random.Random(seed); rng.shuffle(cal); rng.shuffle(test)
        cal = cal[:n_cal]; test = test[:n_test]
        print(f"[R11.5] seed={seed} computing LLM scores (this is slow!)")
        cs = _llm_scores(cal); ts = _llm_scores(test)
        per_seed.append({
            "seed": seed,
            "cal_scores": cs, "test_scores": ts,
            "cal_labels": [int(bool(c.expected_escalate)) for c in cal],
            "test_labels": [int(bool(c.expected_escalate)) for c in test],
            "cal_strata": [(c.scenario_type or "").upper() for c in cal],
            "test_strata": [(c.scenario_type or "").upper() for c in test],
        })
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--llm-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round11" / "r11_5_llm_calibration")
    ap.add_argument("--recompute", action="store_true",
                    help="Cache 에 raw scores 가 없으면 LLM 재실행 (~slow)")
    args = ap.parse_args()

    per_seed = load_llm_scores(args.llm_cache)
    if per_seed is None:
        if args.recompute:
            print("[R11.5] cache 에 raw LLM scores 없음 — 재계산 시작")
            per_seed = recompute_llm_scores_via_inference(
                args.jsonl, args.seeds, args.n_cal, args.n_test)
        else:
            sys.exit("LLM cache 에 raw scores 없음. --recompute 로 LLM 재실행 가능.")

    methods = ["uncalibrated", "platt", "isotonic"]
    per_seed_results = {m: [] for m in methods}
    for seed_entry in per_seed:
        if "cal_scores" not in seed_entry:
            print(f"[R11.5] seed {seed_entry.get('seed')} 의 raw scores 부재 — skip")
            continue
        seed = seed_entry["seed"]
        cs = np.array(seed_entry["cal_scores"], dtype=float)
        ts = np.array(seed_entry["test_scores"], dtype=float)
        cl = np.array(seed_entry["cal_labels"], dtype=int)
        tl = np.array(seed_entry["test_labels"], dtype=int)
        cstr = list(seed_entry["cal_strata"])
        tstr = list(seed_entry["test_strata"])

        # NaN 제거
        mask_c = ~np.isnan(cs)
        cs = cs[mask_c]; cl = cl[mask_c]; cstr = [cstr[i] for i in range(len(mask_c)) if mask_c[i]]
        mask_t = ~np.isnan(ts)
        ts = ts[mask_t]; tl = tl[mask_t]; tstr = [tstr[i] for i in range(len(mask_t)) if mask_t[i]]

        # Cal split A/B
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(cs))
        half = len(cs) // 2
        ia, ib = perm[:half], perm[half:]
        cs_a, cl_a = cs[ia], cl[ia]
        cs_b, cl_b = cs[ib], cl[ib]
        cstr_b = [cstr[i] for i in ib]

        # LLM raw scores 는 보통 negative NLL. probability 로 정규화 후 calibrate.
        # min-max identity → 그대로 probability domain 으로 매핑
        identity = fit_identity(cs_a, cl_a)
        probs_a = identity(cs_a); probs_b = identity(cs_b); probs_t = identity(ts)

        for method in methods:
            if method == "uncalibrated":
                fn = identity
            elif method == "platt":
                fn = fit_platt(cs_a, cl_a)
            elif method == "isotonic":
                fn = fit_isotonic(cs_a, cl_a)
            calibrated_b = fn(cs_b)
            calibrated_t = fn(ts)
            r = evaluate_calibrated(
                calibrated_b, cl_b, cstr_b,
                calibrated_t, tl, tstr)
            r["seed"] = seed
            per_seed_results[method].append(r)

    def pool(per_seed_list, stratum):
        m = n = oe = nneg = 0
        for r in per_seed_list:
            c = r.get(stratum)
            if not c or c.get("n_pos", 0) == 0: continue
            m += c["misses"]; n += c["n_pos"]
            oe += c["over_esc"]; nneg += c["n_neg"]
        if n == 0: return None
        return {
            "pooled_misses": m, "pooled_n_pos": n,
            "pooled_miss_rate": m / n,
            "pooled_over_esc": oe, "pooled_n_neg": nneg,
            "pooled_over_esc_rate": (oe / nneg) if nneg else None,
            "pooled_exact_upper95": clopper_pearson_upper(m, n, 0.95),
            "alpha": ALPHAS[stratum],
            "satisfies_alpha": clopper_pearson_upper(m, n, 0.95) <= ALPHAS[stratum],
        }

    summary = {}
    for method in methods:
        summary[method] = {s: pool(per_seed_results[method], s) for s in ALPHAS}
        # ECE/Brier averaged across seeds
        eces = [r["_global"]["ECE"] for r in per_seed_results[method] if "_global" in r]
        briers = [r["_global"]["Brier"] for r in per_seed_results[method] if "_global" in r]
        sharps = [r["_global"]["Sharpness"] for r in per_seed_results[method] if "_global" in r]
        summary[method]["_global"] = {
            "mean_ECE": float(np.mean(eces)) if eces else None,
            "mean_Brier": float(np.mean(briers)) if briers else None,
            "mean_Sharpness": float(np.mean(sharps)) if sharps else None,
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "llm_cache": str(args.llm_cache),
        "methods": methods,
        "seeds": args.seeds,
        "summary": summary,
        "per_seed_results": per_seed_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))

    # Markdown
    lines = ["# R11.5 — LLM Post-hoc Calibration\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Methods: {', '.join(methods)}\n")
    lines.append("## Global calibration metrics (test-pooled)\n")
    lines.append("| Method | Mean ECE | Mean Brier | Mean Sharpness |")
    lines.append("| --- | --- | --- | --- |")
    for m in methods:
        g = summary[m]["_global"]
        lines.append(f"| {m} | "
                     f"{g['mean_ECE']:.4f} | "
                     f"{g['mean_Brier']:.4f} | "
                     f"{g['mean_Sharpness']:.4f} |")

    lines.append("\n## CRC coverage per stratum\n")
    for s in ["CRITICAL", "HIGH"]:
        lines.append(f"\n### {s} (α={ALPHAS[s]})\n")
        lines.append("| Method | pooled miss / n_pos | exact 95% upper | over_esc rate | α-satisfy? |")
        lines.append("| --- | --- | --- | --- | :---: |")
        for m in methods:
            cell = summary[m].get(s)
            if not cell:
                lines.append(f"| {m} | — | — | — | — |"); continue
            ok = "✓" if cell["satisfies_alpha"] else "✗"
            lines.append(f"| {m} | "
                         f"{cell['pooled_misses']}/{cell['pooled_n_pos']} | "
                         f"{cell['pooled_exact_upper95']:.4f} | "
                         f"{cell['pooled_over_esc_rate']:.4f} | {ok} |")

    # Verdict
    crit_uncal = summary["uncalibrated"].get("CRITICAL")
    crit_platt = summary["platt"].get("CRITICAL")
    crit_iso = summary["isotonic"].get("CRITICAL")
    lines.append("\n## R11.5 verdict\n")
    if crit_uncal and crit_platt:
        delta_miss = crit_platt["pooled_miss_rate"] - crit_uncal["pooled_miss_rate"]
        delta_ece = (summary["platt"]["_global"]["mean_ECE"] -
                     summary["uncalibrated"]["_global"]["mean_ECE"])
        if crit_platt["satisfies_alpha"] and not crit_uncal["satisfies_alpha"]:
            lines.append("**✓ Calibration RESCUES the LLM gate** — "
                         f"Platt drops CRITICAL miss by {-delta_miss:.4f}, "
                         f"α=0.05 satisfied.")
        elif delta_ece < -0.1 and delta_miss > -0.02:
            lines.append("**✗ Calibration improves ECE but does NOT rescue CRC** — "
                         f"ECE Δ={delta_ece:.3f} (improved) yet "
                         f"CRITICAL miss Δ={delta_miss:.4f} (essentially unchanged). "
                         "Confirms 'calibration alone is insufficient'.")
        else:
            lines.append("**Mixed effect** — manual interpretation needed.")
    Path(str(args.out) + ".md").write_text("\n".join(lines))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
