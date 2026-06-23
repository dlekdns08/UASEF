"""
Round 11 R11.1 — Method-agnostic CRC re-run with MINIMAL leakage-safe features.

Round 10 R10.4 의 headline ("RandomForest is unique winner, 0/1293") 가
vacuous solution (over_esc = 1.0 in all 5 seeds) 임을 발견 + R10.4 feature
vector 가 R10.7 의 leakage suspect (Charlson, specialty_baseline_rate) 를
포함하는 self-inconsistency 발견.

R11.1 은 R10.4 를 다음 두 가지 fix 와 함께 재실행:

  1. **MINIMAL features only**: age_bucket, adm_emerg, specialty_id,
     n_lab_flags 의 4개만 사용. R10.7 의 leakage suspect (Charlson,
     specialty_baseline_rate, n_vital_flags) 완전 제거.

  2. **Over-escalation rate 명시 보고**: 각 cell 의 over_esc 까지
     출력하여 "vacuous escalate-all" solution 인지 자동 verdict.
     모든 classifier 의 over_esc = 1.0 이면 "VACUOUS CRC threshold
     collapse — framework limit, not classifier win" 으로 framing.

산출: results/round11/r11_1_method_agnostic_minimal.{json,md}

Wallclock: ~64h (LLM 60k calls) + tabular instant
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

from data.loader import _load_mimic4_jsonl
from experiments.metrics_utils import clopper_pearson_upper, patient_level_split
from experiments.round10_method_agnostic import (
    _make_classifier, _llm_scores, _subject_id,
)
from models.stratified_crc import StratifiedConformalRiskControl

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"


def _feature_vector_minimal(case) -> list[float]:
    """
    Truly leakage-safe minimal features (R11).

    제거됨 (R10.7 L25-L26 의 leakage suspect):
      - charlson_index             (current-admission ICD 사용)
      - specialty_baseline_rate    (cohort-level statistic)
      - n_vital_flags              (chartevents coverage 미보장)

    유지 (decision-time only):
      - age_bucket  (admission registration time 정보)
      - adm_emerg   (admission_type, decision-time hard fact)
      - spec_idx    (services.curr_service first record)
      - n_labs      (early lab order count, first 6h)
    """
    meta = case.meta_info or ""
    age_bucket = {"unknown": 0, "<18": 1, "18-34": 2, "35-49": 3,
                  "50-64": 4, "65-79": 5, "80+": 6}
    age_idx = 0
    for tok in meta.split():
        if tok.startswith("age_bucket="):
            age_idx = age_bucket.get(tok.split("=", 1)[1].strip(), 0)
    adm_emerg = 1.0 if "EMERG" in meta or "URG" in meta else 0.0
    spec_map = {"cardiology": 1, "neurology": 2, "internal_medicine": 3,
                "surgery": 4, "obstetrics": 5, "psychiatry": 6,
                "pediatrics": 7, "cardiothoracic_surgery": 8}
    spec_idx = spec_map.get(case.specialty or "", 0)
    n_labs = 0
    for tok in meta.split():
        if tok.startswith("labs="):
            n_labs = len(tok.split("=", 1)[1].split(","))
    return [float(age_idx), adm_emerg, float(spec_idx), float(n_labs)]


def _tabular_scores_minimal(name: str, cal_cases, test_cases):
    """sklearn classifier with 4-feature minimal vector."""
    clf = _make_classifier(name)
    X_cal = [_feature_vector_minimal(c) for c in cal_cases]
    y_cal = [bool(c.expected_escalate) for c in cal_cases]
    if not clf.fit(X_cal, y_cal):
        return None
    cal_scores = [clf.score(x) for x in X_cal]
    test_scores = [clf.score(_feature_vector_minimal(c)) for c in test_cases]
    return cal_scores, test_scores


def evaluate_one_seed(classifier_name: str, cases, seed: int,
                      n_cal: int, n_test: int) -> dict:
    import random
    print(f"\n── classifier={classifier_name} seed={seed} ──", flush=True)
    cal, test = patient_level_split(cases, group_of=_subject_id,
                                     cal_frac=0.8, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(cal); rng.shuffle(test)
    cal = cal[:n_cal]
    test = test[:n_test]
    print(f"  cal={len(cal)} test={len(test)} (minimal 4-feature)", flush=True)

    is_llm = classifier_name in ("gpt_oss_120b", "lmstudio")
    if is_llm:
        cal_scores = _llm_scores(cal)
        test_scores = _llm_scores(test)
    else:
        result = _tabular_scores_minimal(classifier_name, cal, test)
        if result is None:
            return {"classifier": classifier_name, "seed": seed,
                    "error": "fit failed"}
        cal_scores, test_scores = result

    cal_labels = [bool(c.expected_escalate) for c in cal]
    cal_strata = [(c.scenario_type or "").upper() for c in cal]
    test_labels = [bool(c.expected_escalate) for c in test]
    test_strata = [(c.scenario_type or "").upper() for c in test]

    def _clean(s, l, st_):
        out_s, out_l, out_st = [], [], []
        for sc, lb, x in zip(s, l, st_):
            if sc != sc: continue
            out_s.append(sc); out_l.append(lb); out_st.append(x)
        return out_s, out_l, out_st
    cs, cl, cst = _clean(cal_scores, cal_labels, cal_strata)
    ts, tl, tst = _clean(test_scores, test_labels, test_strata)
    if not cs:
        return {"classifier": classifier_name, "seed": seed,
                "error": "no cal scores"}

    crc = StratifiedConformalRiskControl(alphas=ALPHAS)
    crc.fit(cs, cl, cst)

    per_stratum = {}
    for s, alpha in ALPHAS.items():
        idx = [i for i, x in enumerate(tst) if x == s]
        if not idx:
            per_stratum[s] = None; continue
        lam = crc.threshold_for(s)
        n_pos = sum(tl[i] for i in idx)
        n_neg = sum(1 for i in idx if not tl[i])
        misses = sum(1 for i in idx if tl[i] and ts[i] <= lam)
        over_esc = sum(1 for i in idx if (not tl[i]) and ts[i] > lam)
        miss_rate = (misses / n_pos) if n_pos else None
        over_rate = (over_esc / n_neg) if n_neg else None
        # Vacuous detection: over_esc = 1.0 → "escalate-all" solution
        is_vacuous = (over_rate is not None and over_rate >= 0.99)
        per_stratum[s] = {
            "n": len(idx), "n_pos": n_pos, "n_neg": n_neg,
            "misses": misses, "miss_rate": miss_rate,
            "over_esc": over_esc, "over_esc_rate": over_rate,
            "exact_upper95": clopper_pearson_upper(misses, n_pos, 0.95)
                              if n_pos else None,
            "alpha": alpha,
            "satisfies_alpha": (clopper_pearson_upper(misses, n_pos, 0.95)
                                <= alpha) if n_pos else None,
            "is_vacuous_escalate_all": is_vacuous,
        }
    return {"classifier": classifier_name, "seed": seed,
            "n_cal": len(cs), "n_test": len(ts),
            "per_stratum": per_stratum}


def aggregate_across_seeds(results: list[dict]) -> dict:
    out = {}
    for s in ALPHAS:
        misses_total = 0; n_pos_total = 0
        over_total = 0; n_neg_total = 0
        vacuous_seeds = 0
        for r in results:
            cell = (r.get("per_stratum") or {}).get(s)
            if not cell or cell.get("n_pos", 0) == 0:
                continue
            misses_total += cell["misses"]
            n_pos_total += cell["n_pos"]
            over_total += cell["over_esc"]
            n_neg_total += cell["n_neg"]
            if cell.get("is_vacuous_escalate_all"):
                vacuous_seeds += 1
        if n_pos_total == 0:
            out[s] = None; continue
        pooled_upper = clopper_pearson_upper(misses_total, n_pos_total, 0.95)
        out[s] = {
            "pooled_misses": misses_total,
            "pooled_n_pos": n_pos_total,
            "pooled_miss_rate": misses_total / n_pos_total,
            "pooled_over_esc": over_total,
            "pooled_n_neg": n_neg_total,
            "pooled_over_esc_rate": over_total / n_neg_total if n_neg_total else None,
            "pooled_exact_upper95": pooled_upper,
            "alpha": ALPHAS[s],
            "satisfies_alpha": pooled_upper <= ALPHAS[s],
            "n_vacuous_seeds": vacuous_seeds,
            "is_uniformly_vacuous": vacuous_seeds == 5,
        }
    return out


def write_md(report: dict, out_md: Path):
    lines = ["# Round 11 R11.1 — Method-agnostic CRC with MINIMAL leakage-safe features\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Classifiers: {', '.join(report['classifiers'])}")
    lines.append(f"- n_cal: {report['n_cal']}, n_test: {report['n_test']}")
    lines.append(f"- Seeds: {report['seeds']}\n")
    lines.append("Minimal features: [age_bucket, adm_emerg, spec_idx, n_labs] — "
                  "R10.7 leakage suspect (Charlson, specialty_baseline_rate, n_vital_flags) removed.\n")
    lines.append("**Vacuous CRC detection**: over_esc_rate ≥ 0.99 → escalate-all "
                  "(not genuine win, framework collapse).\n")
    for clf, agg in report["per_classifier"].items():
        lines.append(f"\n## classifier = {clf}\n")
        lines.append("| stratum | α | pooled miss/n_pos | exact 95% upper | "
                     "over_esc/n_neg | over_esc_rate | α satisfies? | VACUOUS? |")
        lines.append("| --- | --- | --- | --- | --- | --- | :---: | :---: |")
        for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            cell = agg.get(s)
            if not cell:
                lines.append(f"| {s} | {ALPHAS[s]} | — | — | — | — | — | — |"); continue
            ok = "✓" if cell["satisfies_alpha"] else "✗"
            vac = "⚠️ ALL" if cell["is_uniformly_vacuous"] else (
                f"{cell['n_vacuous_seeds']}/5 seeds" if cell["n_vacuous_seeds"] > 0 else "✓ no"
            )
            lines.append(
                f"| {s} | {cell['alpha']} | "
                f"{cell['pooled_misses']}/{cell['pooled_n_pos']} | "
                f"{cell['pooled_exact_upper95']:.4f} | "
                f"{cell['pooled_over_esc']}/{cell['pooled_n_neg']} | "
                f"{cell['pooled_over_esc_rate']:.4f} | {ok} | {vac} |"
            )

    # Auto verdict
    lines.append("\n## R11.1 verdict\n")
    by_clf = report["per_classifier"]
    crit_results = {clf: agg.get("CRITICAL", {}) for clf, agg in by_clf.items() if agg.get("CRITICAL")}
    all_vacuous_crit = all(r.get("is_uniformly_vacuous", False) for r in crit_results.values())
    any_genuine = any(
        (not r.get("is_uniformly_vacuous", False)) and r.get("satisfies_alpha", False)
        for r in crit_results.values()
    )
    if all_vacuous_crit:
        lines.append("**🚨 All 5 classifiers produce uniformly vacuous CRC thresholds on CRITICAL.** "
                     "R10.4 의 RF win 이 leakage artifact 가 *아니라* CRC framework 자체의 "
                     "minimal-feature regime 에서의 fundamental limit. paper 의 R10.4 headline "
                     "철회 + 'framework limit at minimal features' 로 reframe.")
    elif any_genuine:
        lines.append("**✓ Some classifier(s) achieve genuine α-satisfaction without "
                     "escalate-all collapse.** R10.4 finding 의 *partial* validation — "
                     "leakage 가 *일부* 원인이지만 fundamental win 도 존재.")
    else:
        lines.append("**Mixed**: 일부 classifier 가 α-satisfy 하나 vacuous; 결과 해석 필요.")

    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", nargs="+",
                    default=["gpt_oss_120b", "logreg", "gbdt",
                             "randomforest", "xgboost"])
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round11" / "r11_1_method_agnostic_minimal")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"JSONL missing: {args.jsonl}")

    print(f"[R11.1] loading {args.jsonl} ...")
    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=42)
    print(f"[R11.1] loaded {len(cases)} cases")
    print(f"[R11.1] MINIMAL features only — Charlson, spec_rate, vital_flags excluded")

    # Per-classifier cache (R10.4 패턴 — 64h wallclock 보호)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out.parent / "r11_1_cache"
    cache_dir.mkdir(exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "classifiers": args.classifiers,
        "n_cal": args.n_cal, "n_test": args.n_test,
        "seeds": args.seeds,
        "features_used": ["age_bucket", "adm_emerg", "spec_idx", "n_labs"],
        "features_removed_for_leakage": [
            "charlson_index (R10.7 L25)",
            "specialty_baseline_rate (R10.7 L26)",
            "n_vital_flags (coverage gaps)",
        ],
        "per_seed": {}, "per_classifier": {},
    }
    for clf_name in args.classifiers:
        cache_path = cache_dir / f"{clf_name}.json"
        if cache_path.exists():
            print(f"\n[CACHE HIT] {clf_name} ← {cache_path}")
            per_seed = json.loads(cache_path.read_text())["per_seed"]
        else:
            per_seed = []
            for seed in args.seeds:
                try:
                    r = evaluate_one_seed(clf_name, cases, seed,
                                          args.n_cal, args.n_test)
                except Exception as e:
                    r = {"classifier": clf_name, "seed": seed,
                         "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    print(f"  [seed {seed} error] {r['error']}", flush=True)
                per_seed.append(r)
                cache_path.write_text(json.dumps({"per_seed": per_seed},
                                                  indent=2, default=str))
            print(f"\n[CACHE SAVED] {clf_name} → {cache_path}")
        report["per_seed"][clf_name] = per_seed
        try:
            report["per_classifier"][clf_name] = aggregate_across_seeds(per_seed)
        except Exception as e:
            report["per_classifier"][clf_name] = {"error": str(e)}

    Path(str(args.out) + ".json").write_text(
        json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")
    print(f"  cache: {cache_dir}/")


if __name__ == "__main__":
    main()
