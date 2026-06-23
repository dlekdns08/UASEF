"""
Round 11 R11.3 — eICU cross-center replication of audit discipline.

H1 (pre-registered): eICU 코호트에서 *동일한 leakage features* (charlson-like,
specialty_baseline_rate, apache_predicted_mort) 가 *동일한 RF vacuous win* 을
재현한다 → audit discipline 의 generalizability 입증.

Two passes:
   Pass A (full features, R10.4 등가 + APACHE leakage):
      [age, adm_emerg, spec, n_labs, charlson, n_vital, spec_rate,
       apache_score, apache_predicted_mort]
   Pass B (minimal, R11.1 등가):
      [age, adm_emerg, spec, n_labs]

기대 결과:
   - Pass A: RF over_esc = 1.0 (모든 case 를 escalate) + 0/n miss = leakage-induced vacuous
   - Pass B: 4 features → CRITICAL miss 5-15%, no α-satisfy
   - Pattern 이 MIMIC-IV 와 일치 → audit discipline 의 단순 cross-center 효과

산출: results/round11/r11_3_eicu_{pass_a,pass_b}.{json,md}
Wallclock: ~30분 tabular, ~64h LLM (optional, --skip-llm)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from experiments.metrics_utils import clopper_pearson_upper, patient_level_split
from experiments.round10_method_agnostic import _make_classifier, _llm_scores

DEFAULT_JSONL = ROOT / "data" / "raw" / "eicu_cases_v11.jsonl"
ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}


# ─────────────────────────────────────────────────────────────────────────────
# Feature vectors — Pass A vs Pass B
# ─────────────────────────────────────────────────────────────────────────────

AGE_BUCKET = {"unknown": 0, "<18": 1, "18-34": 2, "35-49": 3,
              "50-64": 4, "65-79": 5, "80+": 6}
SPEC_MAP = {"cardiology": 1, "neurology": 2, "internal_medicine": 3,
            "surgery": 4, "obstetrics": 5, "psychiatry": 6,
            "pediatrics": 7, "cardiothoracic_surgery": 8}


def _parse_meta(case):
    meta = case.get("meta_info") or ""
    age_idx = 0
    for tok in meta.split():
        if tok.startswith("age_bucket="):
            age_idx = AGE_BUCKET.get(tok.split("=", 1)[1].strip(), 0)
    adm_emerg = 1.0 if ("EMERG" in meta or "URG" in meta) else 0.0
    spec_idx = SPEC_MAP.get(case.get("specialty") or "", 0)
    n_labs = 0
    for tok in meta.split():
        if tok.startswith("labs="):
            v = tok.split("=", 1)[1]
            n_labs = len(v.split(",")) if v else 0
    return age_idx, adm_emerg, spec_idx, n_labs


def fv_pass_b(case) -> list[float]:
    """Minimal 4-feature (R11.1 등가)."""
    a, e, s, l = _parse_meta(case)
    return [float(a), e, float(s), float(l)]


def fv_pass_a(case) -> list[float]:
    """Full 9-feature with eICU leakage suspects."""
    a, e, s, l = _parse_meta(case)
    return [
        float(a), e, float(s), float(l),
        float(case.get("_charlson_index", 0)),
        float(case.get("_n_vital_flags", 0)),
        float(case.get("_specialty_baseline_rate", 0.0)),
        float(case.get("_apache_score", 0)),
        float(case.get("_apache_predicted_mort", 0)),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Lite EHRCase wrapper (loader 안 거치고 dict 그대로 사용)
# ─────────────────────────────────────────────────────────────────────────────

class EHRCaseLite:
    """Lightweight wrapper providing the few attrs CRC/loader needs."""
    __slots__ = ("hadm_id", "subject_id", "specialty", "scenario_type",
                 "expected_escalate", "meta_info", "question",
                 "_charlson_index", "_n_vital_flags",
                 "_specialty_baseline_rate", "_apache_score",
                 "_apache_predicted_mort")
    def __init__(self, d: dict):
        self.hadm_id = d.get("hadm_id"); self.subject_id = d.get("subject_id")
        self.specialty = d.get("specialty")
        self.scenario_type = d.get("stratum")
        self.expected_escalate = bool(d.get("expected_escalate"))
        self.meta_info = d.get("meta_info") or ""
        self.question = d.get("question") or ""
        self._charlson_index = d.get("_charlson_index", 0)
        self._n_vital_flags = d.get("_n_vital_flags", 0)
        self._specialty_baseline_rate = d.get("_specialty_baseline_rate", 0.0)
        self._apache_score = d.get("_apache_score", 0)
        self._apache_predicted_mort = d.get("_apache_predicted_mort", 0)


def load_eicu_jsonl(p: Path) -> list[EHRCaseLite]:
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            out.append(EHRCaseLite(d))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CRC evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _subject_id(c) -> str:
    return c.subject_id


def evaluate_one(classifier_name: str, cases, seed, n_cal, n_test,
                 fv_fn, use_llm=False) -> dict:
    cal, test = patient_level_split(cases, group_of=_subject_id,
                                      cal_frac=0.8, seed=seed)
    rng = random.Random(seed); rng.shuffle(cal); rng.shuffle(test)
    # 0 또는 음수면 split 전체 사용 (small cohort 대응)
    if n_cal and n_cal > 0:  cal = cal[:n_cal]
    if n_test and n_test > 0: test = test[:n_test]

    def _case_to_dict(c):
        return {k: getattr(c, k) for k in EHRCaseLite.__slots__}

    if use_llm:
        cal_scores = _llm_scores(cal); test_scores = _llm_scores(test)
    else:
        clf = _make_classifier(classifier_name)
        X_cal = [fv_fn(_case_to_dict(c)) for c in cal]
        y_cal = [bool(c.expected_escalate) for c in cal]
        if not clf.fit(X_cal, y_cal):
            return {"classifier": classifier_name, "seed": seed,
                    "error": "fit failed"}
        cal_scores = [clf.score(x) for x in X_cal]
        X_test = [fv_fn(_case_to_dict(c)) for c in test]
        test_scores = [clf.score(x) for x in X_test]

    cal_labels = [bool(c.expected_escalate) for c in cal]
    cal_strata = [(c.scenario_type or "").upper() for c in cal]
    test_labels = [bool(c.expected_escalate) for c in test]
    test_strata = [(c.scenario_type or "").upper() for c in test]

    def _clean(s, l, st):
        os, ol, ost = [], [], []
        for sc, lb, x in zip(s, l, st):
            if sc != sc: continue
            os.append(sc); ol.append(lb); ost.append(x)
        return os, ol, ost
    cs, cl, cst = _clean(cal_scores, cal_labels, cal_strata)
    ts, tl, tst = _clean(test_scores, test_labels, test_strata)
    if not cs:
        return {"classifier": classifier_name, "seed": seed, "error": "no cal"}

    from models.stratified_crc import StratifiedConformalRiskControl
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
        upper = clopper_pearson_upper(misses, n_pos, 0.95) if n_pos else None
        is_vac = (over_esc / n_neg) >= 0.99 if n_neg else False
        per_stratum[s] = {
            "n": len(idx), "n_pos": n_pos, "n_neg": n_neg,
            "misses": misses,
            "miss_rate": (misses / n_pos) if n_pos else None,
            "over_esc": over_esc,
            "over_esc_rate": (over_esc / n_neg) if n_neg else None,
            "exact_upper95": upper, "alpha": alpha,
            "satisfies_alpha": (upper <= alpha) if upper is not None else None,
            "is_vacuous_escalate_all": is_vac,
        }
    return {
        "classifier": classifier_name, "seed": seed,
        "n_cal": len(cs), "n_test": len(ts),
        "per_stratum": per_stratum,
    }


def aggregate(results):
    out = {}
    for s in ALPHAS:
        misses = npos = oe = nneg = 0
        vac = 0
        for r in results:
            cell = (r.get("per_stratum") or {}).get(s)
            if not cell or cell.get("n_pos", 0) == 0: continue
            misses += cell["misses"]; npos += cell["n_pos"]
            oe += cell["over_esc"]; nneg += cell["n_neg"]
            if cell.get("is_vacuous_escalate_all"): vac += 1
        if npos == 0: out[s] = None; continue
        upper = clopper_pearson_upper(misses, npos, 0.95)
        out[s] = {
            "pooled_misses": misses, "pooled_n_pos": npos,
            "pooled_miss_rate": misses / npos,
            "pooled_over_esc": oe, "pooled_n_neg": nneg,
            "pooled_over_esc_rate": (oe / nneg) if nneg else None,
            "pooled_exact_upper95": upper, "alpha": ALPHAS[s],
            "satisfies_alpha": upper <= ALPHAS[s],
            "n_vacuous_seeds": vac,
            "is_uniformly_vacuous": vac == len(results),
        }
    return out


def write_md(report, out_md):
    lines = [f"# R11.3 eICU — {report['pass_name']}\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Pass: **{report['pass_name']}**")
    lines.append(f"- Features: {report['feature_set']}")
    lines.append(f"- Classifiers: {', '.join(report['classifiers'])}")
    lines.append(f"- n_cal: {report['n_cal']}, n_test: {report['n_test']}, seeds: {report['seeds']}\n")
    for clf, agg in report["per_classifier"].items():
        lines.append(f"\n## {clf}\n")
        lines.append("| stratum | α | pooled miss/n_pos | exact 95% upper | over_esc rate | α-satisfy? | vacuous? |")
        lines.append("| --- | --- | --- | --- | --- | :---: | :---: |")
        for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            cell = agg.get(s)
            if not cell:
                lines.append(f"| {s} | {ALPHAS[s]} | — | — | — | — | — |"); continue
            ok = "✓" if cell["satisfies_alpha"] else "✗"
            vac = ("⚠ ALL" if cell["is_uniformly_vacuous"]
                   else (f"{cell['n_vacuous_seeds']}/{len(report['seeds'])}"
                         if cell["n_vacuous_seeds"] > 0 else "✓ no"))
            lines.append(
                f"| {s} | {cell['alpha']} | "
                f"{cell['pooled_misses']}/{cell['pooled_n_pos']} | "
                f"{cell['pooled_exact_upper95']:.4f} | "
                f"{cell['pooled_over_esc_rate']:.4f} | {ok} | {vac} |"
            )
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--passes", nargs="+", default=["pass_a", "pass_b"])
    ap.add_argument("--classifiers", nargs="+",
                    default=["logreg", "gbdt", "randomforest", "xgboost"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--include-llm", action="store_true",
                    help="LLM gpt_oss_120b 도 평가 (느림 ~64h)")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "round11")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"eICU JSONL missing: {args.jsonl}. Run round11_eicu_preprocess.py first.")

    print(f"[R11.3] loading {args.jsonl}")
    cases = load_eicu_jsonl(args.jsonl)
    print(f"[R11.3] loaded {len(cases)} cases\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    classifiers = list(args.classifiers)
    if args.include_llm and "gpt_oss_120b" not in classifiers:
        classifiers = ["gpt_oss_120b"] + classifiers

    fv_map = {"pass_a": (fv_pass_a, "full 9-feature (leakage suspects: charlson, spec_rate, APACHE)"),
              "pass_b": (fv_pass_b, "minimal 4-feature (age, adm_emerg, spec, n_labs)")}

    for pass_name in args.passes:
        fv_fn, feat_label = fv_map[pass_name]
        print(f"\n══ {pass_name}: {feat_label} ══")
        report = {
            "timestamp": datetime.now().isoformat(),
            "pass_name": pass_name, "feature_set": feat_label,
            "classifiers": classifiers, "seeds": args.seeds,
            "n_cal": args.n_cal, "n_test": args.n_test,
            "per_seed": {}, "per_classifier": {},
        }
        for clf_name in classifiers:
            per_seed_res = []
            use_llm = clf_name == "gpt_oss_120b"
            for seed in args.seeds:
                print(f"  {clf_name} seed={seed}", flush=True)
                try:
                    r = evaluate_one(clf_name, cases, seed,
                                       args.n_cal, args.n_test, fv_fn,
                                       use_llm=use_llm)
                except Exception as e:
                    r = {"classifier": clf_name, "seed": seed,
                         "error": f"{type(e).__name__}: {str(e)[:200]}"}
                per_seed_res.append(r)
            report["per_seed"][clf_name] = per_seed_res
            report["per_classifier"][clf_name] = aggregate(per_seed_res)

        out_base = args.out_dir / f"r11_3_eicu_{pass_name}"
        Path(str(out_base) + ".json").write_text(json.dumps(report, indent=2, default=str))
        write_md(report, Path(str(out_base) + ".md"))
        print(f"  ✅ {out_base}.{{json,md}}")

    # H1 verdict 자동 도출
    pa = args.out_dir / "r11_3_eicu_pass_a.json"
    pb = args.out_dir / "r11_3_eicu_pass_b.json"
    if pa.exists() and pb.exists():
        a = json.loads(pa.read_text())
        b = json.loads(pb.read_text())
        crit_a_rf = (a.get("per_classifier", {}).get("randomforest") or {}).get("CRITICAL", {})
        crit_b_rf = (b.get("per_classifier", {}).get("randomforest") or {}).get("CRITICAL", {})
        verdict = {}
        if crit_a_rf and crit_b_rf:
            a_vac = crit_a_rf.get("is_uniformly_vacuous", False)
            a_satisfies = crit_a_rf.get("satisfies_alpha", False)
            b_satisfies = crit_b_rf.get("satisfies_alpha", False)
            if a_vac and a_satisfies and not b_satisfies:
                verdict["H1"] = "CONFIRMED"
                verdict["msg"] = ("eICU Pass A RF: vacuous + α-satisfy (vacuous win 재현). "
                                  "Pass B RF: α 미만족 (leakage 제거 시 collapse). "
                                  "MIMIC-IV 의 R10.4 → R11.1 pattern 이 eICU 에서도 동일하게 재현됨 → "
                                  "audit discipline 의 cross-center generalizability 입증.")
            elif a_satisfies and b_satisfies:
                verdict["H1"] = "REJECTED"
                verdict["msg"] = ("eICU 에서는 RF 가 minimal feature 만으로도 α-satisfy. "
                                  "MIMIC-IV pattern 이 cohort-specific 였음.")
            else:
                verdict["H1"] = "PARTIAL"
                verdict["msg"] = f"Pass A RF α-satisfy={a_satisfies}, Pass B RF α-satisfy={b_satisfies} — manual interpretation"
        (args.out_dir / "r11_3_eicu_verdict.json").write_text(
            json.dumps(verdict, indent=2, default=str))
        print(f"\n  H1 verdict: {verdict.get('H1', 'unclear')}")
        print(f"  {verdict.get('msg', '')}")


if __name__ == "__main__":
    main()
