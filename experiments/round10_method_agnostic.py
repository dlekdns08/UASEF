"""
Round 10 R10.4 — Method-agnostic Stratified CRC head-to-head (HEADLINE).

5 underlying classifier × 동일 StratifiedConformalRiskControl layer
× 5 seed bootstrap CI:

  1. openai/gpt-oss-120b  (LMStudio LLM — Round 9 와 동일)
  2. sklearn LogisticRegression
  3. sklearn GradientBoostingClassifier
  4. sklearn RandomForestClassifier
  5. xgboost.XGBClassifier

핵심 가설 (H1): LLM-120 vs LogReg McNemar pooled p > 0.05 → method-
agnostic claim 지지. H2 (LLM 우세) 도 publishable.

산출: results/round10/r10_4_method_agnostic.{json,md}
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

from data.loader import _load_mimic4_jsonl
from experiments.metrics_utils import (
    clopper_pearson_upper, patient_level_split, bootstrap_ci,
)
from models.stratified_crc import StratifiedConformalRiskControl

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"


# ── Feature vector for tabular classifiers ──────────────────────────────────

def _feature_vector(case) -> list[float]:
    """Decision-time features (leakage-safe) → flat float vector."""
    meta = case.meta_info or ""
    age_bucket = {"unknown": 0, "<18": 1, "18-34": 2, "35-49": 3,
                  "50-64": 4, "65-79": 5, "80+": 6}
    # parse age_bucket from meta
    age_idx = 0
    for tok in meta.split():
        if tok.startswith("age_bucket="):
            age_idx = age_bucket.get(tok.split("=", 1)[1].strip(), 0)
    # admission type one-hot
    adm_emerg = 1.0 if "EMERG" in meta or "URG" in meta else 0.0
    # specialty one-hot (top 8)
    spec_map = {"cardiology": 1, "neurology": 2, "internal_medicine": 3,
                "surgery": 4, "obstetrics": 5, "psychiatry": 6,
                "pediatrics": 7, "cardiothoracic_surgery": 8}
    spec_idx = spec_map.get(case.specialty or "", 0)
    # lab flags (count)
    n_labs = 0
    for tok in meta.split():
        if tok.startswith("labs="):
            n_labs = len(tok.split("=", 1)[1].split(",")) if "=" in tok else 0
    # Round 10 extensions (Charlson, vital flags count, specialty rate)
    # 이 값들은 case.meta_info 에서 추출되거나 case 자체에 추가되어 있어야 함.
    # R10 preprocessing 산출 JSONL 의 row dict 에서 case 만들 때 보존되도록 가정.
    charlson = float(getattr(case, "_charlson_index", 0))
    n_vital = float(getattr(case, "_n_vital_flags", 0))
    spec_rate = float(getattr(case, "_specialty_baseline_rate", 0.0))
    return [
        float(age_idx), adm_emerg, float(spec_idx), float(n_labs),
        charlson, n_vital, spec_rate,
    ]


def _attach_r10_features(cases: list, jsonl_path: Path) -> None:
    """JSONL row 의 R10 extension 을 case 객체에 부착 (loader 가 잃어버린 것)."""
    with open(jsonl_path) as f:
        rows_by_hadm = {}
        for line in f:
            r = json.loads(line)
            rows_by_hadm[str(r.get("hadm_id", ""))] = r
    for c in cases:
        hid = ""
        for tok in (c.meta_info or "").split():
            if tok.startswith("hadm_id="):
                hid = tok.split("=", 1)[1]
        row = rows_by_hadm.get(hid, {})
        c._charlson_index = row.get("charlson_index", 0)
        c._n_vital_flags = len(row.get("vital_flags", []) or [])
        c._specialty_baseline_rate = row.get("specialty_baseline_rate", 0.0)


# ── Classifier wrappers — uniform .fit(X, y) → score(x) ─────────────────────

class _SklearnWrap:
    """sklearn classifier → 비적합 점수 = -predict_proba(positive)."""
    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def fit(self, X, y):
        # sklearn 은 y 가 다양해도 OK 지만 단일 class 면 fail → catch
        ys = list(set(y))
        if len(ys) < 2:
            print(f"  [skip] {self.name}: single class label, no fit possible")
            return False
        self.model.fit(X, y)
        return True

    def score(self, x):
        p = self.model.predict_proba([x])[0]
        # positive class probability 의 negation = 비적합
        # higher → less confident in positive (escalate)
        # classes_ 가 [False, True] 또는 [0, 1] 순서로 가정
        classes = list(self.model.classes_)
        pos_idx = classes.index(True) if True in classes else (
            classes.index(1) if 1 in classes else len(classes) - 1
        )
        return -float(p[pos_idx])


def _make_classifier(name: str):
    """Classifier factory."""
    if name in ("gpt_oss_120b", "lmstudio"):
        return None  # LLM 은 별도 path
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import (
            GradientBoostingClassifier, RandomForestClassifier
        )
    except ImportError as e:
        sys.exit(f"sklearn 필요: uv pip install scikit-learn\n{e}")
    if name == "logreg":
        return _SklearnWrap(
            LogisticRegression(max_iter=1000, random_state=42),
            "LogReg",
        )
    if name == "gbdt":
        return _SklearnWrap(
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            "GBDT",
        )
    if name == "randomforest":
        return _SklearnWrap(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "RandomForest",
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print(f"  [warn] xgboost 미설치, RandomForest 로 fallback")
            from sklearn.ensemble import RandomForestClassifier as RF
            return _SklearnWrap(RF(n_estimators=100, random_state=42), "XGBoost-fallback")
        return _SklearnWrap(
            XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                          use_label_encoder=False, eval_metric="logloss"),
            "XGBoost",
        )
    raise ValueError(f"Unknown classifier: {name}")


# ── LLM scoring (Round 9 baseline 재활용) ──────────────────────────────────

def _llm_scores(cases: list, backend: str = "lmstudio") -> list[float]:
    """Round 9 와 동일 logprob 비적합 — verbose progress."""
    from models.uqm import UQM, compute_nonconformity_score
    from models.model_interface import query_model
    import time
    sys_prompt = UQM.SYSTEM_PROMPT
    scores = []
    t_start = time.perf_counter()
    log_every = max(1, len(cases) // 20)
    for i, c in enumerate(cases):
        phi_taint = (c.source or "").startswith("mimic4")
        try:
            resp = query_model(backend, sys_prompt, c.question, temperature=0.0,
                                phi_taint=phi_taint)
            scores.append(compute_nonconformity_score(resp))
        except Exception as e:
            scores.append(float("nan"))
            if (i < 3):
                print(f"  [skip {i}] {type(e).__name__}: {str(e)[:100]}", flush=True)
        if (i + 1) % log_every == 0 or i + 1 == len(cases):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed else 0
            print(f"  [LLM] {i+1}/{len(cases)} ({rate:.2f}/s)", flush=True)
    return scores


def _subject_id(case) -> str:
    for tok in (case.meta_info or "").split():
        if tok.startswith("subject_id="):
            return tok.split("=", 1)[1]
    return f"h:{id(case)}"


# ── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_one_seed(classifier_name: str, cases: list, seed: int,
                      n_cal: int, n_test: int) -> dict:
    print(f"\n── classifier={classifier_name} seed={seed} ──", flush=True)
    cal, test = patient_level_split(cases, group_of=_subject_id,
                                     cal_frac=0.8, seed=seed)
    # cap to (n_cal, n_test) 한도
    rng = random.Random(seed)
    rng.shuffle(cal); rng.shuffle(test)
    cal = cal[:n_cal]
    test = test[:n_test]
    print(f"  cal={len(cal)} test={len(test)}", flush=True)

    # score
    is_llm = classifier_name in ("gpt_oss_120b", "lmstudio")
    if is_llm:
        cal_scores = _llm_scores(cal)
        test_scores = _llm_scores(test)
    else:
        clf = _make_classifier(classifier_name)
        X_cal = [_feature_vector(c) for c in cal]
        y_cal = [bool(c.expected_escalate) for c in cal]
        if not clf.fit(X_cal, y_cal):
            return {"classifier": classifier_name, "seed": seed,
                    "error": "single-class fit"}
        cal_scores = [clf.score(x) for x in X_cal]
        X_test = [_feature_vector(c) for c in test]
        test_scores = [clf.score(x) for x in X_test]

    cal_labels = [bool(c.expected_escalate) for c in cal]
    cal_strata = [(c.scenario_type or "").upper() for c in cal]
    test_labels = [bool(c.expected_escalate) for c in test]
    test_strata = [(c.scenario_type or "").upper() for c in test]

    # filter NaN (LLM 실패)
    def _clean(scores, labels, strata):
        out_s, out_l, out_st = [], [], []
        for sc, lb, st_ in zip(scores, labels, strata):
            if sc != sc:  # NaN
                continue
            out_s.append(sc); out_l.append(lb); out_st.append(st_)
        return out_s, out_l, out_st

    cs, cl, cst = _clean(cal_scores, cal_labels, cal_strata)
    ts, tl, tst = _clean(test_scores, test_labels, test_strata)
    if not cs:
        return {"classifier": classifier_name, "seed": seed,
                "error": "no valid cal scores"}

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
        # over-esc
        n_neg = sum(1 for i in idx if not tl[i])
        over_esc = sum(1 for i in idx if (not tl[i]) and ts[i] > lam)
        per_stratum[s] = {
            "n": len(idx), "n_pos": n_pos, "misses": misses,
            "miss_rate": (misses / n_pos) if n_pos else None,
            "exact_upper95": clopper_pearson_upper(misses, n_pos, 0.95)
                              if n_pos else None,
            "over_esc_rate": (over_esc / n_neg) if n_neg else None,
            "alpha": alpha,
            "satisfies_alpha": (clopper_pearson_upper(misses, n_pos, 0.95)
                                <= alpha) if n_pos else None,
        }
    return {"classifier": classifier_name, "seed": seed,
            "n_cal": len(cs), "n_test": len(ts),
            "per_stratum": per_stratum}


def aggregate_across_seeds(results: list[dict]) -> dict:
    """5 시드 pooled — pooled exact CP upper + bootstrap CI of miss_rate."""
    out = {}
    for s in ALPHAS:
        misses_total = 0
        n_pos_total = 0
        miss_rates = []
        for r in results:
            cell = (r.get("per_stratum") or {}).get(s)
            if not cell or cell.get("n_pos", 0) == 0:
                continue
            misses_total += cell["misses"]
            n_pos_total += cell["n_pos"]
            miss_rates.append(cell["miss_rate"])
        if n_pos_total == 0:
            out[s] = None; continue
        pooled_upper = clopper_pearson_upper(misses_total, n_pos_total, 0.95)
        boot = bootstrap_ci(miss_rates, statistic_fn=lambda x: sum(x) / len(x),
                             n_iter=1000, seed=42) if len(miss_rates) >= 5 else None
        out[s] = {
            "n_seeds_pooled": len(miss_rates),
            "pooled_misses": misses_total,
            "pooled_n_pos": n_pos_total,
            "pooled_miss_rate": misses_total / n_pos_total,
            "pooled_exact_upper95": pooled_upper,
            "miss_rate_mean": st.mean(miss_rates),
            "miss_rate_std": st.stdev(miss_rates) if len(miss_rates) > 1 else 0.0,
            "miss_rate_ci95": boot,
            "alpha": ALPHAS[s],
            "satisfies_alpha": pooled_upper <= ALPHAS[s],
        }
    return out


def write_md(report: dict, out_md: Path):
    lines = ["# Round 10 R10.4 — Method-agnostic CRC head-to-head (MIMIC-IV)\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Seeds: {report['seeds']}")
    lines.append(f"- Classifiers: {', '.join(report['classifiers'])}")
    lines.append(f"- n_cal target / n_test target: {report['n_cal']} / {report['n_test']}\n")
    lines.append("각 cell: pooled `misses/n_pos`, exact CP 95% upper, α 만족 여부.\n")
    for clf, agg in report["per_classifier"].items():
        lines.append(f"## classifier = {clf}\n")
        lines.append("| stratum | α | n_seeds | pooled miss / n_pos | exact 95% upper | bootstrap CI of miss | satisfies α? |")
        lines.append("| --- | --- | --- | --- | --- | --- | :---: |")
        for s, a in agg.items():
            if a is None:
                lines.append(f"| {s} | {ALPHAS[s]} | 0 | — | — | — | — |"); continue
            ci = a.get("miss_rate_ci95")
            ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
            ok = "✓" if a["satisfies_alpha"] else "✗"
            lines.append(
                f"| {s} | {a['alpha']} | {a['n_seeds_pooled']} | "
                f"{a['pooled_misses']}/{a['pooled_n_pos']} | "
                f"{a['pooled_exact_upper95']:.5f} | {ci_s} | {ok} |"
            )
        lines.append("")
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
                    default=ROOT / "results" / "round10" / "r10_4_method_agnostic")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"[R10.4] preprocessed JSONL 미존재: {args.jsonl}\n"
                 f"  먼저 round10_mimic4_preprocess.py 실행.")

    print(f"[R10.4] loading {args.jsonl} ...")
    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=42)
    print(f"[R10.4] loaded {len(cases)} cases")
    _attach_r10_features(cases, args.jsonl)

    report = {
        "timestamp": datetime.now().isoformat(),
        "classifiers": args.classifiers,
        "n_cal": args.n_cal, "n_test": args.n_test,
        "seeds": args.seeds,
        "per_seed": {}, "per_classifier": {},
    }
    for clf_name in args.classifiers:
        per_seed = []
        for seed in args.seeds:
            r = evaluate_one_seed(clf_name, cases, seed, args.n_cal, args.n_test)
            per_seed.append(r)
        report["per_seed"][clf_name] = per_seed
        report["per_classifier"][clf_name] = aggregate_across_seeds(per_seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(
        json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
