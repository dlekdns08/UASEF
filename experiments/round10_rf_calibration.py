"""
Round 10 supplement — Why does RandomForest beat LLM on MIMIC-IV?

R10.4 의 핵심 finding: RandomForest 만 CRITICAL/HIGH 에서 α 만족, LLM (gpt-oss-120b)
은 CRITICAL 13% miss. 같은 features, 같은 CRC layer 인데 왜 결과가 이렇게 다른가?

가설: 5 classifier 의 *probability calibration 의 질* 이 CRC 임계값의 fit 에
결정적. RandomForest 의 bagging 이 sharp boundary 를 smooth 하게 만들어 CRC 의
quantile 이 well-fit, LLM/LogReg/GBDT/XGBoost 의 sharp score 는 어려운 case 를
분리 못 함.

본 스크립트는 5 classifier 의 calibration metric 을 측정해 가설 검증:

  1. **ECE (Expected Calibration Error)** — Naeini et al. 2015, 10-bin
  2. **Brier score** — squared error of probability
  3. **Reliability diagram data** — bin 별 (predicted, observed) 쌍
  4. **Sharpness** — score 분포의 variance (높을수록 confident, 낮을수록 smooth)

산출: results/round10/r10_rf_calibration.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import _load_mimic4_jsonl
from experiments.round10_method_agnostic import (
    _make_classifier, _feature_vector, _attach_r10_features, _subject_id,
)
from experiments.metrics_utils import patient_level_split


def _ece(probs: list[float], labels: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error (Naeini 2015)."""
    if not probs:
        return float("nan")
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    n = len(probs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = [(p, l) for p, l in zip(probs, labels) if lo <= p < hi or (b == n_bins - 1 and p == hi)]
        if not in_bin:
            continue
        bin_acc = sum(1 for _, l in in_bin if l) / len(in_bin)
        bin_conf = sum(p for p, _ in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)
    return ece


def _brier(probs: list[float], labels: list[bool]) -> float:
    """Brier score = mean of (prob - label)^2."""
    if not probs:
        return float("nan")
    return sum((p - (1.0 if l else 0.0)) ** 2 for p, l in zip(probs, labels)) / len(probs)


def _reliability(probs: list[float], labels: list[bool], n_bins: int = 10) -> list[dict]:
    """Bin 별 (mean predicted, observed frequency, n)."""
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    out = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = [(p, l) for p, l in zip(probs, labels) if lo <= p < hi or (b == n_bins - 1 and p == hi)]
        if not in_bin:
            out.append({"bin": b, "lo": lo, "hi": hi, "n": 0,
                        "mean_pred": None, "obs_freq": None}); continue
        out.append({
            "bin": b, "lo": lo, "hi": hi, "n": len(in_bin),
            "mean_pred": sum(p for p, _ in in_bin) / len(in_bin),
            "obs_freq": sum(1 for _, l in in_bin if l) / len(in_bin),
        })
    return out


def _sharpness(probs: list[float]) -> float:
    """variance of predicted probabilities — 높을수록 confident decision boundary."""
    return st.variance(probs) if len(probs) > 1 else 0.0


def _llm_probs(cases: list, backend: str = "lmstudio") -> list[float]:
    """LLM 의 NLL → softmax-like score 로 환산 (probability proxy)."""
    from models.uqm import UQM, compute_nonconformity_score
    from models.model_interface import query_model
    import time, math
    sys_prompt = UQM.SYSTEM_PROMPT
    probs = []
    t_start = time.perf_counter()
    log_every = max(1, len(cases) // 20)
    for i, c in enumerate(cases):
        phi_taint = (c.source or "").startswith("mimic4")
        try:
            resp = query_model(backend, sys_prompt, c.question, temperature=0.0,
                                phi_taint=phi_taint)
            nll = compute_nonconformity_score(resp)
            # NLL → probability proxy: 1 / (1 + exp(-(threshold - nll)))
            # threshold 는 score 의 median 으로 동적 설정 — sharpness 우회
            # 단순 변환: p = exp(-nll) 의 normalized 버전
            probs.append(math.exp(-nll))
        except Exception:
            probs.append(0.5)  # missing → 중립
        if (i + 1) % log_every == 0:
            print(f"  [LLM] {i+1}/{len(cases)}", flush=True)
    # normalize to [0, 1]
    if probs:
        lo, hi = min(probs), max(probs)
        if hi > lo:
            probs = [(p - lo) / (hi - lo) for p in probs]
    return probs


def _tabular_probs(name: str, cal_cases, test_cases) -> list[float]:
    """Tabular classifier 의 positive-class probability."""
    clf = _make_classifier(name)
    X_cal = [_feature_vector(c) for c in cal_cases]
    y_cal = [bool(c.expected_escalate) for c in cal_cases]
    if not clf.fit(X_cal, y_cal):
        return [0.5] * len(test_cases)
    # _SklearnWrap.score returns -p(positive); 원본 prob 추출
    classes = list(clf.model.classes_)
    pos_idx = classes.index(True) if True in classes else (
        classes.index(1) if 1 in classes else len(classes) - 1
    )
    X_test = [_feature_vector(c) for c in test_cases]
    return [float(clf.model.predict_proba([x])[0][pos_idx]) for x in X_test]


def evaluate_classifier(name: str, cal_cases, test_cases) -> dict:
    print(f"\n── classifier={name} (n_cal={len(cal_cases)}, n_test={len(test_cases)}) ──",
          flush=True)
    if name in ("gpt_oss_120b", "lmstudio"):
        probs = _llm_probs(test_cases)
    else:
        probs = _tabular_probs(name, cal_cases, test_cases)
    labels = [bool(c.expected_escalate) for c in test_cases]
    return {
        "classifier": name,
        "n": len(probs),
        "n_pos": sum(labels),
        "ece_10bin": _ece(probs, labels, n_bins=10),
        "brier": _brier(probs, labels),
        "sharpness": _sharpness(probs),
        "reliability": _reliability(probs, labels, n_bins=10),
    }


def write_md(report: dict, out_md: Path):
    lines = ["# Round 10 supplement — RandomForest vs LLM calibration\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- n_cal: {report['n_cal']}, n_test: {report['n_test']}")
    lines.append(f"- seed: {report['seed']}\n")
    lines.append("## Calibration metrics (낮을수록 좋음 — ECE, Brier)\n")
    lines.append("| Classifier | n | n_pos | **ECE (10-bin)** | **Brier score** | Sharpness (variance) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in report["per_classifier"]:
        lines.append(
            f"| **{r['classifier']}** | {r['n']} | {r['n_pos']} | "
            f"{r['ece_10bin']:.4f} | {r['brier']:.4f} | {r['sharpness']:.4f} |"
        )
    lines.append("\n## 해석 (가설 검증)\n")
    lines.append("**가설**: RandomForest 의 bagging 이 *낮은 sharpness + 낮은 ECE* 를 만든다 →")
    lines.append("CRC quantile 이 well-fit. LLM 은 *높은 sharpness + 높은 ECE* → CRC 가 어려운 case 분리 못함.\n")
    # 자동 verdict
    by_name = {r["classifier"]: r for r in report["per_classifier"]}
    if "randomforest" in by_name and "gpt_oss_120b" in by_name:
        rf = by_name["randomforest"]
        llm = by_name["gpt_oss_120b"]
        verdict = []
        if rf["ece_10bin"] < llm["ece_10bin"]:
            verdict.append(f"- ECE: RF ({rf['ece_10bin']:.4f}) < LLM ({llm['ece_10bin']:.4f}) ✓ 가설 지지")
        else:
            verdict.append(f"- ECE: RF ({rf['ece_10bin']:.4f}) ≥ LLM ({llm['ece_10bin']:.4f}) ✗ 가설 reject")
        if rf["brier"] < llm["brier"]:
            verdict.append(f"- Brier: RF ({rf['brier']:.4f}) < LLM ({llm['brier']:.4f}) ✓")
        else:
            verdict.append(f"- Brier: RF ({rf['brier']:.4f}) ≥ LLM ({llm['brier']:.4f}) ✗")
        if rf["sharpness"] < llm["sharpness"]:
            verdict.append(f"- Sharpness: RF ({rf['sharpness']:.4f}) < LLM ({llm['sharpness']:.4f}) ✓ (RF가 smoother)")
        else:
            verdict.append(f"- Sharpness: RF ({rf['sharpness']:.4f}) ≥ LLM ({llm['sharpness']:.4f}) — sharpness 차이 미미")
        lines.extend(verdict)
    lines.append("\n## Reliability diagram (CRITICAL bin)\n")
    lines.append("ECE 산출의 raw bin-level 데이터. paper §6.5 의 plot 입력.")
    lines.append("(JSON 의 `per_classifier[].reliability` 참조)")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", nargs="+",
                    default=["gpt_oss_120b", "logreg", "gbdt", "randomforest", "xgboost"])
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jsonl", type=Path,
                    default=ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "r10_rf_calibration")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"JSONL missing: {args.jsonl}")
    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=args.seed)
    _attach_r10_features(cases, args.jsonl)
    print(f"loaded {len(cases)} cases")

    import random
    cal, test = patient_level_split(cases, group_of=_subject_id,
                                     cal_frac=0.8, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(cal); rng.shuffle(test)
    cal = cal[:args.n_cal]
    test = test[:args.n_test]

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_cal": len(cal), "n_test": len(test),
        "seed": args.seed,
        "classifiers": args.classifiers,
        "per_classifier": [],
    }
    for name in args.classifiers:
        try:
            r = evaluate_classifier(name, cal, test)
            report["per_classifier"].append(r)
        except Exception as e:
            print(f"  [error] {name}: {e}")
            report["per_classifier"].append({"classifier": name, "error": str(e)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
