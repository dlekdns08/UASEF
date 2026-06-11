"""
Round 10 R10.7 — Expanded-feature validation.

Round 9 (basic features only) vs Round 10 (Charlson + vital quartile +
specialty rate) 의 MODERATE/LOW miss rate 개선 효과 측정.

LogReg + CRC 로 빠르게 (LLM 호출 없음).
산출: results/round10/r10_7_feature_expansion.{json,md}
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
    _make_classifier, _feature_vector, _attach_r10_features, _subject_id,
)
from models.stratified_crc import StratifiedConformalRiskControl

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"
DEFAULT_JSONL_R9 = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases.jsonl"


def _feature_vector_basic(case):
    """Round 9 만의 basic feature — R10 extension 제거."""
    full = _feature_vector(case)
    # full = [age_idx, adm_emerg, spec_idx, n_labs, charlson, n_vital, spec_rate]
    return full[:4]  # last 3 (R10 extensions) 제거


def _evaluate(cases, feature_fn, seed: int):
    cal, test = patient_level_split(cases, group_of=_subject_id,
                                     cal_frac=0.8, seed=seed)
    clf = _make_classifier("logreg")
    X_cal = [feature_fn(c) for c in cal]
    y_cal = [bool(c.expected_escalate) for c in cal]
    if not clf.fit(X_cal, y_cal):
        return None
    cal_scores = [clf.score(x) for x in X_cal]
    cal_labels = y_cal
    cal_strata = [(c.scenario_type or "").upper() for c in cal]
    test_scores = [clf.score(feature_fn(c)) for c in test]
    test_labels = [bool(c.expected_escalate) for c in test]
    test_strata = [(c.scenario_type or "").upper() for c in test]

    crc = StratifiedConformalRiskControl(alphas=ALPHAS)
    crc.fit(cal_scores, cal_labels, cal_strata)
    per_stratum = {}
    for s, alpha in ALPHAS.items():
        idx = [i for i, x in enumerate(test_strata) if x == s]
        if not idx:
            per_stratum[s] = None; continue
        lam = crc.threshold_for(s)
        n_pos = sum(test_labels[i] for i in idx)
        misses = sum(1 for i in idx if test_labels[i] and test_scores[i] <= lam)
        miss_rate = (misses / n_pos) if n_pos else None
        per_stratum[s] = {
            "n_pos": n_pos, "misses": misses, "miss_rate": miss_rate,
            "exact_upper95": clopper_pearson_upper(misses, n_pos, 0.95)
                              if n_pos else None,
        }
    return per_stratum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--backends", nargs="+", default=["lmstudio"])  # CLI parity
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round10" / "r10_7_feature_expansion")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"[R10.7] preprocessed JSONL 미존재: {args.jsonl}")

    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=42)
    _attach_r10_features(cases, args.jsonl)
    print(f"[R10.7] loaded {len(cases)} cases (v10 JSONL with extended features)")

    report = {
        "timestamp": datetime.now().isoformat(),
        "seeds": args.seeds,
        "per_seed": {"basic": [], "expanded": []},
        "summary": {},
    }
    for seed in args.seeds:
        print(f"\n── seed={seed} ──", flush=True)
        report["per_seed"]["basic"].append(_evaluate(cases, _feature_vector_basic, seed))
        report["per_seed"]["expanded"].append(_evaluate(cases, _feature_vector, seed))

    # aggregate
    for variant in ["basic", "expanded"]:
        agg = {}
        for s in ALPHAS:
            misses_list = []
            for r in report["per_seed"][variant]:
                if r is None: continue
                cell = (r or {}).get(s)
                if cell and cell.get("miss_rate") is not None:
                    misses_list.append(cell["miss_rate"])
            if not misses_list:
                agg[s] = None; continue
            agg[s] = {
                "miss_rate_mean": st.mean(misses_list),
                "miss_rate_std": st.stdev(misses_list) if len(misses_list) > 1 else 0.0,
                "n_seeds": len(misses_list),
            }
        report["summary"][variant] = agg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))

    # md
    lines = ["# Round 10 R10.7 — Expanded-feature validation\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Seeds: {args.seeds}\n")
    lines.append("Round 9 basic features vs Round 10 expanded (Charlson + vital quartile + specialty rate)\n")
    lines.append("| stratum | basic miss (mean ± std) | expanded miss (mean ± std) | improvement |")
    lines.append("| --- | --- | --- | --- |")
    for s in ALPHAS:
        b = report["summary"]["basic"].get(s)
        e = report["summary"]["expanded"].get(s)
        if b is None or e is None:
            lines.append(f"| {s} | — | — | — |"); continue
        diff = b["miss_rate_mean"] - e["miss_rate_mean"]
        lines.append(f"| {s} | "
                     f"{b['miss_rate_mean']:.4f} ± {b['miss_rate_std']:.4f} | "
                     f"{e['miss_rate_mean']:.4f} ± {e['miss_rate_std']:.4f} | "
                     f"{diff:+.4f} |")
    Path(str(args.out) + ".md").write_text("\n".join(lines))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
