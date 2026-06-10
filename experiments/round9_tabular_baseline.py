"""
Round 9 R9.6 — Tabular (classical-ML) baselines on MIMIC-IV  (REVISION_PLAN P0-6)
══════════════════════════════════════════════════════════════════════════════

리뷰 #5 방어: decision-time 구조화 feature 만으로 LogReg / GBDT 를 학습해 동일한
StratifiedConformalRiskControl 에 연결하고, "LLM score + CRC" 와 같은 평가 틀에서
per-stratum miss/over-escalation 을 비교한다. + admission-type / high-risk trivial
baseline.

⚠️ LLM 호출 없음 — MIMIC-IV preprocessed JSONL + scikit-learn(/xgboost) 만 필요.
   patient-level split 사용 (subject_id group split).

산출: results/round9/tabular_baseline.{json,md}

주장(약): LLM 이 tabular 를 *압도하지는 않더라도* CRC framework 가 tabular decision
support 에도 안전하게 적용된다 — 즉 안전성 보장은 score source 에 불변.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_mimic4_by_stratum
from models.stratified_crc import StratifiedConformalRiskControl
from experiments.metrics_utils import (
    patient_level_split, clopper_pearson_upper, safe_rate,
)
from experiments.baselines.tabular import (
    TabularCRCBaseline, AdmissionTypeHeuristic, HighRiskAllEscalate,
)

ALPHAS_R9 = {"CRITICAL": 0.001, "HIGH": 0.01, "MODERATE": 0.05, "LOW": 0.10}
STRATA = ["CRITICAL", "HIGH", "MODERATE", "LOW"]


def _subject_id(case) -> str:
    for tok in (case.meta_info or "").split():
        if tok.startswith("subject_id="):
            return tok.split("=", 1)[1]
    return "h:" + str(id(case))


def _patient_split(bucket: dict, seed: int) -> tuple[list, list]:
    # GLOBAL patient split: 한 subject_id 가 여러 stratum 에 걸칠 수 있으므로
    # 전 case 를 모아 한 번만 환자 단위로 분할 (stratum 은 case 에서 복원).
    all_cases = [c for cases in bucket.values() for c in cases]
    return patient_level_split(all_cases, group_of=_subject_id, cal_frac=0.8, seed=seed)


def _eval_crc_model(model, cal, test, alphas) -> dict:
    """tabular score → CRC → per-stratum miss/over-escalation."""
    model.fit(cal)
    cal_scores = model.scores(cal)
    cal_labels = [c.expected_escalate for c in cal]
    cal_strata = [(c.scenario_type or "").upper() for c in cal]
    crc = StratifiedConformalRiskControl(alphas=alphas)
    crc.fit(cal_scores, cal_labels, cal_strata)

    per = {}
    for s in STRATA:
        idx = [i for i, c in enumerate(test) if (c.scenario_type or "").upper() == s]
        if not idx:
            per[s] = None; continue
        lam = crc.threshold_for(s)
        tp = fn = fp = tn = 0
        for i in idx:
            sc = model.score(test[i]); lbl = test[i].expected_escalate
            escalate = sc > lam
            if lbl and escalate: tp += 1
            elif lbl and not escalate: fn += 1
            elif (not lbl) and escalate: fp += 1
            else: tn += 1
        n_pos = tp + fn
        per[s] = {
            "n": len(idx), "n_pos": n_pos, "misses": fn,
            "miss_rate": safe_rate(fn, n_pos),
            "over_escalation_rate": safe_rate(fp, fp + tn),
            "exact_upper95_miss": round(clopper_pearson_upper(fn, n_pos), 6) if n_pos else None,
            "alpha_target": alphas[s],
        }
    return {"info": model.info(), "per_stratum": per}


def _eval_heuristic(model, test) -> dict:
    per = {}
    for s in STRATA:
        idx = [i for i, c in enumerate(test) if (c.scenario_type or "").upper() == s]
        if not idx:
            per[s] = None; continue
        tp = fn = fp = tn = 0
        for i in idx:
            lbl = test[i].expected_escalate; escalate = model.predict(test[i])
            if lbl and escalate: tp += 1
            elif lbl and not escalate: fn += 1
            elif (not lbl) and escalate: fp += 1
            else: tn += 1
        n_pos = tp + fn
        per[s] = {
            "n": len(idx), "n_pos": n_pos, "misses": fn,
            "miss_rate": safe_rate(fn, n_pos),
            "over_escalation_rate": safe_rate(fp, fp + tn),
        }
    return {"info": model.info(), "per_stratum": per}


def write_md(report: dict, out_md: Path) -> None:
    L = ["# Round 9 R9.6 — Tabular baselines (MIMIC-IV, decision-time features)\n"]
    L.append(f"- Generated: {report['timestamp']}")
    L.append(f"- seed: {report['seed']}, n_per_stratum: {report['n_per_stratum']}")
    L.append(f"- split: **patient-level (subject_id)**, leakage-safe features only\n")
    L.append("각 셀: `miss_rate (exact 95% upper) | over-esc`. α: "
             + ", ".join(f"{k}={v}" for k, v in ALPHAS_R9.items()) + "\n")
    for name, res in report["methods"].items():
        L.append(f"## {name}\n")
        L.append("| stratum | n | n_pos | miss/n_pos | miss_rate | exact 95% upper | over-esc |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for s in STRATA:
            r = res["per_stratum"].get(s)
            if not r:
                L.append(f"| {s} | 0 | — | — | — | — | — |"); continue
            up = r.get("exact_upper95_miss")
            up_s = f"{up:.5f}" if up is not None else "—"
            mr = r["miss_rate"]; mr_s = f"{mr:.5f}" if mr is not None else "N/A"
            oe = r["over_escalation_rate"]; oe_s = f"{oe:.4f}" if oe is not None else "N/A"
            L.append(f"| {s} | {r['n']} | {r['n_pos']} | {r['misses']}/{r['n_pos']} | "
                     f"{mr_s} | {up_s} | {oe_s} |")
        L.append("")
    out_md.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-stratum", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round9" / "tabular_baseline")
    args = ap.parse_args()

    try:
        bucket = load_mimic4_by_stratum(n_per_stratum=args.n_per_stratum, seed=args.seed,
                                        verbose=True)
    except FileNotFoundError as e:
        print(f"[R9.6] preprocessed MIMIC-IV missing: {e}")
        sys.exit(2)

    cal, test = _patient_split(bucket, args.seed)
    print(f"  cal={len(cal)} test={len(test)} (patient-level)")

    report = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed, "n_per_stratum": args.n_per_stratum,
        "alphas": ALPHAS_R9, "methods": {},
    }
    report["methods"]["Tabular-logreg+CRC"] = _eval_crc_model(
        TabularCRCBaseline("logreg"), cal, test, ALPHAS_R9)
    report["methods"]["Tabular-gbdt+CRC"] = _eval_crc_model(
        TabularCRCBaseline("gbdt"), cal, test, ALPHAS_R9)
    report["methods"]["AdmissionType-heuristic"] = _eval_heuristic(
        AdmissionTypeHeuristic(), test)
    report["methods"]["HighRisk-all-escalate"] = _eval_heuristic(
        HighRiskAllEscalate(), test)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.json / .md")


if __name__ == "__main__":
    main()
