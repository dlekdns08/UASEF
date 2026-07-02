"""
Round 16 — Can a GENUINE (non-leakage) win be found with richer decision-time
features? Uses the unit-tested conformal core.

The r11_1 "minimal" vector was impoverished: [age, adm_emerg, spec_idx, n_labs]
— it used only the COUNT of early labs, discarding WHICH labs were abnormal.
The JSONL already carries clinically-meaningful, decision-time-safe early lab
flags (lactate_high, troponin_high, creatinine_high, leukocytosis, ...). This
round builds a richer, still leakage-safe vector and asks whether AUROC (hence
the achievable escalation gate) improves.

Feature sets compared (all leakage-safe — NO charlson, NO specialty_baseline_rate,
NO ICU-conditional vital flags):
  MINIMAL   : [age, adm_emerg, spec_idx, n_labs]                     (= r11_1)
  RICH_FLAGS: MINIMAL + one-hot of the 10 early lab flags + sex + race + vitals
  (optionally FULL_LABVALUES from raw MIMIC — separate script if needed)

Output: results/round16/r16_richer_features.{json,md}
"""
from __future__ import annotations

import argparse, json, sys, random
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import _load_mimic4_jsonl
from experiments.metrics_utils import patient_level_split
from experiments.round10_method_agnostic import _make_classifier, _subject_id
from models.conformal_escalation import StandardCRC, BoundedCRC, check_orientation

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}

AGE_BUCKET = {"unknown": 0, "<18": 1, "18-34": 2, "35-49": 3,
              "50-64": 4, "65-79": 5, "80+": 6}
SPEC_MAP = {"cardiology": 1, "neurology": 2, "internal_medicine": 3,
            "surgery": 4, "obstetrics": 5, "psychiatry": 6,
            "pediatrics": 7, "cardiothoracic_surgery": 8}
# The 10 legitimate decision-time early lab flags present in the JSONL.
LAB_FLAGS = ["lactate_high", "troponin_high", "creatinine_high", "leukocytosis",
             "hyperkalemia", "hyponatremia", "low_bicarb", "acidemia",
             "thrombocytopenia", "ldh_high"]
RACE_MAP = {"WHITE": 1, "BLACK": 2, "HISPANIC": 3, "ASIAN": 4}


def _raw_of(case):
    """Recover the raw JSONL dict attached by the loader (meta_info holds ids)."""
    return getattr(case, "_raw_row", None)


def fv_minimal(row) -> list[float]:
    age = AGE_BUCKET.get((row.get("demographics") or {}).get("age_bucket", "unknown"), 0)
    adm = row.get("admission_type", "") or ""
    adm_emerg = 1.0 if ("EMER" in adm or "URG" in adm) else 0.0
    spec = SPEC_MAP.get(row.get("specialty") or "", 0)
    elf = (row.get("structured") or {}).get("early_lab_flags", []) or []
    n_labs = len(elf)
    return [float(age), adm_emerg, float(spec), float(n_labs)]


def fv_rich(row) -> list[float]:
    base = fv_minimal(row)
    elf = set((row.get("structured") or {}).get("early_lab_flags", []) or [])
    onehot = [1.0 if f in elf else 0.0 for f in LAB_FLAGS]
    demo = row.get("demographics") or {}
    sex = 1.0 if (demo.get("sex", "") or "").startswith("F") else 0.0
    race = float(RACE_MAP.get(demo.get("race", ""), 0))
    evq = (row.get("structured") or {}).get("early_vital_quartiles", []) or []
    n_vital = float(len(evq))
    return base + onehot + [sex, race, n_vital]


def run(cases_rows, fv_fn, n_cal, n_test, seeds):
    """cases_rows: list of (case_obj, raw_dict). Returns per-stratum pooled."""
    # pool cal/test scores across seeds per stratum, per classifier
    results = {}
    for clf_name in ["randomforest", "logreg", "gbdt", "xgboost"]:
        per_seed = []
        for seed in seeds:
            cal, test = patient_level_split(cases_rows,
                                             group_of=lambda cr: str(cr[1].get("subject_id")),
                                             cal_frac=0.8, seed=seed)
            rng = random.Random(seed); rng.shuffle(cal); rng.shuffle(test)
            cal = cal[:n_cal]; test = test[:n_test]
            clf = _make_classifier(clf_name)
            Xc = [fv_fn(cr[1]) for cr in cal]; yc = [bool(cr[0].expected_escalate) for cr in cal]
            if not clf.fit(Xc, yc):
                continue
            cs = [clf.score(x) for x in Xc]
            Xt = [fv_fn(cr[1]) for cr in test]
            ts = [clf.score(x) for x in Xt]
            per_seed.append({
                "cal_scores": cs, "cal_labels": [int(bool(cr[0].expected_escalate)) for cr in cal],
                "cal_strata": [(cr[0].scenario_type or "").upper() for cr in cal],
                "test_scores": ts, "test_labels": [int(bool(cr[0].expected_escalate)) for cr in test],
                "test_strata": [(cr[0].scenario_type or "").upper() for cr in test],
            })
        results[clf_name] = {}
        for s, alpha in ALPHAS.items():
            cal_s, cal_y, test_s, test_y = [], [], [], []
            for e in per_seed:
                cs = np.array(e["cal_scores"]); cl = np.array(e["cal_labels"]); cst = np.array(e["cal_strata"])
                ts = np.array(e["test_scores"]); tl = np.array(e["test_labels"]); tst = np.array(e["test_strata"])
                cm = cst == s; tm = tst == s
                cal_s.append(cs[cm]); cal_y.append(cl[cm]); test_s.append(ts[tm]); test_y.append(tl[tm])
            cal_s = np.concatenate(cal_s); cal_y = np.concatenate(cal_y)
            test_s = np.concatenate(test_s); test_y = np.concatenate(test_y)
            if (cal_y == 1).sum() < 10 or (test_y == 1).sum() < 5:
                results[clf_name][s] = {"insufficient": True}; continue
            au = check_orientation(test_s, test_y)
            sc = StandardCRC(alpha=alpha).fit(cal_s, cal_y, check_orient=False).evaluate(test_s, test_y)
            b = BoundedCRC(alpha=alpha, c_miss=0.9, c_over=0.1).fit(cal_s, cal_y, check_orient=False).evaluate(test_s, test_y)
            results[clf_name][s] = {"auroc": au, "standard": sc, "bcrc_0.1": b}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl")
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "round16" / "r16_richer_features")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load raw rows alongside case objects
    print(f"[R16] loading {args.jsonl}")
    raw_rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line: raw_rows.append(json.loads(line))
    cases = _load_mimic4_jsonl(args.jsonl, n=10**9, seed=42)
    # align by hadm_id
    by_hadm = {str(r.get("hadm_id")): r for r in raw_rows}
    cases_rows = []
    for c in cases:
        hid = None
        for tok in (c.meta_info or "").split():
            if tok.startswith("hadm_id="): hid = tok.split("=", 1)[1]
        raw = by_hadm.get(str(hid))
        if raw is None: continue
        cases_rows.append((c, raw))
    print(f"[R16] aligned {len(cases_rows)} case/raw pairs")

    out = {"timestamp": datetime.now().isoformat(), "feature_sets": {}}
    for name, fv in [("MINIMAL", fv_minimal), ("RICH_FLAGS", fv_rich)]:
        print(f"[R16] feature set: {name}")
        out["feature_sets"][name] = run(cases_rows, fv, args.n_cal, args.n_test, args.seeds)

    Path(str(args.out) + ".json").write_text(json.dumps(out, indent=2, default=str))

    L = ["# Round 16 — Richer leakage-safe features vs minimal (verified core)\n"]
    L.append(f"- Generated: {out['timestamp']}")
    L.append("- Question: does using the WHICH-labs one-hot (not just count) + demographics")
    L.append("  raise AUROC enough for a genuine escalation gate — WITHOUT leakage?\n")
    for name in ["MINIMAL", "RICH_FLAGS"]:
        L.append(f"\n## {name}\n")
        L.append("| clf | stratum | AUROC | StdCRC miss | StdCRC over_esc | genuine? | high-conf? |")
        L.append("|---|---|---|---|---|---|---|")
        for clf, sd in out["feature_sets"][name].items():
            for s in ALPHAS:
                c = sd.get(s, {})
                if c.get("insufficient"):
                    L.append(f"| {clf} | {s} | — | — | — | — | — |"); continue
                sc = c["standard"]
                if sc.get("infeasible"):
                    L.append(f"| {clf} | {s} | {c['auroc']:.3f} | INFEAS | — | — | — |"); continue
                L.append(f"| {clf} | {s} | {c['auroc']:.3f} | {sc['miss_rate']:.3f} | "
                         f"{sc['over_esc_rate']:.3f} | {'✓' if sc['genuine_win'] else '✗'} | "
                         f"{'✓' if sc['high_conf_coverage'] else '✗'} |")
    Path(str(args.out) + ".md").write_text("\n".join(L))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
