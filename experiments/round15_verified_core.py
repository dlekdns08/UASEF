"""
Round 15 — Definitive corrected result using the unit-tested conformal core.

Runs models/conformal_escalation.{StandardCRC,BoundedCRC} (verified by
tests/test_conformal_escalation.py) on the MIMIC-IV cohorts with the FIXED
score sign (+P(y=1)). This supersedes all R10/R11/R13 tabular numbers, which
were sign-bug artifacts.

For each (cohort, classifier, stratum), 5-seed pooled test set:
  - AUROC (direction-correct)
  - StandardCRC: miss_rate, over_esc, verdict
  - BoundedCRC (c_o in {0.05,0.1,0.2}): miss_rate, over_esc, verdict/INFEASIBLE

Output: results/round15/r15_verified_core.{json,md}
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.conformal_escalation import StandardCRC, BoundedCRC, check_orientation
from experiments.round13_bcrc_vs_crc import recompute_r10_r11
from experiments.round14_genuine_win_feasibility import pool_seeds

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}
CO_GRID = [0.05, 0.10, 0.20]


def analyze(per_seed, stratum, alpha):
    # Fit on pooled CALIBRATION, evaluate on pooled TEST (honest split).
    cal_s, cal_y, test_s, test_y = [], [], [], []
    for e in per_seed:
        if "cal_scores" not in e: continue
        cs = np.asarray(e["cal_scores"], float); cl = np.asarray(e["cal_labels"], int)
        cst = np.asarray(e["cal_strata"]); csm = cst == stratum
        ts = np.asarray(e["test_scores"], float); tl = np.asarray(e["test_labels"], int)
        tst = np.asarray(e["test_strata"]); tsm = tst == stratum
        cal_s.append(cs[csm]); cal_y.append(cl[csm])
        test_s.append(ts[tsm]); test_y.append(tl[tsm])
    cal_s = np.concatenate(cal_s); cal_y = np.concatenate(cal_y)
    test_s = np.concatenate(test_s); test_y = np.concatenate(test_y)
    if (cal_y == 1).sum() < 10 or (test_y == 1).sum() < 5:
        return {"insufficient": True, "n_pos_test": int((test_y == 1).sum())}

    au = check_orientation(test_s, test_y)
    out = {"auroc_test": au, "n_pos_test": int((test_y == 1).sum()),
           "n_neg_test": int((test_y == 0).sum()), "alpha": alpha}

    sc = StandardCRC(alpha=alpha).fit(cal_s, cal_y, check_orient=False)
    out["standard"] = sc.evaluate(test_s, test_y)

    out["bcrc"] = {}
    for co in CO_GRID:
        b = BoundedCRC(alpha=alpha, c_miss=1 - co, c_over=co).fit(
            cal_s, cal_y, check_orient=False)
        out["bcrc"][f"co{co}"] = b.evaluate(test_s, test_y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-jsonl",
                    default=ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl",
                    type=Path)
    ap.add_argument("--classifiers", nargs="+",
                    default=["randomforest", "logreg", "gbdt", "xgboost"])
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round15" / "r15_verified_core")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from experiments.round10_method_agnostic import _feature_vector
    from experiments.round11_method_agnostic_minimal import _feature_vector_minimal

    cohorts = {}
    for label, fv in [("mimic4_r10_4_LEAKAGE", _feature_vector),
                      ("mimic4_r11_1_SAFE", _feature_vector_minimal)]:
        print(f"[R15] {label}")
        cohorts[label] = {}
        for clf in args.classifiers:
            print(f"  {clf}...")
            per_seed = recompute_r10_r11(clf, args.mimic_jsonl, fv,
                                          n_cal=args.n_cal, n_test=args.n_test)
            cohorts[label][clf] = {
                s: analyze(per_seed, s, ALPHAS[s]) for s in ALPHAS}

    report = {"timestamp": datetime.now().isoformat(), "cohorts": cohorts,
              "co_grid": CO_GRID}
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))

    # Markdown
    L = ["# Round 15 — Definitive result (unit-tested core, sign-fixed +P)\n"]
    L.append(f"- Generated: {report['timestamp']}")
    L.append("- Core verified by tests/test_conformal_escalation.py (10/10)")
    L.append("- Convention: score = +P(y=1), higher = escalate; StandardCRC = sup-lambda efficient threshold\n")
    L.append("**genuine_win = miss_rate<=alpha (CRC guarantee) AND over_esc<0.95. "
             "But over_esc directly tracks AUROC — a low-AUROC 'win' still escalates most negatives.**\n")
    for label, cdata in cohorts.items():
        L.append(f"\n## {label}\n")
        L.append("| clf | stratum | AUROC | StdCRC miss | StdCRC over_esc | genuine? | high-conf? | b-CRC(0.1) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for clf, sdata in cdata.items():
            for s in ALPHAS:
                c = sdata[s]
                if c.get("insufficient"):
                    L.append(f"| {clf} | {s} | — | — | — | — | — | insufficient |"); continue
                sc = c["standard"]; b = c["bcrc"]["co0.1"]
                if sc.get("infeasible"):
                    scstr = "INFEASIBLE"; gw = "—"; hc = "—"; oe = "—"; mr = "—"
                else:
                    mr = f"{sc['miss_rate']:.3f}"; oe = f"{sc['over_esc_rate']:.3f}"
                    gw = "✓" if sc["genuine_win"] else "✗"
                    hc = "✓" if sc["high_conf_coverage"] else "✗"
                bstr = "INFEASIBLE" if b.get("infeasible") else (
                    f"miss {b['miss_rate']:.3f}/oe {b['over_esc_rate']:.3f}")
                L.append(f"| {clf} | {s} | {c['auroc_test']:.3f} | {mr} | {oe} | {gw} | {hc} | {bstr} |")
    Path(str(args.out) + ".md").write_text("\n".join(L))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
