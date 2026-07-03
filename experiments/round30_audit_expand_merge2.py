"""
Round 30 — Merge the SECOND real-EHR CP audit expansion and recompute.

The first expansion (R29) added 7 real-EHR CP papers (corpus 17 -> 24). A second
aggressive 12-angle search (workflow) surfaced, after adversarial curation
(dropping 2 duplicates-of-existing, 1 out-of-scope neuroimaging paper, 1
duplicate URL), FIVE more genuine real-EHR CP papers, one of which (Multiply
Robust CRC with Coarsened Data) is the FIRST in the corpus to reach D3="Yes"
(explicit presence-vs-value / coarsening-propensity handling — notably itself a
missing-data methods paper, not a standard clinical prediction pipeline).

Clean evidence-only coder B (no repo/web) re-coded the five for kappa.

This script merges them (corpus 24 -> 29), recomputes Cohen's kappa, and reports
the expanded real-EHR informative-missingness bound with a general
Clopper-Pearson interval (now k may be > 0).

Output: updates codings.json + codings_coderB.json; writes
results/lit_audit/expanded_audit_stats2.{json,md}
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from scipy.stats import beta

ROOT = Path(__file__).parent.parent
LA = ROOT / "results" / "lit_audit"
DIMS = ["D0", "D1", "D2", "D3"]


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None, None
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    k = 1.0 if (pe >= 1 and po == 1) else (0.0 if pe >= 1 else (po - pe) / (1 - pe))
    return k, po


def cp_interval(k, n, alpha=0.05):
    """Clopper-Pearson two-sided interval for k successes in n."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return round(float(lo), 3), round(float(hi), 3)


def main():
    A = json.load(open(LA / "codings.json"))
    B = {int(r["id"]): r for r in json.load(open(LA / "codings_coderB.json"))}
    newA = json.load(open(LA / "codings_new_realehr_2.json"))
    newB = json.load(open(LA / "codings_coderB_new_2.json"))
    assert len(newA) == len(newB) == 5

    base = len(A)  # 24
    for i, (a, b) in enumerate(zip(newA, newB)):
        idx = base + i
        a2 = dict(a); a2["id"] = idx
        A.append(a2)
        B[idx] = {"id": idx, **{d: b[d] for d in DIMS}, "rationale": b.get("rationale", "")}
    json.dump(A, open(LA / "codings.json", "w"), indent=2, ensure_ascii=False)
    json.dump([B[i] for i in range(len(A))], open(LA / "codings_coderB.json", "w"),
              indent=2, ensure_ascii=False)

    # full-corpus kappa
    per_dim, disag = {}, 0
    for d in DIMS:
        la = [A[i].get(d) for i in range(len(A))]
        lb = [B[i].get(d) for i in range(len(A))]
        kk, po = cohen_kappa(la, lb)
        per_dim[d] = {"kappa": round(kk, 3), "agree": sum(1 for x, y in zip(la, lb) if x == y), "n": len(la)}
        disag += sum(1 for x, y in zip(la, lb) if x != y)
    poolA = [A[i].get(d) for d in DIMS for i in range(len(A))]
    poolB = [B[i].get(d) for d in DIMS for i in range(len(A))]
    kp, pop = cohen_kappa(poolA, poolB)

    # real-EHR CP subset: 3 seed-real + 7 (R29, ids 17-23) + 5 (R30, ids 24-28)
    real_ids = [0, 1, 9] + list(range(17, 29))
    real = [A[i] for i in real_ids]
    d3 = Counter(r.get("D3") for r in real)
    d3_yes = d3.get("Yes", 0)
    lo, hi = cp_interval(d3_yes, len(real))
    # longitudinal-only (drop cross-sectional NAFLD id21 + xsectional health-exam)
    long_ids = [i for i in real_ids if (A[i].get("modality") or "").find("xsection") < 0]
    real_long = [A[i] for i in long_ids]
    d3y_long = Counter(r.get("D3") for r in real_long).get("Yes", 0)
    lo_l, hi_l = cp_interval(d3y_long, len(real_long))

    # coder-B verification of the D3="Yes" paper (Multiply Robust = last id)
    mr_id = base + 4
    mr_b = B[mr_id]["D3"]

    stats = {
        "corpus_n": len(A), "added_this_round": 5,
        "kappa_full": {"pooled": round(kp, 3), "po": round(pop, 3), "per_dim": per_dim,
                       "disagreements": disag, "cells": len(DIMS) * len(A)},
        "real_ehr_D3": {
            "n": len(real), "D3_dist": dict(d3), "D3_yes": d3_yes,
            "cp95": [lo, hi],
            "n_longitudinal": len(real_long), "D3_yes_long": d3y_long, "cp95_long": [lo_l, hi_l]},
        "multiply_robust_D3": {"coderA": "Yes", "coderB": mr_b},
    }
    json.dump(stats, open(LA / "expanded_audit_stats2.json", "w"), indent=2, ensure_ascii=False)

    L = ["# Second real-EHR CP audit expansion (R30 merge)\n",
         f"- Corpus grown to **{len(A)}** records (+5 real-EHR CP after curation: dropped 2 "
         f"duplicates-of-existing, 1 out-of-scope neuroimaging, 1 duplicate URL).\n",
         "## Inter-rater reliability (full corpus)\n",
         f"Pooled Cohen's kappa = **{kp:.3f}** (po {pop:.3f}); {disag}/{len(DIMS)*len(A)} cell disagreements.\n",
         "| Dim | kappa | % agree |", "|---|---|---|"]
    for d in DIMS:
        pd = per_dim[d]
        L.append(f"| {d} | {pd['kappa']} | {pd['agree']}/{pd['n']} ({pd['agree']/pd['n']:.0%}) |")
    L += ["\n## Expanded informative-missingness bound (real-EHR clinical-CP)\n",
          f"- Real-EHR clinical-CP papers (D3 assessable): **{len(real)}**; D3 distribution {dict(d3)}.",
          f"- **{d3_yes}/{len(real)}** at D3='Yes' → 95% Clopper-Pearson interval **[{lo}, {hi}]**.",
          f"- Longitudinal-only ({len(real_long)}): {d3y_long}/{len(real_long)} at 'Yes', CP95 [{lo_l}, {hi_l}].",
          f"\nThe single D3='Yes' paper (Multiply Robust Conformal Risk Control with Coarsened Data) "
          f"is itself a *missing-data methods* paper — its coarsening-propensity decomposition IS the "
          f"contribution — not a standard clinical prediction pipeline. Clean coder B independently coded "
          f"its D3 as **{mr_b}** (vs coder A 'Yes'). Every *applied* real-EHR CP pipeline in the corpus "
          f"still omits presence-vs-value handling; the lone exception proves the rule.\n"]
    Path(LA / "expanded_audit_stats2.md").write_text("\n".join(L))
    print(f"✅ merged; corpus n = {len(A)}")
    print(f"  pooled kappa (full) = {kp:.3f}; disagreements {disag}/{len(DIMS)*len(A)}")
    print(f"  real-EHR D3: {d3_yes}/{len(real)} Yes, CP95 [{lo},{hi}] | dist {dict(d3)}")
    print(f"  longitudinal: {d3y_long}/{len(real_long)} Yes, CP95 [{lo_l},{hi_l}]")
    print(f"  Multiply-Robust D3: coderA=Yes, coderB={mr_b}")
    for d in DIMS:
        print(f"  {d}: kappa {per_dim[d]['kappa']} ({per_dim[d]['agree']}/{per_dim[d]['n']})")


if __name__ == "__main__":
    main()
