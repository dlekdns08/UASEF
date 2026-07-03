"""
Round 29 — Merge the expanded real-EHR CP audit corpus and recompute reliability.

Part B of the audit hardening. A background workflow searched for real
longitudinal-EHR conformal-prediction papers; after adversarial curation
(dropping 1 duplicate-of-existing, 2 non-CP supervised-ML papers that were
self-flagged out of scope, and 1 intra-batch duplicate) 7 genuine new real-EHR
CP papers remain (coder A: results/lit_audit/codings_new_realehr.json). A CLEAN
independent coder B (evidence-only, no repo/web access — the workflow's own
coder B was discarded for contaminating several codings with details from THIS
repository) re-coded them: results/lit_audit/codings_coderB_new.json.

This script merges new codings into the master files, recomputes Cohen's kappa
over the full corpus, and reports the expanded real-EHR informative-missingness
bound with Clopper-Pearson intervals.

Output: updates codings.json + codings_coderB.json; writes
results/lit_audit/expanded_audit_stats.{json,md}
"""
from __future__ import annotations
import json, math, sys
from collections import Counter
from pathlib import Path

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


def cp_upper(k, n, alpha=0.05):
    """Clopper-Pearson upper bound for k successes in n (here k=0)."""
    if n == 0:
        return 1.0
    if k == 0:
        return 1 - (alpha / 2) ** (1 / n)
    # general (not needed here) — use beta quantile
    from math import inf
    return None


def main():
    A = json.load(open(LA / "codings.json"))
    B = {int(r["id"]): r for r in json.load(open(LA / "codings_coderB.json"))}
    newA = json.load(open(LA / "codings_new_realehr.json"))
    newB_path = LA / "codings_coderB_new.json"
    if not newB_path.exists():
        sys.exit(f"missing {newB_path}; save clean coder-B output first")
    newB = json.load(open(newB_path))
    assert len(newA) == len(newB) == 7, f"expected 7/7, got {len(newA)}/{len(newB)}"

    base = len(A)  # 17
    # append new coder-A (assign ids continuing) and new coder-B (aligned ids)
    for i, (a, b) in enumerate(zip(newA, newB)):
        idx = base + i
        a2 = dict(a); a2["id"] = idx
        A.append(a2)
        B[idx] = {"id": idx, **{d: b[d] for d in DIMS}, "rationale": b.get("rationale", "")}

    # persist merged master files
    json.dump(A, open(LA / "codings.json", "w"), indent=2, ensure_ascii=False)
    json.dump([B[i] for i in range(len(A))], open(LA / "codings_coderB.json", "w"),
              indent=2, ensure_ascii=False)

    # kappa over full corpus
    per_dim, disag = {}, 0
    for d in DIMS:
        la = [A[i].get(d) for i in range(len(A))]
        lb = [B[i].get(d) for i in range(len(A))]
        k, po = cohen_kappa(la, lb)
        per_dim[d] = {"kappa": round(k, 3), "po": round(po, 3),
                      "agree": sum(1 for x, y in zip(la, lb) if x == y), "n": len(la)}
        disag += sum(1 for x, y in zip(la, lb) if x != y)
    poolA = [A[i].get(d) for d in DIMS for i in range(len(A))]
    poolB = [B[i].get(d) for d in DIMS for i in range(len(A))]
    kp, pop = cohen_kappa(poolA, poolB)

    # kappa on just the 7 NEW papers (clean coder B)
    nptA = [newA[i][d] for d in DIMS for i in range(7)]
    nptB = [newB[i][d] for d in DIMS for i in range(7)]
    kNew, poNew = cohen_kappa(nptA, nptB)

    # expanded real-EHR CP informative-missingness bound
    # real longitudinal/tabular-EHR clinical-CP with D3 assessable (exclude synthetic & non-EHR)
    def is_real_ehr(rec):
        m = (rec.get("modality") or "").lower()
        return m.startswith("ehr")  # only the 7 new carry modality; existing handled below
    new_real = [r for r in newA]  # all 7 curated are real-EHR CP by construction
    new_longitudinal = [r for r in newA if r["modality"] != "ehr_tabular_xsectional"]
    # existing real-EHR (from prior synthesis): CPMORS#0, Time-Series sepsis#1, THCM#9
    existing_real_ids = [0, 1, 9]
    existing_real = [A[i] for i in existing_real_ids]

    def d3_yes(recs):
        return sum(1 for r in recs if r.get("D3") == "Yes")

    real_all = existing_real + new_real              # 3 + 7 = 10
    real_long = existing_real + new_longitudinal      # 3 + 6 = 9
    stats = {
        "corpus_n": len(A), "new_papers": 7, "dropped": {
            "duplicate_of_existing": 1, "non_conformal_supervised": 2, "intra_batch_duplicate": 1},
        "kappa_full_corpus": {"pooled": round(kp, 3), "po": round(pop, 3),
                              "per_dim": per_dim, "disagreements": disag, "cells": len(DIMS) * len(A)},
        "kappa_new7_clean_coderB": {"pooled": round(kNew, 3), "po": round(poNew, 3)},
        "real_ehr_D3": {
            "real_ehr_all_n": len(real_all), "D3_yes": d3_yes(real_all),
            "cp95_upper": round(cp_upper(d3_yes(real_all), len(real_all)), 3),
            "real_longitudinal_n": len(real_long), "D3_yes_long": d3_yes(real_long),
            "cp95_upper_long": round(cp_upper(d3_yes(real_long), len(real_long)), 3),
        },
        "new7_dims": {d: dict(Counter(r[d] for r in newA)) for d in DIMS},
    }
    json.dump(stats, open(LA / "expanded_audit_stats.json", "w"), indent=2, ensure_ascii=False)

    L = ["# Expanded real-EHR conformal-prediction reporting audit (R29 merge)\n"]
    L.append(f"- Corpus grown to **{len(A)}** records (+7 real-EHR CP after curation: dropped "
             f"1 duplicate-of-existing, 2 non-conformal supervised-ML, 1 intra-batch duplicate).\n")
    L.append("## Inter-rater reliability (full corpus, dual-coded)\n")
    L.append(f"Pooled Cohen's kappa = **{kp:.3f}** (po {pop:.3f}); {disag}/{len(DIMS)*len(A)} cell disagreements.\n")
    L.append("| Dim | kappa | % agree |")
    L.append("|---|---|---|")
    for d in DIMS:
        pd = per_dim[d]
        L.append(f"| {d} | {pd['kappa']} | {pd['agree']}/{pd['n']} ({pd['agree']/pd['n']:.0%}) |")
    L.append(f"\nClean coder-B on the 7 NEW papers only: kappa = {kNew:.3f} (po {poNew:.3f}).\n")
    L.append("## Expanded informative-missingness bound (real-EHR clinical-CP)\n")
    ra = stats["real_ehr_D3"]
    L.append(f"- Real-EHR clinical-CP papers (D3 assessable): **{ra['real_ehr_all_n']}**; "
             f"D3='Yes' (full presence-vs-value handling): **{ra['D3_yes']}**.")
    L.append(f"- **0/{ra['real_ehr_all_n']}** → 95% Clopper-Pearson upper bound **{ra['cp95_upper']}** "
             f"(vs 0.52 at the pre-expansion n=5).")
    L.append(f"- Restricting to real *longitudinal* EHR (drop cross-sectional): 0/{ra['real_longitudinal_n']}, "
             f"upper {ra['cp95_upper_long']}.\n")
    L.append("## New-paper dimension distribution (coder A)\n")
    for d in DIMS:
        L.append(f"- {d}: {stats['new7_dims'][d]}")
    L.append("\nAll 7 new real-EHR CP papers are D3=No — none implement presence-vs-value "
             "decomposition or a missingness ablation. The expansion sharpens, and does not "
             "soften, the informative-missingness gap. We keep the claim below 'field-wide': "
             "the CP upper bound (~0.3) still exceeds a powered 0.17 threshold.")
    Path(LA / "expanded_audit_stats.md").write_text("\n".join(L))
    print("✅ merged; corpus n =", len(A))
    print(f"  pooled kappa (full) = {kp:.3f}; new-7 clean coder-B kappa = {kNew:.3f}")
    print(f"  real-EHR D3=Yes: 0/{ra['real_ehr_all_n']} (CP upper {ra['cp95_upper']}); "
          f"longitudinal 0/{ra['real_longitudinal_n']} (upper {ra['cp95_upper_long']})")
    for d in DIMS:
        print(f"  {d}: kappa {per_dim[d]['kappa']} ({per_dim[d]['agree']}/{per_dim[d]['n']})")


if __name__ == "__main__":
    main()
