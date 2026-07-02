"""
Round 26 (P2) — Inter-rater reliability (Cohen's kappa) for the reporting audit.

The §6 audit was single-coder (one guaranteed reviewer hit in a methods journal).
Here we add an INDEPENDENT second coder (a separate agent, blind to coder A's
labels) who re-codes all 17 papers from the same verbatim evidence passages under
the same codebook. We compute Cohen's kappa per dimension (D0/D1/D2/D3), overall,
and adjudicate disagreements. This measures codebook reliability — whether the
D0-D3 definitions are applied consistently by two independent raters.

Inputs:
  results/lit_audit/codings.json           (coder A — original)
  results/lit_audit/codings_coderB.json    (coder B — independent second pass)
Output:
  results/lit_audit/kappa.json + kappa.md
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
CODINGS = ROOT / "results" / "lit_audit" / "codings.json"
CODERB = ROOT / "results" / "lit_audit" / "codings_coderB.json"
DIMS = ["D0", "D1", "D2", "D3"]


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label lists (categorical)."""
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:  # degenerate: all one category in expectation
        return 1.0 if po == 1.0 else 0.0, po, pe
    return (po - pe) / (1 - pe), po, pe


def main():
    A = json.load(open(CODINGS))
    if not CODERB.exists():
        sys.exit(f"missing {CODERB}; run coder B pass first (see round26 agent step)")
    Braw = json.load(open(CODERB))
    B = {int(r["id"]): r for r in Braw}
    assert len(A) == len(B), f"coder mismatch {len(A)} vs {len(B)}"

    per_dim = {}
    disagreements = []
    for d in DIMS:
        la = [A[i].get(d) for i in range(len(A))]
        lb = [B[i].get(d) for i in range(len(A))]
        k, po, pe = cohen_kappa(la, lb)
        per_dim[d] = {"kappa": round(k, 3), "p_observed": round(po, 3),
                      "p_expected": round(pe, 3),
                      "agree_n": sum(1 for x, y in zip(la, lb) if x == y),
                      "n": len(la)}
        for i in range(len(A)):
            if la[i] != lb[i]:
                disagreements.append({"id": i, "dim": d, "paper": A[i].get("paper", "")[:60],
                                      "coderA": la[i], "coderB": lb[i]})

    # pooled kappa over all 4 dims x 17 papers (one combined categorical task)
    poolA = [A[i].get(d) for d in DIMS for i in range(len(A))]
    poolB = [B[i].get(d) for d in DIMS for i in range(len(A))]
    kp, pop, pep = cohen_kappa(poolA, poolB)
    pooled = {"kappa": round(kp, 3), "p_observed": round(pop, 3),
              "p_expected": round(pep, 3), "n": len(poolA)}

    def interp(k):
        return ("almost perfect" if k >= 0.81 else "substantial" if k >= 0.61
                else "moderate" if k >= 0.41 else "fair" if k >= 0.21
                else "slight" if k > 0 else "poor")

    report = {"n_papers": len(A), "dims": DIMS, "per_dim": per_dim,
              "pooled": pooled, "pooled_interpretation": interp(kp),
              "n_disagreements": len(disagreements), "disagreements": disagreements}
    outp = ROOT / "results" / "lit_audit" / "kappa"
    Path(str(outp) + ".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    L = ["# Round 26 — Inter-rater reliability of the reporting audit (Cohen's kappa)\n"]
    L.append(f"- Two independent coders, {len(A)} papers, dimensions {', '.join(DIMS)}.")
    L.append("- Coder B coded blind to coder A's labels, from the same verbatim evidence "
             "passages under the same codebook.\n")
    L.append("| Dimension | Cohen's kappa | % agreement | interpretation |")
    L.append("|---|---|---|---|")
    for d in DIMS:
        pd = per_dim[d]
        L.append(f"| {d} | {pd['kappa']} | {pd['agree_n']}/{pd['n']} "
                 f"({pd['agree_n']/pd['n']:.0%}) | {interp(pd['kappa'])} |")
    L.append(f"| **Pooled (all dims)** | **{pooled['kappa']}** | "
             f"{int(pop*pooled['n'])}/{pooled['n']} ({pop:.0%}) | {interp(kp)} |")
    L.append(f"\n**Pooled Cohen's kappa = {pooled['kappa']} ({interp(kp)} agreement).** "
             f"{len(disagreements)} cell-level disagreements out of {len(DIMS)*len(A)} "
             f"({len(disagreements)/(len(DIMS)*len(A)):.0%}).\n")
    if disagreements:
        L.append("## Disagreements (for adjudication)\n")
        L.append("| id | dim | paper | coder A | coder B |")
        L.append("|---|---|---|---|---|")
        for x in disagreements:
            L.append(f"| {x['id']} | {x['dim']} | {x['paper']} | {x['coderA']} | {x['coderB']} |")
    Path(str(outp) + ".md").write_text("\n".join(L))
    print(f"✅ {outp}.{{json,md}}")
    for d in DIMS:
        print(f"  {d}: kappa={per_dim[d]['kappa']} ({per_dim[d]['agree_n']}/{per_dim[d]['n']})")
    print(f"  POOLED kappa={pooled['kappa']} ({interp(kp)}); disagreements={len(disagreements)}")


if __name__ == "__main__":
    main()
