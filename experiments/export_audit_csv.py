"""
Export the literature-audit coding table to CSV (both coders + agreement).

Produces results/lit_audit/codings.csv — one row per audited paper with the
reference number, venue/identifier, clinical-CP flag, coder-A and coder-B D0-D3
labels, and a per-item agreement column. This is the machine-readable coding
table referenced by §6 / Appendix D of the paper; the Cohen's kappa over it is
computed by experiments/round26_interrater_kappa.py.
"""
from __future__ import annotations
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
LA = ROOT / "results" / "lit_audit"
DIMS = ["D0", "D1", "D2", "D3"]
ALREADY = {"2208.02814": 3}  # Angelopoulos CRC already ref [3]


def ident(url):
    u = url or ""
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", u)
    if m:
        return f"arXiv:{m.group(1)}"
    if "jmir.org" in u:
        return "JMIR"
    if "jamiaopen" in u:
        return "JAMIA Open"
    m = re.search(r"medrxiv\.org/content/[\d.]*?/?(20\d\d\.\d\d\.\d\d)", u)
    if m:
        return f"medRxiv {m.group(1)}"
    m = re.search(r"(PMC\d+)", u)
    if m:
        return m.group(1)
    return u[:40]


def main():
    A = json.load(open(LA / "codings.json"))
    B = {int(r["id"]): r for r in json.load(open(LA / "codings_coderB.json"))}
    rows = []
    n = 16
    for i, r in enumerate(A):
        url = r.get("url", "")
        ids = ident(url)
        ak = re.search(r"(\d{4}\.\d{4,5})", url or "")
        if ak and ak.group(1) in ALREADY:
            ref = ALREADY[ak.group(1)]
        else:
            ref = n; n += 1
        b = B.get(i, {})
        row = {
            "idx": i, "ref": ref, "paper": r.get("paper", ""), "venue_id": ids,
            "url": url, "modality": r.get("modality", ""),
            "clinical_cp": 1 if r.get("clinical_cp") else 0,
        }
        for d in DIMS:
            row[f"A_{d}"] = r.get(d, "")
            row[f"B_{d}"] = b.get(d, "")
            row[f"agree_{d}"] = int(r.get(d) == b.get(d))
        rows.append(row)

    cols = (["idx", "ref", "paper", "venue_id", "url", "modality", "clinical_cp"]
            + [f"{c}_{d}" for d in DIMS for c in ("A", "B", "agree")])
    out = LA / "codings.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    agree_cells = sum(rows[i][f"agree_{d}"] for i in range(len(rows)) for d in DIMS)
    print(f"✅ {out} — {len(rows)} papers x {len(DIMS)} items; "
          f"cell agreement {agree_cells}/{len(rows)*len(DIMS)} "
          f"({agree_cells/(len(rows)*len(DIMS)):.0%})")
    print(f"   columns: {', '.join(cols)}")


if __name__ == "__main__":
    main()
