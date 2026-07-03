"""
Build the audit-corpus references + Appendix C table from the final codings.

The reporting audit (§6) codes a corpus of clinical conformal-prediction papers.
Reviewers ask that every audited paper appear in the bibliography. This script
reads results/lit_audit/codings.json and emits, for pasting into the paper:
  (1) Appendix C — a table of every audited paper with its citation number,
      venue/identifier, modality, and D0-D3 codes.
  (2) numbered reference entries (starting at START_REF) for papers not already
      in the base bibliography.

We identify each paper by title + a permanent identifier (arXiv ID / PMC / DOI /
journal URL) — appropriate for an audit whose "data" are the papers themselves.

Output: results/lit_audit/audit_references.md
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
LA = ROOT / "results" / "lit_audit"
START_REF = 16  # base bibliography currently ends at 15

# audited papers already present in the base bibliography -> reuse their number
ALREADY = {
    "2208.02814": 3,  # Angelopoulos et al., Conformal Risk Control (ICLR 2024) = ref [3]
}


def ident(url: str):
    """Return (identifier_str, year_str, venue_str) from a paper URL."""
    u = url or ""
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4})\.(\d{4,5})", u)
    if m:
        yy = int(m.group(1)[:2])
        year = 2000 + yy
        return f"arXiv:{m.group(1)}.{m.group(2)}", str(year), "arXiv preprint"
    m = re.search(r"jmir\.org/(\d{4})/", u)
    if m:
        return f"JMIR ({m.group(1)})", m.group(1), "J. Med. Internet Res."
    m = re.search(r"medrxiv\.org/content/[\d.]*?/?(20\d\d)\.(\d\d)\.(\d\d)", u)
    if m:
        return f"medRxiv {m.group(1)}.{m.group(2)}.{m.group(3)}", m.group(1), "medRxiv preprint"
    if "jamiaopen" in u:
        my = re.search(r"/(\d{4})\b", u)
        return "JAMIA Open", "2026", "JAMIA Open"
    m = re.search(r"(PMC\d+)", u)
    if m:
        return m.group(1), "", "PubMed Central"
    return (u[:48] or "n/a"), "", ""


def arxiv_key(url):
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", url or "")
    return m.group(1) if m else None


def main():
    recs = json.load(open(LA / "codings.json"))
    lines_tbl = ["| # | Ref | Paper (short) | Venue / ID | CP? | D0 | D1 | D2 | D3 |",
                 "|---|---|---|---|---|---|---|---|---|"]
    refs = []
    n = START_REF
    ref_map = []  # (record_index, ref_number)
    for i, r in enumerate(recs):
        url = r.get("url", "") or ""
        ids, year, venue = ident(url)
        ak = arxiv_key(url)
        if ak and ak in ALREADY:
            refn = ALREADY[ak]
        else:
            refn = n
            n += 1
            title = (r.get("paper") or "").strip().rstrip(".")
            # strip a trailing venue/id parenthetical to avoid doubling with `ids`
            title = re.sub(r"\s*\((?:arXiv|JMIR|medRxiv|JAMIA|PMC|doi)[^)]*\)\s*$", "", title, flags=re.I).strip().rstrip(".")
            refs.append(f"{refn}. {title}. {ids}. {url}")
        ref_map.append((i, refn))
        cp = "Y" if r.get("clinical_cp") else "—"
        short = (r.get("paper") or "")[:46].replace("|", "/")
        lines_tbl.append(f"| {i} | [{refn}] | {short} | {ids} | {cp} | "
                         f"{r.get('D0','')} | {r.get('D1','')} | {r.get('D2','')} | {r.get('D3','')} |")

    out = ["# Appendix C — Audit corpus (for paper paste)\n",
           f"Corpus n = {len(recs)}; clinical-CP = {sum(1 for r in recs if r.get('clinical_cp'))}. "
           f"Each paper is cited below; new reference numbers start at [{START_REF}].\n",
           "## Appendix C table\n", *lines_tbl,
           "\n## Reference entries to append (after [15])\n", *refs]
    Path(LA / "audit_references.md").write_text("\n".join(out))
    print(f"✅ {LA/'audit_references.md'}")
    print(f"  corpus n={len(recs)}; new refs [{START_REF}..{n-1}] ({len(refs)} entries); "
          f"reused base refs: {sorted(set(ALREADY.values()))}")


if __name__ == "__main__":
    main()
