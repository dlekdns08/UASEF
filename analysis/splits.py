"""
Common split manifest — ONE item-grouped, dataset-stratified fold assignment reused by
EVERY analysis (calibration, q-proxy training, threshold selection, final evaluation).

Why: the same canonical item appears in many cells (answerer × verifier × mode ×
original/shuffled). If folds were drawn per-analysis, an item could train a model in one
analysis and test it in another (leakage), and PubMedQA (317 items) would fragment.
Rule: ALL rows derived from one canonical item live in the SAME fold, everywhere.

Deterministic: per dataset, item_ids are sorted then Random(0)-shuffled and dealt
round-robin into K=5 folds — no timestamp, no environment dependence; re-running yields
byte-identical output. Usage is cross-fitting: fit on 4 folds, apply to the held-out
fold, rotate (item-grouped 3-way splits would leave PubMedQA test sets too small).

Run:  python analysis/splits.py        (writes results/consolidated/00_split_manifest.csv)
Use:  from analysis.splits import load_folds; folds = load_folds()  # item_id -> fold_id
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

K = 5
OUT = ROOT / "results" / "consolidated" / "00_split_manifest.csv"


def build():
    from analysis.consolidate import matrix_item_ids
    from analysis.manifest import dataset_of
    ids = matrix_item_ids()          # the canonical 1500 (shuffle-400 is a subset)
    by_ds = defaultdict(list)
    for i in ids:
        by_ds[dataset_of(i)].append(i)
    rows = []
    for ds in sorted(by_ds):
        items = sorted(by_ds[ds])
        random.Random(0).shuffle(items)
        for j, iid in enumerate(items):
            rows.append({"item_id": iid, "dataset": ds, "fold_id": j % K})
    rows.sort(key=lambda r: r["item_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "dataset", "fold_id"])
        w.writeheader(); w.writerows(rows)
    return rows


def load_folds() -> dict:
    """item_id -> fold_id. Builds the manifest on first use; identical on rebuild."""
    if not OUT.exists():
        build()
    return {r["item_id"]: int(r["fold_id"]) for r in csv.DictReader(open(OUT))}


if __name__ == "__main__":
    rows = build()
    from collections import Counter
    c = Counter((r["dataset"], r["fold_id"]) for r in rows)
    print(f"[splits] {len(rows)} items -> {K} folds (item-grouped, dataset-stratified)")
    for ds in sorted({r["dataset"] for r in rows}):
        print(f"  {ds}: " + " ".join(f"f{k}={c[(ds, k)]}" for k in range(K)))
    print(f"  wrote {OUT}")
