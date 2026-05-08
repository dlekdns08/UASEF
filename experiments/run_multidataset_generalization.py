"""
Multi-Dataset Generalization Sweep (Round 8 P1-4)
══════════════════════════════════════════════════════════════════════════════

Runs Tables 1 + 4 across multiple datasets ({medabstain, medqa_usmle, pubmedqa})
on a single backend and emits a unified summary that supplementary §D can cite.

Usage
-----
    python experiments/run_multidataset_generalization.py \
        --backend openai \
        --datasets medabstain medqa_usmle pubmedqa \
        --n-cal 200 --n-test 100 --seed 42 \
        --out results/round8/multidataset_summary.json

Implementation notes
--------------------
- Re-uses the existing experiments/round7_table{1,4}_*.py scripts as
  subprocesses (no logic duplication); each writes per-dataset JSON to
  results/round7/, which we then read back and consolidate.
- Honest framing for the paper: this is **single-seed evidence of
  generalization**, not multi-seed empirical claims. Multi-seed across
  multiple datasets is not in scope; supplementary §D cites this as
  "directional evidence."
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run_table1(python: str, backend: str, dataset: str, n_cal: int, n_test: int, alpha: float, seed: int) -> Path | None:
    cmd = [
        python, "experiments/round7_table1_coverage.py",
        "--backend", backend, "--dataset", dataset,
        "--n-cal", str(n_cal), "--n-test", str(n_test),
        "--alpha-global", str(alpha), "--seed", str(seed),
    ]
    print(f"  → {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  [warn] Table 1 returned {rc} for {dataset}")
        return None
    suf = "" if dataset == "medabstain" else f"_{dataset}"
    p = Path(f"results/round7/table1_coverage{suf}.json")
    return p if p.exists() else None


def _run_table4(python: str, backend: str, dataset: str, n_cal: int, n_test: int, alpha: float, seed: int) -> Path | None:
    cmd = [
        python, "experiments/round7_table4_baseline.py",
        "--backend", backend, "--dataset", dataset,
        "--n-cal", str(n_cal), "--n-test", str(n_test),
        "--alpha", str(alpha), "--seed", str(seed),
    ]
    print(f"  → {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  [warn] Table 4 returned {rc} for {dataset}")
        return None
    suf = "" if dataset == "medabstain" else f"_{dataset}"
    p = Path(f"results/round7/table4_baseline{suf}.json")
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--datasets", nargs="+", default=["medabstain", "medqa_usmle", "pubmedqa"])
    ap.add_argument("--n-cal", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/round8/multidataset_summary.json"))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    archive = args.out.parent / f"raw_{args.backend}_seed{args.seed}"
    archive.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend,
        "seed": args.seed,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "alpha": args.alpha,
        "datasets": {},
    }

    for ds in args.datasets:
        print(f"\n══ dataset = {ds} ══")
        t1 = _run_table1(args.python, args.backend, ds, args.n_cal, args.n_test, args.alpha, args.seed)
        t4 = _run_table4(args.python, args.backend, ds, args.n_cal, args.n_test, args.alpha, args.seed)

        entry: dict = {"table1_path": str(t1) if t1 else None,
                       "table4_path": str(t4) if t4 else None}
        if t1 and t1.exists():
            entry["table1"] = json.loads(t1.read_text())
            shutil.copy2(t1, archive / t1.name)
        if t4 and t4.exists():
            entry["table4"] = json.loads(t4.read_text())
            shutil.copy2(t4, archive / t4.name)
        summary["datasets"][ds] = entry

    # Compact view: per-dataset CRITICAL safety recall + total cost for v2
    compact: list[dict] = []
    for ds, e in summary["datasets"].items():
        t4 = e.get("table4") or {}
        v2 = next((m for m in t4.get("methods", []) if m.get("name", "").startswith("UASEF Round 7")), {})
        crit = (v2.get("per_stratum") or {}).get("CRITICAL") or {}
        compact.append({
            "dataset": ds,
            "v2_critical_safety_recall": crit.get("safety_recall"),
            "v2_total_cost": v2.get("total_cost"),
            "n_methods": len(t4.get("methods", [])),
        })
    summary["compact_v2"] = compact

    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = ["# Round 8 — Multi-Dataset Generalization (supplementary §D)", "",
          f"- timestamp: {summary['timestamp']}",
          f"- backend: {args.backend}, seed: {args.seed} (single-seed; supplementary §D cites as directional evidence)",
          "", "## v2 (Round 7) per-dataset summary", "",
          "| Dataset | CRITICAL Safety Recall | Total Cost | n_methods |",
          "| --- | --- | --- | --- |"]
    for c in compact:
        md.append(
            f"| {c['dataset']} | "
            f"{c['v2_critical_safety_recall'] if c['v2_critical_safety_recall'] is not None else '—'} | "
            f"{c['v2_total_cost'] if c['v2_total_cost'] is not None else '—'} | {c['n_methods']} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✅ saved: {args.out} (+ .md)")


if __name__ == "__main__":
    main()
