"""
Per-Stratum AUROC Equity Audit (Round 8 P2-5, Supplementary §I)
══════════════════════════════════════════════════════════════════════════════

Reads `results/run_<ts>/<backend>/table4_baseline.json` and computes:

  - per-stratum AUROC variance (across CRITICAL / HIGH / MODERATE / LOW)
  - per-stratum Safety-Recall and Over-Esc-Rate spread

A wide AUROC spread (e.g., 0.95 on CRITICAL but 0.60 on MODERATE) signals
that the v2 pipeline allocates capacity asymmetrically across risk classes.
We report this as a *diagnostic*, not as a main-paper claim.

Usage
-----
    python experiments/round8_equity_audit.py \
        --table4 results/run_<ts>/openai/table4_baseline.json \
        --out results/round8/equity_audit_openai.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def per_stratum_audit(methods: list[dict]) -> dict:
    out: dict = {}
    for m in methods:
        name = m.get("name", "?")
        per = m.get("per_stratum") or {}
        recalls = []
        oers = []
        aurocs = []
        for s in ("CRITICAL", "HIGH", "MODERATE", "LOW"):
            cell = per.get(s) or {}
            if cell.get("safety_recall") is not None:
                recalls.append(cell["safety_recall"])
            if cell.get("over_esc_rate") is not None:
                oers.append(cell["over_esc_rate"])
        # Method-level AUROC may be a single number; we still report it.
        if m.get("auroc") is not None:
            aurocs.append(m["auroc"])
        def _spread(xs):
            if len(xs) < 2:
                return None
            return round(max(xs) - min(xs), 4)
        def _stdev(xs):
            return round(statistics.stdev(xs), 4) if len(xs) > 1 else None
        out[name] = {
            "n_strata_with_recall": len(recalls),
            "recall_min": min(recalls) if recalls else None,
            "recall_max": max(recalls) if recalls else None,
            "recall_spread": _spread(recalls),
            "recall_stdev": _stdev(recalls),
            "over_esc_min": min(oers) if oers else None,
            "over_esc_max": max(oers) if oers else None,
            "over_esc_spread": _spread(oers),
            "over_esc_stdev": _stdev(oers),
            "auroc": aurocs[0] if aurocs else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table4", type=Path, required=True,
                    help="Path to table4_baseline.json (e.g., results/run_<ts>/<backend>/table4_baseline.json)")
    ap.add_argument("--out", type=Path, default=Path("results/round8/equity_audit.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.table4.exists():
        raise SystemExit(f"[error] table4 not found: {args.table4}")

    table4 = json.loads(args.table4.read_text())
    audit = per_stratum_audit(table4.get("methods", []))

    payload = {
        "timestamp": datetime.now().isoformat(),
        "source": str(args.table4),
        "audit": audit,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = ["# Round 8 — Per-Stratum Equity Audit (Supplementary §I)", "",
          f"source: `{args.table4}`", "",
          "Wider spread = more capacity asymmetry across risk strata.",
          "",
          "| Method | recall min | recall max | recall spread | over-esc spread | AUROC |",
          "| --- | --- | --- | --- | --- | --- |"]
    for name, info in audit.items():
        md.append(
            f"| {name} | {info['recall_min']} | {info['recall_max']} | "
            f"{info['recall_spread']} | {info['over_esc_spread']} | {info['auroc']} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✅ saved: {args.out} (+ .md)")


if __name__ == "__main__":
    main()
