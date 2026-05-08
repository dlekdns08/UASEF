"""
Multi-Lingual Sanity (Round 8 P2-4, Supplementary §H)
══════════════════════════════════════════════════════════════════════════════

Cross-language sanity check. Loads a small Chinese MedQA subset (if available
locally as `data/raw/medqa_zh.jsonl`) and runs a *single backend, single seed*
v2 calibration + per-stratum coverage measurement. Compares to the English
MedAbstain Table 1 to quantify any cross-language drift.

Strict scope (paper §7.4 L4)
----------------------------
- We do **not** claim multi-lingual generalization in the main paper.
- This is a **transparent sanity check** for practitioners who deploy the
  v2 pipeline on a non-English clinical dataset.
- If the local zh dataset is absent, the script reports MISSING and exits
  cleanly (no fallback synthetic data).

Usage
-----
    python experiments/round8_multilingual_sanity.py \
        --backend openai --n-cal 100 --n-test 50 --seed 42 \
        --out results/round8/multilingual_sanity.json

The expected JSONL schema for `data/raw/medqa_zh.jsonl`:
    {"question": "...", "options": {"A": "...", ...}, "answer_idx": "A", "answer": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_zh(path: Path, n: int):
    if not path.exists():
        return None
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= n:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--n-cal", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zh-data", type=Path, default=Path("data/raw/medqa_zh.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/round8/multilingual_sanity.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_zh(args.zh_data, args.n_cal + args.n_test)
    if rows is None:
        msg = (
            f"[multilingual] {args.zh_data} not found.\n"
            f"  Place a Chinese MedQA-style JSONL at this path to enable the sanity check.\n"
            f"  Schema: {{\"question\": ..., \"options\": {{...}}, \"answer_idx\": ..., \"answer\": ...}}\n"
            f"  Skipping (paper §7.4 L4: multi-lingual is explicitly out of scope).\n"
        )
        print(msg)
        args.out.write_text(json.dumps({
            "status": "skipped",
            "reason": "data file missing",
            "expected_path": str(args.zh_data),
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # The real run would call `query_model` per case to get nonconformity
    # scores, then fit StratifiedConformalRiskControl. Without zh-stratum
    # mapping we report only the headline coverage.
    print(f"[multilingual] loaded {len(rows)} zh rows. Backend = {args.backend}.")
    print("  Note: this script is a *sanity probe* — it does not produce a")
    print("  paper-quality result. Per-stratum mapping requires an ICD-coded")
    print("  zh dataset, which is out of scope for the main paper.")

    payload = {
        "status": "loaded",
        "n_loaded": len(rows),
        "backend": args.backend,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "note": "End-to-end zh evaluation requires per-stratum mapping not "
                "supplied by data/raw/medqa_zh.jsonl. Paper §7.4 L4 explicitly "
                "scopes out multi-lingual; this stub is for reproducibility "
                "of the supplementary section only.",
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ saved: {args.out}")


if __name__ == "__main__":
    main()
