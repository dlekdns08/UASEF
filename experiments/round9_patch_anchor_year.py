"""
Round 9 hotfix — 기존 mimic4_cases.jsonl 에 anchor_year_group 필드 추가.

MIMIC-IV 의 admittime 은 deidentification 으로 환자별 random year-shift 가
적용되어 2110-2212 범위에 있다. 실제 calendar era 는 patients.csv.gz 의
anchor_year_group 컬럼 (2008-2010, 2011-2013, 2014-2016, 2017-2019,
2020-2022) 에 있다. R9.4 temporal shift 가 동작하려면 이 필드가 필요.

Preprocess 를 전체 재실행하지 않고 기존 JSONL 에 patch 만 적용.

Usage
-----
    python experiments/round9_patch_anchor_year.py \\
        --mimic-dir $MIMIC4_DIR \\
        --jsonl data/raw/mimic-iv/mimic4_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", type=Path,
                    default=Path(os.environ.get("MIMIC4_DIR", "")))
    ap.add_argument("--jsonl", type=Path,
                    default=ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases.jsonl")
    args = ap.parse_args()

    if not args.mimic_dir or not args.mimic_dir.exists():
        sys.exit(f"--mimic-dir 누락 또는 미존재: {args.mimic_dir}")
    if not args.jsonl.exists():
        sys.exit(f"JSONL 미존재: {args.jsonl}. preprocess 먼저 실행 필요.")

    print(f"[1/3] Loading patients.csv.gz ...")
    import pandas as pd
    pat = pd.read_csv(args.mimic_dir / "hosp" / "patients.csv.gz",
                      compression="gzip",
                      usecols=["subject_id", "anchor_year_group"])
    grp_map = dict(zip(pat["subject_id"].astype(str), pat["anchor_year_group"]))
    print(f"      {len(grp_map)} subjects loaded")

    print(f"[2/3] Reading existing JSONL ...")
    rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    print(f"      {len(rows)} cases loaded")

    print(f"[3/3] Patching anchor_year_group ...")
    patched = 0
    missing = 0
    group_counts: dict[str, int] = {}
    for row in rows:
        sid = str(row.get("subject_id", ""))
        grp = grp_map.get(sid)
        if grp is None:
            missing += 1
            row["anchor_year_group"] = None
        else:
            row["anchor_year_group"] = grp
            patched += 1
            group_counts[grp] = group_counts.get(grp, 0) + 1

    # atomic write
    tmp = args.jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.jsonl)

    print(f"\n  patched : {patched}")
    print(f"  missing : {missing}")
    print(f"  groups  :")
    for g in sorted(group_counts):
        print(f"    {g}: {group_counts[g]}")
    print(f"\n✅ {args.jsonl}")


if __name__ == "__main__":
    main()
