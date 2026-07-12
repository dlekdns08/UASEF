"""
Safe repair prep — removes ONLY invalid rows (None risk / blank self-answer) so the
next resume run re-queries exactly those items and NEVER regenerates valid rows.

Guarantees enforced here (the delete step), paired with the scripts' own resume logic
(the regenerate step — every generator builds a done-set of item_ids and SKIPS them):
  1. refuses to run while any experiment process is live (a mid-run delete is how the
     earlier duplicate-row corruption happened)
  2. deletes ONLY rows whose payload is invalid — valid rows are byte-preserved
  3. atomic rewrite (tmp + replace) + timestamped .bak backup
  4. prints exact kept/removed counts; the relaunched job's "N cached, M to judge"
     header MUST show cached == kept (verify before letting it run on)

Usage:
  python analysis/purge_invalid.py data/raw/shuffle_judge_qwen35__gpt_high.jsonl
  python analysis/purge_invalid.py data/raw/selfanswer_gptossHigh.jsonl
Then re-run the ORIGINAL generation command (larger --max-tokens if truncation caused
the invalids) and confirm the cached count, then run analysis/preflight.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def invalid_reason(stem, r):
    """return a reason string if the row is invalid for its file family, else None."""
    if stem.startswith(("verifier_", "shuffle_judge_")):
        if r.get("verifier_risk") is None:
            return "risk=None"
    elif stem.startswith("selfanswer_"):
        if not (r.get("self_answer") or "").strip():
            return "blank self_answer"
    elif stem.startswith(("drafts_", "shuffle_answer_")):
        if not (r.get("decision_answer") or r.get("answerer_output_label") or "").strip():
            return "blank answer"
    return None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: purge_invalid.py <jsonl path>")
    p = Path(sys.argv[1])
    stem = p.stem
    if not p.exists():
        sys.exit(f"없음: {p}")

    # 1. live-process guard — never edit a file a job may be appending to
    live = subprocess.run(["pgrep", "-fl", "phase2_|phase0_|phase1_"],
                          capture_output=True, text=True).stdout.strip()
    if live:
        sys.exit(f"⛔ 실험 프로세스 실행 중 — 먼저 종료 후 실행:\n{live}")

    rows = [json.loads(l) for l in open(p) if l.strip()]
    keep, drop = [], []
    for r in rows:
        why = invalid_reason(stem, r)
        (drop if why else keep).append((r, why))

    if not drop:
        print(f"[purge] {stem}: 무효 행 0 — 변경 없음 ({len(keep)}행 전부 유효)")
        return

    # 2-3. backup + atomic rewrite of VALID rows only
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = p.with_suffix(f".jsonl.bak_{ts}")
    shutil.copy2(p, bak)
    tmp = p.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r, _ in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

    print(f"[purge] {stem}")
    print(f"  유지(유효): {len(keep)}행 — 재생성되지 않음 (resume이 item_id로 skip)")
    print(f"  삭제(무효): {len(drop)}행 -> 재생성 대상")
    for (r, why) in drop[:10]:
        print(f"    - {r.get('item_id')}: {why}")
    if len(drop) > 10:
        print(f"    ... 외 {len(drop) - 10}개")
    print(f"  백업: {bak.name}")
    print(f"  ▶ 다음: 원래 생성 명령 재실행 → 헤더 'cached'가 {len(keep)}인지 확인 → preflight.py")


if __name__ == "__main__":
    main()
