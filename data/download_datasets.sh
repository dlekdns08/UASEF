#!/usr/bin/env bash
#
# UASEF — Dataset download script (Round 8 P0-2)
# ════════════════════════════════════════════════════════════════════════════
#
# Downloads MedQA-USMLE and MedAbstain into data/raw/ for paper reproduction.
# MedAbstain의 라이선스/출처는 변경 가능하므로 수동 단계 안내 포함.
#
# Usage:
#   bash data/download_datasets.sh
#
# After running, verify with:
#   UASEF_PAPER_REPRODUCTION=1 python -c "from data.loader import load_medqa_cases, load_medabstain_cases; print(len(load_medqa_cases())); print(len(load_medabstain_cases(variants=['AP','NAP','A','NA'])))"

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${ROOT}/data/raw"
mkdir -p "$RAW_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  UASEF — Dataset download"
echo "════════════════════════════════════════════════════════════════════"
echo "  Target dir : $RAW_DIR"
echo ""

# ── MedQA-USMLE ───────────────────────────────────────────────────────────
# HuggingFace: GBaker/MedQA-USMLE-4-options
# 자동 다운로드는 loader 내부에서도 시도하므로 여기서는 캐시 사전 적재.
echo "── MedQA-USMLE (HuggingFace GBaker/MedQA-USMLE-4-options) ─────────"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("[error] `datasets` not installed: pip install datasets")

raw = Path(__file__).resolve().parent / "raw" if False else Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)

ds = load_dataset("GBaker/MedQA-USMLE-4-options")
for split in ("train", "test"):
    out = raw / f"medqa_{split}.jsonl"
    if out.exists():
        print(f"  skip (exists): {out}")
        continue
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds[split]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n:>5} rows → {out}")
PY

# ── MedAbstain ────────────────────────────────────────────────────────────
# 출처가 시간에 따라 변할 수 있어 자동 다운로드는 best-effort.
# 1) sravanthi6m/MedAbstain (GitHub mirror) — README에 안내된 경로 사용
# 2) Machcha et al. 2026 EACL 공식 부록 — 라이선스 확인 필요
echo ""
echo "── MedAbstain (Machcha et al. 2026 EACL / sravanthi6m/MedAbstain) ─"

cat <<'NOTE'
  MedAbstain은 license 변경 가능성이 있어 수동 단계가 필요합니다.

  옵션 A — Machcha et al. 2026 EACL 공식 부록 (권장)
    1) https://aclanthology.org/ 에서 paper 검색 → supplementary download
    2) 4개 변형(AP/NAP/A/NA) JSONL을 다음 경로에 위치:
         data/raw/medabstain_AP.jsonl
         data/raw/medabstain_NAP.jsonl
         data/raw/medabstain_A.jsonl
         data/raw/medabstain_NA.jsonl

  옵션 B — sravanthi6m/MedAbstain mirror (확인 필요)
    git clone https://github.com/sravanthi6m/MedAbstain "$RAW_DIR/_medabstain_mirror" || true
    # 해당 repo의 jsonl을 medabstain_*.jsonl 형식으로 변환 후 위치

  레이아웃 검증:
    python -c "from pathlib import Path; \
      assert all((Path('data/raw') / f'medabstain_{v}.jsonl').exists() \
        for v in ['AP','NAP','A','NA']), 'missing variants'"
NOTE

# ── 검증 ──────────────────────────────────────────────────────────────────
echo ""
echo "── 검증 ────────────────────────────────────────────────────────────"
ok=0
for v in AP NAP A NA; do
    p="$RAW_DIR/medabstain_${v}.jsonl"
    if [ -f "$p" ]; then
        n=$(wc -l < "$p" | tr -d ' ')
        echo "  ✓ $p ($n rows)"
        ok=$((ok+1))
    else
        echo "  ✗ $p (missing)"
    fi
done
[ -f "$RAW_DIR/medqa_train.jsonl" ] && echo "  ✓ medqa_train.jsonl" || echo "  ✗ medqa_train.jsonl (missing)"
[ -f "$RAW_DIR/medqa_test.jsonl" ]  && echo "  ✓ medqa_test.jsonl"  || echo "  ✗ medqa_test.jsonl (missing)"

echo ""
if [ "$ok" -eq 4 ]; then
    echo "  ✅ MedAbstain 4 variants 모두 OK. paper 재현 가능."
    echo "  실행 예시: UASEF_PAPER_REPRODUCTION=1 bash run_full_evaluation.sh"
else
    echo "  ⚠ MedAbstain 변형이 ${ok}/4 만 존재. 위 매뉴얼 단계 진행 후 다시 실행."
    exit 1
fi
