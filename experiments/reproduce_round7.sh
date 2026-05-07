#!/usr/bin/env bash
#
# UASEF Round 7 — 논문 표 4종을 한 번에 생성하는 재현 스크립트.
#
# 사용:
#     bash experiments/reproduce_round7.sh             # default (n_cal=200, n_test=100)
#     N_CAL=1500 N_TEST=500 bash experiments/reproduce_round7.sh    # 논문 quality
#
# 환경변수:
#     BACKEND        — openai (default) | lmstudio
#     N_CAL          — calibration data per stratum (논문: ≥1000 for α=0.001)
#     N_TEST         — holdout test data per stratum
#     N_TRIALS       — Table 2 FWER simulation 반복 (default 5000)
#     N_PER_STRATUM  — Table 3 합성 데이터 (default 300)
#
# 출력: results/round7/table{1,2,3,4}.{json,md} + results/round7/summary.md

set -euo pipefail

BACKEND="${BACKEND:-openai}"
N_CAL="${N_CAL:-200}"
N_TEST="${N_TEST:-100}"
N_TRIALS="${N_TRIALS:-5000}"
N_PER_STRATUM="${N_PER_STRATUM:-300}"

OUT_DIR="results/round7"
mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  UASEF Round 7 reproduce — backend=$BACKEND, n_cal=$N_CAL, n_test=$N_TEST"
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "[1/4] Table 1 — Per-Stratum Coverage Validity (Pivot A)"
echo "      ⚠ LLM 호출 ~$((N_CAL * 4 + N_TEST * 4))회"
python experiments/round7_table1_coverage.py \
    --backend "$BACKEND" --n-cal "$N_CAL" --n-test "$N_TEST" \
    || echo "  [SKIP] Table 1 실패 (LLM 키 없음 가능)"

echo ""
echo "[2/4] Table 2 — Multi-Trigger FWER (Pivot B, 합성 데이터)"
python experiments/round7_table2_fwer.py \
    --n-trials "$N_TRIALS" --n-cal 200 --alpha 0.05

echo ""
echo "[3/4] Table 3 — Cost-Weighted Performance (Pivot C, 합성 데이터)"
python experiments/round7_table3_cost.py --n-per-stratum "$N_PER_STRATUM"

echo ""
echo "[4/4] Table 4 — Baseline Comparison (TECP/Quach/SE vs Round 6/7)"
echo "      Table 1과 동일 LLM 호출 풀 재사용 권장: --reuse-table1"
python experiments/round7_table4_baseline.py \
    --backend "$BACKEND" --n-cal "$N_CAL" --n-test "$N_TEST" --reuse-table1 \
    || echo "  [SKIP] Table 4 실패 (LLM 키 없음 가능)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  완료 — 산출물: $OUT_DIR/"
ls -la "$OUT_DIR/" 2>/dev/null

# summary.md 생성
{
    echo "# UASEF Round 7 — Summary"
    echo ""
    echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Config: backend=$BACKEND, n_cal=$N_CAL, n_test=$N_TEST"
    echo ""
    for t in table1_coverage table2_fwer table3_cost table4_baseline; do
        if [ -f "$OUT_DIR/$t.md" ]; then
            cat "$OUT_DIR/$t.md"
            echo ""
            echo "---"
            echo ""
        fi
    done
} > "$OUT_DIR/summary.md"

echo ""
echo "  ✅ 통합 보고서: $OUT_DIR/summary.md"
