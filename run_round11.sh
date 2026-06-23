#!/usr/bin/env bash
#
# UASEF Round 11 — R10.4 verification (minimal-feature re-run)
# ════════════════════════════════════════════════════════════════════════════
#
# 단일 실험: R11.1
#   - R10.4 와 동일한 5 classifier × 5 seed 구조
#   - 단, feature vector 는 4 minimal features only
#       (Charlson, specialty_baseline_rate, n_vital_flags 제거 — R10.7 leakage suspect)
#   - over_esc rate 명시 보고 → vacuous CRC 자동 검출
#
# Wallclock: ~64 hours (LLM dominates)
# External API cost: $0
#
# 사용 예
# ──────
#   export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
#   export UASEF_BACKEND_NEVER_SEND_PHI=1
#   caffeinate -dimsu bash run_round11.sh
#
#   # tabular only (~30s, LLM 건너뛰기):
#   CLASSIFIERS="logreg gbdt randomforest xgboost" bash run_round11.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

CLASSIFIERS="${CLASSIFIERS:-gpt_oss_120b logreg gbdt randomforest xgboost}"
SEEDS="${SEEDS:-42 43 44 45 46}"
N_CAL="${N_CAL:-3000}"
N_TEST="${N_TEST:-3000}"

# stale .pyc cache 방지 (2026-06-23 사례)
find "${ROOT}/experiments/__pycache__" -name "round1*.pyc" -delete 2>/dev/null || true

export UASEF_BACKEND_NEVER_SEND_PHI="${UASEF_BACKEND_NEVER_SEND_PHI:-1}"
export UASEF_QUERY_TIMEOUT_S="${UASEF_QUERY_TIMEOUT_S:-60}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/results/round11/all_in_one_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/run.log"
exec > >(tee -a "$LOG") 2>&1

C_BOLD=$'\033[1m'; C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_RST=$'\033[0m'

echo ""
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
echo "${C_BOLD}  UASEF Round 11 — R10.4 verification (minimal feature)${C_RST}"
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
echo "  TIMESTAMP    : $TIMESTAMP"
echo "  RUN_DIR      : $RUN_DIR"
echo "  Classifiers  : $CLASSIFIERS"
echo "  Seeds        : $SEEDS"
echo "  N_cal/N_test : $N_CAL / $N_TEST"
echo "  PHI guard    : ${UASEF_BACKEND_NEVER_SEND_PHI:-not set}"
echo ""
echo "  Features (minimal, leakage-safe):"
echo "    - age_bucket, adm_emerg, spec_idx, n_labs"
echo "  Features removed (R10.7 leakage suspect):"
echo "    - charlson_index, specialty_baseline_rate, n_vital_flags"
echo ""

t0=$(date +%s)
"$PYTHON" experiments/round11_method_agnostic_minimal.py \
    --classifiers $CLASSIFIERS \
    --seeds $SEEDS \
    --n-cal "$N_CAL" --n-test "$N_TEST" \
    --out "${ROOT}/results/round11/r11_1_method_agnostic_minimal"
rc=$?
elapsed=$(( $(date +%s) - t0 ))

echo ""
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
if [ $rc -eq 0 ]; then
    echo "${C_BOLD}${C_OK}  Round 11 R11.1 완료 in ${elapsed}s${C_RST}"
    echo ""
    echo "  📄 결과:"
    echo "    - results/round11/r11_1_method_agnostic_minimal.{json,md}"
    echo "    - results/round11/r11_1_cache/<classifier>.json (per-seed)"
    echo ""
    echo "  📋 다음 액션:"
    echo "    1. r11_1_method_agnostic_minimal.md 의 'R11.1 verdict' 섹션 확인"
    echo "    2. R10.4 의 RF win 이 vacuous 인지 진실 인지 결정"
    echo "    3. paper/UASEF_FINAL.md §5.4, §6, §7 수정 (R11 결과 기반)"
else
    echo "${C_BOLD}${C_ERR}  Round 11 R11.1 FAILED rc=$rc after ${elapsed}s${C_RST}"
    echo "  log: $LOG"
fi
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
