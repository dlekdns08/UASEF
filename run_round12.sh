#!/usr/bin/env bash
#
# UASEF Round 12 — Original vision revival
# ════════════════════════════════════════════════════════════════════════════
#
# R11 → R12 pivot 이유:
#   R10-R11 은 tabular structured features 로 outcome prediction — LLM 이
#   당연히 못하는 문제였음. 원래 UASEF vision (R6-R8) 은:
#     Clinical QA → LLM answer + logprobs → mean-NLL nonconformity → CP threshold
#   즉 LLM 이 primary decision maker + CP 가 LLM uncertainty 를 gate.
#
# R12.1 = MedAbstain 에서 원래 vision 재구현 + R11 audit discipline 적용:
#   - LLM answer 의 mean-NLL 을 nonconformity score
#   - Stratified CRC per {CRIT, HIGH, MOD, LOW}
#   - over_esc rate 명시 (R11.3 lesson)
#   - Strict verdict: α-satisfy AND pooled_over_esc < 0.95 = GENUINE_WIN
#
# 사용 예
# ──────
#   export UASEF_BACKEND_NEVER_SEND_PHI=1  # optional, MedAbstain 은 public
#   caffeinate -dimsu bash run_round12.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

SEEDS="${SEEDS:-42 43 44 45 46}"
N_CAL="${N_CAL:-400}"
N_TEST="${N_TEST:-400}"
POS_DATASET="${POS_DATASET:-${ROOT}/data/raw/medabstain_A.jsonl}"
NEG_DATASET="${NEG_DATASET:-${ROOT}/data/raw/medabstain_NA.jsonl}"
BACKEND="${BACKEND:-lmstudio}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
BACKGROUND="${BACKGROUND:-0}"

find "${ROOT}/experiments/__pycache__" -name "round12*.pyc" -delete 2>/dev/null || true

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/results/round12/all_in_one_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/run.log"
exec > >(tee -a "$LOG") 2>&1

C_BOLD=$'\033[1m'; C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_RST=$'\033[0m'

echo ""
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
echo "${C_BOLD}  UASEF Round 12 — LLM-as-primary + CP escalation gate${C_RST}"
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
echo "  TIMESTAMP    : $TIMESTAMP"
echo "  Pos dataset  : $POS_DATASET"
echo "  Neg dataset  : $NEG_DATASET"
echo "  Seeds        : $SEEDS"
echo "  n_cal/n_test : $N_CAL / $N_TEST"
echo "  Backend      : $BACKEND / model=$MODEL"
echo ""
echo "  Nonconformity: s(x) = mean token NLL of LLM answer"
echo "  Vision       : LLM primary + CP gate (R6-R8 original)"
echo ""

CMD=(
    "$PYTHON" experiments/round12_medabstain_llm_gate.py
    --pos-dataset "$POS_DATASET"
    --neg-dataset "$NEG_DATASET"
    --seeds $SEEDS
    --n-cal "$N_CAL" --n-test "$N_TEST"
    --backend "$BACKEND" --model "$MODEL"
    --out "${ROOT}/results/round12/r12_1_medabstain_llm_gate"
)
if [ "$BACKGROUND" = "1" ]; then
    echo "  ▶ BACKGROUND mode — caffeinate + nohup"
    caffeinate -dimsu nohup "${CMD[@]}" > "${RUN_DIR}/nohup.log" 2>&1 &
    PID=$!
    echo "  PID: $PID"
    echo "  Log: ${RUN_DIR}/nohup.log"
    echo "  진행 확인: tail -f ${RUN_DIR}/nohup.log"
    exit 0
fi

t0=$(date +%s)
"${CMD[@]}"
rc=$?
elapsed=$(( $(date +%s) - t0 ))

echo ""
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
if [ $rc -eq 0 ]; then
    echo "${C_BOLD}${C_OK}  Round 12 R12.1 완료 in ${elapsed}s${C_RST}"
    echo ""
    echo "  📄 산출물:"
    echo "    - results/round12/r12_1_medabstain_llm_gate.{json,md}"
    echo "    - results/round12/r12_1_cache/seed_*.json"
    echo ""
    echo "  📋 다음 액션:"
    echo "    1. r12_1_medabstain_llm_gate.md 의 'strict verdict' 확인"
    echo "    2. GENUINE_WIN 이면 → paper §9 (R12 revisited vision) 작성"
    echo "    3. VACUOUS_WIN / FAIL 이면 → 정직한 negative reporting"
else
    echo "${C_BOLD}${C_ERR}  Round 12 R12.1 FAILED rc=$rc after ${elapsed}s${C_RST}"
fi
echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
