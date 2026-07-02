#!/usr/bin/env bash
#
# Camera-ready blocking item — R11.3 eICU LLM Pass A + B
# ════════════════════════════════════════════════════════════════════════════
#
# 5-seed × 2-pass × 6000 calls @ 0.10 calls/s = ~166h ≈ 7 days.
# Runs in background with caffeinate + nohup so terminal can close.
#
# 사용:
#   bash run_r11_3_full_llm.sh                # 백그라운드 시작 + PID 출력
#
# 진행 확인:
#   tail -f results/round11/r11_3_full_llm.log
#   grep "seed=" results/round11/r11_3_full_llm.log | tail -3
#   grep -c "3000/3000" results/round11/r11_3_full_llm.log   # 20 = complete
#
# 중단:
#   ps aux | grep round11_eicu_repl | grep -v grep | awk '{print $2}' | xargs kill

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

# Prerequisites 검증
if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:1234/v1/models | grep -q "200"; then
    echo "❌ LMStudio not reachable at localhost:1234"
    echo "   Start LMStudio and load openai/gpt-oss-120b first."
    exit 1
fi
echo "✓ LMStudio OK"

JSONL="${ROOT}/data/raw/eicu_cases_v11_full.jsonl"
if [ ! -f "$JSONL" ]; then
    echo "❌ Missing $JSONL"
    echo "   Run: .venv/bin/python experiments/round11_eicu_preprocess.py"
    exit 1
fi
echo "✓ eICU JSONL present ($(wc -l < "$JSONL") lines)"

# Stale .pyc clear
find "${ROOT}/experiments/__pycache__" -name "round11*.pyc" -delete 2>/dev/null || true

export UASEF_BACKEND_NEVER_SEND_PHI="${UASEF_BACKEND_NEVER_SEND_PHI:-1}"
export UASEF_QUERY_TIMEOUT_S="${UASEF_QUERY_TIMEOUT_S:-60}"

LOG="${ROOT}/results/round11/r11_3_full_llm.log"
mkdir -p "$(dirname "$LOG")"
# Preserve previous log if any
[ -f "$LOG" ] && mv "$LOG" "${LOG%.log}_$(date +%Y%m%d-%H%M%S).log"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  R11.3 eICU LLM Pass A + B — background launch"
echo "════════════════════════════════════════════════════════════════════"
echo "  Wallclock estimate: ~166h ≈ 7 days"
echo "  n_cal=3000, n_test=3000, seeds=42-46, 2 passes"
echo "  Classifier: gpt_oss_120b ONLY (tabular already cached)"
echo "  Log: $LOG"
echo ""

caffeinate -dimsu nohup "$PYTHON" \
    experiments/round11_eicu_replication.py \
    --jsonl "$JSONL" \
    --include-llm \
    --classifiers gpt_oss_120b \
    --n-cal 3000 --n-test 3000 \
    --seeds 42 43 44 45 46 \
    > "$LOG" 2>&1 &

PID=$!
echo "✓ Background PID: $PID"
echo ""
echo "Monitoring commands:"
echo "  tail -f $LOG"
echo "  grep 'seed=' $LOG | tail -3"
echo "  grep -c '3000/3000' $LOG    # 20 = complete"
echo ""
echo "Stop command:"
echo "  kill $PID"
