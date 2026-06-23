#!/usr/bin/env bash
#
# UASEF — ML4H 2026 submission pipeline (Option A)
# ════════════════════════════════════════════════════════════════════════════
#
# 3 단계로 paper 를 ML4H 2026 submission-ready 로:
#
#   [Step 1] RF calibration 분석 (1-2시간 LLM + tabular)
#            → results/round10/r10_rf_calibration.{json,md}
#            → paper §6.5 의 "왜 RF 가 LLM 보다 잘 했나" 핵심 evidence
#
#   [Step 2] IRB case extraction (instant)
#            → data/raw/audit_r10_5/ (commit 금지)
#            → paper/irb_audit_package/ (physician 에게 전달)
#            * 별도 4주 외부 process 시작 — paper 의 final piece
#
#   [Step 3] Paper sync (instant)
#            → paper/UASEF_Round10.{md,KO.md} 의 §5 placeholder 를
#              실제 R10 결과로 교체
#
# 사용 예
# ──────
#   bash run_ml4h_submission.sh                    # 전체 3 단계
#   SKIP_RF_CALIBRATION=1 bash run_ml4h_submission.sh  # IRB + sync 만 (instant)
#   DRY_RUN=1 bash run_ml4h_submission.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

SKIP_RF_CALIBRATION="${SKIP_RF_CALIBRATION:-0}"
SKIP_IRB_EXTRACT="${SKIP_IRB_EXTRACT:-0}"
SKIP_PAPER_SYNC="${SKIP_PAPER_SYNC:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Round 10 의 환경 그대로
export UASEF_BACKEND_NEVER_SEND_PHI="${UASEF_BACKEND_NEVER_SEND_PHI:-1}"
export UASEF_QUERY_TIMEOUT_S="${UASEF_QUERY_TIMEOUT_S:-60}"

C_BOLD=$'\033[1m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_RST=$'\033[0m'
banner() { echo ""; echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
           echo "${C_BOLD}  $1${C_RST}"
           echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"; }
ok()   { echo "  ${C_OK}✓${C_RST} $1"; }
warn() { echo "  ${C_WARN}⚠${C_RST} $1"; }
err()  { echo "  ${C_ERR}✗${C_RST} $1"; }

run_step() {
    local key="$1"; shift
    local desc="$1"; shift
    local skip="$1"; shift
    if [ "$1" != "--" ]; then err "[$key] missing --"; return 2; fi
    shift
    if [ "$skip" = "1" ]; then warn "[$key] SKIP — $desc"; return 0; fi
    echo "  [$key] ▶ $desc"
    if [ "$DRY_RUN" = "1" ]; then echo "    DRY: $*"; return 0; fi
    local t0; t0=$(date +%s); local rc=0
    "$@" || rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then ok "[$key] done in ${elapsed}s"
    else err "[$key] FAILED rc=$rc after ${elapsed}s"; fi
    return $rc
}

banner "ML4H 2026 submission pipeline (Option A)"
echo "  ROOT       : $ROOT"
echo "  PYTHON     : $PYTHON"
echo "  PHI guard  : ${UASEF_BACKEND_NEVER_SEND_PHI:-not set}"
echo ""

# Clear stale .pyc cache — fix 한 source 가 stale bytecode 와 불일치하는 경우 방지.
# (2026-06-23: round10_5_irb_extract.py 가 stale .pyc 로 KeyError 한 사례)
find "${ROOT}/experiments/__pycache__" -name "round10*.pyc" -delete 2>/dev/null || true

# ── Step 1: RF calibration analysis ────────────────────────────────────────
banner "[Step 1] RF vs LLM calibration analysis"
echo "  목적: paper §6.5 — 왜 RandomForest 가 LLM 을 outperform 했나"
echo "  소요: ~1-2 hours (LLM 2000 calls + tabular instant)"
run_step CALIB "RF calibration analysis (ECE, Brier, sharpness)" "$SKIP_RF_CALIBRATION" -- \
    "$PYTHON" experiments/round10_rf_calibration.py \
        --n-cal 2000 --n-test 2000 --seed 42

# ── Step 2: IRB case extraction ────────────────────────────────────────────
banner "[Step 2] IRB physician audit infrastructure"
echo "  목적: 100 cases (50 CRITICAL + 50 HIGH) 추출 + physician package 생성"
echo "  소요: instant"
echo "  Next: physician 3명에게 paper/irb_audit_package/ 전달 (4 weeks external)"
run_step IRB "extract 100 cases + physician package" "$SKIP_IRB_EXTRACT" -- \
    "$PYTHON" experiments/round10_5_irb_extract.py \
        --n-critical 50 --n-high 50 --seed 42

# ── Step 3: Paper sync ─────────────────────────────────────────────────────
banner "[Step 3] Paper sync (R10 result → paper §5)"
echo "  목적: paper/UASEF_Round10.{md,KO.md} 의 §5 placeholder → 실제 수치"
echo "  소요: instant"
run_step SYNC "auto-sync paper §5 from results JSON" "$SKIP_PAPER_SYNC" -- \
    "$PYTHON" experiments/round10_paper_sync.py

banner "ML4H 2026 submission 준비 완료"
echo ""
echo "  ✅ 산출물:"
echo "    - results/round10/r10_rf_calibration.{json,md}    (paper §6.5)"
echo "    - paper/irb_audit_package/                         (physician 에게 전달)"
echo "    - data/raw/audit_r10_5/                            (internal, commit 금지)"
echo "    - paper/UASEF_Round10.md                           (R10 실제 결과 sync)"
echo "    - paper/UASEF_Round10_KO.md                        (KO mirror)"
echo ""
echo "  📋 다음 액션:"
echo "    1. paper/irb_audit_package/ 를 3 board-certified physician 에게 전달"
echo "    2. 4 weeks wait — physician 의 response JSONL 도착 후:"
echo "       .venv/bin/python experiments/round10_physician_audit.py \\"
echo "         --physician-labels data/raw/physician_audit.jsonl \\"
echo "         --outcome-labels data/raw/audit_r10_5/r10_5_audit_ground_truth.csv"
echo "    3. paper §5.5 의 κ 결과를 paper 에 추가 (manual)"
echo "    4. ML4H 2026 submission window 에 paper/UASEF_Round10.md 제출"
