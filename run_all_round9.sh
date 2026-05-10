#!/usr/bin/env bash
#
# UASEF Round 9 — MIMIC-IV integration master runner
# ════════════════════════════════════════════════════════════════════════════
#
# 단계 (improvements/round9_PLAN.md §4 참조):
#   [P0]   Preprocessing: ~/Downloads/mimic-iv-3.1/{hosp,icu}/*.csv.gz → JSONL
#   [R9.1] α_CRITICAL = 0.001 empirical validation         (Table 1c)
#   [R9.2] Table 4-MIMIC head-to-head (8 method × 5 seed)
#   [R9.3] Real-EHR distribution shift (services-table)
#   [R9.4] Temporal shift (2008-14 cal vs 2015-19 test)
#   [R9.5] Demographic equity audit
#   [Agg]  통합 보고서
#
# 환경변수
# ────────
#   MIMIC4_DIR                  $HOME/Downloads/mimic-iv-3.1 등 — preprocessing 입력
#   UASEF_BACKEND_NEVER_SEND_PHI=1   필수: external API 송신 차단
#   N_PER_STRATUM   default 1500   preprocessing 샘플 수 (CRITICAL/HIGH/MOD/LOW × N)
#   N_CAL_PER       default 200    Table 4-MIMIC calibration per stratum
#   N_TEST_PER      default 100    Table 4-MIMIC test per stratum
#   ALPHA           default 0.10
#   ALPHA_CRITICAL  default 0.001  R9.1 헤드라인
#   SEEDS           default "42 43 44 45 46"
#   BACKENDS        default "openai lmstudio"
#
# Skip 플래그 (1로 설정 시 해당 단계 생략)
# ──────────────────────────────────────
#   SKIP_PREPROCESS, SKIP_R9_1, SKIP_R9_2, SKIP_R9_3, SKIP_R9_4, SKIP_R9_5,
#   SKIP_AGGREGATE
#
# 사용 예
# ──────
#   export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
#   export UASEF_BACKEND_NEVER_SEND_PHI=1
#   bash run_all_round9.sh                     # 전체 — ~$80, ~5h
#
#   SKIP_PREPROCESS=1 bash run_all_round9.sh   # JSONL 이미 있으면
#   SKIP_R9_1=1 SKIP_R9_2=1 bash run_all_round9.sh  # subset
#   STRICT_FAIL=1 DRY_RUN=1 bash run_all_round9.sh  # what-if

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

# ── env defaults ────────────────────────────────────────────────────────────
MIMIC4_DIR="${MIMIC4_DIR:-$HOME/Downloads/mimic-iv-3.1}"
N_PER_STRATUM="${N_PER_STRATUM:-1500}"
N_CAL_PER="${N_CAL_PER:-200}"
N_TEST_PER="${N_TEST_PER:-100}"
ALPHA="${ALPHA:-0.10}"
ALPHA_CRITICAL="${ALPHA_CRITICAL:-0.001}"
SEEDS="${SEEDS:-42 43 44 45 46}"
BACKENDS="${BACKENDS:-openai lmstudio}"

SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_R9_1="${SKIP_R9_1:-0}"
SKIP_R9_2="${SKIP_R9_2:-0}"
SKIP_R9_3="${SKIP_R9_3:-0}"
SKIP_R9_4="${SKIP_R9_4:-0}"
SKIP_R9_5="${SKIP_R9_5:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

STRICT_FAIL="${STRICT_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ── PHI guard 강제 ──────────────────────────────────────────────────────────
if [ -z "${UASEF_BACKEND_NEVER_SEND_PHI:-}" ]; then
    echo "⚠️ UASEF_BACKEND_NEVER_SEND_PHI 가 설정되어 있지 않습니다."
    echo "   Round 9 는 PhysioNet DUA 보호를 위해 PHI guard 활성화를 권장합니다."
    echo "   계속하시려면 5초 후 자동 진행. 중단: Ctrl+C."
    sleep 5
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/results/round9/all_in_one_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/run.log"
STATUS_DIR="${RUN_DIR}/_status"
mkdir -p "$STATUS_DIR"

exec > >(tee -a "$LOG") 2>&1

# ── helpers ─────────────────────────────────────────────────────────────────
C_BOLD=$'\033[1m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_RST=$'\033[0m'
banner() { echo ""; echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
           echo "${C_BOLD}  $1${C_RST}"
           echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"; }
note()   { echo "  $1"; }
ok()     { echo "  ${C_OK}✓${C_RST} $1"; }
warn()   { echo "  ${C_WARN}⚠${C_RST} $1"; }
err()    { echo "  ${C_ERR}✗${C_RST} $1"; }

set_status()   { echo "$2" > "$STATUS_DIR/$1.status"; }
set_duration() { echo "$2" > "$STATUS_DIR/$1.duration"; }

run_step() {
    local key="$1"; shift
    local desc="$1"; shift
    local skip="$1"; shift
    if [ "$1" != "--" ]; then err "[$key] internal: missing -- separator"; return 2; fi
    shift
    if [ "$skip" = "1" ]; then
        set_status "$key" "skip"; set_duration "$key" "0"
        warn "[$key] SKIP — $desc"; return 0
    fi
    note "[$key] ▶ $desc"
    if [ "$DRY_RUN" = "1" ]; then echo "    DRY: $*"; set_status "$key" "dry"; set_duration "$key" "0"; return 0; fi
    local t0; t0=$(date +%s); local rc=0
    "$@" || rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then ok "[$key] done in ${elapsed}s"; set_status "$key" "ok"
    else err "[$key] FAILED rc=$rc after ${elapsed}s"; set_status "$key" "fail($rc)"
        if [ "$STRICT_FAIL" = "1" ]; then exit $rc; fi
    fi
    set_duration "$key" "$elapsed"
    return 0
}

# ── banner ──────────────────────────────────────────────────────────────────
banner "UASEF Round 9 — MIMIC-IV integration"
note "TIMESTAMP    : $TIMESTAMP"
note "RUN_DIR      : $RUN_DIR"
note "MIMIC4_DIR   : $MIMIC4_DIR"
note "PHI guard    : ${UASEF_BACKEND_NEVER_SEND_PHI:-not set}"
note "N_PER_STRATUM: $N_PER_STRATUM"
note "Backends     : $BACKENDS"
note "Seeds        : $SEEDS"

# ── P0: preprocessing ──────────────────────────────────────────────────────
banner "[P0] Preprocessing — MIMIC-IV hosp+icu → JSONL"
JSONL="${ROOT}/data/raw/mimic-iv/mimic4_cases.jsonl"
if [ -f "$JSONL" ] && [ "$SKIP_PREPROCESS" = "0" ]; then
    warn "JSONL 이미 존재: $JSONL — skip 하려면 SKIP_PREPROCESS=1"
fi
run_step P0 "preprocessing → $JSONL" "$SKIP_PREPROCESS" -- \
    "$PYTHON" experiments/round9_mimic4_preprocess.py \
        --mimic-dir "$MIMIC4_DIR" \
        --output "$JSONL" \
        --n-per-stratum "$N_PER_STRATUM" --seed 42

# ── R9.1: α=0.001 empirical ────────────────────────────────────────────────
banner "[R9.1] α_CRITICAL=$ALPHA_CRITICAL empirical (n=$N_PER_STRATUM × 5 seeds)"
run_step R9.1 "alpha_critical_real (Table 1c)" "$SKIP_R9_1" -- \
    "$PYTHON" experiments/round9_alpha_critical_real.py \
        --n-critical "$N_PER_STRATUM" \
        --alpha-critical "$ALPHA_CRITICAL" \
        --seeds $SEEDS \
        --backends $BACKENDS \
        --out "${ROOT}/results/round9/alpha_critical_real"

# ── R9.2: Table 4-MIMIC ────────────────────────────────────────────────────
banner "[R9.2] Table 4-MIMIC head-to-head"
run_step R9.2 "table4_mimic (8 method × $(echo $BACKENDS | wc -w | tr -d ' ') backend × $(echo $SEEDS | wc -w | tr -d ' ') seed)" "$SKIP_R9_2" -- \
    "$PYTHON" experiments/round9_table4_mimic.py \
        --n-cal-per-stratum "$N_CAL_PER" \
        --n-test-per-stratum "$N_TEST_PER" \
        --alpha "$ALPHA" \
        --seeds $SEEDS \
        --backends $BACKENDS \
        --out "${ROOT}/results/round9/table4_mimic"

# ── R9.3: distribution shift ───────────────────────────────────────────────
banner "[R9.3] Real-EHR distribution shift (cardiology → others, weighted CP)"
run_step R9.3 "dist_shift_real" "$SKIP_R9_3" -- \
    "$PYTHON" experiments/round9_distribution_shift.py \
        --source cardiology \
        --targets neurology internal_medicine surgery \
        --n-per-spec 300 \
        --seeds $SEEDS \
        --backends $BACKENDS \
        --out "${ROOT}/results/round9/dist_shift_real"

# ── R9.4: temporal shift ───────────────────────────────────────────────────
banner "[R9.4] Temporal shift (2008-14 → 2015-19)"
run_step R9.4 "temporal_shift" "$SKIP_R9_4" -- \
    "$PYTHON" experiments/round9_temporal_shift.py \
        --cal-years 2008 2014 \
        --test-years 2015 2019 \
        --n-cal 600 --n-test 300 \
        --seeds $SEEDS \
        --backends $BACKENDS \
        --out "${ROOT}/results/round9/temporal_shift"

# ── R9.5: equity audit ─────────────────────────────────────────────────────
banner "[R9.5] Demographic equity audit"
run_step R9.5 "equity_audit_real (single seed)" "$SKIP_R9_5" -- \
    "$PYTHON" experiments/round9_equity_real.py \
        --n-cal 800 --n-test 600 \
        --seed 42 --backend openai \
        --out "${ROOT}/results/round9/equity_audit_real"

# ── Aggregate ──────────────────────────────────────────────────────────────
banner "[Agg] Aggregate Round 9 보고서"
run_step Agg "round9_report.md" "$SKIP_AGGREGATE" -- \
    "$PYTHON" experiments/round9_aggregate_report.py \
        --in-dir "${ROOT}/results/round9" \
        --out "${ROOT}/results/round9/round9_report.md"

# ── 요약 ────────────────────────────────────────────────────────────────────
banner "Round 9 완료"
echo "  RUN_DIR : $RUN_DIR"
echo "  보고서  : ${ROOT}/results/round9/round9_report.md"
echo ""
echo "  Step 상태:"
shopt -s nullglob
for f in "$STATUS_DIR"/*.status; do
    k=$(basename "$f" .status)
    s=$(cat "$f")
    d=$(cat "${f%.status}.duration" 2>/dev/null || echo "?")
    printf "    %-8s  %-12s  %ss\n" "$k" "$s" "$d"
done
shopt -u nullglob
