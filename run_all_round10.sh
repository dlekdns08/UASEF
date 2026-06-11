#!/usr/bin/env bash
#
# UASEF Round 10 — Method-Agnostic CRC × Multi-seed × MIMIC-IV
# ═══════════════════════════════════════════════════════════════════════════════
#
# 단계 (improvements/round10_PLAN.md §3 참조):
#   [P0]    Preprocessing — Round 9 leakage-safe 위에 R10.7 확장 feature
#   [R10.0] Multi-seed infrastructure 검증
#   [R10.1] Powered α=0.05 empirical (n=3000 CRITICAL × 5 seed)
#   [R10.2] Multi-seed Table 4-MIMIC (8 method × 5 seed McNemar pooled)
#   [R10.3] Distribution shift mitigation (3 strategy)
#   [R10.4] Method-agnostic CRC head-to-head (5 classifier) ← HEADLINE
#   [R10.5] IRB physician audit (별도 4주 process — 본 스크립트는 결과 처리만)
#   [R10.6] 4-D cost matrix sweep (81 조합)
#   [R10.7] Expanded-feature 검증
#   [Agg]   통합 보고서
#
# 환경변수
# ────────
#   MIMIC4_DIR                       Round 9 와 동일
#   UASEF_BACKEND_NEVER_SEND_PHI=1   필수
#   SEEDS                            default "42 43 44 45 46"
#   BACKENDS                         default "lmstudio"
#   N_CRITICAL_R10_1                 default 3000
#   R10_4_CLASSIFIERS                default "gpt_oss_120b logreg gbdt randomforest xgboost"
#                                    (LLM 백엔드는 gpt-oss-120b 만 사용; size scaling 비교 없음)
#   UASEF_QUERY_TIMEOUT_S            default 60
#
# Skip 플래그 (1 로 설정 시 해당 단계 생략)
# ──────────────────────────────────────
#   SKIP_PREPROCESS, SKIP_R10_0, SKIP_R10_1, ..., SKIP_R10_7, SKIP_AGGREGATE
#
# 사용 예
# ──────
#   # Phase 1 (infra + preprocessing)
#   SKIP_R10_1=1 SKIP_R10_2=1 SKIP_R10_3=1 SKIP_R10_4=1 SKIP_R10_5=1 SKIP_R10_6=1 \
#     bash run_all_round10.sh
#
#   # Phase 2 (모든 실험 — LLM 포함 시 ~18일)
#   SKIP_PREPROCESS=1 SKIP_R10_0=1 SKIP_R10_5=1 bash run_all_round10.sh
#
#   # Phase 3 (aggregate)
#   SKIP_PREPROCESS=1 SKIP_R10_0=1 SKIP_R10_1=1 SKIP_R10_2=1 \
#   SKIP_R10_3=1 SKIP_R10_4=1 SKIP_R10_5=1 SKIP_R10_6=1 SKIP_R10_7=1 \
#     bash run_all_round10.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

# ── env defaults ────────────────────────────────────────────────────────────
MIMIC4_DIR="${MIMIC4_DIR:-$HOME/Downloads/mimic-iv-3.1}"
SEEDS="${SEEDS:-42 43 44 45 46}"
BACKENDS="${BACKENDS:-lmstudio}"
N_CRITICAL_R10_1="${N_CRITICAL_R10_1:-3000}"
R10_4_CLASSIFIERS="${R10_4_CLASSIFIERS:-gpt_oss_120b logreg gbdt randomforest xgboost}"
# 주의: LLM 백엔드는 openai/gpt-oss-120b 만 사용. 20B 또는 다른 사이즈 비교 없음.

SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_R10_0="${SKIP_R10_0:-0}"
SKIP_R10_1="${SKIP_R10_1:-0}"
SKIP_R10_2="${SKIP_R10_2:-0}"
SKIP_R10_3="${SKIP_R10_3:-0}"
SKIP_R10_4="${SKIP_R10_4:-0}"
SKIP_R10_5="${SKIP_R10_5:-1}"   # IRB physician — 기본 skip
SKIP_R10_6="${SKIP_R10_6:-0}"
SKIP_R10_7="${SKIP_R10_7:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

STRICT_FAIL="${STRICT_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"

export UASEF_QUERY_TIMEOUT_S="${UASEF_QUERY_TIMEOUT_S:-60}"
export UASEF_QUERY_MAX_RETRIES="${UASEF_QUERY_MAX_RETRIES:-2}"

# ── PHI guard 확인 ──────────────────────────────────────────────────────────
if [ -z "${UASEF_BACKEND_NEVER_SEND_PHI:-}" ]; then
    echo "⚠️ UASEF_BACKEND_NEVER_SEND_PHI 가 설정되어 있지 않습니다."
    echo "   Round 10 도 PhysioNet DUA 보호를 위해 PHI guard 활성화 권장."
    sleep 5
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/results/round10/all_in_one_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/run.log"
STATUS_DIR="${RUN_DIR}/_status"
mkdir -p "$STATUS_DIR"

exec > >(tee -a "$LOG") 2>&1

# ── helpers (Round 8/9 와 동일) ─────────────────────────────────────────────
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
    if [ "$1" != "--" ]; then err "[$key] missing --"; return 2; fi
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
banner "UASEF Round 10 — Method-Agnostic CRC × Multi-seed × MIMIC-IV"
note "TIMESTAMP    : $TIMESTAMP"
note "RUN_DIR      : $RUN_DIR"
note "MIMIC4_DIR   : $MIMIC4_DIR"
note "PHI guard    : ${UASEF_BACKEND_NEVER_SEND_PHI:-not set}"
note "Backends     : $BACKENDS"
note "Seeds        : $SEEDS"
note "R10.4 cls    : $R10_4_CLASSIFIERS"
note "Query TO     : ${UASEF_QUERY_TIMEOUT_S}s × ${UASEF_QUERY_MAX_RETRIES} retries"

# ── P0: Preprocessing (R10.7 features included) ─────────────────────────────
banner "[P0] R10 preprocessing — expanded decision-time features"
JSONL_V10="${ROOT}/data/raw/mimic-iv/mimic4_cases_v10.jsonl"
run_step P0 "preprocess (Charlson + vital quartile + specialty rate)" "$SKIP_PREPROCESS" -- \
    "$PYTHON" experiments/round10_mimic4_preprocess.py \
        --mimic-dir "$MIMIC4_DIR" \
        --output "$JSONL_V10" \
        --n-per-stratum 3500 --seed 42

# ── R10.0: infrastructure ──────────────────────────────────────────────────
banner "[R10.0] Multi-seed infrastructure 검증"
run_step R10.0 "pytest bootstrap CI / Clopper-Pearson helper" "$SKIP_R10_0" -- \
    "$PYTHON" -m pytest tests/test_round10_aggregate.py -q --no-header

# ── R10.1: Powered α=0.05 empirical ────────────────────────────────────────
banner "[R10.1] Powered α=0.05 empirical (n=$N_CRITICAL_R10_1 CRITICAL × 5 seed)"
run_step R10.1 "alpha_005_empirical (proper power)" "$SKIP_R10_1" -- \
    "$PYTHON" experiments/round10_alpha_005_empirical.py \
        --n-cal "$N_CRITICAL_R10_1" --n-test "$N_CRITICAL_R10_1" \
        --seeds $SEEDS --backends $BACKENDS \
        --out "${ROOT}/results/round10/r10_1_alpha_005_empirical"

# ── R10.2: multi-seed Table 4-MIMIC ────────────────────────────────────────
banner "[R10.2] Multi-seed Table 4-MIMIC (8 method × 5 seed)"
run_step R10.2 "table4_multiseed (McNemar pooled)" "$SKIP_R10_2" -- \
    "$PYTHON" experiments/round10_table4_multiseed.py \
        --n-cal-per-stratum 200 --n-test-per-stratum 100 \
        --alpha 0.10 \
        --seeds $SEEDS --backends $BACKENDS \
        --out "${ROOT}/results/round10/r10_2_table4_multiseed"

# ── R10.3: distribution shift mitigation ───────────────────────────────────
banner "[R10.3] Distribution shift mitigation (3 strategy)"
run_step R10.3 "mitigation (online_recal, KMM, group_conditional)" "$SKIP_R10_3" -- \
    "$PYTHON" experiments/round10_distshift_mitigation.py \
        --strategies online_recal kmm group_conditional \
        --seeds $SEEDS --backends $BACKENDS \
        --out "${ROOT}/results/round10/r10_3_mitigation"

# ── R10.4: method-agnostic CRC head-to-head (HEADLINE) ─────────────────────
banner "[R10.4] Method-agnostic CRC head-to-head (HEADLINE)"
run_step R10.4 "5 classifier × CRC × $(echo $SEEDS | wc -w | tr -d ' ') seed" "$SKIP_R10_4" -- \
    "$PYTHON" experiments/round10_method_agnostic.py \
        --classifiers $R10_4_CLASSIFIERS \
        --n-cal "$N_CRITICAL_R10_1" --n-test "$N_CRITICAL_R10_1" \
        --seeds $SEEDS \
        --out "${ROOT}/results/round10/r10_4_method_agnostic"

# ── R10.5: IRB physician audit (default SKIP, separate process) ────────────
banner "[R10.5] IRB physician audit (별도 4주 process)"
run_step R10.5 "physician κ + confusion matrix" "$SKIP_R10_5" -- \
    "$PYTHON" experiments/round10_physician_audit.py \
        --physician-labels "${ROOT}/data/raw/physician_audit.jsonl" \
        --outcome-labels "$JSONL_V10" \
        --out "${ROOT}/results/round10/r10_5_physician_audit"

# ── R10.6: 4-D cost matrix sweep ───────────────────────────────────────────
banner "[R10.6] 4-D cost matrix sweep (81 조합)"
run_step R10.6 "cost_sweep_4d" "$SKIP_R10_6" -- \
    "$PYTHON" experiments/round10_cost_sweep_4d.py \
        --grid 10 100 1000 \
        --seed 42 \
        --out "${ROOT}/results/round10/r10_6_cost_sweep_4d"

# ── R10.7: expanded-feature 검증 ───────────────────────────────────────────
banner "[R10.7] Expanded-feature 검증 (Round 9 vs Round 10 feature)"
run_step R10.7 "feature_expansion (MODERATE/LOW improvement)" "$SKIP_R10_7" -- \
    "$PYTHON" experiments/round10_feature_expand.py \
        --seeds $SEEDS --backends $BACKENDS \
        --out "${ROOT}/results/round10/r10_7_feature_expansion"

# ── Aggregate ──────────────────────────────────────────────────────────────
banner "[Agg] Round 10 통합 보고서"
run_step Agg "round10_aggregate_report" "$SKIP_AGGREGATE" -- \
    "$PYTHON" experiments/round10_aggregate_report.py \
        --in-dir "${ROOT}/results/round10" \
        --out "${ROOT}/results/round10/ROUND10_FINAL_REPORT.md"

# ── 요약 ────────────────────────────────────────────────────────────────────
banner "Round 10 완료"
echo "  RUN_DIR : $RUN_DIR"
echo "  보고서  : ${ROOT}/results/round10/ROUND10_FINAL_REPORT.md"
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
