#!/usr/bin/env bash
#
# UASEF Round 11 — Path 2 (Solid Proceedings) master runner
# ════════════════════════════════════════════════════════════════════════════
#
# 5 단계로 R10.4 retraction 완결 + Proceedings 트랙 자격 확보:
#
#   [R11.7]  Paper-JSON 수치 자동 검증 (~2분)
#            → results/round11/r11_7_paper_audit.{json,md}
#            → 모든 paper numeric claim 이 JSON artifact 와 일치하는지
#
#   [R11.4]  MOD/LOW failure 의 information-theoretic 규명 (~5분)
#            → results/round11/r11_4_modlow_mi.{json,md}
#            → "framework limit 가 아닌 data limit" 정식 증명
#
#   [R11.5]  LLM post-hoc calibration (Platt + isotonic) (~30분)
#            → results/round11/r11_5_llm_calibration.{json,md}
#            → "calibration alone is insufficient" verdict
#            (LLM cache 의 raw scores 없으면 --recompute 로 ~64h 재실행)
#
#   [R11.3a] eICU preprocessing (~5-30분, 다운로드 크기에 비례)
#            → data/raw/eicu_cases_v11.jsonl  (commit 금지)
#
#   [R11.3b] eICU cross-center replication, Pass A + Pass B (~30분 tabular)
#            → results/round11/r11_3_eicu_{pass_a,pass_b}.{json,md}
#            → audit discipline H1 verdict (cross-center generalizability)
#
#   [R11.2]  LLM minimal-feature re-run (background, ~64시간)
#            → results/round11/r11_1_cache/gpt_oss_120b.json
#            → R11.1 표의 LLM 행 완성
#            * LLM 64h 는 마지막 단계 — 다른 빠른 R11.x 가 먼저 완료
#
# 사용 예
# ──────
#   export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
#   export UASEF_BACKEND_NEVER_SEND_PHI=1
#   bash run_all_round11.sh                     # tabular only (~70분)
#   INCLUDE_LLM=1 bash run_all_round11.sh       # + LLM R11.2/R11.3 (~140h)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

SKIP_AUDIT="${SKIP_AUDIT:-0}"
SKIP_MI="${SKIP_MI:-0}"
SKIP_CALIB="${SKIP_CALIB:-0}"
SKIP_EICU_PREP="${SKIP_EICU_PREP:-0}"
SKIP_EICU_REP="${SKIP_EICU_REP:-0}"
SKIP_LLM_MINIMAL="${SKIP_LLM_MINIMAL:-1}"  # default skip (64h)
INCLUDE_LLM="${INCLUDE_LLM:-0}"  # eICU 의 LLM pass 도 포함 여부
DRY_RUN="${DRY_RUN:-0}"

# stale .pyc cache 방지 (2026-06-23 사례)
find "${ROOT}/experiments/__pycache__" -name "round1*.pyc" -delete 2>/dev/null || true

export UASEF_BACKEND_NEVER_SEND_PHI="${UASEF_BACKEND_NEVER_SEND_PHI:-1}"
export UASEF_QUERY_TIMEOUT_S="${UASEF_QUERY_TIMEOUT_S:-60}"

C_BOLD=$'\033[1m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_RST=$'\033[0m'
banner() { echo ""; echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
           echo "${C_BOLD}  $1${C_RST}"
           echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"; }
ok()   { echo "  ${C_OK}✓${C_RST} $1"; }
warn() { echo "  ${C_WARN}⚠${C_RST} $1"; }
err()  { echo "  ${C_ERR}✗${C_RST} $1"; }

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/results/round11/all_in_one_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/run.log"
exec > >(tee -a "$LOG") 2>&1

run_step() {
    local key="$1"; shift
    local desc="$1"; shift
    local skip="$1"; shift
    [ "$1" != "--" ] && { err "[$key] missing --"; return 2; }; shift
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

banner "UASEF Round 11 — Path 2 (Solid Proceedings)"
echo "  TIMESTAMP    : $TIMESTAMP"
echo "  RUN_DIR      : $RUN_DIR"
echo "  PYTHON       : $PYTHON"
echo "  PHI guard    : ${UASEF_BACKEND_NEVER_SEND_PHI}"
echo "  INCLUDE_LLM  : $INCLUDE_LLM"
echo ""

# ── R11.7 — Paper-JSON audit ───────────────────────────────────────────────
banner "[R11.7] Paper-JSON numerical consistency audit"
echo "  목적: paper 의 수치 claim 이 results/*.json 과 일치하는지 검증"
run_step R11_7 "paper-JSON audit" "$SKIP_AUDIT" -- \
    "$PYTHON" experiments/round11_paper_audit.py

# ── R11.4 — MOD/LOW information-theoretic ──────────────────────────────────
banner "[R11.4] MOD/LOW failure as data limit"
echo "  목적: §5.7/5.8 의 100% miss 가 data limit 임을 mutual information 으로 증명"
run_step R11_4 "MOD/LOW MI analysis" "$SKIP_MI" -- \
    "$PYTHON" experiments/round11_modlow_mi.py \
        --jsonl "${ROOT}/data/raw/mimic-iv/mimic4_cases_v10.jsonl"

# ── R11.5 — LLM post-hoc calibration ───────────────────────────────────────
banner "[R11.5] LLM post-hoc calibration (Platt + isotonic)"
echo "  목적: LLM gate 가 calibration 으로 임상 deployable 한지 정량화"
echo "  주의: LLM cache 에 raw scores 없으면 자동 skip — --recompute 로 LLM 재실행 가능"
run_step R11_5 "LLM calibration" "$SKIP_CALIB" -- \
    "$PYTHON" experiments/round11_llm_calibration.py

# ── R11.3a — eICU preprocessing ────────────────────────────────────────────
banner "[R11.3a] eICU-CRD preprocessing"
echo "  목적: eICU 5 csv → JSONL (decision-time features + outcome)"
run_step R11_3A "eICU preprocess" "$SKIP_EICU_PREP" -- \
    "$PYTHON" experiments/round11_eicu_preprocess.py \
        --eicu-dir "${ROOT}/data/raw/eicu-crd" \
        --out "${ROOT}/data/raw/eicu_cases_v11.jsonl"

# ── R11.3b — eICU cross-center replication ─────────────────────────────────
banner "[R11.3b] eICU cross-center replication (Pass A + Pass B)"
echo "  목적: H1 — audit discipline 이 다른 코호트에서도 leakage 를 잡는다"
EICU_EXTRA=""
[ "$INCLUDE_LLM" = "1" ] && EICU_EXTRA="--include-llm"
run_step R11_3B "eICU replication" "$SKIP_EICU_REP" -- \
    "$PYTHON" experiments/round11_eicu_replication.py \
        --jsonl "${ROOT}/data/raw/eicu_cases_v11.jsonl" \
        $EICU_EXTRA

# ── R11.2 — LLM minimal-feature re-run (optional, 64h) ─────────────────────
banner "[R11.2] LLM minimal-feature re-run (~64h)"
echo "  목적: R11.1 표의 LLM 행 완성"
echo "  주의: 64h wallclock — SKIP_LLM_MINIMAL=0 로 활성화"
run_step R11_2 "LLM minimal re-run" "$SKIP_LLM_MINIMAL" -- \
    "$PYTHON" experiments/round11_method_agnostic_minimal.py \
        --classifiers gpt_oss_120b \
        --seeds 42 43 44 45 46 \
        --n-cal 3000 --n-test 3000 \
        --out "${ROOT}/results/round11/r11_1_method_agnostic_minimal"

banner "Round 11 Path 2 완료"
echo ""
echo "  📄 산출물:"
echo "    - results/round11/r11_7_paper_audit.{json,md}      (paper consistency)"
echo "    - results/round11/r11_4_modlow_mi.{json,md}         (MOD/LOW data limit verdict)"
echo "    - results/round11/r11_5_llm_calibration.{json,md}   (LLM calibration verdict)"
echo "    - data/raw/eicu_cases_v11.jsonl                     (eICU JSONL, commit 금지)"
echo "    - results/round11/r11_3_eicu_{pass_a,pass_b}.{json,md}  (cross-center)"
echo "    - results/round11/r11_3_eicu_verdict.json            (H1 verdict)"
echo ""
echo "  📋 다음 액션:"
echo "    1. r11_7_paper_audit.md 확인 — paper 의 stale 수치 수정"
echo "    2. r11_4_modlow_mi.md 의 verdict 확인 → paper §5.7.1 작성"
echo "    3. r11_5_llm_calibration.md → paper §6.6 작성"
echo "    4. r11_3_eicu_verdict.json → paper §5.10 작성"
echo "    5. (선택) SKIP_LLM_MINIMAL=0 bash run_all_round11.sh — R11.2 64h 실행"
