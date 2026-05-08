#!/usr/bin/env bash
# shellcheck shell=bash
#
# IMPORTANT: This script requires bash in non-POSIX mode. It uses process
# substitution `>(...)`, $'...' ANSI escapes, and `local` — none POSIX.
# Two failure modes are handled:
#   (a) invoked under a different shell (dash, zsh, ksh) → BASH_VERSION unset
#   (b) invoked as `sh run_all_round8.sh` on macOS, where /bin/sh is bash 3.2
#       in POSIX mode — BASH_VERSION IS set but process substitution is
#       parse-rejected. Detect via `set -o` and re-exec.
if [ -z "${BASH_VERSION:-}" ]; then
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    fi
    echo "[error] this script requires bash. Run as: bash run_all_round8.sh" >&2
    exit 1
fi
# bash but in POSIX mode (e.g., /bin/sh on macOS) → re-exec out of POSIX mode.
if set -o 2>/dev/null | awk '/^posix/ {exit ($NF == "on") ? 0 : 1}'; then
    exec bash "$0" "$@"
fi

# UASEF — Round 8 ALL-IN-ONE Master Runner
# ════════════════════════════════════════════════════════════════════════════
#
# 한 번 호출로 다음을 모두 실행한다:
#
#   [Pre]   환경 점검 + MedAbstain raw JSONL 검증
#   [P0]    pytest 회귀 안전망 (140 tests)
#   [P0.5]  Calibration pipeline (run_calibration_pipeline.py) — base_config.yaml 자동 보정
#   [P1.1]  Single-seed full evaluation (run_full_evaluation.sh, seed=42)
#           = run_all_experiments.py (agent + baseline + medabstain + pareto)
#             + Table 2/3/α-critical (synthetic) + Table 1/4 (LLM)
#   [P1.2]  5-seed bootstrap (run_multiseed_evaluation.sh) — Table 1/4 95% CI
#   [P1.3]  LLM-judge self-consistency relabeling (gpt-5.5 + claude-opus-4-7)
#   [P1.4]  Standalone primary+ablation logprob CP (run_experiment.py)
#   [P2.1]  Multi-dataset generalization (medabstain + medqa_usmle + pubmedqa)
#   [P2.2]  Pivot B case study (m ∈ {3,5,8,12} FWER + 기관 customization)
#   [P3.1]  Distribution-shift sanity (specialty-mismatch coverage 위반)
#   [P3.2]  Multi-lingual sanity (zh, data 있을 때만)
#   [P3.3]  Per-stratum equity audit (single-seed run의 Table 4 기반)
#   [P3.4]  Paper-claim regression test (LLM snapshot 활성화 후 9/9)
#   [P4.1]  Figure generation (visualize_results.py) — Pareto + comparison bar
#   [P4.2]  Cross-tag run comparison (compare_runs.py) — 사용자가 --run-tag로 분리한 ablation 비교 (optional)
#   [Post]  통합 리포트 results/round8/all_in_one_<ts>/ALL_IN_ONE_REPORT.md
#
# 사용:
#   bash run_all_round8.sh                                # 전체 (~$190 + ~5h)
#   SKIP_MULTISEED=1 bash run_all_round8.sh               # single-seed only
#   SKIP_LLM_JUDGE=1 SKIP_MULTIDATASET=1 bash …           # synthetic only
#   DRY_RUN=1 bash run_all_round8.sh                      # what-if (no execution)
#
# 환경변수:
#   BACKENDS              default "openai lmstudio"
#   SEEDS_FOR_MULTISEED   default "42 43 44 45 46"
#   N_CAL                 default 200 (논문 quality는 500)
#   N_TEST                default 100
#   ALPHA                 default 0.10
#   JUDGE_OPENAI_MODEL    default gpt-5.5
#   JUDGE_ANTHROPIC_MODEL default claude-opus-4-7
#   N_LLM_JUDGE           default 200
#
#   SKIP_*                각 step skip (1 = skip)
#   STRICT_FAIL           1 = 어느 step이든 fail하면 즉시 종료 (default 0)
#   PYTHON                default ./.venv/bin/python
#
# 비용 추산 (default):
#   P1.1 (1 seed × 2 backend):    ~$25
#   P1.2 (5 seeds × 2 backend):   ~$125 + ~50min  ← 가장 비쌈
#   P1.3 (LLM-judge × 2 model):   ~$25
#   P2.1 (3 datasets, openai):    ~$30
#   P2.2 / P3.x:                  $0 (synthetic)
#   ─────────────────────────────────
#   Total:                        ~$205 + ~3h

set -uo pipefail   # 의도적으로 -e 미사용 — step별 continue-on-fail.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── 기본 설정 ────────────────────────────────────────────────────────────────
BACKENDS="${BACKENDS:-openai lmstudio}"
SEEDS_FOR_MULTISEED="${SEEDS_FOR_MULTISEED:-42 43 44 45 46}"
N_CAL="${N_CAL:-200}"
N_TEST="${N_TEST:-100}"
ALPHA="${ALPHA:-0.10}"
JUDGE_OPENAI_MODEL="${JUDGE_OPENAI_MODEL:-gpt-5.5}"
JUDGE_ANTHROPIC_MODEL="${JUDGE_ANTHROPIC_MODEL:-claude-opus-4-7}"
N_LLM_JUDGE="${N_LLM_JUDGE:-200}"

SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_CALIBRATION="${SKIP_CALIBRATION:-0}"   # P0.5
SKIP_SINGLE_SEED="${SKIP_SINGLE_SEED:-0}"
SKIP_MULTISEED="${SKIP_MULTISEED:-0}"
SKIP_LLM_JUDGE="${SKIP_LLM_JUDGE:-0}"
SKIP_RUN_EXPERIMENT="${SKIP_RUN_EXPERIMENT:-0}"   # P1.4 — produces results/experiment_results.json (input for P4.1)
SKIP_MULTIDATASET="${SKIP_MULTIDATASET:-0}"
SKIP_PIVOTB="${SKIP_PIVOTB:-0}"
SKIP_DIST_SHIFT="${SKIP_DIST_SHIFT:-0}"
SKIP_MULTILINGUAL="${SKIP_MULTILINGUAL:-0}"
SKIP_EQUITY="${SKIP_EQUITY:-0}"
SKIP_PAPER_CLAIM="${SKIP_PAPER_CLAIM:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"            # P4.1
SKIP_COMPARE_RUNS="${SKIP_COMPARE_RUNS:-1}"  # P4.2 default skip — needs --run-tag setup
COMPARE_TAGS="${COMPARE_TAGS:-}"             # space-separated tags for P4.2

STRICT_FAIL="${STRICT_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/results/round8/all_in_one_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/run.log"

# Step status는 file system으로 관리 (bash 3.2 호환).
STATUS_DIR="${RUN_DIR}/_status"
mkdir -p "$STATUS_DIR"

# tee everything to the log file.
exec > >(tee -a "$LOG") 2>&1

# ── 색상 + 헬퍼 ──────────────────────────────────────────────────────────────
C_BOLD=$'\033[1m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_RST=$'\033[0m'
banner() {
    echo ""
    echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
    echo "${C_BOLD}  $1${C_RST}"
    echo "${C_BOLD}════════════════════════════════════════════════════════════════════${C_RST}"
}
note() { echo "  $1"; }
ok()   { echo "  ${C_OK}✓${C_RST} $1"; }
warn() { echo "  ${C_WARN}⚠${C_RST} $1"; }
err()  { echo "  ${C_ERR}✗${C_RST} $1"; }

set_status()   { echo "$2" > "$STATUS_DIR/$1.status"; }
set_duration() { echo "$2" > "$STATUS_DIR/$1.duration"; }
get_status()   { [ -f "$STATUS_DIR/$1.status" ] && cat "$STATUS_DIR/$1.status" || echo "not-run"; }
get_duration() { [ -f "$STATUS_DIR/$1.duration" ] && cat "$STATUS_DIR/$1.duration" || echo "—"; }

# ── Step runner: continue-on-fail unless STRICT_FAIL=1 ─────────────────────
# Usage: run_step KEY DESC SKIP_FLAG_VALUE -- CMD ...
run_step() {
    local key="$1"; shift
    local desc="$1"; shift
    local skip="$1"; shift
    if [ "$1" != "--" ]; then
        err "[$key] internal: missing -- separator"
        return 2
    fi
    shift  # eat --

    if [ "$skip" = "1" ]; then
        set_status "$key" "skip"
        set_duration "$key" "0"
        warn "[$key] SKIP — $desc"
        return 0
    fi
    note "[$key] ▶ $desc"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY: $*"
        set_status "$key" "dry"
        set_duration "$key" "0"
        return 0
    fi
    local t0; t0=$(date +%s)
    local rc=0
    "$@" || rc=$?
    local t1; t1=$(date +%s)
    set_duration "$key" "$((t1 - t0))"
    if [ "$rc" -eq 0 ]; then
        set_status "$key" "ok"
        ok "[$key] done in $((t1 - t0))s"
    else
        set_status "$key" "fail($rc)"
        err "[$key] FAILED with rc=$rc after $((t1 - t0))s"
        if [ "$STRICT_FAIL" = "1" ]; then
            err "STRICT_FAIL=1 — aborting"
            exit "$rc"
        fi
    fi
    return 0
}

# ── PRE-FLIGHT ──────────────────────────────────────────────────────────────
banner "[Pre] Environment + Data sanity"
note "TIMESTAMP : $TIMESTAMP"
note "RUN_DIR   : $RUN_DIR"
note "PYTHON    : $PYTHON"
note "BACKENDS  : $BACKENDS"
note "SEEDS     : $SEEDS_FOR_MULTISEED  (used by P1.2)"
note "N_CAL/N_TEST : $N_CAL / $N_TEST  (α=$ALPHA)"

if [ "$DRY_RUN" = "1" ]; then
    warn "DRY_RUN=1 — commands will be printed but NOT executed."
fi

if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    err "Python ≥ 3.10 required (got $($PYTHON --version 2>&1))"
    exit 1
fi
ok "python $($PYTHON --version 2>&1)"

# .env 로드: 라인에 # 코멘트가 포함되어 있을 수 있어 set -a 방식 대신 한 줄씩 export.
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"   # strip inline comments
        case "$line" in
            ''|export*) continue ;;
            *=*) export "${line%%=*}=${line#*=}" 2>/dev/null || true ;;
        esac
    done < .env
    ok ".env loaded"
fi

# OpenAI / Anthropic key sanity
if [ "$SKIP_LLM_JUDGE" != "1" ] || [ "$SKIP_SINGLE_SEED" != "1" ] || [ "$SKIP_MULTISEED" != "1" ] || [ "$SKIP_MULTIDATASET" != "1" ]; then
    if [ -z "${OPENAI_API_KEY:-}" ] || [ "${OPENAI_API_KEY:-sk-your-key-here}" = "sk-your-key-here" ]; then
        warn "OPENAI_API_KEY not set or placeholder — OpenAI-dependent steps will fail"
    else
        ok "OPENAI_API_KEY present"
    fi
fi
if [ "$SKIP_LLM_JUDGE" != "1" ]; then
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        warn "ANTHROPIC_API_KEY not set — LLM-judge will fail (P1.3)"
    else
        ok "ANTHROPIC_API_KEY present"
    fi
fi

# MedAbstain raw data check
mkdir -p "$ROOT/data/raw"
MISSING_VARIANTS=""
for V in AP NAP A NA; do
    [ -f "$ROOT/data/raw/medabstain_${V}.jsonl" ] || MISSING_VARIANTS="$MISSING_VARIANTS $V"
done
if [ -n "$MISSING_VARIANTS" ]; then
    warn "MedAbstain variants missing:$MISSING_VARIANTS"
    warn "Run: bash data/download_datasets.sh    (manual step required for MedAbstain)"
    warn "Continuing — fallback will be used by data/loader (CP guarantee invalid)."
    warn "For paper reproduction set UASEF_PAPER_REPRODUCTION=1 to force-fail instead."
else
    ok "MedAbstain raw JSONL: 4/4 present"
fi

PRIMARY_BACKEND="$(echo "$BACKENDS" | awk '{print $1}')"

# ── Step list (for final report) ────────────────────────────────────────────
ALL_STEPS="P0 P0.5 P1.1 P1.2 P1.3 P1.4 P2.1 P2.2 P3.1 P3.2 P3.3 P3.4 P4.1 P4.2"

# ── P0: pytest ──────────────────────────────────────────────────────────────
banner "[P0] pytest regression safety net"
run_step P0 "pytest tests/ -q (140 tests)" "$SKIP_TESTS" -- \
    "$PYTHON" -m pytest tests/ -q --tb=short

# ── P0.5: Calibration pipeline ──────────────────────────────────────────────
# RTC multipliers / entropy threshold / EDE 계수를 데이터에서 역산 → base_config.yaml 갱신.
# v2(Pivot A/B/C)는 base_config의 v1 fallback 값으로도 동작하지만, v1 ablation 측정의
# 정합성을 위해 calibration pipeline을 한 번 돌려놓는다. 결과는 base_config.yaml 인플레이스 갱신.
banner "[P0.5] Calibration pipeline (RTC multipliers / entropy / EDE)"
run_step P0.5 "$PRIMARY_BACKEND backend, n-cal=$N_CAL, n-labeled=50" "$SKIP_CALIBRATION" -- \
    "$PYTHON" experiments/run_calibration_pipeline.py \
        --backend "$PRIMARY_BACKEND" \
        --n-cal "$N_CAL" --n-labeled 50 --seed 42

# ── P1.1: Single-seed full evaluation ──────────────────────────────────────
# 내부적으로 다음을 모두 호출:
#   - experiments/run_all_experiments.py
#       └─ run_agent_experiment.py     (LangGraph ReAct agent)
#       └─ run_baseline_comparison.py  (BertScore / Self-Consistency / Hybrid 등)
#       └─ eval_medabstain.py          (MedAbstain CP eval)
#       └─ pareto_sweep.py             (Coverage ↔ Escalation Pareto)
#   - experiments/round7_table2_fwer.py
#   - experiments/round7_table3_cost.py (1D + 4D sweep)
#   - experiments/round7_alpha_critical_validation.py
#   - experiments/round7_table1_coverage.py
#   - experiments/round7_table4_baseline.py
banner "[P1.1] Single-seed full evaluation"
run_step P1.1 "Tables 1·2·3·4 single seed=42 × $BACKENDS" "$SKIP_SINGLE_SEED" -- \
    env BACKENDS="$BACKENDS" N_CAL="$N_CAL" N_TEST="$N_TEST" ALPHA="$ALPHA" SEED=42 \
    bash "$ROOT/run_full_evaluation.sh"

# Snapshot the latest run directory for downstream equity audit.
LATEST_RUN=""
if [ "$(get_status P1.1)" = "ok" ]; then
    LATEST_RUN="$(ls -td "$ROOT"/results/run_2*/ 2>/dev/null | head -n1 | sed 's:/$::')"
    echo "$LATEST_RUN" > "$RUN_DIR/latest_single_seed_run.txt"
    note "single-seed run dir: $LATEST_RUN"
fi

# ── P1.2: 5-seed bootstrap ────────────────────────────────────────────────
banner "[P1.2] 5-seed bootstrap"
run_step P1.2 "5 seeds × $BACKENDS → bootstrap CIs" "$SKIP_MULTISEED" -- \
    env BACKENDS="$BACKENDS" SEEDS="$SEEDS_FOR_MULTISEED" N_CAL="$N_CAL" N_TEST="$N_TEST" ALPHA="$ALPHA" \
    bash "$ROOT/run_multiseed_evaluation.sh"

AGG_DIR=""
if [ "$(get_status P1.2)" = "ok" ]; then
    AGG_DIR="$(ls -td "$ROOT"/results/run_*_aggregate/ 2>/dev/null | head -n1 | sed 's:/$::')"
    echo "$AGG_DIR" > "$RUN_DIR/latest_aggregate_dir.txt"
    note "aggregate dir: $AGG_DIR"
fi

# ── P1.3: LLM-judge self-consistency ───────────────────────────────────────
banner "[P1.3] LLM-judge self-consistency relabeling (gpt-5.5 + claude-opus-4-7)"
run_step P1.3 "Cohen's κ over n=$N_LLM_JUDGE cases" "$SKIP_LLM_JUDGE" -- \
    "$PYTHON" experiments/llm_judge_relabel.py \
        --n "$N_LLM_JUDGE" --seed 42 \
        --judges openai anthropic \
        --openai-model "$JUDGE_OPENAI_MODEL" \
        --anthropic-model "$JUDGE_ANTHROPIC_MODEL" \
        --out "$RUN_DIR/llm_judge_relabel.json"

# ── P1.4: Standalone primary+ablation logprob CP ───────────────────────────
# P1.1의 run_full_evaluation.sh와 별개로 results/experiment_results.json을 생성한다
# (P4.1 visualize_results.py의 입력). primary(OpenAI logprob) + ablation(LMStudio
# logprob) 양쪽을 동시에 돌리는 단일 entry-point — paper §5 그림용.
# Note: run_experiment.py는 --alpha 미지원 (base_config.yaml의 uqm.alpha 사용).
banner "[P1.4] Standalone primary+ablation logprob CP"
run_step P1.4 "experiments/run_experiment.py primary + ablation" "$SKIP_RUN_EXPERIMENT" -- \
    "$PYTHON" experiments/run_experiment.py \
        --n-cal "$N_CAL" --n-test "$N_TEST"

# ── P2.1: Multi-dataset generalization ─────────────────────────────────────
banner "[P2.1] Multi-dataset generalization"
run_step P2.1 "Tables 1·4 across 3 datasets ($PRIMARY_BACKEND)" "$SKIP_MULTIDATASET" -- \
    "$PYTHON" experiments/run_multidataset_generalization.py \
        --backend "$PRIMARY_BACKEND" \
        --datasets medabstain medqa_usmle pubmedqa \
        --n-cal "$N_CAL" --n-test "$N_TEST" --alpha "$ALPHA" --seed 42 \
        --out "$RUN_DIR/multidataset_summary.json" \
        --python "$PYTHON"

# ── P2.2: Pivot B case study ───────────────────────────────────────────────
banner "[P2.2] Pivot B case study"
run_step P2.2 "Variable-m FWER + m=8 institutional cost" "$SKIP_PIVOTB" -- \
    "$PYTHON" experiments/round8_pivotB_case_study.py \
        --n-trials 5000 --n-cal 200 --alpha 0.05 --seed 42 \
        --m-values 3 5 8 12 \
        --out "$RUN_DIR/pivotB_case_study.json"

# ── P3.1: Distribution-shift sanity ────────────────────────────────────────
banner "[P3.1] Distribution-shift sanity"
run_step P3.1 "specialty-mismatch coverage violation" "$SKIP_DIST_SHIFT" -- \
    "$PYTHON" experiments/round8_distribution_shift.py \
        --calib-specialty emergency_medicine \
        --n-cal 500 --n-test 200 --seed 42 \
        --out "$RUN_DIR/distribution_shift.json"

# ── P3.2: Multi-lingual sanity ─────────────────────────────────────────────
banner "[P3.2] Multi-lingual sanity (zh)"
run_step P3.2 "zh sanity — graceful skip if data missing" "$SKIP_MULTILINGUAL" -- \
    "$PYTHON" experiments/round8_multilingual_sanity.py \
        --backend "$PRIMARY_BACKEND" \
        --n-cal 100 --n-test 50 --seed 42 \
        --out "$RUN_DIR/multilingual_sanity.json"

# ── P3.3: Equity audit (per backend, from single-seed run) ──────────────────
banner "[P3.3] Per-stratum equity audit"
if [ "$SKIP_EQUITY" = "1" ]; then
    run_step P3.3 "skipped" "1" -- :
elif [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN" ]; then
    rc_total=0
    for B in $BACKENDS; do
        T4="${LATEST_RUN}/${B}/table4_baseline.json"
        if [ -f "$T4" ]; then
            note "[P3.3.$B] auditing $T4"
            if [ "$DRY_RUN" = "1" ]; then
                echo "    DRY: $PYTHON experiments/round8_equity_audit.py --table4 $T4 --out $RUN_DIR/equity_audit_${B}.json"
            else
                "$PYTHON" experiments/round8_equity_audit.py \
                    --table4 "$T4" \
                    --out "$RUN_DIR/equity_audit_${B}.json" \
                    || rc_total=$?
            fi
        else
            warn "table4_baseline.json missing for $B at $T4"
        fi
    done
    if [ "$rc_total" -eq 0 ]; then
        set_status P3.3 "ok"
        ok "[P3.3] equity audit done"
    else
        set_status P3.3 "fail($rc_total)"
        err "[P3.3] partial fail rc=$rc_total"
        if [ "$STRICT_FAIL" = "1" ]; then exit "$rc_total"; fi
    fi
    set_duration P3.3 "—"
else
    set_status P3.3 "skip"
    set_duration P3.3 "0"
    warn "[P3.3] SKIP — needs P1.1 single-seed run output"
fi

# ── P3.4: Paper-claim regression test (with LLM snapshot active) ───────────
banner "[P3.4] Paper-claim regression test"
run_step P3.4 "9/9 expected after P1.1" "$SKIP_PAPER_CLAIM" -- \
    "$PYTHON" -m pytest tests/test_paper_claims.py -v --tb=short

# ── P4.1: Figure generation ────────────────────────────────────────────────
# visualize_results.py는 results/experiment_results.json (P1.4의 출력)을 읽어
# Pareto frontier + comparison bar + latency 차트를 생성한다. 입력 파일이 없으면
# graceful skip — P1.4를 먼저 돌려야 한다는 안내만 출력.
banner "[P4.1] Figure generation (visualize_results)"
if [ "$SKIP_FIGURES" = "1" ]; then
    run_step P4.1 "skipped" "1" -- :
elif [ -f "$ROOT/results/experiment_results.json" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY: $PYTHON experiments/visualize_results.py  (writes results/*.png)"
        set_status P4.1 "dry"; set_duration P4.1 "0"
    else
        t0=$(date +%s); rc=0
        "$PYTHON" experiments/visualize_results.py || rc=$?
        # Snapshot generated PNGs into RUN_DIR/figures/
        mkdir -p "$RUN_DIR/figures"
        for fn in pareto_frontier.png comparison_bar.png latency_comparison.png; do
            [ -f "$ROOT/results/$fn" ] && cp "$ROOT/results/$fn" "$RUN_DIR/figures/$fn"
        done
        t1=$(date +%s); set_duration P4.1 "$((t1 - t0))"
        if [ "$rc" -eq 0 ]; then
            set_status P4.1 "ok"; ok "[P4.1] figures generated"
        else
            set_status P4.1 "fail($rc)"; err "[P4.1] failed rc=$rc"
            if [ "$STRICT_FAIL" = "1" ]; then exit "$rc"; fi
        fi
    fi
else
    set_status P4.1 "skip"; set_duration P4.1 "0"
    warn "[P4.1] SKIP — results/experiment_results.json not found (run P1.4 first)"
fi

# ── P4.2: Cross-tag run comparison (optional) ──────────────────────────────
# COMPARE_TAGS="base instructed confidence" 처럼 여러 ablation 태그가 results/<tag>/
# 구조로 존재할 때만 의미 있음. default SKIP_COMPARE_RUNS=1.
banner "[P4.2] Cross-tag run comparison (optional)"
if [ "$SKIP_COMPARE_RUNS" = "1" ] || [ -z "$COMPARE_TAGS" ]; then
    set_status P4.2 "skip"; set_duration P4.2 "0"
    warn "[P4.2] SKIP — set COMPARE_TAGS=\"tag1 tag2 …\" SKIP_COMPARE_RUNS=0 to enable"
else
    run_step P4.2 "comparing tags: $COMPARE_TAGS" "0" -- \
        "$PYTHON" experiments/compare_runs.py $COMPARE_TAGS
fi

# ── POST: Aggregate report ────────────────────────────────────────────────
banner "[Post] All-in-one report"

REPORT="$RUN_DIR/ALL_IN_ONE_REPORT.md"
{
    echo "# UASEF Round 8 — All-in-One Report"
    echo ""
    echo "- timestamp: $TIMESTAMP"
    echo "- run_dir: \`$RUN_DIR\`"
    echo "- log: \`$LOG\`"
    echo ""
    echo "## Step status"
    echo ""
    echo "| Step | Description | Status | Duration (s) |"
    echo "| --- | --- | --- | --- |"
    desc_for() {
        case "$1" in
            P0)   echo "pytest regression";;
            P0.5) echo "calibration pipeline (RTC/entropy/EDE)";;
            P1.1) echo "single-seed full evaluation (run_full_evaluation.sh)";;
            P1.2) echo "5-seed multi-seed bootstrap";;
            P1.3) echo "LLM-judge self-consistency";;
            P1.4) echo "primary+ablation logprob CP (run_experiment.py)";;
            P2.1) echo "multi-dataset generalization";;
            P2.2) echo "Pivot B case study";;
            P3.1) echo "distribution shift sanity";;
            P3.2) echo "multi-lingual sanity";;
            P3.3) echo "equity audit";;
            P3.4) echo "paper-claim regression";;
            P4.1) echo "figure generation (visualize_results)";;
            P4.2) echo "cross-tag run comparison (compare_runs)";;
            *) echo "?";;
        esac
    }
    for K in $ALL_STEPS; do
        echo "| $K | $(desc_for $K) | $(get_status $K) | $(get_duration $K) |"
    done
    echo ""

    # P1.2 aggregate summary preview
    if [ -f "$RUN_DIR/latest_aggregate_dir.txt" ]; then
        AGG="$(cat "$RUN_DIR/latest_aggregate_dir.txt")"
        if [ -f "$AGG/aggregate_seeds.md" ]; then
            echo "## P1.2 Multi-seed bootstrap summary"
            echo ""
            echo "Source: \`$AGG/aggregate_seeds.md\`"
            echo ""
            sed -n '1,80p' "$AGG/aggregate_seeds.md" || true
            echo ""
        fi
    fi

    # P1.3 κ
    if [ -f "$RUN_DIR/llm_judge_relabel.json" ]; then
        echo "## P1.3 LLM-judge κ"
        echo ""
        "$PYTHON" - "$RUN_DIR/llm_judge_relabel.json" 2>/dev/null <<'PY' || warn "could not parse"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"- κ = **{d.get('cohen_kappa')}**")
print(f"- n_consensus = {d.get('n_consensus')}, n_disagree = {d.get('n_disagreement')}, parse_fail = {d.get('n_parse_fail')}")
print(f"- judges: {d.get('judges')} → models {d.get('judge_models')}")
print(f"- heuristic_vs_consensus_agreement = {d.get('heuristic_vs_consensus_agreement')}")
PY
        echo ""
    fi

    # P2.1 multi-dataset compact
    if [ -f "$RUN_DIR/multidataset_summary.md" ]; then
        echo "## P2.1 Multi-dataset compact"
        echo ""
        cat "$RUN_DIR/multidataset_summary.md" || true
        echo ""
    fi

    # P2.2 Pivot B
    if [ -f "$RUN_DIR/pivotB_case_study.md" ]; then
        echo "## P2.2 Pivot B case study"
        echo ""
        cat "$RUN_DIR/pivotB_case_study.md" || true
        echo ""
    fi

    # P3.1
    if [ -f "$RUN_DIR/distribution_shift.md" ]; then
        echo "## P3.1 Distribution shift"
        echo ""
        cat "$RUN_DIR/distribution_shift.md" || true
        echo ""
    fi

    # P3.3 equity
    for B in $BACKENDS; do
        if [ -f "$RUN_DIR/equity_audit_${B}.md" ]; then
            echo "## P3.3 Equity audit ($B)"
            echo ""
            cat "$RUN_DIR/equity_audit_${B}.md" || true
            echo ""
        fi
    done

    echo "## Reproduction"
    echo ""
    echo "\`\`\`bash"
    echo "BACKENDS=\"$BACKENDS\" SEEDS_FOR_MULTISEED=\"$SEEDS_FOR_MULTISEED\" \\"
    echo "    N_CAL=$N_CAL N_TEST=$N_TEST ALPHA=$ALPHA \\"
    echo "    JUDGE_OPENAI_MODEL=$JUDGE_OPENAI_MODEL \\"
    echo "    JUDGE_ANTHROPIC_MODEL=$JUDGE_ANTHROPIC_MODEL \\"
    echo "    bash run_all_round8.sh"
    echo "\`\`\`"
} > "$REPORT"

ok "Report written: $REPORT"

# ── Final summary line ─────────────────────────────────────────────────────
banner "Done"
N_OK=0; N_SKIP=0; N_FAIL=0; N_DRY=0
for K in $ALL_STEPS; do
    case "$(get_status $K)" in
        ok)   N_OK=$((N_OK + 1)) ;;
        skip) N_SKIP=$((N_SKIP + 1)) ;;
        fail*) N_FAIL=$((N_FAIL + 1)) ;;
        dry)  N_DRY=$((N_DRY + 1)) ;;
    esac
done
echo "  Summary: ${C_OK}${N_OK} ok${C_RST} / ${C_WARN}${N_SKIP} skip${C_RST} / ${C_ERR}${N_FAIL} fail${C_RST} / ${N_DRY} dry"
echo "  Report : $REPORT"
echo "  Log    : $LOG"

if [ "$N_FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
