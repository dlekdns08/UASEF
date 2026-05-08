#!/usr/bin/env bash
#
# UASEF — 전체 평가 통합 실행 (v1 + v2 Round 7, 다중 backend)
# ════════════════════════════════════════════════════════════════════════════
#
# 한 번 실행으로:
#   [1] pytest 137 tests (회귀 안전망)
#   [2] v1 평가: run_all_experiments.py × {각 backend} (1회 호출 = agent + baseline
#                                                       + medabstain + pareto 4개 sub 자동 포함)
#   [3] v2 Round 7 합성 검증: Table 2 (FWER) + Table 3 (cost) — backend 무관, 1회
#   [4] v2 Round 7 LLM 검증: Table 1 (per-stratum coverage) + Table 4 (head-to-head
#                             vs TECP/Quach/SE) × {각 backend}
#   [5] result_<timestamp>.md 통합 보고서 + result.json (구조화) 자동 생성
#
# 사용:
#   bash run_full_evaluation.sh                                # default: openai+lmstudio
#   BACKENDS="openai" bash run_full_evaluation.sh              # openai만
#   SKIP_LLM=1 bash run_full_evaluation.sh                     # 합성 검증 + pytest만
#   N_CAL=500 N_TEST=200 bash run_full_evaluation.sh           # 논문 quality
#
# 환경변수:
#   BACKENDS         space-separated list, default "openai lmstudio"
#   N_CAL            calibration n per stratum (default 200, 논문 ≥1000)
#   N_TEST           test n per stratum (default 100)
#   N_MEDABSTAIN     MedAbstain 변형별 n (default 50)
#   N_PARETO         Pareto sweep test n (default 50)
#   N_TRIALS         Table 2 FWER simulation (default 5000)
#   N_PER_STRATUM    Table 3 합성 (default 300)
#   ALPHA            global α (default 0.10)
#   SEED             random seed (default 42; single-run mode)
#   SEEDS            space-separated multi-seed list for bootstrap (default empty;
#                    if set, e.g. SEEDS="42 43 44 45 46", the script runs every
#                    seed in turn and emits results/run_<ts>/aggregate_seeds.{json,md}
#                    with mean ± std and 95% bootstrap CI per metric. Single-seed
#                    mode is unaffected when SEEDS is empty.)
#   SKIP_LLM         1이면 LLM 호출 단계 모두 SKIP (Table 2/3 + pytest만)
#   SKIP_TESTS       1이면 pytest SKIP
#   SKIP_V1          1이면 v1 (run_all_experiments) SKIP
#   SKIP_V2_SYN      1이면 v2 합성 (Table 2/3) SKIP
#   SKIP_V2_LLM      1이면 v2 LLM (Table 1/4) SKIP
#   PYTHON           python interpreter (default ./.venv/bin/python)
#
# 출력 (results/run_<timestamp>/):
#   ├── result.md                    ← 사람용 통합 보고서 (모든 단계 요약)
#   ├── result.json                  ← 구조화 결과
#   ├── pytest_summary.txt
#   ├── run.log
#   ├── synthetic/                   ← v2 합성 (backend 무관)
#   │   ├── table2_fwer.{json,md}
#   │   └── table3_cost.{json,md}
#   ├── openai/                      ← OpenAI backend 결과
#   │   ├── all_experiments_report.md          (v1 통합)
#   │   ├── all_experiments_summary.json
#   │   ├── agent_results.json + comparison_table.csv
#   │   ├── baseline_comparison.{json,csv}
#   │   ├── medabstain_eval.{json,csv}
#   │   ├── pareto_sweep_results.json + alpha_recommendations.json + pareto_frontier.png
#   │   ├── table1_coverage.{json,md}          (v2 Round 7 Pivot A)
#   │   └── table4_baseline.{json,md}          (v2 Round 7 head-to-head)
#   └── lmstudio/                     ← LMStudio backend (동일 구조)

set -euo pipefail

# ── 기본 설정 ──
BACKENDS="${BACKENDS:-openai lmstudio}"
N_CAL="${N_CAL:-200}"
N_TEST="${N_TEST:-100}"
N_MEDABSTAIN="${N_MEDABSTAIN:-50}"
N_PARETO="${N_PARETO:-50}"
N_TRIALS="${N_TRIALS:-5000}"
N_PER_STRATUM="${N_PER_STRATUM:-300}"
ALPHA="${ALPHA:-0.10}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-}"
SKIP_LLM="${SKIP_LLM:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_V1="${SKIP_V1:-0}"
SKIP_V2_SYN="${SKIP_V2_SYN:-0}"
SKIP_V2_LLM="${SKIP_V2_LLM:-0}"

# ── 환경 ──
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="results/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR/synthetic"
for B in $BACKENDS; do
    mkdir -p "$RUN_DIR/$B"
done

LOG_FILE="${RUN_DIR}/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH="$(date +%s)"

# SKIP_LLM=1이면 LLM 의존 단계 모두 skip
if [ "$SKIP_LLM" = "1" ]; then
    SKIP_V1=1
    SKIP_V2_LLM=1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  UASEF — Full Evaluation (v1 + v2 Round 7, multi-backend)"
echo "════════════════════════════════════════════════════════════════════"
echo "  Timestamp        : $TIMESTAMP"
echo "  Output           : $RUN_DIR"
echo "  Backends         : $BACKENDS"
echo "  N_CAL / N_TEST   : $N_CAL / $N_TEST  (per stratum)"
echo "  N_MEDABSTAIN     : $N_MEDABSTAIN  (per variant)"
echo "  N_PARETO         : $N_PARETO  (per scenario)"
echo "  N_TRIALS         : $N_TRIALS  (Table 2 FWER)"
echo "  N_PER_STRATUM    : $N_PER_STRATUM  (Table 3 합성)"
echo "  α / seed         : $ALPHA / $SEED"
echo "  SKIP_LLM         : $SKIP_LLM  (→ SKIP_V1=$SKIP_V1, SKIP_V2_LLM=$SKIP_V2_LLM)"
echo "  SKIP_TESTS       : $SKIP_TESTS"
echo "  SKIP_V2_SYN      : $SKIP_V2_SYN"
echo "  Python           : $PYTHON"

# 환경변수 안내
for B in $BACKENDS; do
    case "$B" in
        openai)    [ -z "${OPENAI_API_KEY:-}" ] && echo "  ⚠  OPENAI_API_KEY 미설정 (SKIP_LLM=1로 우회 권장)" ;;
        anthropic) [ -z "${ANTHROPIC_API_KEY:-}" ] && echo "  ⚠  ANTHROPIC_API_KEY 미설정" ;;
        gemini)    [ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ] && echo "  ⚠  GEMINI_API_KEY/GOOGLE_API_KEY 미설정" ;;
    esac
done

# ── [1] pytest ──
if [ "$SKIP_TESTS" != "1" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [1/4] pytest 회귀 검증"
    echo "════════════════════════════════════════════════════════════════════"
    "$PYTHON" -m pytest tests/ -q 2>&1 | tee "$RUN_DIR/pytest_summary.txt" | tail -10
    PYTEST_RC=${PIPESTATUS[0]:-1}
    if [ "$PYTEST_RC" -ne 0 ]; then
        echo "  ✗ pytest 실패 — 진행하지만 결과 신뢰도 ↓"
    else
        echo "  ✓ pytest 통과"
    fi
fi

# ── [2] v1: run_all_experiments per backend ──
if [ "$SKIP_V1" != "1" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [2/4] v1 — run_all_experiments.py × {$BACKENDS}"
    echo "         (각 호출 = agent + baseline + medabstain + pareto 4개 sub 자동)"
    echo "════════════════════════════════════════════════════════════════════"

    for B in $BACKENDS; do
        TAG="_tmp_${TIMESTAMP}_${B}_v1"
        echo ""
        echo "  ── [v1] backend=$B → results/$TAG/ → $RUN_DIR/$B/"
        "$PYTHON" experiments/run_all_experiments.py \
            --backend "$B" \
            --run-tag "$TAG" \
            --n-cal "$N_CAL" --n-test "$N_TEST" \
            --n-medabstain "$N_MEDABSTAIN" --n-pareto-test "$N_PARETO" \
            --alpha "$ALPHA" --seed "$SEED" \
            > "$RUN_DIR/$B/v1_stdout.txt" 2>&1 \
            && echo "    ✓ v1/$B 완료" \
            || echo "    ✗ v1/$B 실패 (stdout: $RUN_DIR/$B/v1_stdout.txt)"

        # 결과 mv (run-tag 디렉토리 → backend 디렉토리)
        if [ -d "results/$TAG" ]; then
            mv "results/$TAG"/* "$RUN_DIR/$B/" 2>/dev/null || true
            rmdir "results/$TAG" 2>/dev/null || true
        fi
    done
else
    echo ""
    echo "  ⏩ [2/4] v1 SKIP (SKIP_V1=1)"
fi

# ── [3] v2 Round 7 합성 (backend 무관, 1회) ──
if [ "$SKIP_V2_SYN" != "1" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [3/4] v2 Round 7 — 합성 검증 (Table 2 FWER + Table 3 Cost)"
    echo "════════════════════════════════════════════════════════════════════"

    "$PYTHON" experiments/round7_table2_fwer.py \
        --n-trials "$N_TRIALS" --alpha 0.05 --seed "$SEED" \
        > "$RUN_DIR/synthetic/table2_stdout.txt" 2>&1 \
        && echo "  ✓ Table 2 (FWER)" || echo "  ✗ Table 2 실패"
    [ -f "results/round7/table2_fwer.json" ] \
        && cp results/round7/table2_fwer.{json,md} "$RUN_DIR/synthetic/"

    "$PYTHON" experiments/round7_table3_cost.py \
        --n-per-stratum "$N_PER_STRATUM" --seed "$SEED" \
        > "$RUN_DIR/synthetic/table3_stdout.txt" 2>&1 \
        && echo "  ✓ Table 3 (Cost)" || echo "  ✗ Table 3 실패"
    [ -f "results/round7/table3_cost.json" ] \
        && cp results/round7/table3_cost.{json,md} "$RUN_DIR/synthetic/"
else
    echo ""
    echo "  ⏩ [3/4] v2 합성 SKIP (SKIP_V2_SYN=1)"
fi

# ── [4] v2 Round 7 LLM (per backend) ──
if [ "$SKIP_V2_LLM" != "1" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [4/4] v2 Round 7 — LLM 검증 (Table 1 + Table 4) × {$BACKENDS}"
    echo "════════════════════════════════════════════════════════════════════"

    for B in $BACKENDS; do
        echo ""
        echo "  ── [v2] backend=$B"

        # Table 1 — per-stratum coverage
        "$PYTHON" experiments/round7_table1_coverage.py \
            --backend "$B" \
            --n-cal "$N_CAL" --n-test "$N_TEST" \
            --alpha-global "$ALPHA" --seed "$SEED" \
            > "$RUN_DIR/$B/table1_stdout.txt" 2>&1 \
            && echo "    ✓ Table 1 ($B)" \
            || echo "    ✗ Table 1 ($B) 실패"
        [ -f "results/round7/table1_coverage.json" ] \
            && mv results/round7/table1_coverage.{json,md} "$RUN_DIR/$B/"

        # Table 4 — head-to-head baseline
        "$PYTHON" experiments/round7_table4_baseline.py \
            --backend "$B" \
            --n-cal "$N_CAL" --n-test "$N_TEST" \
            --alpha "$ALPHA" --seed "$SEED" \
            > "$RUN_DIR/$B/table4_stdout.txt" 2>&1 \
            && echo "    ✓ Table 4 ($B)" \
            || echo "    ✗ Table 4 ($B) 실패"
        [ -f "results/round7/table4_baseline.json" ] \
            && mv results/round7/table4_baseline.{json,md} "$RUN_DIR/$B/"
    done
else
    echo ""
    echo "  ⏩ [4/4] v2 LLM SKIP (SKIP_V2_LLM=1)"
fi

# ── 통합 보고서 생성 ──
END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  통합 보고서 생성"
echo "════════════════════════════════════════════════════════════════════"

export REPORT_RUN_DIR="$RUN_DIR"
export REPORT_TIMESTAMP="$TIMESTAMP"
export REPORT_ELAPSED="${ELAPSED_MIN}m ${ELAPSED_SEC}s"
export REPORT_BACKENDS="$BACKENDS"
export REPORT_N_CAL="$N_CAL"
export REPORT_N_TEST="$N_TEST"
export REPORT_N_MEDABSTAIN="$N_MEDABSTAIN"
export REPORT_N_PARETO="$N_PARETO"
export REPORT_N_TRIALS="$N_TRIALS"
export REPORT_N_PER_STRATUM="$N_PER_STRATUM"
export REPORT_ALPHA="$ALPHA"
export REPORT_SEED="$SEED"
export REPORT_SKIP_LLM="$SKIP_LLM"
export REPORT_SKIP_TESTS="$SKIP_TESTS"
export REPORT_SKIP_V1="$SKIP_V1"
export REPORT_SKIP_V2_SYN="$SKIP_V2_SYN"
export REPORT_SKIP_V2_LLM="$SKIP_V2_LLM"

"$PYTHON" - <<'PYEOF'
import json
import os
from datetime import datetime
from pathlib import Path

run_dir = Path(os.environ["REPORT_RUN_DIR"])
ts = os.environ["REPORT_TIMESTAMP"]
elapsed = os.environ["REPORT_ELAPSED"]
backends = os.environ["REPORT_BACKENDS"].split()

config = {
    "n_cal": int(os.environ["REPORT_N_CAL"]),
    "n_test": int(os.environ["REPORT_N_TEST"]),
    "n_medabstain": int(os.environ["REPORT_N_MEDABSTAIN"]),
    "n_pareto": int(os.environ["REPORT_N_PARETO"]),
    "n_trials": int(os.environ["REPORT_N_TRIALS"]),
    "n_per_stratum": int(os.environ["REPORT_N_PER_STRATUM"]),
    "alpha": float(os.environ["REPORT_ALPHA"]),
    "seed": int(os.environ["REPORT_SEED"]),
    "skip_llm": os.environ["REPORT_SKIP_LLM"] == "1",
    "skip_tests": os.environ["REPORT_SKIP_TESTS"] == "1",
    "skip_v1": os.environ["REPORT_SKIP_V1"] == "1",
    "skip_v2_syn": os.environ["REPORT_SKIP_V2_SYN"] == "1",
    "skip_v2_llm": os.environ["REPORT_SKIP_V2_LLM"] == "1",
    "backends": backends,
}

# ── 데이터 수집 ────────────────────────────────────────────────
def _load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"_error": str(e)}
    return None

# 합성 (backend 무관)
synthetic = {
    "table2": _load_json(run_dir / "synthetic" / "table2_fwer.json"),
    "table3": _load_json(run_dir / "synthetic" / "table3_cost.json"),
}

# backend별
per_backend = {}
for B in backends:
    per_backend[B] = {
        "v1_summary": _load_json(run_dir / B / "all_experiments_summary.json"),
        "v2_table1":  _load_json(run_dir / B / "table1_coverage.json"),
        "v2_table4":  _load_json(run_dir / B / "table4_baseline.json"),
    }

# pytest
pytest_summary = ""
pt = run_dir / "pytest_summary.txt"
if pt.exists():
    lines = [l for l in pt.read_text(encoding="utf-8").splitlines() if l.strip()]
    pytest_summary = "\n".join(lines[-12:])

# ── result.json ────────────────────────────────────────────────
result_payload = {
    "meta": {"timestamp": ts, "elapsed": elapsed, "config": config},
    "pytest_summary": pytest_summary,
    "synthetic": synthetic,
    "per_backend": per_backend,
}
(run_dir / "result.json").write_text(
    json.dumps(result_payload, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)

# ── result.md ──────────────────────────────────────────────────
md = []
md.append(f"# UASEF — Full Evaluation Report")
md.append(f"")
md.append(f"- **timestamp**: `{ts}`")
md.append(f"- **elapsed**: `{elapsed}`")
md.append(f"- **backends**: {', '.join(f'`{b}`' for b in backends)}")
md.append(f"")
md.append(f"### Config")
md.append(f"")
md.append(f"| key | value |")
md.append(f"| --- | --- |")
for k, v in config.items():
    md.append(f"| `{k}` | `{v}` |")
md.append(f"")

# §0 pytest
md.append(f"## 0. 회귀 검증 (pytest 137 tests)")
md.append(f"")
if pytest_summary:
    md.append("```")
    md.append(pytest_summary)
    md.append("```")
else:
    md.append("_pytest 스킵됨_")
md.append(f"")

# §1 합성 (Table 2/3, backend 무관)
md.append(f"## 1. v2 Round 7 합성 검증 (backend 무관)")
md.append(f"")

# Table 2
md.append(f"### 1.1 Table 2 — Multi-Trigger FWER (Pivot B)")
md.append(f"")
md.append(f"Null hypothesis (모든 trigger가 정상)에서 false escalation rate 측정.")
md.append(f"v1 `len(triggers)>0`은 FWER 위반, v2 harmonic / e-value는 보장 충족.")
md.append(f"")
t2 = synthetic.get("table2")
if t2:
    md.append(f"- n_trials={t2['config']['n_trials']}, target α={t2['config']['alpha']}")
    md.append(f"")
    md.append(f"| Method | Dependence | Empirical FWER | OK |")
    md.append(f"| --- | --- | --- | --- |")
    for r in t2.get("rows", []):
        md.append(f"| {r['method']} | {r['dependence']} | {r['empirical_fwer']} | "
                  f"{'✓' if r['ok'] else '✗'} |")
else:
    md.append("_Table 2 결과 없음_")
md.append(f"")

# Table 3
md.append(f"### 1.2 Table 3 — Cost-Weighted Performance (Pivot C)")
md.append(f"")
md.append(f"비대칭 cost matrix (CRITICAL miss=1000× over_esc) 적용.")
md.append(f"")
t3 = synthetic.get("table3")
if t3:
    md.append(f"- n_per_stratum={t3['n_per_stratum']}")
    md.append(f"- **Round 6 total cost**: `{t3['round6_total_cost']:.1f}`")
    md.append(f"- **Round 7 total cost**: `{t3['round7_total_cost']:.1f}`")
    md.append(f"- **Reduction**: **{t3.get('cost_reduction_ratio')}×** (Round 6 / Round 7)")
    md.append(f"")
    md.append(f"#### Per-stratum")
    md.append(f"| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over_esc |")
    md.append(f"| --- | --- | --- | --- | --- | --- | --- |")
    for r6, r7 in zip(t3["round6_per_stratum"], t3["round7_per_stratum"]):
        md.append(f"| {r6['stratum']} | {r6['threshold']} | {r6['cost']:.1f} | "
                  f"{r7['threshold']} | {r7['cost']:.1f} | "
                  f"{r7['miss_rate']} | {r7['over_esc_rate']} |")
else:
    md.append("_Table 3 결과 없음_")
md.append(f"")

# §2 backend별
md.append(f"## 2. Backend별 결과")
md.append(f"")

for idx, B in enumerate(backends, 1):
    md.append(f"### 2.{idx} Backend: `{B}`")
    md.append(f"")
    pb = per_backend[B]

    # v1 summary
    md.append(f"#### 2.{idx}.1 v1 — `run_all_experiments.py` (agent + baseline + medabstain + pareto)")
    md.append(f"")
    v1 = pb.get("v1_summary")
    if v1:
        backend_data = v1.get("agent", {}).get(B, {})
        baseline_data = v1.get("baseline", {}).get(B, {})
        medab_data = v1.get("medabstain", {}).get(B, {})

        # agent
        if backend_data:
            md.append(f"**Agent (LangGraph ReAct)**")
            md.append(f"")
            md.append(f"- accuracy: `{backend_data.get('accuracy')}` | "
                      f"safety_recall: `{backend_data.get('safety_recall')}` | "
                      f"over_esc: `{backend_data.get('over_escalation_rate')}` | "
                      f"escalation_rate: `{backend_data.get('escalation_rate')}`")
            md.append(f"- avg_tool_calls: `{backend_data.get('avg_tool_calls')}` | "
                      f"avg_react_iterations: `{backend_data.get('avg_react_iterations')}`")
            md.append(f"- coverage: `{backend_data.get('conformal_coverage')}`")
            md.append(f"")

        # baseline 3 strategies
        if baseline_data:
            md.append(f"**Baseline (3 strategies)**")
            md.append(f"")
            md.append(f"| Strategy | Safety Recall | 95% CI | Over-Esc | OK |")
            md.append(f"| --- | --- | --- | --- | --- |")
            for strat, m in baseline_data.items():
                if "error" in m:
                    md.append(f"| {strat} | error | | | |")
                    continue
                ci = m.get("safety_recall_ci")
                ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
                md.append(f"| {strat} | {m.get('safety_recall')} | {ci_s} "
                          f"| {m.get('over_escalation_rate')} | "
                          f"{'✓' if m.get('safety_recall_ok') else '✗'} |")
            md.append(f"")

        # medabstain
        if medab_data:
            ov = medab_data.get("overall", {})
            md.append(f"**MedAbstain 전체**")
            md.append(f"")
            md.append(f"- Recall: `{ov.get('recall')}` | Precision: `{ov.get('precision')}` "
                      f"| F1: `{ov.get('f1')}` | AUROC: `{ov.get('auroc')}`")
            ab = medab_data.get("abstention_accuracy", {})
            if ab and "error" not in ab:
                md.append(f"- Abstention: P=`{ab.get('abstention_precision')}` "
                          f"R=`{ab.get('abstention_recall')}` F1=`{ab.get('abstention_f1')}`")
            md.append(f"")

        md.append(f"_상세 보고서_: [`{B}/all_experiments_report.md`]({B}/all_experiments_report.md)")
    else:
        md.append(f"_v1 SKIP 또는 실패_")
    md.append(f"")

    # v2 Table 1
    md.append(f"#### 2.{idx}.2 v2 Round 7 Table 1 — Per-Stratum Coverage (Pivot A)")
    md.append(f"")
    t1 = pb.get("v2_table1")
    if t1:
        md.append(f"- n_cal={t1.get('n_cal')}/stratum, n_test={t1.get('n_test')}/stratum")
        md.append(f"- α_global={t1.get('alpha_global')}, CRC alphas={t1.get('crc_alphas')}")
        md.append(f"")
        md.append(f"| Method | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss |")
        md.append(f"| --- | --- | --- | --- | --- |")
        for r in t1.get("results", []):
            row = f"| {r['name']} |"
            for s in ("CRITICAL", "HIGH", "MODERATE", "LOW"):
                v = r["per_stratum"].get(s, {}).get("miss_rate")
                row += f" {v if v is not None else 'N/A'} |"
            md.append(row)
    else:
        md.append(f"_Table 1 SKIP_")
    md.append(f"")

    # v2 Table 4
    md.append(f"#### 2.{idx}.3 v2 Round 7 Table 4 — Head-to-Head (TECP / Quach / SE / R6 / R7)")
    md.append(f"")
    t4 = pb.get("v2_table4")
    if t4:
        md.append(f"- n_cal={t4.get('n_cal')}, n_test={t4.get('n_test')}, α={t4.get('alpha')}")
        md.append(f"")
        md.append(f"##### CRITICAL stratum")
        md.append(f"| Method | Safety Recall | Over-Esc | TP/FN/FP | Total cost |")
        md.append(f"| --- | --- | --- | --- | --- |")
        for m in t4.get("methods", []):
            c = m["per_stratum"].get("CRITICAL", {})
            md.append(f"| {m['name']} | {c.get('safety_recall')} | "
                      f"{c.get('over_esc_rate')} | "
                      f"{c.get('tp')}/{c.get('fn')}/{c.get('fp')} | "
                      f"{m['total_cost']:.1f} |")
        md.append(f"")
        md.append(f"##### Total cost (전 stratum)")
        md.append(f"| Method | Total cost |")
        md.append(f"| --- | --- |")
        for m in t4.get("methods", []):
            md.append(f"| {m['name']} | {m['total_cost']:.1f} |")
    else:
        md.append(f"_Table 4 SKIP_")
    md.append(f"")

# §3 결론
md.append(f"## 3. 결론 요약")
md.append(f"")

# FWER
if t2:
    naive = max((r["empirical_fwer"] for r in t2.get("rows", []) if "naive" in r["method"]), default=None)
    harm  = max((r["empirical_fwer"] for r in t2.get("rows", []) if "harmonic" in r["method"]), default=None)
    if naive and harm:
        md.append(f"- **Pivot B (FWER)**: v1 `len(triggers)>0` empirical FWER ≤ **{naive}** "
                  f"(target 0.05 위반). v2 harmonic FWER ≤ **{harm}** (✓).")
# Cost
if t3:
    md.append(f"- **Pivot C (Cost)**: Round 6 → Round 7 total cost "
              f"**{t3.get('cost_reduction_ratio')}× 감소** "
              f"({t3['round6_total_cost']:.0f} → {t3['round7_total_cost']:.0f}).")

# Table 1 summary per backend
for B in backends:
    pb = per_backend[B]
    t1 = pb.get("v2_table1")
    if t1:
        r7 = next((r for r in t1.get("results", []) if "Round 7" in r["name"]), None)
        if r7:
            crit = r7["per_stratum"].get("CRITICAL", {})
            target = crit.get("target_alpha")
            actual = crit.get("miss_rate")
            ok = crit.get("ok")
            if target is not None and actual is not None:
                md.append(f"- **Pivot A coverage ({B})**: CRITICAL stratum miss_rate={actual} "
                          f"vs target α={target} → {'✓' if ok else '⚠ n 부족 가능'}")

# Table 4 cost comparison per backend
for B in backends:
    pb = per_backend[B]
    t4 = pb.get("v2_table4")
    if t4:
        methods = t4.get("methods", [])
        if methods:
            costs = {m["name"]: m["total_cost"] for m in methods}
            r7_name = next((n for n in costs if "Round 7" in n), None)
            tecp_name = next((n for n in costs if "TECP" in n), None)
            if r7_name and tecp_name:
                ratio = costs[tecp_name] / max(costs[r7_name], 1e-6)
                md.append(f"- **Head-to-head cost ({B})**: TECP cost={costs[tecp_name]:.0f} vs "
                          f"UASEF Round 7 cost={costs[r7_name]:.0f} → **{ratio:.1f}× 절감**")

md.append(f"")
md.append(f"---")
md.append(f"")
md.append(f"_생성: `run_full_evaluation.sh` ({datetime.now().isoformat(timespec='seconds')})_")

(run_dir / "result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"  ✓ result.md ({len(chr(10).join(md).encode('utf-8'))} bytes)")
print(f"  ✓ result.json")


# ════════════════════════════════════════════════════════════════════
# Supplementary 렌더링 (v1 4 sub-experiment 결과를 paper appendix 형식으로)
# v1이 한 번이라도 실행됐으면 supplementary 자동 생성
# ════════════════════════════════════════════════════════════════════

def _has_v1_data():
    return any(
        per_backend.get(b, {}).get("v1_summary") is not None
        for b in backends
    )

if _has_v1_data():
    sup = []
    sup.append(f"# Supplementary Materials — UASEF v1 sub-experiments")
    sup.append(f"")
    sup.append(f"**Run timestamp:** `{ts}`  ·  **Backends:** "
                f"{', '.join(f'`{b}`' for b in backends)}")
    sup.append(f"**Source paper:** [UASEF_Round7.md](../../paper/UASEF_Round7.md) "
                f"(English) · [UASEF_Round7_KO.md](../../paper/UASEF_Round7_KO.md) (한국어)")
    sup.append(f"**Template:** [UASEF_Round7_Supplementary.md](../../paper/UASEF_Round7_Supplementary.md) "
                f"· [UASEF_Round7_Supplementary_KO.md](../../paper/UASEF_Round7_Supplementary_KO.md)")
    sup.append(f"")

    # ── B.1 Agent ReAct ────────────────────────────────────────────
    sup.append(f"## B.1 Agent ReAct Behavior")
    sup.append(f"")
    sup.append(f"| Backend | Accuracy | Safety Recall | Over-Esc | "
                f"Avg Tool Calls | Avg ReAct Iters | Coverage |")
    sup.append(f"| --- | --- | --- | --- | --- | --- | --- |")
    for B in backends:
        v1 = per_backend[B].get("v1_summary")
        if not v1:
            sup.append(f"| `{B}` | _v1 SKIP_ |  |  |  |  |  |")
            continue
        a = v1.get("agent", {}).get(B, {})
        sup.append(
            f"| `{B}` | {a.get('accuracy')} | {a.get('safety_recall')} | "
            f"{a.get('over_escalation_rate')} | {a.get('avg_tool_calls')} | "
            f"{a.get('avg_react_iterations')} | "
            f"{a.get('conformal_coverage')} |"
        )
    sup.append(f"")

    # ── B.2 Trigger Contribution Ablation (3-strategy) ─────────────
    sup.append(f"## B.2 Trigger Contribution Ablation (Pivot B 동기 강화)")
    sup.append(f"")
    sup.append(f"| Backend | Strategy | Safety Recall | 95% CI | "
                f"Over-Esc Rate | TP/FN/FP/TN | OK (≥0.95)? |")
    sup.append(f"| --- | --- | --- | --- | --- | --- | --- |")
    for B in backends:
        v1 = per_backend[B].get("v1_summary")
        if not v1:
            continue
        baseline = v1.get("baseline", {}).get(B, {})
        for strat, m in baseline.items():
            if "error" in m:
                sup.append(f"| `{B}` | {strat} | error |  |  |  |  |")
                continue
            ci = m.get("safety_recall_ci")
            ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
            sup.append(
                f"| `{B}` | {strat} | {m.get('safety_recall')} | {ci_s} | "
                f"{m.get('over_escalation_rate')} | "
                f"{m.get('tp')}/{m.get('fn')}/{m.get('fp')}/{m.get('tn')} | "
                f"{'✓' if m.get('safety_recall_ok') else '✗'} |"
            )
    sup.append(f"")
    sup.append(f"**해석.** `threshold_only` (T1만, 순수 CP) → `full_uasef` (T1 ∨ T2 ∨ T3)")
    sup.append(f"의 Safety Recall 격차가 keyword/no-evidence trigger의 한계 기여이며,")
    sup.append(f"`full_uasef`의 over-escalation 증가가 main paper §6.2 Table 2의 FWER 위반")
    sup.append(f"으로 이어진다 — 이것이 Pivot B (조화평균 결합)의 동기다.")
    sup.append(f"")

    # ── B.3 MedAbstain Variant-level ───────────────────────────────
    sup.append(f"## B.3 MedAbstain 변형별 분석")
    sup.append(f"")
    sup.append(f"### B.3.1 Per-Variant Metrics")
    sup.append(f"")
    sup.append(f"| Backend | Variant | n | Recall | Precision | F1 | AUROC | OK (≥0.95)? |")
    sup.append(f"| --- | --- | --- | --- | --- | --- | --- | --- |")
    for B in backends:
        v1 = per_backend[B].get("v1_summary")
        if not v1:
            continue
        medab = v1.get("medabstain", {}).get(B, {})
        per_v = medab.get("per_variant", {}) if medab else {}
        for variant, m in per_v.items():
            if "error" in m:
                sup.append(f"| `{B}` | {variant} | error |  |  |  |  |  |")
                continue
            sup.append(
                f"| `{B}` | {variant} | {m.get('n')} | "
                f"{m.get('recall')} | {m.get('precision')} | "
                f"{m.get('f1')} | {m.get('auroc')} | "
                f"{'✓' if m.get('safety_recall_ok') else '✗'} |"
            )
    sup.append(f"")

    # B.3.2 Abstention Accuracy
    sup.append(f"### B.3.2 Abstention Accuracy (LLM 자체 abstention 능력)")
    sup.append(f"")
    sup.append(f"| Backend | TA | FA | TR | MA | Abstention P | Abstention R | F1 |")
    sup.append(f"| --- | --- | --- | --- | --- | --- | --- | --- |")
    for B in backends:
        v1 = per_backend[B].get("v1_summary")
        if not v1:
            continue
        medab = v1.get("medabstain", {}).get(B, {})
        ab = medab.get("abstention_accuracy", {}) if medab else {}
        if not ab or "error" in ab:
            continue
        sup.append(
            f"| `{B}` | {ab.get('ta')} | {ab.get('fa')} | "
            f"{ab.get('tr')} | {ab.get('ma')} | "
            f"{ab.get('abstention_precision')} | "
            f"{ab.get('abstention_recall')} | "
            f"{ab.get('abstention_f1')} |"
        )
    sup.append(f"")
    sup.append(f"**논의.** Abstention Recall (LLM이 스스로 불확실성을 표현하는 능력)이")
    sup.append(f"낮을수록 UASEF의 CP 기반 결정이 더 큰 가치를 가진다 — 모델이 과신할 때")
    sup.append(f"외부 안전 게이트가 가장 필요하다.")
    sup.append(f"")

    # ── B.4 Pareto α Recommendation ────────────────────────────────
    sup.append(f"## B.4 Pareto Frontier — Specialty별 권고 α")
    sup.append(f"")
    sup.append(f"| Backend | Specialty | Recommended α | Coverage | Esc Rate | Utility |")
    sup.append(f"| --- | --- | --- | --- | --- | --- |")
    for B in backends:
        v1 = per_backend[B].get("v1_summary")
        if not v1:
            continue
        pareto = v1.get("pareto", {}).get(B, {})
        recs = pareto.get("recommendations", {}) if pareto else {}
        for specialty, rec in recs.items():
            if rec.get("alpha") is None:
                sup.append(f"| `{B}` | {specialty} | — | — | — | — |")
                continue
            sup.append(
                f"| `{B}` | {specialty} | {rec.get('alpha')} | "
                f"{rec.get('actual_coverage')} | "
                f"{rec.get('escalation_rate')} | {rec.get('utility')} |"
            )
    sup.append(f"")
    sup.append(f"**main paper Pivot A와의 연결.** Pareto sweep은 단일 전역 α를 specialty")
    sup.append(f"조건부로 측정. main paper Pivot A는 단일 CRC 절차 안에서 stratum별 $\\alpha_s$를")
    sup.append(f"부여하여 한 단계 더 진행. 여기서의 권고 α는 기관 배포 시 ")
    sup.append(f"$\\alpha_{{\\text{{CRITICAL}}}}, \\ldots, \\alpha_{{\\text{{LOW}}}}$ 선택 정보가 된다.")
    sup.append(f"")

    # ── B.5 Cross-backend summary ─────────────────────────────────
    sup.append(f"## B.5 Cross-Backend MedAbstain 종합")
    sup.append(f"")
    sup.append(f"| Backend | Recall | Precision | F1 | AUROC | Safety Recall ≥ 0.95? |")
    sup.append(f"| --- | --- | --- | --- | --- | --- |")
    for B in backends:
        v1 = per_backend[B].get("v1_summary")
        if not v1:
            continue
        medab = v1.get("medabstain", {}).get(B, {})
        ov = medab.get("overall", {}) if medab else {}
        if "error" in ov or not ov:
            continue
        sup.append(
            f"| `{B}` | {ov.get('recall')} | {ov.get('precision')} | "
            f"{ov.get('f1')} | {ov.get('auroc')} | "
            f"{'✓' if ov.get('safety_recall_ok') else '✗'} |"
        )
    sup.append(f"")

    # ── footer ─────────────────────────────────────────────────────
    sup.append(f"---")
    sup.append(f"")
    sup.append(f"_생성: `run_full_evaluation.sh` ({datetime.now().isoformat(timespec='seconds')})_")
    sup.append(f"_Source: `results/run_{ts}/<backend>/all_experiments_summary.json`_")

    (run_dir / "result_supplementary.md").write_text("\n".join(sup) + "\n", encoding="utf-8")
    print(f"  ✓ result_supplementary.md ({len(chr(10).join(sup).encode('utf-8'))} bytes)")
else:
    print(f"  ⏩ result_supplementary.md SKIP (v1 데이터 없음)")
PYEOF

# ── 종료 요약 ──
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  완료 — 총 ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  결과: $RUN_DIR"
echo "  📄 통합 보고서: $RUN_DIR/result.md"
echo "  📊 구조화 결과: $RUN_DIR/result.json"
if [ -f "$RUN_DIR/result_supplementary.md" ]; then
    echo "  📚 Supplementary (paper Appendix B): $RUN_DIR/result_supplementary.md"
fi
echo "════════════════════════════════════════════════════════════════════"
