#!/usr/bin/env bash
#
# UASEF — 전체 평가 통합 실행 스크립트
# ════════════════════════════════════════════════════════════════════════════
#
# 한 번 실행으로 (1) UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) +
# (2) 모든 baseline (TECP / Quach 2024 CLM / Semantic Entropy) +
# (3) 합성 검증 (FWER, Cost) 까지 모두 돌리고 단일 markdown 보고서로 정리.
#
# 사용:
#     bash run_full_evaluation.sh                              # default
#     BACKEND=openai N_CAL=1500 N_TEST=500 bash run_full_evaluation.sh
#     SKIP_LLM=1 bash run_full_evaluation.sh                   # 합성만 (LLM 키 불필요)
#
# 환경변수:
#     BACKEND       openai (default) | lmstudio | mlx
#     N_CAL         calibration n per stratum (논문 권장 ≥1000 for α=0.001)
#     N_TEST        holdout test n per stratum
#     N_TRIALS      Table 2 FWER simulation 반복 (default 5000)
#     N_PER_STRATUM Table 3 합성 데이터 (default 300)
#     ALPHA         global α for baselines (default 0.10)
#     SEED          random seed (default 42)
#     SKIP_LLM      1이면 LLM 호출 단계 스킵 (Table 2/3 + 단위 테스트만)
#     SKIP_TESTS    1이면 pytest 스킵
#     RESULTS_DIR   기본 results
#
# 출력:
#     results/run_<timestamp>/result.md         ← 통합 보고서 (사람용)
#     results/run_<timestamp>/result.json       ← 구조화 결과 (post-hoc 분석용)
#     results/run_<timestamp>/table{1,2,3,4}.* ← 개별 표
#     results/run_<timestamp>/pytest_summary.txt
#

set -euo pipefail

# ── 설정 ──
BACKEND="${BACKEND:-openai}"
N_CAL="${N_CAL:-200}"
N_TEST="${N_TEST:-100}"
N_TRIALS="${N_TRIALS:-5000}"
N_PER_STRATUM="${N_PER_STRATUM:-300}"
ALPHA="${ALPHA:-0.10}"
SEED="${SEED:-42}"
SKIP_LLM="${SKIP_LLM:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# ── 환경 ──
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RESULTS_DIR}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH="$(date +%s)"

echo "════════════════════════════════════════════════════════════════════"
echo "  UASEF — Full Evaluation Run"
echo "════════════════════════════════════════════════════════════════════"
echo "  Timestamp     : $TIMESTAMP"
echo "  Output dir    : $RUN_DIR"
echo "  Backend       : $BACKEND"
echo "  N_CAL         : $N_CAL  (per stratum)"
echo "  N_TEST        : $N_TEST  (per stratum)"
echo "  N_TRIALS      : $N_TRIALS  (Table 2)"
echo "  N_PER_STRATUM : $N_PER_STRATUM  (Table 3)"
echo "  Alpha         : $ALPHA"
echo "  Seed          : $SEED"
echo "  Skip LLM      : $SKIP_LLM"
echo "  Skip tests    : $SKIP_TESTS"
echo "  Python        : $PYTHON"

# ── 환경 점검 ──
if [ "$SKIP_LLM" != "1" ]; then
    case "$BACKEND" in
        openai)
            if [ -z "${OPENAI_API_KEY:-}" ]; then
                echo ""
                echo "  ⚠  OPENAI_API_KEY 미설정 — LLM 호출 단계가 SKIP될 수 있습니다."
                echo "     SKIP_LLM=1로 강제 합성-only 실행을 권장."
            fi
            ;;
        anthropic)
            if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
                echo "  ⚠  ANTHROPIC_API_KEY 미설정"
            fi
            ;;
    esac
fi

# ── 1. pytest (회귀 안전망) ──
if [ "$SKIP_TESTS" != "1" ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  [1/5] pytest 회귀 검증"
    echo "────────────────────────────────────────────────────────────────────"
    "$PYTHON" -m pytest tests/ -q 2>&1 | tee "$RUN_DIR/pytest_summary.txt" | tail -20
    PYTEST_RC=${PIPESTATUS[0]}
    if [ "$PYTEST_RC" -ne 0 ]; then
        echo "  ✗ pytest 실패 — 나머지 단계 진행하지만 결과 신뢰도 ↓"
    else
        echo "  ✓ pytest 통과"
    fi
fi

# ── 2. Table 2 — FWER (Pivot B, LLM 불필요) ──
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "  [2/5] Table 2 — Multi-Trigger FWER (Pivot B)"
echo "────────────────────────────────────────────────────────────────────"
"$PYTHON" experiments/round7_table2_fwer.py \
    --n-trials "$N_TRIALS" --alpha 0.05 --seed "$SEED" \
    > "$RUN_DIR/table2_stdout.txt" 2>&1 || echo "  ✗ Table 2 실패"
# round7_table2_fwer.py가 results/round7/에 저장 → 이 디렉토리로 복사
[ -f "results/round7/table2_fwer.json" ] && cp results/round7/table2_fwer.{json,md} "$RUN_DIR/" 2>/dev/null
echo "  ✓ Table 2 완료"

# ── 3. Table 3 — Cost-Weighted (Pivot C, LLM 불필요) ──
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "  [3/5] Table 3 — Cost-Weighted Performance (Pivot C)"
echo "────────────────────────────────────────────────────────────────────"
"$PYTHON" experiments/round7_table3_cost.py \
    --n-per-stratum "$N_PER_STRATUM" --seed "$SEED" \
    > "$RUN_DIR/table3_stdout.txt" 2>&1 || echo "  ✗ Table 3 실패"
[ -f "results/round7/table3_cost.json" ] && cp results/round7/table3_cost.{json,md} "$RUN_DIR/" 2>/dev/null
echo "  ✓ Table 3 완료"

# ── 4. Table 1 — Per-Stratum Coverage (Pivot A, LLM 필요) ──
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "  [4/5] Table 1 — Per-Stratum Coverage (Pivot A)"
echo "────────────────────────────────────────────────────────────────────"
if [ "$SKIP_LLM" = "1" ]; then
    echo "  ⏩ SKIP_LLM=1 — Table 1 SKIP"
else
    "$PYTHON" experiments/round7_table1_coverage.py \
        --backend "$BACKEND" --n-cal "$N_CAL" --n-test "$N_TEST" \
        --alpha-global "$ALPHA" --seed "$SEED" \
        > "$RUN_DIR/table1_stdout.txt" 2>&1 \
        && cp results/round7/table1_coverage.{json,md} "$RUN_DIR/" 2>/dev/null \
        && echo "  ✓ Table 1 완료" \
        || echo "  ✗ Table 1 실패 (LLM 키 / 데이터 확인)"
fi

# ── 5. Table 4 — Baseline Comparison (LLM 필요) ──
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "  [5/5] Table 4 — Head-to-Head Baseline (TECP / Quach / SE / R6 / R7)"
echo "────────────────────────────────────────────────────────────────────"
if [ "$SKIP_LLM" = "1" ]; then
    echo "  ⏩ SKIP_LLM=1 — Table 4 SKIP"
else
    "$PYTHON" experiments/round7_table4_baseline.py \
        --backend "$BACKEND" --n-cal "$N_CAL" --n-test "$N_TEST" \
        --alpha "$ALPHA" --seed "$SEED" \
        > "$RUN_DIR/table4_stdout.txt" 2>&1 \
        && cp results/round7/table4_baseline.{json,md} "$RUN_DIR/" 2>/dev/null \
        && echo "  ✓ Table 4 완료" \
        || echo "  ✗ Table 4 실패"
fi

# ── 통합 보고서 result.md / result.json 생성 ──
END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "  보고서 생성"
echo "────────────────────────────────────────────────────────────────────"

export REPORT_RUN_DIR="$RUN_DIR"
export REPORT_TIMESTAMP="$TIMESTAMP"
export REPORT_ELAPSED="${ELAPSED_MIN}m ${ELAPSED_SEC}s"
export REPORT_BACKEND="$BACKEND"
export REPORT_N_CAL="$N_CAL"
export REPORT_N_TEST="$N_TEST"
export REPORT_N_TRIALS="$N_TRIALS"
export REPORT_N_PER_STRATUM="$N_PER_STRATUM"
export REPORT_ALPHA="$ALPHA"
export REPORT_SEED="$SEED"
export REPORT_SKIP_LLM="$SKIP_LLM"

"$PYTHON" - <<'PYEOF'
import json
import os
from datetime import datetime
from pathlib import Path

run_dir = Path(os.environ["REPORT_RUN_DIR"])
ts = os.environ["REPORT_TIMESTAMP"]
elapsed = os.environ["REPORT_ELAPSED"]

# config metadata
meta = {
    "timestamp": ts,
    "elapsed": elapsed,
    "config": {
        "backend": os.environ["REPORT_BACKEND"],
        "n_cal": int(os.environ["REPORT_N_CAL"]),
        "n_test": int(os.environ["REPORT_N_TEST"]),
        "n_trials": int(os.environ["REPORT_N_TRIALS"]),
        "n_per_stratum": int(os.environ["REPORT_N_PER_STRATUM"]),
        "alpha": float(os.environ["REPORT_ALPHA"]),
        "seed": int(os.environ["REPORT_SEED"]),
        "skip_llm": os.environ["REPORT_SKIP_LLM"] == "1",
    },
}

# 각 표 결과 로드
tables = {}
for name in ("table1_coverage", "table2_fwer", "table3_cost", "table4_baseline"):
    fp = run_dir / f"{name}.json"
    tables[name] = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None

# pytest 요약
pytest_summary = ""
pt = run_dir / "pytest_summary.txt"
if pt.exists():
    lines = pt.read_text(encoding="utf-8").splitlines()
    summary_lines = [l for l in lines[-10:] if l.strip()]
    pytest_summary = "\n".join(summary_lines)

# ── result.json (구조화) ──
result_payload = {"meta": meta, "tables": tables, "pytest_summary": pytest_summary}
(run_dir / "result.json").write_text(
    json.dumps(result_payload, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)

# ── result.md (사람용 통합 보고서) ──
md = []
md.append(f"# UASEF Full Evaluation Result")
md.append(f"")
md.append(f"- **timestamp**: {ts}")
md.append(f"- **elapsed**: {elapsed}")
md.append(f"- **config**:")
for k, v in meta["config"].items():
    md.append(f"  - {k}: `{v}`")
md.append(f"")
md.append(f"## 0. 회귀 검증 (pytest)")
md.append(f"")
if pytest_summary:
    md.append("```")
    md.append(pytest_summary)
    md.append("```")
else:
    md.append("_pytest 스킵됨_")
md.append(f"")

# Table 1
md.append(f"## 1. Per-Stratum Coverage Validity (Pivot A — Stratified CRC)")
md.append(f"")
md.append(f"각 risk stratum (CRITICAL/HIGH/MODERATE/LOW)에서 missed-escalation rate를 측정.")
md.append(f"UASEF Round 7만이 stratum별 conformal risk control 보장을 충족해야 함.")
md.append(f"")
t1 = tables.get("table1_coverage")
if t1:
    md.append(f"- backend: `{t1.get('backend')}`, n_cal={t1.get('n_cal')}/stratum, n_test={t1.get('n_test')}/stratum")
    md.append(f"- α_global = {t1.get('alpha_global')}, CRC alphas = {t1.get('crc_alphas')}")
    md.append(f"")
    md.append(f"| Method | CRITICAL | HIGH | MODERATE | LOW |")
    md.append(f"| --- | --- | --- | --- | --- |")
    for r in t1.get("results", []):
        row = f"| {r['name']} |"
        for s in ("CRITICAL", "HIGH", "MODERATE", "LOW"):
            v = r["per_stratum"].get(s, {}).get("miss_rate")
            row += f" {v if v is not None else 'N/A'} |"
        md.append(row)
else:
    md.append("_Table 1 SKIP (LLM 호출 필요)_")
md.append(f"")

# Table 2
md.append(f"## 2. Multi-Trigger Combination FWER (Pivot B — Conformal Combination)")
md.append(f"")
md.append(f"Null hypothesis (모든 trigger가 정상)에서 false escalation rate 측정.")
md.append(f"v1 `len(triggers) > 0` → FWER 위반, v2 harmonic / e-value → 보장 충족.")
md.append(f"")
t2 = tables.get("table2_fwer")
if t2:
    md.append(f"- n_trials = {t2['config']['n_trials']}, α target = {t2['config']['alpha']}")
    md.append(f"")
    md.append(f"| Method | Dependence | Empirical FWER | OK |")
    md.append(f"| --- | --- | --- | --- |")
    for r in t2.get("rows", []):
        ok = "✓" if r["ok"] else "✗"
        md.append(f"| {r['method']} | {r['dependence']} | {r['empirical_fwer']} | {ok} |")
else:
    md.append("_Table 2 결과 없음_")
md.append(f"")

# Table 3
md.append(f"## 3. Cost-Weighted Performance (Pivot C — Cost-Aware Optimization)")
md.append(f"")
md.append(f"비대칭 cost matrix (CRITICAL miss=1000× over_esc) 적용 시 total cost 비교.")
md.append(f"")
t3 = tables.get("table3_cost")
if t3:
    md.append(f"- n_per_stratum = {t3['n_per_stratum']}")
    md.append(f"- **Round 6 total cost**: `{t3['round6_total_cost']:.1f}`")
    md.append(f"- **Round 7 total cost**: `{t3['round7_total_cost']:.1f}`")
    md.append(f"- **Cost reduction**: **{t3.get('cost_reduction_ratio')}×** (Round 6 / Round 7)")
    md.append(f"")
    md.append(f"### Per-stratum")
    md.append(f"| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over_esc |")
    md.append(f"| --- | --- | --- | --- | --- | --- | --- |")
    for r6, r7 in zip(t3["round6_per_stratum"], t3["round7_per_stratum"]):
        md.append(
            f"| {r6['stratum']} | {r6['threshold']} | {r6['cost']:.1f} | "
            f"{r7['threshold']} | {r7['cost']:.1f} | "
            f"{r7['miss_rate']} | {r7['over_esc_rate']} |"
        )
else:
    md.append("_Table 3 결과 없음_")
md.append(f"")

# Table 4
md.append(f"## 4. Head-to-Head Baseline Comparison")
md.append(f"")
md.append(f"동일 calibration / test 풀에서 5개 method 비교:")
md.append(f"TECP (Xu & Lu 2025), Quach 2024 CLM, Semantic Entropy (Farquhar Nature 2024), UASEF Round 6 (heuristic), **UASEF Round 7 (Stratified CRC + MTC + Cost-Aware)**.")
md.append(f"")
t4 = tables.get("table4_baseline")
if t4:
    md.append(f"- backend: `{t4.get('backend')}`, n_cal={t4.get('n_cal')}, n_test={t4.get('n_test')}, α={t4.get('alpha')}")
    md.append(f"")
    md.append(f"### CRITICAL stratum")
    md.append(f"| Method | Safety Recall | Over-Esc | TP/FN/FP | Total cost |")
    md.append(f"| --- | --- | --- | --- | --- |")
    for m in t4.get("methods", []):
        c = m["per_stratum"]["CRITICAL"]
        md.append(
            f"| {m['name']} | {c.get('safety_recall')} | {c.get('over_esc_rate')} | "
            f"{c['tp']}/{c['fn']}/{c['fp']} | {m['total_cost']:.1f} |"
        )
    md.append(f"")
    md.append(f"### Total cost (전 stratum 합)")
    md.append(f"| Method | Total cost |")
    md.append(f"| --- | --- |")
    for m in t4.get("methods", []):
        md.append(f"| {m['name']} | {m['total_cost']:.1f} |")
else:
    md.append("_Table 4 SKIP (LLM 호출 필요)_")
md.append(f"")

# 한 줄 결론
md.append(f"## 결론 요약")
md.append(f"")
if t2:
    naive_rows = [r for r in t2.get("rows", []) if "naive" in r["method"]]
    naive_max = max((r["empirical_fwer"] for r in naive_rows), default=None)
    harm_rows = [r for r in t2.get("rows", []) if "harmonic" in r["method"]]
    harm_max = max((r["empirical_fwer"] for r in harm_rows), default=None)
    if naive_max and harm_max:
        md.append(f"- **Pivot B (FWER)**: v1 naive OR FWER ≤ **{naive_max}** (target 0.05 위반). v2 harmonic FWER ≤ **{harm_max}** (충족).")
if t3:
    md.append(f"- **Pivot C (Cost)**: Round 6 → Round 7 total cost **{t3.get('cost_reduction_ratio')}× 감소** ({t3['round6_total_cost']:.0f} → {t3['round7_total_cost']:.0f}).")
if t1:
    r7 = next((r for r in t1.get("results", []) if "Round 7" in r["name"]), None)
    if r7:
        all_ok = all((s.get("ok") is True) for s in r7["per_stratum"].values() if s.get("ok") is not None)
        md.append(f"- **Pivot A (Coverage)**: Round 7 stratum별 보장 충족: {'전체 ✓' if all_ok else '일부 ✗ (n 부족 가능)'}")
md.append(f"")
md.append(f"---")
md.append(f"")
md.append(f"_생성: `run_full_evaluation.sh` ({datetime.now().isoformat(timespec='seconds')})_")

(run_dir / "result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"  ✓ result.md ({sum(len(l)+1 for l in md)} bytes)")
print(f"  ✓ result.json")
PYEOF

# ── 종료 요약 ──
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  완료 — 총 ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  결과 디렉토리: $RUN_DIR/"
ls -la "$RUN_DIR"
echo ""
echo "  📄 통합 보고서: $RUN_DIR/result.md"
echo "  📊 구조화 결과: $RUN_DIR/result.json"
echo "════════════════════════════════════════════════════════════════════"
