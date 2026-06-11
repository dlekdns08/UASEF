# Round 10 — Execution Runbook

> **TL;DR — 3 phase execution**
> ```bash
> # Phase 1 (infrastructure + preprocessing, ~6h)
> export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
> export UASEF_BACKEND_NEVER_SEND_PHI=1
> SKIP_R10_1=1 SKIP_R10_2=1 SKIP_R10_3=1 SKIP_R10_4=1 SKIP_R10_5=1 SKIP_R10_6=1 \
>   bash run_all_round10.sh
>
> # Phase 2 (experiments, ~18 days wallclock with LLM; ~6h without LLM)
> SKIP_PREPROCESS=1 SKIP_R10_0=1 SKIP_R10_5=1 \
>   bash run_all_round10.sh
>
> # Phase 3 (aggregation + paper, ~30 min)
> SKIP_PREPROCESS=1 SKIP_R10_0=1 SKIP_R10_1=1 SKIP_R10_2=1 \
> SKIP_R10_3=1 SKIP_R10_4=1 SKIP_R10_5=1 SKIP_R10_6=1 SKIP_R10_7=1 \
>   bash run_all_round10.sh
> ```
>
> plan: [round10_PLAN.md](round10_PLAN.md)

---

## 0. 사전 점검

### 0.1 데이터 (사용자 로컬)

```bash
ls $MIMIC4_DIR/hosp/{admissions,patients,services,diagnoses_icd,labevents,d_labitems}.csv.gz
ls $MIMIC4_DIR/icu/{icustays,chartevents}.csv.gz
# chartevents 가 새로 필요 — R10.7 의 vital sign quartile feature 산출용
```

### 0.2 venv 의존성 (uv-managed)

```bash
uv pip install scikit-learn xgboost   # R10.4 의 tabular baseline
uv pip install scipy                  # R10.5 의 Cohen's κ
```

### 0.3 LMStudio 확인

R10.1, R10.2, R10.4 (LLM branch), R10.7 가 LLM 호출 필요:

```bash
curl -s http://localhost:1234/v1/models | jq .data[].id
# "openai/gpt-oss-120b" 가 보여야 함
```

**LLM 백엔드는 `openai/gpt-oss-120b` 만 사용** (사용자 결정 2026-06-11).
size-scaling 비교를 위한 gpt-oss-20b 추가는 Round 10 에서 제외;
within-LLM scaling 연구는 future work (paper §7.2 L24 참조).

### 0.4 IRB / Physician 조정 (R10.5)

R10.5 는 LLM 실험과 독립적인 4주 process — 미리 시작:

```bash
# IRB review 신청
# 3 physician 섭외 (응급, 내과, 가정의학)
# Compensation: USD 80/hr × ~10hr × 3 = $2,400
```

---

## 1. Phase 1 — Infrastructure (R10.0) + Preprocessing (R10.7)

### Step 1.1 R10.0 — Multi-seed infrastructure

```bash
.venv/bin/python -m pytest tests/test_round10_aggregate.py -v
# bootstrap CI helper 가 통계적으로 올바른지 검증
```

### Step 1.2 R10.7 — Expanded decision-time features preprocessing

```bash
.venv/bin/python experiments/round10_mimic4_preprocess.py \
    --mimic-dir "$MIMIC4_DIR" \
    --output data/raw/mimic-iv/mimic4_cases_v10.jsonl \
    --n-per-stratum 3500 \
    --seed 42
# CRITICAL ≥ 3000 보장 (n_pos 부족 strata 는 가용한 만큼)
# 새 feature: charlson_index, vital_HR_Q4, vital_BP_Q1, specialty_admit_rate
```

산출:
- `data/raw/mimic-iv/mimic4_cases_v10.jsonl` (commit 금지 — DUA)
- Stratum 별 ICD/lab/vital feature 분포 통계 (commit 가능)

---

## 2. Phase 2 — Experiments (R10.1-R10.6)

### Step 2.1 R10.1 — Properly-powered α=0.05 empirical

```bash
.venv/bin/python experiments/round10_alpha_005_empirical.py \
    --n-cal 3000 --n-test 3000 \
    --alphas CRITICAL=0.05 HIGH=0.10 MODERATE=0.15 LOW=0.20 \
    --seeds 42 43 44 45 46 \
    --backends lmstudio \
    --out results/round10/r10_1_alpha_005_empirical
# Wallclock: ~120h on Mac Studio LMStudio (6000 calls × 5 seed × ~11s)
```

### Step 2.2 R10.2 — Multi-seed Table 4-MIMIC

```bash
.venv/bin/python experiments/round10_table4_multiseed.py \
    --n-cal-per-stratum 200 --n-test-per-stratum 100 \
    --alpha 0.10 --seeds 42 43 44 45 46 \
    --backends lmstudio \
    --out results/round10/r10_2_table4_multiseed
# Wallclock: ~25h (1200 calls × 5 seed)
```

### Step 2.3 R10.3 — Distribution shift mitigation

```bash
.venv/bin/python experiments/round10_distshift_mitigation.py \
    --strategies online_recal kmm group_conditional \
    --temporal-cal "2008 - 2010,2011 - 2013,2014 - 2016" \
    --temporal-test "2017 - 2019,2020 - 2022" \
    --specialty-source cardiology \
    --specialty-targets neurology,internal_medicine,surgery \
    --seeds 42 43 44 45 46 \
    --out results/round10/r10_3_mitigation
```

### Step 2.4 R10.4 — Method-agnostic CRC head-to-head (HEADLINE)

```bash
.venv/bin/python experiments/round10_method_agnostic.py \
    --classifiers gpt_oss_120b logreg gbdt randomforest xgboost \
    --n-cal 3000 --n-test 3000 \
    --seeds 42 43 44 45 46 \
    --out results/round10/r10_4_method_agnostic
# LLM (gpt-oss-120b 만): ~120h
# Tabular (4): ~30s total (sklearn 즉시)
# Wallclock dominate 는 LLM
```

### Step 2.5 R10.5 — IRB physician adjudication (별도 4주 process)

이 단계는 코드가 아니라 사람 작업. R10.5 의 결과 JSON 이 생기면 후속 처리:

```bash
.venv/bin/python experiments/round10_physician_audit.py \
    --physician-labels data/raw/physician_audit_2026-08.jsonl \
    --outcome-labels data/raw/mimic-iv/mimic4_cases_v10.jsonl \
    --out results/round10/r10_5_physician_audit
# Cohen's κ + confusion matrix 산출
```

### Step 2.6 R10.6 — 4-D cost matrix sweep

```bash
.venv/bin/python experiments/round10_cost_sweep_4d.py \
    --grid 10 100 1000 \
    --stratum CRITICAL \
    --seed 42 \
    --out results/round10/r10_6_cost_sweep_4d
# CPU-only, 6시간 wallclock
```

---

## 3. Phase 3 — Aggregate + Paper

### Step 3.1 통합 보고서

```bash
.venv/bin/python experiments/round10_aggregate_report.py \
    --in-dir results/round10 \
    --out results/round10/ROUND10_FINAL_REPORT.md
```

### Step 3.2 Paper revision

```bash
# paper/UASEF_Round10.md 의 _to be filled_ 를 실제 수치로 교체
.venv/bin/python experiments/round10_sync_paper.py \
    --results-dir results/round10 \
    --paper paper/UASEF_Round10.md
```

---

## 4. 한 줄 통합 실행

```bash
bash run_all_round10.sh                  # 전체 ~18일 wallclock
SKIP_PREPROCESS=1 bash run_all_round10.sh  # preprocessing 이미 완료 시
DRY_RUN=1 bash run_all_round10.sh         # 명령만 출력
```

`run_all_round10.sh` 환경변수:

- `SKIP_PREPROCESS`, `SKIP_R10_0` (infra), `SKIP_R10_1` ~ `SKIP_R10_7`, `SKIP_AGGREGATE`
- `STRICT_FAIL=1` — 한 단계 실패 시 중단
- `DRY_RUN=1` — 명령만 출력
- `SEEDS="42 43 44 45 46"` — 시드 (default 5 seed)
- `BACKENDS="lmstudio"` — Round 9 와 동일 (default lmstudio only)

---

## 5. 산출물 위치

```
results/round10/
├── r10_1_alpha_005_empirical.{json,md}      # Powered α=0.05
├── r10_2_table4_multiseed.{json,md}         # 5-seed Table 4-MIMIC
├── r10_3_mitigation.{json,md}               # 3 strategy 비교
├── r10_4_method_agnostic.{json,md}          # HEADLINE: 5 classifier × CRC
├── r10_5_physician_audit.{json,md}          # Cohen's κ
├── r10_6_cost_sweep_4d.{json,md}            # 81 조합
├── r10_7_feature_expansion_validation.{json,md}  # MODERATE/LOW 개선 효과
└── ROUND10_FINAL_REPORT.md                  # 통합

data/raw/mimic-iv/
├── mimic4_cases.jsonl                       # Round 9 (구) — 보존
└── mimic4_cases_v10.jsonl                   # Round 10 (확장 feature)
```

---

## 6. 트러블슈팅

| 증상 | 해결 |
|---|---|
| LMStudio 가 18일 wallclock 동안 hibernate | `caffeinate -dimsu bash run_all_round10.sh` |
| `xgboost` install 실패 | macOS arm64: `brew install libomp && uv pip install xgboost` |
| R10.1 의 n_test=3000 으로 wallclock 너무 김 | `--n-test 1500` 으로 절반, α=0.05 의 exact upper 조금 헐거워짐 |
| R10.4 의 sklearn classifier 가 score 변환 실패 | `predict_proba` 의 positive class 확인 — `classes_` attribute 점검 |
| R10.5 의 physician κ < 0.5 | paper 에 finding 으로 정직 보고, "outcome label 이 더 보수적" 으로 framing |
| R10.7 의 vital quartile 산출이 너무 느림 | chartevents 의 첫 6h 만 chunked read; 또는 `polars` 로 전환 |

---

## 7. Round 9 와의 차이 요약

| 측면 | Round 9 | Round 10 |
|---|---|---|
| Seed | 단일 (42) | 5 seed bootstrap CI |
| α | 0.001 (n_pos 부족 → 증명 fail) | 0.05 (n_pos 충분 → 진짜 증명) |
| Classifier | LLM only | 5 classifier × CRC (LLM + 4 tabular) |
| Shift | Detection only | Detection + 3 mitigation strategy |
| Stratum 라벨 | outcome-derived | + IRB physician 100-case audit |
| Cost matrix | 1-D sensitivity | 4-D 81 조합 sweep on real data |
| Decision-time feature | admission_type, age, service, lab flag | + Charlson, vital quartile, specialty rate |
| Wallclock | 34h | ~18 일 (LLM dominate) |
| Headline | "v2 가 single-α 대비 10× recall" | "method-agnostic CRC framework — LLM/tabular 동등" |

---

_Round 10 RUNBOOK 작성: 2026-06-11. Plan: [round10_PLAN.md](round10_PLAN.md)._
