# Round 9 — Execution Runbook (MIMIC-IV Integration)

> ⚠️ **개정 (2026-06-10) — leakage-safe 재설계.** preprocessing 이 decision-time 위험군
> $G(X_{t_0})$ 과 독립적 미래 outcome $Y$ 를 분리하도록 바뀌었으므로 **반드시 JSONL 을 새로
> 생성**해야 한다(구 JSONL 폐기: `rm -f data/raw/mimic-iv/mimic4_cases.jsonl` 후 Step 1).
> split 은 patient-level(subject_id), α=0.001 은 exact 이항 상한으로 보고된다.
> 근거: [paper/REVISION_PLAN.md](../paper/REVISION_PLAN.md). 신규 단계 **R9.6 = tabular baseline**
> (LLM 무관, 아래 §1 Step 3b). (구 RUNBOOK 의 "R9.6 LLM-judge κ" 는 Phase 2 의 R9.7 로 재배치.)

> **TL;DR — local-only headline**
> ```bash
> export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
> export UASEF_BACKEND_NEVER_SEND_PHI=1
> bash run_all_round9.sh                  # default: lmstudio only, $0, ~3-5h
> ```
>
> **변경 (2026-05-11)**: BACKENDS default 가 `"lmstudio"` 만 (외부 API 제외).
> PhysioNet DUA 보수적 해석 + 실 병원 deployment 와 일치하는 framing.
> OpenAI 비교 필요 시: `BACKENDS="openai lmstudio" bash run_all_round9.sh`
> (단, `UASEF_BACKEND_NEVER_SEND_PHI=1` 도 unset 해야 송신 가능 — guard 가
> MIMIC-IV case 의 외부 API 송신을 차단).
>
> 자세한 plan: [round9_PLAN.md](round9_PLAN.md)

---

## 0. 사전 점검

### 0.1 데이터 (사용자 로컬, ~/Downloads/mimic-iv-3.1/)

```bash
ls $MIMIC4_DIR/hosp/admissions.csv.gz $MIMIC4_DIR/icu/icustays.csv.gz \
   $MIMIC4_DIR/hosp/services.csv.gz $MIMIC4_DIR/hosp/diagnoses_icd.csv.gz
# 4 파일 모두 존재해야 Phase 1 통과
```

### 0.2 PhysioNet credential 확인

- [ ] CITI 교육 이수증 (PhysioNet account 페이지에서 확인)
- [ ] DUA 서명 완료 (`LICENSE.txt` 동봉됨)
- [ ] **commit 금지** 확인 — `.gitignore` 에 `data/raw/mimic-iv/` 블록됨

### 0.3 PHI guard env

```bash
export UASEF_BACKEND_NEVER_SEND_PHI=1   # 필수
```

이 변수가 켜져 있을 때 `models/model_interface.py` 는 case `source` 가
`mimic4_note*` 로 시작하면 OpenAI/Anthropic 백엔드 호출을 거부합니다.
Phase 1 의 structured proxy (`source=mimic4_struct`) 는 OpenAI 송신 허용.

---

## 1. Phase 1 — Structured Data (hosp + icu only)

### Step 1: Preprocessing (수동, 한 번만)

```bash
# 입력: $MIMIC4_DIR/{hosp,icu}/*.csv.gz
# 출력: data/raw/mimic-iv/mimic4_cases.jsonl  (~50 MB)
.venv/bin/python experiments/round9_mimic4_preprocess.py \
    --mimic-dir "$MIMIC4_DIR" \
    --n-per-stratum 1500 \
    --output data/raw/mimic-iv/mimic4_cases.jsonl \
    --seed 42
```

기본 6000 case (CRITICAL/HIGH/MOD/LOW × 1500). CSV.gz chunked read 라
~80 GB 임에도 메모리 < 4 GB. CPU bound, ~2h.

### Step 2: 빠른 검증 (LLM 호출 없음, $0)

```bash
.venv/bin/pytest tests/test_mimic4_loader.py -v
# data 없으면 graceful skip, 있으면 schema 검증
```

### Step 3: R9.1 — α=0.001 empirical (Table 1c)

```bash
.venv/bin/python experiments/round9_alpha_critical_real.py \
    --n-critical 1500 --alpha-critical 0.001 \
    --seeds 42 43 44 45 46 \
    --backends openai lmstudio \
    --out results/round9/alpha_critical_real
# 산출: alpha_critical_real.{json,md}
```

목표: per-example loss `E[ℓ_CRITICAL] ≤ 0.001 + 2σ` 만족 확인.

### Step 4: R9.2 — Table 4-MIMIC head-to-head

```bash
.venv/bin/python experiments/round9_table4_mimic.py \
    --n-test 100 --alpha 0.10 \
    --seeds 42 43 44 45 46 \
    --backends openai lmstudio \
    --out results/round9/table4_mimic
```

8 method (TECP / Quach / SE / R6 / R7 / TECP-strat / Cost-Sens single-α / v1-cost-aware) × 2 backend × 5 seed.

### Step 5: R9.3 — Distribution shift (real)

```bash
.venv/bin/python experiments/round9_distribution_shift.py \
    --calibrate-on cardiology \
    --test-on neurology general_medicine surgery \
    --weighted-cp \
    --seeds 42 43 44 45 46 \
    --out results/round9/dist_shift_real
```

### Step 6: R9.4 — Temporal shift

```bash
.venv/bin/python experiments/round9_temporal_shift.py \
    --calibrate-years 2008-2014 \
    --test-years 2015-2019 \
    --seeds 42 43 44 45 46 \
    --out results/round9/temporal_shift
```

### Step 7: R9.5 — Demographic equity

```bash
.venv/bin/python experiments/round9_equity_real.py \
    --groupby race sex age_bucket \
    --seed 42 \
    --out results/round9/equity_audit_real
```

1 시드 (단일 분포 보고), demographic stratification 만 변동.

### Step 8: 통합 보고서

```bash
.venv/bin/python experiments/round9_aggregate_report.py \
    --in-dir results/round9 \
    --out results/round9/round9_report.md
```

---

## 2. 한 줄 실행 (모든 단계)

```bash
export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1
bash run_all_round9.sh
```

`run_all_round9.sh` 환경변수:
- `SKIP_PREPROCESS=1` — JSONL 이미 있으면
- `SKIP_R9_1=1`, `SKIP_R9_2=1`, ... — 단계별 건너뛰기
- `STRICT_FAIL=1` — 한 단계라도 실패하면 중단
- `DRY_RUN=1` — 명령만 출력

---

## 3. 산출물 위치

```
results/round9/
├── alpha_critical_real.{json,md}        # R9.1 — Table 1c
├── table4_mimic.{json,md}               # R9.2 — Table 4-MIMIC
├── dist_shift_real.{json,md}            # R9.3 — §G 강화
├── temporal_shift.{json,md}             # R9.4
├── equity_audit_real.{json,md}          # R9.5 — §I 강화
└── round9_report.md                     # 통합

data/raw/mimic-iv/                       # ⚠️ commit 금지
└── mimic4_cases.jsonl                   # preprocessing 산출
```

---

## 4. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `FileNotFoundError: $MIMIC4_DIR/hosp/admissions.csv.gz` | 환경변수 미설정 또는 압축 해제 안 됨 |
| `[PHI guard] OpenAI 백엔드 거부` | 정상. Phase 2 free-text 는 lmstudio 만 |
| preprocessing 이 4h 이상 | CSV.gz chunked read 실패. `pandas` 또는 `polars` 메모리 점검 |
| α=0.001 violation | 실 EHR noise. paper §7.2 limitation 표현 약화로 fallback (plan §7 참조) |
| `services` 컬럼 없음 | MIMIC-IV-3.1 의 `hosp/services.csv.gz` 필요. 다운로드 확인 |

---

## 5. Phase 2 (옵션, PhysioNet 추가 승인 필요)

PhysioNet 에서 **MIMIC-IV-Note v2.2** 와 **MIMIC-IV-ED v2.2** 별도 신청·승인 후:

```bash
export MIMIC4_NOTE_DIR=~/Downloads/mimic-iv-note-2.2
export MIMIC4_ED_DIR=~/Downloads/mimic-iv-ed-2.2
bash run_all_round9.sh --phase2
```

추가 단계:
- R9.7 LLM-judge κ on MIMIC discharge note (n=100, lmstudio only) ✏️ (구 R9.6 → R9.6 은 tabular 로 재배치)
- R9.8 ESI proxy validation — Phase 1 의 risk-group 가정이 ESI≤2 와 얼마나 일치하는지

---

_Round 9 RUNBOOK 작성: 2026-05-10. plan 의존: improvements/round9_PLAN.md._
