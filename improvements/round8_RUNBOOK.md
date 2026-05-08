# Round 8 — Execution Runbook

> **TL;DR — 한 줄 명령**
> ```bash
> bash run_all_round8.sh        # 전체 14 단계, ~$205 + ~3시간
> ```
> SKIP_*, STRICT_FAIL, DRY_RUN 등 환경변수로 부분 실행 / what-if 가능.
> 자세한 옵션은 `run_all_round8.sh` 헤더 주석 참조.
>
> **상태**: Round 8 코드/문서 변경은 모두 commit 가능한 상태로 stage됨.
> **이 문서는 master script `run_all_round8.sh`의 14개 단계를 풀어 설명하고,
> expensive run을 트리거하기 전 점검 사항을 정리한다.**

---

## 0. 변경 요약 (코드 + 문서, 모두 완료)

### 신규 파일
| 파일 | 역할 |
|---|---|
| `data/download_datasets.sh` | MedQA + MedAbstain raw JSONL 다운로드 + 검증 |
| `experiments/llm_judge_relabel.py` | gpt-5.5 + claude-opus-4-7 self-consistency relabeling |
| `experiments/run_multidataset_generalization.py` | MedAbstain + MedQA-USMLE + PubMedQA 3-dataset sweep |
| `experiments/round8_pivotB_case_study.py` | m ∈ {3,5,8,12} FWER + 기관 customization (m=8) |
| `experiments/round8_distribution_shift.py` | specialty-mismatch coverage violation audit |
| `experiments/round8_multilingual_sanity.py` | 영어 vs zh 교차 sanity (data 없으면 graceful skip) |
| `experiments/round8_equity_audit.py` | per-stratum AUROC/recall variance 진단 |
| `tests/test_paper_claims.py` | 페이퍼 핵심 수치 regression guard (5/9 pass, 4 LLM-skip) |
| `improvements/round8_PLAN.md` | Round 8 마스터 플랜 |

### 수정된 파일
| 파일 | 변경 |
|---|---|
| `data/loader.py` | `UASEF_PAPER_REPRODUCTION=1` 추가 — fallback 절대 차단 |
| `experiments/aggregate_multiseed.py` | seed-pooled McNemar (binom + Fisher combined) |
| `experiments/round7_table4_baseline.py` | mcnemar `{b, c, n, p_value}` emit |
| `run_full_evaluation.sh` | paper-reproduction guard + Tables 1/4를 results/round7/에 latest copy 유지 |
| `paper/UASEF_Round7.md` | §1.3 contribution 1에 α=0.001 caveat, "137" → "140" |
| `paper/UASEF_Round7_KO.md` | "137" → "140" |
| `paper/UASEF_Round7_Supplementary.md` | §A Theorem 1 full proof, §C LLM-judge, §D multi-dataset, §E Pivot B 케이스, §F HCUP cost 정당화, §G shift, §H multi-lingual, §I equity |
| `paper/IRB_PROTOCOL.md` | §7.1 LLM-judge fallback 절차 명시 |

---

## 1. Phase 1 — 출판 차단 항목 해결 (4주, 비용 ~$140)

### Step 1: MedAbstain 데이터 확보 (사람 손이 필요)
```bash
bash data/download_datasets.sh
# MedQA는 자동 다운로드. MedAbstain은 라이선스 사유로 매뉴얼 단계 출력됨.
# 출력 안내에 따라 medabstain_{AP,NAP,A,NA}.jsonl을 data/raw/에 위치시킨다.
```

검증:
```bash
ls data/raw/medabstain_*.jsonl
# AP, NAP, A, NA 4개 파일 모두 존재해야 함
```

### Step 2: 페이퍼 재현 모드로 dry-run (빠른 검증, ~5분, $0)
```bash
UASEF_PAPER_REPRODUCTION=1 SKIP_LLM=1 bash run_full_evaluation.sh
# pytest 140 + Table 2/3 (synthetic, no LLM calls)
# fallback 자동 차단 — data/raw 비면 즉시 fail
```

### Step 3: 단일 seed full run (~$25 + 10분 LMStudio)
```bash
UASEF_PAPER_REPRODUCTION=1 SEED=42 BACKENDS="openai lmstudio" \
    bash run_full_evaluation.sh
# 기존 single-seed 결과 재현. results/run_<ts>/ + results/round7/table*_<backend>.json
```

### Step 4: Multi-seed (5 seeds, ~$125 + 50분)
```bash
UASEF_PAPER_REPRODUCTION=1 SEEDS="42 43 44 45 46" \
    BACKENDS="openai lmstudio" \
    bash run_multiseed_evaluation.sh
# results/run_<ts>_aggregate/aggregate_seeds.{json,md}
# - Table 1·4 mean ± std + 95% bootstrap CI per metric
# - seed-pooled McNemar (binom + Fisher combined p)
```

### Step 5: LLM-judge relabeling (~$10 + $15 + 30분)
```bash
mkdir -p results/round8
.venv/bin/python experiments/llm_judge_relabel.py \
    --n 200 --seed 42 \
    --judges openai anthropic \
    --openai-model gpt-5.5 \
    --anthropic-model claude-opus-4-7 \
    --out results/round8/llm_judge_relabel.json
# Cohen's κ ≥ 0.7면 supplementary §C에 보조 ground truth로 인용 가능.
```

### Step 6: 페이퍼 §6 표 갱신
```bash
# aggregate_seeds.md 결과를 paper/UASEF_Round7.md §6.1·§6.4에 mean±std±CI 형태로 반영
# tests/test_paper_claims.py가 backend snapshot 활성화되어 9/9 pass 확인
.venv/bin/python -m pytest tests/test_paper_claims.py -q
```

---

## 2. Phase 2 — 강한 리뷰 지적 보강 (3주, 비용 ~$40)

### Step 7: Multi-dataset 일반화 (~$30 + 1시간)
```bash
.venv/bin/python experiments/run_multidataset_generalization.py \
    --backend openai \
    --datasets medabstain medqa_usmle pubmedqa \
    --n-cal 200 --n-test 100 --seed 42 \
    --out results/round8/multidataset_summary.json
# supplementary §D에 인용
```

### Step 8: Pivot B case study ($0, 5분)
```bash
.venv/bin/python experiments/round8_pivotB_case_study.py \
    --n-trials 5000 --alpha 0.05 --seed 42 \
    --out results/round8/pivotB_case_study.json
# supplementary §E에 인용. m={3,5,8,12} 모두 표
```

---

## 3. Phase 3 — 강화 (선택, $0–10)

### Step 9: Distribution shift sanity ($0, 1분)
```bash
.venv/bin/python experiments/round8_distribution_shift.py \
    --calib-specialty emergency_medicine \
    --n-cal 500 --n-test 200 --seed 42 \
    --out results/round8/distribution_shift.json
# supplementary §G에 인용
```

### Step 10: Equity audit ($0, 1분, multi-seed run 이후)
```bash
.venv/bin/python experiments/round8_equity_audit.py \
    --table4 results/run_<ts>/openai/table4_baseline.json \
    --out results/round8/equity_audit_openai.json
.venv/bin/python experiments/round8_equity_audit.py \
    --table4 results/run_<ts>/lmstudio/table4_baseline.json \
    --out results/round8/equity_audit_lmstudio.json
# supplementary §I
```

### Step 11 (optional): Multi-lingual ($0, data 있을 때만)
```bash
# data/raw/medqa_zh.jsonl 위치 → 자동 실행. 없으면 graceful skip.
.venv/bin/python experiments/round8_multilingual_sanity.py \
    --backend openai --n-cal 100 --n-test 50 --seed 42 \
    --out results/round8/multilingual_sanity.json
```

### Step 12 (optional, IRB 일정에 따라): physician relabeling
- IRB_PROTOCOL.md 일정에 따라 별도 진행. LLM-judge 결과(§C)는 그 동안의 partial validation.

---

## 4. 비용·시간 총합

| Phase | 비용 (USD) | 시간 |
|---|---|---|
| 1: P0 해결 (data + 5-seed + LLM-judge) | ~$150 | ~3시간 + setup |
| 2: P1 보강 (multi-dataset + Pivot B) | ~$30 | ~1.5시간 |
| 3: P2 강화 (shift/equity/multi-lingual) | ~$0–10 | ~10분 |
| **합계** | **~$180–190** | **~5시간** |

---

## 5. 의존성 / 순서 그래프

```
[Step 1: data 확보] ──▶ [Step 2: dry-run] ──▶ [Step 3: single-seed]
                                                       │
                                                       ▼
                                          [Step 4: 5-seed bootstrap]
                                                       │
            ┌──────────────────────────────────────────┼───────────────┐
            ▼                                          ▼               ▼
    [Step 5: LLM-judge]                       [Step 7: multi-dataset]  [Step 10: equity]
            │                                          │
            ▼                                          ▼
    [Step 6: 페이퍼 §6 update + 페이퍼-claim test 9/9]
                                                       │
            ┌──────────────────────────────────────────┼───────────────┐
            ▼                                          ▼               ▼
    [Step 8: Pivot B case]               [Step 9: distribution shift]   [Step 11: zh]
                                                       │
                                                       ▼
                                          [Step 12: IRB physician (별도 일정)]
```

---

## 6. 위험 / 완화

| 위험 | 완화 |
|---|---|
| MedAbstain 라이선스 이슈 | 자동 다운로드 대신 Machcha et al. 2026 EACL supplementary 매뉴얼 단계 안내 (download_datasets.sh) |
| gpt-5.5 모델 미지원 (API access 부족) | `--openai-model gpt-4o`로 fallback. 결과는 동일 형식. |
| claude-opus-4-7 미지원 | `--anthropic-model claude-3-5-sonnet-latest`로 fallback. |
| 5-seed에서 단일-seed 우위 사라짐 | aggregate_seeds.md의 95% CI를 paper §6에 정직 보고. "comparable performance with FWER guarantee"로 재포지셔닝. |
| LLM-judge κ < 0.7 | LLM-judge consensus 라벨 사용 안 함. supplementary §C는 κ만 보고. IRB가 load-bearing. |
| IRB 지연 | paper.IRB_PROTOCOL §7 일정 슬립 시 follow-up paper로 분리 |
