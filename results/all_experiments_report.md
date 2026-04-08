# UASEF 전체 실험 보고서

- **실행 시각**: 2026-04-07T19:34:33.941058
- **총 소요 시간**: 1519m 32s
- **백엔드**: all
- **Scoring Method**: auto
- **α**: 0.15
- **n_cal**: 500 | **n_test**: 50 | **n_medabstain**: 50 | **n_pareto_test**: 20

> **실험 구조**
> - `[Primary]` **OpenAI** — logprob-based CP: token-level logprobs 기반 비적합 점수. **논문 주요 결과.**
> - `[Ablation]` **LMStudio GGUF** — logprob-based CP: LM Studio OpenAI-compatible API를 통해 token-level logprobs 추출. 로컬 GGUF 모델 적용 가능성 검증.
> - 두 백엔드 모두 동일한 logprob 비적합 함수를 사용합니다. 모델 차이에 의한 성능 비교가 가능합니다.


## 1. LangGraph 에이전트 실험

| Backend | Role | Accuracy | Safety Recall | Over-Esc. Rate | Escalation Rate | Avg Tool Calls | Avg Iters | Coverage |
| ---------- | ----------- | ---------- | -------------- | ---------------- | ---------------- | ---------------- | ---------- | ---------- |
| lmstudio | [Primary] | 0.6884 | 0.6854 | 0.2857 | 0.6432 | 0.2700 | 1.3 | 0.8100 |
| openai | [Primary] | 0.8500 | 0.9050 | 0.6190 | 0.8750 | 1.2600 | 1.7 | 0.8900 |

## 2. 베이스라인 비교 실험

| Backend | Role | 전략 | Safety Recall | Over-Esc. Rate | TP | FN | FP | TN | OK(≥0.95) |
| ---------- | ----------- | ---------------------- | -------------- | ---------------- | ---- | ---- | ---- | ---- | ---------- |
| lmstudio | [Ablation] | no_escalation | 0.0000 | 0.0000 | 0 | 179 | 0 | 21 | ✗ |
| lmstudio | [Ablation] | threshold_only | 0.5754 | 0.0476 | 103 | 76 | 1 | 20 | ✗ |
| lmstudio | [Ablation] | full_uasef | 0.7095 | 0.1905 | 127 | 52 | 4 | 17 | ✗ |
| openai | [Primary] | no_escalation | 0.0000 | 0.0000 | 0 | 179 | 0 | 21 | ✗ |
| openai | [Primary] | threshold_only | 0.7151 | 0.1905 | 128 | 51 | 4 | 17 | ✗ |
| openai | [Primary] | full_uasef | 0.7374 | 0.2381 | 132 | 47 | 5 | 16 | ✗ |

## 3. MedAbstain 분류 정확도 평가


### Backend: lmstudio

**전체** — Safety Recall: 0.6566 ✗ | Precision: 0.4851 | F1: 0.5579 | AUROC: N/A

| Variant | n | Recall | Precision | F1 | AUROC | OK(≥0.95) |
| -------- | ------ | -------- | ---------- | ------ | ------- | ---------- |
| AP | 49 | 0.6122 | 1.0000 | 0.7595 | N/A | ✗ |
| NAP | 50 | 0.7000 | 1.0000 | 0.8235 | N/A | ✗ |
| A | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |
| NA | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |

**LLM Abstention Accuracy** — Precision: 0.6818 | Recall: 0.1515 | F1: 0.2479

### Backend: openai

**전체** — Safety Recall: 0.6400 ✗ | Precision: 0.4812 | F1: 0.5494 | AUROC: N/A

| Variant | n | Recall | Precision | F1 | AUROC | OK(≥0.95) |
| -------- | ------ | -------- | ---------- | ------ | ------- | ---------- |
| AP | 50 | 0.5800 | 1.0000 | 0.7342 | N/A | ✗ |
| NAP | 50 | 0.7000 | 1.0000 | 0.8235 | N/A | ✗ |
| A | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |
| NA | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |

**LLM Abstention Accuracy** — Precision: 0.6486 | Recall: 0.2400 | F1: 0.3504

## 4. Pareto Frontier α Sweep


### Backend: lmstudio

| α | Mean Coverage | Mean Esc. Rate | # Points |
| ------ | -------------- | -------------- | -------- |
| 0.01 | 0.9900 | 0.2500 | 3 |
| 0.05 | 0.9467 | 0.3167 | 3 |
| 0.1 | 0.8867 | 0.3667 | 3 |
| 0.15 | 0.7500 | 0.4000 | 3 |
| 0.2 | 0.6700 | 0.4833 | 3 |
| 0.3 | 0.5500 | 0.5500 | 3 |

**α 권고** (min_coverage=0.95, max_esc_rate=0.15)

| Specialty | α | Coverage | Esc. Rate | Utility | 근거 |
| ------------------------ | ----- | --------- | --------- | -------- | ---------------------------------------- |
| emergency_medicine | 0.01 | 0.9900 | 0.7500 | -0.5100 | coverage 제약(≥0.95) 충족 중 escalation_rate 최소 선택 (esc_rate=0.750 > 0.15 초과) |
| internal_medicine | 0.01 | 0.9900 | 0.0000 | 0.9900 | coverage=0.990(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |
| general_practice | 0.01 | 0.9900 | 0.0000 | 0.9900 | coverage=0.990(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |

### Backend: openai

| α | Mean Coverage | Mean Esc. Rate | # Points |
| ------ | -------------- | -------------- | -------- |
| 0.01 | 1.0000 | 0.5000 | 3 |
| 0.05 | 0.9700 | 0.5667 | 3 |
| 0.1 | 0.9333 | 0.5833 | 3 |
| 0.15 | 0.9267 | 0.6000 | 3 |
| 0.2 | 0.8967 | 0.6333 | 3 |
| 0.3 | 0.7967 | 0.7167 | 3 |

**α 권고** (min_coverage=0.95, max_esc_rate=0.15)

| Specialty | α | Coverage | Esc. Rate | Utility | 근거 |
| ------------------------ | ----- | --------- | --------- | -------- | ---------------------------------------- |
| emergency_medicine | 0.01 | 1.0000 | 1.0000 | -1.0000 | coverage 제약(≥0.95) 충족 중 escalation_rate 최소 선택 (esc_rate=1.000 > 0.15 초과) |
| internal_medicine | 0.01 | 1.0000 | 0.5000 | 0.0000 | coverage 제약(≥0.95) 충족 중 escalation_rate 최소 선택 (esc_rate=0.500 > 0.15 초과) |
| general_practice | 0.01 | 1.0000 | 0.0000 | 1.0000 | coverage=1.000(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |

## 핵심 지표 요약

> Safety Recall ≥ 0.95 달성 여부를 중심으로 각 실험 결과를 요약합니다.


### Backend: lmstudio

- **[에이전트]** Safety Recall: **0.6854** ✗ | Accuracy: 0.6884 | Over-Esc: 0.2857
- **[베이스라인 full_uasef]** Safety Recall: **0.7095** ✗ | Over-Esc: 0.1905
- **[MedAbstain 전체]** Safety Recall: **0.6566** ✗ | AUROC: N/A
- **[Pareto 권고 α]** emergency_medicine→α=0.01, internal_medicine→α=0.01, general_practice→α=0.01

### Backend: openai

- **[에이전트]** Safety Recall: **0.9050** ✗ | Accuracy: 0.8500 | Over-Esc: 0.6190
- **[베이스라인 full_uasef]** Safety Recall: **0.7374** ✗ | Over-Esc: 0.2381
- **[MedAbstain 전체]** Safety Recall: **0.6400** ✗ | AUROC: N/A
- **[Pareto 권고 α]** emergency_medicine→α=0.01, internal_medicine→α=0.01, general_practice→α=0.01

---
_Generated by `experiments/run_all_experiments.py`_
