# UASEF 전체 실험 보고서

- **실행 시각**: 2026-04-10T00:19:30.167433
- **총 소요 시간**: 1566m 43s
- **백엔드**: all
- **Scoring Method**: auto
- **α**: None
- **n_cal**: 500 | **n_test**: 50 | **n_medabstain**: 50 | **n_pareto_test**: 20

> **실험 구조**
> - `[Primary]` **OpenAI** — logprob-based CP: token-level logprobs 기반 비적합 점수. **논문 주요 결과.**
> - `[Ablation]` **LMStudio GGUF** — logprob-based CP: LM Studio OpenAI-compatible API를 통해 token-level logprobs 추출. 로컬 GGUF 모델 적용 가능성 검증.
> - 두 백엔드 모두 동일한 logprob 비적합 함수를 사용합니다. 모델 차이에 의한 성능 비교가 가능합니다.

c
## 1. LangGraph 에이전트 실험

| Backend | Role | Accuracy | Safety Recall | Over-Esc. Rate | Escalation Rate | Avg Tool Calls | Avg Iters | Coverage |
| ---------- | ----------- | ---------- | -------------- | ---------------- | ---------------- | ---------------- | ---------- | ---------- |
| lmstudio | [Primary] | 0.4020 | 0.3371 | 0.0476 | 0.3065 | 0.2500 | 1.2 | 0.9200 |
| openai | [Primary] | 0.6200 | 0.5866 | 0.0952 | 0.5350 | 1.3300 | 1.7 | 0.9500 |

## 2. 베이스라인 비교 실험

| Backend | Role | 전략 | Safety Recall | Over-Esc. Rate | TP | FN | FP | TN | OK(≥0.95) |
| ---------- | ----------- | ---------------------- | -------------- | ---------------- | ---- | ---- | ---- | ---- | ---------- |
| lmstudio | [Ablation] | no_escalation | 0.0000 | 0.0000 | 0 | 179 | 0 | 21 | ✗ |
| lmstudio | [Ablation] | threshold_only | 0.2514 | 0.0000 | 45 | 134 | 0 | 21 | ✗ |
| lmstudio | [Ablation] | full_uasef | 0.4190 | 0.0476 | 75 | 104 | 1 | 20 | ✗ |
| openai | [Primary] | no_escalation | 0.0000 | 0.0000 | 0 | 179 | 0 | 21 | ✗ |
| openai | [Primary] | threshold_only | 0.3911 | 0.0000 | 70 | 109 | 0 | 21 | ✗ |
| openai | [Primary] | full_uasef | 0.4246 | 0.2381 | 76 | 103 | 5 | 16 | ✗ |

## 3. MedAbstain 분류 정확도 평가


### Backend: lmstudio

**전체** — Safety Recall: 0.1667 ✗ | Precision: 0.7576 | F1: 0.2732 | AUROC: N/A

| Variant | n | Recall | Precision | F1 | AUROC | OK(≥0.95) |
| -------- | ------ | -------- | ---------- | ------ | ------- | ---------- |
| AP | 50 | 0.1200 | 1.0000 | 0.2143 | N/A | ✗ |
| NAP | 50 | 0.2000 | 1.0000 | 0.3333 | N/A | ✗ |
| A | 50 | 0.1800 | 1.0000 | 0.3051 | N/A | ✗ |
| NA | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |

**LLM Abstention Accuracy** — Precision: 0.9167 | Recall: 0.0733 | F1: 0.1358

### Backend: openai

**전체** — Safety Recall: 0.2667 ✗ | Precision: 0.8000 | F1: 0.4000 | AUROC: N/A

| Variant | n | Recall | Precision | F1 | AUROC | OK(≥0.95) |
| -------- | ------ | -------- | ---------- | ------ | ------- | ---------- |
| AP | 50 | 0.2800 | 1.0000 | 0.4375 | N/A | ✗ |
| NAP | 50 | 0.3400 | 1.0000 | 0.5075 | N/A | ✗ |
| A | 50 | 0.1800 | 1.0000 | 0.3051 | N/A | ✗ |
| NA | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |

**LLM Abstention Accuracy** — Precision: 0.8158 | Recall: 0.2067 | F1: 0.3298

## 4. Pareto Frontier α Sweep


### Backend: lmstudio

| α | Mean Coverage | Mean Esc. Rate | # Points |
| ------ | -------------- | -------------- | -------- |
| 0.01 | 1.0000 | 0.1750 | 2 |
| 0.05 | 0.9100 | 0.3000 | 2 |
| 0.1 | 0.8450 | 0.4000 | 2 |
| 0.15 | 0.8050 | 0.4250 | 2 |
| 0.2 | 0.7750 | 0.4500 | 2 |
| 0.3 | 0.6050 | 0.5500 | 2 |

**α 권고** (min_coverage=0.95, max_esc_rate=0.15)

| Specialty | α | Coverage | Esc. Rate | Utility | 근거 |
| ------------------------ | ----- | --------- | --------- | -------- | ---------------------------------------- |
| emergency_medicine | 0.01 | 1.0000 | 0.3500 | 0.3000 | coverage 제약(≥0.95) 충족 중 escalation_rate 최소 선택 (esc_rate=0.350 > 0.15 초과) |
| internal_medicine | 0.01 | 1.0000 | 0.0000 | 1.0000 | coverage=1.000(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |

### Backend: openai

| α | Mean Coverage | Mean Esc. Rate | # Points |
| ------ | -------------- | -------------- | -------- |
| 0.01 | 0.9933 | 0.2500 | 3 |
| 0.05 | 0.9633 | 0.3667 | 3 |
| 0.1 | 0.9400 | 0.4667 | 3 |
| 0.15 | 0.8833 | 0.5167 | 3 |
| 0.2 | 0.8867 | 0.5333 | 3 |
| 0.3 | 0.7767 | 0.5667 | 3 |

**α 권고** (min_coverage=0.95, max_esc_rate=0.15)

| Specialty | α | Coverage | Esc. Rate | Utility | 근거 |
| ------------------------ | ----- | --------- | --------- | -------- | ---------------------------------------- |
| emergency_medicine | 0.01 | 1.0000 | 0.7000 | -0.4000 | coverage 제약(≥0.95) 충족 중 escalation_rate 최소 선택 (esc_rate=0.700 > 0.15 초과) |
| internal_medicine | 0.01 | 0.9900 | 0.0500 | 0.8900 | coverage=0.990(≥0.95) & esc_rate=0.050(≤0.15) 충족 — utility 최대 |
| general_practice | 0.01 | 0.9900 | 0.0000 | 0.9900 | coverage=0.990(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |

## 핵심 지표 요약

> Safety Recall ≥ 0.95 달성 여부를 중심으로 각 실험 결과를 요약합니다.


### Backend: openai

- **[에이전트]** Safety Recall: **0.5866** ✗ | Accuracy: 0.6200 | Over-Esc: 0.0952
- **[베이스라인 full_uasef]** Safety Recall: **0.4246** ✗ | Over-Esc: 0.2381
- **[MedAbstain 전체]** Safety Recall: **0.2667** ✗ | AUROC: N/A
- **[Pareto 권고 α]** emergency_medicine→α=0.01, internal_medicine→α=0.01, general_practice→α=0.01

### Backend: lmstudio

- **[에이전트]** Safety Recall: **0.3371** ✗ | Accuracy: 0.4020 | Over-Esc: 0.0476
- **[베이스라인 full_uasef]** Safety Recall: **0.4190** ✗ | Over-Esc: 0.0476
- **[MedAbstain 전체]** Safety Recall: **0.1667** ✗ | AUROC: N/A
- **[Pareto 권고 α]** emergency_medicine→α=0.01, internal_medicine→α=0.01

---
_Generated by `experiments/run_all_experiments.py`_
