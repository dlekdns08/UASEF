# UASEF 전체 실험 보고서

- **실행 시각**: 2026-04-06T01:24:57.054114
- **총 소요 시간**: 1624m 04s
- **백엔드**: all
- **Scoring Method**: auto
- **α**: 0.05
- **n_cal**: 500 | **n_test**: 50 | **n_medabstain**: 50 | **n_pareto_test**: 20

> **실험 구조**
> - `[Primary]` **OpenAI** — logprob-based CP: token-level logprobs 기반 비적합 점수. **논문 주요 결과.**
> - `[Ablation]` **LMStudio GGUF** — logprob-based CP: LM Studio OpenAI-compatible API를 통해 token-level logprobs 추출. 로컬 GGUF 모델 적용 가능성 검증.
> - 두 백엔드 모두 동일한 logprob 비적합 함수를 사용합니다. 모델 차이에 의한 성능 비교가 가능합니다.


## 1. LangGraph 에이전트 실험

| Backend | Role | Accuracy | Safety Recall | Over-Esc. Rate | Escalation Rate | Avg Tool Calls | Avg Iters | Coverage |
| ---------- | ----------- | ---------- | -------------- | ---------------- | ---------------- | ---------------- | ---------- | ---------- |
| lmstudio | [Primary] | 0.5253 | 0.4689 | 0.0000 | 0.4192 | 0.2600 | 1.3 | 0.9800 |
| openai | [Primary] | 0.7200 | 0.6983 | 0.0952 | 0.6350 | 1.2700 | 1.7 | 0.9500 |

## 2. 베이스라인 비교 실험

| Backend | Role | 전략 | Safety Recall | Over-Esc. Rate | TP | FN | FP | TN | OK(≥0.95) |
| ---------- | ----------- | ---------------------- | -------------- | ---------------- | ---- | ---- | ---- | ---- | ---------- |
| lmstudio | [Ablation] | no_escalation | 0.0000 | 0.0000 | 0 | 179 | 0 | 21 | ✗ |
| lmstudio | [Ablation] | threshold_only | 0.3855 | 0.0000 | 69 | 110 | 0 | 21 | ✗ |
| lmstudio | [Ablation] | full_uasef | 0.4525 | 0.0952 | 81 | 98 | 2 | 19 | ✗ |
| openai | [Primary] | no_escalation | 0.0000 | 0.0000 | 0 | 179 | 0 | 21 | ✗ |
| openai | [Primary] | threshold_only | 0.4804 | 0.0000 | 86 | 93 | 0 | 21 | ✗ |
| openai | [Primary] | full_uasef | 0.8492 | 0.6667 | 152 | 27 | 14 | 7 | ✗ |

## 3. MedAbstain 분류 정확도 평가


### Backend: lmstudio

**전체** — Safety Recall: 0.4600 ✗ | Precision: 0.6970 | F1: 0.5542 | AUROC: N/A

| Variant | n | Recall | Precision | F1 | AUROC | OK(≥0.95) |
| -------- | ------ | -------- | ---------- | ------ | ------- | ---------- |
| AP | 50 | 0.4400 | 1.0000 | 0.6111 | N/A | ✗ |
| NAP | 50 | 0.4800 | 1.0000 | 0.6486 | N/A | ✗ |
| A | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |
| NA | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |

**LLM Abstention Accuracy** — Precision: 0.8182 | Recall: 0.2700 | F1: 0.4060

### Backend: openai

**전체** — Safety Recall: 0.7200 ✗ | Precision: 0.5333 | F1: 0.6128 | AUROC: N/A

| Variant | n | Recall | Precision | F1 | AUROC | OK(≥0.95) |
| -------- | ------ | -------- | ---------- | ------ | ------- | ---------- |
| AP | 50 | 0.7000 | 1.0000 | 0.8235 | N/A | ✗ |
| NAP | 50 | 0.7400 | 1.0000 | 0.8506 | N/A | ✗ |
| A | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |
| NA | 50 | 0.0000 | 0.0000 | 0.0000 | N/A | ✗ |

**LLM Abstention Accuracy** — Precision: 0.5158 | Recall: 0.4900 | F1: 0.5026

## 4. Pareto Frontier α Sweep


### Backend: lmstudio

| α | Mean Coverage | Mean Esc. Rate | # Points |
| ------ | -------------- | -------------- | -------- |
| 0.01 | 1.0000 | 0.2000 | 3 |
| 0.05 | 0.9600 | 0.3167 | 3 |
| 0.1 | 0.9100 | 0.3333 | 3 |
| 0.15 | 0.8600 | 0.3500 | 3 |
| 0.2 | 0.8000 | 0.4000 | 3 |
| 0.3 | 0.7000 | 0.4333 | 3 |

**α 권고** (min_coverage=0.95, max_esc_rate=0.15)

| Specialty | α | Coverage | Esc. Rate | Utility | 근거 |
| ------------------------ | ----- | --------- | --------- | -------- | ---------------------------------------- |
| emergency_medicine | 0.01 | 1.0000 | 0.6000 | -0.2000 | coverage 제약(≥0.95) 충족 중 escalation_rate 최소 선택 (esc_rate=0.600 > 0.15 초과) |
| internal_medicine | 0.01 | 1.0000 | 0.0000 | 1.0000 | coverage=1.000(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |
| general_practice | 0.01 | 1.0000 | 0.0000 | 1.0000 | coverage=1.000(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |

### Backend: openai

| α | Mean Coverage | Mean Esc. Rate | # Points |
| ------ | -------------- | -------------- | -------- |
| 0.01 | 1.0000 | 0.3667 | 3 |
| 0.05 | 0.9733 | 0.4500 | 3 |
| 0.1 | 0.9100 | 0.4833 | 3 |
| 0.15 | 0.8900 | 0.5167 | 3 |
| 0.2 | 0.8950 | 0.5000 | 2 |
| 0.3 | 0.7700 | 0.5750 | 2 |

**α 권고** (min_coverage=0.95, max_esc_rate=0.15)

| Specialty | α | Coverage | Esc. Rate | Utility | 근거 |
| ------------------------ | ----- | --------- | --------- | -------- | ---------------------------------------- |
| emergency_medicine | 0.01 | 1.0000 | 0.9500 | -0.9000 | coverage 제약(≥0.95) 충족 중 escalation_rate 최소 선택 (esc_rate=0.950 > 0.15 초과) |
| internal_medicine | 0.01 | 1.0000 | 0.1500 | 0.7000 | coverage=1.000(≥0.95) & esc_rate=0.150(≤0.15) 충족 — utility 최대 |
| general_practice | 0.01 | 1.0000 | 0.0000 | 1.0000 | coverage=1.000(≥0.95) & esc_rate=0.000(≤0.15) 충족 — utility 최대 |

## 핵심 지표 요약

> Safety Recall ≥ 0.95 달성 여부를 중심으로 각 실험 결과를 요약합니다.


### Backend: openai

- **[에이전트]** Safety Recall: **0.6983** ✗ | Accuracy: 0.7200 | Over-Esc: 0.0952
- **[베이스라인 full_uasef]** Safety Recall: **0.8492** ✗ | Over-Esc: 0.6667
- **[MedAbstain 전체]** Safety Recall: **0.7200** ✗ | AUROC: N/A
- **[Pareto 권고 α]** emergency_medicine→α=0.01, internal_medicine→α=0.01, general_practice→α=0.01

### Backend: lmstudio

- **[에이전트]** Safety Recall: **0.4689** ✗ | Accuracy: 0.5253 | Over-Esc: 0.0000
- **[베이스라인 full_uasef]** Safety Recall: **0.4525** ✗ | Over-Esc: 0.0952
- **[MedAbstain 전체]** Safety Recall: **0.4600** ✗ | AUROC: N/A
- **[Pareto 권고 α]** emergency_medicine→α=0.01, internal_medicine→α=0.01, general_practice→α=0.01

---
_Generated by `experiments/run_all_experiments.py`_
