# UASEF

Uncertainty-Aware Safe Escalation Framework for Medical LLM Agents

LLM 기반 의료 에이전트가 **자신의 불확실성을 정량화**하고, **위험도를 판단**하여 **인간 전문가에게 자동 인계**하는 연구 프레임워크입니다.

---

## 목차

1. [연구 배경 및 동기](#1-연구-배경-및-동기)
2. [핵심 설계 철학](#2-핵심-설계-철학)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [아키텍처 상세](#4-아키텍처-상세)
   - 4.1 [UQM — Uncertainty Quantification Module](#41-uqm--uncertainty-quantification-module)
   - 4.2 [RTC — Risk-Threshold Calibrator](#42-rtc--risk-threshold-calibrator)
   - 4.3 [EDE — Escalation Decision Engine](#43-ede--escalation-decision-engine)
   - 4.4 [LangGraph 에이전트](#44-langgraph-에이전트)
5. [데이터셋](#5-데이터셋)
6. [실험 설계](#6-실험-설계)
   - 6.1 [순차 파이프라인 실험](#61-순차-파이프라인-실험-run_experimentpy)
   - 6.2 [LangGraph 에이전트 실험](#62-langgraph-에이전트-실험-run_agent_experimentpy)
   - 6.3 [MedAbstain 분류 정확도 평가](#63-medabstain-분류-정확도-평가-eval_medabstainpy)
   - 6.4 [Pareto Frontier Alpha Sweep](#64-pareto-frontier-alpha-sweep-pareto_sweeppy)
7. [평가 지표](#7-평가-지표)
8. [설치 및 환경 구성](#8-설치-및-환경-구성)
9. [실험 실행](#9-실험-실행)
10. [출력 파일](#10-출력-파일)
11. [논문 권장 설정](#11-논문-권장-설정)
12. [참고문헌](#12-참고문헌)

---

## 1. 연구 배경 및 동기

### 문제 정의

LLM을 의료 현장에 배포할 때 가장 큰 장벽은 **"모델이 언제 틀리는지 모른다"** 는 점입니다. 의료 도메인에서 잘못된 자신감(overconfidence)은 치명적 결과로 이어질 수 있습니다. 기존 접근법들은 세 가지 한계를 가집니다.

| 기존 접근법 | 한계 |
|------------|------|
| 단순 threshold 기반 | 임계값이 임의적이며 통계적 보장 없음 |
| Human-in-the-loop (항상) | 자율 처리 불가 → 운영 비용, 지연 |
| 확률 보정(calibration) | 도메인 이동(distribution shift) 시 보장 붕괴 |

### UASEF의 제안

UASEF는 **Conformal Prediction(CP)** 이론을 의료 LLM에 적용하여 세 가지를 동시에 달성합니다.

1. **통계적 보장**: `P(s_test ≤ q̂) ≥ 1 - α` — 이론적으로 증명된 커버리지
2. **동적 위험도 반영**: 전문과목·시나리오에 따라 임계값 자동 조정
3. **선택적 에스컬레이션**: 자율 처리 가능한 케이스는 AI가, 불확실하거나 고위험인 케이스만 전문의에게 인계

---

## 2. 핵심 설계 철학

### 왜 Conformal Prediction인가?

#### 기존 확률 보정의 문제점 (temperature scaling, Platt scaling)

- 모델이 "70% 확신" 이라고 말해도, 실제로 70% 맞을 것이라는 보장이 없음
- 학습 분포와 다른 도메인에서 보정 성질이 유지되지 않음

#### Conformal Prediction의 강점

- 분포 가정 불필요. 교환가능성(exchangeability)만 가정
- `q̂ = ⌈(n+1)(1-α)⌉/n 번째 순위 비적합 점수` → 이 임계값 하나로 `P(s_test ≤ q̂) ≥ 1-α` 성립
- 따라서 "α = 0.05로 설정 → 실제 에스컬레이션 누락률 ≤ 5%"가 수학적으로 보장됨

### 왜 Nonconformity Score로 token logprob을 쓰는가?

비적합 점수(nonconformity score)는 "이 테스트 포인트가 캘리브레이션 셋과 얼마나 다른가"를 수치화한 것입니다. UASEF는 **평균 negative log-likelihood**를 사용합니다.

```text
s(x) = -mean(log P(t_i | context, t_1, ..., t_{i-1}))
```

이 선택의 이유:

- 모델이 생성한 각 토큰의 확률을 그대로 반영 → 답변 생성 과정 자체의 불확실성
- Temperature = 0일 때도 의미 있음 (greedy decoding이지만 logprob은 여전히 분포를 반영)
- API 추가 호출 불필요 (generate 한 번으로 score와 답변을 동시에 얻음)

### 왜 세 모듈로 분리했는가?

```text
UQM  →  "이 질문이 얼마나 어려운가?" (CP 기반 통계 측정)
RTC  →  "얼마나 어려워야 에스컬레이션할 것인가?" (위험도 기반 임계값)
EDE  →  "최종적으로 에스컬레이션할 것인가?" (다중 신호 통합 결정)
```

세 관심사를 분리함으로써:

- UQM은 CP 이론 컴포넌트로만 교체 가능 (weighted CP, conformal risk control 등)
- RTC의 전문과목 위험도 온톨로지는 임상 전문가 피드백으로 독립 업데이트 가능
- EDE의 트리거 정책은 기관별 프로토콜에 맞게 조정 가능

### 왜 LangGraph ReAct 구조인가?

단순 쿼리-응답이 아닌 **추론-행동 루프(Reasoning + Acting)**를 택한 이유:

- 의료 질문은 단일 답변보다 도구 활용(약물 상호작용 DB, 가이드라인 검색)이 필요
- UASEF는 에이전트 내부가 아닌 **외부에서 독립적으로 판정** → 에이전트 출력을 감사(audit)하는 구조
- LangGraph의 StateGraph가 ReAct 루프와 UASEF 체크 노드를 명확히 분리

---

## 3. 프로젝트 구조

```text
UASEF/
├── models/                         # 핵심 모듈
│   ├── model_interface.py          # LMStudio / OpenAI 통합 추상화 레이어
│   ├── uqm.py                      # Uncertainty Quantification Module (CP 기반)
│   └── rtc_ede.py                  # Risk-Threshold Calibrator + Escalation Decision Engine
│
├── agent/                          # LangGraph ReAct 에이전트
│   ├── graph.py                    # StateGraph 조립
│   ├── nodes.py                    # 노드 함수 + AgentComponents
│   ├── state.py                    # MedicalAgentState TypedDict
│   └── tools.py                    # 의료 도구 4종 (drug, guideline, lab, DDx)
│
├── data/
│   ├── loader.py                   # MedQA / MedAbstain / MIMIC-III 로더
│   ├── raw/                        # 로컬 JSONL 파일 위치 (.gitignore)
│   └── README.md                   # 데이터 소스 및 다운로드 가이드
│
├── experiments/
│   ├── configs/                    # 시나리오별 YAML 설정
│   │   ├── base_config.yaml        # 공통 기본값
│   │   ├── scenario_emergency.yaml
│   │   ├── scenario_rare_disease.yaml
│   │   └── scenario_multimorbidity.yaml
│   ├── run_experiment.py           # 순차 파이프라인 실험 (LMStudio vs OpenAI)
│   ├── run_agent_experiment.py     # LangGraph 에이전트 실험
│   ├── eval_medabstain.py          # MedAbstain AP/NAP 분류 정확도 평가
│   ├── pareto_sweep.py             # α sweep → Pareto frontier + α 권고
│   └── visualize_results.py        # 결과 시각화
│
├── results/                        # 실험 결과 (자동 생성, .gitignore)
├── pyproject.toml
└── .env.example
```

---

## 4. 아키텍처 상세

### 4.1 UQM — Uncertainty Quantification Module

**파일**: `models/uqm.py`

UQM은 단일 질문의 불확실성을 **통계적으로 보장된 수치**로 변환합니다.

#### 내부 흐름

```text
질문 입력
   ↓
_get_score(): LLM 호출 → token logprobs 수집
   ↓
compute_nonconformity_score(): s = -mean(logprobs)
   ↓
calibrator.threshold와 비교 → should_escalate
   ↓
compute_entropy(): top_logprobs로 위치별 조건부 엔트로피 계산
   ↓
UncertaintyResult 반환
```

#### Conformal Calibration 수식

보정 집합 `{s_1, ..., s_n}` (비적합 점수들)에서 임계값 계산:

```text
q̂ = s_{(⌈(n+1)(1-α)⌉)}  ← n번째 순위 점수

보장: P(s_test ≤ q̂) ≥ 1 - α
```

실제 구현에서는 `numpy.quantile`을 사용하며, level을 `min(1.0, ⌈(n+1)(1-α)⌉/n)` 으로 보정하여 유한 표본에서의 보수성을 확보합니다.

#### Scoring Method 비교

| 방식 | 수식 | 특징 | 논문 위치 |
|------|------|------|----------|
| **logprob** (Primary) | `s = -mean(token logprobs)` | CP 보장 ✓, 단일 쿼리 | 주요 기여 |
| **self_consistency** (Ablation) | `s = Jaccard_diversity × 5` | CP 보장 ✓, N회 쿼리 | Ablation study |
| **auto** | 런타임 감지 | 재현성 저하 위험 | 비권장 |

> **왜 logprob이 Primary인가?**
> 자연어 다양성보다 모델 내부 확률 분포가 CP의 교환가능성 가정에 더 적합하며, 단일 쿼리로 score와 답변을 동시에 얻어 비용과 지연을 절반으로 줄입니다.

#### 엔트로피 계산

`compute_entropy(response: ModelResponse)` 는 `top_logprobs`가 있을 때만 유효한 엔트로피를 반환합니다.

```python
# 각 토큰 위치에서 상위 k개 logprob으로 조건부 분포 근사
probs = softmax(top_k_logprobs)   # 정규화
H_pos = -sum(p * log(p))          # 위치별 엔트로피
H_avg = mean(H_pos)               # 전체 평균 (nats/token)
```

`top_logprobs`가 없으면 `float("nan")` 반환 — 개별 토큰 logprob으로는 Shannon 엔트로피를 계산할 수 없기 때문입니다 (각 값이 완전한 어휘 분포를 구성하지 않음).

#### Distribution Shift 처리

```python
# 보정: MedQA 분포
uqm.calibrate(cal_questions, distribution_source="medqa")

# 평가: MIMIC-III 분포 (다른 분포!) → 자동 경고 + Weighted CP 전환
uqm.evaluate(question, distribution_source="mimic3")
```

Weighted CP (Tibshirani et al., 2019)는 교환가능성 위반 시 커버리지 보장을 복원합니다.

```text
w_i = 1 + k × Jaccard(cal_i, test)   # 밀도비 근사

q̂_w = inf{q : Σ_{s_i ≤ q} w_i / (Σ w_i + w_{n+1}) ≥ 1-α}
```

`w_{n+1}` (테스트 포인트 자신의 weight)를 분모에 포함해야 CP 하한 보장이 성립합니다. `w_{n+1} = 1 + k` (Jaccard(test, test) = 1.0 이므로 최대 유사도).

#### UncertaintyResult 주요 필드

| 필드 | 설명 |
|------|------|
| `nonconformity_score` | 비적합 점수 — 클수록 불확실 |
| `margin` | `threshold - score` — 양수=안전 여유, 음수=임계값 초과 |
| `confidence_entropy` | 위치별 조건부 엔트로피 (nats/token). `top_logprobs` 없으면 `nan` |
| `should_escalate` | `score > threshold` 여부 |
| `weighted_cp_used` | Weighted CP 적용 여부 |

---

### 4.2 RTC — Risk-Threshold Calibrator

**파일**: `models/rtc_ede.py`

UQM이 반환한 기본 임계값 `q̂`를 전문과목과 시나리오의 위험도에 따라 **동적으로 조정**합니다.

#### 조정 수식

```text
adjusted_threshold = q̂ × risk_multiplier × scenario_multiplier
```

| 위험 등급 | 배율 | 해당 전문과목 |
|----------|------|--------------|
| CRITICAL | ×0.60 | 응급의학, 중환자의학, 외상외과 |
| HIGH     | ×0.75 | 심장내과, 신경과, 종양학, 심흉외과 |
| MODERATE | ×1.00 | 내과, 외과, 소아과, 산부인과 |
| LOW      | ×1.30 | 일반 외래, 예방의학, 피부과, 정신건강의학과 |

`emergency` / `rare_disease` 시나리오에는 추가 ×0.85 적용됩니다.

> **설계 이유**: 응급의학에서 에스컬레이션 누락(False Negative)의 비용은 일반 외래에 비해 훨씬 큽니다. 임계값을 낮추면 더 많은 케이스가 에스컬레이션되지만, 위험한 케이스를 놓칠 확률이 줄어듭니다. 이 트레이드오프를 전문과목 온톨로지로 인코딩했습니다.

#### Pareto Frontier 분석

```python
rtc.pareto_frontier(sweep_results)
```

`pareto_sweep.py`의 실측 데이터를 받아 각 `(α, specialty)` 조합에서 `(coverage, escalation_rate)` 쌍을 반환합니다. 이를 통해 실제로 측정된 trade-off를 시각화하고 최적 α를 권고합니다.

---

### 4.3 EDE — Escalation Decision Engine

**파일**: `models/rtc_ede.py`

세 가지 트리거를 통합하여 최종 에스컬레이션 여부를 결정합니다.

#### 트리거 구조

```text
Trigger 1 — UNCERTAINTY_EXCEEDED:
    nonconformity_score > adjusted_threshold
    → CP 이론의 직접 신호 (주 트리거)

Trigger 2 — HIGH_RISK_ACTION:
    CRITICAL_KEYWORDS 감지   (EOL 결정, Code Blue)  → 항상 트리거
    PROCEDURAL_KEYWORDS 감지 (intubation, 승압제)   → UNCERTAINTY_MODIFIERS 동반 시만 트리거

Trigger 3 — NO_EVIDENCE:
    "I am not certain", "insufficient evidence" 등 근거 부재 표현 감지

하나라도 활성화 → should_escalate = True
```

> **Trigger 2 설계 이유**: "에피네프린을 아나필락시스에 투여하세요" 같은 정상적인 처치 권고가 키워드만으로 에스컬레이션되는 False Positive를 방지합니다. 따라서 시술 키워드는 불확실 표현(`consider`, `may need`, `if deteriorates` 등)과 함께 나타날 때만 활성화합니다. 반면 DNR, withdraw care 같은 EOL 결정은 AI가 단독으로 판단해서는 안 되므로 항상 에스컬레이션합니다.

#### Confidence 계산

```text
confidence = min(1.0,
    len(triggers) / 3
    + 0.4   if UNCERTAINTY_EXCEEDED in triggers
    + 0.15  if entropy > 2.0 nats/token
)
```

엔트로피는 별도 트리거가 아닌 **신뢰도 가중치**로만 사용됩니다.

---

### 4.4 LangGraph 에이전트

**파일**: `agent/graph.py`, `agent/nodes.py`, `agent/state.py`

#### 그래프 흐름

```
START → reason → [tool_calls?] → act ──→ reason  (ReAct 루프, 최대 5회)
                                         ↓
                               uasef_check  ← 원본 질문 독립 재판
                               ↙          ↘
                          escalate      finalize
                             ↓               ↓
                           END             END
```

#### 주요 설계 결정

##### ① uasef_check는 에이전트와 독립

`uasef_check` 노드는 에이전트의 메시지 히스토리를 보지 않고 **원본 질문을 직접 UQM에 전달**합니다. 에이전트가 도구로 정보를 많이 수집했더라도 UASEF는 별도로 판정합니다.

이 설계 이유:

- 에이전트가 틀린 정보를 수집해도 UASEF가 안전망 역할
- 에이전트 출력을 감사(audit)하는 외부 컴포넌트 패턴

##### ② LLM 재호출 최소화

`reason` 노드에서 이미 `logprobs=True`로 LLM을 호출합니다. `uasef_check`에서 마지막 AIMessage의 `response_metadata`에서 logprobs를 추출해 `pre_computed_response`로 UQM에 전달하면, logprob 모드에서 두 번째 LLM 호출을 생략합니다.

```python
pre_resp = _extract_model_response(last_ai_message, backend)
unc = components.uqm.evaluate(question, pre_computed_response=pre_resp)
# pre_resp가 있으면 LLM 재호출 없이 score 계산
```

##### ③ AgentComponents를 functools.partial로 바인딩

LangGraph State에 비직렬화 객체(UQM, RTC, EDE)를 넣지 않고, `functools.partial`로 각 노드 함수에 클로저로 전달합니다. State는 JSON 직렬화 가능한 데이터만 포함합니다.

##### ④ 의료 도구 4종 (Mock 구현)

| 도구 | 역할 | 실제 연구 교체 대상 |
|------|------|-------------------|
| `drug_interaction_checker` | 약물 상호작용 확인 | Drugs@FDA API / Lexicomp |
| `clinical_guideline_search` | 임상 가이드라인 검색 | UpToDate / PubMed E-utilities |
| `lab_reference_lookup` | 검사 참고치 조회 | LOINC / 기관 내 LIS |
| `differential_diagnosis` | 감별 진단 | Isabel DDx / 기관 내 CDR |

---

## 5. 데이터셋

### 자동 로딩 우선순위

```
1. data/raw/*.jsonl       (로컬 JSONL 파일)
2. HuggingFace datasets   (자동 다운로드)
3. 내장 fallback          (개발/테스트 전용, 30개)
```

### MedQA (USMLE 4-options)

- **역할**: Calibration + 기본 시나리오 테스트
- **출처**: Jin et al., 2021 — "What Disease does this Patient Have?"
- **HuggingFace ID**: `GBaker/MedQA-USMLE-4-options`
- **사용 split**: `train` (calibration), `test` (테스트 시나리오)

USMLE(미국 의사면허시험) 스타일의 4지선다 문제로 구성됩니다. 정답만 알아도 되는 것이 아니라, 왜 틀렸는지를 통해 불확실성을 측정하는 데 적합합니다.

### MedAbstain

- **역할**: 희귀질환·불확실 시나리오, safety 평가의 핵심
- **출처**: Zhu et al., 2023 — "Can LLMs Express Their Uncertainty?"

4가지 변형이 있으며, AP와 NAP가 safety 평가의 핵심입니다.

| 변형 | 설명 | `expected_escalate` | 사용 시나리오 |
|------|------|---------------------|--------------|
| **AP** | Abstention + Perturbed | `True` | 희귀질환 (불확실 + 변형 질문) |
| **NAP** | Normal + Perturbed | `True` | 희귀질환 (정상 답변이지만 변형 질문) |
| **A** | Abstention only | `True` | 일반 불확실 케이스 |
| **NA** | Normal | `False` | 정상 케이스 (True Negative 검증) |

> **왜 AP/NAP가 핵심인가?**: AP와 NAP는 원래 질문을 미묘하게 변형(perturb)하여 모델의 안정성을 테스트합니다. 이 변형된 질문에 자신 있게 답하는 모델은 에스컬레이션해야 하는 상황을 놓칠 위험이 있습니다.

### MIMIC-III (선택 사항)

- **역할**: 실제 ICU 임상 기록으로 distribution shift 실험
- **조건**: PhysioNet DUA(Data Use Agreement) 서명 필요
- **사용 목적**: Weighted CP가 분포 이동 상황에서도 커버리지 보장을 복원하는지 검증

> CP 보장은 calibration과 evaluation이 같은 분포에서 나올 때만 유효합니다 (exchangeability). MedQA로 보정한 뒤 MIMIC-III로 평가하면 CP 보장이 깨지며, 이를 Weighted CP로 복원하는 것이 실험의 핵심입니다.

---

## 6. 실험 설계

### 전체 실험 파이프라인

```
MedQA (calibration)
       ↓
   UQM.calibrate()    ← Split CP: 80%로 q̂ 계산, 20%로 실측 coverage 검증
       ↓
 q̂ → RTC.adjust()   ← specialty × scenario multiplier 적용
       ↓
  MedQA/MedAbstain
  (test scenarios)
       ↓
   UQM.evaluate()     ← s(x) 계산
       ↓
   EDE.decide()       ← 3 트리거 통합 → should_escalate
       ↓
  Safety Recall / Over-Escalation Rate / Conformal Coverage
```

---

### 6.1 순차 파이프라인 실험 (`run_experiment.py`)

LangGraph 에이전트 없이 UQM → RTC → EDE를 순서대로 실행하는 **기본 파이프라인**입니다.

#### 실험 조건

- 백엔드: LMStudio (로컬, meta-llama-3.1-8b-instruct) vs OpenAI (GPT-4o-mini)
- 시나리오: Emergency / Rare Disease / Multimorbidity
- Scoring: logprob (primary) vs self_consistency (ablation)

#### Config 오버라이드 계층

```yaml
# base_config.yaml → scenario_emergency.yaml → CLI 인자
# 오른쪽이 왼쪽을 덮어씁니다.

uqm:
  alpha: 0.05
  scoring_method: logprob
  holdout_fraction: 0.2
data:
  n_calibration: 30      # 논문 권장: 500
  n_test_per_scenario: 3 # 논문 권장: 50
```

#### 실험 흐름

1. `_build_datasets()`: Config에 따라 MedQA / MedAbstain 로드
2. `UQM.calibrate()`: calibration set으로 q̂ 계산 + hold-out으로 실측 coverage 검증
3. 시나리오별로 `UQM.evaluate()` → `EDE.decide()` 실행
4. `compute_metrics()`: TP/FN/FP/TN → Safety Recall, Over-Escalation Rate 계산
5. JSON + CSV로 저장

---

### 6.2 LangGraph 에이전트 실험 (`run_agent_experiment.py`)

ReAct 에이전트가 도구를 활용해 추론하고, UASEF가 독립적으로 에스컬레이션을 판정합니다. 순차 파이프라인 실험과 동일한 케이스를 에이전트로 실행하여 **도구 사용의 효과**를 비교합니다.

#### 추가 측정 항목

- `react_iterations`: reason 노드 호출 횟수 (추론 깊이)
- `tool_calls`: 도구별 사용 횟수
- `avg_tool_calls_per_case`: 케이스당 평균 도구 호출 수

**시나리오 → 전문과목 매핑:**

| 시나리오 | 전문과목 | RTC 위험도 | 임계값 배율 |
|---------|---------|-----------|------------|
| emergency | emergency_medicine | CRITICAL | ×0.60 × 0.85 = ×0.51 |
| rare_disease | neurology | HIGH | ×0.75 × 0.85 = ×0.64 |
| multimorbidity | internal_medicine | MODERATE | ×1.00 |

**에이전트 그래프 실행 상세:**

```python
graph.invoke(
    initial_state,
    config={"recursion_limit": 25}  # 무한 루프 방지
)
```

`max_iterations=5`와 `recursion_limit=25`는 독립적입니다. `max_iterations`는 reason 노드 호출 횟수를 제한하고, `recursion_limit`은 LangGraph 레벨의 전체 노드 전환 횟수를 제한합니다.

---

### 6.3 MedAbstain 분류 정확도 평가 (`eval_medabstain.py`)

MedAbstain 4개 변형에서 UASEF가 에스컬레이션을 올바르게 감지하는지 **이진 분류** 문제로 평가합니다.

#### 측정 지표

| 지표 | 수식 | 중요성 |
|------|------|--------|
| Safety Recall | TP / (TP + FN) | **핵심 — 타협 불가** |
| Precision | TP / (TP + FP) | 불필요한 에스컬레이션 비율 |
| F1 | 2 × Precision × Recall / (P + R) | 균형 지표 |
| Specificity | TN / (TN + FP) | 정상 케이스 자율 처리 비율 |
| AUROC | 순위 성능 | 임계값 독립적 판별력 |

**변형별 비교의 의미:**

- AP recall < NAP recall → 모델이 Abstention + Perturbation 조합을 더 어려워함
- A recall < AP recall → Perturbation이 없어도 불확실한 케이스를 놓침
- NA specificity가 낮으면 → 정상 케이스를 과도하게 에스컬레이션 (Over-Escalation 문제)

**Weighted CP 비교 실험:**

```bash
# 표준 CP
python experiments/eval_medabstain.py --backend openai

# Weighted CP (분포 이동 상황 시뮬레이션)
python experiments/eval_medabstain.py --backend openai --weighted-cp
```

두 결과의 차이가 Weighted CP의 기여를 정량화합니다.

---

### 6.4 Pareto Frontier Alpha Sweep (`pareto_sweep.py`)

Coverage-Escalation Rate **트레이드오프의 실제 측정**입니다. α를 여러 값으로 스윕하며 각 `(α, specialty)` 조합에서 실측 `(coverage, escalation_rate)`를 측정합니다.

**스윕 범위:**

```python
ALPHAS     = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
SPECIALTIES = [
    ("emergency_medicine", "emergency"),
    ("internal_medicine",  "multimorbidity"),
    ("general_practice",   "routine"),
]
```

**총 실험 수**: `6 × 3 × 2 (백엔드) = 36` 포인트

**Pure CP 모드:**

Pareto sweep에서는 Trigger 2 (키워드)와 Trigger 3 (근거 부재)를 제외하고 **CP Trigger만 사용**합니다. 이는 순수한 Conformal Prediction의 효과만 측정하기 위함입니다.

```python
# 순수 CP Trigger만
escalated = unc.nonconformity_score > rtc_config.adjusted_threshold
```

**α 권고 알고리즘:**

```
입력: (α, specialty) 별 실측 (coverage, escalation_rate)
목표: specialty별 최적 α 선택

우선순위:
  1. coverage ≥ 0.95 AND escalation_rate ≤ 0.15 → utility = coverage - 2×esc_rate 최대
  2. coverage ≥ 0.95만 충족 → escalation_rate 최소
  3. 아무것도 충족 안 됨 → utility 최대 (fallback)
```

이 알고리즘은 **안전 제약(coverage)을 효율(escalation_rate)보다 항상 우선**합니다. 의료 도메인에서 coverage 미충족은 생명 위험과 직결되기 때문입니다.

---

## 7. 평가 지표

### 핵심 지표

| 지표 | 목표 | 수식 | 의미 |
|------|------|------|------|
| **Safety Recall** | ≥ 0.95 | TP / (TP + FN) | 에스컬레이션해야 할 케이스를 놓치지 않음 |
| **Over-Escalation Rate** | ≤ 0.15 | FP / (FP + TN) | 자율 처리 가능한 케이스를 불필요하게 넘기지 않음 |
| **Conformal Coverage** | ≥ 1-α | hold-out에서 s ≤ q̂인 비율 | CP 이론 보장의 실측 검증 |

### 지표 해석

- **Safety Recall 0.95**는 "에스컬레이션이 필요한 100개 케이스 중 95개 이상 감지"를 의미합니다. 이는 논의 여지 없는 최소 요구사항입니다.
- **Over-Escalation Rate 0.15**는 "자율 처리 가능한 케이스 중 15% 이하만 불필요하게 전문의에게 전달"을 의미합니다. 너무 낮으면 운영 비용이 증가합니다.
- **Conformal Coverage**가 `1-α`보다 낮으면 CP 이론이 실제로 작동하지 않는 것입니다. 이 경우 calibration 데이터 부족(n < 30) 또는 distribution shift가 원인일 수 있습니다.

### 지표 간 트레이드오프

```
α 낮춤 → Coverage ↑, Safety Recall ↑, Over-Escalation Rate ↑
α 높임 → Coverage ↓, Safety Recall ↓, Over-Escalation Rate ↓

RTC multiplier 낮춤 → adjusted_threshold ↓ → 더 많은 에스컬레이션
RTC multiplier 높임 → adjusted_threshold ↑ → 적은 에스컬레이션
```

이 트레이드오프를 전문과목별로 최적화하는 것이 Pareto Sweep의 목적입니다.

---

## 8. 설치 및 환경 구성

```bash
# uv 설치 (없으면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 환경 변수 설정
cp .env.example .env
# .env에서 OPENAI_API_KEY, LMSTUDIO_MODEL 수정
```

### LMStudio (로컬 모델)

1. LMStudio 앱 실행 → 모델 다운로드 (권장: `meta-llama-3.1-8b-instruct`)
2. **Local Server** 탭 → **Start Server** (기본 포트: 1234)
3. `.env`의 `LMSTUDIO_MODEL`을 로드된 모델명으로 수정

### LangSmith 트레이싱 (선택)

에이전트 실험의 ReAct 루프를 시각적으로 추적할 수 있습니다.

```bash
# .env에 추가
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=UASEF-agent
```

---

## 9. 실험 실행

### 순차 파이프라인 실험

```bash
# 기본 실행 (base_config.yaml 사용)
python experiments/run_experiment.py

# 시나리오별 config 적용
python experiments/run_experiment.py --config experiments/configs/scenario_emergency.yaml

# 논문 품질 (권장 설정)
python experiments/run_experiment.py --n-cal 500 --n-test 50

# Ablation: self_consistency 방식
python experiments/run_experiment.py --scoring-method self_consistency

# 결과 시각화
python experiments/visualize_results.py
```

### LangGraph 에이전트 실험

```bash
# 기본 실행
python experiments/run_agent_experiment.py

# 단일 백엔드
python experiments/run_agent_experiment.py --backend openai --n-cal 500 --n-test 50

# LMStudio + logprob
python experiments/run_agent_experiment.py --backend lmstudio --scoring-method logprob
```

### MedAbstain 분류 정확도 평가

```bash
# 전체 변형 (AP, NAP, A, NA)
python experiments/eval_medabstain.py --backend openai

# 핵심 safety 케이스만 (AP/NAP)
python experiments/eval_medabstain.py --backend openai --variants AP NAP --n 100

# Weighted CP 비교
python experiments/eval_medabstain.py --backend openai --weighted-cp
```

### Pareto Frontier + α 권고

```bash
# α sweep 실행
python experiments/pareto_sweep.py --backend openai --n-cal 500

# 기존 sweep 결과에서 권고만 재계산
python -c "
from experiments.pareto_sweep import recommend_alpha, print_recommendations
recs = recommend_alpha()
print_recommendations(recs)
"
```

### 개별 모듈 테스트

```bash
# 모델 연결 확인 (logprobs 지원 여부 포함)
python models/model_interface.py

# UQM 단독 (logprob vs self_consistency 비교)
python models/uqm.py

# RTC + EDE 단독 (가상 UncertaintyResult로 트리거 확인)
python models/rtc_ede.py
```

---

## 10. 출력 파일

| 파일 | 생성 스크립트 | 설명 |
|------|-------------|------|
| `results/experiment_results.json` | `run_experiment.py` | 백엔드별, 시나리오별 전체 케이스 결과 |
| `results/comparison_table.csv` | `run_experiment.py` | Safety Recall / Over-Escalation Rate / Coverage 요약표 |
| `results/agent_results.json` | `run_agent_experiment.py` | 에이전트 실험 전체 결과 (tool_calls, react_iterations 포함) |
| `results/agent_comparison_table.csv` | `run_agent_experiment.py` | 에이전트 비교 요약 |
| `results/medabstain_eval.json` | `eval_medabstain.py` | 변형별 Precision / Recall / F1 / AUROC 전체 결과 |
| `results/medabstain_eval_summary.csv` | `eval_medabstain.py` | 백엔드 × 변형 요약표 |
| `results/pareto_sweep_results.json` | `pareto_sweep.py` | α × specialty 실측 (coverage, escalation_rate) |
| `results/pareto_frontier.png` | `pareto_sweep.py` | α 별 trajectory + 이상적 영역 |
| `results/alpha_recommendations.json` | `pareto_sweep.py` | specialty별 최적 α 및 권고 이유 |
| `results/comparison_bar.png` | `visualize_results.py` | 백엔드별 Safety Recall / Over-Escalation Rate 바차트 |
| `results/latency_comparison.png` | `visualize_results.py` | 로컬 vs 클라우드 응답 지연 비교 |

---

## 11. 논문 권장 설정

```yaml
# experiments/configs/base_config.yaml
uqm:
  alpha: 0.05
  scoring_method: logprob    # primary. self_consistency는 ablation study 전용
  holdout_fraction: 0.2
data:
  n_calibration: 500         # CP 보장 실용 하한
  n_test_per_scenario: 50    # 시나리오별 케이스 수
```

> **현재 기본값(`n_calibration=30`)은 개발/디버그 전용입니다.**
> n이 너무 작으면 q̂가 보수적(over-coverage)이 되어 지표가 낙관적으로 보입니다.
> 논문 품질 결과를 위해 반드시 `n ≥ 500`을 사용하세요.

### CP 이론 보증 (Angelopoulos & Bates, 2021)

```
q̂ = ⌈(n+1)(1-α)⌉/n 번째 순위 비적합 점수

P(s_test ≤ q̂) ≥ 1 - α   (이론적 하한)

n = 500, α = 0.05 → 실측 coverage ≈ 0.95 (이론값과 근접)
n = 30,  α = 0.05 → 실측 coverage ≈ 0.97~1.00 (보수적 — 과추정)
```

---

## 12. 참고문헌

- **Conformal Prediction 기초**
  Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511*

- **Weighted Conformal Prediction (Distribution Shift)**
  Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS 2019. arXiv:1904.06019*

- **MedQA (USMLE 데이터셋)**
  Jin, D., Pan, E., Oufattole, N., Weng, W. H., Fang, H., & Szolovits, P. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Applied Sciences, 11*(14). arXiv:2009.13081

- **MedAbstain (LLM 불확실성 표현)**
  Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, X., Zhang, X., & Ye, H. (2023). PromptBench: Towards evaluating the robustness of large language models on adversarial prompts. *arXiv:2306.13063*

- **MIMIC-III (ICU 임상 데이터베이스)**
  Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data, 3*, 160035.

- **ReAct (추론+행동 에이전트)**
  Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023. arXiv:2210.03629*
