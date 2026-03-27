# UASEF

Uncertainty-Aware Safe Escalation Framework for Medical LLM Agents

LLM 의료 에이전트가 불확실성을 정량화하고, 위험도를 판단하여 인간 전문가에게 자동으로 인계하는 연구 프레임워크입니다.

---

## 핵심 구조

```
UASEF/
├── models/                         # 핵심 모듈
│   ├── model_interface.py          # LMStudio / OpenAI 통합 레이어
│   ├── uqm.py                      # Uncertainty Quantification Module (CP 기반)
│   └── rtc_ede.py                  # Risk-Threshold Calibrator + Escalation Decision Engine
│
├── agent/                          # LangGraph ReAct 에이전트
│   ├── graph.py                    # StateGraph 조립 (START → reason → act → uasef_check → END)
│   ├── nodes.py                    # 노드 함수 + AgentComponents
│   ├── state.py                    # MedicalAgentState TypedDict
│   └── tools.py                    # 의료 도구 4종 (drug interaction, guideline, lab, DDx)
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
│   ├── run_experiment.py           # 순차 파이프라인 실험
│   ├── run_agent_experiment.py     # LangGraph 에이전트 실험
│   ├── eval_medabstain.py          # MedAbstain AP/NAP 분류 정확도 평가
│   ├── pareto_sweep.py             # α sweep → Pareto frontier + α 권고
│   └── visualize_results.py        # 결과 시각화
│
├── results/                        # 실험 결과 (자동 생성, .gitignore)
├── pyproject.toml                  # uv 의존성
└── .env.example
```

---

## 설치

```bash
# uv 설치 (없으면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 환경 변수 설정
cp .env.example .env
# .env에서 OPENAI_API_KEY, LMSTUDIO_MODEL 수정
```

### LMStudio 사용 시

1. LMStudio 앱 실행 → 모델 다운로드 (권장: `meta-llama-3.1-8b-instruct`)
2. **Local Server** 탭 → **Start Server** (기본 포트: 1234)
3. `.env`의 `LMSTUDIO_MODEL`을 로드된 모델명으로 수정

### LangSmith 트레이싱 (선택)

```bash
# .env에 추가
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=UASEF-agent
```

---

## 실행

### 순차 파이프라인 실험

```bash
# 기본 실행 (base_config.yaml 사용)
python experiments/run_experiment.py

# 설정 오버라이드
python experiments/run_experiment.py --config experiments/configs/scenario_emergency.yaml
python experiments/run_experiment.py --n-cal 500 --n-test 50 --scoring-method logprob

# 결과 시각화
python experiments/visualize_results.py
```

### LangGraph 에이전트 실험

```bash
# 기본 실행
python experiments/run_agent_experiment.py

# 단일 백엔드, 인자 조정
python experiments/run_agent_experiment.py --backend openai --n-cal 500 --n-test 50
python experiments/run_agent_experiment.py --backend lmstudio --scoring-method logprob
```

### MedAbstain 분류 정확도 평가

```bash
# 전체 변형 (AP, NAP, A, NA)
python experiments/eval_medabstain.py --backend openai

# AP/NAP만 (핵심 safety 케이스)
python experiments/eval_medabstain.py --backend openai --variants AP NAP --n 100

# Weighted CP 비교
python experiments/eval_medabstain.py --backend openai --weighted-cp
```

### Pareto Frontier + α 권고

```bash
# α sweep 실행 → frontier 시각화 → specialty별 최적 α 자동 권고
python experiments/pareto_sweep.py --backend openai --n-cal 500

# 기존 sweep 결과에서 권고만 재계산
python -c "
from experiments.pareto_sweep import recommend_alpha, print_recommendations
recs = recommend_alpha()
print_recommendations(recs)
"
```

---

## 주요 출력 파일

| 파일 | 생성 스크립트 | 설명 |
| ---- | ----------- | ---- |
| `results/experiment_results.json` | `run_experiment.py` | 순차 파이프라인 전체 결과 |
| `results/comparison_table.csv` | `run_experiment.py` | 백엔드 비교 요약표 |
| `results/agent_results.json` | `run_agent_experiment.py` | 에이전트 실험 결과 |
| `results/agent_comparison_table.csv` | `run_agent_experiment.py` | 에이전트 비교 요약 |
| `results/medabstain_eval.json` | `eval_medabstain.py` | MedAbstain AP/NAP 정밀 평가 |
| `results/medabstain_eval_summary.csv` | `eval_medabstain.py` | 변형별 Precision/Recall/F1 |
| `results/pareto_sweep_results.json` | `pareto_sweep.py` | α × specialty 실측 데이터 |
| `results/pareto_frontier.png` | `pareto_sweep.py` | Pareto frontier 시각화 |
| `results/alpha_recommendations.json` | `pareto_sweep.py` | specialty별 최적 α 권고 |

---

## 아키텍처

### 세 가지 핵심 모듈

#### UQM — Uncertainty Quantification Module

Conformal Prediction으로 통계적 coverage 보장이 있는 불확실성을 측정합니다.

```python
uqm = UQM(backend="openai", alpha=0.05, scoring_method="logprob")
uqm.calibrate(cal_questions, distribution_source="medqa")  # q̂ 계산
result = uqm.evaluate(question, distribution_source="medqa")
# result.nonconformity_score, result.should_escalate, result.weighted_cp_used
```

| 파라미터 | 옵션 | 의미 |
| ------- | ---- | ---- |
| `scoring_method` | `"logprob"` (primary) | -mean(token logprobs), CP 보장 |
| | `"self_consistency"` (ablation) | Jaccard diversity × N queries |
| `use_weighted_cp` | `True` | 분포 이동 시 Weighted CP 자동 사용 |
| `alpha` | 0.01 ~ 0.30 | 1-α = coverage 보장 수준 |

분포 이동 감지: `calibrate(distribution_source="medqa")` 후 `evaluate(distribution_source="mimic3")` 호출 시 `UserWarning` + Weighted CP 자동 전환.

#### RTC — Risk-Threshold Calibrator

전문과목 위험도에 따라 임계값을 동적으로 조정합니다.

| 위험도 | 배율 | 전문과목 |
| ----- | ---- | ------- |
| CRITICAL | ×0.60 | 응급의학, 중환자의학 |
| HIGH | ×0.75 | 심장내과, 신경과, 종양학 |
| MODERATE | ×1.00 | 내과, 외과 |
| LOW | ×1.30 | 일반 외래, 예방의학 |

`emergency` / `rare_disease` 시나리오에는 추가 ×0.85 적용.

#### EDE — Escalation Decision Engine

3가지 트리거 중 하나라도 활성화되면 에스컬레이션:

1. `UNCERTAINTY_EXCEEDED` — nonconformity_score > adjusted_threshold (CP 직접 신호)
2. `HIGH_RISK_ACTION` — 고위험 키워드 감지 (intubation, alteplase, norepinephrine 등)
3. `NO_EVIDENCE` — 근거 부재 표현 감지 ("I am not certain", "off-label" 등)

엔트로피 > 2.0 nats/token 시 confidence +0.15 boosting (에스컬레이션 트리거 아님).

### LangGraph 에이전트 구조

```text
START → reason → (tool_calls?) → act ──┐
          ↑                             │
          └─────────────────────────────┘
          ↓ (최종 답변 or 반복 한계)
       uasef_check  ← 원본 질문 독립 재판
       ↙          ↘
  escalate      finalize
     ↓               ↓
   END             END
```

`uasef_check` 노드는 에이전트 메시지 히스토리와 **독립적으로** 원본 질문을 재평가합니다. 에이전트가 도구로 정보를 수집했더라도 UASEF는 별도로 판단합니다.

---

## 데이터

### 자동 로딩 우선순위

```text
1. data/raw/*.jsonl       (로컬 파일)
2. HuggingFace datasets   (자동 다운로드)
3. 내장 fallback          (개발/테스트 전용)
```

### 데이터셋별 설치

| 데이터셋 | 용도 | 설치 방법 |
| ------- | ---- | ------- |
| **MedQA** | Calibration + 기본 시나리오 | `pip install datasets` 후 자동 다운로드 |
| **MedAbstain** | AP/NAP safety 평가 | [GitHub](https://github.com/HowieSiao/medabstain) |
| **MIMIC-III** | 분포 이동 실험 | [PhysioNet DUA 필요](https://physionet.org/content/mimiciii/1.4/) |

자세한 내용은 [data/README.md](data/README.md)를 참고하세요.

### MIMIC-III 실험 시 주의

CP 보장을 유지하려면 반드시 MIMIC 데이터로 재보정해야 합니다.

```python
from data.loader import load_mimic_calibration
cal = load_mimic_calibration(n=500)
uqm.calibrate(cal, distribution_source="mimic3")    # ✓
uqm.evaluate(question, distribution_source="mimic3") # ✓
```

---

## 논문 권장 설정

```yaml
# experiments/configs/base_config.yaml
uqm:
  alpha: 0.05
  scoring_method: logprob   # primary; self_consistency는 ablation 전용
  holdout_fraction: 0.2
data:
  n_calibration: 500        # CP 보장 실용 하한
  n_test_per_scenario: 50   # 시나리오별 케이스 수
```

> **현재 기본값(`n_calibration=30`)은 개발용입니다.**
> CP 이론 보증이 실용 수준으로 작동하려면 n ≥ 500을 권장합니다.

---

## 평가 지표

| 지표 | 목표 | 의미 |
| ---- | ---- | ---- |
| **Safety Recall** | ≥ 0.95 | 에스컬레이션해야 할 케이스를 놓치지 않음 (타협 불가) |
| **Over-Escalation Rate** | ≤ 0.15 | 자율 처리 가능한 케이스를 불필요하게 넘기지 않음 |
| **Conformal Coverage** | ≥ 1-α | hold-out에서 실측 coverage ≥ 이론값 |
| **AUROC (MedAbstain)** | — | AP/NAP 분류 순위 성능 |

---

## 개별 모듈 테스트

```bash
# 모델 연결 확인
python models/model_interface.py

# UQM 단독 테스트 (logprob vs self_consistency 비교)
python models/uqm.py

# RTC + EDE 단독 테스트
python models/rtc_ede.py
```

---

## 참고문헌

- Angelopoulos & Bates (2021). *A gentle introduction to conformal prediction.* arXiv:2107.07511
- Tibshirani et al. (2019). *Conformal Prediction Under Covariate Shift.* NeurIPS 2019.
- Jin et al. (2021). *What disease does this patient have?* (MedQA) arXiv:2009.13081
- Zhu et al. (2023). *Can LLMs express their uncertainty?* (MedAbstain) arXiv:2306.13063
- Johnson et al. (2016). *MIMIC-III, a freely accessible critical care database.* Scientific Data.
- Yao et al. (2023). *ReAct: Synergizing reasoning and acting in language models.* ICLR 2023.
