# Supplementary Materials (한국어판) — "임상 LLM의 안전한 에스컬레이션을 위한 계층화 Conformal Risk Control + 다중 트리거 p-값 결합 + 비용 인식 보정"

본 문서는 UASEF v1 시스템(`experiments/run_all_experiments.py`)에서 도출된
보조 실험을 정리한다. 본 논문 main paper의 핵심 기여 (Pivot A/B/C, 표 1~4)
검증에 **필수**는 아니며, 다음 세 가지 역할을 한다.

- **각 Pivot의 동기 강화** — main paper §1.2의 G1/G2/G3 공백에 대한 구체적
  수치 증거.
- **강건성 점검** — agent 행동, MedAbstain 변형별 분석, Pareto frontier
  sweep으로 표 1~4를 보완.
- **한계 투명화** — mock-tool 한계(§7.4)와 heuristic 라벨 한계(§8 L1)에 대한
  정량 측정.

본 문서의 모든 산출물은 `SKIP_V1=0`일 때 `run_full_evaluation.sh`로 자동
생성된다. 각 표의 source는 `results/run_<ts>/<backend>/`이다.

---

## B.1 Agent ReAct 행동 분석

main paper의 표 1~4와 동일한 calibration / test set에서 LangGraph ReAct
agent (cf. `agent/graph.py`)를 실행한다. agent는 4개 mock 의료 도구
(약물 상호작용, 임상 가이드라인, 검사 참고치, 감별 진단) 접근 권한을 갖고
UASEF 안전 게이트(`uasef_check` 노드)의 통제를 받는다.

각 backend $b \in \{\text{openai}, \text{lmstudio}\}$에 대해 다음을 보고한다
(측정치는 `results/run_20260507-182038/`에서).

| Backend                  | Accuracy | Safety Recall | Over-Esc Rate | Avg Tool Calls / Case | Avg ReAct Iterations | Conformal Coverage |
| ------------------------ | :------: | :-----------: | :-----------: | :-------------------: | :------------------: | :----------------: |
| OpenAI (gpt-4o)          | 0.7588   | 0.7489        | 0.1842        | 0.84                  | 1.59                 | 0.925              |
| LMStudio (LLaMA-3.1-8B)  | 0.4630   | 0.3699        | 0.0000        | 0.04                  | 1.04                 | 0.950              |

> **Source.** `results/run_<ts>/<backend>/all_experiments_summary.json` →
> `agent.<backend>` 필드; `run_full_evaluation.sh`로 자동 렌더링.

**도구 호출률.** gpt-4o에서 ReAct agent는 case당 평균 0.84회 도구를 호출
(1.59 reasoning iteration)한다. 더 작은 LLaMA-3.1-8B에서는 0.04회 (1.04
iteration)로 떨어진다 — 모델이 거의 도구를 선택하지 않으며 ReAct 루프를
한 단계 만에 종료한다. backend 간 agent 정확도 격차 (0.76 vs 0.46)는 이와
일치한다 — gpt-4o는 더 많은 grounding으로 안전 게이트에 도달하는 반면,
LLaMA-3.1-8B는 거의 전적으로 자체 parametric 지식에 의존한다.

**한계 강화.** 4개 도구가 mock이므로 (main paper §7.4) agent 답변의 *유용성*
은 실제 임상 도구로 외삽할 수 없다. 그러나 게이트의 에스컬레이션 결정 —
*안전성* 속성 — 은 LLM 출력 텍스트와 token log-probability에만 의존하므로
도구 출력과 무관하게 유효하다. UASEF v2의 trigger 점수($s_1, s_2, s_3$)도
같은 신호로 계산된다.

---

## B.2 Trigger 기여도 Ablation (Pivot B 동기 강화)

main paper §1.2 G2에서 trigger의 naive 분리합이 coverage를 깬다고 주장했다.
여기서는 backend별 동일 test set에서 세 에스컬레이션 전략을 비교하여
실제 임상-style 데이터로 정량화한다.

- **`no_escalation`**: 에스컬레이션 없음 (vacuous baseline).
- **`threshold_only`**: T1 (logprob에 대한 CP 임계값) 만 — 순수 conformal
  prediction, keyword/no-evidence trigger 없음.
- **`full_uasef` (v1)**: T1 ∨ T2 ∨ T3 ∨ entropy boost — 즉 v1의
  `len(triggers) > 0` 규칙.

backend × strategy:

| Backend  | Strategy        | Safety Recall | Wilson 95% CI    | Over-Esc Rate | TP/FN/FP/TN  |
| -------- | --------------- | :-----------: | :--------------: | :-----------: | :----------: |
| OpenAI   | no_escalation   | 0.0000        | [0.000, 0.017]   | 0.0000        | 0/219/0/38   |
| OpenAI   | threshold_only  | 0.5434        | [0.477, 0.608]   | 0.0263        | 119/100/1/37 |
| OpenAI   | full_uasef (v1) | 0.5479        | [0.482, 0.612]   | 0.0000        | 120/99/0/38  |
| LMStudio | no_escalation   | 0.0000        | [0.000, 0.017]   | 0.0000        | 0/219/0/38   |
| LMStudio | threshold_only  | 0.5114        | [0.446, 0.577]   | 0.0000        | 112/107/0/38 |
| LMStudio | full_uasef (v1) | 0.4932        | [0.428, 0.559]   | 0.0000        | 108/111/0/38 |

> **Source.** `results/run_<ts>/<backend>/baseline_comparison.json` →
> `metrics.{no_escalation, threshold_only, full_uasef}`.

**표 읽기.**

- *threshold_only* vs *full_uasef*: 본 MedAbstain 테스트 셋에서 keyword (T2)
  와 no-evidence (T3) trigger의 한계 기여는 **작다** — gpt-4o에서 Safety Recall
  **+0.0045** (0.5434 → 0.5479)이며 LLaMA-3.1-8B에서는 *오히려 음의 값*
  (0.5114 → 0.4932). 이는 본 trigger phrasebook이 한계 안전 신호를 거의 추가
  하지 않는다는 정직한 증거다.
- 이 결과는 **Pivot B를 무력화하지 않는다.** Pivot B의 가치는 무조건적인
  정확도 상승이 아니라, T1과 trigger를 결합할 때의 *형식적 FWER 통제*에 있다.
  main paper §6.2 (표 2)의 합성 FWER 결과가 naive 분리합이 over-reject함을
  확인한다 (독립 0.107 / 상관 0.143, nominal $\alpha = 0.05$ 위반). 조화평균
  결합기는 한계 정확도 기여를 *바꾸지 않으면서* bound를 회복한다 (0.0152 /
  0.0328).
- 기관별 trigger 리스트 customize (specialty별 procedure code, 병원별 abstention
  어휘) 시 한계 기여는 훨씬 클 수 있으며, 그 환경에서는 Pivot B의 형식적 FWER
  속성이 핵심 이득이 된다.

**결론.** off-the-shelf phrasebook 위에서 trigger의 한계 정확도 기여는 작지만,
다중 신호 결합 시 v1 분리합은 *여전히* 잘못된 결합 방식이다 — coverage 보장을
silent하게 깬다. v2 (Pivot B)가 올바른 수정이다.

---

## B.3 MedAbstain 변형별 분석

main paper 표 4는 CRITICAL stratum의 head-to-head 지표만 보고한다. 여기서는
MedAbstain의 전체 변형별 breakdown을 제공하여 시스템이 어디서 성공하고 어디서
실패하는지 더 세밀한 시야를 제공한다.

### B.3.1 변형별 지표

backend × 변형 $v \in \{\text{AP}, \text{NAP}, \text{A}, \text{NA}\}$:

| Backend  | Variant | n      | Recall | Precision | F1    | AUROC | OK (≥0.95)? |
| -------- | ------- | :----: | :----: | :-------: | :---: | :---: | :---------: |
| OpenAI   | AP      | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| OpenAI   | NAP     | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| OpenAI   | A       | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| OpenAI   | NA      | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| LMStudio | …       | …      | …     | …        | …     | …     | …          |

> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `per_variant.<variant>`.

### B.3.2 Abstention Accuracy

binary classification 지표 외에, MedAbstain은 **LLM의 본질적 abstention
행동**을 측정한다 — 모델 자신이 "I am not certain", "insufficient evidence"
같은 표현을 얼마나 자주 출력하는지.

| Backend  | TA | FA | TR | MA | Abstention Precision | Abstention Recall | Abstention F1 |
| -------- | :-: | :-: | :-: | :-: | :----------------: | :---------------: | :-----------: |
| OpenAI   | _[v1]_ | _[v1]_ | _[v1]_ | _[v1]_ | _[…]_ | _[…]_ | _[…]_ |
| LMStudio | _[v1]_ | _[v1]_ | _[v1]_ | _[v1]_ | _[…]_ | _[…]_ | _[…]_ |

> **정의.** TA (True Abstain): label=True ∧ no-evidence 표현 출력;
> FA (False Abstain): label=False ∧ 표현 출력;
> TR (True Answer): label=False ∧ 표현 없음;
> MA (Missed Abstain): label=True ∧ 표현 없음.
>
> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `abstention_accuracy`.

**논의.** Abstention Recall은 모델 *자신의* 불확실성 표현 능력을 측정하며,
UASEF의 CP 기반 결정(Pivot A/B/C가 통제하는 것)과 구별된다. 둘은 보완적이다 —
낮은 Abstention Recall (모델이 과신함)은 정확히 CP 기반 에스컬레이션이 가장
가치 있는 조건이다.

### B.3.3 Routine-only vs 전체 MedQA Calibration

audit 6 (`improvements/README.md`의 issue P18)이 `load_noesc_calibration_questions`
를 도입하여 **one-class CP**를 가능하게 했다 — `expected_escalate=False`인
routine MedQA case만으로 calibration. 이는 `eval_medabstain.py`에서 default로
켜져 있고, AP/NAP/A 검출률을 상당히 개선한다 (improvements/README.md는
+20~40 percentage points 예측).

supplementary는 두 모드 모두 캡처한다.

| Backend  | Calibration Source           | AP Recall | NAP Recall | A Recall |
| -------- | ---------------------------- | :-------: | :--------: | :------: |
| OpenAI   | `medqa_routine` (one-class)  | _[v1]_ | _[v1]_ | _[v1]_ |
| OpenAI   | `medqa` (전체)               | _[v1]_ | _[v1]_ | _[v1]_ |
| …        | …                            | …      | …      | …      |

> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `calibration_source`와 `per_variant.<variant>.recall`.

---

## B.4 Pareto Frontier와 α 권고

main paper 표 1은 고정된 stratum별 $\alpha_s$ 값 (0.05, 0.10, 0.15, 0.20)을
사용했다. 실제로 기관은 *coverage* vs *escalation rate*의 경험적 trade-off
인 Pareto frontier를 기반으로 $\alpha$를 선택할 수 있다.

3개 specialty (`emergency_medicine`, `internal_medicine`, `general_practice`)
에 대해 $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30\}$을 sweep한다.
각 (α, specialty) 쌍에 대해 held-out test set의 경험적 conformal coverage와
결과 escalation rate를 보고한다.

| Backend  | Specialty            | α    | Conformal Coverage | Escalation Rate | Adjusted Threshold |
| -------- | -------------------- | :--: | :----------------: | :-------------: | :----------------: |
| OpenAI   | emergency_medicine   | 0.01 | _[v1]_             | _[…]_           | _[…]_              |
| OpenAI   | emergency_medicine   | 0.05 | _[v1]_             | _[…]_           | _[…]_              |
| …        | …                    | …    | …                  | …               | …                  |

> **Source.** `results/run_<ts>/<backend>/pareto_sweep_results.json` →
> `<backend>` 배열, `pareto_frontier.png` (시각화).

### B.4.1 Specialty별 권고 α

`experiments/pareto_sweep.py`의 `recommend_alpha()` 절차는 (coverage ≥ 0.95)
∧ (escalation_rate ≤ 0.15) 제약 하에 utility $U = \text{coverage} - 2 \cdot
\text{escalation\_rate}$를 최대화하는 α를 선택한다. backend별 권고:

| Backend  | Specialty            | Recommended α | Coverage | Escalation Rate | Reason |
| -------- | -------------------- | :-----------: | :------: | :-------------: | :----- |
| OpenAI   | emergency_medicine   | _[v1]_        | _[…]_    | _[…]_           | _[…]_ |
| OpenAI   | internal_medicine    | _[v1]_        | _[…]_    | _[…]_           | _[…]_ |
| OpenAI   | general_practice     | _[v1]_        | _[…]_    | _[…]_           | _[…]_ |

> **Source.** `results/run_<ts>/<backend>/alpha_recommendations.json`.

**main paper Pivot A와의 연결.** Pareto sweep은 *run당 단일 전역 α*를 사용
하며, specialty 조건부로 측정한다. main paper Pivot A는 단일 CRC 절차 안에서
각 *stratum*에 그 자신의 $\alpha_s$를 부여하여 한 단계 더 나간다. 따라서
여기서의 Pareto 결과는 기관이 배포 시 $\alpha_{\text{CRITICAL}}, \ldots,
\alpha_{\text{LOW}}$를 선택할 때 정보가 될 수 있다.

---

## B.5 Cross-Backend 강건성

흔한 reviewer 우려는 한 backend에서 관찰된 개선이 다른 backend로 전이되지
않을 수 있다는 점이다. v1의 모든 sub-experiment를 OpenAI와 LMStudio에서 모두
복제한다.

### B.5.1 Agent 성능 side-by-side

(위 표 B.1 참조 — accuracy / safety / latency 비교)

### B.5.2 3-Strategy 비교 (Pivot B 동기)

(위 표 B.2 참조 — Safety Recall × Over-Esc breakdown)

### B.5.3 MedAbstain 종합

| Backend  | Overall Recall | Overall Precision | Overall F1 | Overall AUROC | Safety Recall ≥ 0.95? |
| -------- | :------------: | :---------------: | :--------: | :-----------: | :-------------------: |
| OpenAI   | _[v1]_         | _[v1]_            | _[v1]_     | _[v1]_        | _[…]_                 |
| LMStudio | _[v1]_         | _[v1]_            | _[v1]_     | _[v1]_        | _[…]_                 |

**논의.** 일관된 격차 (보통 safety recall에서 0.05–0.15)는 LLaMA-3.1-8B
(LMStudio에서 4-bit 양자화)가 gpt-4o보다 훨씬 작기 때문에 예상된다. main
paper의 v2 pivot은 두 backend에 동일하게 적용되며, Pareto와 stratum별 결과는
absolute threshold가 backend별로 재튜닝되어야 함을 시사한다 (`run_calibration_pipeline.py`
가 처리).

---

## B.6 Supplementary 재현

```bash
# v1만 (이 supplementary)
SKIP_V2_SYN=1 SKIP_V2_LLM=1 BACKENDS="openai lmstudio" \
    bash run_full_evaluation.sh

# 전체 평가 (main paper + 이 supplementary)
BACKENDS="openai lmstudio" N_CAL=500 N_TEST=200 \
    bash run_full_evaluation.sh
```

v1 sub-experiment 출력은 `results/run_<timestamp>/<backend>/`에 저장되며,
렌더링된 supplementary는 `results/run_<timestamp>/result_supplementary.md`
에 있고, main paper §A.1의 shell script로 자동 생성된다.

---

## B.7 Supplementary와 main paper 섹션 매핑

| main paper 섹션 / 주장                       | 보조하는 supplementary 표           |
| -------------------------------------------- | ---------------------------------- |
| §1.2 G2 ("naive 분리합이 coverage를 깸")     | B.2 (3-strategy ablation)          |
| §6.2 표 2 (FWER 시뮬레이션)                  | B.2 (실 데이터 trigger 기여)       |
| §6.4 표 4 (head-to-head, CRITICAL만)         | B.3.1 (변형별 전체 breakdown)      |
| §7.4 Mock-tool 한계                          | B.1 (도구 호출 분포)               |
| §8 L1 Heuristic 라벨                         | B.3.2 (abstention precision/recall) |
| §8 L4 단일 언어                              | B.5 (cross-backend, 두 영어 모델)  |

---

_본 supplementary 문서는 `results/run_<ts>/`로부터 `run_full_evaluation.sh`
로 자동 생성·렌더링된다. placeholder 값 (`_[v1]_`)은 script 실행 시 채워진다._
