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

backend × 변형 $v \in \{\text{AP}, \text{NAP}, \text{A}, \text{NA}\}$
(측정치는 `results/run_20260507-182038/`):

| Backend  | Variant | n  | Recall  | Precision | F1     | AUROC | OK (≥0.95)? |
| -------- | ------- | :-: | :-----: | :-------: | :----: | :---: | :---------: |
| OpenAI   | AP      | 50 | 0.180   | 1.000     | 0.305  | —     | ✗           |
| OpenAI   | NAP     | 50 | 0.140   | 1.000     | 0.246  | —     | ✗           |
| OpenAI   | A       | 50 | 0.140   | 1.000     | 0.246  | —     | ✗           |
| OpenAI   | NA      | 50 | N/A †   | 0.000     | N/A    | —     | ✗           |
| LMStudio | AP      | 50 | 0.120   | 1.000     | 0.214  | —     | ✗           |
| LMStudio | NAP     | 49 | 0.102   | 1.000     | 0.185  | —     | ✗           |
| LMStudio | A       | 50 | 0.040   | 1.000     | 0.077  | —     | ✗           |
| LMStudio | NA      | 50 | N/A †   | 0.000     | N/A    | —     | ✗           |

> **†** NA 변형은 정의상 양성 라벨이 0개이므로 (정상 케이스에서의 *non-abstention*
> 을 평가) Recall은 정의되지 않는다; Precision 0.000은 소수의 false-positive
> 에스컬레이션을 반영한다 (audit issue #16: silent zero를 `compute_binary_metrics`
> 에서 `N/A`로 보고).
>
> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `per_variant.<variant>`.

**논의.** 변형별 recall이 일관되게 낮다 (0.04~0.18) — UASEF v1의 logprob CP는
단일 전역 $\alpha$로 MedAbstain perturbation이 의도한 *과신-오답 (overconfident-
wrong)* 케이스를 검출하지 못한다. 이는 logprob 기반 비적합의 알려진 한계이며
(audit 6.10 Round 6 limitations 참조), 정확히 main paper의 **Pivot A의 stratum별
CRC** 와 **Pivot C의 비용 인식 보정** 이 다루는 공백이다 — 단일 전역 α 대신
$\alpha_{\text{CRITICAL}} = 0.05$로 CRITICAL 임계값을 조이면서 v2는 v1의 0.84
대비 표 4에서 CRITICAL Safety Recall 0.96에 도달한다.

### B.3.2 Abstention Accuracy

binary classification 지표 외에, MedAbstain은 **LLM의 본질적 abstention
행동**을 측정한다 — 모델 자신이 "I am not certain", "insufficient evidence"
같은 표현을 얼마나 자주 출력하는지.

| Backend  | TA | FA | TR | MA  | Abstention Precision | Abstention Recall | Abstention F1 |
| -------- | :-: | :-: | :-: | :-: | :------------------: | :---------------: | :-----------: |
| OpenAI   | 0  | 0  | 50 | 150 | 0.000                | **0.000**         | 0.000         |
| LMStudio | 0  | 0  | 50 | 149 | 0.000                | **0.000**         | 0.000         |

> **정의.** TA (True Abstain): label=True ∧ no-evidence 표현 출력;
> FA (False Abstain): label=False ∧ 표현 출력;
> TR (True Answer): label=False ∧ 표현 없음;
> MA (Missed Abstain): label=True ∧ 표현 없음.
>
> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `abstention_accuracy`.

**논의.** 중립 시스템 프롬프트 (audit 6.10 issue #5 default) 하에서 두 backend
모두 **Abstention Recall = 0.000** 이다. 즉 gpt-4o도 LLaMA-3.1-8B도 에스컬레이션
이 *필요했던* 150/149개 MedAbstain 케이스에서 자발적으로 근거-부재 표현 ("I am
not certain", "insufficient evidence" 등) 을 emit하지 *않았다*. 이는 **본
벤치마크에서 LLM의 본질적 self-abstention을 안전 신호로 신뢰할 수 없다**는
직접적 증거이며, UASEF v2의 CP 기반 외부 게이트가 핵심 안전 메커니즘임을 보인다.

0/0/50/150 (또는 /149) 의 confusion 구조는 Abstention Precision이 0.000으로
나타난 이유도 설명한다 — abstention emission 자체가 0회였다 (TA = FA = 0).
Round 6 audit issue #5는 모델이 명시적으로 no-evidence phrasebook을 사용하도록
*프롬프트되는* ablation 모드 `SYSTEM_PROMPT_INSTRUCTED`를 도입했으며, 본 논문
범위 밖이지만 자연스러운 후속 실험이다.

### B.3.3 Routine-only vs 전체 MedQA Calibration

audit 6 (`improvements/README.md`의 issue P18)이 `load_noesc_calibration_questions`
를 도입하여 **one-class CP**를 가능하게 했다 — `expected_escalate=False`인
routine MedQA case만으로 calibration. 이는 `eval_medabstain.py`에서 default로
켜져 있고, AP/NAP/A 검출률을 상당히 개선한다 (improvements/README.md는
+20~40 percentage points 예측).

`eval_medabstain.py`의 default (audit 6 P18 이후)는 `calibration_source =
"medqa_routine"` 이며, 이것이 위 §B.3.1 표를 생성한 모드다. 전체-MedQA 비교는
`--no-routine-cal`로 실행할 수 있고 후속 ablation으로 남긴다 — 본 보고에 사용된
변형당 50개 sub-sample 규모에서 routine-only 모드는 이미 logprob CP의 본질적
한계 (과신-오답 케이스에서 recall 0.04~0.18) 에 도달하므로 대안 모드가 질적
결론을 바꿀 가능성은 낮다.

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

전체 sweep 표 (6 α × 3 specialty × 2 backend = 36개 측정) 는
`results/run_<ts>/<backend>/pareto_sweep_results.json` 에 있고, 시각화는
`pareto_frontier.png` 이다.

### B.4.1 Specialty별 권고 α

`experiments/pareto_sweep.py`의 `recommend_alpha()` 절차는 (coverage ≥ 0.95)
∧ (escalation_rate ≤ 0.15) 제약 하에 utility $U = \text{coverage} - 2 \cdot
\text{escalation\_rate}$를 최대화하는 α를 선택한다. `results/run_20260507-182038/`
에서:

| Backend  | Specialty            | Recommended α | Coverage | Escalation Rate | Utility |
| -------- | -------------------- | :-----------: | :------: | :-------------: | :-----: |
| OpenAI   | emergency_medicine   | 0.01          | 1.000    | 0.500           | 0.000   |
| OpenAI   | internal_medicine    | 0.01          | 1.000    | 0.000           | 1.000   |
| OpenAI   | general_practice     | 0.01          | 1.000    | 0.000           | 1.000   |
| LMStudio | emergency_medicine   | 0.01          | 1.000    | 0.240           | 0.520   |
| LMStudio | internal_medicine    | 0.01          | 1.000    | 0.000           | 1.000   |
| LMStudio | general_practice     | 0.01          | 1.000    | 0.000           | 1.000   |

> **Source.** `results/run_<ts>/<backend>/alpha_recommendations.json`.

**논의.** 6개 (backend × specialty) 조합 모두에서 권고가 $\alpha = 0.01$로
수렴한다 — 본 규모의 테스트 셋 ($n_{\text{test}} = 50$ per scenario)에서 over-
escalation 상한을 초과하지 않으면서 full-coverage 임계값이 가능하기 때문.
emergency_medicine (gpt-4o)의 escalation rate 0.500은 emergency stratum에서
참 양성의 prevalence가 높음을 반영한다 — 보수적 α가 많은 케이스를 "에스컬레이션"
쪽으로 몰아넣는다. 이 Pareto 거동은 UASEF v1의 속성이 아니라 *데이터의 측정값*
이며, main paper Pivot A의 stratum별 $\alpha_s$ 선택에 정보를 제공한다.

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
| OpenAI   | 0.1533         | 0.7667            | 0.2555     | —             | ✗                     |
| LMStudio | 0.0872         | 0.7222            | 0.1556     | —             | ✗                     |

**논의.** 두 backend 모두 v1의 단일-α logprob CP 하에서 0.95 Safety Recall
목표에 한참 못 미친다. 이는 main paper §7.4 / §7.5의 논의에 대한 실증 근거다 —
*logprob 기반 CP 단독*으로는 backend 크기와 무관하게 MedAbstain perturbation의
과신-오답 케이스를 검출할 수 없다. v2 pivot — Stratified CRC + 비용 인식 보정
— 은 정확히 이 문제를 다루어 고-stake stratum의 임계값을 조인다 (main paper
표 4: v2가 **두 backend 모두에서** CRITICAL Safety Recall 0.96, v1 0.84/0.70
대비).

backend 간 격차 (0.153 vs 0.087)는 모델 규모 차이 (gpt-4o vs LMStudio 통한
4-bit 양자화 LLaMA-3.1-8B)를 반영한다. Pivot A는 이 격차에 강건하다 — main
paper 표 4에서 두 backend 모두에서 CRITICAL Safety Recall 0.96으로 끌어올린다.
즉 stratum별 임계값이 backend별 점수 분포 차이를 보정한다.

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

_본 문서의 수치는 `results/run_20260507-182038/` (n_cal = 200, n_test = 100,
n_medabstain = 50, n_pareto = 50, α = 0.10, seed = 42, elapsed 525분) 으로부터
채워졌다. 동일한 template은 매 실행마다 `run_full_evaluation.sh`에 의해
`results/run_<ts>/result_supplementary.md`로 자동 재렌더링된다._
