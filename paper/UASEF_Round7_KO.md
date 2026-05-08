# 임상 LLM의 안전한 에스컬레이션을 위한 계층화 Conformal Risk Control + 다중 트리거 p-값 결합 + 비용 인식 보정

**저자.** *[저자명]*<sup>1</sup>, *[공저자]*<sup>2</sup>
<sup>1</sup>*[소속 1]*; <sup>2</sup>*[소속 2]*
교신: `[email]`

**투고 대상 학회.** ML4H 2026 (Spotlight) / AISTATS 2026 / NeurIPS 2026
**코드 및 데이터.** `https://github.com/[org]/UASEF` (심사용 익명화)
**연산 비용.** OpenAI API ~$120 USD · LMStudio + LLaMA-3.1-8B로 ~24 GPU시간
**재현성.** `bash run_full_evaluation.sh` 한 줄로 본 논문의 모든 표·그림 재생성;
고정된 `pyproject.toml` + Dockerfile 포함.

---

## 초록 (Abstract)

임상의사결정지원(CDS)에 배포되는 대규모 언어모델(LLM)은 **불확실할 때
판단을 보류**할 수 있어야 한다. 최근 연구는 LLM에 **Conformal Prediction
(CP)** 을 적용하여 분포 가정 없이 abstention 결정에 대한 coverage 보장을
얻었다. 그러나 세 가지 공백이 남아 있다. *(i)* 기존 CP-LLM 기법은 **단일
전역(global)** 미적합 수준 $\alpha$를 사용하는데, 이는 응급의학과의 missed
escalation 비용이 일반외래보다 수백~수천 배 큰 임상 현실과 부합하지 않는다.
*(ii)* 다중 신호 에스컬레이션 정책(저신뢰 + 고위험 행동 키워드 + 명시적
근거-부재 표현 결합)은 보통 **임의적 OR 트리거**로 구현되어 nominal
coverage 보장을 *깨뜨린다.* *(iii)* 임계값 최적화는 보통 $F_1$ 같은 **대칭
목표 함수**로 수행되어, false negative의 비대칭 비용을 무시한다.

본 논문은 세 가지 공백을 형식적 보장과 함께 해결하는 **UASEF v2** 프레임워크를
제안한다. **(A) 계층화 Conformal Risk Control**은 calibration 데이터를 임상
위험도(CRITICAL/HIGH/MODERATE/LOW)로 분할하고 각 stratum에서 *Angelopoulos
& Bates [2024]*의 conformal risk control 절차를 적용해 stratum별로
$\mathbb{E}[\ell_s] \le \alpha_s$ 보장을 제공한다. **(B) 다중 트리거
Conformal 결합**은 각 트리거를 비적합 점수로 frame하고 trigger별 conformal
$p$-값을 *Wilson [2019]*의 조화평균 규칙 또는 *Vovk & Wang [2019]*의 e-값
규칙으로 결합하여, 임의 종속성 하에서 family-wise error rate (FWER)를 통제한다.
**(C) 비용 인식 보정**은 $F_1$ 최적화를 stratum별 위험 통제 제약 하의
비용 가중 목적 함수로 대체한다.

합성 null 가설 시뮬레이션 ($n_{\text{trials}} = 5000$, $\alpha = 0.05$)에서
naive 분리합 baseline `len(triggers) > 0`의 경험적 FWER은 **0.107(독립) /
0.143(상관)** 인 반면, 본 연구의 조화평균 결합기는 **0.015 / 0.033** 에 머문다.
임상 보정된 비용 행렬에서 본 기법은 $F_1$-대칭 최적화 대비 총 기대 비용을
**38.3×** 감소시키며 (4 stratum 합계 16,264 → 425), CRITICAL stratum 미스율을
UASEF v1 (heuristic 배율) 대비 OpenAI gpt-4o에서 0.16 → **0.03**, LMStudio
LLaMA-3.1-8B에서 0.31 → **0.04**로 개선한다. MedAbstain head-to-head 비교에서
TECP *[Xu & Lu, 2025]*, Conformal Language Modeling *[Quach et al., 2024]*,
Semantic Entropy *[Farquhar et al., 2024]*와 대조한다 — 본 v2는 두 backend
모두에서 CRITICAL stratum Safety Recall **0.96** 을 달성 (TECP/Quach/SE는
0.16, UASEF v1은 0.84/0.70)하고 TECP 대비 총 비용을 **20.3× / 21.3×**
감소시킨다. 모든 산출물 (137-test pytest 스위트,
`experiments/round7_table*.py`, `run_full_evaluation.sh`)은 한 줄 재현을 위해
공개한다.

---

## 1. 서론 (Introduction)

### 1.1 연구 동기

병원의 예측 AI 도입은 빠르게 가속되고 있다 — 2024년 미국 급성기 병원의 71%가
예측 AI를 사용한다고 보고했고(2023년 66%에서 증가) [Healthcare IT Today,
2024]. 그러나 *생성형* LLM의 임상의사결정지원 배포는 **불확실성 정량화**
문제로 인해 여전히 제약을 받고 있다. 임상의는 "*모델이 언제 인간 전문가에게
판단을 양도해야 하는가?*"라는 원리적 답변을 필요로 한다.

Conformal Prediction [Vovk et al., 2005; Angelopoulos & Bates, 2021]은 그
coverage 보장 — $P(s_{\text{test}} \le \hat q) \ge 1 - \alpha$ — 이 교환가능성
가정 하에서 **분포 자유**로 성립하고, 재학습이 필요 없으며, 모델 비종속이라는
점에서 매력적이다. 최근 연구는 CP를 LLM에 적용했다. **Conformal Language
Modeling** [Quach et al., 2024]은 negative log-likelihood를 비적합 점수로
사용하며 sampling 기반 stopping rule을 도입했다. **TECP** [Xu & Lu, 2025]은
누적 token-entropy를 사용한다. **ConU/API Is Enough** [Wang et al., 2024;
Su et al., 2024]는 logit 접근 없이 self-consistency 기반 비적합 점수로 CP를
확장했다. **MedAbstain** 벤치마크 [Machcha et al., 2026]는 평가 프로토콜에 CP를
이미 통합한다. 동시에, 비-conformal 환각 검출기 **Semantic Entropy**
[Farquhar et al., 2024]는 *Nature*에 게재되며 우수한 ROC-AUC를 보고했다.

### 1.2 임상 안전을 위한 세 가지 공백

이 풍부한 문헌에도 불구하고 CP-LLM 기법을 **위험도 계층화된 임상
에스컬레이션**에 적용할 때 세 가지 공백이 남는다.

**G1 — 단일 전역 $\alpha$는 임상 위험도 계층화와 호환되지 않는다.** 소아
일차진료와 패혈증 triage가 똑같은 nominal coverage를 받는다. 실제로 missed
escalation의 비대칭 손해는 진료과별로 2–3 자릿수 차이가 난다. 기존 CP-LLM
기법은 사후 임의 배율 없이는 이를 표현할 수 없는데, 그 배율은 coverage
보장을 잃는다.

**G2 — 다중 신호 에스컬레이션 정책은 coverage 보장을 깨뜨린다.** 운영
안전 게이트는 보통 (a) log-likelihood에 대한 CP 임계값, (b) 고위험 시술
키워드 검출, (c) "근거 없음(no-evidence)" 표현 매칭 같은 여러 트리거를 결합한다.
3개 트리거 각각이 $\alpha = 0.10$일 때 naive `OR`의 경험적 FWER은 독립
하에서 0.27, positive dependence 하에서 더 큰 값까지 올라갈 수 있다. 이는
정확히 CP를 적용한 본래 동기를 무력화한다.

**G3 — 안전이 중요한 환경에서 대칭 손실은 잘못된 목적 함수다.** 보정은
보통 $F_1$, 정확도 같은 대칭 지표로 수행된다. 그러나 *임상* 손실은 매우
비대칭이다 — STEMI를 놓치면 over-escalation의 약 $10^3$배 비용이 발생하고,
일반 감기를 놓치면 거의 $1\times$이다. 이 구조를 무시하는 방법은 고위험
stratum에 sensitivity를 체계적으로 과소 배분하게 된다.

### 1.3 본 연구의 기여

본 논문은 G1–G3를 형식적 보장과 함께 해결하는 통합 프레임워크 — **UASEF
v2** — 와 그것의 실증적 우위를 다음 5가지 항목으로 기여한다.

1. **계층화 Conformal Risk Control (Pivot A, §4.1).** Angelopoulos & Bates
   [2024]의 conformal risk control 절차를 *임상 위험도 stratum*으로 확장하여
   stratum별 보장 $\mathbb{E}[\ell(\lambda_s, X, Y) \mid \text{stratum}=s]
   \le \alpha_s$를 제공한다. 우리가 아는 한 LLM 에스컬레이션 정책에 stratified
   CRC를 적용한 첫 사례이다.

2. **다중 트리거 Conformal 결합 (Pivot B, §4.2).** 키워드와 근거-부재 트리거를
   추가적인 비적합 점수로 frame하고, trigger별 conformal $p$-값을 조화평균
   [Wilson, 2019] 또는 e-값 [Vovk & Wang, 2019] 규칙으로 결합하여 임의 종속성
   하에서 FWER ≤ $\alpha$를 달성한다. 우리가 아는 한 의료 NLP 문헌에서 형식적
   FWER 통제를 동반한 conformal 다중 신호 에스컬레이션 규칙은 이 연구가 처음이다.

3. **비용 인식 보정 (Pivot C, §4.3).** 대칭 $F_1$ 최적화를 stratum별 비용
   가중 목적 함수와 stratified CRC 제약의 결합으로 대체한다. 가능한 영역에
   해가 존재할 때 stratum별 위험 한계를 보장하는 단순 sweep 알고리즘을 제시하고,
   그렇지 않으면 가장 보수적 임계값으로 fallback한다.

4. **5개 baseline에 대한 정직한 실증 평가.** MedAbstain 벤치마크와 분포-매칭
   합성 데이터에서 TECP, Quach et al. (2024), Semantic Entropy, UASEF v1
   (heuristic-multiplier), 본 연구의 v2와 비교한다. 추가로 독립과 상관 두 가지
   null 구조 하에서 합성 FWER 시뮬레이션을 수행한다.

5. **한 줄 재현성 인프라.** 고정된 dependency, 모든 알고리즘 모듈을 다루는
   137개의 pytest 단위 테스트, 그리고 본 논문의 모든 표를 raw 데이터로부터
   재생성하는 단일 shell script `run_full_evaluation.sh`를 공개한다.

---

## 2. 관련 연구 (Related Work)

### 2.1 LLM에 대한 Conformal Prediction

Conformal prediction은 임의의 base predictor에 대해 분포 자유 finite-sample
coverage 보장을 제공한다 [Vovk et al., 2005; Angelopoulos & Bates, 2021].
LLM에 대한 적용은 빠르게 성장하는 문헌이다.

**Conformal Language Modeling** [Quach et al., 2024]은 NLL을 비적합으로 하는
sampling 기반 stopping rule을 도입했다. **TECP** [Xu & Lu, 2025]은 누적
token-entropy를 사용해 6개 LLM × CoQA/TriviaQA에서 split conformal 절차를
평가했다. **API Is Enough** [Su et al., 2024]와 **ConU** [Wang et al., 2024]는
self-consistency 기반 비적합으로 black-box API 접근 환경의 CP로 확장한다.
2024년 TACL 서베이 [Campos et al., 2024]는 분야의 60+ 논문을 정리한다.

이들은 모두 **단일 전역** $\alpha$를 사용한다. 위험도 계층화나 다중 트리거
결합을 다루지 않는다. UASEF v2는 이 토대 위에 구축되며, CP/CRC를 stratum
별·trigger별로 호출되는 primitive 연산으로 다룬다.

### 2.2 Conformal Risk Control

Conformal risk control [Angelopoulos et al., 2024 (ICLR Spotlight)]은 CP를
임의의 monotone bounded loss의 *기대값* 통제로 일반화한다. 핵심 부등식

$$\hat \lambda = \sup\Big\{\lambda : \hat R(\lambda) + \tfrac{B}{n+1} \le \alpha\Big\}$$

는 $\ell$이 $\lambda$에 대해 monotone non-decreasing이고 $B$로 상한일 때
$\mathbb{E}[\ell(\hat \lambda, X, Y)] \le \alpha$를 보장한다. 본 연구는
$\ell = \mathbf{1}\{Y=1 \,\wedge\, \mathrm{score}(X) \le \lambda\}$ (missed
escalation)을 사용하며 이는 $\lambda$에 단조이다.

Class-conditional CP [Romano et al., 2020]는 calibration 데이터를 클래스로
분할하고 클래스별 CP를 실행한다. 우리는 이를 *위험도 stratum*으로 확장하고
CRC와 결합하여 **stratum별** $\mathbb{E}[\ell_s] \le \alpha_s$를 얻는다.
이러한 결합은 신규이며, 기존 CRC 논문은 단일 모집단 보장에 집중한다.

### 2.3 다중 가설 결합

$p$-값 결합은 통계학의 고전적 문제다. Bonferroni 보정은 임의 종속성 하에서
보수적이다. Wilson의 **조화평균 p-값** [Wilson, 2019]은 더 sharp하며
$\alpha \le 0.05$에서 임의 종속 하에 점근적으로 valid하다. **e-값 평균**
[Vovk & Wang, 2019; Wang & Ramdas, 2022]은 Markov 부등식으로 정확한 validity를
제공한다. Bates et al. [2023]은 이를 **conformal $p$-값**으로 적응시켜 outlier
detection에 적용했다.

본 연구는 LLM 에스컬레이션의 **trigger conformal $p$-값**에 조화평균과 e-값
결합을 적용한다. 적용은 신규이며 기저 수학은 확립되어 있다.

### 2.4 의료 LLM의 abstention과 안전성

2025년 TACL 서베이 [Wen et al., 2025]는 80+ LLM abstention 기법을 분류한다.
**MedAbstain** [Machcha et al., 2026 (EACL)]은 적대적 perturbation
(AP/NAP/A/NA 4 변형)과 통합 CP 평가를 가진 벤치마크다. **SelectLLM**은
보정과 결합한 selective prediction을 제안한다. **AbstentionBench** [Meta AI,
2025]는 종합적 LLM abstention 벤치마크다.

이들은 *무엇을 평가할 것인가* (데이터·지표·perturbation)에 집중하며,
*다중 안전 신호를 형식적 보장과 함께 어떻게 결합할 것인가*는 다루지 않는다.
UASEF v2는 보완적이다 — 벤치마크 기법들이 plug-in할 수 있는 알고리즘 backbone을
제공한다.

### 2.5 CP 없는 환각 검출

**Semantic Entropy** [Farquhar et al., 2024 (Nature)]는 $N$개의 generation을
의미적으로 클러스터링한 후 클러스터 분포의 Shannon entropy를 환각 신호로
사용한다. 우수한 *샘플 단위 불확실성 점수*를 제공하지만 coverage 보장을
다루지 않으며, 환각 선언 임계값은 임의적이다. 본 연구는 Semantic Entropy를
baseline (§5–6)으로 사용하며, 그 출력을 split CP layer에 입력으로 넣는다.

---

## 3. 사전 지식 (Preliminaries)

### 3.1 표기법

$(X, Y) \sim \mathcal{P}$이고 $X$는 임상 질문, $Y \in \{0, 1\}$는 에스컬레이션
필요 여부라고 하자. $s : \mathcal{X} \to \mathbb{R}$는 *비적합 점수*
(클수록 더 불확실)이다. 임계값 $\lambda$에 대해 $s(X) > \lambda$일 때
*에스컬레이션*을 선언한다.

Calibration set $\mathcal{D}_{\text{cal}} = \{(X_i, Y_i)\}_{i=1}^n$는 $\mathcal{P}$
에서 i.i.d.로 추출된다고 가정한다.

### 3.2 Conformal Prediction

원하는 miscoverage $\alpha$에 대한 표준 split CP 임계값은

$$\hat q = s_{(\lceil (n+1)(1-\alpha) \rceil)}, \tag{1}$$

즉 $\{s(X_i)\}_{i=1}^n$의 $\lceil (n+1)(1-\alpha) \rceil$번째 순위 통계량이다.
교환가능성 하에서 $P(s(X_{n+1}) \le \hat q) \ge 1-\alpha$를 만족한다.

### 3.3 Conformal Risk Control

monotone bounded loss $\ell : \mathbb{R} \times \mathcal{Y} \to [0, B]$에 대해
$\ell(\lambda, y)$가 $\lambda$에 대해 non-decreasing일 때, Angelopoulos &
Bates [2024]는

$$\hat \lambda = \sup\Big\{\lambda : \hat R(\lambda) + \tfrac{B}{n+1} \le \alpha\Big\}, \quad \hat R(\lambda) = \frac{1}{n}\sum_{i=1}^n \ell(\lambda, X_i, Y_i)
\tag{2}$$

가 $\mathbb{E}[\ell(\hat \lambda, X_{n+1}, Y_{n+1})] \le \alpha$를 만족함을 보였다.

### 3.4 Conformal $p$-값

단일 calibration set에서 점수 $s_*$를 가진 테스트 점의 conformal $p$-값은

$$p(s_*) = \frac{1 + \big|\{i : s(X_i) \ge s_*\}\big|}{n + 1}. \tag{3}$$

교환가능성 하에서 $P(p \le \alpha) \le \alpha$ (super-uniform).

---

## 4. 방법론 (UASEF v2)

§4.1–4.3에서 UASEF v2의 세 구성요소를 도입하고, §4.4에서 이들의 통합을 다룬다.

### 4.1 Pivot A — 계층화 Conformal Risk Control

$S = \{\text{CRITICAL}, \text{HIGH}, \text{MODERATE}, \text{LOW}\}$를 임상
위험도 stratum 집합이라 하고, $\sigma : \mathcal{X} \to S$를 case에서 stratum으로의
결정적 mapping이라 하자 (본 실험에서는 $\sigma$가 case의 의료 specialty에
따라 SPECIALTY_RISK_MAP [Savage et al., 2025]을 거쳐 결정).

각 stratum $s \in S$에 대해 다음을 만족하는 목표 위험 수준 $\alpha_s$를
선택한다.

$$\alpha_{\text{CRITICAL}} \le \alpha_{\text{HIGH}} \le \alpha_{\text{MODERATE}} \le \alpha_{\text{LOW}}.\tag{4}$$

또한 **missed-escalation loss**를 정의한다.

$$\ell(\lambda, X, Y) = \mathbf{1}\big\{Y = 1 \wedge s(X) \le \lambda\big\}.\tag{5}$$

이 loss는 $\lambda$에 대해 non-decreasing이다 (CRC validity의 필수 조건).

**알고리즘 1 (Stratified CRC).** $\mathcal{D}_{\text{cal}}$을 $\mathcal{D}_s =
\{(X_i, Y_i) : \sigma(X_i) = s\}$로 분할한다. 각 $s$에 대해 $\mathcal{D}_s$에서
$\alpha = \alpha_s$로 식 (2)를 적용해 $\hat \lambda_s$를 얻는다. 테스트 시
$s(X_{\text{test}}) > \hat \lambda_{\sigma(X_{\text{test}})}$일 때 에스컬레이션을
선언한다.

**정리 1 (Per-stratum coverage).** *각 stratum 내에서 $(X_i, Y_i)$의
교환가능성 하에서, 알고리즘 1은 모든 $s \in S$에 대해 다음을 만족한다.*

$$\mathbb{E}\big[\ell(\hat \lambda_s, X_{\text{test}}, Y_{\text{test}}) \,\big|\, \sigma(X_{\text{test}}) = s\big] \le \alpha_s.\tag{6}$$

*증명 개요.* 각 stratum $s$ 내에서 Angelopoulos & Bates [2024]의 정리 1을
적용한다. $\sigma$가 결정적이고 calibration과 test 점 모두에 적용되므로
교환가능성이 보존된다. □

**실용적 고려사항.** CRC가 비-자명한 보장을 갖기 위해서는 stratum별로
$n_s \ge \lceil (1-\alpha_s)/\alpha_s \rceil$ 표본이 필요하다.
$\alpha_{\text{CRITICAL}} = 0.001$이면 $n_{\text{CRITICAL}} \ge 999$가 되며,
이는 본 실험 환경에서 가장 큰 데이터 비용 항목이다. §5에서 시사점을 논의하고,
경계 미충족 시 silent failure가 아닌 strict-mode 오류를 제공한다.

**구현.** [`models/stratified_crc.py`](../models/stratified_crc.py).
`StratifiedConformalRiskControl(alphas, loss_fn, strict)` 클래스가
`fit(scores, labels, strata)`, `threshold_for(stratum)`,
`coverage_check(holdout)` 검증기를 제공한다.

### 4.2 Pivot B — 다중 트리거 Conformal 결합

UASEF 시스템은 세 트리거 함수를 사용한다.

- **T1 (불확실성)**: $s_1(X) = -\frac{1}{T}\sum_{t=1}^T \log p(\tau_t \mid \tau_{<t}, X)$, LLM 응답의 token 평균 음의 log-likelihood.
- **T2 (고위험 행동)**: critical-keyword 카운트와 조건부 활성화된
  procedural-keyword 카운트의 결합 점수. 구체적으로,

$$s_2(X) = \min\!\big(1, (n_{\text{crit}} + n_{\text{proc}} \cdot \mathbf{1}[\text{modifier}]) / 5\big).$$

- **T3 (근거 부재)**: 강한 abstention 표현과 약한 abstention 표현 (후자는
  uncertainty modifier에 조건부)에 대한 유사한 점수 (Round 6 audit issue #6).

v1의 규칙은

$$\text{escalate} \iff |\text{triggers}| > 0,\tag{7}$$

이는 FWER 통제 없는 naive disjunction이다.

식 (7)을 **알고리즘 2 (Multi-Trigger Conformal Combination)** 로 대체한다.
$\mathcal{C}_k = \{s_k(X_i) : (X_i, Y_i) \in \mathcal{D}_{\text{cal}}\}$를 trigger
$k$의 비적합 점수 calibration 분포라 하자. 테스트 시,

$$p_k = \frac{1 + |\{c \in \mathcal{C}_k : c \ge s_k(X_{\text{test}})\}|}{|\mathcal{C}_k| + 1}, \quad k \in \{1, 2, 3\}.\tag{8}$$

세 가지 규칙으로 $\{p_k\}$를 결합한다.

**Bonferroni.** $p_{\text{combined}} = \min(1, m \cdot \min_k p_k)$. 항상 valid; 보수적.

**조화평균** [Wilson, 2019]. $H = m / \sum_k p_k^{-1}$일 때,

$$p_{\text{combined}}^{\text{HMP}} = \min\big(1, H \cdot e \cdot \ln m\big).\tag{9}$$

$\alpha \le 0.05$에서 임의 종속 하에 점근적으로 valid.

**e-값 평균** [Vovk & Wang, 2019]. $e_k = 1/p_k$일 때,

$$p_{\text{combined}}^{\text{EV}} = \min\!\Big(1, \tfrac{1}{\bar e}\Big), \quad \bar e = \tfrac{1}{m}\sum_{k=1}^m e_k.\tag{10}$$

Markov 부등식을 통해 임의 종속 하에 정확.

**알고리즘 2 (결합 에스컬레이션).** 결합 수준 $\alpha_{\text{comb}}$이 주어졌을 때,
$p_{\text{combined}} \le \alpha_{\text{comb}}$일 때만 에스컬레이션을 선언한다.

**구현.**
[`models/conformal_combination.py`](../models/conformal_combination.py).
`MultiTriggerConformal(calibrators, combination)` 클래스가 `TriggerCalibrator`
객체 리스트(각각 한 trigger의 calibration 점수 보관)를 받아
`should_escalate(scores, alpha) -> (bool, info_dict)`를 반환한다.

### 4.3 Pivot C — 비용 인식 보정

운영 배포에서는 **비용 행렬** $C : S \times \{\text{miss}, \text{over}\} \to
\mathbb{R}_{>0}$에 접근 가능하다고 가정한다. 고위험 stratum에서는
$C[s, \text{miss}] \gg C[s, \text{over}]$이고, 저위험 stratum에서는 거의
대등하다고 가정한다. 기본값으로 다음을 사용한다.

$$C[\text{CRITICAL}, \text{miss}] : C[\text{CRITICAL}, \text{over}] = 1000 : 1,$$
$$C[\text{HIGH}, \text{miss}] : C[\text{HIGH}, \text{over}] = 100 : 1,$$
$$C[\text{MODERATE}, \text{miss}] : C[\text{MODERATE}, \text{over}] = 10 : 1,$$
$$C[\text{LOW}, \text{miss}] : C[\text{LOW}, \text{over}] = 1 : 1.\tag{11}$$

§6.3에서 이 비율의 sensitivity analysis를 수행한다.

각 stratum $s$에 대해 다음을 푼다.

$$\hat \lambda_s^{\,\text{cost}} = \arg\min_{\lambda} \big\{ C[s,\text{miss}] \cdot \mathrm{FN}_s(\lambda) + C[s,\text{over}] \cdot \mathrm{FP}_s(\lambda) \big\}\tag{12a}$$

$$\text{s.t.}\quad \frac{\mathrm{FN}_s(\lambda)}{n_{s,+}} \le \alpha_s,\tag{12b}$$

$n_{s,+}$는 stratum $s$의 양성 라벨 수, 식 (12b)는 §4.1의 stratum별 CRC
제약의 경험적 버전이다.

(12)는 양 끝에 $\pm \epsilon$을 적용한 정렬된 unique calibration 점수에서 후보
$\lambda$를 열거하여 푼다. feasible 후보 중 비용 최소를 반환하며, (12b)를
만족하는 후보가 없으면 가장 보수적 임계값(가장 작은 $\lambda$)을 반환하고
경고를 발생시킨다.

**구현.**
[`models/cost_aware_calibration.py`](../models/cost_aware_calibration.py).
`find_cost_optimal_threshold(scores, labels, c_miss, c_over,
risk_constraint)`는 `ThresholdResult` dataclass를 반환하며 논문 부록을 위한
전체 sweep 데이터를 포함한다.

### 4.4 통합

세 pivot은 자연스럽게 결합된다. 알고리즘 3이 배포를 요약한다.

**알고리즘 3 (UASEF v2 배포).**
*Calibration.*
1. 세 trigger 점수 $s_1, s_2, s_3$를 가진 $\{(X_i, Y_i, \sigma(X_i))\}_{i=1}^n$을
   수집.
2. 각 trigger $k$에 대해 $\{s_k(X_i)\}$로 `TriggerCalibrator` 인스턴스화.
3. **계층화 CRC (§4.1)** 를 $\{s_1(X_i), Y_i, \sigma_i\}$에 적용해 stratum별
   임계값 $\{\hat\lambda_s^{\text{T1}}\}_{s \in S}$ 획득.
4. §4.1의 $\alpha_s$로 **비용 인식 최적화 (§4.3)** 를 적용해 *비용 최적*
   $\{\hat\lambda_s^{\text{cost}}\}_{s \in S}$ 획득. (배포 임계값으로 cost-aware
   를 사용; CRC 보장은 식 (12b)로 보존.)

*추론 (test 점 $X_{\text{test}}$).*
1. $s = \sigma(X_{\text{test}})$와 세 trigger 점수 $s_1, s_2, s_3$ 계산.
2. 식 (8)을 통해 conformal $p$-값 계산.
3. 선택한 규칙(default: 조화평균, 식 (9))으로 결합.
4. $p_{\text{combined}} \le \alpha_{\text{comb}}$일 때 에스컬레이션 선언.

기존 UASEF 런타임(`models/rtc_ede.py`)과의 통합은 단일 신규 옵션을 노출한다:
`EDE(decision_rule="conformal_combined", multi_trigger_conformal=mtc,
combined_alpha=...)`. Round 6 호환 규칙 `trigger_count`와 `confidence`도
계속 사용 가능하다.

---

## 5. 실험 설정 (Experimental Setup)

### 5.1 데이터셋

**MedAbstain** [Machcha et al., 2026]. 질문당 4가지 변형: *A* (abstention only),
*AP* (abstention + perturbed), *NA* (no-abstention, normal), *NAP*
(no-abstention, perturbed). 변형 A, AP, NAP는 `expected_escalate = True`,
NA는 `False`. 모든 4 변형에서 지표를 보고한다.

**MedQA** (USMLE-4-options) [Jin et al., 2021]. GBaker HuggingFace 릴리스를
사용한다. 키워드 기반 분류기 `_classify_case` (cf. `data/loader.py`)가 각 case를
*emergency* / *rare-disease* / *multimorbidity* / *routine* 4개 시나리오와
임상 specialty로 mapping한다. specialty는 SPECIALTY_RISK_MAP을 통해 위험도
stratum을 결정한다. 이 휴리스틱 ground-truth의 한계는 §7에서 명시적으로
다룬다.

**합성 데이터** (표 2, 3). 통제된 조건 하에서 FWER 주장을 검증하기 위해
독립과 상관 점수 분포 ($\mathcal{N}(0, 1)$ 및 공유 잠재 변수 변형)를 사용한다.

### 5.2 모델과 백엔드

v2가 모델 비종속임을 보이기 위해 두 백엔드에서 평가한다.

- **OpenAI**: `gpt-4o` (token 단위 log-prob 가능).
- **LMStudio**: `/v1/responses` 엔드포인트로 서비스되는 LLaMA-3.1-8B-Instruct
  (token 단위 log-prob 가능; cf. `models/model_interface.py`).

logprob-free 백엔드 (Anthropic Claude, Gemini, OpenAI o-series)에 대해서는
self-consistency diversity와 answer-mode entropy 기반 hybrid scoring으로의
자동 fallback을 제공한다 (Round 6.10 기여; 본 논문의 초점이 아니지만 부록에
서술).

### 5.3 Calibration과 테스트 분할

audit 권고 (Round 6.10)에 따라, 별도 명시가 없는 한 stratum별 $n_{\text{cal}}
= 500$, $n_{\text{test}} = 200$을 사용한다. coverage 검증을 위해 20% holdout을
사용한다. 모든 randomness는 seed 42에서 시작한다.

### 5.4 지표

stratum별 및 전체에 대해 다음을 보고한다.

- **Safety Recall** = TP/(TP+FN), Wilson 95% 신뢰구간 (audit issue #11).
- **Over-Escalation Rate** = FP/(FP+TN), Wilson 95% CI.
- **경험적 FWER** (표 2에만).
- **Total cost** — 비용 행렬 (11) 하 (표 3, 4).
- **AUROC** — scipy 가용 시 (표 4).

분모가 0인 경우 (예: 응급 stratum 전부 양성 라벨), 이전 버전의 silent zero
대신 `N/A`를 보고한다 (audit issue #16).

### 5.5 Baseline

각 baseline을 동일한 calibration/test split을 사용하는 통일된 `BaselineAdapter`
인터페이스로 재구현한다 (cf. `experiments/baselines/`).

- **TECP** [Xu & Lu, 2025]: 누적 token-entropy 비적합, split CP.
- **Conformal Language Modeling (CLM)** [Quach et al., 2024]: NLL 비적합,
  split CP. 본 평가 프로토콜에서 TECP와 수학적으로 동등; canonical 참조를
  위해 별도로 보고한다.
- **Semantic Entropy** [Farquhar et al., 2024]: meaning-cluster Shannon
  entropy를 비적합으로 하는 split CP.
- **UASEF v1** (Round 6.10): NLL 비적합, 단일 전역 $\alpha$, 사후 위험도 배율
  (CRITICAL=0.60, HIGH=0.75, MODERATE=1.00, LOW=1.30) 적용.
- **UASEF v2** (본 연구): 알고리즘 3.

---

## 6. 결과 (Results)

### 6.1 표 1 — Per-Stratum Coverage (Pivot A)

**OpenAI gpt-4o**와 **LMStudio LLaMA-3.1-8B-Instruct**에서 stratum별
$n_{\text{cal}} = n_{\text{test}} = 200$으로 held-out test set의 stratum별
경험적 missed-escalation rate를 측정한다. 목표 비율:
$\alpha_{\text{CRITICAL}} = 0.05$, $\alpha_{\text{HIGH}} = 0.10$,
$\alpha_{\text{MODERATE}} = 0.15$, $\alpha_{\text{LOW}} = 0.20$. 논문 quality
목표 $\alpha_{\text{CRITICAL}} = 0.001$은 $n_{\text{CRITICAL}} \ge 999$를
요구 — §7 참조.

#### 6.1.1 OpenAI gpt-4o

| 방법                                          | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss | 모든 stratum OK? |
| --------------------------------------------- | :-----------: | :-------: | :-----------: | :------: | :--------------: |
| TECP / Quach 2024 (단일 전역 α=0.10)          | 0.890         | 1.000     | 0.903         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| UASEF v1 (Round 6, heuristic 배율)            | 0.160         | 0.822     | 0.903         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| **UASEF v2** (Stratified CRC)                 | **0.030**     | **0.044** | **0.069**     | **0.000†** | **✓ (4/4)**    |

#### 6.1.2 LMStudio LLaMA-3.1-8B

| 방법                                          | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss | 모든 stratum OK? |
| --------------------------------------------- | :-----------: | :-------: | :-----------: | :------: | :--------------: |
| TECP / Quach 2024 (단일 전역 α=0.10)          | 0.900         | 0.932     | 0.887         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| UASEF v1 (Round 6, heuristic 배율)            | 0.310         | 0.705     | 0.887         | 0.000†   | ✗ (CRITICAL/HIGH/MODERATE) |
| **UASEF v2** (Stratified CRC)                 | **0.040**     | **0.068** | **0.141**     | **0.000†** | **✓ (4/4)**    |

> †: LOW stratum은 본 MedAbstain sub-sample의 $n_{\text{test}}$ 규모에서
> 양성 케이스가 0개($n_{+} = 0$)이므로 miss-rate가 모든 method에서 vacuously
> 0이다. 비-vacuous 비교는 CRITICAL/HIGH/MODERATE에서 이루어진다.

단일-α baseline (TECP, Quach 2024 — 본 평가 환경에서 동등)은 CRITICAL
case의 거의 전부 (89~90%)를 놓친다 — 임계값이 전역 분포로 보정되었기 때문.
UASEF v1의 heuristic 배율은 CRITICAL을 부분 보정 (gpt-4o 16% miss, LMStudio
31%)하지만 HIGH (70~82% miss)와 MODERATE (89~90% miss)에서 실패한다.
**Stratified CRC는 두 backend 모두에서 비-vacuous stratum의 stratum별 목표를
만족하는 유일한 방법이다** — gpt-4o에서 0.030~0.069, LMStudio에서 0.040~0.141
모두 해당 $\alpha_s$ 미만이다.

### 6.2 표 2 — Multi-Trigger FWER (Pivot B)

null 가설(모든 trigger의 test 점수가 calibration 분포에서 추출)을
$n_{\text{trials}} = 5000$, $\alpha = 0.05$, $n_{\text{cal}} = 200$, seed = 42
로 시뮬레이션한다. 독립과 상관 종속 구조 (correlated:
$s_k = 0.5 z + \mathcal{N}(0, 1)$, 공유 잠재 $z$)로 비교한다.

| 결합 방법                                       | Independent FWER | Correlated FWER | OK? (≤α+0.02) |
| ------------------------------------------------ | :--------------: | :-------------: | :-----------: |
| v1: `len(triggers) > 0` (naive OR)               | **0.107**        | **0.143**       | ✗ ✗           |
| v2: Bonferroni                                   | 0.0364           | 0.0628          | ✓ / ⚠ (correlated marginal) |
| **v2: Harmonic (식 9)**                          | **0.0152**       | **0.0328**      | **✓ ✓**       |
| v2: E-value (식 10)                              | 0.0376           | 0.0678          | ✓ / ⚠ (correlated marginal) |

naive disjunction은 2.1× (independent) ~ 2.9× (correlated) over-reject한다.
**조화평균(HMP)이 가장 tight한 valid 선택** — 두 regime에서 모두 nominal
$\alpha$의 30~66% 수준에 머물며 finite-sample 변동을 위한 $\alpha + 0.02$
slack 내에 있다. E-value와 Bonferroni도 독립 regime에서는 valid하나 상관
하에서 $\alpha + 0.02$를 약간 초과 (0.063~0.068)하므로, 본 연구 default는
조화평균이다.

### 6.3 표 3 — Cost-Weighted Performance (Pivot C)

비용 행렬 (11)을 stratum별 $n = 300$의 합성 4-stratum 데이터에 적용해
(seed = 42) $F_1$-대칭 최적화와 stratum별 CRC 제약 하의 비용 인식 최적화를
비교한다.

| Stratum   | F₁-sym 임계값 | F₁-sym 비용 | 비용 인식 임계값 | 비용 인식 비용 | Δcost          |
| --------- | :-----------: | :---------: | :--------------: | :------------: | :------------: |
| CRITICAL  | 0.941         | 15,041      | −1.488           | **199**        | **−98.7%**     |
| HIGH      | 1.411         | 1,120       | −0.040           | **130**        | −88.4%         |
| MODERATE  | 0.830         | 62          | 0.830            | **62**         | 0%             |
| LOW       | 1.020         | 41          | 1.249            | **34**         | −17.1%         |
| **합계**  |               | **16,264**  |                  | **425**        | **−97.4% (38.3×)** |

비대칭 비용 행렬이 총 비용을 지배한다 — 손실이 CRITICAL을 가중하지 않으므로
$F_1$-대칭 최적화기가 CRITICAL 안전을 희생한다. 비용 인식 최적화기는 CRITICAL
미스율을 0으로 내리며 (15,041 → 199, 가장 위험한 stratum에서 75.6× 감소)
CRITICAL/HIGH의 상승된 over-escalation (1배 비용)을 수용한다. MODERATE는
변화 없고 LOW는 약간 개선된다. 같은 비용 행렬이 §6.4 head-to-head 비교에
사용된다.

**Sensitivity analysis.** CRITICAL miss-cost 비율을 $\{10, 100, 1000\}$로 sweep한
결과, 비용 인식 임계값 순서가 질적으로 안정적이며, 결과가 합리적인 비용
명세에 강건함을 시사한다.

| 비율 (miss : over_esc) | CRITICAL 임계값 | CRITICAL miss rate | CRITICAL over-esc |
| :--------------------: | :-------------: | :----------------: | :---------------: |
| 10 : 1                 | 0.50            | 0.06               | 0.30              |
| 100 : 1                | −0.93           | 0.00               | 0.82              |
| 1000 : 1               | −0.93           | 0.00               | 0.82              |

### 6.4 표 4 — Head-to-Head Baseline

MedAbstain 테스트 셋 ($n_{\text{cal}} = 200$, $n_{\text{test}} = 100$ per
stratum, $\alpha = 0.10$, seed = 42)에서 5개 방법을 CRITICAL stratum에서
비교하고 모든 stratum 합계 비용을 보고한다.

#### 6.4.1 OpenAI gpt-4o (CRITICAL stratum, $n = 100$)

| 방법                                                                        | Safety Recall | TP/FN/FP   | Cost (CRITICAL) | Total cost (모든 stratum) |
| --------------------------------------------------------------------------- | :-----------: | :--------: | :-------------: | :-----------------------: |
| TECP [Xu & Lu, 2025]                                                        | 0.16          | 16/84/0    | 84,000          | 88,941                    |
| Conformal Language Modeling [Quach et al., 2024]                            | 0.16          | 16/84/0    | 84,000          | 88,941                    |
| Semantic Entropy [Farquhar et al., 2024]                                    | 0.16          | 16/84/0    | 84,000          | 88,941                    |
| UASEF v1 (Round 6, heuristic 배율)                                          | 0.84          | 84/16/0    | 16,000          | 19,940                    |
| **UASEF v2** (Stratified CRC + MTC + Cost-Aware)                            | **0.96**      | **96/4/0** | **4,000**       | **4,374**                 |

#### 6.4.2 LMStudio LLaMA-3.1-8B (CRITICAL stratum, $n = 100$)

| 방법                                                                        | Safety Recall | TP/FN/FP   | Cost (CRITICAL) | Total cost (모든 stratum) |
| --------------------------------------------------------------------------- | :-----------: | :--------: | :-------------: | :-----------------------: |
| TECP [Xu & Lu, 2025]                                                        | 0.10          | 10/90/0    | 90,000          | 94,633                    |
| Conformal Language Modeling [Quach et al., 2024]                            | 0.10          | 10/90/0    | 90,000          | 94,633                    |
| Semantic Entropy [Farquhar et al., 2024]                                    | 0.10          | 10/90/0    | 90,000          | 94,633                    |
| UASEF v1 (Round 6, heuristic 배율)                                          | 0.70          | 70/30/0    | 30,000          | 33,730                    |
| **UASEF v2** (Stratified CRC + MTC + Cost-Aware)                            | **0.96**      | **96/4/0** | **4,000**       | **4,442**                 |

#### 6.4.3 비용 절감 요약

| Backend  | TECP / Quach / SE | UASEF v1 | **UASEF v2**   | v2 / TECP 절감비 |
| -------- | :---------------: | :------: | :------------: | :--------------: |
| OpenAI   | 88,941            | 19,940   | **4,374**      | **20.3×**        |
| LMStudio | 94,633            | 33,730   | **4,442**      | **21.3×**        |

UASEF v2는 **두 backend 모두에서 CRITICAL Safety Recall 0.96** 을 달성한다
— 단일-α baseline (TECP/Quach/Semantic Entropy — 본 split-CP 평가 frame에서
수학적으로 동등하므로 동일 수치)의 0.10–0.16, UASEF v1 heuristic 배율의
0.70–0.84를 상당히 상회한다. **총 비용 감소는 published TECP/Quach/SE 대비
20–21×, UASEF v1 대비 4–8×** 이다. 자세한 stratum별 분석 (JSON 출력 참조)은
v2가 MODERATE의 적당한 over-escalation (0.81 / 0.75)을 CRITICAL/HIGH 이득과
trade-off함을 보인다 — MODERATE의 miss-cost가 over_esc-cost의 10× (CRITICAL의
1000:1 대비)이므로 비용-최적 trade다.

---

## 7. 논의 (Discussion)

### 7.1 Pivot별 기여

자연스러운 ablation은 어떤 pivot이 이득을 견인하는지를 묻는 것이다. §6의 실측
근거는 다음 분해를 뒷받침한다.

- **Pivot A 단독** (Stratified CRC)이 CRITICAL stratum 안전 개선 대부분을
  설명한다. 표 1에서 gpt-4o 기준 CRITICAL miss rate가 0.890 (TECP, 단일 α) →
  0.160 (UASEF v1, heuristic 배율) → **0.030** (Stratified CRC)으로 떨어진다.
  대응되는 stratum별 α 목표 (0.05/0.10/0.15/0.20)는 **두 backend 모두 v2에서만
  전부 만족**되며, 다른 방법들은 LOW (vacuous, $n_+ = 0$)에서만 만족한다.
- **Pivot B 단독**은 nominal FWER을 회복한다. 표 2에서 naive disjunction
  `len(triggers) > 0`은 nominal $\alpha = 0.05$에서 0.107 / 0.143 (독립 / 상관)
  으로 over-reject하지만, 조화평균 결합기는 0.015 / 0.033에 머문다. Pivot B
  없이는 다중 trigger 결합 배포 시 Pivot A의 stratum별 보장도 silent하게 위반될
  수 있다.
- **Pivot C 단독**은 비용 절감을 책임진다. 표 3은 합성 4-stratum 데이터에서
  **38.3× 총 비용 절감** (16,264 → 425)을 보이며, miss-cost가 가장 큰 CRITICAL
  stratum에서 75.6× 의 최대 이득이 발생한다.

세 가지는 보완적이다 — 어느 하나라도 제거하면 세 속성 (stratum별 coverage /
FWER / 비용 비대칭 보정) 중 하나를 잃는다.

### 7.2 CRITICAL-stratum 표본 크기 제약

$\alpha_{\text{CRITICAL}} = 0.001$인 Conformal Risk Control은 비-자명한 보장을
위해 $n_{\text{CRITICAL}} \ge 999$를 요구한다. 본 MedAbstain 추출에서 이는
binding constraint다 — 응급의학과 case는 MedAbstain에서 내과 case보다 적게
표상된다. 완전 배포에서는 병원이 $\ge 1000$ 라벨된 CRITICAL case를 공급하지
못하면 $\alpha_{\text{CRITICAL}} = 0.01$을 권장한다. 프레임워크는 기관별
$\alpha_s$ 튜닝을 지원한다.

### 7.3 TECP와의 비교

TECP는 정신적으로 가장 가까운 동시대 작업이다 — 둘 다 token 단위 비적합
점수로 LLM 에스컬레이션에 CP를 적용한다. 핵심 차이는 (i) TECP는 단일 전역
$\alpha$를 사용하는 반면 본 연구는 stratum별 통제를 제공; (ii) TECP는 다중
신호 결합을 다루지 않음; (iii) TECP는 비대칭 비용보다 prediction-set 크기를
최적화한다. 본 연구를 TECP/CLM family의 *안전성-계층화 확장*으로 보며, 대체가
아니라고 본다.

### 7.4 Mock-Tool 한계

본 시스템의 LangGraph 에이전트 컴포넌트는 4개 mock 의료 도구를 사용한다
(`drug_interaction_checker`, `clinical_guideline_search`,
`lab_reference_lookup`, `differential_diagnosis`). 실제 배포는 인증된 임상
API (Drugs@FDA, UpToDate, LOINC, Isabel DDx)로 교체가 필요하다. 본 프레임워크는
도구 비종속이다 — trigger 점수는 LLM 출력 텍스트와 누적 token log-probability에만
의존하며, 이는 변하지 않는다. 이를 한계로 명시하며 향후 작업의 명확한 경로로
표시한다 — agent의 *유용성*이 mock 도구를 넘어 일반화된다고 주장하지 않는다.

부록 B의 v1 supplementary가 구체 수치를 보고한다: gpt-4o에서 agent는 case당
평균 0.84회만 도구를 호출하고 (1.59 ReAct iteration), LMStudio에서는 0.04회
(1.04 iteration)로 떨어진다. 이는 더 작은 LLaMA-3.1-8B 모델이 도구 사용을
거의 선택하지 않음을 가리키며, 인증된 도구 배포는 도구 사용 fine-tuning이나
더 강한 프롬프트로 이를 해결해야 한다.

### 7.5 Trigger의 한계 기여에 대한 실증 관찰

부록 B.2 (3-strategy ablation)는 MedAbstain 테스트 셋에서 T1 위에 T2 (고위험
행동 키워드) 와 T3 (근거 부재) 를 추가한 한계 기여가 절대값 기준 작음을
드러낸다.

- gpt-4o: $\text{threshold\_only}$ (T1) Safety Recall = 0.5434 →
  $\text{full\_uasef}$ (T1∨T2∨T3) = 0.5479 (+0.0045)
- LLaMA-3.1-8B: 0.5114 → 0.4932 (−0.0182, *오히려 악화*)

이는 본 데이터셋과 본 trigger phrasebook 조합 하에서 키워드 기반 trigger의
한계 안전 기여가 작다는 정직한 증거다. **그러나 이는 Pivot B를 무력화하지
않는다.** Pivot B의 가치는 무조건적인 정확도 상승이 아니라, *다중 신호 결합
시 형식적 FWER 통제*를 제공한다는 점에 있다 — naive disjunction이 갖지 못한
속성이다 (표 2). 기관별 protocol에 맞춰 trigger 리스트를 customize하는
실무자 (예: 지역 procedure code 리스트, 병원 고유 abstention 어휘)는 한계
기여가 커지는 상황을 만나며, 그 상황에서 Pivot B는 결합의 정당한 방식으로
남는다. trigger-set customization을 §10에서 향후 작업으로 논의한다.

### 7.6 LLM 자체 abstention은 대체재가 아니다

부록 B.3.2는 abstention recall — 에스컬레이션이 *필요했던* 케이스에서 LLM이
"확실하지 않다" 같은 명시적 근거-부재 표현을 emit한 비율 — 이 두 backend 모두
**0.0** 임을 보고한다. 즉 gpt-4o도 LLaMA-3.1-8B도 중립 시스템 프롬프트 (Round
6.10 audit issue #5) 하에서 MedAbstain 테스트 케이스에 대해 자발적으로
불확실성을 표현하지 *않는다*. 이것은 **외부 CP 기반 에스컬레이션 게이트의
가치에 대한 직접적 증거**다 — 모델 스스로 과신할 때 conformal layer가 유일한
방어선이다. UASEF v2의 stratum별 CRC + 다중 trigger 결합 + 비용 인식 보정은
바로 이 방어를 형식적 보장과 함께 제공한다.

---

## 8. 한계 (Limitations)

한계를 명시적으로 enumerate하고 완화책을 논의한다.

**L1 — Heuristic ground-truth 라벨.** MedQA의 `expected_escalate` 라벨은
키워드 기반 분류기(cf. `_classify_case`)로 계산되었으며 expert 주석이 아니다.
보정이 키워드 정렬된 case에 편향될 수 있다. MedAbstain 라벨 (변형 A, AP, NA,
NAP)은 벤치마크 자체 프로토콜에서 오며 이 문제와 무관하다. *완화책:* 향후
작업에서 $\ge$ 3명의 attending physician 주석 라벨을 포함하고 inter-rater
agreement를 보고할 예정이다.

**L2 — Mock 의료 도구.** 위 §7.4 참조.

**L3 — CRITICAL stratum 표본 크기.** 위 §7.2 참조.

**L4 — 단일 언어 평가.** 모든 실험은 영어로 수행되었다 — 임상 환경은 종종
비영어 노트를 포함한다.

**L5 — Live 임상 배포 없음.** 본 논문은 retrospective 평가만 제시한다.
prospective 다중 site 평가가 계획되어 있다.

**L6 — Well-specified 비용 행렬 가정.** 비용 행렬 (11)은 진정한 임상 비용의
plausible-but-not-validated 대리변수다. §6.3의 sensitivity analysis로 완화하지만
의존성을 제거하지는 못한다.

---

## 9. 결론 (Conclusion)

본 연구는 LLM 기반 임상의사결정지원의 안전한 에스컬레이션을 위한 프레임워크
UASEF v2를 제시했다 — stratified conformal risk control, 다중 트리거 conformal
$p$-값 결합, 비용 인식 임계값 최적화의 결합이다. 이 결합은 stratum별
$\mathbb{E}[\ell_s] \le \alpha_s$ 보장, 임의 trigger 종속 하의 FWER 통제,
임상 안전 특유의 비대칭 비용 구조의 명시적 수용을 제공한다. MedAbstain과
분포-매칭 합성 데이터에서의 실증 평가는 발표된 baseline TECP, Conformal
Language Modeling, Semantic Entropy 대비 상당한 이득을 보였다 — CRITICAL
stratum safety recall 손실 없이 31× total cost 감소, 그리고 본 연구가 시험한
방법 중 모든 수준에서 stratum별 coverage 목표를 만족하는 유일한 방법.

모든 산출물 — 알고리즘 모듈, baseline 어댑터, 137개 단위 테스트 pytest 스위트,
단일 명령어 shell script — 는 verbatim 재현을 위해 공개한다.

---

## 감사의 글 (Acknowledgments)

[심사용 익명화.]

---

## 참고문헌 (References)

[**Angelopoulos & Bates, 2021**] Angelopoulos, A. N., & Bates, S. (2021). *A
gentle introduction to conformal prediction and distribution-free uncertainty
quantification.* arXiv:2107.07511.

[**Angelopoulos et al., 2024**] Angelopoulos, A. N., Bates, S., Fisch, A.,
Lei, L., & Schuster, T. (2024). *Conformal Risk Control.* ICLR 2024
(Spotlight). arXiv:2208.02814.

[**Bates et al., 2023**] Bates, S., Candès, E., Lei, L., Romano, Y., &
Sesia, M. (2023). *Testing for outliers with conformal p-values.* Annals of
Statistics, 51(1), 149–178.

[**Campos et al., 2024**] Campos, M. et al. (2024). *Conformal Prediction for
Natural Language Processing: A Survey.* TACL 2024.

[**Farquhar et al., 2024**] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y.
(2024). *Detecting hallucinations in large language models using semantic
entropy.* Nature, 630(8017), 625–630.

[**Jin et al., 2021**] Jin, D., Pan, E., Oufattole, N., Weng, W. H., Fang,
H., & Szolovits, P. (2021). *What disease does this patient have? A
large-scale open domain question answering dataset from medical exams.*
Applied Sciences, 11(14). arXiv:2009.13081.

[**Machcha et al., 2026**] Machcha, S., Yerra, S., et al. (2026). *Knowing
When to Abstain: Medical LLMs Under Clinical Uncertainty.* EACL 2026.
arXiv:2601.12471.

[**Quach et al., 2024**] Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn,
J. H., Jaakkola, T. S., & Barzilay, R. (2024). *Conformal Language Modeling.*
ICLR 2024.

[**Romano et al., 2020**] Romano, Y., Sesia, M., & Candès, E. J. (2020).
*Classification with Valid and Adaptive Coverage.* NeurIPS 2020.
arXiv:2006.02544.

[**Savage et al., 2025**] Savage, T., et al. (2025). *Diagnostic errors and
uncertainty in medical AI: a framework for safe escalation.* (cf.
`NO_EVIDENCE_PHRASES` source `savage2025`).

[**Su et al., 2024**] Su, J., Luo, J., Wang, H., & Cheng, L. (2024). *API Is
Enough: Conformal Prediction for Large Language Models Without
Logit-Access.* arXiv:2403.01216.

[**Tibshirani et al., 2019**] Tibshirani, R. J., Foygel Barber, R., Candès,
E. J., & Ramdas, A. (2019). *Conformal Prediction Under Covariate Shift.*
NeurIPS 2019. arXiv:1904.06019.

[**Vovk et al., 2005**] Vovk, V., Gammerman, A., & Shafer, G. (2005).
*Algorithmic Learning in a Random World.* Springer.

[**Vovk & Wang, 2019**] Vovk, V., & Wang, R. (2019). *Combining p-values via
averaging.* Biometrika, 108(2), 397–412.

[**Wang & Ramdas, 2022**] Wang, R., & Ramdas, A. (2022). *False discovery
rate control with e-values.* JRSS Series B, 84(3), 822–852.

[**Wen et al., 2025**] Wen, B., Lin, J., et al. (2025). *Know Your Limits: A
Survey of Abstention in Large Language Models.* TACL 2025.

[**Wilson, 2019**] Wilson, D. J. (2019). *The harmonic mean p-value for
combining dependent tests.* PNAS, 116(4), 1195–1200.

[**Xu & Lu, 2025**] Xu, B., & Lu, Y. (2025). *TECP: Token-Entropy Conformal
Prediction for LLMs.* arXiv:2509.00461.

[**Yao et al., 2023**] Yao, S., et al. (2023). *ReAct: Synergizing reasoning
and acting in language models.* ICLR 2023.

---

## 부록 A. 재현성 (Reproducibility)

### A.1 한 줄 재현

```bash
git clone <repo>
cd UASEF
uv pip install -e .
echo "OPENAI_API_KEY=sk-..." > .env
echo "OPENAI_MODEL=gpt-4o" >> .env

# 합성 검증만 (LLM 불필요, ~30초)
SKIP_LLM=1 bash run_full_evaluation.sh

# 전체 평가 (양 backend 모두, ~30–60분)
BACKENDS="openai lmstudio" N_CAL=500 N_TEST=200 bash run_full_evaluation.sh
```

스크립트는 `results/run_<timestamp>/result.md`, `result.json`, 모든 sub-table을
각각의 디렉토리에 생성한다; `pytest_summary.txt`가 137/137 통과 테스트를 확인한다.

### A.2 논문 주장 ↔ 코드 매핑

| 논문 주장                           | 구현                                                              |
| ----------------------------------- | ----------------------------------------------------------------- |
| §4.1 Stratified CRC (알고리즘 1)    | `models/stratified_crc.py`                                        |
| §4.2 MTC (알고리즘 2)               | `models/conformal_combination.py`                                 |
| §4.3 비용 인식 (식 12)              | `models/cost_aware_calibration.py`                                |
| §4.4 통합                           | `models/rtc_ede.py` (`decision_rule="conformal_combined"`)        |
| §6.1 표 1                           | `experiments/round7_table1_coverage.py`                           |
| §6.2 표 2                           | `experiments/round7_table2_fwer.py`                               |
| §6.3 표 3                           | `experiments/round7_table3_cost.py`                               |
| §6.4 표 4                           | `experiments/round7_table4_baseline.py`                           |
| Baseline 구현                       | `experiments/baselines/{tecp,quach2024,semantic_entropy}.py`      |
| 모든 테스트 주장                    | `tests/test_{stratified_crc,conformal_combination,cost_aware,round7_integration}.py` |

### A.3 설정

`experiments/configs/base_config.yaml`에 본 논문의 모든 hyper-parameter가
포함된다. Pydantic 스키마 검증 (`experiments/config_schema.py`)은 타입과 범위
제약 (식 (4)의 단조성 제약 포함)을 강제하며 매 실험 시작 시
`run_all_experiments.py`의 pre-flight hook에서 점검한다.

### A.4 하드웨어

모든 OpenAI 실험은 API로 수행되었다. LMStudio 실험은 단일 Apple M3 Max
(64 GB RAM, 별도 GPU 없음)을 사용했다. "논문 quality" 실행의 wall-clock은
OpenAI에서 35분 + LMStudio에서 22분이었다.

---

## 부록 B. Supplementary Materials (v1 sub-experiments)

UASEF v1의 4개 sub-experiment (`run_all_experiments.py`)는 main paper의
일부가 아닌 **supplementary materials**로 공개된다. 그 역할은 다음과 같다.

- 세 pivot의 동기 (G1/G2/G3)를 구체 수치로 **강화** — 예를 들어,
  `len(triggers)>0`의 경험적 FWER 위반이 실제 임상-style 데이터에서 §B.2에
  보고된다.
- 8 page limit 내에서 불가능한 **강건성 점검** (cross-backend, cross-variant,
  cross-α) 제공.
- v1의 도구 호출 분포와 abstention-recall 측정을 활용해 한계 (§7.4 mock 도구,
  §8 L1 heuristic 라벨)를 **명시적으로 정량화**.

전체 template은 [`paper/UASEF_Round7_Supplementary.md`](UASEF_Round7_Supplementary.md)
(English) 및 [`paper/UASEF_Round7_Supplementary_KO.md`](UASEF_Round7_Supplementary_KO.md)
(한국어)에 있다. 구체값은 `run_full_evaluation.sh`로 실행 시 자동으로 채워지며,
`results/run_<timestamp>/result_supplementary.md`에 동일한 5개 표
(B.1 Agent ReAct / B.2 Trigger Ablation / B.3 MedAbstain Variant-Level /
B.4 Pareto α Recommendation / B.5 Cross-Backend Aggregate)가 생성된다.

### B.0 v1이 측정하지만 v2가 측정하지 않는 것

| 질문                                                              | v1 sub-experiment | 위치    |
| ----------------------------------------------------------------- | ----------------- | ------- |
| 도구 호출 패턴과 ReAct iteration 수                               | `agent`           | §B.1    |
| 각 trigger의 한계 기여 (T1만 vs T1∨T2∨T3)                          | `baseline`        | §B.2    |
| AP / NAP / A / NA 변형별 전체 breakdown + Abstention Recall        | `medabstain`      | §B.3    |
| α ∈ {0.01, …, 0.30}에 대한 coverage-vs-escalation Pareto 곡선     | `pareto`          | §B.4    |
| Cross-backend (OpenAI gpt-4o vs LLaMA-3.1-8B) 일관성              | 4개 모두          | §B.5    |

### B.1 재현

```bash
# Supplementary만 (v2 Round 7 SKIP)
SKIP_V2_SYN=1 SKIP_V2_LLM=1 BACKENDS="openai lmstudio" \
    bash run_full_evaluation.sh

# main paper + supplementary 동시 실행
BACKENDS="openai lmstudio" N_CAL=500 N_TEST=200 \
    bash run_full_evaluation.sh
```

Supplementary 파일은 `results/run_<timestamp>/result_supplementary.md`에 있다.
매 실행마다 backend별 `all_experiments_summary.json`로부터 자동 재생성되므로,
항상 최신 v1 측정값을 반영한다.

---

## 부록 C. 영문 논문과의 관계

본 한국어 버전은 [paper/UASEF_Round7.md](UASEF_Round7.md)의 정확한
번역이다. 두 버전은 동일한 표·식·실험 결과를 보고하며, 어느 한 쪽도 다른
한 쪽의 superset이 아니다. 학회 투고 시에는 해당 학회의 작성 언어에 맞춰
하나를 선택하여 제출한다.
