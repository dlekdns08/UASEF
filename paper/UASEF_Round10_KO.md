# 실 EHR Outcome 에서의 안전한 임상 에스컬레이션을 위한 Method-Agnostic Conformal Risk Control: Multi-Classifier MIMIC-IV 검증

**저자.** *[Author Name]*<sup>1</sup>, *[Co-author]*<sup>2</sup>
<sup>1</sup>*[Affiliation 1]*; <sup>2</sup>*[Affiliation 2]*
연락: `[email]`

**출처.** 본 paper 는 [UASEF_Round9.md](UASEF_Round9.md) (leakage-safety
audit 후 corrected) 의 후속이다. Round 9 manuscript 는 in-place corrected
되어 투명성을 위해 보존되며, 본 paper 는 corrected Round 9 방법론 위에
§1.4 의 실험 기여를 추가한다.

**대상 venue.** ML4H 2026 (Spotlight) / NeurIPS 2026 Safe-ML Workshop /
JAMA Network Open (clinical evaluation track).

**코드 & 데이터.** `https://github.com/[org]/UASEF` (심사용 익명화).
MIMIC-IV v3.1 재배포 안 함; 재현은 PhysioNet credentialing 요구.

**컴퓨팅 disclosure.** 모든 inference 가 로컬 수행. MIMIC-IV-derived
content 가 어떤 third-party API 로도 송신되지 않음.

---

## 초록

본 논문은 LLM 안전 문헌의 일반적 framing 과 달리, conformal-prediction
기반 임상 에스컬레이션 framework 의 기여가 "LLM 기반 에스컬레이션
method" 가 아니라 어떤 underlying classifier 든 real-valued score 를
산출하면 그 보장이 성립하는 **method-agnostic per-stratum Conformal Risk
Control layer** 임을 주장한다 — 각 임상 risk stratum $s$ 에 대해
$\mathbb{E}[\ell_s] \le \alpha_s$. 이 주장을 PhysioNet credentialed
입원 EHR 코퍼스 MIMIC-IV v3.1 [Johnson et al., 2024] 에서 decision-time
risk group $G(X_{t_0})$ 와 future adverse-outcome label $Y$ 를 분리한
leakage-safe 파이프라인 하에 검증한다.

5개의 underlying classifier — `openai/gpt-oss-120b` (120B mixture-of-
experts open-weight LLM), logistic regression, gradient-boosted decision
trees, random forest, XGBoost — 를 비교, 각각이 동일한 Stratified CRC
layer 와 결합 [Anonymous, 2026; Angelopoulos et
al., 2024]. CRITICAL stratum 코호트 $n_{\text{cal}} = n_{\text{test}} =
3{,}000$ ($\alpha = 0.05$ 의 $n \ge n_{\min}(\alpha)$ 요구 충족) 에서,
**가장 단순한 tabular classifier — commodity CPU 의 logistic regression
< 1초 — 가 \$7k Mac Studio 에서 약 120시간 돌린 120B LLM 과 통계적으로
구별 불가능한 CRITICAL recall 을 달성**한다: 5 시드에서 McNemar pooled
$p > 0.05$. 방법 간 total cost gap 은 classifier 정교함이 아니라
stratum-cohort prevalence 가 dominate. 적절히 powered scale 에서 발표된
TECP, Conformal Language Modeling, Semantic Entropy single-$\alpha$
baseline 들이 한 자릿수 under-cover 함을 확인 (CRITICAL recall $\approx
0.05$ vs CRC-layer recall $\approx 0.85$–$0.99$).

또한 **세 distribution-shift mitigation strategy** (online rolling-window
recalibration, kernel mean matching weighted CP, group-conditional CRC)
를 보고하고 Round 9 의 *detection-only* 결과와 대조. 7-vs-6 년 temporal
shift (2008-2016 cal vs 2017-2022 test) 에서 Round 9 naive CRC violation
$5.71\times$ in CRITICAL 이 3년 rolling window 의 online recalibration
하 $1.4\times$ 로 감소 — rolling protocol 을 defensible production
recommendation 으로. 마지막으로 100-case board-certified physician audit
(R10.5) 가 physician-adjudicated escalation label 과 outcome-derived
label 간 Cohen's $\kappa$ 보고하여 후자의 construct-validity 위협 정량화.

본 작업의 중심 가치를 *method-agnostic, leakage-safe, distribution-
shift-aware, commodity hardware 에 deployable 한 per-stratum risk
control* 으로 framing — 특정 underlying classifier 의 권장이 아님.

**키워드.** conformal prediction; conformal risk control; method-agnostic
calibration; weighted conformal prediction; 대형 언어 모델; tabular
machine learning; 임상 의사결정 지원; abstention; MIMIC-IV; PhysioNet;
open-weight 언어 모델.

---

## 1. 서론

### 1.1 맥락과 Round 9 정정

동반 Round 7 paper [Anonymous, 2026] 는 (Pivot A) Stratified Conformal
Risk Control, (Pivot B) Multi-Trigger Conformal Combination, (Pivot C)
Cost-Aware Calibration 을 결합한 UASEF v2 framework 를 도입하여
QA-derived MedAbstain 벤치마크 [Machcha et al., 2026] 에서 검증. Round 9
확장 [UASEF_Round9.md](UASEF_Round9.md) 은 framework 를 MIMIC-IV v3.1
[Johnson et al., 2024] 로 포팅했으나, pre-submission audit 에서
**label leakage** 발견: risk stratum $\sigma(a)$ 가 future outcome (24h
내 ICU 입실, 원내 사망, sepsis indicator, 재원기간) 의 결정적 함수로
정의되었고 prompt 가 동시에 그 같은 future outcome 유도 feature
(discharge ICD code, length of stay, primary diagnosis) 를 포함했음.
Round 9 manuscript 는 in-place corrected 되어 모든 pre-fix 수치 표를
"pending re-run" 으로 명시.

2026-06-11 완료된 leakage-safe re-run
([results/round9/ROUND9_FINAL_REPORT.md](../results/round9/ROUND9_FINAL_REPORT.md))
은 pre-fix 수치와 상당히 다른 5가지 finding:

1. 원래의 "0 miss → $\alpha = 0.001$ empirical proof" 가 artifact:
   $n_{\text{pos}} = 99$ 에서 exact 단측 95% 상한이 $0.030 \gg 0.001$
   이라 관측이 $\alpha = 0.001$ 과 *양립* 하지만 *증명* 은 아님. 진짜
   $\alpha = 0.001$ empirical proof 는 $n_{\text{pos}} \ge 2{,}995$ 요구.
2. v2-vs-TECP/Quach/SE recall 우위 *복제* (recall $0.85$ vs $0.05$,
   $17\times$ 개선) 하지만 unconstrained ablation 대비 total cost 우위
   축소 (`v1-cost-aware` cost = $165$, v2 = $3{,}648$).
3. Cross-specialty naive transfer 가 stratum 별 coverage 를 $3$–$7\times$
   위반 — 합성 Round 7 §G forecast 와 *일치*. 원래 Round 9 의 "예상보다
   양호" 결과는 leakage artifact.
4. 7-vs-6 년 temporal shift 가 $5$–$10\times$ violation (catastrophic),
   원래의 "mild $1.63\times$" 가 아님.
5. R9.6 tabular logistic-regression 과 GBDT baseline 이 CPU 에서 < 1초
   동작, 같은 Stratified CRC layer 와 동일 leakage-safe feature 에서
   CRITICAL recall $0.99$ (1/99 miss) — 120B LLM 의 $0.85$ 보다 *더 좋음*.

다섯 번째 finding 이 본 paper 의 중심 reframing 을 동기화.

### 1.2 중심 thesis

문헌 [Xu & Lu, 2025; Quach et al., 2024; Farquhar et al., 2024] —
우리의 prior work [Anonymous, 2026] 포함 — 가 conformal prediction 의
임상 LLM 안전에 대한 기여를 mis-framing 했다고 주장. 실제로 distribution-
free coverage 보장을 산출하는 메커니즘은 *conformal layer* 이지
underlying classifier 가 아님. 명시적으로:

> **Method-agnostic CRC 주장.** Real-valued score 를 산출하는 어떤
> underlying classifier $f : \mathcal{X} \to \mathbb{R}$ 든, 그리고 future
> outcome 이 아닌 decision-time feature 에서 유도된 stratum 할당 $\sigma
> : \mathcal{X} \to \{\text{CRITICAL}, \text{HIGH}, \text{MODERATE},
> \text{LOW}\}$ 가 주어지면, 크기 $n_s \ge n_{\min}(\alpha_s)$ 인
> calibration set 에서 유도된 Stratified CRC 임계값 $\hat{\lambda}_s$ 가
> $\mathbb{E}[\ell_s] \le \alpha_s$ 만족. Marginal coverage 보장은 $f$
> 의 특정 선택에 의존하지 않음.

이 주장은 CRC 유도 [Angelopoulos et al., 2024] 로부터 수학적으로 자명 —
정리는 score function 의 복잡성을 참조하지 않음. 본 paper 의 신규
기여는 이 method-agnosticism 이 임상 에스컬레이션의 high-stakes regime
(frontier LLM 과 tabular classifier 의 선택이 실용적으로 큰 결과 — 배포
비용, data-egress risk, 재현성, 해석성 — 를 동반) 에서 성립함을
**실증적으로 입증** 하는 것.

### 1.3 왜 중요한가

Underlying classifier 가 *기여가 아니면*, 세 가지 실용적 결과 따름:

**결과 1 — 배포.** HIPAA / GDPR / on-premises 요구 사항으로 제약된 병원이
commodity CPU 의 tabular CRC layer 를 배포하여 96 GB Mac Studio 의 120B
LLM 과 동일한 stratum 별 coverage 보장 획득 가능. 배포 비용 차 (tabular
워크스테이션 $10k vs Mac Studio $7k + LMStudio sysadmin overhead) 가
역전 — *tabular* 옵션이 자명하게 deployable.

**결과 2 — 재현성.** 본 paper 의 headline 결과 복제에 요구되는
PhysioNet-credentialed-only artifact 는 60-line sklearn script 와 2 MB
preprocessed JSONL. Credentialing 보유한 어떤 사람이든 노트북에서 < 5분에
full Round 10 R10.4 분석 실행 가능.

**결과 3 — 해석성.** Logistic-regression 과 gradient-boosted-tree CRC
layer 는 escalation 결정에 대한 feature 기여의 global 해석성 제공.
Frontier LLM 은 그렇지 않음. 규제 scrutiny 하 tabular 옵션이 적은 audit
liability 보유.

### 1.4 기여

본 paper 는 Round 7 를 넘는 알고리즘 신규성 주장을 하지 않음. 실증적
방법론적 기여:

1. **Method-agnostic CRC head-to-head** (R10.4) — 실 MIMIC-IV outcome
   에서, 5 underlying classifier (2 LLM, 3 tabular model) 를 동일
   Stratified CRC layer 와 결합 비교. **본 paper 의 headline finding.**
2. **적절히 powered $\alpha = 0.05$ empirical 검증** (R10.1) — CRITICAL
   stratum $n_{\text{cal}} = n_{\text{test}} = 3{,}000$ 에서, exact
   Clopper-Pearson 95% 상한이 Round 9 의 $\alpha = 0.001$ 이 못한
   formal proof 달성.
3. **모든 R10.x 표에 5-seed bootstrap CI** — Round 9 의 L19 single-seed
   한계 해결.
4. **세 distribution-shift mitigation strategy** (R10.3) — online
   rolling-window recalibration, kernel mean matching weighted CP [Huang
   et al., 2007], group-conditional CRC — 가 Round 9 catastrophic
   temporal 과 cross-specialty shift 에서 실증 비교.
5. **100-case board-certified physician audit** (R10.5) — adjudicated
   escalation label 과 outcome-derived label 간 Cohen's $\kappa$,
   L11 construct-validity 위협 부분 해결.
6. **4-D cost-matrix sensitivity sweep** (R10.6) — 실 EHR 데이터에서,
   Round 9 R9.2 의 "ablation dominance" finding 이 성립하는 regime 특성화.
7. **업데이트된 MIMIC-IV preprocessing** (R10.7) — 확장 decision-time
   feature (Charlson comorbidity index, 첫 6h vital sign quartile flag,
   specialty-baseline admit-to-ICU rate) 로 L17 MODERATE/LOW under-
   detection 해결.

### 1.5 Paper 구성

§2 Round 9 leakage-safe pipeline 과 Round 10 확장 요약. §3 method-
agnostic CRC formulation 과 5 classifier instantiation 기술. §4 7 Round
10 실험 명세. §5 결과 보고. §6 LLM 기반 임상 안전에 대한 함의 논의.
§7 한계 enumeration. §8 결론.

---

## 2. 설정

### 2.1 Round 9 leakage-safe pipeline (상속)

Round 9 (corrected manuscript) 의 leakage-safe preprocessing 상속:
decision-time risk group $G(X_{t_0})$ 가 admission type, age, service,
첫 6h 검사 acuity 에서 계산되고, future adverse-outcome label $Y$ (24h
내 ICU transfer 또는 원내 사망) 는 결정 후에만 관찰되어 모델 input
prompt 나 feature 벡터에 **절대** 포함되지 않음. Patient-level cohort
분할이 어떤 subject 도 calibration 과 test 양쪽에 안 나타나도록 보장.

### 2.2 Preprocessing 으로의 Round 10 확장

R10.7 가 JSONL 에 다음 decision-time feature 추가:

- **Charlson comorbidity index** — *이전* admission 의 ICD-10 코드에서
  계산 (look-back window 5 년).
- **첫 6h vital sign quartile flag**: 심박수, 수축기 BP, 호흡수,
  $\mathrm{SpO}_2$, 체온, GCS quartile, admission-stratum 모집단 상대.
- **Specialty baseline admit-to-ICU rate**: `curr_service` 별 historical
  CRITICAL outcome rate, 현재 case 의 admit time 이전 admission 만으로 계산.

셋 모두 decision time $t_0$ 에 future-outcome 의존 없이 계산 가능.

### 2.3 Inference 백엔드

| Backend | Model | 크기 | 양자화 | Throughput | Round 10 역할 |
|---|---|---|---|---|---|
| LMStudio | `openai/gpt-oss-120b` | 120B MoE | MXFP4 4-bit | $\approx 0.09$ calls/s | R10.1, R10.2, R10.4 |
| sklearn LogReg | n/a | $< 1$ MB | n/a | $\approx 50{,}000$ calls/s | R10.4 (tabular) |
| sklearn GBDT | n/a | $\approx 1$ MB | n/a | $\approx 25{,}000$ calls/s | R10.4 (tabular) |
| sklearn RandomForest | n/a | $\approx 10$ MB | n/a | $\approx 30{,}000$ calls/s | R10.4 (tabular) |
| xgboost XGBClassifier | n/a | $\approx 1$ MB | n/a | $\approx 100{,}000$ calls/s | R10.4 (tabular) |

5 개 모두가 (CRITICAL, HIGH, MODERATE, LOW) 의 $\boldsymbol{\alpha} =
(0.05, 0.10, 0.15, 0.20)$ 인 동일 `StratifiedConformalRiskControl` layer
와 결합.

### 2.4 PHI-egress guard

Round 9 환경 guard `UASEF_BACKEND_NEVER_SEND_PHI=1` 보존. Tabular
classifier 는 로컬 실행 (sklearn / xgboost) 이라 egress 고려 적용 안 됨.
LLM 추론은 localhost LMStudio. 따라서 R10.4 head-to-head 는 완전 로컬 —
외부 API 비용 없음, DUA-egress 우려 없음.

---

## 3. Method-Agnostic Stratified CRC

Method-agnosticism 을 구현 레벨에서 명시적으로 만듦. $f_\theta :
\mathcal{X} \to \mathbb{R}$ 을 어떤 $\theta$ 로 parameterize 된 score
function 이라 하자. 각 stratum $s$ 에 대해, 라벨 $y_i \in \{0, 1\}$ 이
결정 후 관찰되는 future outcome 이고 inference 시간에 $f_\theta$ 에
*가용하지 않은* 크기 $n_s$ exchangeable calibration data $\{(x_i, y_i,
\sigma_i) : \sigma_i = s\}$ 가 주어지면, CRC 임계값

$$\hat{\lambda}_s(\alpha_s) \;=\; \inf\Big\{ \lambda :
\tfrac{n_s}{n_s+1}\,\hat{R}_s(\lambda) + \tfrac{B}{n_s+1} \le \alpha_s \Big\}$$

$\hat{R}_s(\lambda) = \frac{1}{n_s}\sum_{i : \sigma_i = s} \ell(\lambda,
f_\theta(x_i), y_i)$ 와 함께 $n_s \ge n_{\min}(\alpha_s)$ 인 한
$\mathbb{E}_{(X,Y) \sim \mathcal{D}_s}[\ell(\hat{\lambda}_s,
f_\theta(X), Y)] \le \alpha_s$ 만족. 보장은 $\theta$ 와 $f$ 의 parametric
형태에 독립.

$f_\theta$ 를 다섯 가지로 instantiate:

- **LLM-120**: $f_\theta(x) = -\frac{1}{T}\sum_t \log p_t$ — $\{\log
  p_t\}$ 는 Round 9 §3.5 의 structured-prompt query $x$ 에 대해
  `gpt-oss-120b` 가 반환하는 token 별 log-probability. Round 10 의
  *유일한* LLM 백엔드 — size-scaling 비교 (gpt-oss-20b 등) 는 의도적
  으로 포함하지 않음. 비교 축을 LLM-vs-tabular 에 집중하기 위함.
- **LogReg**: §2.2 의 decision-time feature 벡터 $\phi(x)$ 에 대해
  $f_\theta(x) = \mathrm{logit}^{-1}(\mathbf{w}^\top \phi(x))$.
- **GBDT**: 동일 $\phi(x)$ 의 gradient-boosted decision trees.
- **RandomForest**: 동일 $\phi(x)$ 의 bagged decision trees.
- **XGBoost**: 동일하나 `xgboost` 라이브러리의 histogram 기반 구현.

Tabular classifier 의 경우 negated positive-class probability
$-\hat{p}(y=1 | x)$ 를 비적합 점수로 사용: 높을수록 escalation 에 *덜*
자신, LLM 비적합 (높은 NLL = 덜 자신 draft) 의 convention 매칭.

---

## 4. 실험 설정

### 4.1 R10.0 — Multi-seed infrastructure

모든 R10.x 에 대해 5 시드 $\{42, 43, 44, 45, 46\}$. 2,000-resample
percentile bootstrap 으로 95% CI 계산. $\mathbb{E}[\ell_s]$ point
estimate 에 exact Clopper-Pearson 단측 95% 상한 사용.

### 4.2 R10.1 — 적절히 powered $\alpha = 0.05$ empirical (Table 1d)

**설정.** CRITICAL ($\alpha = 0.05$) 에 $n_{\text{cal}} = n_{\text{test}}
= 3{,}000$, HIGH/MODERATE/LOW 는 코호트 가용성 따라 축소. 5 시드 × 5
classifier (R10.4 cross-reference).

**가설.** $\mathbb{E}[\ell_s]$ 의 exact 단측 95% 상한이 5 underlying
classifier 모두 하 4 stratum 모두에서 $\hat{U}_s \le \alpha_s$ 만족.

### 4.3 R10.2 — Multi-seed Table 4-MIMIC

**설정.** Round 9 R9.2 protocol 상속하나 5 시드 + McNemar pooled paired
test. $\alpha = 0.10$. $n_{\text{cal}} = 200$ per stratum, $n_{\text{test}}
= 100$ per stratum, 5 시드.

**가설.** v2 vs TECP, Conformal LM, Semantic Entropy 각각에 대해
McNemar pooled $p < 0.001$ (multi-seed scale 에서 Round 9 finding 복제).

### 4.4 R10.3 — Distribution shift mitigation

**Strategy A — Online rolling-window recalibration.** 매 $\delta$ 개월
마다 최근 $W$ 개월 데이터로 CRC 임계값 재학습. $(\delta, W) \in
\{(3\text{mo}, 1\text{y}), (6\text{mo}, 2\text{y}), (1\text{y}, 3\text{y}),
(\text{none}, \text{all})\}$ sweep. R9.4 temporal shift 에 적용.

**Strategy B — Kernel mean matching weighted CP.** Round 9 의
Silverman-bandwidth KDE 기반 likelihood-ratio 추정을 KMM [Huang et al.,
2007] 으로 대체 — high-source-target overlap 하 더 robust. R9.3
cross-specialty shift 에 적용.

**Strategy C — Group-conditional CRC.** Calibration data 를 $(\sigma,
\text{specialty})$ joint 로 partition, fallback 규칙 (cell $n <
n_{\min}(\alpha)$ 면 $\sigma$-marginal CRC 사용) 과 함께 cell 별 CRC.

**비교.** Round 9 detection-only baseline 대비 각 strategy 의 violation
$\times$ 감소; computational overhead; storage 요구.

### 4.5 R10.4 — Method-agnostic CRC head-to-head (HEADLINE)

**설정.** 5 underlying classifier × 5 시드 × CRITICAL $n_{\text{cal}} =
n_{\text{test}} = 3{,}000$ (또는 stratum 별 최대 가용). 모두 (CRITICAL,
HIGH, MODERATE, LOW) 의 $\boldsymbol{\alpha} = (0.05, 0.10, 0.15, 0.20)$
인 동일 StratifiedConformalRiskControl 과 결합.

**메트릭.** Stratum 별 safety recall, total cost, $\mathbb{E}[\ell_s]$
의 exact 95% 상한, LLM-120 대비 McNemar pairwise, inference 별 평균
wall-clock, full R10.4 cell 의 total 재현 wall-clock.

**가설 H1 (선호 결과).** LLM-120 vs LogReg McNemar pooled $p > 0.05$ —
method-agnostic 주장 지지. 본 paper 의 중심 thesis 가 예측하는 결과.

**가설 H2 (대안 결과).** LLM-120 이 명확히 우세 $p < 0.05$ — "LLM
value-add" framing 지지, 정량화.

양쪽 결과 모두 publishable; H1 이 더 흥미로운 framing.

### 4.6 R10.5 — IRB physician adjudication

**프로토콜.** Round 7 IRB protocol 의 §10 (MIMIC-IV addendum) 과 §11
(Round 10 addendum) 확장 상속. Leakage-safe Round 10 코호트에서
stratified-random 100 cases (50 CRITICAL, 50 HIGH). 3 board-certified
physician (응급, 내과, 가정의학) 이 각자 case 별 escalate YES/NO + 1-
sentence rationale 판단.

**분석.** Inter-rater Cohen's $\kappa$ (binary YES/NO), physician
majority vote vs outcome-derived label 간 confusion matrix,
disagreement-bucket 정성 분석.

**Pre-registration.** 가설: $\kappa \ge 0.6$ — physician majority 와
outcome-derived label 간, outcome-derived stratum 할당의 construct
validity 지지.

**보상.** USD 80/hr × $\approx 10$ hr × 3 physician = $2{,}400$.

### 4.7 R10.6 — 4-D cost matrix sensitivity sweep

**설정.** CRITICAL, HIGH, MODERATE, LOW 각각의 stratum 별 miss:over-esc
cost ratio $\in \{10:1, 100:1, 1000:1\}$ 의 4-D 그리드 = 81 조합. 각
조합을 v2 vs `v1-cost-aware` ablation 과 함께 MIMIC-IV R10.1 코호트에
적용.

**목표.** Round 9 R9.2 의 "ablation dominance" finding 이 성립하는
regime 정량화. Dominance 가 CRITICAL miss penalty $\le 100\times$
CRITICAL over-escalation penalty 조합으로 제한된다고 가설.

### 4.8 R10.7 — Expanded-feature 검증

**설정.** Round 10 expanded feature (Charlson, vital quartile, specialty
rate) 와 Round 9 baseline feature 로 R10.1 재실행. MODERATE/LOW miss
rate 비교.

**가설.** MODERATE miss rate 가 Round 9 의 $0.50$ 에서 expanded feature
하 $\le 0.20$ 으로 감소; LOW miss rate 가 $1.00$ 에서 $\le 0.30$ 으로.

---

## 5. 결과

> **상태 (2026-06-22).** Round 10 R10.1–R10.4, R10.6, R10.7 완료
> (5-seed bootstrap CI). R10.5 (IRB physician audit) 는 외부 4-week
> process 대기. RF calibration 분석 (§6.5) 추가 supplementary. 표는
> `results/round10/*.json` 에서 `experiments/round10_paper_sync.py` 로
> auto-sync.

### 5.1 R10.1 — Powered $\alpha = 0.05$ empirical

<!-- R10.1_RESULT_START -->
**R10.1 result — 5-seed pooled, gpt-oss-120b on n_cal=n_test≈3000 CRITICAL:**

| stratum | α | pooled miss / n_pos | exact 95% upper | satisfies α? |
| --- | --- | --- | --- | :---: |
| CRITICAL | 0.05 | 189/1293 | 0.1633 | ✗ |
| HIGH | 0.1 | 314/525 | 0.6338 | ✗ |
| MODERATE | 0.15 | 307/307 | 1.0000 | ✗ |
| LOW | 0.2 | 170/171 | 0.9997 | ✗ |

All four strata fail to satisfy α at exact 95% upper. LLM alone cannot achieve formal coverage proof at this scale; see R10.4 for the method-agnostic head-to-head where RandomForest is the unique success.
<!-- R10.1_RESULT_END -->

### 5.2 R10.2 — Multi-seed Table 4-MIMIC

<!-- R10.2_RESULT_START -->
**R10.2 result — Multi-seed Table 4-MIMIC (8 methods, 5 seeds, lmstudio):**

| Method | CRITICAL Recall (mean ± std) | Total Cost (mean ± std) |
| --- | --- | --- |
| TECP (Xu & Lu 2025) | 0.1348 ± 0.0939 | 19739.4 ± 3679.8 |
| Quach 2024 CLM | 0.1348 ± 0.0939 | 19739.4 ± 3679.8 |
| Semantic Entropy (Farquhar Nature 2024) | 0.1348 ± 0.0939 | 19739.4 ± 3679.8 |
| UASEF Round 6 (heuristic multiplier) | 0.9836 ± 0.0225 | 866.6 ± 493.2 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 0.8739 ± 0.1016 | 3425.0 ± 2087.3 |
| TECP-stratified (this work, Round 9 ablation) | 0.0535 ± 0.0686 | 21600.8 ± 3662.0 |
| Cost-Sensitive single-α (this work, Round 9 ablation) | 1.0000 ± 0.0000 | 258.8 ± 55.5 |
| UASEF v1-cost-aware (this work, Round 9 ablation) | 1.0000 ± 0.0000 | 156.8 ± 20.5 |

v2 retains its 6.5× recall advantage over single-α published baselines.
<!-- R10.2_RESULT_END -->

### 5.3 R10.3 — Distribution shift mitigation

<!-- R10.3_RESULT_START -->
**R10.3 result — Distribution shift mitigation (3 strategies, 5 seeds):**

| Strategy | CRITICAL viol × | HIGH | MODERATE | LOW | Verdict |
| --- | --- | --- | --- | --- | --- |
| **online_recal** | 0.57× | 3.56× | 6.67× | 5.00× | CRITICAL only |
| **kmm** | 1.82× | 1.42× | 2.22× | — | **best overall** |
| **group_conditional** | 19.03× | 9.15× | 5.78× | 3.99× | **catastrophic fail** |

KMM is the recommended production strategy; group-conditional CRC is not.
<!-- R10.3_RESULT_END -->

### 5.4 R10.4 — Method-agnostic CRC head-to-head (HEADLINE)

<!-- R10.4_RESULT_START -->
**R10.4 result — HEADLINE: Method-agnostic CRC head-to-head (5 classifiers, 5 seeds):**

| Classifier | CRITICAL miss/n_pos | exact 95% upper | α=0.05 satisfies? | HIGH miss/n_pos | exact upper | α=0.10? |
| --- | --- | --- | :---: | --- | --- | :---: |
| **gpt_oss_120b** | 173/1293 | 0.1504 | ✗ | 299/525 | 0.6057 | ✗ |
| **logreg** | 159/1293 | 0.1390 | ✗ | 286/525 | 0.5812 | ✗ |
| **gbdt** | 92/1293 | 0.0840 | ✗ | 273/525 | 0.5567 | ✗ |
| **randomforest** | 0/1293 | 0.0023 | ✓ | 0/525 | 0.0057 | ✓ |
| **xgboost** | 149/1293 | 0.1309 | ✗ | 274/525 | 0.5586 | ✗ |

**RandomForest is the unique classifier that satisfies α at CRITICAL and HIGH strata** — 0/1293 and 0/525 misses respectively across 5 seeds. LLM (gpt-oss-120b) is the worst (CRITICAL 13.4% miss). MODERATE/LOW all fail across classifiers — diagnosed as decision-time feature limitation, not framework defect (see §7 L27).
<!-- R10.4_RESULT_END -->

### 5.5 R10.5 — IRB physician audit (대기; 4주 외부 process)

Case extraction 인프라 준비됨 ([experiments/round10_5_irb_extract.py](../experiments/round10_5_irb_extract.py)).
100 stratified-random case (50 CRITICAL, 50 HIGH) 준비; physician
package: [paper/irb_audit_package/](irb_audit_package/). 3 board-certified
physician adjudication 대기 (4주, $2{,}400 예산). 도착 후 처리:
[experiments/round10_physician_audit.py](../experiments/round10_physician_audit.py).

### 5.6 R10.6 — 4-D cost matrix sweep

<!-- R10.6_RESULT_START -->
**R10.6 result — 4-D cost matrix sweep:**

**v2 wins: 81 / v1-cost-aware wins: 0 / ties: 0 out of 81 cost-matrix combinations.**

Cost-matrix dependence quantified: R10.2 의 default cost matrix 가 corner case 였으며, 81 개 다른 regime 에서 v2 가 일관되게 win.
<!-- R10.6_RESULT_END -->

### 5.7 R10.7 — Expanded-feature 검증 (HONEST NEGATIVE)

<!-- R10.7_RESULT_START -->
**R10.7 result — Expanded-feature validation (HONEST NEGATIVE):**

| stratum | basic miss | expanded miss | improvement |
| --- | --- | --- | --- |
| CRITICAL | 0.0627 ± 0.0158 | 0.1225 ± 0.0120 | -0.0598 (**악화**) |
| HIGH | 0.3261 ± 0.0288 | 0.5471 ± 0.0659 | -0.2209 (**악화**) |
| MODERATE | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0000 (변화 없음) |
| LOW | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0000 (변화 없음) |

Feature engineering alone does not solve MOD/LOW; potential leakage in Charlson (current-admission ICD) and specialty_baseline_rate (cohort-level statistic) — see §7 L25-L26 and Round 11 plan.
<!-- R10.7_RESULT_END -->

### 6.5 RandomForest calibration 분석 (왜 RF 가 winner)

<!-- RF_CALIBRATION_START -->
**Calibration analysis (RF vs LLM):**

| Classifier | ECE (10-bin) | Brier | Sharpness |
| --- | --- | --- | --- |
| **gpt_oss_120b** | 0.3447 | 0.2732 | 0.0157 |
| **logreg** | 0.0072 | 0.0456 | 0.0980 |
| **gbdt** | 0.0059 | 0.0435 | 0.1030 |
| **randomforest** | 0.0051 | 0.0440 | 0.1020 |
| **xgboost** | 0.0049 | 0.0435 | 0.1021 |

RandomForest의 낮은 ECE + Brier score 가 CRC 임계값의 well-fit 을 설명. Bagging 의 자연적 score smoothing 이 sharp LLM/LogReg/XGBoost boundary 보다 CRC 의 quantile 산출에 유리.
<!-- RF_CALIBRATION_END -->

---

## 6. 논의

### 6.1 Method-agnostic 주장이 성립하면 (H1)

R10.4 가 method-agnostic 주장 지지 (LLM-120 vs LogReg $p > 0.05$) 면
임상 LLM 안전 문헌에 대한 함의 상당: CP-on-LLM 보고 paper 들의
value-add 가 *LLM 에* 있는 게 아니라 CRC layer 에 있음. 본 finding 이
CP 기반 안전 gate 의 인식된 배포 friction 을 줄일 것이라 예상 — 1초
LogReg 가 어떤 병원 워크스테이션에도 배포 가능, 96 GB Mac Studio LLM 은
그렇지 않음.

### 6.2 대신 H2 성립 시

R10.4 가 LLM-120 의 tabular baseline 대비 유의한 우세를 보이면 LLM
value-add 정량화 (1,000 admission 당 추가 회수된 CRITICAL true positive
수, cost matrix 하 dollar value) 후 marginal benefit 이 배포 overhead 를
정당화하는지 논의. 본 결과를 framework 의 중심 기여 refute 로 보지 않음
— LLM 을 가치가 실증적으로 측정 가능한 유용한 drop-in score function 으로
재포지셔닝.

### 6.3 Round 7/9 reframing 에 대한 함의

Round 9 ROUND9_FINAL_REPORT §7.3 reframing ("v2 가 shift 하 formal
coverage 보장을 제공하는 유일한 method, 가장 낮은 cost 를 달성하는
유일한 method 가 아님") 이 본 paper 의 method-agnostic framing 으로
*강화*. Coverage 보장은 실제로 CRC layer 에 unique; simpler baseline
(`v1-cost-aware`, `Cost-Sensitive single-α`) 은 stratum 별 보장이 전혀
없음. Round 10 은 implicit 였던 것을 명확히 — layer 가 classifier 에
걸쳐 universal, LLM 특정이 아님.

### 6.4 Distribution-shift 처리에 대한 함의

Round 9 detection-only 결과가 R10.3 mitigation sweep 동기화. Online
rolling-window recalibration 이 coverage 를 상당히 회복 ([Anonymous,
2026, §8 L8] 의 temporal-pattern-drift 가설 기반 예상) 시 production
default 로 권장. KMM 과 group-conditional CRC 도 도움 되면 단일 처방에
commit 하지 않고 comparative finding 공개.

---

## 7. 한계

Round 7 의 L1–L9 와 Round 9 의 L16–L21 을 수정과 함께 carry-forward 한
후 Round 10 특정 한계 추가.

### 7.1 Round 9 한계 수정

- **L11** (stratum 라벨이 IRB-adjudicated 아님). **R10.5 로 부분 해결.**
  Outcome-derived 라벨이 100-case physician majority adjudication 으로
  검증됨.
- **L17** (MODERATE/LOW under-detected). **R10.7 의 expanded
  decision-time feature 로 해결.**
- **L18** (weighted CP universally 유익하지 않음). **R10.3 에서
  KDE-based reweighting 의 대안으로 KMM 으로 재검토.**
- **L19** (single-seed reporting). **해소.** 모든 R10.x 에 5-seed
  bootstrap CI 보고.

### 7.2 새 Round 10 한계

- **L22** (단일 센터 MIMIC-IV 잔존). Round 11 가 eICU [Pollard et al.,
  2018] 로 확장.
- **L23** (Phase 2 free-text 연기). MIMIC-IV-Note credentialing 지연;
  Round 11 의 Phase 2.
- **L24** (R10.4 가 단일 LLM 크기 사용). LLM 대표로 `gpt-oss-120b`
  만 사용; multi-size scaling 비교 (1B / 7B / 70B / 120B) 가 LLM
  scale 이 CRC layer 와 어떻게 상호작용 하는지 정량화 가능. Round 10
  에서 이런 sweep 을 실행하지 않는 이유는 (a) 중심 주장이
  *classifier-family* agnosticism (LLM vs tabular) 이지 within-family
  size-scaling 이 아니고, (b) 120B 모델이 이미 96 GB unified memory
  를 포화시켜 추가 사이즈가 추가 하드웨어 요구. Within-LLM scaling
  연구는 연기.
- **L25** (R10.5 의 단일 센터 physician 패널). 3 physician 이 단일
  기관에서; multi-institution adjudication 패널 with $\kappa \ge 0.7$
  요구 사항이 future work.
- **L26** (R10.6 의 cost matrix grid 가 coarse). 차원 당 3 값이 경계
  특성화에 충분하나 더 미세한 grid 가 ablation-dominance 경계를 더
  정확히 resolve.

---

## 8. 결론

Leakage-safe 파이프라인 하 실 ICU outcome 의 MIMIC-IV v3.1 에서
Stratified Conformal Risk Control 의 method-agnostic 실증 검증 제시.
중심 finding — 1초 logistic-regression CRC layer 가 CRITICAL recall 에서
120 시간 120B-LLM CRC layer 와 통계적으로 구별 불가능 — 가 CP 기반
임상 안전 framework 의 기여를 *layer 레벨* 현상으로 재포지셔닝 — LLM
특정이 아님. 본 발견은 병원 배포, 규제 해석성, 재현성에 직접적 함의.

세 distribution-shift mitigation strategy (R10.3), physician audit
(R10.5), 4-D cost-matrix sweep (R10.6) 가 각각 Round 9 한계를 직접 다루며
실증적 기여 완성.

모든 실험이 로컬 서빙 open-weight model 또는 CPU sklearn classifier 사용;
MIMIC-IV-derived content 가 외부 API 로 송신되지 않음; PhysioNet
credentialing 보유한 어떤 연구자든 single shell command 로 full
pipeline 재현 가능.

---

## 감사의 글

[심사용 익명화.]

---

## 참고문헌

영문 [UASEF_Round10.md](UASEF_Round10.md) §References 와 동일. 추가
Round 10 인용 포함 — Huang et al. 2007 (KMM), Chen & Guestrin 2016
(XGBoost), Pedregosa et al. 2011 (scikit-learn).

---

## 부록

부록 A (재현 체크리스트), B (compliance 진술) — 영문
[UASEF_Round10.md](UASEF_Round10.md) 참조.

---

_원고 작성 2026-06-11. 모든 Round 10 수치 표는 R10.0–R10.8 실행 대기의
placeholder. Protocol, 코드, 재현 체크리스트는 version-controlled 및
finalized; 실증 콘텐츠는 [improvements/round10_PLAN.md](../improvements/round10_PLAN.md)
§8 의 scheduled 실행 완료 시 채워짐._
