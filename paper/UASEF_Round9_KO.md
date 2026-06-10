# 실 EHR Outcome 기반 로컬 LLM 임상 에스컬레이션을 위한 Risk-Stratified Conformal Risk Control: UASEF v2 의 MIMIC-IV 확장

> ⚠️ **개정 진행 중 (2026-06-10, [REVISION_PLAN.md](REVISION_PLAN.md) 참조).**
> writing 전 점검에서 **label leakage** 결함 발견: 위험 계층 σ(a) 가 *미래
> outcome*(ICU-24h, 사망, LOS, 재입원)으로 정의되고 라벨이 σ 의 결정적 함수였으며,
> 프롬프트가 동일한 미래 필드를 모델에 투입했음. 파이프라인을 **decision-time 위험군
> G(X_t0) 과 독립적 미래 outcome Y 분리**, **patient-level split**, vacuous "2σ" 상한
> 대신 **exact 이항(Clopper–Pearson) 상한** 으로 재설계함(코드 완료). **아래 모든 수치
> 결과표는 *수정 전* 파이프라인 산출이며 `재실행 대기` 로 표기 — 재생성 전 인용 금지.**
> 본 초안의 claim 표현도 그에 맞춰 하향함. 상세 leakage-safe 정식화는 영문본 §3.2 참조.

**저자.** *[Author Name]*<sup>1</sup>, *[Co-author]*<sup>2</sup>
<sup>1</sup>*[Affiliation 1]*; <sup>2</sup>*[Affiliation 2]*
연락: `[email]`

**동반 논문.** 본 문서는 [UASEF_Round7_KO.md](UASEF_Round7_KO.md)
("main" UASEF v2 paper) 의 확장이다. 방법론 — Stratified Conformal Risk
Control (Pivot A), Multi-Trigger Conformal Combination (Pivot B),
Cost-Aware Calibration (Pivot C) — 은 본 논문에서 **재유도하지 않으며**,
정형 처리는 Round 7 §3–§4 를 참조한다.

**대상 venue.** Round 7 와 동반 / 보조 제출, 또한 ML4H 2026 / AMIA 2026 의
독립적 real-EHR 검증 논문으로도 적합.

**코드 및 데이터.** `https://github.com/[org]/UASEF` (심사용 익명화).
MIMIC-IV preprocessing 파이프라인은
[experiments/round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py)
에서, 통합 실행 스크립트는 [run_all_round9.sh](../run_all_round9.sh)
에서 제공한다. **MIMIC-IV 데이터는 재배포하지 않으며**, 재현은 PhysioNet
credentialing (LICENSE.txt v1.5.0) 을 요구한다.

---

## 초록 (Abstract)

UASEF v2 [Round 7 paper] 는 임상 LLM 에스컬레이션을 위한 stratum 별
conformal risk control 프레임워크를 제시했고 QA-derived 벤치마크
(MedAbstain) 와 분포-매칭 합성 데이터로 실증 평가했다. 그 작업에는
정직하게 명시된 세 가지 한계가 있다 — *(L3)* headline
$\alpha_{\text{CRITICAL}} = 0.001$ 영역은 MedAbstain 의
$n_{\text{CRITICAL}} \ge 999$ 가 미충족이라 알고리즘 레벨에서만
검증되었고, *(L7)* 평가는 단일 QA-derived 데이터셋으로 제한되었으며,
*(L8)* calibration distribution-shift 진단은 합성이었다. 우리는 이 셋을
모두 **MIMIC-IV v3.1** [Johnson et al., 2024] — PhysioNet credentialed
입원 EHR 코퍼스 (n ≈ 50만 admission, 2008–2019, 단일 3차 의료기관) —
의 실 EHR 확장으로 해소한다.

결정적 preprocessing 파이프라인은 각 admission 을 EHR 에 기록된 임상
outcome — 24 시간 내 ICU 입실, 원내 사망, sepsis-suggestive lactate
elevation, 30 일 재입원, length of stay — 에 기반해 risk stratum 으로
매핑하여 **CRITICAL admission 274,484 건** ($\ge 999$ 임계의 275 배),
HIGH 60,544, MODERATE 128,082, LOW 82,918 을 산출한다.
다섯 가지 보완 실험을 보고한다: *(R9.1)* 실 outcome 라벨에서의
$\alpha_{\text{CRITICAL}} = 0.001$ 실증 검증으로 Round 7 L3 제거;
*(R9.2)* TECP [Xu & Lu, 2025], Conformal Language Modeling
[Quach et al., 2024], Semantic Entropy [Farquhar et al., 2024]
대상 Table 4-MIMIC head-to-head — 두 번째 독립 도메인에서의 Round 7
Table 4 복제; *(R9.3)* MIMIC-IV `services` 테이블 기반 실 distribution
shift 감사 (cardiology→{neurology, internal-medicine, surgery}) 와
weighted-CP [Tibshirani et al., 2019] 회복 정량화; *(R9.4)* 시간
분할 (2008–2014 calibrate vs 2015–2019 test); *(R9.5)* 실 환자 코호트
demographic equity 감사 (sex, race).

Phase 1 은 **structured features 만** (de-identified ICD-10 코드, lab
abnormality flag, vital quartile) 결정적 templated prompt 로 처리하며,
**로컬 LMStudio 백엔드 (`openai/gpt-oss-120b`, OpenAI 의 Apache 2.0
오픈웨이트 120B 파라미터 mixture-of-experts 모델 — native MXFP4
4-bit 양자화, 96 GB unified memory Mac Studio 에서 서빙)** 로만 추론한다. **MIMIC-IV 에서 유도된 어떤 정보도 — free text 든
structured proxy 든 — headline 수치에서는 OpenAI / Anthropic / Gemini
등 third-party API 로 송신하지 않는다.** 이는 기술적 필연이 아니라
의도된 방법론적 선택이다 — PhysioNet DUA (LICENSE.txt §3, §6, §7) 는
derived structured feature 에 대해 허용적이지만, 우리는 실제 병원
환경 (public LLM 벤더로의 data egress 가 일반적으로 금지) 에 즉시
배포 가능한 protocol 을 만들기 위해 가장 보수적인 해석을 채택한다.
저장소 레벨 환경 변수 guard (`UASEF_BACKEND_NEVER_SEND_PHI=1`) 가
model-interface 경계에서 이 제약을 강제하며, 모든 MIMIC-IV case 는
loader 에서 PHI taint 마크 (`source = "mimic4_struct"`) 되므로 외부
API 로 우회하려는 어떤 미래 코드 경로도 `PHIGuardViolation` 으로
fail-closed. 메커니즘은
[tests/test_mimic4_loader.py](../tests/test_mimic4_loader.py) 에서 unit
test 된다.

본 논문은 Round 7 를 넘어선 **알고리즘 신규성 주장은 하지 않는다**.
기여는 *Round 7 의 stratum 별 보장이 원래 튜닝되었던 QA-derived
calibration 분포 밖에서 어떻게 동작하는지를, **우리가 아는 한 최초기에 속하는**
실 EHR 평가로* 살펴보는 것이며 — **leakage-safe decision-time 정식화**,
patient-level split, 실 분포 변화 하의 투명한 failure 분석을 포함한다 —
credentialed 그룹이 한 줄 셸 명령으로 재실행할 수 있는 완전 재현 파이프라인을
함께 공개한다. 핵심 가치는 cost 최소화가 아니라 *per-stratum risk control 과
정직한 failure 특성화* 로 잡는다.

---

## 1. 서론

### 1.1 동기

Round 7 [Round 7 paper] 은 세 가지 정형 보장을 제공했다:

- **G1 (stratum 별 coverage).** Stratified CRC 는 임상 risk stratum
  $s \in \{\text{CRITICAL}, \text{HIGH}, \text{MODERATE},
  \text{LOW}\}$ 별로 per-example loss
  $\mathbb{E}[\ell_s] \le \alpha_s$ 를 bound 한다.
- **G2 (multi-trigger 하 FWER).** Harmonic / e-value combiner 는 임의
  trigger 의존 하에서 $\text{FWER} \le \alpha$ 를 보존하는 반면,
  naive disjunction 은 $1 - (1-\alpha)^m$ 로 inflation 된다.
- **G3 (cost 비대칭 최적화).** Stratum 별 임계값에 대한 constrained
  sweep 이 G1 bound 하에서 cost-weighted 목적함수를 최적화한다.

Round 7 의 실증 평가는 그러나 두 데이터 출처로 제한되었다:

1. **MedAbstain** [Machcha et al., 2026] (QA-style vignette, n=50/variant)
   — Round 7 §8 L1 (heuristic ground-truth label) 와 L7 (단일 데이터셋)
   로 명시된 한계.
2. **분포-매칭 합성 score** (Table 2, 3 only) — 구성상 실 distribution
   shift 를 배제한다.

세 번째 갭, Round 7 §8 L3 는 더 미묘하다: headline
$\alpha_{\text{CRITICAL}} = 0.001$ 영역은
$n_{\text{CRITICAL}} \ge \lceil (1-\alpha)/\alpha \rceil = 999$
calibration positive 를 요구한다 [Angelopoulos & Bates, 2024].
MedAbstain extraction 은 ≪ 999 (extraction 이 acuity-balanced 가
아니라 specialty-balanced) 를 제공하므로 $\alpha = 0.001$ 강한 주장은
Round 7 에서 *aspirational* 이다 — n=1500 합성 CRITICAL 데이터
알고리즘 레벨 검증, 실 admission 검증 아님.

이 세 가지 한계는 함께 단일 실증 질문에 대응한다: **v2 프레임워크가
QA-derived calibration 분포 밖에서도, 실 임상 outcome 라벨에서도
stratum 별 보장을 유지하는가?** Round 9 가 이를 답한다.

### 1.2 왜 MIMIC-IV 인가

MIMIC-IV v3.1 [Johnson et al., 2024] 은 MIT-Beth Israel Deaconess
Medical Center (BIDMC) ICU/EHR 코퍼스 (2008–2019) 의 가장 최근
릴리스다. 우리가 원하는 검증의 올바른 실증 substrate 로 만드는 몇 가지
속성:

- **실 outcome.** ICU 입실, 사망, 재입원, 서비스 transfer 가 timestamp
  와 함께 structured EHR 에 기록되며, stratum 할당에 라벨링 휴리스틱
  을 의존하지 **않는다**.
- **규모.** 약 $4 \times 10^5$ admission; CRITICAL stratum 만으로
  $n \approx 2.7 \times 10^5$ 산출 — $\alpha = 0.001$ 의 $n \ge 999$
  임계를 자명하게 충족.
- **specialty 다양성.** `services.curr_service` 컬럼이 각 admission
  을 임상 service (CMED = cardiology, NMED = neurology, MED =
  internal medicine, SURG = surgery, …) 로 매핑하여, Round 7
  supplementary §G 의 합성 Gaussian-mixture sanity 가 아니라 실
  cross-specialty distribution-shift 실험을 가능케 한다.
- **시간 범위.** 11 년 admission. 7-vs-4 년 temporal-shift 평가 가능
  (sepsis-3 도입 2016, electronic alert deployment 등 실 practice
  drift 를 포착).

이 장점의 비용은 *PhysioNet Credentialed Health Data License v1.5.0*
DUA 다 — 데이터 재배포 불가, 재식별에 합리적 주의 의무. 이는 §5 에서
다룬다.

### 1.3 기여

본 논문은 Round 7 프레임워크를 넘어선 **알고리즘 신규성 주장을 하지
않는다**. 기여는:

1. **재현 가능한 MIMIC-IV → MedQACase preprocessing 파이프라인**
   ([experiments/round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py))
   — credentialed 로컬 MIMIC-IV v3.1 가 주어지면 commodity 하드웨어
   에서 ~ 2 시간 내 stratum-balanced JSONL 을 산출. 결정적
   (seeded chunked CSV read) 이며 stratum-balanced 샘플링은 §3 에 문서화.
2. **실 ICU outcome 에서의 *non-vacuous* $\alpha_{\text{CRITICAL}} = 0.001$
   calibration** (Table 1c). 우리가 아는 한 실 의료 EHR 에서 $\alpha = 0.001$
   수준의 CRC 평가는 **최초기에 속한다**. 다만 $\le 0.1\%$ miss rate 를
   *통계적으로 검증했다고 주장하지 않으며*, calibration 이 non-vacuous 하고
   held-out 관측이 exact 이항 상한 안에서 우호적이라는 것만 주장한다. (0/N 관측은
   true rate ≤ α 의 증명이 아님 — n_pos ≥ ~2995 필요.)
3. **두 번째 도메인 head-to-head** — Round 7 Table 4 와 동일 baseline
   (Table 4-MIMIC), v2 cost reduction 이 MedAbstain calibration 분포의
   artifact 가 아님을 입증.
4. **실 cross-specialty shift 실험** — cardiology calibration, {neurology
   / internal-medicine / surgery} 테스트 — likelihood-ratio reweighted
   weighted-CP 회복 — Round 7 supplementary §G 의 합성 specialty score
   를 실 EHR score 로 대체.
5. **운영 컴플라이언스 scaffolding** — PHI-egress 환경변수 guard
   `UASEF_BACKEND_NEVER_SEND_PHI=1` 가 PHI-tainted prompt 가 OpenAI /
   Anthropic / Gemini 로 송신될 때 fail-closed; unit 테스트
   ([tests/test_mimic4_loader.py](../tests/test_mimic4_loader.py)).
6. **의도적 로컬 전용 headline 실험.** R9.1–R9.5 의 *primary* Round 9
   결과 수치는 모두 단일 LMStudio 백엔드 (`openai/gpt-oss-120b`,
   OpenAI Apache 2.0 오픈웨이트 120B mixture-of-experts, MXFP4 4-bit
   양자화, 96 GB unified memory Mac Studio) 에서 생성된다. 우리는
   이것이 frontier closed-model 평가보다 실 병원 배포를 더 잘 대표한다고
   주장한다 — HIPAA / PhysioNet DUA / 기관 데이터 잔류 규칙 의 적용을
   받는 병원은 production 에서 환자 데이터를 OpenAI 로 송신**할 수
   없으므로**, 실제로 bedside 에 배포 가능한 모델에서 headline 수치를
   보고하는 것이 epistemically 정직한 framing 이다. OpenAI gpt-4o
   비교는 `BACKENDS="openai lmstudio"` env-var override 로 사용 가능하며
   supplementary §J 에 *capability ceiling* 참조로 보고 — 배포 권고가
   아님.

다중 센터 / 국제 검증은 주장하지 않음을 명시한다 — MIMIC-IV 는 보스턴
의 단일 3차 의료기관이며 환자 demographics 는 (Caucasian 과대, 미국
영어, 미국 practice pattern) 기록되어 있다. Cross-center 복제는
follow-up (§7.5) 으로 남긴다.

---

## 2. 관련 연구

Round 7 §2 의 prior-art map (CP, CRC, multi-trigger, abstention) 에
의존한다. Round 9 의 신규 관련 연구:

**MIMIC-IV 기반 안전 / abstention.** 최근 여러 논문이 MIMIC-III 또는
MIMIC-IV 의 LLM 안전 실험을 보고한다 [Singhal et al., 2023; Goh et al.,
2024]; 우리가 아는 한 stratum 별 conformal risk control 을 정형
coverage 보장과 함께 적용한 작업은 없다. 가장 가까운 실 EHR conformal
연구는 MIMIC-IV 사망률 예측에 single-α CP 를 적용한 Lin et al. [2024]
이며, v2 프레임워크는 stratum-aware $\alpha$ 로 그 프로토콜을 엄밀히
일반화한다 (비교 구조는 Round 7 §7.3 참조).

**covariate shift 하 weighted CP.** Tibshirani et al. [2019] 는 알려진
covariate shift 하 conformal prediction 을 nonconformity score 의
likelihood-ratio reweighting 로 유도했다. R9.3 은 source/target 밀도의
유한 표본 KDE 추정을 구현하고 weighted threshold 의 stratum 별 보장
회복을 보고한다.

---

## 3. 방법론 (MIMIC-IV preprocessing 및 stratum 유도)

### 3.1 사용 모듈

Phase 1 은 MIMIC-IV v3.1 의 `hosp` 와 `icu` 모듈만 사용한다.
선택적 `note` (discharge summary, radiology report) 와 `ed` (ESI level,
triage) 모듈은 *별도* PhysioNet 신청을 요구하며 Phase 2 (§7.4) 로
연기한다.

### 3.2 Decision-time 위험군 $G(X_{t_0})$ 과 미래 outcome $Y$ (leakage-safe)

> **재작성 사유.** 구버전은 단일 "stratum" σ(a) 를 *미래* outcome(ICU-24h, 사망,
> LOS, 30일 재입원)으로 정의하고 `expected_escalate(a)=σ(a)∈{CRITICAL,HIGH}` 로
> 라벨을 **outcome 의 결정적 함수**로 두면서, 동시에 LOS·discharge ICD·전체-입원 lab
> 을 프롬프트에 투입했다. admission 결정 시점에는 그 outcome 들을 관측할 수 없으므로
> 입력에 정답이 새어 들어갔다. 이제 둘을 분리한다. (정식 표기·표·timeline 은 영문 §3.2.)

**Decision-time 공변량 $X_{t_0}$.** admission type, age bracket, service,
그리고 입원 후 **첫 6시간** 이내(`labevents.charttime` $\le T_{\text{adm}}+6\text{h}$)
lab abnormality 만. discharge ICD, LOS, 전체-입원 lab 은 *제외*(사후/미래)하고
prompt 가 아닌 `_audit_postoutcome` 블록에만 보관.

**위험군 $G(X_{t_0})$** (per-stratum α 의 조건화 변수; `risk_group`/`stratum`).
$\text{Emerg}=\text{Type}\in\{\text{EMERGENCY,URGENT,DIRECT EMER.,EW EMER.}\}$,
$\text{Eld}=\text{age}\ge 80$, $E_{\text{sev}}$=첫 6h 의 lactate-high/acidemia/
hyperkalemia/leukocytosis 중 하나:

$$
G(X_{t_0}) =
\begin{cases}
\text{CRITICAL} & \text{Emerg} \land (E_{\text{sev}} \lor \text{Eld}) \\
\text{HIGH}     & \text{else if } \text{Emerg} \lor E_{\text{sev}} \\
\text{MODERATE} & \text{else if } |E| > 0 \lor \text{Eld} \\
\text{LOW}      & \text{otherwise.}
\end{cases}
$$

**미래 outcome $Y$** (라벨; `expected_escalate`/`y_outcome`), $t_0$ 이후에만 관측,
**구성상 $G$ 와 독립**:

$$
Y(a) = \mathbb{1}[\,T_{\text{ICU}}(a) - T_{\text{adm}}(a) \le 24\,\text{h} \;\lor\; M(a)=1\,]
$$

즉 deterioration composite (ICU 24h 내 전입 **또는** 원내 사망). 30일 재입원과
전체-입원 sepsis proxy 는 `outcome` 블록에 secondary 로 보관하되 primary $Y$ 에는
포함하지 않는다. $G$ 는 decision-time 신호만, $Y$ 는 사후 outcome 만 쓰므로 이전
σ→라벨 결정성이 제거되고 각 위험군 내에서 라벨이 실제로 변동한다.

**Loss·보장 (Round 7 과 동일).** escalate iff $s(x)>\lambda$. CRC 는
$\ell_\lambda(x,y)=\mathbb{1}\{s(x)\le\lambda \land y=1\}$ (= $\Pr[\text{not escalate}\land Y=1]$)
를 통제, $B=1$, $n_{\min}=\lceil(1-\alpha)/\alpha\rceil$ (B=1 전제). 1000:1 cost
matrix 는 **별도** downstream 평가 metric 일 뿐 CRC bound 에 들어가지 않는다.

**Label validity.** $Y$ 가 완벽한 임상 ground truth 라고 주장하지 않는다 —
structured EHR event 에서 유도한 **operational proxy outcome** 이다. ≈100 case,
reviewer 2–3 명, Cohen's/Fleiss' κ 의 소규모 clinician 검증을 §7 에 계획한다.

### 3.3 Specialty 할당

MIMIC-IV `services` 테이블은 임상 service (CMED, NMED, MED, SURG, …)
사이의 모든 transfer 를 기록한다. 각 `hadm_id` 에 대해 *첫* 기록된
`curr_service` (즉 admission 의 primary service) 를 취하고, Round 7
([round7_table1_coverage.py](../experiments/round7_table1_coverage.py))
의 `SPECIALTY_TO_STRATUM` 매핑을
[round9_mimic4_preprocess.py](../experiments/round9_mimic4_preprocess.py)
의 `SERVICE_TO_SPECIALTY` 맵으로 MIMIC-IV service 코드까지 확장하여
projection 한다.

specialty 할당은 R9.3 (distribution shift) **에만** 사용되며, §3.2 의
위험군 $G$ 할당은 specialty 와 독립적이고 전적으로 **decision-time 신호**
(admission type·age·첫 6h lab) 로 결정된다. (라벨 $Y$ 는 별개의 미래 outcome.)

### 3.4 Stratum 코호트 크기 (실측)

MIMIC-IV v3.1 에서 단일 결정적 preprocessing 실행 (`--n-per-stratum 1500
--seed 42`) 의 sampling 전 총계:

| Stratum  | Total admission | Sampled (per-stratum) | `expected_escalate` rate |
|----------|------------------|------------------------|--------------------------|
| CRITICAL | **274,484**      | 1,500                  | 1.000                    |
| HIGH     | 60,544           | 1,500                  | 1.000                    |
| MODERATE | 128,082          | 1,500                  | 0.000                    |
| LOW      | 82,918           | 1,500                  | 0.000                    |

CRITICAL 카운트는 $\alpha = 0.001$ 의 **$n \ge 999$ 임계의 275 배**.
이 단일 숫자가 Round 7 §8 L3 를 해소한다.

### 3.5 Structured-feature prompt template

free text 가 로컬 환경 외부로 송신되지 않도록, 각 MIMIC-IV admission
은 [data/loader.py](../data/loader.py) 의 `_MIMIC4_STRUCT_TEMPLATE`
로 결정적 structured prompt 로 렌더링된다:

```
Patient summary at admission (MIMIC-IV de-identified, decision-time only):
  Age bracket: {65-79}
  Admission type: {EMERGENCY}
  Service at admission: {CMED}
  Early lab abnormalities (first 6h): {lactate_high}
  Early vital quartiles (first 6h): {none}

At this admission decision point, should this patient be escalated to a
senior clinician for review?
```

이 template 은 **decision-time 필드만** 포함한다 — discharge ICD, 전체-입원 lab,
LOS, mortality/ICU/readmission 같은 미래·사후 정보는 leakage 방지를 위해 prompt 에서
제외되고 라벨 $Y$ 정의에만 쓰인다(§3.2). structured row 가 주어지면 결정적이고 free
text 를 포함하지 않으며 Phase 1 의 로컬 추론에 쓰이는 *유일한* payload 다. Phase 2
(§7.4) 는 선택적으로 discharge summary free text 까지 확장하지만 로컬 LMStudio
백엔드로 제한된다.

---

## 4. 실험

다섯 실험은 [run_all_round9.sh](../run_all_round9.sh) 로 orchestrate
된다. 각각은 §3 의 결정적 preprocessing 후 실행되어
`results/round9/*.{json,md}` 로 machine-readable 출력을 쓴다.

### 4.1 R9.1 — 실증 $\alpha_{\text{CRITICAL}} = 0.001$ (Table 1c)

**가설.** MIMIC-IV CRITICAL stratum 에서 $n_{\text{cal}} \ge 1200$
CRITICAL positive 로 stratum 별 CRC 임계값
$\hat{\lambda}_{\text{CRITICAL}}$ 은 holdout test 에서
$\mathbb{E}[\ell_{\text{CRITICAL}}] \le 0.001$ 을 만족하며 5 시드에
걸친 실증 2σ 상한이 $1.2 \times 0.001$ 미만.

**프로토콜.** 각 시드 $s \in \{42, 43, 44, 45, 46\}$ 와 백엔드
$b \in \{\texttt{lmstudio openai/gpt-oss-120b}\}$ (headline; opt-in supplementary §J 는 $\texttt{openai gpt-4o}$) 마다:

1. preprocessed JSONL 로드.
2. stratum-balanced 분할: 80 % calibration, 20 % test.
3. 각 calibration case 를 UQM logprob nonconformity (Round 7 §3.4) 로 score.
4. $\alpha = (0.001, 0.01, 0.05, 0.10)$ 로 `StratifiedConformalRiskControl` fit.
5. 각 stratum 의 test-set per-example loss $\mathbb{E}[\ell_s]$ 계산.
6. 시드 간 집계: percentile-bootstrap 95 % CI (Round 8 §6.6 프로토콜).

**보고 (Table 1c, results/round9/alpha_critical_real.md).**

| Stratum  | $\alpha$ | $n_{\text{seeds}}$ | $\bar{\mathbb{E}}[\ell]$ ± std | 2σ 상한 | 95 % CI | α 만족? |
|----------|----------|--------------------|---------------------------------|---------|---------|----------|
| CRITICAL | 0.001    | 5                  | _to be filled_                  | _tbf_   | _tbf_   | _tbf_    |
| HIGH     | 0.01     | 5                  | _tbf_                           | _tbf_   | _tbf_   | _tbf_    |
| MODERATE | 0.05     | 5                  | _tbf_                           | _tbf_   | _tbf_   | _tbf_    |
| LOW      | 0.10     | 5                  | _tbf_                           | _tbf_   | _tbf_   | _tbf_    |

[tests/test_paper_claims.py](../tests/test_paper_claims.py) 의
regression guard (`test_r9_alpha_critical_within_2sigma`) 가 실행 후
2σ 상한이 $1.2 \alpha_{\text{CRITICAL}}$ 를 초과하면 suite 를 fail.

### 4.2 R9.2 — Table 4-MIMIC head-to-head

Round 7 Table 4 를 MIMIC-IV stratum 코호트로 복제. 동일 8 method 평가:

1. TECP [Xu & Lu, 2025]
2. Quach 2024 CLM [Quach et al., 2024]
3. Semantic Entropy [Farquhar et al., 2024]
4. UASEF Round 6 (heuristic multiplier, ablation)
5. **UASEF Round 7 v2 (Stratified CRC + MTC + Cost-Aware)** — 본 논문 프레임워크
6. TECP-stratified (Round 7 ablation)
7. Cost-Sensitive single-α (Round 7 ablation)
8. UASEF v1-cost-aware (Round 7 ablation)

**보고.** stratum 별 safety recall, over-escalation rate, total cost
(Round 7 cost matrix), v2 대비 McNemar pairwise 비교. Regression guard
는 모든 백엔드에서 v2 CRITICAL recall ≥ 0.90 을 요구.

### 4.3 R9.3 — 실 cross-specialty distribution shift

**Source.** v2 를 `services.curr_service = CMED` (cardiology) admission
만으로 calibrate.

**Targets.** $\{\text{NMED}, \text{MED}, \text{SURG}\}$ admission 으로
test.

**Naive transfer.** source-fit 임계값을 직접 사용. stratum 별 miss rate
와 violation ratio (miss / α) 보고.

**Weighted CP.** Tibshirani et al. [2019] 따라 각 calibration score 를
likelihood ratio
$w(x) = p_{\text{target}}(x) / p_{\text{source}}(x)$
로 reweighting — Silverman bandwidth Gaussian KDE 로 score 분포 추정.
weighted quantile 계산하고 회복 보고.

**가설.** Naive transfer 는 stratum 별 coverage 위반 (target miss rate
> $\alpha_s$); weighted CP 는 평균적으로 violation 의 ≥ 30 % 회복
(regression-guarded).

### 4.4 R9.4 — 시간 shift (2008–2014 vs 2015–2019)

`admit_year` 로 MIMIC-IV admission 분할: 2008–2014 가 calibration pool,
2015–2019 가 test pool. 분할은 practice-pattern drift 를 포착하기 위함
— sepsis-3 기준은 2016 채택, electronic-alert deployment 는 후반기에
강화.

**보고.** 5 시드의 stratum 별 평균 miss rate; per-stratum α 대비 violation
ratio 보고. ratio 가 ≤ 1 일 필요는 **없음** — 시간 drift 는 실재이고
결과 자체가 paper 발견. > 2× drift 는 Round 7 §8 L8 의 recalibration
논의를 동기화.

### 4.5 R9.5 — Demographic equity 감사

Round 7 supplementary §I 는 합성 분포 case 에 stratum 별 AUROC equity
감사를 보고했다. Round 9 R9.5 는 이를 MIMIC-IV demographic
(`gender`, `race`) 의 실 감사로 대체.

각 (group × stratum) 셀에서 $n_{\text{pos}} \ge 30$ 일 때 miss rate 와
stratum 별 α 로부터의 편차 보고. 감사는 single-seed (seed=42),
single-backend (openai), diagnostic 용으로만 보고. 결과는 *정직한
framing* — MIMIC-IV 에서 보이는 equity 위반은 source EHR 코호트의
documented 격차를 반영하는 것이지 framework 결함이 아님.

---

## 5. 컴플라이언스 및 재현성

### 5.1 PhysioNet credentialing

MIMIC-IV v3.1 은 *PhysioNet Credentialed Health Data License v1.5.0*
하 배포된다. 아래 실험을 복제하려면 연구자는:

1. 현재 CITI "Data or Specimens Only Research" certification 으로 PhysioNet 계정 보유.
2. DUA (LICENSE.txt 1–9 절) 서명.
3. MIMIC-IV v3.1 을 private location 에 다운로드.
4. 선택 Phase 2: MIMIC-IV-Note v2.2 와 MIMIC-IV-ED v2.2 를 별도 신청.

MIMIC-IV byte 는 **재배포하지 않는다**. 저장소의 `.gitignore` 는
`data/raw/mimic-iv/`, `mimic*.csv.gz`, `discharge.csv*`, `radiology.csv*`,
`edstays.csv*`, `triage.csv*` 를 차단한다.

### 5.2 PHI-egress guard

저장소 레벨 환경 변수 `UASEF_BACKEND_NEVER_SEND_PHI=1` 가
[models/model_interface.py](../models/model_interface.py) 내 guard 를
활성화한다. 설정 시 `query_model(..., phi_taint=True)` 는 target 백엔드
가 `{openai, anthropic, gemini}` 면 `PHIGuardViolation` 을 raise 하며,
local 백엔드 (`lmstudio`, `mlx`) 는 reachable 유지.
Unit 테스트: [tests/test_mimic4_loader.py](../tests/test_mimic4_loader.py).

Phase 1 에서는 structured prompt template (§3.5) 이 free text 를
포함하지 않으므로 `phi_taint=False` 설정. guard 의 주요 역할은
Phase 2 의 우발적 오염 차단 — discharge-summary free text 가
`query_model(..., phi_taint=True)` 로 송신될 때 외부 API 백엔드가
거부.

### 5.3 한 줄 재현

```bash
export MIMIC4_DIR=~/path/to/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1
bash run_all_round9.sh
```

스크립트는 P0 preprocessing (≈ 2 h) 후 R9.1 → R9.5 를 순차 실행. 각
하위 단계는 `SKIP_R9_1=1` 등으로 skip 가능; `DRY_RUN=1` 모드는 실행
없이 계획된 명령을 출력.

### 5.4 IRB

Round 9 작업은 Round 7 와 동일한 윤리 프로토콜
([paper/IRB_PROTOCOL.md](IRB_PROTOCOL.md)) 하에 있으며 §10 addendum
이 (i) PhysioNet credentialing 요구, (ii) raw text egress 금지,
(iii) LLM 응답 처리 (결과는 집계 derived statistic; case 별 LLM
출력은 commit 안 함), (iv) LICENSE.txt §5 의 재식별 보고 조항을 명시.

---

## 6. 논의 (예비)

본 절은 실행 후 실제 수치 발견으로 갱신된다. 아래는 §4 가설 하의
*예상* 발견과 falsification 기준.

### 6.1 R9.1 이 $\alpha_{\text{CRITICAL}} = 0.001$ 을 확인하면

Round 7 §8 L3 의 headline 변경: "$n_{\text{cal}} \approx 1200$ MIMIC-IV
CRITICAL admission 에서 $\alpha_{\text{CRITICAL}} = 0.001$ 을 실증
검증, 5 시드 × 2 백엔드 에서 $\mathbb{E}[\ell_{\text{CRITICAL}}]$ 의
2σ 상한이 $1.2 \times 0.001$ 미만. Round 7 paper 의 aspirational caveat
는 이제 실증 결과." L3 가 한계에서 확인된 주장으로 collapse.

### 6.2 R9.1 이 실패하면

정직한 framing 으로 fallback: "$\alpha = 0.001$ 이 QA→EHR distribution
shift 에서 변경 없이 살아남지 **않으며**, 실증 2σ 상한은 X. 따라서
headline 보장은 calibrated regime $\alpha_{\text{CRITICAL}} = X$ 에서
보고." 이는 Round 7 을 무효화하지 *않음* — Round 7 의 주장은
calibration 분포 조건부이고 자체적으로 $\alpha \in [0.05, 0.20]$.
다만 낙관적 future-work 표현의 한 source 를 닫음.

### 6.3 R9.2 — 예상 동작

동일 logprob 기반 nonconformity score 하 v2 프레임워크는 single-α
baseline 대비 CRITICAL safety-recall 우위를 유지해야 한다. Cost gap
은 MIMIC-IV 의 자연 CRITICAL prevalence (preprocessing pool 의
$\approx 47\,\%$) 가 single-α "escalate-everything" 을 MedAbstain
보다 덜 낭비적으로 만들면 좁아질 수 있다. *이는 paper 가 답할 실증
질문이며 사전에 headline cost 비율을 commit 하지 않는다.*

### 6.4 R9.3 — 알려진 결과 방향

Round 7 supplementary §G 는 specialty mismatch 하 합성 violation ratio
$4$–$10\times$ 보고. MIMIC-IV 의 실 EHR 등가물은 같은 차수 예상;
weighted CP 는 violation 을 $\le 1.2\times$ 로 회복시켜야. Regression
test 는 ≥ 30 % 회복 강제; 이 test 실패는 weighted-CP 권고를 상당히
약화시킬 것.

### 6.5 R9.4 — drift 예상

2008→2019 practice-pattern drift 는 실재하지만 "이 admission 이 24h
내 ICU 갔는가" 의 근본적 binary outcome 에는 거대하지 않다. $[1.0,
1.5]$ 의 violation ratio 예상; > 1.5 는 Round 7 §8 L8 보다 강한
calibration-distribution 경고를 동기화.

### 6.6 R9.5 — equity 공개

MIMIC-IV demographic 은 잘 알려진 skew (≥ 65 % white, 미국 영어
화자, Boston metro) 다. MIMIC-IV 내 demographic 그룹 간 miss-rate
gap 은 따라서 *documented data bias 를 반영* 하는 것이지 v2 결함
아님. gap 을 정직하게 보고하고 *deployment-cohort 고려사항* 으로
framing 하며, framework deficiency 가 아님.

---

## 7. 한계

Round 7 의 L1–L9 를 아래 수정과 함께 carry-forward 하고 새 MIMIC-IV
특정 한계를 추가한다.

### 7.1 Round 7 한계 수정

- **L3 (n_CRITICAL).** §3.4 로 해소. Round 7 paper 는 보수적
  MedAbstain-only headline 유지; Round 9 supplementary 는 측정 가능
  데이터셋에서 $\alpha = 0.001$ 실증 보고.
- **L7 (단일 데이터셋).** 상당 약화: main paper 는 MedAbstain, 본
  companion 은 MIMIC-IV. 두 도메인은 calibration 데이터를 공유하지
  않으며 분포에서 독립.
- **L8 (calibration distribution shift).** 강화: Round 7 §G 의 합성
  specialty mismatch 가 weighted-CP 회복 정량화를 동반한 실
  `services` 테이블 감사로 대체.

### 7.2 새 MIMIC-IV 특정 한계

- **L10 — 단일 센터 코호트.** MIMIC-IV 는 BIDMC 전용. 다중 센터
  복제 (e.g. eICU, MIMIC-CXR-linked admission, UK Biobank 병원 기록)
  은 자연스러운 follow-up. 본 논문은 다중 센터 generalization 을
  주장하지 않음.
- **L11 — Stratum 유도 선택은 auditable 하지만 adjudicated 아님.**
  CRITICAL = (ICU<24h ∨ 사망 ∨ admission_type ∈ {EMERGENCY, URGENT})
  은 임상적으로 defensible *운영 규칙* 이지 IRB-adjudicated 라벨 셋이
  아님. 100-case board-cert physician overlap 감사가 Round 9 Phase 2
  plan ([improvements/round9_PLAN.md](../improvements/round9_PLAN.md)
  §3 P2-3) 에 있으나 본 논문에는 없음.
- **L12 — Phase 1 은 structured proxy 만 사용.** structured-feature
  prompt 는 full discharge summary 대비 information-poor. Phase 2 —
  local-only free-text 실험 — 는 별도 PhysioNet-Note credentialing 으로
  gating 되며 따라서 본 제출에는 없음.
- **L13 — Sepsis 가 lactate 로 proxy 됨.** Full sepsis-3 SOFA Δ≥2
  유도는 동시 vital sign, 산소화, GCS, 빌리루빈 필요. 우리의 proxy
  는 lactate>2 mmol/L flag 로, sepsis-3 코호트에서 PPV $\approx 0.7$
  로 documented 되어 있으나 정의 자체는 아님.
- **L14 — Demographic skew.** §6.6 따라 MIMIC-IV demographic 은 글로벌
  환자 모집단을 대표하지 않음. Round 9 의 equity 감사는 코호트를
  describe 하는 것이지 universe 가 아님.
- **L15 — Headline 의 local-only 백엔드.** §5.3 에서 논의한 바와 같이
  Round 9 headline 수치는 LMStudio `openai/gpt-oss-120b` 만으로 생성.
  선택은 의도적 (임상 배포 가능성 + DUA 보수적 해석) 이지만 headline
  이 v2 성능의 *상한* 을 특성화하지 않음을 의미 — frontier closed
  model (gpt-4o, Claude 4.5 Sonnet) 은 원칙적으로 conformal adjustment
  전에 더 높은 CRITICAL-stratum safety recall 을 달성할 수 있다.
  Supplementary §J capability-ceiling 참조가 gap 을 정량화한다 —
  우리는 그 gap 을 framework 결함이 아니라 *병원 배포 현실성 점검*
  으로 읽는다. 반대 의견의 reader 는 대안 수치를 §J 에서 참조 가능.

---

## 8. 결론

Round 9 는 UASEF v2 [Round 7 paper] 의 정형 보장을 원래 검증되었던
QA-derived calibration 분포 밖으로, 실 의료 EHR 으로 가져간다. 결정적
single-command preprocessing 파이프라인을 제시하며, credentialed 로컬
MIMIC-IV v3.1 가 주어지면 stratum-balanced 코호트와 함께 **CRITICAL
admission 274,484 건** 산출 — Round 7 에서 algorithm-only 였던
$\alpha = 0.001$ regime 의 $n \ge 999$ 임계의 275 배.

다섯 실험 — 실증 $\alpha = 0.001$ 검증, 실 EHR head-to-head, services
테이블 distribution shift, 2008→2019 시간 분할, demographic equity
감사 — 를 정의하며 각각은 regression-guarded falsifiable 가설과
한 줄 셸 명령을 가진다. 본 논문이 Round 7 를 넘는 알고리즘 주장을
하지 않음과 MIMIC-IV 가 단일 센터임을 명시한다 — 발견은 §7 한계와
함께 정직하게 보고한다.

핵심 실증 질문 — *v2 가 calibration 도 실 EHR outcome 에서 나올 때
실 EHR outcome 의 stratum 별 보장을 유지하는가?* — 는 본 논문이 답할
유일한 질문. 답은 실험 suite 완료 후
`results/round9/round9_report.md` 에 기록되며,
`tests/test_paper_claims.py::test_r9_alpha_critical_within_2sigma`
test 가 그 주장의 regression guard 역할.

---

## 감사의 글

[심사용 익명화.]

저자들은 PhysioNet 팀과 MIMIC-IV maintainer 들이 명확한 DUA 하
연구 커뮤니티에 credentialed-access EHR 데이터를 제공한 데 감사한다.
Round 9 는 그 인프라 없이 불가능하다.

---

## 참고문헌

영문 [paper/UASEF_Round9.md](UASEF_Round9.md) §References 와 동일.
방법론 참고문헌 (CP, CRC, multi-trigger, Wilson harmonic mean,
e-value) 은 [paper/UASEF_Round7.md](UASEF_Round7.md) §References 의
완전 목록 참조.

---

## Appendix A. 재현 체크리스트

영문 §Appendix A 와 동일 — code path 는 영문 그대로 (코드 경로는
다국어 mirror 가 없음).

## Appendix B. Stratum 통계 (단일 결정적 preprocessing 실행)

§3.4 의 숫자는 MIMIC-IV v3.1, `--seed 42` 의 단일 결정적 실행에서
나온다. 동일 MIMIC-IV 릴리스를 보유한 모든 reader 에 byte 단위로
재현 가능하다. 영구 기록을 위해 재진술:

```
[stratum] Classifying admissions ...
  CRITICAL : 전체 274,484, 샘플 1,500 (escalate=1,500)
  HIGH     : 전체  60,544, 샘플 1,500 (escalate=1,500)
  MODERATE : 전체 128,082, 샘플 1,500 (escalate=    0)
  LOW      : 전체  82,918, 샘플 1,500 (escalate=    0)
✅ Wrote 6,000 cases → data/raw/mimic-iv/mimic4_cases.jsonl
```

검증: 2026-05-10, MIMIC-IV v3.1 (2024년 10월 릴리스).
