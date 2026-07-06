# 진행 중인 작업 상세 설명문 — Audited Conformal Escalation for Medical LLM QA

> 이 문서 하나로 프로젝트 전체(동기·수학·프로토콜·결과·통계·한계·현재상태)를 이해할 수
> 있도록 작성했다. 요약본이 아니라 기술 설명서다.
> 관련: 설계 [phase0_3_redesign.md](phase0_3_redesign.md) · 코어 검증
> [../models/label_conditional_conformal.py](../models/label_conditional_conformal.py) ·
> 재현 [../REPRODUCE.md](../REPRODUCE.md)

---

## 0. 한눈에

의료 LLM(gpt-oss-120b)이 답을 내기 **전에**, 그 답이 틀렸을 위험 `r(x,a)`를 계산해
**공개(release) vs 전문가 인계(escalate)** 를 결정하는 게이트를 만든다. 게이트는
`공개 ⟺ r(x,a) < τ` 형태이며, 임계 τ는 **틀린 답의 위험 분포**에서 conformal하게 뽑아
다음을 **유한표본·분포무관**으로 보장한다:

$$ \Pr(\text{공개} \mid \text{실제로 틀림}) \le \alpha. $$

즉 "틀린 답이 그대로 사용자에게 나가는 비율 ≤ α". 이것이 이 연구의 안전 계약이다.

**포지셔닝(고정)**: 이 연구는 **벤치마크 정의 오류(MCQ 정오답)와 objective-label 위험을
제어**하는 것이지, 실제 임상 위해를 보장하지 않는다. verifier LLM은 **위험 feature**로만
쓰이며 ground truth가 아니다. 임상 배포 전엔 전문의 판정이 있는 open-ended QA 검증이 필요.

---

## 1. 왜 이 문제이고, 왜 이렇게 접근하나

**문제.** LLM 의료 배포의 핵심 난점은 "언제 틀리는지 모른다"이다. 기존 세 접근의 한계:
- 단순 임계값: 임의적, 통계 보장 없음
- 항상 human-in-the-loop: 자율 처리 불가 → 비용·지연
- 확률 보정(temperature/Platt scaling): 분포 이동 시 붕괴

**접근.** Conformal prediction은 분포 가정 없이 교환가능성(exchangeability)만으로 유한표본
보장을 준다. 이 저장소의 선행 연구(Patterns 진단-프레임워크)에서 확립한 **엄밀성 규율**을
그대로 앞으로 적용한다: 검증된 코어, leakage 가드, 방향(orientation) 검증, 단계별 go/no-go,
적대적 검증. 목표는 "그럴듯한 성공"이 실은 아티팩트인지 매 단계 검문하며 **진짜 해법**을 짓는 것.

**전 단계 불변식(코드로 강제):**
1. **No definitional leakage** — gold 정답은 오류 **라벨** 계산에만 쓰고 feature엔 절대 안 넣음.
2. **Orientation** — 모든 위험 신호는 "높을수록 위험"으로 통일, 매번 `AUROC(risk,error)>0.5` 확인.
3. **Locked test** — 최종 주장은 잠긴 test 기준, calibration/test를 훔쳐보지 않음.
4. **Pre-committed go/no-go** — 각 단계 통과 기준을 미리 못박아 가짜 성공을 조기 차단.

---

## 2. 수학적 심장 — label-conditional (Mondrian) conformal

### 2.1 정의

위험 점수 `r(x,a)`(높을수록 오류 가능성). 게이트: `공개 ⟺ r < τ`, `에스컬레이션 ⟺ r ≥ τ`.
목표는 `Pr(r < τ | 오답) ≤ α`. τ를 **오답 케이스의 위험만** 모아 보정한다(그래서
label-conditional / Mondrian):

- calibration의 오답 집합 위험을 오름차순 정렬: `r_(1) ≤ … ≤ r_(n_err)`
- `k = ⌊α·(n_err+1)⌋`
- `τ = r_(k)` (k번째로 작은 오답 위험). 단 `k < 1`이면 `τ = −∞`(아무것도 공개 안 함 = 보수)

**보장(연속 점수 가정).** 새 오답 케이스가 교환가능하면
`Pr(r_new < τ | 오답) = k/(n_err+1) ≤ α`. 증명 개요: n_err+1개 교환가능 값 중 r_new의 순위는
균등분포 → `r_new < r_(k)`일 확률 = k/(n_err+1). 동점은 strict `<`로 보수 처리.

### 2.2 표준 CRC와 무엇이 다른가

기존 `conformal_escalation.py`(endpoint-CRC)는 **miss율**을 nested 임계 위에서 제어한다.
여기는 보장이 **오류 라벨에 조건부**(τ를 오답에만 보정)라, "틀린 답을 공개하지 마라"에
정확히 대응한다. 이 조건부성이 QA 안전의 요구와 맞물린다.

### 2.3 실현가능성(feasibility)과 명시적 INFEASIBLE

`k ≥ 1`은 `n_err ≥ ⌈(1−α)/α⌉` 필요 (α=0.10→9, α=0.05→19; 실무 권장 30/60). 미달이면
보장을 지키는 유일한 방법이 "아무것도 공개 안 함"(escalate-all)뿐 → 코어는 이를 **조용히
붕괴시키지 않고** `feasible=False, τ=−∞`로 **명시**한다(선행 연구의 sign-bug 붕괴를 교훈 삼음).

### 2.4 검증

몬테카를로 200시드([../tests/test_label_conditional_conformal.py](../tests/test_label_conditional_conformal.py),
**11/11 통과**): 경험적 `Pr(공개|오답)` = 0.0495 / 0.0985 / 0.1994 (α=0.05/0.10/0.20, 전부 ≤ α,
bound에 tight). 정보 없는 순수-노이즈 위험에서도 보장 성립(validity ≠ informativeness),
정보 있는 위험은 non-vacuous(공개율 0.22~0.48).

---

## 3. 데이터 & draft 생성 파이프라인

### 3.1 데이터셋 (플랜 요구 반영)

- **MedMCQA** 3000 (21과목 subject-stratified) — 4지선다 의학 시험문제
- **PubMedQA** 800 — yes/no/maybe 생의학 QA
- 합계 **3800 문항**. 기존 로더엔 MedMCQA가 없어 신규 추가([../data/qa_datasets.py](../data/qa_datasets.py)).

### 3.2 hidden draft 생성 ([../models/qa_drafts.py](../models/qa_drafts.py))

문항마다 gpt-oss-120b(로컬 LMStudio, $0)로:
- **decision draft** (temp 0): "Reasoning → Answer → Confidence(0-100)" 형식 요구 → 답·언어화
  confidence·reasoning·token logprobs 추출
- **온도 샘플 k=5** (temp 0.7): 자기일관성/엔트로피용
- **오류 라벨** = (decision 답 ≠ gold) → 0/1. gold는 라벨에만, feature엔 안 씀.
- 결과는 JSONL 캐시(재개 가능). 실제 3800 생성 완료, 품질 진단 결과 **decision·confidence
  99.8% 온전, 잘림 0건**.

---

## 4. 위험 feature (5개, 전부 "높을수록 위험") ([../models/qa_risk_features.py](../models/qa_risk_features.py))

| feature | 정의 | 직관 |
|---|---|---|
| `self_consistency_disagreement` | 1 − (최빈답 비율) over k=5 | 샘플이 갈릴수록 위험 |
| `answer_entropy` | 샘플 답분포의 정규화 Shannon 엔트로피 | 분산될수록 위험 |
| `neg_token_logprob_mean` | decision draft의 평균 NLL | 자기 답 생성이 불확실할수록 위험 |
| `verbalized_uncertainty` | 1 − (언어화 confidence/100) | 스스로 낮게 확신할수록 위험 |
| `hedging_rate` | reasoning 100단어당 헤지 표현 수 | "아마/불확실" 많을수록 위험 |

NaN(신호 없음)은 모델 학습 시 median 임퓨트. **verbalized_uncertainty가 단독 최강**(AUROC ~0.83).

---

## 5. Phase 0 — 게이트키퍼 ("신호가 존재하는가?")

**프로토colo**: draft → feature → **교차검증 AUROC(risk, error)** 만 측정(아직 conformal 아님).
LogReg/HistGBDT 5-fold OOF, pooled·데이터셋별·과목별·단변량.

**결과 (n=3800)**: pooled CV AUROC **0.841** (MedMCQA 0.813 / PubMedQA 0.730), 오답율 0.284
(오답 1079개), 22과목 median 0.80.

**go/no-go**: ≥0.70 → GO. → **GO**. (조건부 0.60–0.70 / no-go <0.60은 미발동.)
동시에 verbalized confidence의 강함(0.83)이 0.85 self-deception 경계에 근접 → Phase 2 감사 예약.

---

## 6. Phase 1 — Stage-A MVP (conformal 게이트)

**프로토colo** ([../experiments/phase1_stage_a.py](../experiments/phase1_stage_a.py)):
- 문항 단위 **subject-stratified 분할 train 40 / cal 30 / test 30** (같은 subject를 같은 비율로)
- train으로 위험 스코어러(HistGBDT) 학습 → cal·test 위험
- cal의 오답 위험으로 **label-conditional conformal 보정** → τ
- **잠긴 test**에서 지표 산출 + 동일 incorrect-release 목표로 보정한 **baseline** 대조

**결과 (n=3800; train 1509 / cal 1140 / test 1151)**:
- 위험 스코어러 test AUROC 0.822
- **conformal 게이트 (핵심)**: α=0.10 → `Pr(공개|오답)` **0.077** (≤0.10 ✅, 공개율 0.379) ·
  α=0.05 → **0.044** (≤0.05 ✅, 공개율 0.272) → **단일 잠긴 test에서 보장 성립**
- baseline: 자기신뢰 임계(B2)는 유사 공개율이나 **보장 없음**; 자기일관성 단독(B3)은
  escalate-all 붕괴; release-all(B0)은 위반(1.0)
- 판정 **OVER_ESCALATION_AUDIT**: 강한 모델 + 엄격 α에서 공개 38% → 자율공개율 중간.
  실패가 아니라 정직한 "과에스컬레이션 필요" 결말(플랜의 분기)

**소표본 주의(작은 표본 파일럿 n=279 때)**: 단일-split은 노이즈로 α를 살짝 넘길 수 있었으나
200-split 몬테카를로 평균 `Pr(공개|오답)`=0.085/0.033 ≤ α로 marginal 보장 확인. → full 3800에서
**단일 잠긴 test로도 안정**됨(위 결과).

---

## 7. Phase 2 — 감사 (진행 중)

### 7.1 다섯 진단기 이식 ([../experiments/phase2_qa_audit.py](../experiments/phase2_qa_audit.py))

Patterns 연구의 5 detector를 QA 위험 점수에 이식:

| 진단기 | 통계량 | 결과 (n=3800) |
|---|---|---|
| orientation | AUROC(risk,error)>0.5 | clean (0.842) |
| escalate-all | 게이트 에스컬레이션율 | clean (0.572, vacuous 아님) |
| definitional-leakage | 최대 단변량 AUROC≥0.90 | clean (0.828 < 0.90) |
| **confidence-dominance** | confidence-only ÷ full above-chance | **FLAGGED (recover 0.963)** |

**confidence-dominance**는 informative-missingness의 QA판이다: confidence-only AUROC 0.829가
full 0.842의 96%를 복원, confidence 제거 시 0.694로 급락 → **게이트가 모델 자기신뢰에 지배됨**.
= 정직한 약점 진단.

### 7.2 교차-모델 verifier (진행 중, full 3800 확장) ([../experiments/phase2_cross_verifier.py](../experiments/phase2_cross_verifier.py))

**동기**: "에스컬레이션 신호가 답변 모델 자신에게서 나온다"는 우려 해소. **다른 모델
gemma-4-31b**가 gpt-oss의 답을 (gold 없이) 판정 → 독립 위험 점수(risk feature, ground truth 아님).

**중간 결과(within-dataset)**: **독립 verifier > 자기신뢰**
- MedMCQA 0.88 vs 0.80, PubMedQA 0.82 vs 0.72 (gap +0.07~0.10)
- 부트스트랩 2000회 95% CI가 0을 배제, **P(gap>0)=100%** → 뒤집힐 위험 없음

**함의**: 독립 모델이 더/동등하게 예측 → 에스컬레이션을 답변 모델과 **탈결합** 가능.
결합(verifier+confidence)이 verifier 단독을 못 넘음 → 자기신뢰는 독립 판정 앞에서 잉여.
**3800으로 확장 중**(~18h): (a) gpt-oss와 대칭, (b) verifier를 **Phase 1 게이트의 feature로
승격**해 "독립 verifier + 보장"을 감사가 아닌 **해법**으로 통합.

### 7.3 dual-shuffle — self-deception 판별 (준비 완료, 허락 대기)

**질문**: gpt-oss confidence(와 gemma 판정)가 **내용 기반**인가 **벤치마크 암기**인가?
**방법**: MCQ 선택지 위치를 섞어 다시 풀리고/판정시켜, 원본 대비 AUROC 유지 여부를 본다.
**두 모델 모두 대칭 감사**:

| 모델 | 셔플 유지 | 셔플 붕괴 |
|---|---|---|
| gpt-oss 자기신뢰 | 내용 기반(진짜 calibration) | 위치 암기 = self-deception |
| **gemma verifier** | 진짜 검증 | verifier도 답안지 암기(오염) |

- gpt-oss 셔플: [../experiments/phase2_shuffle_audit.py](../experiments/phase2_shuffle_audit.py)
  (섞인 선택지·질문·답을 저장해 다음 단계로 넘김)
- gemma 셔플-verify: [../experiments/phase2_shuffle_verify.py](../experiments/phase2_shuffle_verify.py)
  (gemma가 그 셔플된 답을 판정 → 셔플 vs 원본 AUROC 비교)

**추가 신호**: gpt-oss **정확도**가 셔플에서 떨어지면 = 답을 위치로 암기(답-레벨 벤치마크 오염)로,
confidence가 정직해도 별개의 발견이 된다.

---

## 8. 통계적 엄밀성 (직접 잡은 함정)

- **Simpson/pooling 인플레**: pooled AUROC(0.89)가 두 데이터셋 각각(0.88/0.83)보다 컸다. 원인은
  verifier가 PubMedQA(오답 많음)에 +0.25 높은 위험을 매겨 "데이터셋 맞히기"가 pooled에 공짜
  boost를 준 것. → **within-dataset으로 보고**, pooled 인플레를 명시. 비교(verifier vs
  confidence)는 within-dataset에서 +0.07~0.10로 견고.
- **gap 추세 우려**: gap이 전반부 +0.073 → 후반부 +0.037로 좁아졌으나(원인=후반부 confidence
  개선), 부트스트랩 P(gap>0)=100%로 0을 안 넘음. → 작아지지만 견고한 양수로 수렴.

---

## 9. 모델 동작 상세 (왜 gemma≠gpt-oss)

두 모델 모두 "thinking" 계열이지만 소비 토큰이 다르다:
- **gpt-oss-120b(답변자)**: 이 MCQ/yes-no에 생각을 짧게 → 512토큰 안에 생각+답+confidence 완결
  (잘림 0건, confidence 99.8% 온전) → 512로 충분.
- **gemma-4-31b(판정자)**: "이 답이 틀렸나?" 판정에 생각을 길게(~1350토큰) → 512에선 생각 도중
  잘려 **빈 응답(finish_reason=length) 43%** 발생. → **토큰 4096**으로 수정하니 실패 0%.
- 실무적으로 판정 태스크가 더 깊은 숙고를 유발함을 시사. 그래서 verifier만 4096 필요.

**메모리/스왑**: gpt-oss-120b(≈65GB급)와 gemma-31b(20GB)를 동시에 올리면 OOM. 그래서 모델
스왑(내리고 올리기)은 **운영자가 수동**으로 하고, 항상 한 모델만 로드한다(`lms` CLI).

---

## 10. 정직성 장치 — 지금까지 잡아낸/방어한 이슈

| 이슈 | 대응 |
|---|---|
| 같은 모델로 생성+에스컬레이션? | 독립 verifier(gemma) — 오히려 더 나음 실증 |
| pooled AUROC가 부분집합보다 큼 | within-dataset 보고 + 인플레 명시 |
| gemma 빈 응답 43% | thinking 512 초과 → 토큰 4096 수정(실패 0%) |
| gpt-oss도 잘렸을 수 있잖아 | 진단: confidence 99.8% 온전, 잘림-전 0건 → 재생성 불필요 |
| verifier(gemma)도 셔플해야 | dual-shuffle 대칭 감사 설계 |
| 120b+31b 동시 로드 OOM | 수동 스왑, 항상 한 모델만 |
| verifier 표본 1500 vs 3800 비대칭 | 3800 확장(대칭 + Phase 1 통합) |
| conformal 소표본 단일-split 노이즈 | 200-split 몬테카를로 + full 3800로 안정 |

---

## 11. 결과 분기 & 출판 프레이밍 (negative-only가 아니게 설계)

핵심은 **conformal 보장**(백본, 셔플과 무관하게 성립)이 항상 살아 있고, dual-shuffle 결과가
"어떤 논문인가"를 정한다:

| dual-shuffle 결말 | 서사 | 강도 |
|---|---|---|
| gpt-oss 붕괴 + gemma 유지 | "자기신뢰는 가짜(암기), 독립 verifier는 진짜 → verifier 써라" = **폭로+해법** | ⭐⭐⭐ |
| 둘 다 붕괴 | "둘 다 벤치마크 암기 = 오염 폭로" | ⭐⭐ |
| 둘 다 유지 + 정확도 유지 | "신호가 진짜 → conformal이 보장을 씌운 검증형 방법론" | ⭐ |
| 둘 다 유지 + 정확도 붕괴 | "답-레벨 벤치마크 오염 발견" | ⭐⭐ |

어느 결말이든 (a) conformal 보장 방법론 + (b) 독립-verifier 탈결합 + (c) 정직한 감사가 남아
**솔루션 논문**이 된다. 실제로는 완전 붕괴/유지가 아닌 **부분(스펙트럼)** 일 가능성이 높고,
그건 "대체로 내용 기반, 일부 위치 의존"으로 정량 보고한다.

---

## 12. 한계 (정직하게)

1. **벤치마크 오류 ≠ 임상 위해** — MCQ 정오답을 제어할 뿐, 실제 안전 아님.
2. **단일 답변 모델(gpt-oss-120b)** — 일반화엔 둘째 모델 필요.
3. **verifier 의존 전가** — 자기신뢰를 빼면 게이트가 단일 verifier(gemma) 품질에 의존.
   그래서 "보장이 중심, verifier는 교체 가능한 feature"로 프레이밍이 가장 강함.
4. **실용성** — 강한 모델 + 엄격 α에서 공개 38%(=62% 에스컬레이션). "safety-first,
   benchmark-defined"로 프레이밍(배포 준비 아님).
5. verifier도 LLM이라 자체 편향 있음(dual-shuffle로 오염은 검문하나 편향 전체는 아님).

---

## 13. 인프라 & 재현

- **로컬 LMStudio, 외부 API $0, PHI egress 0** — 저장소 원칙 유지
- 모든 LLM 생성은 **재개 가능**(JSONL 캐시); 세션/잡이 죽어도 같은 명령으로 이어짐
- 검증 코어 + 진단기: 순수 함수 + 단위 테스트 **21/21**, 데이터 없이 실행/검증 가능
- 산출물: `results/phase0|1|2/`; 원시 draft·verifier·shuffle 캐시는 `.gitignore`
- 모델 스왑: `~/.lmstudio/bin/lms unload --all` → `lms load <model>` (수동)

---

## 14. 현재 상태 & 다음

```
[완료] Phase 0 GO(0.841) → Phase 1 보장 성립(0.077/0.044≤α) → Phase 2 confidence-dominance 적발
[진행] 교차-verifier full 3800 (~18h, 재개가능) — 독립 verifier > 자기신뢰 견고(within-dataset)
[대기] 완료 → 보고 + 허락 → (수동 gpt-oss 스왑) 옵션-셔플 → (수동 gemma 스왑) 셔플-verify
       → dual-shuffle 결론 + verifier를 Phase 1 feature로 통합 → Phase 2 마무리 → Phase 3(Stage-B)
```

**Phase 3(예정)**: MIMIC/eICU objective-label을 leakage-safe decision-time feature로 만들어,
answer-reliability(Stage A)와 objective-label의 **정보 이론 대조**(I/H) + agreement(κ) 분석.
성능 주장이 아니라 두 도메인의 근본 차이를 정보이론으로 규명(선행 §7과 연결).

**한 줄**: 방법(conformal 보장)은 검증됐고, 약점(자기신뢰 의존)은 스스로 드러냈으며, 그
약점을 독립 verifier로 대체 가능한지 + 신호들이 진짜인지(암기 아닌지)를 지금 대칭적으로
검증 중이다. 어떤 결과가 나와도 정직한 솔루션 논문이 되도록 설계돼 있다.
