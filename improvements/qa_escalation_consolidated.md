# 통합 문서 — Cross-Model Escalation 신호의 정보이론적 해부
### 현재 연구 프레임: 완료 실험 + 계획 (단일 참조)

> 작성 2026-07-09. 관련: 계획 상세 `qa_escalation_master.md`, 결과 원장 `qa_escalation_results.md`.
> 이 문서는 **현재 프레임에 속하는 실험만** 정리 (초기 "안전 게이트" 프레임에서 재편 후).

---

## 0. 연구 프레임 (thesis)

**질문**: 한 LLM의 답을, 다른 LLM(cross-model verifier)이나 그 자신의 self-confidence로 "틀릴 위험"을
매기는 escalation 신호가 **실제로 무엇을 재는가?**

**답 (본 연구의 주장)**:
1. cross-model verifier가 self-confidence보다 오류를 잘 예측하지만, 그 우위는 **독립성이 아니라
   verifier와 answerer의 능력격차 Δ의 함수**다.
2. 신호는 사실상 **"모델 간 불일치 탐지기"** — 정보이론으로 정량화(상호정보·조건부 상호정보·κ).
3. 그리고 그 능력은 **verifier가 실제로 추론(thinking)할 때만** 작동한다 (추론 ablation).
4. 분포무관 conformal 게이트 `Pr(공개|틀림)≤α`는 **불완전·능력얽힘 신호를 안전하게 운용하는
   응용 매개**로 제시.

**포지셔닝 (임상 축소)**: 임상 안전 논문 **아님**. 벤치마크 정의 오류(MCQ 정오답)만 제어. 의료
QA(MedMCQA·PubMedQA)는 신뢰도 질문이 자연히 제기되는 **고부담 testbed**. verifier LLM은 ground
truth 아닌 **위험 feature**. 임상 적용은 future work(전문의 판정 open-ended 검증 선행).

**어떻게 여기 왔나 (연구 진화)**: ① 처음엔 conformal escalation 게이트(보장) → ② 자기감사에서
"신호가 자기신뢰에 96% 의존" 발견 → ③ 독립 verifier 도입(자기신뢰 이김) → ④ 분해에서 "능력 지배"
발견(독립성 아님) → ⑤ 2모델서 재현 → ⑥ 답변자·verifier 능력 다양화 + 추론 on/off로 법칙·ablation →
⑦ insight 논문으로 재편, 임상 축소.

---

## 1. 조직 구조 — 능력격차 그리드

핵심 실험 구조는 **(answerer × verifier) 매트릭스**. 각 셀에서:
- **Δ = acc(verifier) − acc(answerer)** (능력격차)
- **lift = AUROC(verifier_risk) − AUROC(answerer self-confidence)** (verifier 우위)
- 목표: **lift ≈ f(Δ)** 법칙 (A9)

**모델 능력 (자문자답 정확도, 공통 1500)**:
| 모델 | 역할 | acc |
|---|---|---|
| gpt-oss-120b | 답변자#1 / verifier | 0.712 |
| gemma-4-31b | verifier | 0.816 |
| qwen3.6-27b | verifier | 0.805 |
| Qwen3.5-122B (no-think) | 답변자#2 / verifier | 0.795 (mc 0.862 / pm 0.543) |
| Qwen3.5-122B (think) | 답변자#2 | ~0.845 (mc 0.928 / pm 0.567) ※진행중 |

---

## 2. 완료된 실험 (결과)

| ID | 실험 | 무엇을 보이나 | 핵심 결과 | 상태 |
|---|---|---|---|---|
| P0 | 게이트키퍼 | risk가 오류 예측하나 | CV AUROC **0.841** → GO | ✅ |
| P1 | Stage-A conformal 보장 | 분포무관 보장 성립 | P(공개\|오답) **0.077/0.044 ≤ α** (잠긴 test) | ✅ |
| P2 | 자기감사 (detector) | 게이트가 뭘 쓰나 | **confidence-dominance**: conf-only가 신호 **96.3%** 복원 | ✅ |
| A0 | 교차 verifier (gemma·qwen3.6) | 독립 모델이 자기신뢰 이기나 | 이김 (아래 표) | ✅ |
| B1/B1' | 능력 vs 독립성 분해 | 우위가 독립성이냐 능력이냐 | **능력 지배**, 2모델 재현 (아래) | ✅ |
| A6 | verifier-augmented 게이트 | 실용 이득 | 공개율 **+7%p**, α 유지 | ✅ |
| A8 | 정보이론 해부 | 증분정보·불일치 정량화 | I(V;Y\|C)≈0.15b, P(오류\|불일치) 0.84 | ✅ 2셀 |
| B-1b | Qwen3.5 no-think verifier | 추론 없으면? | **self-conf보다 못함**(−lift) | ✅ |

### 2.1 A0 — 교차 verifier (within-dataset, Simpson 회피)

| 데이터셋 | gemma vs 자기신뢰 | 95% CI | qwen3.6 vs 자기신뢰 | 95% CI |
|---|---|---|---|---|
| MedMCQA | 0.877 vs 0.806 (**+0.071**) | [+.037,+.106] 견고 | 0.850 vs 0.800 (**+0.050**) | [+.013,+.086] 견고 |
| PubMedQA | 0.809 vs 0.774 (+0.035) | [−.006,+.077] 경계 | 0.814 vs 0.774 (+0.040) | [−.001,+.081] 경계 |

→ 두 계열 verifier 모두 MedMCQA에서 자기신뢰를 **견고히** 이김 (verifier 교체 가능). PubMedQA는 경계.

### 2.2 B1/B1' — 능력 vs 독립성 분해 (핵심)

verifier가 같은 문항을 스스로 답한 뒤, verifier가 *틀린* 문항에서도 gpt-oss 오류를 예측하나?

| subset | gemma AUROC | qwen3.6 AUROC |
|---|---|---|
| 전체 | 0.882 | 0.878 |
| verifier **맞힌** 문항 | **0.982** | **0.981** |
| verifier **틀린** 문항 | **0.612** | **0.593** |

→ **능력 지배**: verifier가 답을 알 때만 오류를 잘 잡고, 모를 때 붕괴. **두 모델 거의 판박이 재현**
= 특정 모델 quirk 아님. 오답상관 φ≈0.53. **cross-verifier ≈ 모델 간 불일치 탐지기**.
(주의: 두 verifier 모두 답변자보다 강해 "약한 verifier" 검정은 구조적으로 불가.)

### 2.3 A6 — verifier-augmented 게이트 (실용성)

- risk AUROC(test): 5-feature **0.824** → +verifier **0.895**
- 공개율: α=.10 0.362→**0.432(+.070)**, α=.05 0.293→**0.354(+.061)**, **양쪽 α 충족**(안전 유지)
- → "게이트가 62% 에스컬레이션(급소)"을 **결과로** 전환.

### 2.4 A8 — 정보이론 해부 (재편 코어, bits, gpt-oss 답변자 row)

| 양(bits) | gpt-oss×gemma | gpt-oss×qwen3.6 |
|---|---|---|
| H(Y) 오류 불확실성 | 0.861 | 0.862 |
| I(verifier;Y) | 0.348 | 0.333 |
| I(self-conf;Y) | 0.222 | 0.224 |
| **I(verifier;Y \| self-conf)** 증분 | **0.161 (46%)** | **0.149 (45%)** |
| P(오류 \| 두 모델 일치) | 0.11 | 0.12 |
| **P(오류 \| 불일치)** | **0.84** | 0.81 |
| κ(verifier답, 답변자답) | 0.71 | 0.71 |
| 증분정보: verifier-맞힘 / 틀림 | 0.446 / **0.027** | 0.462 / **0.008** |

→ verifier는 self-conf **너머 ~0.15b(≈45%) 증분 정보**를 주지만, 그 증분이 **verifier-틀림에선 ≈0**
(능력 게이팅의 bits판). **"불일치→오류 84% vs 일치 11%"**가 핵심 그림.

### 2.5 B-1b — 추론 ablation의 첫 증거

Qwen3.5를 **no-think**로 gpt-oss 답 판정 → lift **−0.031(mc) / −0.084(pm)** = **자기신뢰보다 못함**.
(thinking verifier gemma/qwen3.6는 +0.04~+0.07). → **"강한 모델도 추론 없이는 verifier로서 무력"** →
verifier 가치는 **추론**에서 옴. 4모델로 확장 예정(§4).

---

## 3. 진행 중

- **Qwen3.5 thinking drafts** (`drafts_qwen35_think.jsonl`): 답변자#2를 native thinking으로 재생성
  (16000토큰, 절단 0). 잠정 acc **~0.84** (within-dataset think>no-think: mc 0.90>0.86, pm 0.59>0.54).
  ~768/1500 진행. no-think판(`drafts_qwen35_nothink.jsonl`, acc 0.795)은 ablation으로 보존.
- **주기적 자동 리로드**: 100문항마다 같은 모델 unload+load(속도 리셋, 메모리 안전). 두 스크립트 모두 적용.

---

## 4. 계획 — 완전 추론×능력 그리드

### 4.1 완전 대칭 ablation 그리드
답변자: **A1=gpt-oss(0.712)**, **A2=Qwen3.5-think(~0.845)**. verifier 4종 × think/no-think.
(같은 모델 self-verification은 self-conf가 대각선이라 제외)

| verifier×mode | A1 판정 | A2 판정 |
|---|---|---|
| gemma-T | ✅A0 | 🆕 |
| gemma-N | 🆕 | 🆕 |
| qwen3.6-T | ✅ | 🆕 |
| qwen3.6-N | 🆕 | 🆕 |
| gpt-oss-T | (self) | 🆕 |
| gpt-oss-N | (self) | 🆕 |
| Qwen3.5-T | 🆕 | (self) |
| Qwen3.5-N | ✅B-1b | (self) |
→ 신규 9 판정 패스.

### 4.2 dual-shuffle (완전 대칭, think+no-think, ~400 MedMCQA)
암기/위치편향 검문 + no-think 붕괴 메커니즘. **의존성 규칙: 답변자 셔플 재답변이 먼저 → verifier 판정.**
셔플 items = **공통 1500의 MedMCQA 부분집합**(원본 baseline이 이미 존재하도록).
| 종류 | 패스 · 규모 |
|---|---|
| 답변자 셔플 재답변 (2) | gpt-oss, Qwen3.5-think — 각 ~400 |
| gpt-oss 셔플답 판정 (6) | gemma-T/N, qwen3.6-T/N, Qwen3.5-T/N — **각 셔플 400만** |
| Qwen3.5 셔플답 판정 (6) | gemma-T/N, qwen3.6-T/N, gpt-oss-T/N — **각 셔플 400만** |
→ **14 패스** (셔플 400/셀). **원본은 전체 재판정 X** — 매트릭스 판정(risk) + 기존 self-answer(불일치-text)에서
조인(item_id paired). 원본 vs 셔플 AUROC 비교로 강건성.
**모든 셀 원본 baseline은 존재/생성 확정**: A1(gpt-oss 답)의 gemma-N·qwen3.6-N은 매트릭스 A1 패스로 생성,
Qwen3.5-N=B-1b, gemma-T·qwen3.6-T=A0; A2(Qwen3.5 답) 6셀은 매트릭스 A2 패스. self-answer는 gemma/qwen3.6/
gpt-oss(drafts)/Qwen3.5(drafts) 전부 1500 존재. → **셔플 때문에 추가 원본 판정 불필요.**

**prompt-confound sanity check (모든 셀, 100개만)**: 각 셀 원본 100개를 shuffle-judge 프롬프트로 재판정 →
기존 원본과 비교. **통과 기준**: answer agreement(text) ≥90–95% · risk Spearman ≥0.80 · AUROC 차 ≤0.03–0.05 ·
lift 부호 불변 · P(오류|불일치) 해석 유지. **실패 셀만 원본 400 전체 재판정**(트리거: AUROC ≥0.07·lift flip·
no-think 불안정). no-think 셀은 100 보수적. 비용: 12셀×100=1,200 (전체 4,800 대신).
스크립트: `phase2_shuffle_answer.py`(원시 12필드·text 채점·결정적 순열·공통셋 추출) + `phase2_shuffle_judge.py`
(verifier에 `letter) text`·**agreement_by_text**·불일치-text) + [만들 것] 원본-reference 생성(LLM0) + sanity 비교(LLM0).

### 4.3 스왑 순서 (9 세션, 수정판 — 의존성 준수, 리로드 활성)
**Phase 1 — 두 답변자 셔플 재답변 먼저** (그래야 verifier가 판정 가능):
```
1. Qwen3.5-T (현재): matrix Qwen3.5-T→A1  +  Qwen3.5 셔플 재답변
2. gpt-oss-T       : matrix gpt-oss-T→A2  +  gpt-oss 셔플 재답변  +  gpt-oss-T→Qwen3.5셔플(준비됨)
   → 이제 gpt-oss셔플·Qwen3.5셔플 둘 다 준비
```
**Phase 2 — 나머지 verifier가 두 셔플셋 판정** (의존성 없음, 로드당 매트릭스+셔플 piggyback):
```
3. gpt-oss-N : A2               + gpt-oss-N→Qwen3.5셔플
4. gemma-T   : A2               + gemma-T→gpt-oss셔플, Qwen3.5셔플
5. gemma-N   : A1,A2            + gemma-N→gpt-oss셔플, Qwen3.5셔플
6. qwen3.6-T : A2               + qwen3.6-T→gpt-oss셔플, Qwen3.5셔플
7. qwen3.6-N : A1,A2            + qwen3.6-N→gpt-oss셔플, Qwen3.5셔플
8. Qwen3.5-T (재방문): Qwen3.5-T→gpt-oss셔플 (이제 준비됨)   ← 상호의존이라 불가피한 1회 추가
9. Qwen3.5-N : Qwen3.5-N→gpt-oss셔플
```
**gpt-oss no-think 답변자 row = skip** (Qwen3.5 데이터셋별 편차로 Δ 이미 광범위).

### 4.4 남은 분석 (LLM 0)
- **A9 능력격차 법칙**: (Δ, lift) 셀 → **within-dataset 필수**(pooling 시 가짜 음의 추세) 산점도+회귀.
  Qwen3.5 편차 커 Δ ∈ 약 [−0.21,+0.23]. ★ **주의(현 데이터)**: 양의 Δ에서 lift가 평평 경향 → **Δ<0 앵커
  (Qwen3.5-강×gpt-oss)에서 lift 죽는지가 법칙의 생사.**
- **추론 ablation 표 (더 강한 카드)**: 4모델 × think/no-think verifier lift. 현 데이터상 think(+0.04~0.07)
  vs no-think(−0.04~−0.08) 간극 ~0.1로 A9의 Δ-기울기보다 크고 깨끗 → **주 헤드라인 후보**.
- **A8 전셀 정보이론**: I(V;Y|C)·κ·P(오류|불일치) 전 셀.
- **dual-shuffle 분석**: 원본 vs 셔플 AUROC(강건성) + **no-think 붕괴 2×2**(추론 없으면 암기라서 셔플에 무너짐).

### 4.5 비용
LLM 패스 **23개** (매트릭스 9×1500 + 셔플 14×400), **9 로드 세션** (4모델×think/no-think + Qwen3.5-T 재방문).
셔플 원본은 재판정 없이 기존 재활용(400/셀). ≈ **~3일**, 컴퓨트 $0 로컬, 전부 재개가능(리로드로 속도 유지).

---

## 5. 모델 · 환경 (Methods)

**HW**: Apple M3 Ultra 96GB, LMStudio (llama.cpp GGUF + MLX), 한 번에 하나 로드(수동 스왑), $0, PHI egress 0.

| 역할 | 모델 | 포맷 | 양자화 | acc |
|---|---|---|---|---|
| 답변자#1/verifier | gpt-oss-120b | GGUF | MXFP4(네이티브) | 0.712 |
| verifier | gemma-4-31b-it | GGUF | Q4_K_M | 0.816 |
| verifier | qwen3.6-27b | MLX | 4-bit affine gs64 | 0.805 |
| 답변자#2/verifier | Qwen3.5-122B-A10B | GGUF | Q3_K_S(공격적) | 0.795(N)/0.845(T) |

**주의(한계에 기재)**: ① 양자화·런타임 혼용(수치 미세차). ② Qwen3.5 Q3_K_S는 품질 하한. ③ **thinking
토글**: Qwen3.5는 Jinja 템플릿 `enable_thinking`으로 on/off — **think/no-think를 능력·ablation 변수로 활용**.
④ verifier max_completion_tokens: thinking 4096~16000(절단 방지), no-think 512.

---

## 6. 논문 척추 (재편 확정)

1. **[주 발견]** escalation 우위 ≈ **능력격차 Δ의 함수** (능력지배 B1/B1', 두 계열 재현; A9 법칙).
2. **[정보이론 해부]** self-conf 너머 증분정보 I(V;Y|C) + "불일치→오류" 정량화 (A8); informative-missingness 연결.
3. **[추론이 원천]** 추론 끄면 verifier가 self-conf보다 못함 (B-1b→4모델 ablation).
4. **[응용 매개]** 불완전 신호도 분포무관 conformal로 `Pr(공개|틀림)≤α` 운용 + 공개율 +7%p (P1·A6).

**한계·범위**: 벤치마크 오류 ≠ 임상 위해(scope). MCQ 2벤치마크·답변자 2개(1500 subset). verifier=feature.

## 7. 산출물

- 결과: `results/phase2/phase2_{qa_audit, cross_verifier, decomp_gemma, decomp_qwen27,
  verifier_augmented, infotheory_gptoss__gemma, infotheory_gptoss__qwen27}.json`
- 데이터: `data/raw/{verifier_cross, verifier_qwen27, verifier_qwen35_of_gptoss, selfanswer_gemma,
  selfanswer_qwen27, drafts_qwen35_nothink, drafts_qwen35_think, drafts_phase0_all}.jsonl`
- 코드: `models/label_conditional_conformal.py`(21 test) · `qa_risk_features.py` · `qa_drafts.py`;
  `experiments/phase{0_gatekeeper,1_stage_a,2_qa_audit,2_cross_verifier,2_verifier_augmented,
  2_selfanswer,2_infotheory,2_answerer_drafts}.py` (리로드 기능 포함)
