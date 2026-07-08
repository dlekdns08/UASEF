# 마스터 문서 — What Does a Cross-Model Escalation Signal Measure?
### 능력격차의 함수로서의 cross-model escalation 신호: 정보이론적 해부 (분포무관 release 게이트를 응용 매개로)

> 이 프로젝트의 **단일 완전 참조 문서**. 동기·수학·파이프라인·완료결과·남은 전체 로직·
> 모델·비용·출판 프레이밍·한계·재현까지 전부.
> **2026-07-08 재편**: "안전 게이트 논문" → **"escalation 신호의 정보이론적 해부(insight) 논문"**.
> 임상 주장 축소, conformal 게이트는 **헤드라인이 아닌 응용 매개**로 강등.

---

# PART I. 무엇을 / 왜 (재편)

## 1. 한 문장 (thesis)

한 LLM의 답을, **다른** LLM(cross-model verifier)이나 그 자신의 self-confidence로 "틀릴 위험"을
매기는 escalation 신호가 **실제로 무엇을 재는지** 묻는다. 우리는 그 신호의 오류-예측 우위가
**독립성이 아니라 대체로 verifier와 answerer의 능력격차 Δ의 함수**임을 실증하고, 이를
**정보이론적으로 해부**한다(상호정보 I·조건부 상호정보 I(·;error|conf)·agreement κ). 분포무관
label-conditional conformal 게이트 `Pr(공개|틀림)≤α`는 **불완전·능력얽힘 신호를 그럼에도 안전하게
운용**하는 방법을 보이는 **응용 매개**로 제시한다.

**핵심 주장(재편):**
- **Q**: cross-model verifier가 self-confidence보다 오류를 잘 예측하는 이유는 독립성인가 능력인가?
- **A**: 대체로 **능력** — verifier가 *스스로 맞힌* 문항에서만 예측력이 살고(0.98), *틀린* 문항에선
  붕괴(0.61). 두 verifier 모두 answerer보다 강해(Δ>0) 우위가 Δ와 얽힘. → **신호 ≈ 강한 second-opinion
  (모델 불일치), 순수 독립 오류탐지 아님**.
- **정보이론**: verifier가 self-confidence *너머* 더하는 정보 I(verifier;error|conf)를 정량화하고
  Δ와의 관계(law: lift≈f(Δ))를 그린다. (선행 informative-missingness 연구의 I/H 프레임 연결.)
- **매개**: 그런 불완전 신호도 conformal로 감싸면 `Pr(공개|틀림)≤α` 유한표본 보장 → **운용 가능**.

## 2. 포지셔닝 (재편 — 임상 축소)

- **이 논문은 임상 안전 논문이 아니다.** **벤치마크 정의 오류(MCQ 정오답)**만 제어하며 실제 임상
  위해와의 연결은 **주장하지 않는다**. 의료 QA(MedMCQA·PubMedQA)는 "신뢰도 질문이 자연히 제기되는
  **고부담 testbed**"로 쓰는 것이지 배포 준비 주장이 아니다.
- 표기도 축소: "medical LLM **safety**" → "**QA reliability** under a distribution-free release gate".
- verifier LLM은 **위험 feature**일 뿐 ground truth가 아니다(본문 결과가 그 근거를 제공).
- 임상 적용은 **범위 밖(future work)**: 전문의 판정 open-ended QA 검증이 선행되어야 함을 명시.

## 3. 전 단계 불변식 (코드로 강제)

1. **No definitional leakage** — gold는 오류 **라벨**에만, feature엔 절대 안 넣음.
2. **Orientation** — 모든 위험 신호 "높을수록 위험" 통일, 매번 `AUROC(risk,error)>0.5` 확인.
3. **Locked test** — 최종 주장은 잠긴 test 기준.
4. **Pre-committed go/no-go** — 단계 통과 기준 사전 못박아 가짜 성공 조기 차단.

---

# PART II. 수학적 심장 — label-conditional (Mondrian) conformal

## 4. 정의

위험 `r(x,a)`(높을수록 오류). 게이트: `공개 ⟺ r < τ`. τ를 **오답 케이스 위험만** 보정:
- 오답 위험 오름차순 `r_(1) ≤ … ≤ r_(n_err)`, `k = ⌊α·(n_err+1)⌋`, `τ = r_(k)`
- `k < 1`이면 `τ = −∞` (아무것도 공개 안 함 = 보수)

**보장**: 새 오답 케이스가 교환가능 → `Pr(r_new < τ | 오답) = k/(n_err+1) ≤ α`.

## 5. 표준 CRC와 차이

endpoint-CRC(`conformal_escalation.py`)는 miss율을 nested 임계로 제어. 여기는 보장이 **오류
라벨에 조건부**(τ를 오답에만 보정) → "틀린 답 공개 금지"에 정확히 대응.

## 6. Feasibility & INFEASIBLE

`k≥1`은 `n_err ≥ ⌈(1−α)/α⌉` (α=0.10→9, 0.05→19; 권장 30/60). 미달이면 유일 해법이
escalate-all뿐 → **조용히 붕괴 안 시키고** `feasible=False, τ=−∞`로 명시.

## 7. 검증

몬테카를로 200시드(`tests/test_label_conditional_conformal.py`, **11/11**): 경험적
`Pr(공개|오답)` = 0.0495/0.0985/0.1994 (α=0.05/0.10/0.20, 전부 ≤ α). 순수-노이즈 위험에서도
성립(validity ≠ informativeness), 정보 있는 위험은 non-vacuous(공개율 0.22~0.48).

---

# PART III. 데이터 · feature · 파이프라인

## 8. 데이터 (3800)

- **MedMCQA** 3000 (21과목 subject-stratified, 4지선다)
- **PubMedQA** 800 (yes/no/maybe)
- 로더 `data/qa_datasets.py` (MedMCQA 신규 추가).

## 9. draft 생성 (`models/qa_drafts.py`)

문항마다 답변 모델로: decision draft(temp 0, "Reasoning→Answer→Confidence" 형식 → 답·언어화
confidence·reasoning·logprobs) + 온도샘플 k=5(temp 0.7). 오류라벨=(답≠gold). JSONL 캐시(재개).
`--max-tokens`(gpt-oss 512, thinking 모델 4096). gpt-oss 3800 생성 완료·품질 99.8% 온전.

## 10. 위험 feature 5종 (`models/qa_risk_features.py`, 전부 높을수록 위험)

| feature | 정의 |
|---|---|
| self_consistency_disagreement | 1 − 최빈답 비율 (k=5) |
| answer_entropy | 샘플 답분포 정규화 엔트로피 |
| neg_token_logprob_mean | decision draft 평균 NLL |
| verbalized_uncertainty | 1 − 언어화 confidence | 
| hedging_rate | reasoning 100단어당 헤지 수 |

NaN은 median 임퓨트. **verbalized_uncertainty 단독 최강(AUROC ~0.83)**.

---

# PART IV. 완료된 결과 (Phase 0 · 1 · 2 일부)

## 11. Phase 0 — 게이트키퍼: GO

CV AUROC(risk,error) **0.841** (MedMCQA 0.813 / PubMedQA 0.730), n=3800, 오답 1079, 22과목
median 0.80. **≥0.70 → GO.** verbalized confidence 0.83이 0.85 self-deception 경계 근접 → Phase 2 예약.

## 12. Phase 1 — Stage-A: 분포무관 보장 성립

분할 train 40 / cal 30 / test 30 (subject-stratified, 문항단위). full 3800(train1509/cal1140/
test1151): 위험 AUROC(test) 0.822. **conformal 게이트**: α=0.10 → `Pr(공개|오답)` **0.077**
(공개 0.379) · α=0.05 → **0.044** (공개 0.272) → **단일 잠긴 test 보장 성립**. baseline: 자기신뢰
임계(B2) 유사 공개율이나 **보장 없음**, 자기일관성(B3) escalate-all 붕괴, release-all(B0) 위반.
판정 **OVER_ESCALATION_AUDIT**(강한 모델+엄격 α → 공개 38%, 정직한 "과에스컬레이션 필요").
(소표본 파일럿 n=279는 단일-split 노이즈 있었으나 200-split 평균 0.085/0.033≤α로 확인.)

## 13. Phase 2 — 감사

**5 detector 이식** (`phase2_qa_audit.py`, n=3800): orientation clean(0.842) · escalate-all
clean(0.572) · definitional clean(0.828) · **confidence-dominance FLAGGED(recover 0.963)** —
confidence-only 0.829가 full 0.842의 96% 복원, confidence 제거 시 0.694 → **게이트가 자기신뢰
지배**(informative-missingness의 QA판).

**교차-verifier (완료, n=1495)**: gemma가 gpt-oss 답 판정 → within-dataset **독립 verifier > 자기신뢰**
(MedMCQA 0.877 vs 0.806 gap +0.071 CI[+0.037,+0.106] 견고; PubMedQA 0.809 vs 0.774 gap +0.035
CI[−0.006,+0.077] 경계). 함의: 에스컬레이션을 답변 모델과 **탈결합** 가능. 1500 공통 subset(헤드라인은 3800).
None 5(0.3%).

**B1 능력 vs 독립성 분해 (완료, `phase2_decomp_gemma.json`)** — gemma가 같은 1495문항을 스스로 답한 뒤,
gemma가 *틀린* 문항에서도 verifier_risk가 gpt-oss 오류를 예측하나?
| subset | n | gpt-oss 오답 | AUROC |
|---|---|---|---|
| 전체 | 1495 | 424(28%) | 0.882 |
| gemma **맞힘** | 1220 | 206(17%) | **0.982** |
| gemma **틀림** | 275 | 218(79%) | **0.612** (빈응답제외 0.615) |

**결과 = 능력 지배적**(mixed). gemma가 답을 알 때 gpt-oss 오류를 거의 완벽(0.982) 포착, 모를 때 0.612로
급락 → cross-verifier ≈ **모델-간 불일치 탐지기**. 잔차 독립성(0.612>0.5)은 존재하나 약함. 부가:
gemma 자기정확도 **0.816 > gpt-oss 0.712**(더 강한 verifier → "약한 verifier도" 주장 불가), 오답상관
φ=0.535, 틀림subset 오답 79%(불균형→AUROC 저검정력). **→ verifier=위험 feature지 ground truth 아님을 실증.
"약한 verifier" 독립성 검정은 qwen3.6-27b(gpt-oss보다 약함) 분해(B1')로 이월.**

---

# PART V. 남은 전체 로직 (선택 모두 포함)

## 14. 모델 인벤토리 & 역할

| 모델 | ID | 역할 | 크기 | 포맷/런타임 | 양자화 |
|---|---|---|---|---|---|
| gemma-4-31b-it | `google/gemma-4-31b` | verifier 메인 | 31B | GGUF/llama.cpp | Q4_K_M |
| qwen3.6-27b | `qwen/qwen3.6-27b` | verifier #2 (dense) | 27B | **MLX/Apple** | 4-bit affine gs64 |
| gpt-oss-120b | `openai/gpt-oss-120b` | **답변자 #1** + self-verifier + 셔플 | 120B | GGUF/llama.cpp | MXFP4(네이티브) |
| Qwen3.5-122B-A10B | `Qwen3.5-122B-A10B`(unsloth) | **답변자 #2**(일반화) | 122B MoE A10B | GGUF/llama.cpp | **Q3_K_S(공격적)** |
| ~~qwen3-coder-next~~ | `qwen/qwen3-coder-next` | *미사용* (제외) | 80B | GGUF | 4bit |
| ~~qwen3.6-35b-a3b~~ | `qwen/qwen3.6-35b-a3b` | *미사용* | 35B MoE | GGUF | Q4_K_M |

verifier 3종 = **gemma · qwen3.6-27b · gpt-oss(self)**. HW = **Apple M3 Ultra 96GB**, LMStudio, 한 번에 하나 로드.
런타임 혼용(qwen3.6=MLX, 나머지=GGUF)·양자화 confound는 결과원장 §8.5에 상세. Qwen3.5 Q3_K_S는 품질 하한.

오류 라벨 = gold(외부). 스왑은 **운영자 수동**(항상 한 모델만 로드, 메모리).

## 15. 완전 대칭 매트릭스 실행 플랜 (A9 = lift≈f(Δ))

**목표**: (answerer × verifier) **완전 정사각 매트릭스**로 Δ를 넓게 벌려 `lift≈f(Δ)` 법칙 확립.
Δ = acc(verifier) − acc(answerer). 대각선(Δ=0)은 self-confidence 기준선(drafts 자동, verify 불필요).
verifier self-accuracy는 답변자 무관(자문자답 = 문제 스스로 풀기) → **gemma/qwen3.6/gpt-oss 것 이미 확보,
Qwen3.5 것만 새로**. 공통 1500 고정.

| answerer↓ \ verifier→ | gemma(.816) | qwen3.6(~.80) | gpt-oss(.712) | Qwen3.5(X) |
|---|---|---|---|---|
| **gpt-oss(.712)** | +.104 ✅ | +.088 ✅ | 0 ✅conf | **X−.712** 🆕 |
| **Qwen3.5(X)** | .816−X 🆕 | .80−X 🆕 | .712−X 🆕 | 0 🆕conf |

**스왑 시퀀스 (4 로드로 8칸 완성, 로드된 김에 전부):**
```
[완료] gemma·qwen3.6·gpt-oss(self)의 자문자답 = 각 verifier acc 확보
[완료] gpt-oss row: ×gemma(A0) ×qwen3.6(step2) ×self(conf)
[진행] step2 ② qwen3.6 자문자답 (acc 확정용) — 끝나야 스왑 가능
──────── 길 B: 완전 매트릭스 ────────
1. Qwen3.5-122B  ① drafts_qwen35 생성(answerer_drafts.py)  ② gpt-oss 답 판정  ← 로드 김에 둘 다(대칭셀 공짜)
   → acc(Qwen3.5)=X 확정 → Δ 앵커 가치 결정 후 보고
2. gpt-oss       Qwen3.5 답 판정 (cross_verifier --drafts drafts_qwen35)
3. gemma         Qwen3.5 답 판정
4. qwen3.6-27b   Qwen3.5 답 판정
```
- X=0.85 가정 시 Δ ∈ [−0.14, +0.14] 약 7점 → 회귀 가능. X를 1단계서 알게 됨(스왑 전 5문항 절단 테스트).
- **핵심 효율(사용자 지적)**: Qwen3.5 로드 1단계에서 gpt-oss 답까지 판정 → 대칭셀 스왑0.
- dual-shuffle(암기 검문)은 **보조 특성화**로 강등 — 매트릭스/정보이론 확정 후 선택.

**허락 게이트 ★**: 각 스왑은 그때 요청. 1단계 후 acc(Qwen3.5) 보고 → verifier 셀 몇 개 돌릴지 결정.

## 15.1 검정력 & 1500 subset (비판 4 대응)

- 1500 = 3800 결정적 셔플의 **첫 1500**(재현 가능한 고정 부분집합). 오답 ~270(medmcqa 143 +
  pubmedqa 126) → AUROC 부트스트랩 CI 폭 ±0.04, conformal α=0.05 feasibility(n_err≥19) 충분.
- 논문엔 "**1500-item 사전지정 부분집합, 검정력 명시**"로 보고. 필요 시 3800 확장은 turnkey
  (같은 명령 `--n 3800`).

## 16. 각 단계 = 증명 + 출력

| # | 모델 | 작업 | 증명 | 출력 |
|---|---|---|---|---|
| 0 | gemma | gpt-oss 원본 답 판정 1500 | 독립 > 자기신뢰(메인) | `verifier_cross.jsonl` |
| **A6** | — (LLM 0) | verifier feature로 Phase 1 재적합 + 공개율 | **실용성: 같은 α에서 공개율↑ (급소→결과)** | `results/phase2/verifier_augmented.json` |
| **B1** | gemma | 자문자답 1500 (answerer) | **능력 vs 독립성 분해** (틀린 문항서도 예측?) | `drafts_gemma_self.jsonl` |
| 2a | gpt-oss | 셔플 재생성 400 | gpt-oss confidence 암기? | `drafts_medmcqa_shuffled.jsonl` |
| 2b | gpt-oss | 자기 판정 원본1500+셔플400 | same-model 대조 + 셔플 강건성 | `verifier_gptoss.jsonl` |
| 3 | gemma | gpt-oss 셔플 판정 400 | gemma 암기? | `verifier_shuffled.jsonl` |
| 4 | qwen3.6-27b | gpt-oss 원본1500+셔플400 판정 | 다른계열 verifier 작동+암기아님 | `verifier_qwen27*.jsonl` |
| 5 | Qwen3.5 | 답변 drafts 1500 + 셔플 400 | 2nd 답변자(일반화) | `drafts_qwen_*.jsonl` |
| 6 | gemma | Qwen 답1500 + 셔플400 판정 | 독립 verifier 다른 답변자에도 | `verifier_qwen_gemma.jsonl` |
| 7 | qwen3.6-27b | Qwen 답1500 + 셔플400 판정 | 2nd 답변자에도 multi-verifier 대칭 | `verifier_qwen_qwen27.jsonl` |

## 17. 사이 분석 (LLM 0, 스왑 없음)

- **A1 multi-verifier 비교**(1~4 후): 3 verifier(gemma·qwen3.6-27b·gpt-oss) × 공통 1500 →
  within-dataset AUROC + **각 verifier로 conformal 게이트 P(공개|오답)≤α 성립** → "verifier 교체 가능"을 **결과로**.
- **A2 dual-shuffle (전 verifier 대칭)**(2~4 후): gpt-oss confidence + **각 verifier(gemma·qwen3.6·gpt-oss-self)**
  의 셔플 vs 원본 AUROC → "자기신뢰 암기냐 + **모든 독립 verifier가 진짜냐(암기 아님)**" 판정표. (리뷰어의 "다른 verifier도 진짜냐?" 선제 차단.)
- **A3 Qwen Tier1+Phase1**(6a 후): Qwen 자기 feature AUROC[Phase0 복제] + Qwen 게이트 보장 → "feature+게이트 일반화".
- **A4 Qwen dual-shuffle**(6b 후): Qwen confidence 셔플 강건성 → self-deception 일반성.
- **A5 Qwen Tier2**(7 후): gemma가 Qwen 답 예측 AUROC → "독립 verifier가 다른 답변자에도".
- **A6 verifier→Phase1 통합 + 실용성**(1~4 후 언제든): 게이트에 verifier feature 추가 →
  같은 α에서 공개율↑? (실용성을 논거로). **[완료: 0.824→0.895, +7%p, α 성립]**
- **A7 능력 vs 독립성 분해 (verifier별 대칭)**(B1/B1' 후): verifier가 *스스로 틀린* 문항에서도
  verifier_risk가 gpt-oss 오류를 예측하나? **[gemma 완료: 능력지배 0.982→0.612]**. qwen3.6-27b(B1')로 대칭.

### ★ A8 정보이론 해부 (재편 코어, LLM 0) — `phase2_infotheory.py` (만들 것)

signal S(=verifier_risk 또는 self_confidence), error Y∈{0,1}에 대해:
- **I(S; Y)** — 각 신호의 오류에 대한 상호정보(bits). MI로 AUROC를 보완(순위 아닌 정보량).
- **조건부 I(verifier; Y | conf)** — *가장 중요*: verifier가 self-confidence **너머** 더하는 **증분 정보**.
  이게 "탈결합의 실제 가치"의 정직한 척도. (conf/verifier를 분위수 이산화 후 plug-in MI + 편향보정.)
- **H(Y | S)** — 신호별 잔여 불확실성. `1 − H(Y|S)/H(Y)` = 불확실성 감소율.
- **agreement κ**(verifier_answer, answerer_answer)와 P(error | 불일치) — "불일치=오류신호"의 강도.
- **verifier 능력별 층화**: I(verifier;Y | conf)를 verifier-맞힘/틀림 subset으로 분해 → 증분 정보가
  능력에서 오는지 정보이론적으로 재확인(A7의 MI판).
- 선행 informative-missingness 연구의 **I/H·κ 프레임 그대로 재사용** → 데이터과학 통찰 연결.

### ★ A9 능력격차 법칙 (재편 코어) — 답변자×verifier 매트릭스

각 (answerer a, verifier v) 셀에서 **Δ = acc(v) − acc(a)** 와 **lift = AUROC(v_risk) − AUROC(a_conf)**
(및 증분 MI)를 측정 → **lift ≈ f(Δ)** 관계를 산점도+회귀로. 셀:
- (gpt-oss)×(gemma) Δ=+0.104 lift=+0.071 ✅ · (gpt-oss)×(qwen3.6) Δ≈+0.09 lift=+0.050 ✅ [2점 확보]
- (gpt-oss)×(gpt-oss self)=대조 Δ=0 · (Qwen3.5 답변자)×(gemma·qwen3.6·self) → 매트릭스 확장.
- **가설**: lift가 Δ로 단조·설명되면 "cross-model 이득의 원천은 독립성이 아니라 능력격차"가 **정량 법칙**.
  Δ≈0에서도 lift>0인 셀이 있으면 그만큼이 순수 독립 성분(현재까진 미확인).

## 18. 총 비용 (1500 공통 기준, 대략)

| 단계 | 시간 | 우선순위 |
|---|---|---|
| 0 gemma 원본 verify(남은) | ~3h | A |
| **A6** verifier-augmented Phase1 + 공개율 | **~0 (LLM 없음)** | **A(최우선)** ✅ |
| B1 gemma 자문자답 1500 | ~8h | B ✅완료(자기정확 0.816) |
| 2 gpt-oss(셔플재생성 + self-verify 원본+셔플) | ~4h | B |
| 3 gemma 셔플-verify 400 | ~2h | B |
| 4 qwen3.6-27b (원본1500 + 셔플400) | ~7h | B |
| 5 Qwen drafts(9k gen)+셔플 | ~15h | D |
| 6 gemma(Qwen 답1500 + 셔플400) | ~10h | D |
| 7 qwen3.6-27b(Qwen 답1500 + 셔플400) | ~7h | D |
| **LLM 생성 합계** | **~56h** (약 2.3일, 재개가능) | |

※ **A6는 LLM 0** — verifier(1500) 끝나면 즉시. 급소를 가장 싸게 막음. B/D는 이후 순차.

분석 A1~A6은 LLM 0, 각 분 단위.

---

# PART VI. 출판 · 한계

## 19. 헤드라인 (재편 — insight first, 게이트는 매개)

**주장 순서(중요도 순):**
1. **[주 발견] escalation 신호의 우위 ≈ 능력격차 Δ의 함수.** cross-verifier가 answerer의
   self-confidence를 이기지만(gemma +0.071 / qwen3.6 +0.050, MedMCQA 견고), 그 우위는 verifier가
   *스스로 맞힌* 문항에서만 살아있고(AUROC 0.98) *틀린* 문항에선 붕괴(0.61). 두 verifier 모두
   answerer보다 강함(Δ>0). → **신호는 독립적 오류탐지가 아니라 "강한 모델의 second-opinion(불일치)"**.
2. **[정보이론 해부] self-confidence *너머* 증분 정보를 정량화.** I(verifier;error) vs
   조건부 I(verifier;error | self_confidence)로 "탈결합이 실제로 더하는 정보"를 측정하고, 능력격차 Δ와의
   관계 **lift ≈ f(Δ)**를 답변자×verifier 매트릭스로 확립. agreement κ·조건부 엔트로피 H(error|·)로 보강.
   → 선행 **informative-missingness(I/H)** 프레임과 연결되는 데이터과학 통찰.
3. **[응용 매개] 그런 불완전·능력얽힘 신호도 분포무관으로 운용 가능.** label-conditional conformal 게이트
   `Pr(공개|틀림)≤α` 유한표본 보장(잠긴 test 검증) + verifier feature 통합 시 같은 α에서 공개율 **+7%p**(A6).
   → "완벽한 신호가 아니어도 보장으로 감싸 안전하게 쓴다"가 게이트의 역할.

→ **dual-shuffle은 헤드라인 아님** — "신호가 벤치마크 암기인가"를 검문하는 특성화(보조). 아래 표는 그 분기.

> **B1 반영(중요)**: 이전 초안의 "능력이 아닌 독립성에서 옴"은 B1이 **반증**했다. 정직한 서사는
> "탈결합은 실용적으로 작동하나(feature로서 유효), 그 값은 대체로 능력 기반이며 약한 독립 잔차를
> 가진다 → 그래서 ground truth가 아닌 feature로만 쓴다". 약한-verifier 독립성 증거는 qwen3.6(B1')로 검정.

→ 이 셋은 **셔플 결과와 무관하게 성립**하므로 논문의 척추다. **dual-shuffle은 헤드라인이 아니라
"신호가 암기냐 진짜냐"를 부가로 밝히는 특성화 감사**로 강등한다. 그래서 아래 어떤 결말이 나와도
논문이 흔들리지 않는다:

| dual-shuffle 결말 (특성화) | 색채 | 강도 |
|---|---|---|
| gpt-oss 붕괴 + gemma 유지 | 자기신뢰=암기, 독립 verifier=진짜 → verifier 써라 (폭로+해법) | ⭐⭐⭐ |
| 둘 다 붕괴 | 둘 다 벤치마크 암기 = 오염 폭로 | ⭐⭐ |
| 둘 다 유지 + 정확도 유지 | 신호 진짜 → conformal 검증형 방법론 | ⭐ |
| 둘 다 유지 + 정확도 붕괴 | 답-레벨 벤치마크 오염 | ⭐⭐ |

어느 결말이든 (보장 방법론 + 탈결합 + 정직한 감사)로 **솔루션 논문**. 실제론 부분(스펙트럼) 가능.

## 20. 한계 (정직)

1. 벤치마크 오류 ≠ 임상 위해.
2. 답변자 gpt-oss 단일 → **Qwen3.5로 일반화 보강**(위험2).
3. verifier 의존 전가 → **verifier 교체 실험(A1)으로 "교체 가능"을 결과로**(위험1). "보장이 중심, verifier는 교체 가능 feature".
4. 실용성 공개 38% → **A6 확정: verifier 통합 시 같은 α에서 공개율 +7%p(0.36→0.43 @α=.10), 안전 유지** → 약점→해법 루프 완성(위험4 해소).
5. verifier도 LLM → 편향 존재(셔플로 오염만 검문).
6. **verifier 비종결 문항 5/1500 (0.3%)** — thinking 모델이 답을 안 내는 병리적 케이스. None 처리·제외하고 **사실 기재**(강제 채우지 않음). 유효 n=1495.
7. **cross-verifier 신호는 능력 지배적(B1)** — gemma가 답을 알 때만 gpt-oss 오류를 잘 잡음(맞힘 0.982 / 틀림 0.612). 즉 "모델 불일치" 신호이지 순수 독립 오류탐지가 아님. 그래서 verifier는 feature로만 사용. gemma가 gpt-oss보다 강해(0.816>0.712) "약한 verifier도 작동" 주장은 못 함 → qwen3.6(B1')로 검정 이월.

## 20.1 확정 결과 스냅샷 (A0 + A6 + B1)

- **verifier(1495)**: MedMCQA 독립 0.877 vs 자기신뢰 0.806 (gap +0.071, CI[+0.037,+0.106] 견고) ·
  PubMedQA 0.809 vs 0.774 (gap +0.035, CI[−0.006,+0.077] **경계·유의성 약함**). None 5(0.3%).
- **A6 verifier-augmented 게이트**: risk AUROC(test) 0.824→**0.895**; 공개율 α=.10 0.362→**0.432(+0.070)**,
  α=.05 0.293→**0.354(+0.061)**; **둘 다 α 충족**(안전 유지). → "약점→해법" 루프 완성.
- **B1 능력/독립성 분해(gemma)**: 자기정확도 0.816(>gpt-oss 0.712). AUROC(verifier_risk→gpt오류) 전체 0.882 /
  **gemma-맞힘 0.982 / gemma-틀림 0.612**(빈응답제외 0.615). φ=0.535. **판정: 능력 지배적, 약한 독립 잔차** →
  "verifier=feature" 실증. mixed 결과지만 정직 기재. 약한-verifier 독립성 검정은 **B1'(qwen3.6-27b) 대기**.

---

# PART VII. 인프라 · 현재상태 · 이후

## 21. 인프라

- 로컬 LMStudio, 외부 API $0, PHI egress 0. 모든 LLM 생성 **재개 가능**(JSONL 캐시).
- 검증 코어+진단기 순수함수+단위테스트 **21/21**, 데이터 없이 실행/검증.
- 모델 스왑: `~/.lmstudio/bin/lms unload --all` → `lms load <id>` (수동).
- ✅ 스크립트: `phase2_cross_verifier.py`(VERIFIER_MODEL/--out/--repair), `phase2_shuffle_audit.py`,
  `phase2_shuffle_verify.py`, `phase0_gatekeeper.py`(--drafts), `phase1_stage_a.py`(--drafts),
  `qa_drafts.py`(--max-tokens), **`phase2_verifier_augmented.py`(A6)**, **`phase2_selfanswer.py`(B1/B1', --tag/SELFANSWER_MODEL, 분해 자동)**.
- 🔧 만들 것: multi-verifier 비교 스크립트, Qwen용 shuffle/verify 경로 파라미터화(--drafts/--out).

## 22. 현재 상태

```
[완료] Phase 0 GO(0.841) · Phase 1 보장(0.077/0.044≤α) · Phase 2 confidence-dominance FLAGGED
[완료] A0 gemma cross-verifier 1495 — 독립>자기신뢰(MedMCQA +0.071견고, PubMedQA +0.035경계)
[완료] A6 verifier-augmented — 공개율 +7%p, α 성립 (실용성 결과)
[완료] B1 gemma 능력/독립성 분해 — 능력지배(0.982→0.612), gemma>gpt-oss → verifier=feature 실증
[대기·★허락] step2 qwen3.6-27b 스왑: cross-verifier + B1'자문자답(약한-verifier 독립성 핵심검정)
[대기] 2~4(gpt-oss/gemma/qwen3.6 셔플·dual-shuffle) → 5~7(Qwen3.5 일반화) → 결론.
       헤드라인=보장+탈결합(feature)+실용성, shuffle=특성화. B1 반영: "순수 독립성" 주장 철회.
```

## 23. 이후 — Phase 3 (Stage-B, 별도)

MIMIC/eICU objective-label을 leakage-safe decision-time feature로 → answer-reliability(Stage A)
vs objective-label **정보이론 대조(I/H)** + agreement(κ). 성능 아닌 통찰(선행 Patterns §7 연결).

**한 줄**: 방법(conformal 보장)은 검증됐고, 약점(자기신뢰 의존)은 스스로 드러냈으며, 그 약점을
독립 verifier로 대체 가능한지(A1) + 신호가 진짜인지(dual-shuffle) + 다른 모델에도 되는지(Qwen)를
대칭적으로 검증한다. 어떤 결과든 정직한 솔루션 논문이 되도록 설계됨.
