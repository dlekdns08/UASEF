# 마스터 문서 — Audited Conformal Escalation for Medical LLM QA

> 이 프로젝트의 **단일 완전 참조 문서**. 동기·수학·파이프라인·완료결과·남은 전체 로직·
> 모델·비용·출판 프레이밍·한계·재현까지 전부. (기존 phase0_3_current_status.md +
> phase2plus_remaining_plan.md 통합·확장.)
> 작성 시점: gemma cross-verifier 1500 진행 중(~62%).

---

# PART I. 무엇을 / 왜

## 1. 한 문장

의료 LLM(gpt-oss-120b)이 답을 내기 **전에** 그 답이 틀렸을 위험 `r(x,a)`를 계산해
**공개(release) vs 전문가 인계(escalate)** 를 결정하는 게이트를, `공개 ⟺ r < τ` 형태로 만들고,
τ를 **틀린 답의 위험 분포**에서 conformal하게 뽑아 다음을 **유한표본·분포무관**으로 보장한다:

$$ \Pr(\text{공개} \mid \text{실제로 틀림}) \le \alpha. $$

## 2. 포지셔닝 (고정)

**벤치마크 정의 오류(MCQ 정오답)·objective-label 위험을 제어**하는 것이지 실제 임상 위해를
보장하지 않는다. verifier LLM은 **위험 feature**로만 쓰이며 ground truth가 아니다. 임상 배포
전엔 전문의 판정이 있는 open-ended QA 검증이 필요.

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

## 15. 우선순위 재조정된 스왑 시퀀스 (논문 방어 순)

**원칙**: 논문 급소(실용성·독립성)를 먼저 결과로 막고, dual-shuffle은 헤드라인이 아닌 특성화로.
verifier 3종: **gemma · qwen3.6-27b · gpt-oss(self)**.

```
── 우선순위 A: 논문 급소 방어 (LLM 최소) ────────────────
0. [진행중] gemma  gpt-oss 원본 답 verify 1500
   ↳ A6  verifier-augmented Phase 1 + 공개율 비교   [LLM 0]   ← 최우선: 38%를 결과로
   → 완료 보고 + 허락 ★
── 우선순위 B: 독립성 검증 (두 비판 선제 방어) ──────────
B1. gemma  자문자답 1500 (gemma-as-answerer)      [gemma 로드 유지, 스왑X]
    ↳ 능력 vs 독립성 분해: gemma가 *틀린* 문항에서도 gpt-oss 오류 예측? → 독립성(구조적)
2.  gpt-oss  셔플 재생성 400 + self-verify(원본1500+셔플400)  [스왑]  ← 셔플 먼저(대칭)
3.  gemma    gpt-oss 셔플 verify 400                [스왑, 복귀]
4.  qwen3.6-27b  gpt-oss 원본1500 + 셔플400 verify   [스왑]
    ↳ A1 multi-verifier 비교(3 verifier, 원본+셔플) + A2 dual-shuffle(특성화)
── 우선순위 D: 일반화 (2nd 답변자) ──────────────────────
5.  Qwen3.5-122B  답변 drafts 1500 + Qwen 셔플 400   [스왑]
6.  gemma    Qwen 답 1500 + Qwen 셔플 400 verify     [스왑]
7.  qwen3.6-27b  Qwen 답 1500 + Qwen 셔플 400 verify [스왑]  ← point5: 2nd 답변자에도 대칭
    ↳ A3 Tier1(feature)+Phase1(게이트) + A5 Tier2 + verifier-augmented on Qwen
```

**허락 게이트 ★**: 0(gemma verify) 완료 시 A6 돌려 보고 → 허락받고 B1~7. 각 스왑 그때 요청.
**핵심 변경 vs 이전**: (i) A6를 맨 앞(LLM 0)으로, (ii) gemma 자문자답(B1)으로 능력/독립성 분해 추가,
(iii) shuffle을 특성화로 강등, (iv) Qwen에도 qwen3.6-27b verify 추가(대칭).

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
  verifier_risk가 gpt-oss 오류를 예측하나? **[gemma 완료: 능력지배 0.982→0.612]**. qwen3.6-27b(B1',
  gpt-oss보다 약함)로 대칭 검정 → 약한 verifier도 잔차 독립성 보이면 "능력만은 아님" 증거.

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

## 19. 헤드라인 & 결과 분기 (dual-shuffle에 의존하지 않게)

**헤드라인(고정, 셔플 결과와 무관)**:
1. **분포무관 conformal 보장** `P(공개|오답)≤α` — 실데이터·잠긴 test에서 성립 (검증됨).
2. **탈결합** — 독립 모델이 답변자 자기신뢰 *이상*으로 오류 예측(within-dataset) → 에스컬레이션을 답변자와
   분리 가능 (A0). **단 그 신호는 대체로 verifier의 능력(정답 앎)에서 오며**(B1: gemma-맞힘 0.982 vs
   gemma-틀림 0.612), cross-verifier는 사실상 **모델-간 불일치 탐지기** → **verifier는 ground truth가 아닌
   위험 feature**(포지셔닝과 정합). "능력이 아닌 순수 독립성" 주장은 **하지 않는다**(B1이 반증).
3. **실용성** — verifier feature 통합 시 같은 α에서 공개율↑ (A6, +7%p).

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
