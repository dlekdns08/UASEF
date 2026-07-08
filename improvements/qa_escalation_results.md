# 결과 원장 — What Does a Cross-Model Escalation Signal Measure?
### (능력격차의 함수로서의 escalation 신호: 정보이론적 해부 / 게이트는 응용 매개)

> 확정 수치만 모은 단일 참조. 계획/서사는 `qa_escalation_master.md`. 최종 갱신 2026-07-08.
> **재편**: "안전 게이트" → "escalation 신호의 정보이론적 해부 insight". 임상 주장 축소(벤치마크 오류만 제어).
> 데이터: MedMCQA + PubMedQA(고부담 testbed). 답변자 = gpt-oss-120b. 오류라벨 = gold(외부). $0 로컬.

---

## 0. 한눈에 (claim → 증거 → 상태)

| # | 주장 | 핵심 수치 | 상태 |
|---|---|---|---|
| C1 | 분포무관 conformal 보장 성립 | P(공개\|오답) 0.077/0.044 ≤ α (잠긴 test) | ✅ 확정 |
| C2 | 게이트가 자기신뢰에 지배됨(자기감사) | conf-only가 신호 96.3% 복원 | ✅ 확정 |
| C3 | 독립 verifier > 자기신뢰 (feature로서) | gemma MedMCQA +0.071 / PubMedQA +0.035 | ✅ but 경계 |
| C4 | **실용성**: verifier 통합 시 같은 α에서 공개율↑ | +7%p, α 유지 | ✅ 확정 (최강) |
| C5 | verifier 신호는 **능력 지배**(순수 독립성 아님) | gemma 맞힘 0.982 / 틀림 0.612 | ✅ 확정 (mixed) |
| C6 | **다른 계열 verifier(qwen3.6)도 견고하게 우위** | qwen3.6 MedMCQA **+0.050** [+.013,+.086] | ✅ 확정(full 1498) |
| C7 | dual-shuffle: 신호가 암기냐 진짜냐 | — | ⏳ 미착수 |
| C8 | 2nd 답변자(Qwen3.5)로 일반화 | — | ⏳ 미착수 |

---

## 1. Phase 0 — 게이트키퍼 (n=3800)

- CV AUROC(risk → 오류) = **0.841** → GO (risk가 오류를 예측).

## 2. Phase 1 — Stage-A 분포무관 보장 (잠긴 test)

- P(공개 | 오답) = **0.077** (α=0.10), **0.044** (α=0.05) → **둘 다 ≤ α, 보장 성립**.
- label-conditional(Mondrian) split-conformal, τ = k번째 오류-case 위험, k=⌊α(n_err+1)⌋.

## 3. Phase 2 — 자기감사 (detector, n=3800)

- orientation clean 0.842 · escalate-all clean 0.572 · definitional clean 0.828.
- **confidence-dominance FLAGGED**: conf-only 0.829 = full 0.842의 **96.3%** 복원;
  conf 제거 시 0.694. → 게이트가 **자기신뢰 지배**(informative-missingness의 QA판).

## 4. A0 — 교차 verifier: gemma가 gpt-oss 답 판정 (n=1495, within-dataset)

| 데이터셋 | 독립 verifier(gemma) | 자기신뢰(gpt-oss) | gap | 95% CI | 판정 |
|---|---|---|---|---|---|
| MedMCQA | 0.877 | 0.806 | **+0.071** | [+0.037, +0.106] | 견고 |
| PubMedQA | 0.809 | 0.774 | +0.035 | [−0.006, +0.077] | **경계(0 포함)** |

- pooled는 Simpson으로 팽창 → **within-dataset로만 보고**.
- verifier 비종결 None 5건(0.3%) — 제외·사실기재(강제 채우지 않음).

## 5. A6 — verifier-augmented 게이트 (실용성, LLM 0, n=1495)

- risk AUROC(test): base(5 feature) **0.824** → augmented(+verifier) **0.895**.
- 공개율: α=.10 0.362 → **0.432 (+0.070)** · α=.05 0.293 → **0.354 (+0.061)**.
- **양쪽 α 충족**(rel|inc base .047/.016, aug .054/.031 ≤ α) → 안전 유지하며 자율성↑.
- → "38% 공개(급소)"를 **결과로** 전환. 논문 최강 "so what".

## 6. B1 — 능력 vs 독립성 분해 (gemma, n=1495)

gemma가 같은 문항을 스스로 답한 뒤, gemma가 *틀린* 문항에서도 verifier_risk가 gpt-oss 오류 예측?

| subset | n | gpt-oss 오답 | AUROC |
|---|---|---|---|
| 전체 | 1495 | 424 (28%) | 0.882 |
| gemma **맞힘** | 1220 | 206 (17%) | **0.982** |
| gemma **틀림** | 275 | 218 (79%) | **0.612** (빈응답 제외 0.615) |

- gemma 자기정확도 **0.816 > gpt-oss 0.712** → "약한 verifier도 작동" 주장 **불가**.
- 오답 상관 φ = **0.535** (중간). 틀림 subset 오답 79% → AUROC 저검정력(불균형).
- **판정 = 능력 지배적**. cross-verifier ≈ **모델-간 불일치 탐지기**. 잔차 독립성(0.612>0.5)은 약함.
- 함의: verifier = **위험 feature지 ground truth 아님**을 실증. (포지셔닝과 정합.)

## 7. step 2 — qwen3.6-27b (① 완료, ② 진행 중)

**① cross-verifier 완료 (within-dataset, full n=1498, None 2):**

| 데이터셋 | qwen3.6 | 자기신뢰 | gap | 95% CI | 판정 |
|---|---|---|---|---|---|
| MedMCQA (n=1181) | 0.850 | 0.800 | **+0.050** | [+0.013, +0.086] | **견고(0 제외)** |
| PubMedQA (n=317) | 0.814 | 0.774 | +0.040 | [−0.001, +0.081] | 경계 |

- 파싱 실패 0% (4096 적합). risk 60%가 0점(퇴화=이진 동의/불일치)에도 판별 성립.
- **정정**: 예비 250건의 "+0.006(우위 없음)"은 초반 노이즈였음. full은 **+0.050 견고**.
- **함의**: gemma·qwen3.6 **두 계열 모두** 자기신뢰를 견고하게(MedMCQA) 이김 → **A1 "verifier 교체 가능"
  강화**. 단, 둘 다 답변자보다 강한 모델이라 우위는 능력과 얽힘(C5와 정합).

**② B1' 자문자답 분해 (완료, n=1498):**
- qwen3.6 자기정확도 **0.805** > gpt-oss 0.712 → gemma(0.816)처럼 답변자보다 강함(구조적 confound).
- 분해: 전체 0.878 / verifier-맞힘 **0.981** / verifier-틀림 **0.593**.
- **★ gemma와 거의 동일 재현** (gemma 0.882/0.982/0.612) → **능력지배가 gemma 특유 아님, 두 계열서 재현**.
  cross-verifier ≈ 모델-간 불일치 탐지기(family-invariant).

## 7.5 A8 — 정보이론 해부 (완료, gpt-oss row 2셀, bits)

| 양 (bits) | gpt-oss×gemma (n=1495) | gpt-oss×qwen3.6 (n=1498) |
|---|---|---|
| H(Y) 오류 불확실성 | 0.861 | 0.862 |
| I(verifier; Y) | 0.348 | 0.333 |
| I(self-conf; Y) | 0.222 | 0.224 |
| **I(verifier; Y \| self-conf)** 증분 | **0.161 (46%)** | **0.149 (45%)** |
| κ(verifier답, 답변자답) | 0.71 | 0.71 |
| **P(오류 \| 일치)** | **0.11** | 0.12 |
| **P(오류 \| 불일치)** | **0.84** | 0.81 |
| 증분정보 층화 맞힘/틀림 | 0.446b / **0.027b** | 0.462b / **0.008b** |

- **해석**: verifier는 self-conf 너머 **~0.15b(≈45%) 증분 정보** 제공(탈결합 실질값) — 단 그 증분이
  **verifier-틀림 subset에선 ≈0**(능력 게이팅의 bits판, A7과 정합). **"불일치→오류율 84% vs 일치 11%"**가 핵심 그림.
- 두 verifier 계열이 거의 동일 → 재현성. **Δ≈+0.09~0.10에 뭉침 → A9엔 Qwen3.5 답변자로 Δ 변동 필요.**

---

## 7.6 길 B — 2번째 답변자 Qwen3.5 (non-thinking) 매트릭스 (진행 중)

- **acc(Qwen3.5 no-think) = 0.795** (medmcqa **0.862** 강함 / pubmedqa **0.543** 약함). gpt-oss(0.712)와 구별 →
  row 유효. **데이터셋별 편차 큼 → within-dataset Δ가 넓게 벌어짐**(A9에 유리).
- **★ B-1b 보조 발견 (Qwen3.5 no-think를 verifier로)**: gpt-oss 답 판정 시 lift **−0.031(medmcqa) / −0.084(pubmedqa)**
  → **자기신뢰보다 나쁨**. gemma/qwen3.6(thinking, 비슷한 Δ에서 +0.04)와 정반대. → **"강한 모델도 추론 없이는
  verifier로서 무력"** (verifier 가치=추론). 단 config(no-think)라 **A9 주 법칙에선 제외**, 보조 관찰로 기재.
- **설계 확정**: Qwen3.5 = **답변자 전용**. A9 법칙은 **thinking verifier(gemma·qwen3.6·gpt-oss)** 로만 구성.
- **하단 row 진행**: gpt-oss→Qwen3.5(B-2, 음의 앵커 Δ=−0.083) [진행중] → gemma(B-3) → qwen3.6(B-4).

## 8. 종합 판정 (현재)

- **탄탄**: C1 보장(척추) · C4 실용성. 이 둘은 셔플/verifier 결과와 무관하게 성립.
- **의외로 견고**: C3·C6 — gemma·qwen3.6 **두 계열 모두** 자기신뢰를 견고히(MedMCQA) 이김 → verifier 교체
  가능(A1). PubMedQA만 경계.
- **정직한 confound**: 두 verifier(gemma 0.816, qwen3.6 ~0.80)가 모두 답변자 gpt-oss(0.712)보다 **강함**
  → 우위가 능력과 얽힘(C5: 능력 지배). "약한 verifier도 되나"는 검정 불가(구조적). = cross-verifier는
  **강한 second-opinion**이지 순수 독립 오류탐지 아님.
- **재편(채택)**: insight-first 헤드라인 —
  ① **주 발견**: escalation 우위 ≈ **능력격차 Δ의 함수**(C3·C5·C6, 두 verifier 모두 Δ>0·능력지배).
  ② **정보이론 해부**: I(verifier;error|conf) 증분정보 + lift≈f(Δ) 매트릭스(A8·A9, 만들 것) → informative-missingness 연결.
  ③ **응용 매개**: 불완전 신호도 conformal로 `Pr(공개|틀림)≤α` 운용(C1) + 공개율 +7%p(C4).
- **임상 축소(채택)**: "medical safety" → "**QA reliability under a distribution-free gate**". 의료 QA는
  **고부담 testbed**일 뿐 임상 위해 주장 아님. 임상 적용은 future work(전문의 판정 open-ended 검증 선행).
- **천장(정직)**: 벤치마크 오류 ≠ 임상 위해 — 이제 **한계가 아니라 범위 설정(scope)**으로 명시.

## 8.5 실험 환경 · 모델 · 양자화 (보고서 Methods용)

**하드웨어**: Apple **M3 Ultra, 96 GB 통합 메모리** (macOS). 추론 엔진 **LMStudio**(llama.cpp GGUF +
Apple MLX 백엔드 혼용). 외부 API $0, 모든 처리 로컬, PHI egress 0. 모델은 **한 번에 하나만 로드**
(운영자 수동 스왑; 단일 모델 최대 63 GB < 96 GB, 여유 있음).

| 역할 | 모델 | 배포 repo | 포맷/런타임 | **양자화** | 디스크 | ≈bpw |
|---|---|---|---|---|---|---|
| 답변자 #1 | `openai/gpt-oss-120b` | lmstudio-community | GGUF / llama.cpp | **MXFP4** (네이티브) | 59 GB | ~4.2 |
| verifier #1 | `google/gemma-4-31b-it` | lmstudio-community | GGUF / llama.cpp | **Q4_K_M** | 19 GB | ~5.1 |
| verifier #2 | `qwen/qwen3.6-27b` | lmstudio-community | **MLX / Apple** | **4-bit affine, group_size 64** | 15 GB | ~4.8 |
| 답변자 #2 | `Qwen3.5-122B-A10B` (MoE, 활성 10B) | unsloth | GGUF / llama.cpp | **Q3_K_S** | 51 GB | ~3.6 |

- 오류 라벨 = gold(외부 벤치마크). 디코딩: 결정 temp=0.0, self-consistency 샘플 k=5 temp>0.
  verifier max_completion_tokens=4096(thinking 여유). 재현: JSONL 캐시 + 결정적 셔플(seed 0).
- **런타임 혼용 주의**: gpt-oss·gemma·Qwen3.5는 llama.cpp(GGUF), qwen3.6-27b만 **MLX**. 수치 미세차 가능.
- **Qwen3.5 답변자 = non-thinking mode**: Q3_K_S 빌드가 항상 thinking(끌 수 없는 소프트스위치, `/no_think`·
  `enable_thinking=false` API 무시)이라 절단 42%·저속(~75s/call). **Jinja 템플릿에 `{%- set enable_thinking
  = false %}`로 thinking 비활성화** → ~4s/call·빈응답 0. 답변자의 *답*만 쓰므로(과정 아님) 법칙에 무해;
  단 acc가 think-mode보다 낮을 수 있어 **"Qwen3.5 non-thinking"으로 정직 기재**. k=0(결정답+verbalized conf,
  A8/A9 기준선과 정합). think-mode 100행은 폐기(`_drafts_qwen35_thinkmode_discard.jsonl`).

**양자화 confound (한계에 명시)**:
1. 모든 수치는 **위 특정 양자화 수준**에서의 값 — full precision 아님. 답변자 오류율·verifier 판정 모두
   quant 의존. gpt-oss는 **MXFP4가 공식 릴리스 quant**이라 공정하나, 절대 정확도는 quant에 영향받음.
2. **Qwen3.5-122B는 Q3_K_S(≈3.6 bpw)로 가장 공격적** — 답변자 #2 품질이 저하됐을 수 있음. 일반화
   실험 해석 시 반드시 병기(품질 하한 케이스). 필요 시 상위 quant 재실행은 turnkey.
3. gemma는 mmproj(멀티모달 프로젝터) 포함하나 **텍스트 전용**으로 사용.

## 9. 산출물 (파일)

- `data/raw/verifier_cross.jsonl`(gemma) · `verifier_qwen27.jsonl`(진행) · `selfanswer_gemma.jsonl`
- `results/phase2/phase2_decomp_gemma.json` · `phase2_verifier_augmented.json` · `phase2_cross_verifier.json`
- 코어: `models/label_conditional_conformal.py`(21/21 test) · `qa_risk_features.py` · `qa_drafts.py`
