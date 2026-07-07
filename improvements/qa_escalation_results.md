# 결과 원장 — Audited Conformal Escalation for Medical LLM QA

> 확정 수치만 모은 단일 참조. 계획/서사는 `qa_escalation_master.md`. 최종 갱신 2026-07-07.
> 데이터: MedMCQA + PubMedQA. 답변자 = gpt-oss-120b. 오류라벨 = gold(외부). $0 로컬.

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

**② B1' 자문자답 (진행 중, 잠정 505/1500):**
- qwen3.6 자기정확도(잠정) **0.802** > gpt-oss 0.712 → gemma(0.816)처럼 **답변자보다 강함**.
  → "약한 verifier" 독립성 검정은 **두 verifier 모두 불가**(우리가 가진 verifier가 답변자보다 강함=구조적 confound).
- 분해(gemma-틀림 subset 유형)는 ②완료 후 자동 산출 — qwen3.6도 능력지배로 붕괴하는지 확인 예정.

---

## 8. 종합 판정 (현재)

- **탄탄**: C1 보장(척추) · C4 실용성. 이 둘은 셔플/verifier 결과와 무관하게 성립.
- **의외로 견고**: C3·C6 — gemma·qwen3.6 **두 계열 모두** 자기신뢰를 견고히(MedMCQA) 이김 → verifier 교체
  가능(A1). PubMedQA만 경계.
- **정직한 confound**: 두 verifier(gemma 0.816, qwen3.6 ~0.80)가 모두 답변자 gpt-oss(0.712)보다 **강함**
  → 우위가 능력과 얽힘(C5: 능력 지배). "약한 verifier도 되나"는 검정 불가(구조적). = cross-verifier는
  **강한 second-opinion**이지 순수 독립 오류탐지 아님.
- **재프레이밍 권고**: 헤드라인 = **"분포무관 escalation 보장(C1) + 실용 이득(C4) + 강한 독립 second-opinion을
  feature로, 그 한계를 정직히 해부(C2·C5)"**. B1 mixed는 **"신호가 ground truth 아니라 보장으로 감싼다"는
  논거**.
- **천장**: 벤치마크 오류 ≠ 임상 위해. 전문가 판정 전엔 임상 주장 불가(명시).

## 9. 산출물 (파일)

- `data/raw/verifier_cross.jsonl`(gemma) · `verifier_qwen27.jsonl`(진행) · `selfanswer_gemma.jsonl`
- `results/phase2/phase2_decomp_gemma.json` · `phase2_verifier_augmented.json` · `phase2_cross_verifier.json`
- 코어: `models/label_conditional_conformal.py`(21/21 test) · `qa_risk_features.py` · `qa_drafts.py`
