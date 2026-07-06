# 남은 전체 로직 (Phase 2+ / 선택 모두 포함)

> gemma verifier(1500)가 도는 현재 시점 이후의 **완전한 남은 순서**. 선택(🔶) 항목까지
> 전부 포함. 모델 스왑은 **운영자 수동**(메모리상 항상 한 모델만). 공통 subset = 1500.

## 모델 스왑 시퀀스 (7 로드, gemma 3회는 의존성상 불가피)

```
1. gemma        V1  cross-verifier(gpt-oss 답) 1500          [진행 중]  → 완료 보고 + 허락 ★
──── multi-verifier: 나머지 verifier가 gpt-oss 원본 답 판정 ────
2. qwen3.6-27b  V2  cross-verifier 1500                      [스왑]
3. qwen-coder   V3  cross-verifier 1500                      [스왑]
4. gpt-oss      V4  self-verifier 1500  +  S1 셔플 재생성 400 [스왑]  ← gpt-oss 로드 김에 둘 다
──── dual-shuffle(gpt-oss) 마무리 ────
5. gemma        S2  셔플-verify(gpt-oss 셔플 답) 400          [스왑]
──── 답변모델 #2 (Qwen3.5-122B) 일반화 ────
6. Qwen3.5-122B D   답변 drafts 1500  +  QS1 Qwen 셔플 재생성 400  [스왑]  ← 로드 김에 둘 다
7. gemma        T2  Qwen 답 판정 1500  +  QS2 Qwen 셔플 답 판정 400 [스왑]  ← gemma 복귀, 둘 다
```

**허락 게이트 ★**: 1(gemma verifier) 완료 시 보고 → 허락받고 2~7 진행. 각 스왑은 그때 요청.

## 각 단계 = 무엇을 증명하나 + 산출물

| # | 모델 | LLM 작업 | 증명 | 출력 |
|---|---|---|---|---|
| 1 | gemma | gpt-oss 답 판정 1500 | 독립 verifier > 자기신뢰 (메인) | `verifier_cross.jsonl` |
| 2 | qwen3.6 | 〃 | 다른 계열 verifier도 작동 | `verifier_qwen35.jsonl` |
| 3 | qwen-coder | 〃 | off-domain verifier도? | `verifier_qwencoder.jsonl` |
| 4a | gpt-oss | 자기 답 판정 1500 | same-model verifier 대조(순환성) | `verifier_gptoss.jsonl` |
| 4b | gpt-oss | 셔플 재생성 400 | gpt-oss confidence 암기 여부 | `drafts_medmcqa_shuffled.jsonl` |
| 5 | gemma | gpt-oss 셔플 답 판정 400 | gemma verifier 암기 여부 | `verifier_shuffled.jsonl` |
| 6a | Qwen3.5 | 답변 drafts 1500 | 2nd 답변자 생성 | `drafts_qwen_*.jsonl` |
| 6b | Qwen3.5 | Qwen 셔플 재생성 400 | Qwen confidence 암기 여부(일반성) | `drafts_qwen_shuffled.jsonl` |
| 7a | gemma | Qwen 답 판정 1500 | 독립 verifier가 다른 답변자에도 | `verifier_qwen_answers.jsonl` |
| 7b | gemma | Qwen 셔플 답 판정 400 | verifier가 Qwen 셔플에도 강건? | `verifier_qwen_shuffled.jsonl` |

## 사이사이 분석 (LLM 불필요, 스왑 없음)

- **A1. multi-verifier 비교** (1~4a 후): 4 verifier × 공통 1500 → 각 within-dataset AUROC + **각 verifier로 conformal 게이트가 P(release|오답)≤α 성립** → **"verifier 교체 가능"을 결과로** (위험1 해결).
- **A2. gpt-oss dual-shuffle** (4b+5 후): gpt-oss confidence·gemma verifier의 셔플 vs 원본 AUROC → **self-deception/오염 판정표** (핵심 서사).
- **A3. Qwen Tier 1 + Phase 1** (6a 후): Qwen 자기 feature로 AUROC(risk, Qwen-error) [Phase 0 복제] + Qwen drafts로 conformal 게이트 보장 → **"feature+게이트 일반화"** (위험2).
- **A4. Qwen dual-shuffle** (6b 후): Qwen confidence 셔플 강건성 → **self-deception이 model-specific인가 일반적인가**.
- **A5. Qwen Tier 2** (7 후): gemma가 Qwen 답 예측 AUROC → **"독립 verifier가 다른 답변자에도"** (해법 일반화).
- **A6. verifier→Phase 1 feature 통합 + 실용성** (1~4 후 언제든): 게이트에 독립 verifier feature 추가 → **같은 α에서 공개율↑?** (위험4: 실용성을 논거로).

## 총 비용 (대략, 1500 공통 기준)

| 단계 | 시간(대략) |
|---|---|
| 1 gemma(남은) | ~4h |
| 2 qwen3.6 / 3 qwen-coder | ~6h / ~10h |
| 4 gpt-oss (self-verify+셔플) | ~3h |
| 5 gemma 셔플-verify | ~2h |
| 6 Qwen drafts(9k gen)+셔플 | ~15h |
| 7 gemma (Qwen 답+셔플 판정) | ~10h |
| **합계 LLM 생성** | **~50h** (약 2일, 재개가능) |

분석(A1~A6)은 LLM 0, 각 ~분 단위.

## 산출물 → 논문 구성 매핑

- A1 → "verifier replaceable" (결과) · A2 → "self-deception 폭로" (핵심) · A5 → "해법 = 독립 verifier"
- A3+A5 → "다른 frontier 모델(Qwen)에도 일반화" · A4 → "self-deception 일반성" · A6 → "실용성"
- 어느 dual-shuffle 결말이든 (보장 방법론 + 탈결합 + 정직한 감사)로 **솔루션 논문** 성립.

## 이후 (Phase 3, 별도)

MIMIC/eICU objective-label을 leakage-safe feature로 → answer-reliability(Stage A) vs objective-label
정보이론 대조(I/H) + agreement(κ). 성능 아닌 통찰(선행 Patterns §7 연결).

## 인프라 상태 (스크립트)

- ✅ 있음: `phase2_cross_verifier.py`(VERIFIER_MODEL/--out), `phase2_shuffle_audit.py`,
  `phase2_shuffle_verify.py`, `phase0_gatekeeper.py`(--drafts), `phase1_stage_a.py`(--drafts),
  `qa_drafts.py`(--max-tokens)
- 🔧 만들 것: multi-verifier 비교 스크립트(per-model 파일→공통 1500 비교), verifier→Phase1
  feature 통합, Qwen용 shuffle/verify 경로 파라미터화(--drafts/--out 일반화)
