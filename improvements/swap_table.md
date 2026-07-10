# 스왑 실행 표 (9 세션) — 운영자용 · Option B (400+400 clean paired)

> 각 세션: **모델 로드(수동)** → 아래 명령 실행. 공통 앞머리: `cd /Users/idaun/PoC/UASEF && export UASEF_QUERY_TIMEOUT_S=400`
> 생략형 = `.venv/bin/python experiments/<script>.py`.
> **셔플 audit = 원본400 + 셔플400 둘 다 shuffle-judge 프롬프트**(prompt confound 0, self-contained). 재활용/sanity 없음.
> think/no-think = **Jinja 템플릿 토글 후 로드**(수동). 리로드 CTX: qwen3.5=22272 · qwen3.6=13649 · gpt-oss/gemma=로드값.
> **모든 cross_verifier(매트릭스) 명령에 `--verifier-max-tokens 16000` 추가** (Qwen3.5 heavy thinker 절단 방지; gemma/qwen3.6/gpt-oss는 조기종료라 무해). None율 5%→~0.3%.

## 모델 ID / 답변자 파일
| 약칭 | 모델 ID | 답변자 | drafts |
|---|---|---|---|
| gpt-oss | `openai/gpt-oss-120b` | A1 | `data/raw/drafts_phase0_all.jsonl` |
| gemma | `google/gemma-4-31b` | A2 | `data/raw/drafts_qwen35_think.jsonl` |
| qwen3.6 | `qwen/qwen3.6-27b` | 셔플 items | 공통 1500의 MedMCQA 400 |
| Qwen3.5 | `qwen3.5-122b-a10b` | | |

> **원본 답 = `--no-shuffle` 재생성** (reference 재활용 폐기): 셔플과 **같은 조건**(프롬프트·max_tokens·temp)으로
> 원본옵션 재답변 → answerer 생성조건까지 100% 동일. 세션 1·2에서 답변자가 `--tag <a>`(셔플) + `--tag <a>_orig --no-shuffle`(원본) 둘 다 생성.
> 셔플 판정 = 셀당 2회: `shuffle_judge --answerer <a>`(셔플400) + `shuffle_judge --answerer <a>_orig`(원본400).

---

## Phase 1 — 답변자 셔플 재답변 (셔플셋 준비)

### 세션 1 · **Qwen3.5-T** (현재 로드) — 재생성 먼저 → 매트릭스
- 셔플 재답변: `ANSWERER_MODEL=qwen3.5-122b-a10b … shuffle_answer.py --tag qwen35 --n 400 --max-tokens 16000`
- 원본 재답변(--no-shuffle): `ANSWERER_MODEL=qwen3.5-122b-a10b … shuffle_answer.py --tag qwen35_orig --no-shuffle --n 400 --max-tokens 16000`
- 매트릭스 Qwen3.5-T→A1 (재답변 후): `VERIFIER_MODEL=qwen3.5-122b-a10b … cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_q35T_gptoss.jsonl --reload-every 100 --reload-context 22272 --reload-parallel 4`

### 세션 2 · **gpt-oss-T**
- 매트릭스 gpt-oss-T→A2 [음의앵커]: `VERIFIER_MODEL=openai/gpt-oss-120b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gptT_q35.jsonl --reload-every 100 --reload-parallel 4`
- 셔플 재답변: `ANSWERER_MODEL=openai/gpt-oss-120b … shuffle_answer.py --tag gptoss --n 400 --max-tokens 2048`
- 원본 재답변(--no-shuffle): `ANSWERER_MODEL=openai/gpt-oss-120b … shuffle_answer.py --tag gptoss_orig --no-shuffle --n 400 --max-tokens 2048`
- 셔플판정 gpt-oss-T→Qwen3.5: `VERIFIER_MODEL=openai/gpt-oss-120b … shuffle_judge.py --answerer qwen35 --tag gpt_T --max-tokens 2048`
- 원본판정 gpt-oss-T→Qwen3.5원본: `… shuffle_judge.py --answerer qwen35_orig --tag gpt_T --max-tokens 2048`

*→ gpt-oss셔플·Qwen3.5셔플 둘 다 준비됨.*

---

## Phase 2 — verifier 판정 (매트릭스 + 셔플400 + 원본400)

### 세션 3 · **gpt-oss-N** (no-think 토글)
- 매트릭스 gpt-oss-N→A2: `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gptN_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4`
- **[Exp2/B1] self-answer + features (gpt-oss-N, B-1b 대칭짝)**: `SELFANSWER_MODEL=openai/gpt-oss-120b … selfanswer.py --tag gptossN --verifier-file data/raw/verifier_gptN_q35.jsonl --max-tokens 4096 --reload-every 100 --reload-parallel 4` (no-think 자기답변 → drafts_qwen35_nothink의 대칭짝; **매트릭스 후 실행** → item set·verifier_risk 공급; B1 분해 + competence proxy)
- 셔플판정: `… shuffle_judge.py --answerer qwen35 --tag gpt_N --max-tokens 1024`
- 원본판정: `… shuffle_judge.py --answerer qwen35_orig --tag gpt_N --max-tokens 1024`

### 세션 4 · **gemma-T**
- 매트릭스 gemma-T→A2: `VERIFIER_MODEL=google/gemma-4-31b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gemT_q35.jsonl --reload-every 100 --reload-parallel 4`
- **[Exp2] self-answer + features 재생성**: `SELFANSWER_MODEL=google/gemma-4-31b … selfanswer.py --tag gemma --verifier-file data/raw/verifier_cross.jsonl --max-tokens 4096 --reload-every 100 --reload-parallel 4` (verbalized conf·logprob·hedging 저장 → competence proxy)
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag gem_T --max-tokens 4096` · `… --answerer qwen35 --tag gem_T --max-tokens 4096`
- 원본판정×2: `… shuffle_judge.py --answerer gptoss_orig --tag gem_T --max-tokens 4096` · `… --answerer qwen35_orig --tag gem_T --max-tokens 4096`

### 세션 5 · **gemma-N** (no-think 토글)
- 매트릭스 gemma-N→A1: `… cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_gemN_gptoss.jsonl --reload-every 100 --reload-parallel 4`
- 매트릭스 gemma-N→A2: `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gemN_q35.jsonl --reload-every 100 --reload-parallel 4`
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag gem_N --max-tokens 1024` · `… --answerer qwen35 --tag gem_N --max-tokens 1024`
- 원본판정×2: `… --answerer gptoss_orig --tag gem_N --max-tokens 1024` · `… --answerer qwen35_orig --tag gem_N --max-tokens 1024`

### 세션 6 · **qwen3.6-T**
- 매트릭스 qwen3.6-T→A2: `VERIFIER_MODEL=qwen/qwen3.6-27b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_q36T_q35.jsonl --reload-every 100 --reload-context 13649 --reload-parallel 4`
- **[Exp2] self-answer + features 재생성**: `SELFANSWER_MODEL=qwen/qwen3.6-27b … selfanswer.py --tag qwen27 --verifier-file data/raw/verifier_qwen27.jsonl --max-tokens 4096 --reload-every 100 --reload-context 13649 --reload-parallel 4`
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag q36_T --max-tokens 4096` · `… --answerer qwen35 --tag q36_T --max-tokens 4096`
- 원본판정×2: `… --answerer gptoss_orig --tag q36_T --max-tokens 4096` · `… --answerer qwen35_orig --tag q36_T --max-tokens 4096`

### 세션 7 · **qwen3.6-N** (no-think 토글)
- 매트릭스 qwen3.6-N→A1: `… cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_q36N_gptoss.jsonl --reload-every 100 --reload-context 13649 --reload-parallel 4`
- 매트릭스 qwen3.6-N→A2: `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_q36N_q35.jsonl --reload-every 100 --reload-context 13649 --reload-parallel 4`
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag q36_N --max-tokens 1024` · `… --answerer qwen35 --tag q36_N --max-tokens 1024`
- 원본판정×2: `… --answerer gptoss_orig --tag q36_N --max-tokens 1024` · `… --answerer qwen35_orig --tag q36_N --max-tokens 1024`

### 세션 8 · **Qwen3.5-T** (재방문, think)
- 셔플판정 Qwen3.5-T→gpt-oss: `VERIFIER_MODEL=qwen3.5-122b-a10b … shuffle_judge.py --answerer gptoss --tag q35_T --max-tokens 16000`
- 원본판정: `… shuffle_judge.py --answerer gptoss_orig --tag q35_T --max-tokens 16000`

### 세션 9 · **Qwen3.5-N** (no-think 토글)
- (매트릭스 A1 = B-1b 완료, 생략)
- 셔플판정 Qwen3.5-N→gpt-oss: `VERIFIER_MODEL=qwen3.5-122b-a10b … shuffle_judge.py --answerer gptoss --tag q35_N --max-tokens 1024`
- 원본판정: `… shuffle_judge.py --answerer gptoss_orig --tag q35_N --max-tokens 1024`

---

## Phase 3 — 분석 (LLM 0, 스왑 없음) — reviewer 방어 4실험 + 코어

**★ Exp1 (최우선) calibration/sharpness 통제 후 ability gating** `phase2_calibration_gating.py` [완성]
- 각 셀: V·C를 calibrated p로(isotonic) → ECE·Brier·sharpness·AUROC. 중첩 로짓 M0(pC)→M1(+pV)→M3(+Z)→M4(+Z×불일치).
- Z(=verifier 자기정답, gold=mechanism)가 통제 후에도 유의(LRT)? → **"calibration/sharpness로 환원 안 됨" 방어**.
- **부분 결과(gpt-oss row 2셀): ΔNLL +0.073/+0.072, LRT p=0.0 ✅** (전 셀은 매트릭스 완성 후).

**★ Exp2 (필수) 배포가능 verifier competence proxy** `phase2_competence_proxy.py` [만들 것]
- q=P(verifier 자기정답 | verifier uncertainty features) 학습(logistic). features: verbalized conf·logprob·hedging·length·(k>0시 self-consistency).
- gpt-oss-T(drafts_phase0_all) + Qwen3.5-T(drafts_qwen35_think) + Qwen3.5-N(drafts_qwen35_nothink) = 무료(기존). gemma-T·qwen3.6-T = 세션4·6 self-answer 재생성(feature). **gpt-oss-N = 세션3 self-answer(selfanswer_gptossN, B-1b 대칭짝)**.
- **no-think verifier 대칭 현황**: Qwen3.5-N ✅(drafts_qwen35_nothink) · gpt-oss-N ✅(세션3 추가) · gemma-N·qwen3.6-N = 보류(결과 나쁘면 세션5·7에 추가).
- high/mid/low q tertile별 V lift 비교 + Y~pC+pV+q+pV×q → **gold 없이 competence로 signal 조절 가능**.

**★ Exp3 threshold/shift transfer robustness** `phase2_threshold_transfer.py` [만들 것]
- calibration split에서 τ 결정 → within-ds / cross-ds(Med↔Pub) / cross-answerer(gpt↔Qwen) / original→shuffled 이전.
- Pr(공개|오답)·release rate·τ instability 측정. C vs V vs C+V vs C+V+q 안정성 비교 → **thresholded 운영 안정성 + shift 한계 명시**.

**★ Exp4 option-shuffle audit (canonical-id 기준)** `phase2_shuffle_compare.py` [완성]
- 원본400 vs 셔플400: AUROC(risk→오류)·AUROC(불일치-**text/canonical id**→오류)·P(오류|불일치). **think 유지 vs no-think 붕괴(2×2)** = reasoning-mediated 증거.
- answerer robustness(원본/셔플 정확도·content-invariance) + verifier robustness(Δlift).

**코어**: A9 lift≈f(Δ) **within-dataset(per-ds Δ 필수)** · A8 전셀 정보이론 · 4모델 추론 ablation.

**논문 Results 재배치**: 4.1 self-conf 강하나 불완전 → 4.2 thinking cross-verifier 개선 → 4.3 item-level ability-gated →
**4.4 gating≠calibration/sharpness(Exp1)** → 4.5 competence proxy로 부분 배포가능(Exp2) → 4.6 평균Δ만으론 불충분 →
4.7 shuffle audit=reasoning 의존(Exp4) → 4.8 conformal gate로 불완전 신호 운용(+Exp3 robustness).

## 매트릭스 9 출력 체크리스트
- [완료] verifier_cross(gemT→A1) · verifier_qwen27(q36T→A1) · verifier_qwen35_of_gptoss(q35N→A1=B-1b)
- [신규] verifier_q35T_gptoss · verifier_gptT_q35 · verifier_gptN_q35 · verifier_gemT_q35 · verifier_gemN_gptoss · verifier_gemN_q35 · verifier_q36T_q35 · verifier_q36N_gptoss · verifier_q36N_q35

## 셔플 총량 (Option B, --no-shuffle 재생성)
- verifier 판정: 12셀 × (셔플400 + 원본400) = **9,600**
- 답변자 생성: 2모델 × (셔플400 + **원본400 --no-shuffle**) = **1,600** (같은 조건 clean paired)
- sanity·reference 재활용 없음. 프롬프트·생성조건 100% 동일, 옵션 순서만 차이.
