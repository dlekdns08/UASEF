# 스왑 실행 표 (9 세션) — 운영자용 · Option B (400+400 clean paired)

> 각 세션: **모델 로드(수동)** → 아래 명령 실행. 공통 앞머리: `cd /Users/idaun/PoC/UASEF && export UASEF_QUERY_TIMEOUT_S=400`
> 생략형 = `.venv/bin/python experiments/<script>.py`.
> **셔플 audit = 원본400 + 셔플400 둘 다 shuffle-judge 프롬프트**(prompt confound 0, self-contained). 재활용/sanity 없음.
> think/no-think = **Jinja 템플릿 토글 후 로드**(수동). 리로드 CTX: qwen3.5=22272 · qwen3.6=13649 · gpt-oss/gemma=로드값.

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
- 매트릭스 gpt-oss-N→A2: `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gptN_q35.jsonl --reload-every 100 --reload-parallel 4`
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

## Phase 3 — 분석 (LLM 0, 스왑 없음)
- **셔플 강건성 (셀별 clean paired)**: 원본400(`shuffle_judge_<a>_orig__<v>`) vs 셔플400(`shuffle_judge_<a>__<v>`)
  → AUROC(risk→오류)·AUROC(불일치-text→오류)·P(오류\|불일치) 원본 vs 셔플 비교. think 유지 / no-think 붕괴(암기) 2×2.
  *(비교 스크립트 `phase2_shuffle_compare.py` [만들 것, LLM0])*
- **A9 lift≈f(Δ)** within-dataset · **4모델 추론 ablation 표** · **A8 전셀 정보이론** (매트릭스 데이터)

## 매트릭스 9 출력 체크리스트
- [완료] verifier_cross(gemT→A1) · verifier_qwen27(q36T→A1) · verifier_qwen35_of_gptoss(q35N→A1=B-1b)
- [신규] verifier_q35T_gptoss · verifier_gptT_q35 · verifier_gptN_q35 · verifier_gemT_q35 · verifier_gemN_gptoss · verifier_gemN_q35 · verifier_q36T_q35 · verifier_q36N_gptoss · verifier_q36N_q35

## 셔플 총량 (Option B, --no-shuffle 재생성)
- verifier 판정: 12셀 × (셔플400 + 원본400) = **9,600**
- 답변자 생성: 2모델 × (셔플400 + **원본400 --no-shuffle**) = **1,600** (같은 조건 clean paired)
- sanity·reference 재활용 없음. 프롬프트·생성조건 100% 동일, 옵션 순서만 차이.
