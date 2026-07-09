# 스왑 실행 표 (9 세션) — 운영자용

> 각 세션: **모델 로드(수동)** → 아래 명령 실행. 공통 앞머리: `cd /Users/idaun/PoC/UASEF && export UASEF_QUERY_TIMEOUT_S=400`
> 스크립트 경로 생략형 = `.venv/bin/python experiments/<script>.py`. 리로드: `--reload-every 100 --reload-parallel 4 --reload-context <CTX>`.
> think/no-think은 **Jinja 템플릿 토글 후 로드**(수동). CTX: qwen3.5=22272 · qwen3.6=13649 · gpt-oss/gemma=로드값에 맞춤.

## 모델 ID / 답변자 파일
| 약칭 | 모델 ID | | 답변자 | drafts 파일 |
|---|---|---|---|---|
| gpt-oss | `openai/gpt-oss-120b` | | A1 | `data/raw/drafts_phase0_all.jsonl` |
| gemma | `google/gemma-4-31b` | | A2 | `data/raw/drafts_qwen35_think.jsonl` |
| qwen3.6 | `qwen/qwen3.6-27b` | | 셔플 items | 공통 1500의 MedMCQA 400 |
| Qwen3.5 | `qwen3.5-122b-a10b` | | | |

## LLM0 사전 준비 (아무 때나, 모델 무관)
```
# 원본-reference (sanity용). gptoss는 이미 생성됨. qwen35는 A2 drafts 완료 후.
.venv/bin/python experiments/phase2_shuffle_reference.py --tag gptoss --answerer-drafts data/raw/drafts_phase0_all.jsonl --n 400
.venv/bin/python experiments/phase2_shuffle_reference.py --tag qwen35 --answerer-drafts data/raw/drafts_qwen35_think.jsonl --n 400
```

---

## Phase 1 — 답변자 셔플 재답변 (셔플셋 준비)

### 세션 1 · **Qwen3.5-T** (현재 로드)
| 작업 | 명령 |
|---|---|
| 매트릭스 Qwen3.5-T→A1 | `VERIFIER_MODEL=qwen3.5-122b-a10b … cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_q35T_gptoss.jsonl --reload-every 100 --reload-context 22272 --reload-parallel 4` |
| 셔플 재답변(Qwen3.5) | `ANSWERER_MODEL=qwen3.5-122b-a10b … shuffle_answer.py --tag qwen35 --n 400 --max-tokens 16000 --reload-every 100 --reload-context 22272 --reload-parallel 4` |

### 세션 2 · **gpt-oss-T**
| 작업 | 명령 |
|---|---|
| 매트릭스 gpt-oss-T→A2 [음의앵커] | `VERIFIER_MODEL=openai/gpt-oss-120b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gptT_q35.jsonl --reload-every 100 --reload-parallel 4` |
| 셔플 재답변(gpt-oss) | `ANSWERER_MODEL=openai/gpt-oss-120b … shuffle_answer.py --tag gptoss --n 400 --max-tokens 2048 --reload-every 100 --reload-parallel 4` |
| 셔플 판정 gpt-oss-T→Qwen3.5셔플 | `VERIFIER_MODEL=openai/gpt-oss-120b … shuffle_judge.py --answerer qwen35 --tag gpt_T --max-tokens 2048` |
| sanity(100) | `VERIFIER_MODEL=openai/gpt-oss-120b … shuffle_judge.py --answerer qwen35_ref --tag gpt_T --limit 100 --max-tokens 2048` |

*→ 이제 gpt-oss셔플·Qwen3.5셔플 둘 다 준비됨.*

---

## Phase 2 — verifier 판정 (매트릭스 + 셔플 orig-sanity + 셔플)

### 세션 3 · **gpt-oss-N** (템플릿 no-think 토글 후 로드)
| 작업 | 명령 |
|---|---|
| 매트릭스 gpt-oss-N→A2 | `VERIFIER_MODEL=openai/gpt-oss-120b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gptN_q35.jsonl --reload-every 100 --reload-parallel 4` |
| 셔플 판정 gpt-oss-N→Qwen3.5셔플 | `… shuffle_judge.py --answerer qwen35 --tag gpt_N --max-tokens 1024` |
| sanity(100) | `… shuffle_judge.py --answerer qwen35_ref --tag gpt_N --limit 100 --max-tokens 1024` |

### 세션 4 · **gemma-T**
| 작업 | 명령 |
|---|---|
| 매트릭스 gemma-T→A2 | `VERIFIER_MODEL=google/gemma-4-31b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gemT_q35.jsonl --reload-every 100 --reload-parallel 4` |
| 셔플 판정 gemma-T→gpt-oss셔플 | `… shuffle_judge.py --answerer gptoss --tag gem_T --max-tokens 4096` |
| 셔플 판정 gemma-T→Qwen3.5셔플 | `… shuffle_judge.py --answerer qwen35 --tag gem_T --max-tokens 4096` |
| sanity(100)×2 | `… shuffle_judge.py --answerer gptoss_ref --tag gem_T --limit 100 --max-tokens 4096` · `… --answerer qwen35_ref --tag gem_T --limit 100 --max-tokens 4096` |

### 세션 5 · **gemma-N** (no-think 토글)
| 작업 | 명령 |
|---|---|
| 매트릭스 gemma-N→A1 | `… cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_gemN_gptoss.jsonl --reload-every 100 --reload-parallel 4` |
| 매트릭스 gemma-N→A2 | `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gemN_q35.jsonl --reload-every 100 --reload-parallel 4` |
| 셔플 판정 gemma-N→gpt-oss셔플 · Qwen3.5셔플 | `… shuffle_judge.py --answerer gptoss --tag gem_N --max-tokens 1024` · `… --answerer qwen35 --tag gem_N --max-tokens 1024` |
| sanity(100)×2 | `… --answerer gptoss_ref --tag gem_N --limit 100 --max-tokens 1024` · `… --answerer qwen35_ref --tag gem_N --limit 100 --max-tokens 1024` |

### 세션 6 · **qwen3.6-T**
| 작업 | 명령 |
|---|---|
| 매트릭스 qwen3.6-T→A2 | `VERIFIER_MODEL=qwen/qwen3.6-27b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_q36T_q35.jsonl --reload-every 100 --reload-context 13649 --reload-parallel 4` |
| 셔플 판정 →gpt-oss셔플 · Qwen3.5셔플 | `… shuffle_judge.py --answerer gptoss --tag q36_T --max-tokens 4096` · `… --answerer qwen35 --tag q36_T --max-tokens 4096` |
| sanity(100)×2 | `… --answerer gptoss_ref --tag q36_T --limit 100 --max-tokens 4096` · `… --answerer qwen35_ref --tag q36_T --limit 100 --max-tokens 4096` |

### 세션 7 · **qwen3.6-N** (no-think 토글)
| 작업 | 명령 |
|---|---|
| 매트릭스 qwen3.6-N→A1 | `… cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_q36N_gptoss.jsonl --reload-every 100 --reload-context 13649 --reload-parallel 4` |
| 매트릭스 qwen3.6-N→A2 | `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_q36N_q35.jsonl --reload-every 100 --reload-context 13649 --reload-parallel 4` |
| 셔플 판정 →gpt-oss셔플 · Qwen3.5셔플 | `… shuffle_judge.py --answerer gptoss --tag q36_N --max-tokens 1024` · `… --answerer qwen35 --tag q36_N --max-tokens 1024` |
| sanity(100)×2 | `… --answerer gptoss_ref --tag q36_N --limit 100` · `… --answerer qwen35_ref --tag q36_N --limit 100` |

### 세션 8 · **Qwen3.5-T** (재방문, think)
| 작업 | 명령 |
|---|---|
| 셔플 판정 Qwen3.5-T→gpt-oss셔플 | `VERIFIER_MODEL=qwen3.5-122b-a10b … shuffle_judge.py --answerer gptoss --tag q35_T --max-tokens 16000` |
| sanity(100) | `… shuffle_judge.py --answerer gptoss_ref --tag q35_T --limit 100 --max-tokens 16000` |

### 세션 9 · **Qwen3.5-N** (no-think 토글)
| 작업 | 명령 |
|---|---|
| (매트릭스 A1 = B-1b 완료, 생략) | — |
| 셔플 판정 Qwen3.5-N→gpt-oss셔플 | `VERIFIER_MODEL=qwen3.5-122b-a10b … shuffle_judge.py --answerer gptoss --tag q35_N --max-tokens 1024` |
| sanity(100) | `… shuffle_judge.py --answerer gptoss_ref --tag q35_N --limit 100 --max-tokens 1024` |

---

## Phase 3 — 분석 (LLM 0, 스왑 없음)
```
# sanity 통과/확장 판정 (셀별, 예시 1개)
.venv/bin/python experiments/phase2_shuffle_sanity.py --cell gptoss__gem_T \
  --rejudge data/raw/shuffle_judge_gptoss_ref__gem_T.jsonl \
  --existing-risk data/raw/verifier_cross.jsonl \
  --v-selfanswer data/raw/selfanswer_gemma.jsonl \
  --answerer-drafts data/raw/drafts_phase0_all.jsonl
# 이후: A9 lift≈f(Δ)(within-dataset) · 4모델 추론 ablation 표 · A8 전셀 정보이론 · 셔플 원본vs셔플 강건성
```

## 완료 파일 체크리스트 (매트릭스 9)
- [완료] verifier_cross(gemT→A1) · verifier_qwen27(q36T→A1) · verifier_qwen35_of_gptoss(q35N→A1=B-1b)
- [신규] verifier_q35T_gptoss · verifier_gptT_q35 · verifier_gptN_q35 · verifier_gemT_q35 · verifier_gemN_gptoss · verifier_gemN_q35 · verifier_q36T_q35 · verifier_q36N_gptoss · verifier_q36N_q35
