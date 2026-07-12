# UASEF 실험 마스터 문서 (단일 정본)
> **이 문서 하나로 실험 전체가 재구성 가능하도록 작성됨.** (설계·데이터·명령·결과·분석규칙·일정)
> 작성: 2026-07-10 · **상태: 실험 프로토콜·분석 계획·데이터 스키마·실행 순서 = 동결(FROZEN). 남은 confirmation cell 데이터 수집 진행 중** (세션1 매트릭스 실행중). "실험 완료" 표현 금지.
> **동결 범위**: 새 모델·새 데이터셋·ensemble·self-verification 셀 추가 = 하지 않음.
> 관련 정본: 분석규칙=`improvements/analysis_plan.md` · 실행표=`improvements/swap_table.md` · 파일→조건 매핑=`analysis/manifest.py`

---

## 1. 논문 한 장 요약

**제목(working)**: *What Does a Cross-Model Escalation Signal Measure?*
— *Reasoning-Mediated, Item-Level Competence-Gated Disagreement*
(대안: *When Do LLM Verifiers Help? Item-Level Competence Gating, Reasoning Modes, and Robust Escalation*)

**연구 질문**: 한 LLM의 답을 다른 LLM(cross-model verifier)이나 그 자신의 self-confidence로 "틀릴 위험"을 매기는 escalation 신호가 **실제로 무엇을 재는가?**

**4대 주장** (수위 확정):
1. **완료된 thinking-verifier 조건에서** cross-model verifier는 self-confidence 너머의 오류 예측 개선을 보였고, **평균 능력격차 Δ만으로는 그 lift를 충분히 설명할 수 없으며, verifier의 문항수준 competence(Z, 그리고 배포가능 프록시 q)가 신호의 유용성을 강하게 조절**한다 (verifier가 맞힌 곳 AUROC 0.98, 틀린 곳 0.59–0.61; 평균 Δ는 배경 변수). 전 matrix 일관 재현 시 일반화 표현으로 승격.
2. **answerer–verifier 불일치는 escalation 신호의 주요 carrier이지만, 그 정보성은 verifier competence로 게이트**된다 — 불일치만으로는 불충분, 불일치+competence가 정보성 (정보이론 MI·조건부MI·κ로 정량화; 불일치 시 오류율 0.81–0.84). V 전체를 단순 불일치 지표로 환원한다고 주장하지 않음.
3. **[잠정, 전 셀 완료 전]** no-thinking 실행에서 verifier 신호가 붕괴함을 **시사** — 현재 확정은 Qwen3.5 쌍뿐 (T lift +0.102/+0.031 vs N −0.039/−0.084). gemma-N·qwen3.6-N 완료 후 **3모델(Qwen3.5·gemma·qwen3.6) 일관** 시 "reasoning-mediated" headline로 승격, 일부만이면 "model-dependent reasoning-mode effect"로 하향. gpt-oss는 no-think 불가 → **effort ablation(low vs high, 보조)** 로 대체.
4. **conformal 게이트** Pr(release|wrong) ≤ α: 이론 보장은 **단일 split conformal + exchangeability** 조건에서 성립. 실험이 보이는 것은 **평가된 exchangeable split들에서 관측 누출률이 목표 이하**라는 경험적 제어(및 반복 split 강건성)이며, shift transfer는 보장이 아닌 stress test.

**포지셔닝 (임상 축소, 고정)**: 임상 안전 논문이 아님. 통제 대상 = 벤치마크 정의 오류(MCQ 정오)뿐. 의료 QA는 high-stakes 테스트베드. verifier = 위험 **feature**이지 ground truth 아님. 임상 적용 = future work.

**A9 명칭 (하향 고정)**: ~~"ability-gap law"~~ → **"평균 Δ 불충분성 분석 + negative-Δ stress test"** (검정: ① 평균 능력격차가 lift를 충분히 설명하는가 ② negative-Δ에서도 신호가 남는가). 셀 산점도는 descriptive.

---

## 2. 용어·기호 (전 결과물 공통)

| 기호 | 정의 |
|---|---|
| **Y** | answerer 오류 (1 = 벤치마크 gold 기준 오답) |
| **C** | answerer self-confidence 기반 위험 (= 1 − verbalized confidence) |
| **V** | verifier가 answerer 답을 보고 매긴 위험 (0–100 → 0–1) |
| **S** | self-verification: **같은 모델**이 자기 답을 보고 매긴 위험 (대각 baseline) |
| **Z** | verifier가 같은 문항을 **직접 풀었을 때** 정답 여부 (self-answer 기반; gold 사용 = 메커니즘 진단용) |
| **q** | Z의 배포가능 프록시 = P(Z=1 \| verifier 자체 불확실성 feature), cross-fitted. **배포 추론 시 gold 미사용이지만, 학습 target인 Z 라벨은 gold로 정의됨** (inference-time gold-free, not training-free) |
| **Δ** | acc(verifier self-answer) − acc(answerer), **within-dataset·mode별** 계산 |
| **lift** | AUROC(V→Y) − AUROC(C→Y), within-dataset |
| **T / N** | thinking / no-thinking (LM Studio Jinja 템플릿 토글, 동일 모델) — Qwen3.5·gemma·qwen3.6에만 존재 |
| **low / high** | **gpt-oss 전용 mode 축**: gpt-oss-120b는 no-think 불가(Jinja 토글 없음, reasoning effort low/med/high만) → T/N 대신 **effort ablation(보조 축)**. 세션2까지의 전 gpt-oss 데이터 = **low** (운영자 확인: 설치 후 변경 없음). 순수 T/N 토글 주장은 3모델만 |

**verification_type 3버킷 (절대 혼합 금지)**:
- `cross` — 다른 모델이 답 보고 판정 (**핵심 결과**)
- `self_verification` — 같은 모델이 자기 답 보고 판정 (**대각 baseline, core 아님**)
- `self_answer` — 답 안 보고 직접 풂 (**Z·q·Δ의 원천**)

**필수 키 (모든 결과 행)**: `answerer_model, answerer_mode, verifier_model, verifier_mode, dataset, split(=matrix/original400/shuffled400), verification_type, item_id`
**결측 역할 규칙 (고정)**: 해당 행 유형에 없는 역할은 **`NA`** (빈 문자열 금지) — `self_answer` 행: answerer_model/mode=**NA** (verifier만 실측) · `answerer` 행: verifier_model/mode=**NA**. **verifier를 answerer 자리에 복제 금지** (self_verification과 혼동됨). consolidation이 자동 적용.

**통계 기호 정의 (논문 표기 고정)**:
| 기호 | 정확한 정의 |
|---|---|
| **φ** (=0.535) | answerer 오류 지시자와 verifier(자기답) 오류 지시자 사이의 **phi coefficient** (2×2 이항 상관) |
| **κ** (=0.714) | verifier의 자기답 선택지와 answerer 선택지 사이의 **Cohen's kappa** (답 일치도) |
| **증분 정보 비율** (=0.46) | **I(V;Y\|C) / I(V;Y)** — "조건부 정보가 verifier 주변 정보의 46%를 보존" (막연한 "+46%" 표기 금지) |
| **τ-instability range** | 평가된 4개 transfer(within-mc, within-pm, mc→pm, pm→mc)의 test-split Pr(release\|error) **최대−최소 범위** |

---

## 3. 모델 레지스트리 (정본 — 논문·CSV·파일명 전부 이 문자열)

> 실측 확정(`lms ps` + .env). 옛 .tex의 `google/gemma-4-31b-it`, `Qwen3.5-122B-A10B`는 **틀림** → 교정 필요.
> 코드 정본: `analysis/manifest.py`의 `MODELS`.

| alias | **model_id (= LM Studio 런타임 문자열)** | 역할 | 비고 (런타임/컨텍스트) |
|---|---|---|---|
| gpt-oss | `openai/gpt-oss-120b` | A1 답변자(T 고정) + verifier T/N | GGUF MXFP4, llama.cpp |
| Qwen3.5 | `qwen3.5-122b-a10b` | A2 답변자(T; N=ablation) + verifier T/N | GGUF Q3_K_S, ctx 22272, heavy thinker |
| gemma | `google/gemma-4-31b` | verifier T/N | GGUF Q4_K_M |
| qwen3.6 | `qwen/qwen3.6-27b` | verifier T/N | MLX 4-bit, ctx 13649 |

공통 인프라: LM Studio 로컬 서버(:1234, OpenAI 호환), `--parallel 4`, Flash Attention ON, F16 KV cache. `UASEF_QUERY_TIMEOUT_S=400~600`. 전 실행 `--reload-every 100`(동일 모델 unload→reload, Apple Silicon 장기실행 슬로다운 리셋). cross_verifier는 전부 `--verifier-max-tokens 16000`(Qwen3.5 사고절단 방지; 타 모델은 조기종료라 무해). **외부 API $0, 전량 로컬.**

---

## 4. 데이터셋·아이템·보안

- 원천: **MedMCQA 3000 + PubMedQA 800** (`data/qa_datasets.py`, seed 42) → gpt-oss 답변 3800개 = `drafts_phase0_all.jsonl`.
- **canonical 1500** = `Random(0).shuffle(전체3800)[:1500]` = **MedMCQA 1183 + PubMedQA 317**. 모든 매트릭스 판정·self-answer·A2 생성이 **이 동일 1500 문항** (A1·A2 문항 완전 일치 검증됨 → reciprocal 설계 성립).
- **셔플 400** = canonical 1500 중 MedMCQA 첫 400 (canonical 1500의 부분집합 → fold 상속).
- 정오 판정: 셔플에서는 **canonical TEXT 기준** (`correct_by_text`; 라벨 재배치 오류 방지), 매트릭스에서는 decision_answer vs gold.
- **본 논문의 데이터 범위 (논문용 서술)**: 본 논문 실험은 **MedMCQA·PubMedQA만** 사용 — 환자 수준 임상기록(MIMIC-IV/eICU)은 **본 논문 실험에 사용되지 않음**. 전 입출력 로컬 처리, 외부 추론 API 미사용·미전송. ("PHI 송출 0바이트" 같은 표현은 논문에서 실제 PHI를 다룬 것처럼 오해될 수 있으므로 사용 금지.)
- **레포 운영 규칙 (논문 범위 밖, 변경 불가)**: 같은 레포의 과거 라운드가 PhysioNet credentialed 데이터(MIMIC-IV/eICU)를 포함 → 원자료 재배포 금지, `UASEF_BACKEND_NEVER_SEND_PHI=1` 유지, raw 캐시(`data/raw/`)·`results/`·`paper/` gitignored — 코드·문서만 추적.

---

## 5. 실험 설계 (셀 전체)

### 5.1 매트릭스 (cross, 1500 판정/셀) — 14셀 (A1 6 + A2 6 + A2-N 2)
**A1 = gpt-oss-T 답변 평가** (6셀):
| verifier | mode | 파일 | 상태 | 필요도 |
|---|---|---|---|---|
| gemma | T | verifier_cross | ✅ 1500 | 필수 |
| gemma | N | verifier_gemN_gptoss | 세션5 | ablation 필수 |
| qwen3.6 | T | verifier_qwen27 | ✅ 1500 | 필수 |
| qwen3.6 | N | verifier_q36N_gptoss | 세션7 | ablation 필수 |
| Qwen3.5 | T | verifier_q35T_gptoss | 🔄 세션1 (413+) | 필수 |
| Qwen3.5 | N | verifier_qwen35_of_gptoss | ✅ 1500 (=B-1b) | 완료 |

**A2 = Qwen3.5-T 답변 평가** (6셀):
| verifier | mode | 파일 | 상태 | 필요도 |
|---|---|---|---|---|
| gpt-oss | T | verifier_gptT_q35 | 세션2 (**294 cached** — 구 B-2 부분실행 rename) | **negative-Δ anchor 필수** |
| gpt-oss | **high** (N 불가) | verifier_gptHigh_q35 | 세션3 | **within-model effort intervention** (보조 F5; 규칙=analysis_plan §13 — 해석 4경우·Z-전이 4그룹 사전 고정, judgment 쪽만 순수 비교) |
| gemma | T | verifier_gemT_q35 | 세션4 | 필수 |
| gemma | N | verifier_gemN_q35 | 세션5 | ablation 필수 |
| qwen3.6 | T | verifier_q36T_q35 | 세션6 | 필수 |
| qwen3.6 | N | verifier_q36N_q35 | 세션7 | ablation 필수 |

**A2-N = Qwen3.5-N 답변 평가 (answerer T/N ablation, B-lite)** (2셀):
| verifier | 파일 | 세션 |
|---|---|---|
| gemma-T | verifier_gemT_q35N | 4 |
| qwen3.6-T | verifier_q36T_q35N | 6 |

A2-N 답변 = `drafts_qwen35_nothink` **재사용** (A2-T와 동일 파이프라인·동일 1500 검증됨). verifier 고정, answerer mode만 변화 → answerer ablation 성립. A1은 T 고정. **일반화 범위 (고정)**: "limited answerer-mode ablation on Qwen3.5 under two fixed thinking verifiers" — 이 이상 일반화 금지.

### 5.2 self-verification 대각 (baseline, core 아님) — 2셀만
| 셀 | 파일 | 세션 |
|---|---|---|
| gpt-oss(low) 답 → gpt-oss(low) = **same-model, matched-effort baseline** | verifier_gptT_gptoss | 2 |
| Qwen3.5-T 답 → Qwen3.5-T | verifier_q35T_q35 | 8 |

목적: "같은 모델에 재검토시키면 안 되나?" 방어. 분석 = **C+S+V vs C+S** (S를 알고도 V가 증분 정보를 주는가) — 단 **V는 verifier별 개별 평가로 고정**: A1행 = C+S+V_gemma / C+S+V_qwen3.6 / C+S+V_Qwen3.5 각각 vs C+S; A2행 = V_gptoss/V_gemma/V_qwen3.6 각각. **여러 V 조합·사후 best-verifier 선택 금지** (no-ensemble 방침과 일치). 셔플/전체 대각 확장·ensemble = **안 함**.

### 5.3 verifier self-answer (Z/q/Δ) — 8조건 전원 확보
| verifier | T | N |
|---|---|---|
| gpt-oss (low/high) | low = **A1 drafts 재사용** (`reused_answerer_draft`) | high = selfanswer_gptossHigh (세션3) |
| Qwen3.5 | drafts_qwen35_think 재사용 | drafts_qwen35_nothink 재사용 |
| gemma | selfanswer_gemma ✅ (feature 재생성=세션4) | selfanswer_gemmaN (세션5) |
| qwen3.6 | selfanswer_qwen27 ✅ (feature 재생성=세션6) | selfanswer_qwen27N (세션7) |

→ 전 셀에서 Z-gating·q·Δ 가능. T/N 격차의 메커니즘 분리: "N은 문제를 못 풀어서(능력) vs 풀 수 있는데 판정 출력만 나빠서(표현)".
⚠️ self-answer(답 안 보고 풂) ≠ self-verification(답 보고 평가) — 절대 혼합 금지.

### 5.4 셔플 audit (Option B, clean paired) — 12 판정셀
- 답변자 생성: Qwen3.5-T·gpt-oss-T 각각 **셔플400 + 원본400(`--no-shuffle`)** — **옵션 순서(와 그에 따른 positional representation)만 의도적으로 조작**하고 prompt·decoding·parser·scoring 조건은 고정하여 **비의도적 confound를 최소화** ("confound 0" 단정 금지). temp=0로 **평가된 로컬 배포 환경에서 사실상 결정적**(effectively deterministic; 중복 행 동일 출력으로 실증 — bitwise determinism 일반 보장은 주장 안 함).
- 판정 구조: A1(gpt-oss-T 답) ← gemma-T/N·qwen3.6-T/N·Qwen3.5-T/N = 6조건, A2(Qwen3.5-T 답) ← gpt-oss-T/N·gemma-T/N·qwen3.6-T/N = 6조건 → **12 (answerer×verifier-mode) 셀 × 원본/셔플 2 × 400문항 = 9,600 판정**.
- 비교는 전부 **paired** (shuffle_group = canonical item): 정확도 McNemar, lift는 group bootstrap.
- 기대 패턴: verifier-T 유지 vs verifier-N 붕괴 = reasoning-mediated 독립 증거.

---

## 6. 데이터 파일 인벤토리

### 6.1 활성 (스키마 3종)
**답변자/self-answer drafts** (`drafts_*`): `item_id, dataset, subject, decision_answer, samples, token_logprobs, verbalized_confidence, reasoning_text, gold_answer`
**매트릭스 판정** (`verifier_*`): 구형 5필드 `item_id, verifier_risk, vtext, gpt_oss_conf, error` → **세션3부터 신형**: + `answerer_conf, dataset, parser_ok, empty, truncated(finish_reason), output_length, vtext=전체`. (⚠️ `gpt_oss_conf` 필드명은 legacy — 실제 값은 **해당 셀 answerer의 confidence**. consolidation이 `answerer_conf`로 정규화.)
**셔플 답변** (`shuffle_answer_*`): 16필드 (original/shuffled_options, permutation_map, gold 양표기, canonical text 정오 등)
**셔플 판정** (`shuffle_judge_<answerer>__<tag>`): verifier_pred label+text, risk, agreement_by_text/label, **judge_selected_correct**(≠Z — answerer 답을 **본 후** 선택한 답의 정오; confirmation-bias 오염 가능), error(text기준). **독립 Z는 self-answer 파일에서만** — 셔플 행에 join할 수는 있으나 그 Z는 "원본 옵션 순서에서 측정된 competence"이며 shuffle-specific competence라고 주장 금지. 셔플 core 분석(T/N별 lift·paired robustness)에 Z 불필요.
**self-answer** (`selfanswer_*`): 구형 3필드(gemma·qwen27, 세션4·6 feature 재생성 예정) / 신형: + verbalized_confidence, neg_logprob_mean, reasoning_len, parser_ok, self_consistency_disagree, samples

### 6.2 구스키마 파일 처리 (완전 구형 4 + 부분 구형 2; 유도로 수용, 재생성 안 함)
verifier_cross·verifier_qwen27·verifier_qwen35_of_gptoss·verifier_gptT_q35(294) 전체 + verifier_q35T_gptoss 초기분:
dataset←item_id, parser_ok←risk≠None, empty←risk/vtext (전부 무손실) / **truncated=unknown(0 아님)** / output_length는 vtext 300자 우측검열(52행) → core 분석 미사용 항목, failure appendix 각주.

### 6.3 Deprecated (사용 금지)
`drafts_qwen35.jsonl`(구 A2 생성분; 정본=drafts_qwen35_think) · `_drafts_qwen35_thinkmode_discard.jsonl` · `shuffle_answer_gptoss_ref.jsonl`(구 LLM0 reference; --no-shuffle로 대체) · phase2_shuffle_reference.py·phase2_shuffle_sanity.py(Option B로 폐기)

---

## 7. 실행 런북 (9 세션) — 스왑만 수동, 세션 내 전 작업 자동

공통: `cd /Users/idaun/PoC/UASEF && export UASEF_QUERY_TIMEOUT_S=400` · 생략형 `.venv/bin/python experiments/<script>.py` · think/no-think = Jinja 토글 후 로드 · 전 명령 `--reload-every 100` · cross_verifier 전부 `--verifier-max-tokens 16000`
(전체 명령 원문은 `improvements/swap_table.md` — 아래는 작업+시간 요약)

| 세션 | 로드 | 작업 | 예상 |
|---|---|---|---|
| **1** 🔄 | Qwen3.5-T | 셔플400✅+원본400✅ → 매트릭스→A1 (413+/1500 진행중) | ~17.5h |
| **2** | gpt-oss-T | 매트릭스→A2[negative-Δ, **294 cached**] + ⬦self-verif→gpt-oss + 셔플/원본 재답변 + 판정→Qwen3.5(셔+원) | ~26h |
| **3** | gpt-oss-**high** (effort; N 불가) | 매트릭스→A2 + ★self-answer(gptossHigh) + 판정→Qwen3.5(셔+원) | ~3-6h (high라 low보다 느림) |
| **4** | gemma-T | 매트릭스→A2 + **→A2-N(ablation)** + self-answer(feature) + 판정×4 | ~54h |
| **5** | gemma-N | 매트릭스→A1,A2 + ★self-answer(gemmaN) + 판정×4 | ~30h |
| **6** | qwen3.6-T | 매트릭스→A2 + **→A2-N(ablation)** + self-answer(feature) + 판정×4 | ~54h |
| **7** | qwen3.6-N | 매트릭스→A1,A2 + ★self-answer(qwen27N) + 판정×4 | ~30h |
| **8** | Qwen3.5-T | ⬦self-verif→Qwen3.5 + 판정→gpt-oss(셔+원) | ~27h |
| **9** | Qwen3.5-N | 판정→gpt-oss(셔+원) (매트릭스 A1=B-1b 완료) | ~4h |

**합계: 순수 연산 ~262h ≈ 11일** → 스왑 대기·재시작 오버헤드 포함 **현실 ~2주**.
운영 원칙: **단일 잡**(reload가 `lms unload --all`이라 동시 실행 금지) · `python -u` + PYTHONUNBUFFERED · 전 스크립트 resumable(append+item_id skip) · self-answer는 해당 매트릭스 **후** 실행(item set·risk 공급).

---

## 8. 확정된 결과 (수집 완료분, results/*.json에서 전사)

### 8.1 기반 (P0–P2, n=3800)
| 실험 | 결과 |
|---|---|
| P0 gatekeeper | CV AUROC 0.841 (mc 0.813 / pm 0.73), 최강 단일 feature = verbalized_uncertainty 0.832 → **GO** |
| P1 conformal | α=0.1: Pr(release\|wrong)=**0.0767**, release 37.9% / α=0.05: **0.0442**, release 27.2% — **평가된 exchangeable split에서 관측 누출률 목표 이하** (200-split 평균 0.0847/0.0333). 이론 보장은 단일 split 절차에 대한 것; 반복/cross-fitted 결과는 경험적 강건성 |
| P2 self-audit | **confidence_dominance FLAGGED**: conf-only가 full risk의 **96.3%** 복원 (0.829/0.842); conf 제외 시 0.694 |

### 8.2 cross-verifier 코어 (A0·B1·B1'·A6·A8, answerer=gpt-oss-T)
| 셀 | 핵심 수치 |
|---|---|
| gemma-T (n=1495) | within-ds lift: mc **+0.071**, pm +0.035[CI −0.006,+0.077 borderline]. 분해: 맞힌곳 **0.982** / 틀린곳 **0.612** (능력지배). self-acc 0.816 vs answerer 0.712, φ=0.535 |
| qwen3.6-T (n=1498) | lift: mc **+0.050**, pm +0.040[borderline]. 분해 **0.981/0.593** — B1 판박이 재현 (가문 무관) |
| A6 게이트 증강 | AUROC 0.824→**0.895**; 공개율 α=0.1: 36.2→43.2% (**+7.0%p**), α=0.05: +6.1%p — **평가된 split에서 관측 오답-공개 누출률은 목표 수준 이하 유지** ("보장 유지" 표현 금지) |
| A8 정보이론 (gemma) | I(V;Y)=0.348b, **I(V;Y\|C)=0.161b** (+46% 증분), κ=0.714, **P(err\|불일치)=0.839** vs P(err\|일치)=0.111. 층화: 맞힌곳 I=0.446b / 틀린곳 I=0.027b |
| A8 (qwen3.6) | I(V;Y\|C)=0.149b(+45%), P(err\|불일치)=0.814. Δ=0.090, lift=0.038 |
| B-1b (Qwen3.5-**N** verifier) | 전체 AUROC 0.77 vs C 0.837 — lift **−0.031(mc)/−0.084(pm)** → **추론 끄면 self-conf보다 나쁨** |

### 8.3 reviewer 방어 실험 — 각 **initial evaluated condition 1셀** (전셀 재현 후 "replicated across answerers, verifiers, and datasets"로 승격; 그 전까지 일반화 금지)
| Exp | 셀 | 결과 |
|---|---|---|
| **Exp1** calibration 통제 (cross-fitted·held-out) | gptoss×gemma-T | M1(pC+pV) NLL 0.3375 → M2(+Z) **0.2562**; ΔNLL **+0.0813 CI[0.0524,0.108]** (0 제외); +Z×D 추가 ΔNLL +0.164 → **calibration/sharpness로 환원 안 됨** |
| **Exp2** competence proxy (cross-fitted q) | qwen35N×gptoss | q→Z AUROC **0.821**; lift low_q −0.046 / high_q **+0.024**; 상호작용 pV×q **+2.318** → **추론 시 gold 없이** 신호 조절 가능 (학습 라벨 Z는 gold 유래) |
| **Exp3** threshold transfer | gptoss×gemma-T | τ-불안정(Pr(rel\|err) range): C 0.599 / V 0.490 / **C+V 0.441 (최안정)**; cross-ds 열화 정직 기재 |
| Exp4 shuffle | — | 스크립트 완성, 셔플 판정 데이터 대기 |

### 8.4 정확도·Δ 기준표 (동일 canonical 1500, 05_accuracy_summary.csv)
| 모델·mode | MedMCQA (1183) | PubMedQA (317) |
|---|---|---|
| gpt-oss-T (A1) | 0.8132 | 0.3502 |
| Qwen3.5-T (A2) | **0.9045** | **0.5994** |
| Qwen3.5-N | 0.8622 | 0.5426 |
| gemma-T | 0.8766 | 0.5804 |
| qwen3.6-T | 0.8673 | 0.5741 |
| Qwen3.5-T 셔플셋: 원본 0.890 / 셔플 0.870 | | |

→ **A2(Qwen3.5-T)가 전 verifier보다 강함** = A2행 전체가 negative-Δ stress test (예: Δ(gemma-T, A2, mc) = 0.8766−0.9045 = **−0.028**; gpt-oss-T→A2 mc **−0.091**). Δ는 반드시 verifier mode별·within-dataset로 계산.

---

## 9. 분석 계획 (전문 = `analysis_plan.md`)

**고정 성격 (정직 표기)**: 완전한 preregistered confirmatory study가 **아님**. **Discovery 셀** = P0–P2, gpt-oss-T×gemma-T, gpt-oss-T×qwen3.6-T, gpt-oss-T×Qwen3.5-N (결과를 본 뒤 계획 확정에 반영됨). **Prospectively locked confirmation 셀** = A2 negative-Δ 행 전체, 남은 T/N 셀, answerer T/N B-lite, self-verification, shuffle audit — 이들의 **완료 전에** 분석 계획이 잠김. 논문 표현: "The analysis plan was prospectively locked before completion of the remaining reciprocal, negative-gap, reasoning-mode, and shuffle conditions." ("모든 분석이 결과 전 preregistered" 표현 금지.)

- **Primary**: ① ΔAUROC(V−C) ② Z=1/Z=0 lift 차 ③ verifier T/N lift 차. **Secondary**: q·CMI·shuffle·self-verif·conformal transfer·answerer T/N.
- **지표**: AUROC(주) + **AUPRC**(class imbalance 보조) + Brier·NLL·ECE. 새 추론 불필요, 분석만 추가.
- **통계**: 행 단위 부트스트랩 금지 — item_id(매트릭스)/shuffle_group(셔플) 단위 paired bootstrap 95% CI, McNemar. 셀 통합은 descriptive + item-cluster.
- **BH-FDR family (사전 정의)**: F1 = V−C lift 검정 전셀 · F2 = Z=1 vs Z=0 검정 전셀 · F3 = T vs N 쌍 검정 (**Qwen3.5·gemma·qwen3.6만**) · F4 = 원본 vs 셔플 검정 · **F5 = gpt-oss effort L/H (별도 보조 family — F3와 pooling 금지)**. **mc·pm 결과는 같은 family 안에 함께 묶어** 보정 (dataset 분리 family 아님).
- **Parser failure 정책 (고정)**: ① 예측 지표(주) = 유효 파싱 행의 complete-case (현행 n=1495/1498 = 이 정책) ② 실패율(parser/empty/truncated)은 role·mode별 별도 보고(17번 CSV) ③ **운영 게이트에서는 parser failure/empty = 자동 escalation(최대 위험 처리)** ④ 민감도 분석: 실패 행을 전부 오류 취급 시 release 성능 재계산.
- **Split**: `00_split_manifest.csv` (item-grouped·dataset-stratified 5-fold, mc 237/pm 63-64씩)를 calibration·q·τ·평가 전부 공유, cross-fitting.
- **Exp1**: fold별 isotonic → held-out M0(pC)/M1(+pV)/M2(+Z)/M3(+Z×D) NLL/Brier/AUROC + item-bootstrap CI. ECE/sharpness는 셀 수준 descriptive만.
- **CMI(A8)**: 보조 결과. 고정 10-quantile binning, bin 민감도, permutation null, bootstrap CI 의무.
- **파이프라인**: raw JSONL → `analysis/manifest.py`(파일명→조건 결정적 매핑) → `analysis/consolidate.py`(01–05 장부) → 분석 스크립트(06–18).

### 최종 산출 CSV — 산출물 18종 (01–18) + 공통 입력 1종 (00)
`00_split_manifest`(공통 fold 입력) `01_items_master` `02_answerer_outputs` `03_verifier_judgments` `04_verifier_self_answers` `05_accuracy_summary` (✅ 생성기 완성·검증) / `06_main_auroc_lift` `07_z_gating` `08_delta_lift` `09_reasoning_mode_ablation` `10_calibration_sharpness` `11_nested_model_comparison` `12_q_proxy_prediction` `13_q_gated_verifier` `14_shuffle_audit_item` `15_shuffle_audit_summary` `16_conformal_threshold_transfer` `17_parser_failure_summary` `18_conditional_mutual_information` (데이터 완성 후 생성; 모든 행에 필수 키+mode 포함)

---

## 10. Reviewer 방어 맵

| 예상 질문 | 대응 |
|---|---|
| self-confidence보다 정말 나은가 | A0 within-ds lift + A6 운용 이득 (+7%p) |
| 같은 모델 재검토로 충분하지 않나 | self-verification 대각 2셀, C+S+V vs C+S |
| verifier가 틀린 문제에서도 되나 | B1/B1' 분해 + Z-gating (0.98 vs 0.59–0.61) |
| 그냥 verifier가 강해서 아닌가 | **negative-Δ**: A2행 전체 (Qwen3.5-T가 최강 answerer) |
| calibration 좋아서 아닌가 (Kiyani) | Exp1 — held-out ΔNLL CI 0 제외 |
| 운영에서 Z(gold)를 어떻게 아나 | Exp2 — q proxy (0.821): **추론 시 gold-free** (학습 라벨 Z는 gold 유래임을 명시) |
| threshold가 이전되나 (Bakman) | Exp3 — C+V 최안정 + shift 열화 정직 기재 |
| 옵션 순서 민감성 아닌가 | Exp4 — canonical-text paired shuffle audit |
| thinking 효과 = 모델 정체성 효과 아닌가 | 동일 모델 Jinja 토글 T/N (매트릭스+셔플 대칭) |
| answerer 추론은? | B-lite: A2-T/A2-N을 같은 verifier가 판정 (2셀) |
| 양자화·런타임 confound | 한계 명시: "고정 로컬 배포 조건의 운영적 정보성" 프레임; T/N 비교가 가장 causal |
| 반복 문항으로 p 부풀림 | item-grouped fold + cluster bootstrap (전 분석) |

---

## 11. 한계 (고정 문구 방향)
1. 모델 간 절대 비교에 family·크기·양자화·런타임·프롬프트 confound — 본 연구는 full-precision 본질 순위가 아니라 **고정된 로컬 배포 조건에서 verifier 신호의 운영적 정보성** 분석. 동일 모델 T/N 토글이 가장 causal한 대비.
2. 벤치마크 정오만 통제 (open-ended·임상 검증은 future work).
3. self-verification은 2셀 → 일반론 주장 안 함.
4. 구스키마 4파일: truncated unknown, output_length 52행 우측검열 (failure appendix 각주).
5. PubMedQA n=317 → CI 넓음, borderline 결과는 borderline으로 보고.

---

## 12. 진행 현황 (2026-07-10)

**완료**: P0·P1·P2 / A0(2셀)·B1·B1'·B-1b·A6·A8(2셀) / A2 답변 1500(think·nothink) / 셔플·원본 400+400(Qwen3.5) / selfanswer gemma·qwen27 / Exp1–3 첫 셀 (신 통계로 재검증) / 장부 01–05 + split manifest + stats 유틸 / 분석계획 사전 고정
**진행 중**: 세션1 매트릭스 Qwen3.5-T→A1 (413+/1500, ~42s/item, 단일 잡)
**대기**: 세션2–9 (스왑당 자동 실행) → 06–18 CSV → 논문 Results 8절 재배치( 4.1 self-conf 불완전 → 4.2 cross-verifier 개선 → 4.3 ability-gated → 4.4 ≠calibration → 4.5 q proxy → 4.6 평균Δ 불충분 → 4.7 shuffle=reasoning → 4.8 conformal 운용)
**조건부**: gemma-N/qwen3.6-N이 의미있는 lift 보이면 해석 확장 (self-answer는 이미 세션5·7에 포함)

**무결성 검증 이력**: 중복 0(6dup 사건은 temp0 동일출력 확인 후 무손실 dedup) · identity-셔플 16개 삭제·재생성(강제 non-identity) · 매트릭스 resume 정확성(target 1500 ⊆, 밖 0) · A1↔A2 문항 동일성 · 294행 B-2 부분실행 → 세션2 파일로 rename 재활용.
