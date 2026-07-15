# 스왑 실행 표 (9 세션) — 운영자용 · Option B (400+400 clean paired)

> 각 세션: **모델 로드(수동)** → **provenance 기록**: `.venv/bin/python analysis/run_provenance.py --session <n> --mode <T|N>` → 아래 명령 실행. 공통 앞머리: `cd /Users/idaun/PoC/UASEF && export UASEF_QUERY_TIMEOUT_S=600`
> 생략형 = `.venv/bin/python experiments/<script>.py`.
> **셔플 audit = 원본400 + 셔플400 둘 다 shuffle-judge 프롬프트**(프롬프트·디코딩·파서 고정, 옵션 순서만 조작 → 비의도 confound 최소화; self-contained). 재활용/sanity 없음.
> think/no-think = **Jinja 템플릿 토글 후 로드**(수동). 리로드 CTX: qwen3.5=22272 · qwen3.6=13649 · gpt-oss/gemma=로드값.
> **[표준·고정] 모든 생성/판정/self-answer 명령 = `--max-tokens 16000`(또는 `--verifier-max-tokens 16000`) + `UASEF_QUERY_TIMEOUT_S=600`** (운영자 지시 2026-07-13; 절단→None/빈답 원천 차단). 모든 cross_verifier(매트릭스) 명령에 `--verifier-max-tokens 16000` (Qwen3.5 heavy thinker 절단 방지; gemma/qwen3.6/gpt-oss는 조기종료라 무해). None율 5%→~0.3%.

## 모델 ID 레지스트리 (SINGLE source of truth — 논문·CSV·파일명 전부 이 문자열)
> ⚠️ 실측 확정(`lms ps` + .env): 아래 `model_id`가 **정확한 LM Studio 런타임 문자열**. 옛 .tex의
> `google/gemma-4-31b-it`·`Qwen3.5-122B-A10B`는 **틀림** → 논문을 이 문자열로 교정할 것.
> `analysis/manifest.py`의 `MODELS`가 동일 레지스트리(정본).

| 약칭(alias) | **model_id (= runtime_name, 정본)** | 답변자 | drafts |
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
- 매트릭스 gpt-oss-T→A2 [음의앵커]: `VERIFIER_MODEL=openai/gpt-oss-120b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gptT_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4` (**전량 1500 판정** — ⚠️ 과거 rename한 294 캐시는 구 `drafts_qwen35`(deprecated) 판정이라 오염 → 삭제·재판정 완료. 재활용 시 파일명 아닌 error/conf 지문으로 draft 버전 검증할 것). **결과: within-ds lift mc −0.073 / pm −0.012 (negative-Δ 확인)**
- **[baseline] self-verification 대각 (gpt-oss-T→gpt-oss 자기답, matrix만)**: `VERIFIER_MODEL=openai/gpt-oss-120b … cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_gptT_gptoss.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4` (reviewer 방어용 대조군, **core 아님**; C vs self-verif vs cross-model 비교; 셔플엔 불필요)
- **[self-answer Z/q] gpt-oss-T self-answer = A1 drafts(`drafts_phase0_all`) 재사용** — 별도 생성 불필요. A1은 gpt-oss가 답을 안 보고 같은 1500을 직접 푼 결과(reasoning_text·conf·samples·logprob 포함)라 gpt-oss-T self-answer와 동일. manifest DUAL_SELFANS가 자동 이중역할 처리 → gpt-oss-T→A2 셀의 **Δ·Z-gating·q 완비**. (⚠️ A1 mode = gpt-oss 기본 thinking = T)
- 셔플 재답변: `ANSWERER_MODEL=openai/gpt-oss-120b … shuffle_answer.py --tag gptoss --n 400 --max-tokens 16000`
- 원본 재답변(--no-shuffle): `ANSWERER_MODEL=openai/gpt-oss-120b … shuffle_answer.py --tag gptoss_orig --no-shuffle --n 400 --max-tokens 16000`
- 셔플판정 gpt-oss-T→Qwen3.5: `VERIFIER_MODEL=openai/gpt-oss-120b … shuffle_judge.py --answerer qwen35 --tag gpt_T --max-tokens 16000`
- 원본판정 gpt-oss-T→Qwen3.5원본: `… shuffle_judge.py --answerer qwen35_orig --tag gpt_T --max-tokens 16000`

*→ gpt-oss셔플·Qwen3.5셔플 둘 다 준비됨.*

---

## Phase 2 — verifier 판정 (매트릭스 + 셔플400 + 원본400)

### 세션 3 · **gpt-oss-high** (reasoning effort HIGH — ⚠️ gpt-oss는 no-think 불가)
> gpt-oss-120b는 Jinja think 토글이 없고 reasoning effort(low/medium/high)만 있음 → **T/N 대신 effort ablation(low vs high, 보조 축)**.
> 지금까지의 전 gpt-oss 데이터 = **effort low** (운영자 확인: 설치 후 변경 없음). 순수 T/N 토글 주장은 Qwen3.5·gemma·qwen3.6 3모델만.
- 매트릭스 gpt-oss-high→A2: `VERIFIER_MODEL=openai/gpt-oss-120b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gptHigh_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4` (**controlled within-model effort intervention** — 고정 질문·해석 4경우·Z-전이 4그룹 = analysis_plan §13; "negative-Δ recovery" 단정 금지, high의 Δ 부호는 self-answer 후 확정)
- **[Exp2/B1] self-answer (gpt-oss-high)**: `SELFANSWER_MODEL=openai/gpt-oss-120b … selfanswer.py --tag gptossHigh --verifier-file data/raw/verifier_gptHigh_q35.jsonl --max-tokens 16000 --reload-every 100 --reload-parallel 4` (**매트릭스 후 실행**; high effort의 Z/q — low는 A1 drafts 재사용)
- 셔플판정: `… shuffle_judge.py --answerer qwen35 --tag gpt_high --max-tokens 16000`
- 원본판정: `… shuffle_judge.py --answerer qwen35_orig --tag gpt_high --max-tokens 16000`

### 세션 4 · **gemma-T**
- 매트릭스 gemma-T→A2: `VERIFIER_MODEL=google/gemma-4-31b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gemT_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4`
- **[answerer T/N ablation] 매트릭스 gemma-T→A2-N**: `… cross_verifier.py --drafts data/raw/drafts_qwen35_nothink.jsonl --n 1500 --out data/raw/verifier_gemT_q35N.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4` (A2-N = drafts_qwen35_nothink 재사용 — A2-T와 동일 파이프라인·동일 1500 검증됨. 같은 verifier가 같은 문항의 T/N 답변 판정 → answerer ablation 성립)
- **[None repair] verifier_cross 5개 복구**: `VERIFIER_MODEL=google/gemma-4-31b … cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_cross.jsonl --repair --verifier-max-tokens 16000`
- **[Exp2] self-answer + features 재생성**: **먼저 구파일 아카이브** `mv data/raw/selfanswer_gemma.jsonl data/raw/_selfanswer_gemma_legacy3field.jsonl` (안 하면 resume이 전부 skip!) → `SELFANSWER_MODEL=google/gemma-4-31b … selfanswer.py --tag gemma --verifier-file data/raw/verifier_cross.jsonl --max-tokens 16000 --reload-every 100 --reload-parallel 4` (feature 저장 + 빈답 20 자연 해소)
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag gem_T --max-tokens 16000` · `… --answerer qwen35 --tag gem_T --max-tokens 16000`
- 원본판정×2: `… shuffle_judge.py --answerer gptoss_orig --tag gem_T --max-tokens 16000` · `… --answerer qwen35_orig --tag gem_T --max-tokens 16000`

### 세션 5 · **gemma-N** (no-think 토글)
- 매트릭스 gemma-N→A1: `… cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_gemN_gptoss.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4`
- 매트릭스 gemma-N→A2: `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_gemN_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-parallel 4`
- **[Exp2/B1] self-answer gemma-N**: `SELFANSWER_MODEL=google/gemma-4-31b … selfanswer.py --tag gemmaN --verifier-file data/raw/verifier_gemN_gptoss.jsonl --max-tokens 16000 --reload-every 100 --reload-parallel 4` (**매트릭스 후 실행**; N verifier의 Z/q/Δ — "능력 부족 vs 판정 불능" 분리)
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag gem_N --max-tokens 16000` · `… --answerer qwen35 --tag gem_N --max-tokens 16000`
- 원본판정×2: `… --answerer gptoss_orig --tag gem_N --max-tokens 16000` · `… --answerer qwen35_orig --tag gem_N --max-tokens 16000`

### 세션 6 · **qwen3.6-T**
- 매트릭스 qwen3.6-T→A2: `VERIFIER_MODEL=qwen/qwen3.6-27b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_q36T_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-context 13649 --reload-parallel 4`
- **[answerer T/N ablation] 매트릭스 qwen3.6-T→A2-N**: `… cross_verifier.py --drafts data/raw/drafts_qwen35_nothink.jsonl --n 1500 --out data/raw/verifier_q36T_q35N.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-context 13649 --reload-parallel 4`
- **[None repair] verifier_qwen27 2개 복구**: `VERIFIER_MODEL=qwen/qwen3.6-27b … cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_qwen27.jsonl --repair --verifier-max-tokens 16000`
- **[Exp2] self-answer + features 재생성**: **먼저 구파일 아카이브** `mv data/raw/selfanswer_qwen27.jsonl data/raw/_selfanswer_qwen27_legacy3field.jsonl` → `SELFANSWER_MODEL=qwen/qwen3.6-27b … selfanswer.py --tag qwen27 --verifier-file data/raw/verifier_qwen27.jsonl --max-tokens 16000 --reload-every 100 --reload-context 13649 --reload-parallel 4` (빈답 9 자연 해소)
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag q36_T --max-tokens 16000` · `… --answerer qwen35 --tag q36_T --max-tokens 16000`
- 원본판정×2: `… --answerer gptoss_orig --tag q36_T --max-tokens 16000` · `… --answerer qwen35_orig --tag q36_T --max-tokens 16000`

### 세션 7 · **qwen3.6-N** (no-think 토글)
- 매트릭스 qwen3.6-N→A1: `… cross_verifier.py --drafts data/raw/drafts_phase0_all.jsonl --n 1500 --out data/raw/verifier_q36N_gptoss.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-context 13649 --reload-parallel 4`
- 매트릭스 qwen3.6-N→A2: `… cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_q36N_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-context 13649 --reload-parallel 4`
- **[Exp2/B1] self-answer qwen3.6-N**: `SELFANSWER_MODEL=qwen/qwen3.6-27b … selfanswer.py --tag qwen27N --verifier-file data/raw/verifier_q36N_gptoss.jsonl --max-tokens 16000 --reload-every 100 --reload-context 13649 --reload-parallel 4` (**매트릭스 후 실행**)
- 셔플판정×2: `… shuffle_judge.py --answerer gptoss --tag q36_N --max-tokens 16000` · `… --answerer qwen35 --tag q36_N --max-tokens 16000`
- 원본판정×2: `… --answerer gptoss_orig --tag q36_N --max-tokens 16000` · `… --answerer qwen35_orig --tag q36_N --max-tokens 16000`

### 세션 8 · **Qwen3.5-T** (재방문, think)
- **[baseline] self-verification 대각 (Qwen3.5-T→Qwen3.5 자기답, matrix만)**: `VERIFIER_MODEL=qwen3.5-122b-a10b … cross_verifier.py --drafts data/raw/drafts_qwen35_think.jsonl --n 1500 --out data/raw/verifier_q35T_q35.jsonl --verifier-max-tokens 16000 --reload-every 100 --reload-context 22272 --reload-parallel 4` (reviewer 방어용 대조군, **core 아님**; 셔플엔 불필요)
- 셔플판정 Qwen3.5-T→gpt-oss: `VERIFIER_MODEL=qwen3.5-122b-a10b … shuffle_judge.py --answerer gptoss --tag q35_T --max-tokens 16000`
- 원본판정: `… shuffle_judge.py --answerer gptoss_orig --tag q35_T --max-tokens 16000`

### 세션 9 · **Qwen3.5-N** (no-think 토글)
- (매트릭스 A1 = B-1b 완료, 생략)
- 셔플판정 Qwen3.5-N→gpt-oss: `VERIFIER_MODEL=qwen3.5-122b-a10b … shuffle_judge.py --answerer gptoss --tag q35_N --max-tokens 16000`
- 원본판정: `… shuffle_judge.py --answerer gptoss_orig --tag q35_N --max-tokens 16000`

---

## Phase 3 — 분석 (LLM 0, 스왑 없음) — reviewer 방어 4실험 + 코어

**★ Baseline (reviewer 방어) self-verification 대각 vs cross-model** [appendix/baseline 표, **core 아님**]
- 질문 "cross-model이 정말 필요한가? answerer가 자기 답 재검토하면 안 되나?"를 막는 대조군.
- **딱 2셀** (thinking만): `verifier_gptT_gptoss`(gpt-oss→gpt-oss-T) · `verifier_q35T_q35`(Qwen3.5-T→Qwen3.5-T). 세션2·8에서 생성. **셔플/전체 대각 확장은 안 함**(논문 중심 흐림).
- 표: **C < self-verification ≤ cross-model V** 패턴 기대 (self-verif는 blind-spot·confirmation bias 공유 → C와 비슷/약함, cross-model만 추가 lift).
- consolidation `verification_type`(self/cross)로 자동 분리 → 메인표엔 cross만, appendix에 self baseline.
- ⚠️ **self-verification(답 보고 평가) ≠ self-answer(답 안 보고 직접 풂, Z/q용)** — 둘은 다른 데이터.

**★ Exp1 (최우선) calibration/sharpness 통제 후 ability gating** `phase2_calibration_gating.py` [완성]
- 각 셀: 공통 fold로 cross-fitted isotonic p_C·p_V → held-out 중첩 로짓 M0(pC)→M1(+pV)→M2(+Z)→M3(+Z×불일치). 판정 = held-out NLL 개선 + item-bootstrap CI (LRT 폐기; analysis_plan §4). ECE/sharpness는 셀 수준 descriptive만.
- Z(=verifier 자기정답, gold=mechanism)가 통제 후에도 개선? → **"calibration/sharpness로 환원 안 됨" 방어**.
- **결과(gptoss×gemma-T, 신 설계): ΔNLL +0.0813 CI[0.0524,0.108] ✅** (전 셀은 매트릭스 완성 후).

**★ Exp2 (필수) 배포가능 verifier competence proxy** `phase2_competence_proxy.py` [만들 것]
- q=P(verifier 자기정답 | verifier uncertainty features) 학습(logistic). features: verbalized conf·logprob·hedging·length·(k>0시 self-consistency).
- gpt-oss-T(drafts_phase0_all) + Qwen3.5-T(drafts_qwen35_think) + Qwen3.5-N(drafts_qwen35_nothink) = 무료(기존). gemma-T·qwen3.6-T = 세션4·6 self-answer 재생성(feature). **gpt-oss-N = 세션3 self-answer(selfanswer_gptossN, B-1b 대칭짝)**.
- **verifier self-answer 전원 확보 (Z/q/Δ 전셀 가능)**:
  - gpt-oss-T = **A1 drafts 재사용**(provenance: reused_answerer_draft) · gpt-oss-N ✅(selfanswer_gptossN, 세션3)
  - Qwen3.5-T ✅(drafts_qwen35_think) · Qwen3.5-N ✅(drafts_qwen35_nothink)
  - gemma-T ✅(세션4) · **gemma-N ✅(세션5 추가)** · qwen3.6-T ✅(세션6) · **qwen3.6-N ✅(세션7 추가)**
  - → T/N 차이의 메커니즘 분리 가능: "N은 문제를 못 풀어서 약한가(능력), 풀 수 있는데 판정 출력만 나쁜가(표현)"
- **answerer T/N ablation (B-lite)**: A2-N = drafts_qwen35_nothink 재사용(동일 파이프라인·1500 검증) → 세션4·6에서 gemma-T·qwen3.6-T가 A2-T와 A2-N을 **모두 판정** (verifier 고정, answerer mode만 변화). A1은 T 고정(gpt-oss 기본 thinking).
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
- [신규] verifier_q35T_gptoss · verifier_gptT_q35 · verifier_gptN_q35 · verifier_gemT_q35 · **verifier_gemT_q35N** · verifier_gemN_gptoss · verifier_gemN_q35 · verifier_q36T_q35 · **verifier_q36T_q35N** · verifier_q36N_gptoss · verifier_q36N_q35 · ⬦verifier_gptT_gptoss · ⬦verifier_q35T_q35 · selfanswer_gptossN · **selfanswer_gemmaN · selfanswer_qwen27N**

## 셔플 총량 (Option B, --no-shuffle 재생성)
- verifier 판정: 12셀 × (셔플400 + 원본400) = **9,600**
- 답변자 생성: 2모델 × (셔플400 + **원본400 --no-shuffle**) = **1,600** (같은 조건 clean paired)
- sanity·reference 재활용 없음. 프롬프트·생성조건 100% 동일, 옵션 순서만 차이.
