# 분석 계획 (prospectively locked) — 남은 조건 완료 전 잠금

> 목적: 결과를 보고 분석법을 바꾸는 것을 방지 (reviewer 신뢰). 이 문서가 규칙의 정본이며,
> 모든 분석 스크립트는 `analysis/splits.py`(공통 fold)와 `analysis/stats.py`(통계 유틸)만 사용한다.
> 고정일: 2026-07-10 (세션1 진행 중, 셔플판정 전체·A2행·T/N 셀 수집 전).
>
> **고정 성격 (정직 표기)**: 완전한 preregistered confirmatory study 아님.
> - **Discovery 셀** (계획 확정 전에 결과를 봄): P0–P2 · gpt-oss-T×gemma-T · gpt-oss-T×qwen3.6-T · gpt-oss-T×Qwen3.5-N.
> - **Prospectively locked confirmation 셀** (완료 전 잠김): A2 negative-Δ 행 전체 · 남은 T/N 셀 · answerer T/N B-lite · self-verification 2셀 · shuffle audit 전체.
> - 논문 표현: "The analysis plan was prospectively locked before completion of the remaining
>   reciprocal, negative-gap, reasoning-mode, and shuffle conditions." ("모두 사전 preregistered" 금지.)

## 1. Outcomes

**Primary (헤드라인, 이 3개로 성패 판단):**
1. ΔAUROC = AUROC(V) − AUROC(C) — 셀별, within-dataset
2. Z-gating: Z=1 vs Z=0 부분집합의 verifier lift 차이
3. Verifier-T vs Verifier-N lift 차이 (동일 모델, 동일 answerer 출력)

**Secondary (보조, 헤드라인 아님):**
q proxy · CMI I(V;Y|C) · shuffle robustness · self-verification 비교 · conformal/threshold transfer · answerer T/N ablation(B-lite 2셀)

## 2. 통계 규칙 (전 분석 공통)
- **행 단위 부트스트랩 금지.** 재표집 단위: matrix 셀=item_id / shuffle=shuffle_group_id(원본+셔플 동행) / 셀 통합=item 클러스터.
- 95% CI = grouped paired bootstrap (`stats.paired_bootstrap_auroc_diff`, seed 고정).
- 정확도 쌍대 비교(original vs shuffled 등) = McNemar(exact).
- **지표 세트**: AUROC(주) + **AUPRC**(class-imbalance 보조, secondary) + Brier + NLL + ECE. 새 추론 불필요.
- **다중 검정 — BH-FDR family (사전 정의, 고정)**:
  - **F1** = V−C lift 검정 (전 셀) · **F2** = Z=1 vs Z=0 lift 차 검정 (전 셀) · **F3** = T vs N 쌍 검정 · **F4** = 원본 vs 셔플 검정.
  - **mc·pm 결과는 같은 family 안에 함께 묶어** 보정한다 (dataset별 분리 family 아님).
- 같은 문항이 여러 셀에 반복 → 셀들을 독립 점으로 취급한 단순 회귀 금지. 셀 통합 주장은 descriptive + item-cluster bootstrap 병기.
- **Parser failure 정책 (고정)**: ① 예측 지표(주) = 유효 파싱 행 complete-case (기존 n=1495/1498 exclusion이 이 정책) ② 실패율은 role·mode·조건별 별도 보고(CSV 17) ③ **운영 게이트: parser failure/empty = 자동 escalation(최대 위험)** ④ 민감도: 실패 행 전부 오류 취급 시 release 성능 재계산.

## 3. Split 규칙
- `results/consolidated/00_split_manifest.csv` (item-grouped, dataset-stratified K=5) 를 **모든** calibration·q 학습·threshold 선택·최종 평가가 공유.
- 사용 방식 = cross-fitting (4 fold 학습 → 1 fold 적용, 회전). 단순 3-way split 금지(PubMedQA 317 파편화).
- 같은 canonical item에서 파생된 모든 행(모델·mode·원본/셔플)은 같은 fold.

## 4. Exp1 (calibration/sharpness 통제) — 수정된 설계
- ~~row-level 회귀에 cell-ECE 복사 투입~~ (표본 부풀림) → **폐기**.
- 1단계: fold별 isotonic calibration으로 C→p_C, V→p_V (cross-fitted).
- 2단계: held-out에서 중첩 모델 비교 — M0: Y~p_C · M1: +p_V · M2: +Z · M3: +Z×D.
  판정 = **held-out NLL/Brier/AUROC 개선 + item-bootstrap CI** (in-sample LRT 아님).
- 3단계: ECE·sharpness는 **셀 수준 descriptive** 표로 별도 보고 (cell lift vs cell ECE/sharpness/verifier-acc 관계).
- 핵심 질문 유지: calibrated p_C,p_V 이후에도 Z/Z×D가 held-out 성능을 개선하는가.

## 5. q proxy — 누수 방지 + 표현 규칙
- q = P(Z=1|features) 학습은 공통 fold로 **cross-fitted** (모든 q 예측치는 out-of-fold).
- 같은 canonical item의 행이 train/test에 갈라지는 것 금지 (fold가 보장).
- tertile·상호작용 분석은 cross-fitted q만 사용.
- **표현 (고정)**: q는 "**배포 추론 시 gold-free**"이며 "학습 단계에서는 gold로 정의한 Z 라벨을 사용"한다고 항상 병기.
  ("q is inference-time gold-free, but trained on calibration data whose competence labels Z are derived from gold answers.")
  단순 "gold 미사용/gold-free" 단독 표기 금지.

## 6. A9 명칭·주장 수위 (하향 고정)
- ~~"ability-gap law", "lift≈f(Δ) 법칙"~~ → **"평균 Δ 불충분성 분석 + negative-Δ stress test"**.
- 검정하는 것: (a) 평균 능력격차가 lift를 충분히 설명하는가, (b) negative-Δ에서도 신호가 남는가.
- 셀 산점도는 descriptive. 셀 간 회귀에 독립성 주장 금지 (공유 문항·공유 모델).

## 7. Self-verification baseline (2셀 한정)
- 비교: C · S(self-verif) · V(cross) · C+S · C+V · **C+S+V vs C+S** (핵심: S를 알고도 V가 증분 정보를 주는가).
- 2셀뿐이므로 self-verification 일반론 주장 금지. appendix/baseline 표.

## 8. CMI (A8) — 보조 결과로 격하 + 강건성 의무
- 고정 binning 규칙(사전: risk·conf 각 10-quantile bin), bin 수 민감도 표, permutation null, item-bootstrap CI 필수.
- 헤드라인은 AUROC/lift/Z-gating. CMI는 해석 보조.

## 9. Shuffle audit
- 결정론 확인됨: 답변자·판정자 모두 temperature=0 (중복 행 동일 출력으로 실증) → sampling noise confound 없음(Methods 명시).
- 통계는 전부 paired: 정확도=McNemar, lift=shuffle_group bootstrap, agreement=paired proportion. T/N 분리 보고.
- **Z 표기 규칙 (고정)**: shuffle_judge의 `judge_selected_correct`(answerer 답 **노출 후** verifier가 고른 답의 정오)는 **독립 Z가 아님** — Z라 부르지 않는다. 독립 Z는 self-answer 파일에서만 오고, 셔플 행에 join 시 그 값은 "원본 옵션 순서에서 측정된 competence"임을 명시 (shuffle-specific competence 주장 금지). 셔플 core 분석(T/N lift·paired robustness)에는 Z 불필요.
- 셀 구조: 12 (answerer×verifier-mode) 셀 × 원본/셔플 2 × 400 = 9,600 판정.

## 10. 해석 범위 규칙
- **N verifier 셀도 Z/q/Δ 전부 계산** (전 verifier T/N self-answer 확보: gpt-oss=A1재사용/gptossN, Qwen3.5=think/nothink, gemma=세션4/5, qwen3.6=세션6/7).
- T/N lift 차이의 메커니즘 분해: ① N self-acc 하락(능력) vs ② self-acc 유지+판정 열화(표현) — self-answer로 구분.
- answerer T/N ablation = B-lite 2셀(gemma-T·qwen3.6-T가 A2-T/A2-N 판정)만. 일반화 주장 금지.
- A1 answerer mode = **T** (gpt-oss 기본 thinking) 명시.

## 11. 한계 명시 (고정 문구 방향)
- 모델 간 절대 비교에는 family·크기·양자화(GGUF MXFP4/Q4_K_M/Q3_K_S·MLX 4bit)·런타임 confound 존재 →
  "full-precision 본질 순위가 아니라 **고정된 로컬 배포 조건에서의 운영적 정보성**을 분석" 프레임.
- 동일 모델 T/N 비교(Jinja 토글만 변화)가 가장 causal한 대비임을 명시.
- old-schema 4파일: truncated=unknown(0 아님), output_length 52행 우측검열, core 분석 미사용, failure appendix 각주.
- **Conformal 문구 규칙 (고정)**: 이론 섹션 = 단일 split conformal의 finite-sample 보장(exchangeability 하). 실험 섹션 = 반복/cross-fitted 결과는 "**평가된 exchangeable split에서 관측 누출률이 목표 이하**"라는 경험적 제어로 서술 ("보장 성립" 단독 표기 금지 — 반복 split 평균에 자동으로 finite-sample 보장이 생기지 않음). shift transfer = 보장 아닌 stress test.

## 12. Provenance
- 모든 결과 행에 answerer_model/mode·verifier_model/mode·dataset·split·verification_type(cross/self_verification/self_answer) 필수.
- self-answer 재사용 시 `self_answer_source=reused_answerer_draft` 표기 (gpt-oss-T=A1, Qwen3.5-T/N=drafts).
- 모델명 정본 = `analysis/manifest.py MODELS` (LM Studio 런타임 문자열). 논문·CSV 동일 문자열.
- **Run manifest (세션 시작마다 기록, `analysis/run_provenance.py`)**: run_id · 로드된 모델 identifier · 로컬 모델 파일 경로/크기/mtime (+옵션 SHA-256, `--hash`) · LM Studio(lms) 버전 · context length · parallel · thinking 토글 상태(T/N; Jinja 토글은 수동이므로 운영자가 --mode로 명시) · UASEF_QUERY_TIMEOUT_S · git commit · 기록 시각 → `results/consolidated/run_provenance.jsonl` append. Jinja 템플릿/프롬프트는 코드 정본(VSYS·VJSYS)이 git으로 버전관리되므로 commit hash가 프롬프트 해시를 겸함.
