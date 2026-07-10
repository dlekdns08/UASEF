# 분석 계획 (사전 고정) — 데이터 수집 완료 전 잠금

> 목적: 결과를 보고 분석법을 바꾸는 것을 방지 (reviewer 신뢰). 이 문서가 규칙의 정본이며,
> 모든 분석 스크립트는 `analysis/splits.py`(공통 fold)와 `analysis/stats.py`(통계 유틸)만 사용한다.
> 고정일: 2026-07-10 (세션1 진행 중, 매트릭스 13/15셀·셔플판정 전체 수집 전).

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
- **다중 검정**: family를 명시하고 family 내 BH-FDR. Family = {Primary-1 전셀}, {Primary-2 전셀}, {Primary-3 쌍}, {각 Secondary 별도}.
- 같은 문항이 여러 셀에 반복 → 셀들을 독립 점으로 취급한 단순 회귀 금지. 셀 통합 주장은 descriptive + item-cluster bootstrap 병기.

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

## 5. q proxy — 누수 방지
- q = P(Z=1|features) 학습은 공통 fold로 **cross-fitted** (모든 q 예측치는 out-of-fold).
- 같은 canonical item의 행이 train/test에 갈라지는 것 금지 (fold가 보장).
- tertile·상호작용 분석은 cross-fitted q만 사용.

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

## 12. Provenance
- 모든 결과 행에 answerer_model/mode·verifier_model/mode·dataset·split·verification_type(cross/self_verification/self_answer) 필수.
- self-answer 재사용 시 `self_answer_source=reused_answerer_draft` 표기 (gpt-oss-T=A1, Qwen3.5-T/N=drafts).
- 모델명 정본 = `analysis/manifest.py MODELS` (LM Studio 런타임 문자열). 논문·CSV 동일 문자열.
