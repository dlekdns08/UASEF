# UASEF Writing-전 보완 계획 (Revision Plan)

작성일: 2026-06-10. 근거: 리뷰 18개 항목 + 실제 코드 검증.

코드 검증으로 확인된 사실(리뷰와 차이 포함):
- **누수(leakage) 확정·심각**: `experiments/round9_mimic4_preprocess.py:261-271` 에서 σ(a)가 미래
  결과(ICU 24h, 사망, LOS, lactate, 30일 재입원)로 만들어지고 `expected_escalate =
  (stratum ∈ {CRITICAL,HIGH})` 로 **Y가 σ(a)의 결정적 함수**. 게다가 프롬프트
  `data/loader.py:728-739` 가 `los_days / lab_flags(lactate_high) / icd_codes` 를 그대로 투입 →
  라벨 정의 필드가 입력으로 새어 들어감.
- **split이 admission(hadm_id) 단위** — `subject_id` group split 없음. 같은 환자가 cal/test 양쪽 가능.
- **CRC loss는 이미 올바름**: `models/stratified_crc.py:71-79` 가 `1{label ∧ score≤λ}` =
  missed-escalation 통제, B=1. 리뷰 #4는 대체로 이미 충족. cost는 분리되어 있음(별도 metric).
- **0/300에 대한 이항 신뢰구간 없음**: Wilson CI는 n=0에서 (0,1) 반환, 단측 exact 상한 미구현.
- **tabular baseline(LR/XGB) 없음** — 모든 baseline이 LLM score 기반.

분류: [P0 필수=writing 전 차단] / [P1 권장] / [APX appendix·병행 가능]
영향: (코드) 코드 수정 / (재실행) 결과표 재생성 필요 / (산문) paper 텍스트만.

---

## P0 — writing 전 반드시 해결 (go/no-go)

### P0-1. Leakage-safe 데이터셋: σ(a)와 Y, 입력과 결과 분리 (리뷰 1·2)
- (코드) `round9_mimic4_preprocess.py`
  - `G(X_t0)` = **decision-time 위험군**: admission_type, age, sex, service, 입원 6h 이내
    가용 vital/lab, 과거 입원/동반질환 ‘이력’ 만으로 정의. (현재 σ에서 ICU24h/사망/LOS/30일
    재입원/discharge-lactate 제거)
  - `Y` = **미래 adverse outcome**: 예) `Y = ICU_transfer_24h OR in_hospital_mortality OR
    early_deterioration`. σ로부터 유도 금지(라인 271 삭제).
  - 출력 JSONL 스키마에 `risk_group`(=G), `y_outcome`(=Y), `decision_time_features` 분리 기록.
- (코드) `data/loader.py:728-739, 947-956` 프롬프트에서 **미래정보 제거**: `los_days` 삭제;
  `lab_flags`/`vital_quartiles`는 ‘입원 후 첫 6h window’ 로 제한; `icd_codes`는 admission
  diagnosis 만(있을 때), discharge ICD 제외.
- (재실행) 라벨/입력이 바뀌므로 R9 전 결과표 재생성 필요.
- (산문) Methods에 X_t0 / G(X_t0) / Y / ℓ_λ 정식 정의 추가.

### P0-2. Feature availability 표 + leakage-safe timeline 그림 (리뷰 1·16)
- (산문) Table: Variable | source table | timestamp | decision-time 가용? | prompt? | label?
- (산문) Figure 2: t0 admission → [allowed feature window] → [future outcome window].

### P0-3. patient-level split (리뷰 9)
- (코드) cal/test split을 `subject_id` group split로. temporal split도 환자 단위 era 일관성 확인
  (multi-admission 환자가 양쪽에 분산되지 않게). readmission 라벨 생성 시 future-admission
  정보가 test로 새지 않는지 점검.
- (재실행) split 바뀌면 결과 재생성.
- (산문) “All calibration/test splits were performed at the patient level (subject_id) to
  prevent repeated-admission leakage.”

### P0-4. α=0.001 주장 + 이항 신뢰구간 (리뷰 3)
- (코드) `experiments/metrics_utils.py` 에 단측 exact upper bound 추가:
  0 사건 관측 시 95% 상한 = `1 - 0.05^(1/n)` (rule of three ≈ 3/n). 0/300 → ≈0.0099.
  Clopper-Pearson 양측도 함께. Wilson의 n=0→(0,1) 케이스 보고에서 이걸로 대체.
- (산문) “최초 실증” → “non-vacuous α=0.001 CRC calibration; held-out test에서 CRITICAL
  miss 0/300 관측(95% 단측 상한 ≈ 1%). 표본만으로 0.1% 이하 miss rate를 통계 검증했다고
  주장하지 않음.” (영문 동일 톤.)

### P0-5. CRC loss 정의 명문화 (리뷰 4) — 코드 OK, 문서화만
- (산문) ℓ_λ(x,y)=1{escalate=0 ∧ y=1}, B=1, n_min=⌈(1-α)/α⌉ 가 B=1 기준임을 명시.
  cost matrix(1000:1)는 **별도 evaluation metric**이며 CRC bound에 들어가지 않음을 분명히.
  v1-cost-aware의 cost=0 은 confusion matrix와 함께 설명.

### P0-6. tabular baseline 최소 2개 (리뷰 5)
- (코드) 신규 `experiments/baselines/tabular.py`: decision-time feature로 LR + XGBoost(또는
  LightGBM) 학습 → 동일 CRC 적용. + admission-type-only heuristic, high-risk→all-escalate
  trivial baseline.
- (재실행) baseline 결과 생성.
- (산문) 약한 주장 채택: “LLM이 tabular를 압도하진 않더라도 CRC framework가 LLM
  decision support에 안전 적용 가능” (데이터 구조상 안전).

---

## P1 — 권장 (writing과 병행 가능, 그러나 강하게 권장)

### P1-1. 강한 표현 하향 (리뷰 7)
- (산문) first/only/unique/최초/유일/가장 엄격 → “to our knowledge, among the first…”,
  “one of the few…”. Round9 §82,157, Round9_KO §76,156 등.

### P1-2. 자연 유병률(prevalence-weighted) 결과 병기 (리뷰 6)
- (코드/재실행) balanced 1500/stratum 결과 + 자연 분포 reweighting 결과 둘 다.
- (산문) Table: method | crit/high miss | over-esc | cost(balanced) | cost(prevalence).

### P1-3. label validity = proxy 명시 (+가능시 소규모 clinician 검증) (리뷰 8)
- (산문) “operational proxy outcomes derived from structured EHR events”. 가능하면 100-case,
  2–3 reviewer, Cohen’s/Fleiss’ κ 의 small validation study.

### P1-4. temporal shift = guarantee 아닌 stress test (리뷰 10)
- (산문) “CRC guarantees under exchangeability; temporal-shift는 가정 위반 시 empirical
  degradation 평가(stress test).” Round9 §416-429 톤 조정.

### P1-5. weighted CP 실패 해석 하향 (리뷰 11)
- (산문) “이론이 틀렸다”가 아니라 “high-overlap·small n_pos 조건에서 우리가 쓴 KDE
  density-ratio 추정이 불안정”.

### P1-6. fairness = subgroup safety audit(exploratory) (리뷰 12)
- (산문) small-cell CI 과대 → “충분 표본 subgroup에선 뚜렷한 disparity 미검출, 소표본은
  불확실”. ‘algorithmic fairness’ 대신 ‘subgroup safety audit’.

### P1-7. contribution 3개로 압축 (리뷰 15)
- (산문) ① real-EHR outcome 기반 risk-stratified CRC evaluation framework ② non-vacuous
  α=0.001 calibration + held-out safety eval ③ cross-specialty/temporal/demographic
  stress test + failure-mode 분석. 나머지는 appendix/implementation.

### P1-8. 최종 framing 채택 (리뷰 18)
- (산문) Title 방향: “Risk-Stratified Conformal Risk Control for Local LLM-Based Clinical
  Escalation on Real EHR Outcomes”. 핵심 메시지: cost 최소화가 아니라 **per-stratum risk
  control + 투명한 failure 분석**.

---

## APX — appendix / 병행 (리뷰 13·14)

### APX-1. PHI/HIPAA/GDPR → data-residency 표현 (리뷰 13)
- (산문) “HIPAA/GDPR 충족” 류 법적 단정 제거 → “data-residency-preserving design; eval
  log에서 external API call 0건 관측. 공식 compliance는 기관 governance/legal 소관.”

### APX-2. reproducibility 상세 (리뷰 14)
- (산문) model checkpoint/revision, quantization, tokenizer, serving backend version, 전체
  prompt template, decoding params(temp/top_p/max_tokens), **logprob 추출 방식 명시**
  (현재 `models/uqm.py:125-139` = 전체 생성 토큰 평균 NLL — 어떤 토큰인지 본문 명기),
  seed(42 + multi-seed), hardware, total calls/runtime, code release.

---

## 실행 순서 (의존성)
1. P0-1/P0-3 (preprocess+split, 코드) → 2. P0-6 (tabular baseline, 코드) →
3. 재실행으로 결과표 재생성 → 4. P0-4/P0-5 CI·loss 문서화 + 결과 반영 →
5. P0-2 표·그림 → 6. P1 산문 일괄 → 7. APX.

주의: P0-1/P0-3/P0-6은 **결과 숫자를 바꾼다**. 재실행 전까지 결과표는 “재실행 대기”로 두고
숫자를 창작하지 않는다.
