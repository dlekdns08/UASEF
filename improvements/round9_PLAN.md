# Round 9 — MIMIC-IV Integration Plan

> **목표**: PhysioNet credentialed dataset (**MIMIC-IV v3.1**, 2024-10) 을 UASEF 평가 파이프라인에 도입.
> Round 8 까지의 핵심 한계 — **L3 (n_CRITICAL<999 → α=0.001 미검증)**, **L7 (single-dataset)**, **L8 (calibration distribution shift 직접 증거 부재)** — 를 한 번에 해소하고, 합성·QA-derived 라벨이 아닌 **실제 임상 outcome 라벨**로 paper claim 을 보강.
>
> **데이터 위치**: `~/Downloads/mimic-iv-3.1/` (사용자 로컬, ~60 GB 압축, hosp + icu 모듈 포함, **note 모듈 별도 신청 필요**)
> **착수 일시**: 2026-05-10
> **선행 조건**: Round 8 P1.3 (LLM-judge κ) 패치 완료, 5-시드 부트스트랩 완료 (run_20260509-013417_aggregate)

**Headline 백엔드 (2026-05-11 결정)**: **LMStudio `openai/gpt-oss-120b` 만** (Mac Studio 96 GB unified memory).
- 모델: OpenAI gpt-oss-120b — Apache 2.0 오픈웨이트, 120B MoE, native MXFP4 4-bit 양자화 (~65 GB), logprobs 지원.
- 하드웨어: Mac Studio 96 GB unified memory (모델 로드 후 inference 헤드룸 ~30 GB).
- 환경변수: `LMSTUDIO_MODEL=openai/gpt-oss-120b` (이미 `.env` 에 설정됨).
- 동기: PhysioNet DUA 보수적 해석 + 실 병원 deployment (HIPAA 환경) 가 외부 API 송신 불가 → headline 수치는 hospital-deployable 모델에서 측정해야 정직. 또한 OpenAI 자체가 같은 회사의 closed model (gpt-4o) 와 open-weight (gpt-oss-120b) 양쪽에서 conformal guarantee 가 살아남는지 정직히 보일 수 있음 — 동일 vendor 비교라 "open vs closed" 의 다른 변인 (학습 데이터 분포, RLHF protocol) 영향이 최소화됨.
- OpenAI gpt-4o 비교는 supplementary §J 옵션 (`BACKENDS="openai lmstudio"`) 으로 capability-ceiling 참조 — 배포 권고가 아님.
- 비용: $0 (로컬 only). 시간: Mac Studio LMStudio 처리량에 의존 (~3–5h wallclock for R9.1–R9.5 single seed).

---

## 0. 왜 MIMIC-IV 인가 (Round 8 까지의 갭과의 직접 매핑)

| Round 8 한계 | MIMIC-IV 해결 메커니즘 |
|---|---|
| **L3** n_CRITICAL < 999 → α=0.001 은 합성에서만 알고리즘-레벨 검증 | MIMIC-IV ED 모듈 (`hosp/triage` 또는 `ed/edstays`) 의 ESI=1 only 만 해도 ≈27k cases. **n_CRITICAL ≥ 999 가 trivial 하게 충족** → α=0.001 의 **실데이터 empirical 검증** 가능. |
| **L7** Single-dataset (MedAbstain n=50/variant) | MedAbstain (QA-derived) ↔ MIMIC-IV (real EHR) **두 개의 서로 독립적인 도메인**. 같은 v2 framework 가 양쪽에서 작동함을 보이면 generalization claim 의 강도 ↑. |
| **L8** Calibration distribution shift — `dist_shift_smoke` 가 합성 specialty 라벨 기반 | MIMIC-IV `services` 테이블 (~100 service code) 로 **진짜 specialty transfer** 실험 가능. weighted CP (Tibshirani 2019) 검증의 직접 증거. |
| **L1** Heuristic ground-truth labels | MIMIC-IV outcome 라벨 (ICU admission / in-hospital mortality / 30-day readmission) 은 **휴리스틱이 아닌 실제 임상 outcome**. LLM-judge κ 와 별개로 ground-truth quality 자체가 한 단계 위. |

⚠️ **하지만 데이터셋 변경 자체는 알고리즘 신규성 추가가 아님** — Round 9 의 contribution framing 은 "v2 가 합성·QA 도메인에서 보였던 guarantee 가 **real-EHR 데이터에서도 성립함을 처음으로 검증**" 으로 갑니다.

---

## 1. 데이터 인벤토리 (사용자 다운로드 분석)

### 1.1 사용 가능한 모듈 (`~/Downloads/mimic-iv-3.1/`)

**hosp/** (병원 전체 EHR, 22 tables):
- `admissions.csv.gz` — 입원 시작·퇴원·death flag (CRITICAL 라벨 산출의 1차 소스)
- `patients.csv.gz` — demographics
- `diagnoses_icd.csv.gz` + `d_icd_diagnoses.csv.gz` — ICD-10 진단 코드
- `procedures_icd.csv.gz` + `d_icd_procedures.csv.gz` — 시술 코드
- `labevents.csv.gz` + `d_labitems.csv.gz` — 검사 결과 (sepsis flag 산출)
- `microbiologyevents.csv.gz` — 배양 결과
- `prescriptions.csv.gz`, `pharmacy.csv.gz` — 약물 처방
- `services.csv.gz` — **specialty 분류의 ground-truth source** (cardiology/neurology/general_medicine 등)
- `transfers.csv.gz` — ICU↔ward 이동 (CRITICAL escalation 라벨)
- `omr.csv.gz` — outpatient measurements (vital signs)

**icu/** (ICU 전용, 9 tables):
- `icustays.csv.gz` — ICU 입실·퇴실 (n≈70k stays). **CRITICAL 라벨의 핵심**.
- `chartevents.csv.gz` — bedside vital signs (수억 row)
- `inputevents.csv.gz` — IV/약물 투여
- `outputevents.csv.gz` — 배출 (urine output 등 sepsis indicator)
- `procedureevents.csv.gz` — 시술 이벤트 (intubation, dialysis 등)

### 1.2 별도 신청 필요한 모듈 (현재 미보유)

- **MIMIC-IV-Note v2.2** (`note/discharge.csv.gz`, `note/radiology.csv.gz`) — **deidentified free text**. PhysioNet credentialing **별도 데이터셋**. → Round 9 Phase 1 은 structured data only 로 진행, free text 는 Phase 2 옵션으로 분리.
- **MIMIC-IV-ED v2.2** (`ed/edstays.csv.gz` + `triage.csv.gz`) — ESI level 직접 사용 가능. **신청 권장**. 없으면 `transfers.csv.gz` + `admissions.csv.gz` 의 `admission_type='URGENT'/'EMERGENCY'` 로 proxy.
- **MIMIC-IV-CXR**, **MIMIC-IV-WDB** — 이미지·waveform. Round 9 범위 외.

### 1.3 라이선스·DUA 의무

- **PhysioNet Credentialed Health Data License v1.5.0** ([LICENSE.txt:1-7](~/Downloads/mimic-iv-3.1/LICENSE.txt#L1-L7))
- **데이터 자체 commit 절대 금지** — `.gitignore` 강화 필수
- **derived statistics 만 commit 허용** — calibrated thresholds, summary tables, anonymized aggregate counts
- **OpenAI API 로 raw note text 송신은 DUA 7항 위반 가능성** — ⚠️ Phase 2 의 free-text 부분은 **로컬 LMStudio backend only** 로 제한
- **Structured features (ICD code, lab abnormality flag, age bucket, vitals quartile)** 는 OpenAI 로 보낼 수 있으나, paper 에 "we only send aggregated structured signals, never raw note text" 명시 필수
- LLM 호출 결과 (escalate yes/no + 1-sentence rationale) 도 PHI 가 들어갈 수 없는 형식으로만 저장

---

## 2. 라벨 스키마 (UASEF MedQACase 매핑)

### 2.1 stratum 정의 (real-outcome 기반, 임상 표준)

| Stratum | MIMIC-IV 라벨 정의 | 예상 n (single-center, 2008–2019) |
|---|---|---|
| **CRITICAL** | (a) ICU admission within 24h of hospital admission ∨ (b) in-hospital mortality ∨ (c) `admission_type='EMERGENCY'` AND ESI≤2 (proxy if ED unavailable) | ≈ **40k–60k** (≫ 999) |
| **HIGH** | (a) sepsis-3 criteria (SOFA Δ≥2 within 48h) ∨ (b) 30-day readmission ∨ (c) blood transfusion within 24h | ≈ 30k |
| **MODERATE** | standard inpatient admission, no ICU, no mortality | ≈ 200k |
| **LOW** | discharged from ED without admission (proxy: short hospital LOS < 24h, discharge home) | ≈ 100k |

**CRITICAL 라벨 산출 SQL pseudocode**:
```sql
expected_escalate = TRUE IFF (
    EXISTS (SELECT 1 FROM icustays WHERE hadm_id=A.hadm_id
            AND intime <= A.admittime + INTERVAL '24 hours')
    OR A.hospital_expire_flag = 1
    OR (A.admission_type IN ('EMERGENCY', 'URGENT'))
)
```

### 2.2 case → MedQACase 변환

이미 [data/loader.py:728](data/loader.py#L728) 에 `_load_mimic_jsonl` 가 있고 `_MIMIC_NOTE_TEMPLATE` 가 있으니, 그 형식을 그대로 따라 `data/raw/mimic4_cases.jsonl` 을 만듦. 한 줄 = 한 입원:

```jsonl
{"hadm_id":"...","note_type":"discharge_summary_OR_structured_proxy",
 "text":"<free text OR structured proxy>",
 "icd_codes":["I50.9","N18.3"],
 "specialty":"cardiology",
 "expected_escalate":true,
 "stratum":"CRITICAL",
 "outcome":{"icu_within_24h":true,"in_hospital_mortality":false,"sepsis":false,"readmit_30d":false}}
```

**Phase 1 (structured proxy)**: text 필드는 `_MIMIC_NOTE_TEMPLATE` 로 ICD-10 + 주요 lab abnormality + chief complaint (admissions 의 `diagnosis` 컬럼) 를 합성한 **구조화 텍스트**. PhysioNet DUA 7 항 OpenAI 송신 issue 회피.

**Phase 2 (free text, 옵션)**: `note/discharge.csv.gz` 첫 800 chars 를 사용. **lmstudio backend 전용**. OpenAI 로는 송신 안 함.

---

## 3. 갭 인벤토리 (Round 9 specific)

### P0 (출판 보강 차단)

| ID | 갭 | 현재 | Round 9 해결 |
|---|---|---|---|
| R9-P0-1 | MIMIC-IV → MedQACase preprocessing 스크립트 부재 | `data/loader.py` 에 mimic3 placeholder 만 있음 | `experiments/round9_mimic4_preprocess.py` 신규. PostgreSQL or pandas chunked CSV.gz 직접 처리. |
| R9-P0-2 | data 가 commit 가능한 위치 (`data/raw/`) 에 들어가지 않도록 보장 | `.gitignore` 에 `data/raw/` 이미 포함 | MIMIC-IV 전용 sub-directory `data/raw/mimic-iv/` 명시 추가 + pre-commit hook |
| R9-P0-3 | OpenAI API 로 raw note 송신 차단 | 차단 코드 없음 | `models/model_interface.py` 에 `UASEF_BACKEND_NEVER_SEND_PHI` env 추가; `case.source.startswith('mimic4_note')` 면 openai backend 거부 |
| R9-P0-4 | MIMIC-IV 가 hosp + icu 만 (note 없음) → free-text 실험 차단 | hosp + icu 만 다운로드됨 | Phase 1 은 structured proxy only 로 진행. note 모듈 별도 신청은 Phase 2. |
| R9-P0-5 | α=0.001 이 paper §3.3·§7.2 에서 "aspirational" 로만 표기 | n_CRITICAL<999 한계 | MIMIC-IV CRITICAL n≈40k → **empirical α=0.001 표 (Round 9 Table 1c)** 로 paper L3 limitation 자체 제거 |

### P1 (강한 리뷰 지적)

| ID | 갭 |
|---|---|
| R9-P1-1 | Cross-specialty distribution shift — Round 8 dist_shift_smoke 는 합성. MIMIC-IV `services` 테이블로 진짜 transfer 실험 (`cardiology → neurology` 등) → §G 강화 |
| R9-P1-2 | Real-EHR 도메인 v2 vs TECP/Quach/SE 헤드-투-헤드 (Round 9 Table 4-MIMIC) |
| R9-P1-3 | MIMIC-IV 환자 cohort 의 **temporal shift** 실험 — 2008–2014 calibration vs 2015–2019 test (paper §8 L8 강화) |
| R9-P1-4 | demographic equity audit — race/sex/age × stratum × miss_rate (Round 8 §I 의 진짜 데이터 버전) |

### P2 (강화)

| ID | 갭 |
|---|---|
| R9-P2-1 | MIMIC-IV-Note 신청·승인 → free-text discharge summary 로 LLM-judge κ 재실행 (Round 8 P1.3 의 cross-domain validation) |
| R9-P2-2 | MIMIC-IV-ED 신청·승인 → ESI level 직접 사용으로 CRITICAL/HIGH proxy 정밀화 |
| R9-P2-3 | IRB 라벨링과 cross-check — board-cert physician 100 cases re-label vs MIMIC outcome 라벨 (κ 보고) |

---

## 4. 단계별 실행 계획

### Phase 1 — Structured Data, hosp + icu only (3주, 비용 ~$80)

**Week 1 — Preprocessing 인프라**
- [R9-P0-2] `.gitignore` 업데이트: `data/raw/mimic-iv/` 명시
- [R9-P0-1] `experiments/round9_mimic4_preprocess.py` 신규
  - 입력: `~/Downloads/mimic-iv-3.1/{hosp,icu}/*.csv.gz` 경로 (env: `MIMIC4_DIR`)
  - 출력: `data/raw/mimic-iv/mimic4_cases.jsonl` (CRITICAL/HIGH/MODERATE/LOW 라벨 포함)
  - 처리: pandas chunked read, hadm_id 단위 join, outcome 산출, stratum 분류
  - 샘플링: `--n-per-stratum 1500` 기본 (CRITICAL 1500, HIGH 1500, MOD 1500, LOW 1500 = 6000 cases)
- [R9-P0-3] `models/model_interface.py` 에 PHI guard 추가
- [data/loader.py] `_load_mimic_jsonl` → `_load_mimic4_jsonl` 로 rename + stratum 필드 인식; `mimic4` dispatcher key 추가
- 테스트: `tests/test_mimic4_loader.py` — 데이터 없을 때 graceful skip, 데이터 있을 때 schema validation

**Week 2 — Round 9 실험 스크립트**
- [R9-P0-5] `experiments/round9_alpha_critical_real.py` — n_CRITICAL=1500 으로 α=0.001 **empirical** 검증
  - 출력: `results/round9/alpha_critical_real.{json,md}` (Table 1c)
- [R9-P1-2] `experiments/round9_table4_mimic.py` — TECP/Quach/SE/v1/v2 head-to-head on MIMIC-IV
  - 출력: `results/round9/table4_mimic.{json,md}`
- [R9-P1-1] `experiments/round9_distribution_shift.py` — `services` 테이블 기반 specialty transfer
  - cardiology → {neurology, general_medicine, surgery} 의 miss-rate violation 측정
  - weighted CP (Tibshirani 2019) 적용 후 회복 정도 측정
  - 출력: `results/round9/distribution_shift_real.{json,md}` (§G 강화)
- [R9-P1-3] `experiments/round9_temporal_shift.py` — 2008–2014 cal vs 2015–2019 test
  - 출력: `results/round9/temporal_shift.{json,md}`

**Week 3 — Equity audit + 통합 실행**
- [R9-P1-4] `experiments/round9_equity_real.py` — demographic × stratum × miss
  - race/sex/age_bucket 마다 per-stratum miss 측정. paper §I 의 합성 → 실데이터 버전.
  - 출력: `results/round9/equity_audit_real.{json,md}`
- 신규 `run_all_round9.sh` — Round 8 와 동일한 구조, 실행 단계: P0 (preprocess) → R9.1 (α=0.001) → R9.2 (Table 4-MIMIC) → R9.3 (dist shift) → R9.4 (temporal) → R9.5 (equity) → 통합 보고서
- `tests/test_paper_claims.py` 에 Round 9 regression guard 추가:
  - α_CRITICAL=0.001 empirical: miss ≤ 0.0012 (2σ upper)
  - Table 4-MIMIC: v2 CRITICAL recall ≥ 0.90
  - dist shift: weighted CP 후 violation ratio ≤ 1.5×

### Phase 2 — Free Text + ED + Note (2–4주, optional, 추가 ~$40)

- MIMIC-IV-Note v2.2 신청·승인 (PhysioNet 평균 5–10 영업일)
- MIMIC-IV-ED v2.2 신청·승인 (위와 동일)
- 승인 후:
  - `experiments/round9_llm_judge_mimic_note.py` — MIMIC discharge summary 100 cases × 2 judge (gpt-5.5 + claude-opus-4-7) — **lmstudio backend only** for the note text
  - `experiments/round9_esi_proxy_validation.py` — Phase 1 의 proxy CRITICAL 라벨이 ESI≤2 와 얼마나 일치하는지 보고

### Phase 3 — Camera-ready 통합 (1주)

- Round 9 결과를 paper §6 (Table 1c, Table 4-MIMIC), §7.2 (L3 제거), §8 (L7 약화, L8 강화) 에 반영
- Supplementary §J (MIMIC-IV preprocessing recipe) 신규
- Abstract 1줄 추가: "We further validate v2 on MIMIC-IV v3.1 (n_CRITICAL ≈ 40k from real ICU admission outcomes), confirming the per-stratum guarantee holds at α_CRITICAL = 0.001 on real EHR data."

---

## 5. 코드/문서 수정 체크리스트

### 코드
- [ ] `.gitignore` — `data/raw/mimic-iv/` 명시 + `**/mimic*.csv.gz` 광역 차단
- [ ] `data/loader.py` — `_load_mimic4_jsonl` (Phase 1: hosp+icu structured proxy), `load_mimic4_cases`, `load_mimic4_scenarios`, dispatcher 에 `"mimic4"` 추가
- [ ] `models/model_interface.py` — `UASEF_BACKEND_NEVER_SEND_PHI` env. `source` 가 `mimic4_note*` 로 시작하면 openai 백엔드 거부
- [ ] `experiments/round9_mimic4_preprocess.py` (신규) — hosp + icu CSV.gz → JSONL
- [ ] `experiments/round9_alpha_critical_real.py` (신규) — Table 1c
- [ ] `experiments/round9_table4_mimic.py` (신규) — Table 4-MIMIC
- [ ] `experiments/round9_distribution_shift.py` (신규) — `services` specialty transfer + weighted CP
- [ ] `experiments/round9_temporal_shift.py` (신규) — 2008–14 vs 2015–19
- [ ] `experiments/round9_equity_real.py` (신규) — demographic equity
- [ ] `run_all_round9.sh` (신규) — Round 8 와 동일 구조
- [ ] `tests/test_mimic4_loader.py` (신규) — graceful skip + schema validation
- [ ] `tests/test_paper_claims.py` — Round 9 regression guard 추가

### 문서 (paper/)
- [ ] `paper/UASEF_Round7.md` — Abstract 마지막 caveat 문단에 Round 9 한 줄 추가, §7.2 (L3 약화), §8 L7 (multi-domain), §8 L8 (real distribution shift), §9 (Conclusion future work 단락)
- [ ] `paper/UASEF_Round7_KO.md` — 동일
- [ ] `paper/UASEF_Round7_Supplementary.md` — §J. "MIMIC-IV Validation Recipe (Round 9)" 신규 섹션, §G distribution shift 강화 (real vs synthetic 두 표)
- [ ] `paper/UASEF_Round7_Supplementary_KO.md` — 동일
- [ ] `paper/IRB_PROTOCOL.md` — MIMIC-IV credential 절차 + DUA 의무 명시 (CITI 교육, PHI guard env, 데이터 commit 금지)

### 외부 운영
- [ ] PhysioNet account → MIMIC-IV-Note v2.2 신청 (Phase 2)
- [ ] PhysioNet account → MIMIC-IV-ED v2.2 신청 (Phase 2)
- [ ] CITI 교육 이수 확인 (이미 MIMIC-IV-3.1 받은 상태면 완료된 것으로 간주)

---

## 6. 비용·시간·자원 예산

| Phase | 항목 | OpenAI | Anthropic | LMStudio | 시간 |
|---|---|---|---|---|---|
| W1 | preprocessing (raw CSV.gz → JSONL) | – | – | – | ~2h CPU (70 GB read) |
| W2 R9.1 | α=0.001 empirical (n_CRIT=1500, 5 seeds) | ~$15 | – | ~30 min | ~1h |
| W2 R9.2 | Table 4-MIMIC (8 method × 2 backend × 5 seeds) | ~$40 | – | ~1h | ~3h |
| W2 R9.3 | dist shift (4 specialty × weighted CP, 5 seeds) | ~$15 | – | ~30 min | ~1h |
| W2 R9.4 | temporal shift (1 split × 5 seeds) | ~$5 | – | ~10 min | ~30 min |
| W3 R9.5 | equity audit (per-demographic, 1 seed) | ~$5 | – | ~10 min | ~30 min |
| **Phase 1 total** | | **~$80** | **–** | **~3h** | **~2 weeks calendar** |
| Phase 2 | LLM-judge on MIMIC note (n=100 × 2 model) | ~$10 | ~$15 | ~30 min | ~1 day after PhysioNet 승인 |
| Phase 2 | ESI proxy validation | – | – | – | ~30 min |
| **Phase 2 total** | | **~$10** | **~$15** | – | **~1–2 weeks calendar (PhysioNet 승인 대기)** |

저장 공간: MIMIC-IV-3.1 압축 ~10 GB, 압축해제 후 ~80 GB. preprocessing 결과 JSONL ~50 MB.

---

## 7. 위험 및 완화

| 위험 | 완화 |
|---|---|
| MIMIC-IV stratum 라벨링 (CRITICAL=ICU<24h ∨ 사망 ∨ EMERGENCY) 이 임상적으로 부정확하다는 리뷰 지적 | board-cert 의사 1–2명에게 100 case 검증 → κ 보고. Phase 2 에 명시 |
| OpenAI 송신 PHI 누출 사고 | (1) `UASEF_BACKEND_NEVER_SEND_PHI=1` env, (2) `source` 가 `mimic4_*` 인 case 는 openai 거부, (3) Phase 1 은 structured proxy 만 사용, (4) Phase 2 raw text 는 lmstudio only. 통합 테스트로 검증 |
| MIMIC-IV preprocessing 이 너무 무거워 reviewer 가 재현 불가 | `--n-per-stratum 100` quick mode 제공 (총 400 case, ~5 min preprocessing). paper Appendix §J 에 두 가지 모드 명시 |
| α=0.001 이 MIMIC-IV 에서도 깨짐 (real EHR 의 noise 가 합성보다 큼) | 정직하게 보고. 깨지면 §7.2 의 limitation 표현을 "α=0.005 까지 empirical, α=0.001 은 algorithm-level only" 로 약화하되 limitation 자체는 paper 에 남김. headline claim 손상 없음 (현재 paper headline 은 α∈[0.05, 0.20]). |
| `services` specialty 분포가 BIDMC single-center bias 때문에 generalize 안 됨 | paper §8 L8 에 "single-center MIMIC-IV-only validation; multi-center eICU validation deferred to follow-up" 명시 |
| Round 8 multi-seed (5 시드, 41h) 비용 재발 | Round 9 는 기본 5 시드 유지하되, structured-data preprocessing 은 결정론적이므로 시드 영향 받는 부분만 멀티시드 (Table 1c, Table 4-MIMIC) 한정. dist-shift / temporal / equity 는 1 시드로 충분 |

---

## 8. Round 8 결과와의 통합 (paper claim 변화)

| Claim | Round 8 (현재) | Round 9 후 |
|---|---|---|
| Abstract: "validated empirically at α_s ∈ [0.05, 0.20]" | 유지 | "validated empirically at α_s ∈ **[0.001, 0.20]** on MIMIC-IV ICU admissions; main MedAbstain table reports α_s ∈ [0.05, 0.20]" |
| §6.3 Table 1 (Per-Stratum Coverage) | MedAbstain only | Table 1a (MedAbstain), Table 1b (synthetic α=0.001 algorithm-level), **Table 1c (MIMIC-IV α=0.001 empirical)** |
| §6.4 Table 4 (Head-to-Head) | MedAbstain only | + **Table 4-MIMIC (real-EHR head-to-head)** in supplementary §J |
| §7.2 (L3) | "n_CRITICAL ≥ 999 unmet → α=0.001 aspirational" | **삭제 또는** "MedAbstain 단독으론 unmet, MIMIC-IV 에선 충족 (n≈40k); main MedAbstain table 은 보수적으로 α∈[0.05,0.20] 유지" |
| §8 L7 (single-dataset) | "MedAbstain only in main paper" | **약화**: "main MedAbstain + supplementary MIMIC-IV (real EHR) — two independent domains" |
| §8 L8 (calibration shift) | "production should pair with drift detection" | **강화**: "MIMIC-IV `services` specialty transfer 실험 결과를 supplementary §G 에 추가; weighted CP 적용 후 violation 회복 정도 정량화" |

---

## 9. 의존성 및 진행 흐름

```
[W1 .gitignore + preprocess + PHI guard]
    ↓
[W2 R9.1 α=0.001]      [W2 R9.2 Table4-MIMIC]      [W2 R9.3 dist shift]      [W2 R9.4 temporal]
    ↓                       ↓                            ↓                         ↓
[W3 R9.5 equity audit] ── [W3 통합 보고서] ── [paper edit + Supplementary §J]
    ↓
[Phase 2: PhysioNet Note/ED 승인 후 R9-Phase2 실험]
    ↓
[Phase 3: camera-ready 통합]
```

W1 은 W2 모두의 입력 (preprocessing 산출 JSONL). W2 의 4 스크립트는 서로 독립 — 병렬 가능. W3 통합 보고서는 W2 모두 필요.

Phase 2 는 W3 와 무관하게 PhysioNet 승인 떨어지면 시작. Phase 3 는 W3 + Phase 2 모두 끝나야.

---

## 10. 한 줄 명령 (Phase 1)

```bash
# 사용자 셋업 (수동, 한 번만)
export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1   # OpenAI 송신 차단

# Phase 1 전체 (preprocess + R9.1–R9.5 + 통합 보고서)
bash run_all_round9.sh                  # ~$80, ~5h wallclock

# 또는 단계별
SKIP_PREPROCESS=1 bash run_all_round9.sh   # JSONL 이미 있으면
SKIP_TABLE4=1 SKIP_TEMPORAL=1 bash run_all_round9.sh   # subset
```

---

_Round 9 plan 작성: 2026-05-10. 데이터 위치: `~/Downloads/mimic-iv-3.1/` (hosp + icu, ~80 GB uncompressed). 본 문서는 commit 가능 (코드/계획만). MIMIC 데이터 자체는 절대 commit 금지._
