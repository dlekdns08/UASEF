# Round 11 — Master Plan (Proceedings track preparation)

> **결정 일자**: 2026-06-23 (revised after Path 2 commitment)
> **목표 venue**: **ML4H 2025 Proceedings track** (8 pages)
> **출발점**: Round 10 R10.4 의 RF win 이 R11.1 에서 leakage artifact 로 retract 됨 (§5.4.1)
> **선택된 path**: **Path 2 — Solid Proceedings (R11.7 + R11.2 + R11.5 + R11.4 + R11.3)**

---

## 0. Reviewer critique 가 요구한 5 가지 + 우리의 응답 mapping

| Reviewer 요구 | 우리의 R11.x |
|---|---|
| (A) positive working contribution 1개 | R11.3 (eICU generalizability) + R11.5 (LLM calibration) |
| (B) MOD/LOW 100% 실패 규명 | R11.4 (information-theoretic limit) |
| (C) 외부 검증 (eICU) | R11.3 |
| (D) LLM minimal-feature 검증 완료 | R11.2 |
| (E) Physician audit | (camera-ready 로 defer — Path 2 에서는 제외) |
| (4) 신뢰성 — paper-JSON 정합성 | R11.7 |

---

## 1. R11.1 — 이미 완료 (2026-06-23 tabular)

R10.4 RF win 의 retraction 확정. 자세한 내용은 §5.4.1 + §5.9 of UASEF_FINAL.md.

---

## 2. R11.2 — LLM minimal-feature re-run

**목적**: §5.9 R11.1 표의 LLM 행 완성. tabular 만으로 RF retraction 했으나 Proceedings 에서는 LLM 까지 동일 protocol 로 통과시켜야 headline retraction 이 *완결*.

### Setup
- gpt-oss-120b, 5 seeds, n_cal=3000, n_test=3000
- minimal 4-feature prompt: age_bucket, adm_emerg, specialty, n_labs only
- 인프라: 기존 `experiments/round11_method_agnostic_minimal.py` 의 `--classifiers gpt_oss_120b` 만

### Wallclock
~64h (LLM 30k calls dominate)

### 산출
- `results/round11/r11_1_cache/gpt_oss_120b.json`
- R11.1 표의 LLM 행 + over_esc rate 명시
- paper §5.9 의 "(LLM R11.1 deferred)" 마침표

### Pre-registered interpretation
- LLM CRITICAL miss < 13.4% (R10.1 leakage 포함 baseline) → leakage feature 가 LLM 에는 *help* 였음
- LLM CRITICAL miss ≈ 13.4% → minimal feature 가 LLM 의 information limit
- LLM CRITICAL miss > 13.4% → leakage feature 가 LLM 에 도움 안 됐었음
- 어느 경우든 α=0.05 통과 가능성 매우 낮음

---

## 3. R11.3 — eICU cross-center validation

**목적**: BIDMC single-center weakness 해소 + audit discipline 의 generalizability 입증. Reviewer 의 가장 큰 한계 지적인 "external validity" 에 직접 응답.

### Hypothesis (pre-registered)

**H1** (audit discipline 보편성): eICU 코호트에서 *동일한 leakage features* (Charlson, specialty_baseline_rate) 가 *동일한 RF 0/n vacuous win* 을 만들어낸다 → audit discipline 이 다른 코호트에서도 leakage 를 잡아냄.

**H2** (minimal-feature limit): eICU 의 minimal 4-feature 결과도 MIMIC-IV 와 유사 (CRITICAL miss 6-14%, no α=0.05 satisfaction) → information limit 가 *데이터셋-독립적*.

### Setup
- Data: eICU-CRD v2.0 (PhysioNet credentialed, Pollard 2018)
  - `patient.csv`, `apachePatientResult.csv`, `lab.csv`, `treatment.csv`, `admissionDx.csv`
  - $n \approx 200{,}000$ ICU stays from 335 US hospitals
- Stratum definition: 동일한 outcome-derived (icu_within_24h, mortality, sepsis_flag 등) — eICU 의 동등 필드
- Classifiers: 5 (gpt_oss_120b, LogReg, GBDT, RF, XGBoost) — R11.1 와 동일
- Two-pass design:
  - Pass A: R10.4 와 동일한 7-feature vector (Charlson, spec_rate 포함)
  - Pass B: minimal 4-feature (R11.1 와 동일)

### Wallclock
- Preprocessing (`round11_eicu_preprocess.py`): ~6-8h (CSV → JSONL, stratum 계산, patient-level dedup)
- Pass A + B tabular: ~30분 each
- LLM Pass A + B: ~64h × 2 = ~128h (또는 LLM 생략 가능, tabular 만으로 H1 검증 가능)

### 산출
- `data/raw/eicu/eicu_cases_v11.jsonl` (commit 금지)
- `results/round11/r11_3_eicu_method_agnostic.{json,md}`
- paper §5.10 — "eICU cross-center replication of audit discipline"
- Reviewer "single-center weakness" 의 직접 응답

### Pre-registered verdict
| eICU 결과 | H1 verdict | paper claim |
|---|---|---|
| eICU full-feature RF 가 vacuous + minimal RF collapse | H1 confirmed | "Audit discipline generalizes" — strong positive contribution |
| MIMIC-IV 와 다른 패턴 | H1 partial | Center-specific 분석 추가 |

---

## 4. R11.4 — MOD/LOW failure information-theoretic 규명

**목적**: §5.7-5.8 의 MOD/LOW 100% miss 를 "data limit" 으로 정식 증명. Framework defect 가 아니라 admission-time features 의 *information-theoretic upper bound* 임을 mutual information 으로 입증.

### Method
- KSG (Kraskov-Stögbauer-Grassberger 2004) estimator 또는 sklearn `mutual_info_classif`
- Per-stratum mutual information:
  $$
  I(X_{t_0}; Y \mid \sigma = s), \quad s \in \{\text{CRIT, HIGH, MOD, LOW}\}
  $$
- CRITICAL/HIGH 의 MI 와 MOD/LOW 의 MI 의 *order-of-magnitude gap* 정량화
- Optional: post-admission features (first 6h labs, charts) 추가 시 MI 회복 측정 → MOD/LOW 가 *post-admission information 으로 풀리는지* 도 정량화

### Wallclock
< 1h (tabular only)

### 산출
- `results/round11/r11_4_modlow_mi.{json,md}`
- paper §5.7.1 — "MOD/LOW failure is an information-theoretic data limit, not a framework limit"
- 결정적 verdict: $\frac{I_\text{MOD}}{I_\text{CRIT}} < 0.1$ 이면 data limit, > 0.5 이면 framework defect

---

## 5. R11.5 — LLM post-hoc calibration

**목적**: "LLM safety gate 가 임상 deployable 하게 만들 수 있는가?" 라는 deployment-level 질문에 정량적 답. Platt scaling / isotonic regression 으로 gpt-oss-120b score 보정 후 ECE / CRC coverage 변화 측정.

### Method
- LLM score cache 재사용 (R10.4 / R11.2)
- 3 calibration methods:
  - Platt scaling (logistic regression on LLM scores)
  - Isotonic regression
  - Beta calibration (Kull et al. 2017) — optional
- Cross-validated calibration (cal split 의 50% 로 calibrator fit, 50% 로 CRC quantile)
- Compare:
  - Uncalibrated LLM: ECE 0.3447, CRITICAL miss ~13%
  - Calibrated LLM: ECE 변화 + CRITICAL miss 변화 + over_esc 변화

### Wallclock
< 30분 (LLM scores 캐시 재사용)

### 산출
- `results/round11/r11_5_llm_calibration.{json,md}`
- paper §6.6 — "Can post-hoc calibration rescue LLM gating?"
- Pre-registered: ECE 가 0.34 → 0.05 이하로 감소하나 CRC 통과 *불가* 일 가능성 (sharpness 0.0157 이 fundamental limit) → "calibration alone is insufficient" 주장

---

## 6. R11.6 — Physician audit (DEFERRED to camera-ready)

Path 2 에서는 제외. R10.5 IRB infrastructure 는 이미 ship 되어 있으므로 camera-ready revision 에서 진행. §5.5 / §7 deferred framing 유지.

---

## 7. R11.7 — Numerical reproducibility audit (1순위)

**목적**: paper 의 모든 수치 table 을 results/*.json 과 자동 대조. 두 번 leakage 겪은 파이프라인은 reviewer 가 재현 시도할 가능성 높으므로 *paper-JSON regression test* 형태 보강.

### Method
- Markdown table parser (R10.4, R11.1, R10.7 등 모든 numeric table)
- 각 cell 의 (classifier, stratum, metric) 트리플 → results/*.json 의 대응 값 lookup
- Mismatch detected → `RAISE` (CI fail)
- 추가: paper sync command 가 모든 table 을 idempotent 하게 재생성

### Wallclock
2-3h (코드)

### 산출
- `experiments/round11_paper_audit.py`
- `tests/test_paper_numerical_consistency.py` — pytest 로 CI 통합 가능
- paper 의 모든 marker block 의 mismatch 자동 검출
- 향후 paper edit 시 stale 수치 방지

---

## 8. R11.2 ~ R11.7 dependency graph

```
R11.7 (immediate, 2-3h)
  ↓ (paper sync 확립)
R11.4 (1h tabular) ──┐
R11.5 (30분)         ├── 모두 paper §5/§6 새 subsection 추가
R11.2 (~64h LLM) ────┤
R11.3 preprocessing (~8h) ──→ R11.3 tabular (~30분) ──→ R11.3 LLM (~64h)
                                       ↓
                                  paper §5.10 추가
```

---

## 9. Total wallclock for Path 2

| 단계 | Wallclock | 누적 |
|---|---|---|
| R11.7 (paper audit) | 2-3h coding | 3h |
| R11.4 (MOD/LOW MI) | 1h | 4h |
| R11.5 (LLM calibration) | 30분 | 4.5h |
| R11.2 (LLM minimal) | 64h | 68.5h |
| R11.3 preprocessing | 8h | 76.5h |
| R11.3 tabular | 30분 | 77h |
| R11.3 LLM Pass A | 64h | 141h |
| R11.3 LLM Pass B | 64h | 205h |
| **Total** | | **~205h (8.5일)** |

Optional optimization:
- R11.3 LLM 은 tabular result 가 strong 하면 *생략 가능* (cross-center generalizability 의 tabular evidence 만으로 충분)
- 그 경우 Total ≈ 77h (3.2일)

External API cost: $0. PHI egress: 0.

---

## 10. Paper § 변화 예상

| § | 변화 |
|---|---|
| Abstract point 1 | R11.2 LLM 결과 추가, "no classifier satisfies α=0.05" 주장 강화 |
| **§5.9** (R11.1) | LLM 행 채워짐 |
| **§5.7.1** (신규) | R11.4 MI 결과 — MOD/LOW data limit 증명 |
| **§5.10** (신규) | R11.3 eICU 교차센터 — audit discipline generalizability |
| **§6.6** (신규) | R11.5 LLM post-hoc calibration |
| §7.4 | L29-L30 update + L31 eICU corroboration + L32 MOD/LOW data limit |
| §8 | "audit discipline 의 보편성" 추가 — Proceedings positive contribution |

---

## 11. 실행 순서 (확정)

1. **R11.7** (즉시 — 2-3h) — paper 수치 검증 인프라 확립
2. **R11.4** (R11.7 후 — 1h) — MOD/LOW data limit 증명
3. **R11.5** (R11.7 후 — 30분) — LLM calibration verdict
4. **R11.2** (background — 64h) — LLM minimal re-run
5. **R11.3** (R11.7 후 — 8h preprocess + 30분 tabular) — eICU 우선 tabular
6. (선택) R11.3 LLM Pass A + B — 추가 128h

각 단계 후 paper 즉시 동기화 (R11.7 의 audit pass 확인).

---

_Round 11 plan revised 2026-06-23. Path 2 commit, R11.7 first._
