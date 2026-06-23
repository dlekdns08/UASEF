# Round 11 — Master Plan (focused: R10.4 verification)

> **결정 일자**: 2026-06-23
> **출발점**: Round 10 R10.4 의 RF win 이 사후 진단에서 vacuous artifact 의심
> **핵심 의도**: Round 10 의 두 번째 leakage discovery — 즉 R10.4 가 R10.7 의 leakage suspect features 를 포함하고, 모든 classifier 의 over_esc=1.0 (escalate-all) 인 사실을 검증 + 정정

---

## 0. R10 → R11 transition 의 trigger

External reviewer 비판 + 사후 진단으로 확인된 문제:

| 문제 | 증거 | paper 의미 |
|---|---|---|
| **R10.4 vacuous CRC**: RF 0/1293 ✓ 이 사실은 over_esc=1.0 (모든 negative case 도 escalate) | R10.4 JSON 의 per-seed over_esc_rate 가 모든 5 시드에서 1.0 (RF, LogReg, GBDT, XGBoost) 또는 0.90-0.96 (LLM) | "RF unique winner" 가 framework 의 정직한 win 이 아닌 escalate-all collapse |
| **R10.4 feature leakage**: feature vector 가 Charlson + specialty_baseline_rate 포함 | `_feature_vector()` line 65-71 | R10.7 의 §7 L25-L26 가 인정한 leakage 가 R10.4 의 headline 결과에도 그대로 포함 — self-inconsistency |
| **ECE-결과 불일치**: RF (ECE 0.0051) ≈ XGBoost (0.0049) 인데 결과는 0/1293 vs 149/1293 | RF calibration 결과 + R10.4 결과 | 보정만으로 RF win 설명 불가 |

세 가지 모두 reviewer 가 지적했고 진단으로 확인됨.

---

## 1. R11 의 핵심 실험 — R11.1

**Single experiment, focused scope.**

### Setup

- Underlying classifiers: 5 (gpt_oss_120b, LogReg, GBDT, RandomForest, XGBoost) — R10.4 와 동일
- Seeds: {42, 43, 44, 45, 46}
- n_cal: 3000, n_test: 3000 patient-level split
- α: (0.05, 0.10, 0.15, 0.20)
- **Critical change**: feature vector 를 4 minimal features 만 사용
  - **유지**: age_bucket, adm_emerg, spec_idx, n_labs
  - **제거** (R10.7 L25-L26 leakage suspect): charlson_index, specialty_baseline_rate, n_vital_flags

### 자동 verdict logic

```
if ALL 5 classifiers' CRITICAL over_esc_rate ≥ 0.99 across ALL 5 seeds:
    verdict = "R10.4 was vacuous CRC collapse — framework limit"
    paper headline retracted
elif at least one classifier achieves α-satisfy AND over_esc < 0.99:
    verdict = "Genuine win exists"
    paper headline preserved/partially preserved
else:
    verdict = "Mixed — manual interpretation"
```

### 산출

- `results/round11/r11_1_method_agnostic_minimal.{json,md}`
- per-classifier per-seed cache (64h wallclock 보호)
- **R10.4 의 RF 0/1293 win 의 진위 결정**

---

## 2. Wallclock + cost

| 단계 | LLM calls | Tabular | Wallclock 추산 |
|---|---|---|---|
| LLM (gpt_oss_120b) × 5 seeds × 6000 cases | 30,000 calls | — | ~64 hours |
| LogReg × 5 seeds | — | <30s total | ~30s |
| GBDT × 5 seeds | — | <30s | ~30s |
| RandomForest × 5 seeds | — | <30s | ~30s |
| XGBoost × 5 seeds | — | <30s | ~30s |
| **Total** | | | **~64h** (LLM dominates) |

External API cost: $0. PHI egress: 0.

---

## 3. R11 의 3 가능한 outcome + paper 반응

### Outcome A — 모든 classifier 가 vacuous (over_esc=1.0)

R10.4 의 RF win 이 **artifact**:
- paper §5.4 headline 철회
- 새 §5.4 narrative: "Minimal-feature regime 에서 CRC threshold 가 모든 classifier 에서 collapse → 'escalate-all' vacuous solution. R10.4 의 RF win 은 leakage suspect features 의 함수, framework 의 진정한 win 이 아님."
- 새 contribution: "두 번째 collapsed finding 발견 — paper 의 메타 lesson 강화"

### Outcome B — RF 만 진짜로 over_esc < 1.0 + α-satisfy

R10.4 의 RF win 이 **partially confirmed**:
- paper §5.4 reframe: "RandomForest 가 leakage suspect features 없이도 α-satisfy → 진정한 calibration-driven win"
- §6 calibration analysis 보강

### Outcome C — Mixed (일부 classifier 가 α-satisfy 하나 vacuous)

- paper §5.4 에 over_esc rate 추가 + 진정한 win 만 highlighting

---

## 4. R11 의 paper 통합 path

R11.1 결과 도착 시:

1. `paper/UASEF_FINAL.md` §5.4 의 R10.4 표를 다음으로 교체:
   - over_esc rate column 추가
   - vacuous CRC marker (⚠️) 추가
   - "R10.4 vs R11.1" 비교 표 추가

2. §6 (calibration analysis) 의 가설 정정:
   - 기존: "RF의 낮은 ECE 가 CRC fit 의 메커니즘"
   - 신규 (R11 결과에 따라): "minimal-feature regime 에서 CRC 가 fundamentally vacuous; calibration 차이는 over_esc 까지 통제했을 때만 의미 있음"

3. §7 (limitations) 에 L29-L30 추가:
   - L29: "R10.4 vacuous CRC detection" — R11.1 fix 사례
   - L30: "Minimal feature regime CRC limit" — 4 features 만으로 α 보장 불가의 fundamental limit

4. §8 conclusion 재정립

---

## 5. R11 단일 실행

```bash
export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1
caffeinate -dimsu bash run_round11.sh
# Total wallclock: ~64h on Mac Studio
# Background 실행 권장 — caffeinate 로 sleep 방지
```

R11.1 만 단일 실행. 다른 R11.x 는 R10.5 physician audit + eICU 등 외부 의존 — paper revision 단계에서.

---

_Round 11 plan 작성: 2026-06-23. R10.4 의 vacuous detection 후속._
