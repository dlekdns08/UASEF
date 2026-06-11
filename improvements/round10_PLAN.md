# Round 10 — Master Plan

> **출발점**: Round 9 (leakage-safe) 가 5월 16일 보고서의 다수 finding 이 leakage artifact 였음을 드러냄.
> **목적**: Round 9 결과를 정직하게 받아 들이고, 그 위에 **통계적으로 적절히 powered**, **method-agnostic**, **distribution-shift-aware** 한 검증 layer 를 쌓아 ML4H 2026 spotlight / NeurIPS 2026 safe-ML workshop 에 제출 가능 수준으로.
>
> **결정 일자**: 2026-06-11
> **착수 가능 일자**: 2026-06-12

---

## 0. Round 9 → Round 10 transition 의 핵심

Round 9 leakage-safe re-run 의 5가지 핵심 finding:

| Finding | Round 9 수치 | Round 10 함의 |
|---|---|---|
| α=0.001 의 0 miss 가 증명이 아님 — n_pos=99 < n_needed=2,995 | exact upper 0.030 | **R10.1**: n_pos≥3,000 으로 cohort 확장하여 진짜 empirical 증명 |
| TECP/Quach/SE 대비 v2 우위 17× recall 유지 (0.85 vs 0.05) | cost 5.5× 절감 | **R10.2**: 보강 — multi-seed CI 로 통계 유의성 강화 |
| simpler ablation (v1-cost-aware, Cost-Sensitive single-α, R6 heuristic) 이 v2 보다 낮은 cost | cost 165 vs 3,648 | **R10.2 + R10.6**: cost matrix sweep + over-esc-aware metric 도입 |
| Cross-specialty / temporal shift 가 catastrophic (5-10× violation) | naive 5.88×, temporal 10× | **R10.3**: detection-only 가 아닌 mitigation strategy (online recal, KMM, group-conditional CRC) |
| R9.6 tabular LogReg/GBDT 가 LLM-v2 와 동등 (recall 0.99 vs 0.85) | sklearn 1초 vs LLM 34시간 | **R10.4**: method-agnostic framework headline 으로 재포지셔닝 |

추가로 Round 9 가 deferred 한 항목:

- L19 single-seed 보고 → **R10.0**: 5-seed bootstrap CI (모든 R10.x)
- L11 stratum 라벨이 outcome-derived (휴리스틱) → **R10.5**: 100-case IRB physician audit
- L17 MODERATE/LOW 100% miss → **R10.7**: decision-time feature 확장
- L20 single-center → Round 11 로 연기 (eICU)
- L21 free-text (Phase 2 free-text) → Round 11 로 연기 (MIMIC-IV-Note credentialing)

---

## 1. Round 10 의 핵심 contribution 재포지셔닝

Round 9 ROUND9_FINAL_REPORT §7.3 의 reframing 을 **paper headline 으로 격상**:

> **The contribution of UASEF is not "an LLM-based escalation framework" but a
> *method-agnostic per-stratum CRC layer* that admits any underlying classifier —
> LLM, gradient-boosted tree, logistic regression — and provides the same formal
> per-stratum coverage guarantee. The LLM is one specific instantiation; we show
> in R10.4 that a 1-second sklearn LogReg/GBDT achieves comparable
> CRITICAL recall (0.99 vs LLM's 0.85) on identical decision-time features.**

이 reframing 의 함의:

1. **결과 비교의 새 차원**: "LLM CP" 와 "LLM no-CP" 비교 (Round 7/9 의 v2 vs TECP/Quach/SE) 만이 아니라, **"any classifier + CRC" vs "any classifier no-CRC"** 비교가 핵심 contribution.
2. **Reproducibility 의 새 차원**: sklearn 1초 baseline 이 LLM 34시간 baseline 과 비슷한 결과 → PhysioNet credentialing 만 있으면 누구나 1초 안에 paper 의 핵심 finding 을 검증 가능. 학계 진입 장벽 대폭 하락.
3. **Deployment 의 새 차원**: 병원 IT 환경에서 LLM 호스팅이 부담스러우면 LogReg + CRC 로 시작 가능. CRC layer 가 동일하면 future LLM upgrade 시 framework swap 가능.

---

## 2. 갭 인벤토리

### P0 (Round 10 submission 차단)

| ID | 갭 | Round 9 상태 | Round 10 해결 |
|---|---|---|---|
| R10-P0-1 | α=0.001 의 empirical proof 불가 (n_pos=99) | exact upper 0.030 | n_pos≥3,000 으로 확장, 5-seed bootstrap |
| R10-P0-2 | single-seed 보고 (L19) | seed=42 only | 5-seed × all R10.x |
| R10-P0-3 | Method-agnostic 주장의 systematic 검증 부재 | R9.6 단발성 | 5 classifier × 동일 CRC layer head-to-head |
| R10-P0-4 | Distribution shift mitigation 없음 — detection only | R9.3 5.88×, R9.4 10× | 3 mitigation strategy 비교 (online recal, KMM, group-conditional CRC) |

### P1 (강한 리뷰 지적)

| ID | 갭 |
|---|---|
| R10-P1-1 | MODERATE/LOW stratum 의 100% miss — decision-time feature 불충분 |
| R10-P1-2 | Stratum 라벨이 IRB-adjudicated 아님 (Round 7 L1, Round 9 L11) — 100-case physician audit |
| R10-P1-3 | 1-D cost matrix sensitivity 만 — Round 7 §6.3 의 4-D sweep 실 데이터 적용 |
| R10-P1-4 | Cohort 가 BIDMC-only — multi-center subgroup (BIDMC 의 outpatient 부서 등) 분석으로 부분 완화 |

### P2 (강화)

| ID | 갭 |
|---|---|
| R10-P2-1 | eICU cross-center 검증 → Round 11 |
| R10-P2-2 | Phase 2 free-text (MIMIC-IV-Note) → Round 11 |
| R10-P2-3 | Pediatric ED corpus 검증 → Round 12 |

---

## 3. 단계별 실험 계획

### R10.0 — Multi-seed infrastructure (1주, 비용 $0)

모든 R10.x 가 5 seed bootstrap CI 를 가지도록 인프라 보강:

- `experiments/round10_multiseed_runner.py` 신규 — Round 10 통합 multi-seed orchestration
- `experiments/aggregate_round10.py` 신규 — 5 시드 통합 + Wilson/Clopper-Pearson CI
- `tests/test_paper_claims.py` 에 R10 regression guard 추가

### R10.1 — Properly-powered α=0.05 empirical validation (Table 1d)

**가설.** $n_{\text{cal}} = 3{,}000$ CRITICAL, $n_{\text{test}} = 3{,}000$ CRITICAL 로 stratum 별 $\mathbb{E}[\ell_s] \le \alpha_s$ 의 단측 exact 95% upper bound 가 $\alpha_s$ 이하 (즉 진짜 empirical proof).

**규모.**
- CRITICAL: $\alpha = 0.05$, $n_{\text{cal}}=n_{\text{test}}=3{,}000$ — exact upper 약 0.040 예상 (보장 충족)
- HIGH: $\alpha = 0.10$, $n_{\text{cal}}=n_{\text{test}}=2{,}000$
- MODERATE: $\alpha = 0.15$, $n_{\text{cal}}=n_{\text{test}}=1{,}500$
- LOW: $\alpha = 0.20$, $n_{\text{cal}}=n_{\text{test}}=1{,}500$
- 5 seeds × 5 underlying classifier (R10.4 와 공유) = **125 시드-classifier 조합**

**규모 정당화.** Round 9 가 α=0.001 을 약속했다가 fail 한 것의 honest reframe — α=0.05 는 임상적으로 deployable level 이고 (5% miss rate 는 "safe but worth-double-checking" tier 에 해당) MIMIC-IV cohort 로 충분히 power 가능.

**산출.** `results/round10/r10_1_alpha_005_empirical.{json,md}` + bootstrap CI

### R10.2 — Multi-seed Table 4-MIMIC (R9.2 의 통계 유의성 강화)

**가설.** v2 의 TECP/Quach/SE 대비 우위가 5-seed bootstrap 95% CI 로 유의 (McNemar pooled p < 0.001).

**규모.** 5 seeds × 동일한 8-method, n_cal=200/stratum × n_test=100/stratum

**추가 분석.** McNemar paired test (seed-pooled), v2 의 CRITICAL recall 95% CI 가 (0.0, 0.20) 의 baseline 과 겹치지 않음 검증.

### R10.3 — Distribution shift mitigation (R9.3 / R9.4 의 detection → mitigation)

Round 9 가 violation 을 **검출** 했다면, Round 10 은 그것을 **완화** 한다.

**Strategy 1 — Online recalibration**. Rolling 3-year window 으로 매년 재보정. R9.4 에 적용 후 violation × 측정.

**Strategy 2 — KMM (Kernel Mean Matching) weighted CP**. R9.3 의 KDE-based 보다 robust 한 likelihood ratio 추정. Huang et al. 2007 의 KMM 사용.

**Strategy 3 — Group-conditional CRC**. (specialty × stratum) joint partition 으로 CRC 적용. n_pos 부족 시 stratum 으로 fallback.

**비교.** 각 strategy 의 violation × 감소 + computational overhead 보고.

### R10.4 — Method-agnostic CRC head-to-head (R10 의 headline)

**5 underlying classifier** (1 LLM + 4 tabular):
1. `openai/gpt-oss-120b` (Round 10 의 유일한 LLM 백엔드 — size-scaling 비교 없음)
2. LogReg + decision-time tabular features
3. GBDT (sklearn `GradientBoostingClassifier`)
4. RandomForest (sklearn `RandomForestClassifier`)
5. XGBoost (`xgboost.XGBClassifier`)

**동일한 CRC layer**: 각 classifier 의 score (LLM 의 logprob, sklearn 의 `predict_proba(positive)`) 에 동일한 `StratifiedConformalRiskControl(alphas=(0.05, 0.10, 0.15, 0.20))` 적용.

**메트릭**:
- Per-stratum CRITICAL recall, Total cost (Round 7 cost matrix)
- exact Clopper-Pearson 95% upper bound for $\mathbb{E}[\ell_s]$
- McNemar pairwise vs LLM-CRC (does any tabular classifier statistically beat LLM?)
- Wallclock per inference + memory footprint
- Reproducibility cost (PhysioNet credentialing + sklearn vs +96GB Mac Studio)

**Hypothesis A (strong).** LogReg + CRC 가 LLM + CRC 보다 CRITICAL recall 에서 통계적으로 worse 하지 않음 (non-inferiority p > 0.05). 이 결과는 "LLM 의 value-add 는 없고 framework 의 가치가 전부" 라는 paper 의 핵심 주장 입증.

**Hypothesis B (weak).** LogReg + CRC 가 LLM + CRC 와 통계적으로 다름 (LLM 이 명확히 우세). 그러면 v2 의 LLM 부분의 가치를 quantify 가능.

### R10.5 — IRB physician adjudication (Phase 2, Round 7 L1 / Round 9 L11)

**프로토콜**.
- 100 cases 무작위 stratified-random (50 CRITICAL, 50 HIGH from MIMIC-IV)
- 3 board-certified physician 이 각자 escalate YES/NO + 1-sentence rationale
- inter-rater Cohen's κ 보고
- physician majority vote 라벨 vs outcome-derived 라벨의 confusion matrix
- 라벨 mismatch 시 outcome-derived 가 어디서 over/under-call 하는지 정량화

**IRB 상태**. paper/IRB_PROTOCOL.md §1-§9 protocol 의 직접 실행 — Round 9 의 §10 (PhysioNet DUA) 와 호환.

**예산**. USD 80/hr × 3 physician × ~10hr = $2,400 (예산 확보 필요)

### R10.6 — 4-D cost matrix sweep on real EHR (Round 7 §6.3 의 실 데이터 적용)

Round 7 supplementary 의 4-D sweep (CRITICAL/HIGH/MODERATE/LOW miss:over-esc ∈ {10:1, 100:1, 1000:1}) 81 조합을 MIMIC-IV CRITICAL stratum 에 적용.

각 조합에서 v2 cost vs simpler ablation cost 비교. **결과 패턴**:
- 1000:1 CRITICAL miss penalty 에서 v2 가 ablation 을 dominate 하는지
- 10:1 에서는 simpler 가 dominate 인지

**Paper 함의**. R9.2 의 "ablation 이 v2 보다 cheap" finding 이 특정 cost ratio 에서만 성립함을 quantify.

### R10.7 — Improved decision-time features for MODERATE/LOW

Round 9 R9.1 의 MODERATE 50% miss, LOW 100% miss 는 **decision-time feature 가 이 stratum 의 positive label 을 식별 못 함** 의 증거.

**Round 10 추가 feature** (preprocessing 업데이트):
- 직전 admission 의 outcome (있다면)
- Comorbidity index (Charlson, Elixhauser — ICD code 기반, 단 이전 admission 의)
- Vital sign quartile (chartevents 첫 6시간 — 현재 reserved 만 되어있음)
- Specialty risk multiplier (CMED, NMED 등의 베이스라인 admit-to-ICU rate)

**예상 효과**. MODERATE/LOW miss rate 가 합리적 수준 (≤ 0.30) 으로 감소. 안 되면 paper 가 "decision-time features 의 fundamental limit" 로 정직 보고.

### R10.8 — 통합 reporting

`experiments/round10_aggregate_report.py` 신규 — 8 단계 (R10.0-R10.7) 산출물 통합. R9 의 aggregate pattern 따름.

---

## 4. Code / 문서 수정 체크리스트

### 신규 파일

| 파일 | 역할 |
|---|---|
| `experiments/round10_multiseed_runner.py` | R10.x 의 5 시드 orchestration |
| `experiments/round10_alpha_005_empirical.py` | R10.1 — n_pos≥3000 CRITICAL |
| `experiments/round10_table4_multiseed.py` | R10.2 — 5-seed v2 vs baselines |
| `experiments/round10_distshift_mitigation.py` | R10.3 — 3 mitigation strategy |
| `experiments/round10_method_agnostic.py` | R10.4 — **headline** 5-classifier × CRC |
| `experiments/round10_physician_audit.py` | R10.5 — Cohen's κ + confusion matrix |
| `experiments/round10_cost_sweep_4d.py` | R10.6 — 81 조합 × MIMIC-IV |
| `experiments/round10_feature_expand.py` | R10.7 — comorbidity + vital quartile + specialty risk |
| `experiments/round10_aggregate_report.py` | 통합 |
| `run_all_round10.sh` | 마스터 runner |
| `tests/test_round10_loader.py` | R10 라벨/feature 검증 |
| `tests/test_round10_aggregate.py` | bootstrap CI 검증 |

### 수정될 파일

| 파일 | 변경 |
|---|---|
| `data/loader.py` | R10.7 의 expanded feature 지원 (`_load_mimic4_v10_jsonl`) |
| `experiments/round9_mimic4_preprocess.py` → `round10_mimic4_preprocess.py` 신규 | comorbidity index + vital quartile + specialty risk 추가 |
| `models/stratified_crc.py` | group-conditional CRC (R10.3) 지원 |
| `tests/test_paper_claims.py` | R10 regression guard |
| `improvements/round10_PLAN.md` (본 문서) | 진행 상황 업데이트 |
| `paper/UASEF_Round9.md` | "previous round" 로 reference 갱신 |

### 신규 paper

| 파일 | 역할 |
|---|---|
| `paper/UASEF_Round10.md` | Round 10 paper (영문) — method-agnostic CRC 헤드라인 |
| `paper/UASEF_Round10_KO.md` | 한국어 미러 |
| `paper/UASEF_Round10_Supplementary.md` | R10.5 IRB 결과, R10.6 4-D sweep, R10.7 feature 확장 |
| `paper/IRB_PROTOCOL.md` (수정) | R10.5 physician audit 의 §11 addendum |

---

## 5. 비용·시간 예산

| Phase | 작업 | OpenAI/API | Anthropic | LMStudio | 사람 | Wallclock |
|---|---|---|---|---|---|---|
| R10.0 | infrastructure | $0 | $0 | $0 | 1 dev × 1주 | — |
| R10.1 | n_pos≥3000 × 5 seed | $0 | $0 | ~120h | — | 5일 |
| R10.2 | 5-seed Table 4-MIMIC | $0 | $0 | ~25h | — | 1일 |
| R10.3 | 3 mitigation × 5 seed | $0 | $0 | ~40h | — | 2일 |
| R10.4 | 5 classifier × 5 seed (LLM 만 LMStudio, 4개는 sklearn 즉시) | $0 | $0 | ~120h | — | 5일 |
| R10.5 | IRB physician × 3 × 10hr | $0 | $0 | $0 | 3 physician = $2,400 | 4주 (실제 시간) |
| R10.6 | 81 × 1 seed × CRITICAL only | $0 | $0 | ~5h | — | 6시간 |
| R10.7 | preprocessing 재실행 + R10.1 재실행 | $0 | $0 | ~120h | — | 5일 |
| **합계** | | $0 | $0 | ~430h ≈ 18일 | $2,400 | ~5주 |

LLM wallclock 이 Mac Studio 의 18일 — 사용자가 다른 일 못함. 옵션:
- (a) 18일 그대로 진행 (헤드라인 quality 우선)
- (b) cohort 를 80% 줄이고 sample size 보고 (n_pos≈600, exact upper 가 0.05 한도 근처)
- (c) caffeinate 로 sleep 방지하고 background 실행, 그동안 R10.3/R10.6 등 CPU-bound 단계 병렬

권장: (c) — LLM 은 background, tabular + 4-D sweep 은 parallel. 그러면 18일 wallclock 이 사용자 작업 차단 안 함.

---

## 6. 우선순위 의존성

```
[R10.0 infra] → [R10.7 expanded features] → [R10.1 powered α=0.05]
                                            ↘
                                              [R10.4 method-agnostic] ← HEADLINE
                                            ↗
[R10.2 multi-seed Table 4] ─────────────────
[R10.3 mitigation] (parallel)
[R10.6 4-D cost sweep] (parallel)
[R10.5 IRB] (parallel, 4 weeks separate)
                                            ↓
                                    [R10.8 aggregate report]
                                            ↓
                                    [paper revision]
```

병렬화 가능한 단계가 많아서 R10.0/R10.7 만 끝나면 R10.1-6 모두 동시 진행 가능.

---

## 7. 위험 및 완화

| 위험 | 완화 |
|---|---|
| R10.7 의 expanded feature 가 MODERATE/LOW 를 여전히 식별 못함 | "decision-time feature 의 fundamental limit" 로 paper §8 L23 추가; LLM 의 narrative-level reasoning 이 어떻게 도움 되는지 R10.4 에서 측정 |
| R10.4 에서 LogReg/GBDT 가 LLM 을 명확히 능가 | paper headline 을 "method-agnostic" 그대로 유지하되 "LLM 의 value-add 는 narrative tasks 에서만" 으로 framing |
| R10.4 에서 LLM 이 LogReg/GBDT 를 명확히 능가 | "framework 가 LLM 의 strength 를 unlock" framing — 양쪽 다 supported 됨 |
| R10.5 의 IRB 라벨이 outcome-derived 와 κ < 0.5 | "outcome 라벨이 IRB 라벨 mismatch — outcome 이 더 보수적/덜 보수적" finding 으로 paper 강화 |
| R10.3 의 mitigation strategy 들이 모두 fail | "real EHR distribution shift 가 fundamental 한 challenge" 로 §8 L24 + Round 11 (multi-center) 의 동기 |
| 18일 wallclock 부담 | `caffeinate -dimsu` 로 sleep 방지하고 background 실행; tabular branch (R10.3/R10.6/R10.7) 와 parallel — 사용자 작업 차단 최소화 |
| Physician 섭외 어려움 | IRB 통과 후 가능 — Round 10 의 Phase 2 로 분리, R10.1-4 결과로 먼저 paper 작성 |

---

## 8. Submission timeline

| 일자 | 마일스톤 |
|---|---|
| 2026-06-12 | Round 10 plan + skeleton code commit |
| 2026-06-19 | R10.0 infra + R10.7 preprocessing 완료 |
| 2026-06-26 | R10.1-R10.6 실행 시작 (병렬) |
| 2026-07-08 | R10.1-R10.4 결과 도착 (preliminary) |
| 2026-07-15 | R10.5 IRB physician 시작 |
| 2026-08-12 | R10.5 IRB physician 완료 |
| 2026-08-19 | R10.8 aggregate report + paper drafting |
| 2026-08-26 | Paper revision (외부 review) |
| 2026-09-02 | ML4H 2026 submission |

---

## 9. 한 줄 명령

```bash
# Round 10 환경
export MIMIC4_DIR=~/Downloads/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1

# Phase 1: infrastructure + preprocessing (R10.0 + R10.7)
SKIP_R10_1=1 SKIP_R10_2=1 SKIP_R10_3=1 SKIP_R10_4=1 SKIP_R10_5=1 SKIP_R10_6=1 \
  bash run_all_round10.sh

# Phase 2: experiments (병렬 가능)
SKIP_PREPROCESS=1 SKIP_R10_0=1 SKIP_R10_5=1 \
  bash run_all_round10.sh

# Phase 3: aggregate + paper
SKIP_PREPROCESS=1 SKIP_R10_0=1 SKIP_R10_1=1 SKIP_R10_2=1 \
SKIP_R10_3=1 SKIP_R10_4=1 SKIP_R10_5=1 SKIP_R10_6=1 SKIP_R10_7=1 \
  bash run_all_round10.sh
```

---

_Round 10 plan 작성: 2026-06-11. Round 9 leakage-safe finding 의 직접 후속._
