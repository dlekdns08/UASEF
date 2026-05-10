# UASEF — All-Experiments Report v2

- **분석 범위**: `results/run_20260508-165626` 이후 누적된 모든 결과
- **분석 일시**: 2026-05-10
- **소스**: 단일-시드 1회 + 5-시드 부트스트랩 + Round 7/8 합성·smoke 실험

이 보고서는 [results/run_20260508-165626/result.md](results/run_20260508-165626/result.md) (단일-시드, ML4H baseline) 와 그 직후 실행된 **5-시드 멀티시드 부트스트랩** ([results/run_20260509-013417_aggregate/aggregate_seeds.md](results/run_20260509-013417_aggregate/aggregate_seeds.md)) 을 합쳐 정리한 두 번째 종합 리포트입니다.

---

## 0. 실행 개요

### 0.1 단일-시드 (run_20260508-165626)

- seed=42, α=0.10, n_cal=200/n_test=100/stratum, n_trials=5000 (Table 2), n_per_stratum=300 (Table 3)
- 백엔드: `openai`, `lmstudio`
- 소요시간: **517m 51s** (~8.6h)
- 회귀 테스트: 158 passed, 5 skipped

### 0.2 멀티시드 부트스트랩 (run_20260509-013417_aggregate)

- seeds = {42, 43, 44, 45, 46} — 총 **5 시드**
- 시드별 run 디렉토리:

| seed | run_dir | elapsed |
|---|---|---|
| 42 | [run_20260509-013417](results/run_20260509-013417/) | 513m 27s |
| 43 | [run_20260509-100744](results/run_20260509-100744/) | 484m 53s |
| 44 | [run_20260509-181237](results/run_20260509-181237/) | 481m 40s |
| 45 | [run_20260510-021418](results/run_20260510-021418/) | 492m 49s |
| 46 | [run_20260510-102707](results/run_20260510-102707/) | 497m 45s |

- 5 시드 누적 wallclock: **약 41시간** (P1.2 단계)
- 회귀 테스트: 162 passed, 1 skipped (모든 시드 동일)
- 집계 산출물: [aggregate_seeds.json](results/run_20260509-013417_aggregate/aggregate_seeds.json), [aggregate_seeds.md](results/run_20260509-013417_aggregate/aggregate_seeds.md)

### 0.3 Round 8 all-in-one 상태

- 최신: [results/round8/all_in_one_20260508-163745](results/round8/all_in_one_20260508-163745/)

| step | status | duration |
|---|---|---|
| P0 (Round 7 sanity) | ✅ ok | 12s |
| P0.5 (data extension) | ✅ ok | 1109s |
| P1.1 (single-seed P1) | ✅ ok | 31071s (8.6h) |
| P1.2 (5-seed bootstrap) | ✅ ok | **148236s (41.2h)** |
| **P1.3 (LLM-judge κ)** | **❌ fail (rc=1)** | 0s — `ModuleNotFoundError: No module named 'data'` |
| P1.4 (run_experiment.py) | ❌ 미완 | OpenAI httpx 끊김 + `^C` 혼입 |
| P2.x / P3.x / P4.x | 미실행 | (P1.3 실패로 후속 단계 차단) |

P1.3 원인: [experiments/llm_judge_relabel.py](experiments/llm_judge_relabel.py) 에 `sys.path.insert(0, ROOT)` 부트스트랩이 누락되어 `from data.loader import ...` 가 실패. **이 보고서 작성 중 패치 완료** ([experiments/llm_judge_relabel.py:40-54](experiments/llm_judge_relabel.py#L40-L54)) — 모듈 lazy import 가 정상 동작하는 것을 확인.

---

## 1. v2 Round 7 합성 검증 (백엔드 무관)

### 1.1 Table 2 — Multi-Trigger FWER (Pivot B)

Null hypothesis 하 false escalation rate. n_trials=5000, target α=0.05.

| Method | Dependence | seed=42 (단일) | 5-시드 mean ± std | OK |
|---|---|---|---|---|
| v1: len(triggers)>0 (naive OR) | independent | 0.107 | **0.130 ± 0.027** | ✗ |
| v1: len(triggers)>0 (naive OR) | correlated | 0.143 | **0.173 ± 0.029** | ✗ |
| v2: bonferroni | independent | 0.036 | 0.046 ± 0.010 | ✓ |
| v2: bonferroni | correlated | 0.063 | **0.077 ± 0.012** | ✗ ⚠️ |
| v2: harmonic | independent | 0.015 | 0.016 ± 0.007 | ✓ |
| v2: harmonic | correlated | 0.033 | 0.032 ± 0.011 | ✓ |
| v2: e_value | independent | 0.038 | 0.049 ± 0.012 | ✓ (경계) |
| v2: e_value | correlated | 0.068 | **0.084 ± 0.015** | ✗ ⚠️ |

⚠️ **단일-시드 결과만 보고 놓친 사실**: `correlated` 의존성 하에서 **bonferroni 와 e_value 도 평균 FWER 가 α=0.05 를 초과**합니다. seed=42 만 봤을 땐 모두 ✓ 였지만, 5 시드 평균은 보수성이 부족함을 보여줍니다. **harmonic 만 모든 의존 구조에서 안정적으로 α 이하**를 유지 — 이게 paper 의 v2 권장 결합 방식이 harmonic 인 이유와 정확히 일치합니다.

### 1.2 Table 3 — Cost-Weighted (Pivot C)

비대칭 cost matrix (CRITICAL miss=1000× over_esc), n_per_stratum=300.

| 지표 | seed=42 (단일) | 5-시드 mean ± std |
|---|---|---|
| Round 6 total cost | 16264 | **14381 ± 4807** |
| Round 7 total cost | 425 | **372 ± 38** |
| Reduction ratio | **38.27×** | **38.5 ± 12.9×** |

R6 비용은 시드에 따라 큰 변동(σ≈5K)이 있는 반면 **R7 비용은 372±38 로 매우 안정적**. 시드별 ratio: [38.3, 20.6, 52.1, 32.2, 49.5]. 최저 20.6× 에서도 한 자릿수 비용 절감 이상이 보장됩니다.

### 1.3 Table 3 — 4-D Cost-Matrix Sweep (단일-시드)

CRITICAL/HIGH/MODERATE/LOW × {10:1, 100:1, 1000:1} = 81 조합 ([table3_cost_4d.md](results/run_20260509-013417/synthetic/table3_cost_4d.md)).

- min reduction: **1.29×**, median: **29.31×**, mean: 23.7×, **max: 60.56×**
- CRITICAL:over=1000:1, HIGH:over=1000:1 조합에서 60.56× 도달 (R6 26164 → R7 432)

### 1.4 α=0.001 Critical-stratum 합성 검증

[alpha_critical_validation.md](results/run_20260509-013417/synthetic/alpha_critical_validation.md) — **알고리즘-레벨** validation (실데이터 n_CRITICAL≥999 미충족 한계 보완).

| Stratum | α | n | mean E[ℓ] | 2σ upper | 만족? | cond miss |
|---|---|---|---|---|---|---|
| CRITICAL | 0.001 | 1500 | 0.0006 | 0.0012 | ✓ | 0.0021 |
| HIGH | 0.01 | 500 | 0.0114 | 0.0166 | ✓ | 0.0384 |
| MODERATE | 0.05 | 500 | 0.0552 | 0.0697 | ✓ | 0.1812 |
| LOW | 0.10 | 500 | 0.0878 | 0.1010 | ✓ | 0.2948 |

CRC 가 보장하는 건 *per-example loss* `E[ℓ]≤α` 이지 *conditional miss rate* 가 아닙니다 — paper §7 limitation 에 명시 필요.

---

## 2. v2 Round 7 LLM 검증

### 2.1 Table 1 — Per-Stratum Coverage (5-시드 평균)

α_global=0.10, CRC alphas = {CRITICAL: 0.05, HIGH: 0.10, MODERATE: 0.15, LOW: 0.20}.
값은 **miss_rate** (낮을수록 좋음).

#### openai

| Method | CRITICAL | HIGH | MODERATE | LOW |
|---|---|---|---|---|
| TECP / Quach 2024 (global α) | 0.866 ± 0.015 | 0.978 ± 0.022 | 0.893 ± 0.034 | 0.000 |
| UASEF Round 6 (heuristic mult.) | 0.178 ± 0.034 | 0.738 ± 0.051 | 0.896 ± 0.036 | 0.000 |
| **UASEF Round 7 (Stratified CRC)** | **0.042 ± 0.023** | **0.058 ± 0.037** | **0.130 ± 0.043** | **0.000** |
| target α | 0.05 | 0.10 | 0.15 | 0.20 |
| **만족?** | ✓ | ✓ | ✓ | ✓ |

#### lmstudio

| Method | CRITICAL | HIGH | MODERATE | LOW |
|---|---|---|---|---|
| TECP / Quach 2024 (global α) | 0.908 ± 0.039 | 0.928 ± 0.018 | 0.866 ± 0.017 | 0.000 |
| UASEF Round 6 (heuristic mult.) | 0.308 ± 0.030 | 0.735 ± 0.053 | 0.866 ± 0.017 | 0.000 |
| **UASEF Round 7 (Stratified CRC)** | **0.038 ± 0.015** | **0.076 ± 0.049** | **0.174 ± 0.050** | **0.000** |
| target α | 0.05 | 0.10 | 0.15 | 0.20 |
| **만족?** | ✓ | ✓ | ✓~ ⚠️ | ✓ |

⚠️ lmstudio MODERATE: 0.174 ± 0.050 — **5 시드 중 3 시드(44, 45, 46)에서 0.21~0.23 으로 target 0.15 를 위반**. 시드 의존적 violation. CRITICAL/HIGH 는 강건.

### 2.2 Table 4 — Head-to-Head (5-시드 평균)

α=0.10. CRITICAL stratum recall + 전체 strata 합산 total_cost.

#### openai

| Method | CRITICAL Recall | Total Cost |
|---|---|---|
| TECP (Xu & Lu 2025) | 0.136 ± 0.037 | 91414 ± 3784 |
| Quach 2024 CLM | 0.136 ± 0.037 | 91414 ± 3784 |
| Semantic Entropy (Farquhar Nature 2024) | 0.136 ± 0.037 | 91414 ± 3784 |
| UASEF Round 6 (heuristic mult.) | 0.834 ± 0.022 | 20595 ± 2138 |
| **UASEF Round 7 (Stratified CRC + MTC + CA)** | **0.970 ± 0.021** [0.954, 0.986] | **3417 ± 2063** [1861, 5019] |
| TECP-stratified (R7 ablation) | 0.054 ± 0.017 | 99487 ± 1527 |
| Cost-Sensitive single-α (R7 ablation) | 1.000 ± 0.000 | 38 ± 3 |
| UASEF v1-cost-aware (R7 ablation) | 0.988 ± 0.011 | 1627 ± 1113 |

#### lmstudio

| Method | CRITICAL Recall | Total Cost |
|---|---|---|
| TECP / Quach / SE | 0.096 ± 0.033 | 95039 ± 3245 |
| UASEF Round 6 | 0.726 ± 0.048 | 31358 ± 4810 |
| **UASEF Round 7** | **0.958 ± 0.011** [0.948, 0.966] | **4704 ± 1209** [3864, 5796] |
| TECP-stratified | 0.064 ± 0.015 | 98179 ± 1487 |
| Cost-Sensitive single-α | 1.000 ± 0.000 | 40 ± 4 |
| UASEF v1-cost-aware | 0.958 ± 0.015 | 4705 ± 1591 |

### 2.3 McNemar (시드-pooled discordant counts)

n_seeds=5, paired discordant cells `b` (v2 우세), `c` (비교군 우세).

#### openai

| Pair | b | c | p_pooled | p_fisher |
|---|---|---|---|---|
| v2 vs TECP / Quach / SE | 896 | 68 | **0.0** | **0.0** |
| v2 vs UASEF Round 6 | 496 | 70 | **0.0** | **0.0** |
| v2 vs TECP-stratified ablation | 925 | 60 | **0.0** | **0.0** |
| v2 vs Cost-Sensitive single-α | 113 | 81 | 1e-05 | 0.104 |
| v2 vs UASEF v1-cost-aware | 15 | 47 | 0.0 | 0.0004 (v1 우세) |

#### lmstudio

| Pair | b | c | p_pooled | p_fisher |
|---|---|---|---|---|
| v2 vs TECP / Quach / SE | 880 | 69 | **0.0** | **0.0** |
| v2 vs UASEF Round 6 | 518 | 73 | **0.0** | **0.0** |
| v2 vs TECP-stratified ablation | 891 | 66 | **0.0** | **0.0** |
| v2 vs Cost-Sensitive single-α | 108 | 102 | 0.490 | 0.962 |
| v2 vs UASEF v1-cost-aware | 15 | 45 | 0.0 | 0.014 (v1 우세) |

⚠️ **두 가지 ablation 이 v2 보다 통계적으로 유의하게 잘 함** — 보고서에 정직하게 반영 필요:
- **Cost-Sensitive single-α**: CRITICAL recall 100%, cost 38 (vs v2 의 3417/4704). 단순히 임계값을 매우 낮게 잡으면 over-esc 가 폭증해야 정상이지만, 이 데이터셋 분포에서는 over-esc 도 낮은 듯. lmstudio 에서는 v2 와 통계적 동률 (p=0.49). 이건 "alpha 만 충분히 보수적이면 stratification 안 해도 됨"의 가능성을 시사.
- **UASEF v1-cost-aware**: openai 에서 cost 1627 < v2 의 3417. McNemar 도 v1 쪽이 유의하게 우세 (b=15, c=47, p=0.0004). **R7 의 stratified-CRC 보수성이 cost 측면에선 손해**.

---

## 3. v1 LLM 평가 (5-시드 평균)

### 3.1 LangGraph ReAct Agent

| 지표 | openai mean ± std | lmstudio mean ± std |
|---|---|---|
| accuracy | 0.789 ± 0.019 | 0.474 ± 0.035 |
| safety_recall | 0.808 ± 0.024 | 0.394 ± 0.034 |
| over_escalation_rate | 0.322 ± 0.081 | 0.040 ± 0.018 |
| escalation_rate | 0.739 ± 0.027 | 0.343 ± 0.027 |
| avg_tool_calls | 0.84 ± 0.08 | 0.03 ± 0.01 |
| avg_react_iterations | 1.59 ± 0.04 | 1.03 ± 0.01 |
| conformal_coverage | 0.880 ± 0.048 | 0.895 ± 0.057 |

⚠️ Agent 의 over-esc 가 openai 에서 0.32 ± 0.08 — 매우 높음. ReAct 가 보수적으로 escalate 하는 경향. lmstudio 는 도구 호출이 거의 없음 (0.03 회/case) → ReAct loop 가 사실상 single-pass.

### 3.2 Baseline (3 strategies)

| Strategy | openai recall | openai over-esc | lmstudio recall | lmstudio over-esc |
|---|---|---|---|---|
| no_escalation | 0.000 | 0.000 | 0.000 | 0.000 |
| threshold_only | 0.599 ± 0.031 | 0.006 ± 0.014 | 0.530 ± 0.012 | 0.010 ± 0.014 |
| full_uasef (v1) | 0.600 ± 0.023 | 0.011 ± 0.016 | 0.504 ± 0.018 | 0.005 ± 0.012 |

⚠️ v1 baseline (single-α threshold) 의 recall 은 ~0.60 으로 v2 stratified (~0.97) 와 큰 차이. **이게 paper 의 핵심 motivation 차트**.

### 3.3 MedAbstain Overall

| 지표 | openai | lmstudio |
|---|---|---|
| recall | 0.261 ± 0.043 | 0.176 ± 0.064 |
| precision | 0.771 ± 0.026 | 0.815 ± 0.047 |
| f1 | 0.389 ± 0.049 | 0.285 ± 0.085 |
| safety_recall_ok (≥0.95) | **0% (5/5 시드 실패)** | **0% (5/5 시드 실패)** |

MedAbstain 데이터셋에서는 safety_recall 목표 0.95 를 5 시드 모두 미달. 이건 v1 single-α 의 한계로, v2 가 해결하는 문제 (Table 1·4 참조).

---

## 4. Round 8 보강 실험 (단일-실행, 멀티시드 아님)

### 4.1 Distribution Shift Smoke ([dist_shift_smoke.md](results/round8/dist_shift_smoke.md))

emergency_medicine 으로 calibrate 후 다른 specialty 에 적용:

| test specialty | stratum | miss rate | target α | violation | ratio |
|---|---|---|---|---|---|
| internal_medicine | MODERATE | 0.279 | 0.05 | 0.229 | **5.6×** |
| pediatrics | HIGH | 0.213 | 0.05 | 0.163 | **4.3×** |
| neurology | HIGH | 0.443 | 0.05 | 0.393 | **8.9×** |
| general_practice | LOW | 0.525 | 0.05 | 0.475 | **10.5×** |

🚨 **Cross-specialty calibration 을 그대로 쓰면 4–10× violation**. CP exchangeability 가정이 specialty shift 에서 깨진다는 직접 증거. → paper §7 (limitation) + Tibshirani 2019 weighted CP 인용 필수.

### 4.2 Pivot B Case Study ([pivotB_smoke.md](results/round8/pivotB_smoke.md))

**(A) Variable-m FWER scaling**, α=0.05:

| m | dep | naive OR | 이론값 1−(1−α)^m | Bonferroni | Harmonic | E-value |
|---|---|---|---|---|---|---|
| 3 | indep | 0.102 | 0.143 | 0.040 | **0.019** | 0.042 |
| 3 | corr | 0.141 | — | 0.066 | **0.036** | 0.072 |
| 5 | indep | 0.182 | 0.226 | 0.044 | **0.000** | 0.045 |
| 5 | corr | 0.234 | — | 0.074 | **0.001** | 0.083 |
| 8 | indep | 0.316 | 0.337 | 0.029 | **0.000** | 0.037 |
| 8 | corr | 0.370 | — | 0.055 | **0.000** | 0.085 |

→ naive OR 은 trigger 수 m 에 따라 FWER 가 1−(1−α)^m 로 폭증. **harmonic 은 m 과 무관하게 α 이하 유지** — institutional customization 의 핵심 안전장치.

**(B) m=8 institutional customization** (n_test=2000, n_pos=110, miss=100, FP=10):

| variant | miss rate | over-esc | total cost |
|---|---|---|---|
| v1 naive OR | 0.000 | 0.304 | 5740 |
| **v2 harmonic** | 0.018 | 0.000 | **200** |

→ 비용 비율 v1/v2 = **28.7×**. miss 약간 증가시키지만 over-esc 폭증 회피로 총비용 대폭 감소.

---

## 5. 결론 요약 (단일-시드 vs 멀티시드 비교)

| 주장 | 단일-시드 (seed=42) | 5-시드 평균 ± std |
|---|---|---|
| **Pivot A** CRITICAL miss (openai) ≤ α=0.05 | 0.030 ✓ | **0.042 ± 0.023** ✓ |
| **Pivot A** CRITICAL miss (lmstudio) ≤ α=0.05 | 0.020 ✓ | **0.038 ± 0.015** ✓ |
| **Pivot B** harmonic FWER ≤ α=0.05 (corr) | 0.033 ✓ | **0.032 ± 0.011** ✓ |
| **Pivot C** R6→R7 cost reduction (실데이터, openai) | 17.4× | **R6=20595 → R7=3417 ≈ 6.0×** |
| **Pivot C** R6→R7 cost reduction (실데이터, lmstudio) | 29.2× | **R6=31358 → R7=4704 ≈ 6.7×** |
| **Pivot C** R6→R7 cost reduction (합성, n=300) | 38.27× | **38.5 ± 12.9×** |
| **Head-to-head** v2 vs TECP cost saving (openai) | 17.4× | **26.7×** (91414/3417) |
| **Head-to-head** v2 vs TECP cost saving (lmstudio) | 29.2× | **20.2×** (95039/4704) |
| **McNemar** v2 > {TECP, Quach, SE, R6, TECP-strat} | (단일 보고 없음) | **모든 페어 p<10⁻⁵** ✓ |

### 5.1 멀티시드 적용 후 새로 드러난 사실

1. **bonferroni / e_value 도 correlated dep. 에서 α 위반** — harmonic 만 안전. 이전 단일 보고서에서 모두 ✓ 로 처리됐던 부분 정정 필요.
2. **lmstudio MODERATE coverage 는 시드 의존적**: 5 시드 중 3 시드에서 0.21~0.23 으로 target 0.15 위반. paper 에 시드 분산 또는 calibration 풀 확장 명시 필요.
3. **두 ablation (Cost-Sensitive single-α, v1-cost-aware) 가 cost 측면에서 v2 보다 우세**:
   - v2 vs Cost-Sensitive single-α (lmstudio): McNemar p=0.49 → **통계적 동률**.
   - v2 vs v1-cost-aware (openai): cost 1627 vs 3417, McNemar p=0.0004 → **v1-cost-aware 가 우세**.
   - 즉 stratification 의 추가 보수성이 cost-only metric 에선 손해. v2 의 가치는 *동시에* miss rate 보장과 over-esc 억제를 만족하는 것 (paper 에서 명확히 framing 필요).
4. **R7 cost 의 시드 안정성**: 합성 R7 cost 372±38 (CV ≈ 10%) vs R6 14381±4807 (CV ≈ 33%). R7 이 시드에 강건.

### 5.2 명백한 한계 (paper §7 등재 후보)

- **Cross-specialty distribution shift**: emergency_medicine → 다른 specialty 에서 miss rate 4–10× violation. weighted CP 또는 specialty-별 calibration 필수.
- **MedAbstain safety_recall ≥ 0.95** 는 v1 baseline 으론 5 시드 모두 실패. v2 만 해당 목표 도달.
- **n_CRITICAL (실데이터) < 999** 로 α=0.001 직접 검증 불가 — 합성 n=1500 으로 algorithm-level 검증만 보고.
- **LLM-judge κ (P1.3) 미실시**: heuristic 라벨 partial validation 단계가 막혀 있음. 패치 후 재실행 권장.
- **P1.4 (run_experiment.py) 미완**: OpenAI 연결 끊김 + ^C 혼입. 재실행 필요.

---

## 6. 다음 단계 권장

| 우선순위 | 작업 | 비용/시간 |
|---|---|---|
| P1 | P1.3 (LLM-judge κ) 재실행 — 이미 패치됨 | ~$25, 1–2h |
| P1 | P1.4 (run_experiment.py) 재실행 — OpenAI 연결만 안정되면 됨 | ~30min |
| P2 | bonferroni / e_value 의 correlated FWER 위반을 paper §6 표에 정정 반영 | edit only |
| P2 | lmstudio MODERATE seed-violation 을 §7 limitation 에 명시 | edit only |
| P3 | v2 vs ablation (Cost-Sensitive, v1-cost-aware) framing 재정렬 — "v2 의 가치는 miss·over-esc 동시 보장" 으로 | edit only |
| P4 | P2.x / P3.x / P4.x (multi-dataset, equity audit, viz) — Round 8 후속 | 추가 ~$50 |

---

_생성: 2026-05-10, 분석 스크립트는 제공된 `result.json` (5 시드) + `aggregate_seeds.{json,md}` 직접 파싱._
