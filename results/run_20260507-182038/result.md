# UASEF — Full Evaluation Report

- **timestamp**: `20260507-182038`
- **elapsed**: `525m 21s`
- **backends**: `openai`, `lmstudio`

### Config

| key | value |
| --- | --- |
| `n_cal` | `200` |
| `n_test` | `100` |
| `n_medabstain` | `50` |
| `n_pareto` | `50` |
| `n_trials` | `5000` |
| `n_per_stratum` | `300` |
| `alpha` | `0.1` |
| `seed` | `42` |
| `skip_llm` | `False` |
| `skip_tests` | `False` |
| `skip_v1` | `False` |
| `skip_v2_syn` | `False` |
| `skip_v2_llm` | `False` |
| `backends` | `['openai', 'lmstudio']` |

## 0. 회귀 검증 (pytest 137 tests)

```
collected 137 items
tests/test_config_schema.py ........                                     [  5%]
tests/test_conformal_combination.py .....................                [ 21%]
tests/test_cost_aware.py ............                                    [ 29%]
tests/test_loader.py ........                                            [ 35%]
tests/test_metrics.py .............                                      [ 45%]
tests/test_round7_integration.py .........                               [ 51%]
tests/test_rtc_ede.py .............                                      [ 61%]
tests/test_runners_smoke.py .............                                [ 70%]
tests/test_stratified_crc.py .............                               [ 80%]
tests/test_uqm.py ...........................                            [100%]
============================= 137 passed in 7.03s ==============================
```

## 1. v2 Round 7 합성 검증 (backend 무관)

### 1.1 Table 2 — Multi-Trigger FWER (Pivot B)

Null hypothesis (모든 trigger가 정상)에서 false escalation rate 측정.
v1 `len(triggers)>0`은 FWER 위반, v2 harmonic / e-value는 보장 충족.

- n_trials=5000, target α=0.05

| Method | Dependence | Empirical FWER | OK |
| --- | --- | --- | --- |
| v1: len(triggers) > 0 (naive OR) | independent | 0.107 | ✗ |
| v2: bonferroni | independent | 0.0364 | ✓ |
| v2: harmonic | independent | 0.0152 | ✓ |
| v2: e_value | independent | 0.0376 | ✓ |
| v1: len(triggers) > 0 (naive OR) | correlated | 0.1434 | ✗ |
| v2: bonferroni | correlated | 0.0628 | ✓ |
| v2: harmonic | correlated | 0.0328 | ✓ |
| v2: e_value | correlated | 0.0678 | ✓ |

### 1.2 Table 3 — Cost-Weighted Performance (Pivot C)

비대칭 cost matrix (CRITICAL miss=1000× over_esc) 적용.

- n_per_stratum=300
- **Round 6 total cost**: `16264.0`
- **Round 7 total cost**: `425.0`
- **Reduction**: **38.27×** (Round 6 / Round 7)

#### Per-stratum
| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over_esc |
| --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.941 | 15041.0 | -1.488 | 199.0 | 0.0 | 0.9476 |
| HIGH | 1.411 | 1120.0 | -0.04 | 130.0 | 0.0 | 0.5328 |
| MODERATE | 0.83 | 62.0 | 0.83 | 62.0 | 0.0 | 0.2222 |
| LOW | 1.02 | 41.0 | 1.249 | 34.0 | 0.1667 | 0.1111 |

## 2. Backend별 결과

### 2.1 Backend: `openai`

#### 2.1.1 v1 — `run_all_experiments.py` (agent + baseline + medabstain + pareto)

**Agent (LangGraph ReAct)**

- accuracy: `0.7588` | safety_recall: `0.7489` | over_esc: `0.1842` | escalation_rate: `0.6654`
- avg_tool_calls: `0.84` | avg_react_iterations: `1.59`
- coverage: `0.925`

**Baseline (3 strategies)**

| Strategy | Safety Recall | 95% CI | Over-Esc | OK |
| --- | --- | --- | --- | --- |
| no_escalation | 0.0 | [0.000,0.017] | 0.0 | ✗ |
| threshold_only | 0.5434 | [0.477,0.608] | 0.0263 | ✗ |
| full_uasef | 0.5479 | [0.482,0.612] | 0.0 | ✗ |

**MedAbstain 전체**

- Recall: `0.1533` | Precision: `0.7667` | F1: `0.2555` | AUROC: `None`
- Abstention: P=`0.0` R=`0.0` F1=`0.0`

_상세 보고서_: [`openai/all_experiments_report.md`](openai/all_experiments_report.md)

#### 2.1.2 v2 Round 7 Table 1 — Per-Stratum Coverage (Pivot A)

- n_cal=200/stratum, n_test=100/stratum
- α_global=0.1, CRC alphas={'CRITICAL': 0.05, 'HIGH': 0.1, 'MODERATE': 0.15, 'LOW': 0.2}

| Method | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss |
| --- | --- | --- | --- | --- |
| TECP / Quach 2024 (global α) | 0.89 | 1.0 | 0.9028 | 0.0 |
| UASEF Round 6 (heuristic multiplier) | 0.16 | 0.8222 | 0.9028 | 0.0 |
| UASEF Round 7 (Stratified CRC) | 0.03 | 0.0444 | 0.0694 | 0.0 |

#### 2.1.3 v2 Round 7 Table 4 — Head-to-Head (TECP / Quach / SE / R6 / R7)

- n_cal=200, n_test=100, α=0.1

##### CRITICAL stratum
| Method | Safety Recall | Over-Esc | TP/FN/FP | Total cost |
| --- | --- | --- | --- | --- |
| TECP (Xu & Lu 2025) | 0.16 | None | 16/84/0 | 88941.0 |
| Quach 2024 CLM | 0.16 | None | 16/84/0 | 88941.0 |
| Semantic Entropy (Farquhar Nature 2024) | 0.16 | None | 16/84/0 | 88941.0 |
| UASEF Round 6 (heuristic multiplier) | 0.84 | None | 84/16/0 | 19940.0 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 0.96 | None | 96/4/0 | 4374.0 |

##### Total cost (전 stratum)
| Method | Total cost |
| --- | --- |
| TECP (Xu & Lu 2025) | 88941.0 |
| Quach 2024 CLM | 88941.0 |
| Semantic Entropy (Farquhar Nature 2024) | 88941.0 |
| UASEF Round 6 (heuristic multiplier) | 19940.0 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 4374.0 |

### 2.2 Backend: `lmstudio`

#### 2.2.1 v1 — `run_all_experiments.py` (agent + baseline + medabstain + pareto)

**Agent (LangGraph ReAct)**

- accuracy: `0.463` | safety_recall: `0.3699` | over_esc: `0.0` | escalation_rate: `0.3152`
- avg_tool_calls: `0.04` | avg_react_iterations: `1.04`
- coverage: `0.95`

**Baseline (3 strategies)**

| Strategy | Safety Recall | 95% CI | Over-Esc | OK |
| --- | --- | --- | --- | --- |
| no_escalation | 0.0 | [0.000,0.017] | 0.0 | ✗ |
| threshold_only | 0.5114 | [0.446,0.577] | 0.0 | ✗ |
| full_uasef | 0.4932 | [0.428,0.559] | 0.0 | ✗ |

**MedAbstain 전체**

- Recall: `0.0872` | Precision: `0.7222` | F1: `0.1556` | AUROC: `None`
- Abstention: P=`0.0` R=`0.0` F1=`0.0`

_상세 보고서_: [`lmstudio/all_experiments_report.md`](lmstudio/all_experiments_report.md)

#### 2.2.2 v2 Round 7 Table 1 — Per-Stratum Coverage (Pivot A)

- n_cal=200/stratum, n_test=100/stratum
- α_global=0.1, CRC alphas={'CRITICAL': 0.05, 'HIGH': 0.1, 'MODERATE': 0.15, 'LOW': 0.2}

| Method | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss |
| --- | --- | --- | --- | --- |
| TECP / Quach 2024 (global α) | 0.9 | 0.9318 | 0.8873 | 0.0 |
| UASEF Round 6 (heuristic multiplier) | 0.31 | 0.7045 | 0.8873 | 0.0 |
| UASEF Round 7 (Stratified CRC) | 0.04 | 0.0682 | 0.1408 | 0.0 |

#### 2.2.3 v2 Round 7 Table 4 — Head-to-Head (TECP / Quach / SE / R6 / R7)

- n_cal=200, n_test=100, α=0.1

##### CRITICAL stratum
| Method | Safety Recall | Over-Esc | TP/FN/FP | Total cost |
| --- | --- | --- | --- | --- |
| TECP (Xu & Lu 2025) | 0.1 | None | 10/90/0 | 94633.0 |
| Quach 2024 CLM | 0.1 | None | 10/90/0 | 94633.0 |
| Semantic Entropy (Farquhar Nature 2024) | 0.1 | None | 10/90/0 | 94633.0 |
| UASEF Round 6 (heuristic multiplier) | 0.7 | None | 70/30/0 | 33730.0 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 0.96 | None | 96/4/0 | 4442.0 |

##### Total cost (전 stratum)
| Method | Total cost |
| --- | --- |
| TECP (Xu & Lu 2025) | 94633.0 |
| Quach 2024 CLM | 94633.0 |
| Semantic Entropy (Farquhar Nature 2024) | 94633.0 |
| UASEF Round 6 (heuristic multiplier) | 33730.0 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 4442.0 |

## 3. 결론 요약

- **Pivot B (FWER)**: v1 `len(triggers)>0` empirical FWER ≤ **0.1434** (target 0.05 위반). v2 harmonic FWER ≤ **0.0328** (✓).
- **Pivot C (Cost)**: Round 6 → Round 7 total cost **38.27× 감소** (16264 → 425).
- **Pivot A coverage (openai)**: CRITICAL stratum miss_rate=0.03 vs target α=0.05 → ✓
- **Pivot A coverage (lmstudio)**: CRITICAL stratum miss_rate=0.04 vs target α=0.05 → ✓
- **Head-to-head cost (openai)**: TECP cost=88941 vs UASEF Round 7 cost=4374 → **20.3× 절감**
- **Head-to-head cost (lmstudio)**: TECP cost=94633 vs UASEF Round 7 cost=4442 → **21.3× 절감**

---

_생성: `run_full_evaluation.sh` (2026-05-08T03:05:59)_
