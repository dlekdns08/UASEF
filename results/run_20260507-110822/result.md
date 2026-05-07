# UASEF — Full Evaluation Report

- **timestamp**: `20260507-110822`
- **elapsed**: `0m 6s`
- **backends**: `openai`, `lmstudio`

### Config

| key | value |
| --- | --- |
| `n_cal` | `200` |
| `n_test` | `100` |
| `n_medabstain` | `50` |
| `n_pareto` | `50` |
| `n_trials` | `300` |
| `n_per_stratum` | `150` |
| `alpha` | `0.1` |
| `seed` | `42` |
| `skip_llm` | `True` |
| `skip_tests` | `False` |
| `skip_v1` | `True` |
| `skip_v2_syn` | `False` |
| `skip_v2_llm` | `True` |
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
============================= 137 passed in 5.86s ==============================
```

## 1. v2 Round 7 합성 검증 (backend 무관)

### 1.1 Table 2 — Multi-Trigger FWER (Pivot B)

Null hypothesis (모든 trigger가 정상)에서 false escalation rate 측정.
v1 `len(triggers)>0`은 FWER 위반, v2 harmonic / e-value는 보장 충족.

- n_trials=300, target α=0.05

| Method | Dependence | Empirical FWER | OK |
| --- | --- | --- | --- |
| v1: len(triggers) > 0 (naive OR) | independent | 0.09 | ✗ |
| v2: bonferroni | independent | 0.03 | ✓ |
| v2: harmonic | independent | 0.0167 | ✓ |
| v2: e_value | independent | 0.03 | ✓ |
| v1: len(triggers) > 0 (naive OR) | correlated | 0.11 | ✗ |
| v2: bonferroni | correlated | 0.0567 | ✓ |
| v2: harmonic | correlated | 0.0333 | ✓ |
| v2: e_value | correlated | 0.0667 | ✓ |

### 1.2 Table 3 — Cost-Weighted Performance (Pivot C)

비대칭 cost matrix (CRITICAL miss=1000× over_esc) 적용.

- n_per_stratum=150
- **Round 6 total cost**: `7608.0`
- **Round 7 total cost**: `253.0`
- **Reduction**: **30.07×** (Round 6 / Round 7)

#### Per-stratum
| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over_esc |
| --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.829 | 7020.0 | -1.488 | 93.0 | 0.0 | 0.9588 |
| HIGH | 1.39 | 508.0 | -0.451 | 87.0 | 0.0 | 0.6905 |
| MODERATE | 0.969 | 53.0 | 0.646 | 52.0 | 0.1333 | 0.237 |
| LOW | 0.919 | 27.0 | 1.08 | 21.0 | 0.1667 | 0.1389 |

## 2. Backend별 결과

### 2.1 Backend: `openai`

#### 2.1.1 v1 — `run_all_experiments.py` (agent + baseline + medabstain + pareto)

_v1 SKIP 또는 실패_

#### 2.1.2 v2 Round 7 Table 1 — Per-Stratum Coverage (Pivot A)

_Table 1 SKIP_

#### 2.1.3 v2 Round 7 Table 4 — Head-to-Head (TECP / Quach / SE / R6 / R7)

_Table 4 SKIP_

### 2.2 Backend: `lmstudio`

#### 2.2.1 v1 — `run_all_experiments.py` (agent + baseline + medabstain + pareto)

_v1 SKIP 또는 실패_

#### 2.2.2 v2 Round 7 Table 1 — Per-Stratum Coverage (Pivot A)

_Table 1 SKIP_

#### 2.2.3 v2 Round 7 Table 4 — Head-to-Head (TECP / Quach / SE / R6 / R7)

_Table 4 SKIP_

## 3. 결론 요약

- **Pivot B (FWER)**: v1 `len(triggers)>0` empirical FWER ≤ **0.11** (target 0.05 위반). v2 harmonic FWER ≤ **0.0333** (✓).
- **Pivot C (Cost)**: Round 6 → Round 7 total cost **30.07× 감소** (7608 → 253).

---

_생성: `run_full_evaluation.sh` (2026-05-07T11:08:28)_
