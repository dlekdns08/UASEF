# UASEF — Full Evaluation Report

- **timestamp**: `20260507-160243`
- **elapsed**: `0m 0s`
- **backends**: `openai`

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
| `skip_llm` | `True` |
| `skip_tests` | `True` |
| `skip_v1` | `True` |
| `skip_v2_syn` | `False` |
| `skip_v2_llm` | `True` |
| `backends` | `['openai']` |

## 0. 회귀 검증 (pytest 137 tests)

_pytest 스킵됨_

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

_v1 SKIP 또는 실패_

#### 2.1.2 v2 Round 7 Table 1 — Per-Stratum Coverage (Pivot A)

_Table 1 SKIP_

#### 2.1.3 v2 Round 7 Table 4 — Head-to-Head (TECP / Quach / SE / R6 / R7)

_Table 4 SKIP_

## 3. 결론 요약

- **Pivot B (FWER)**: v1 `len(triggers)>0` empirical FWER ≤ **0.1434** (target 0.05 위반). v2 harmonic FWER ≤ **0.0328** (✓).
- **Pivot C (Cost)**: Round 6 → Round 7 total cost **38.27× 감소** (16264 → 425).

---

_생성: `run_full_evaluation.sh` (2026-05-07T16:02:43)_
