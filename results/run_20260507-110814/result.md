# UASEF — Full Evaluation Report

- **timestamp**: `20260507-110814`
- **elapsed**: `0m 0s`
- **backends**: `openai`

### Config

| key | value |
| --- | --- |
| `n_cal` | `200` |
| `n_test` | `100` |
| `n_medabstain` | `50` |
| `n_pareto` | `50` |
| `n_trials` | `500` |
| `n_per_stratum` | `200` |
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

- n_trials=500, target α=0.05

| Method | Dependence | Empirical FWER | OK |
| --- | --- | --- | --- |
| v1: len(triggers) > 0 (naive OR) | independent | 0.104 | ✗ |
| v2: bonferroni | independent | 0.046 | ✓ |
| v2: harmonic | independent | 0.022 | ✓ |
| v2: e_value | independent | 0.05 | ✓ |
| v1: len(triggers) > 0 (naive OR) | correlated | 0.132 | ✗ |
| v2: bonferroni | correlated | 0.072 | ✗ |
| v2: harmonic | correlated | 0.042 | ✓ |
| v2: e_value | correlated | 0.08 | ✗ |

### 1.2 Table 3 — Cost-Weighted Performance (Pivot C)

비대칭 cost matrix (CRITICAL miss=1000× over_esc) 적용.

- n_per_stratum=200
- **Round 6 total cost**: `10845.0`
- **Round 7 total cost**: `348.0`
- **Reduction**: **31.16×** (Round 6 / Round 7)

#### Per-stratum
| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over_esc |
| --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.829 | 10030.0 | -1.488 | 128.0 | 0.0 | 0.9481 |
| HIGH | 1.173 | 716.0 | -0.451 | 105.0 | 0.0 | 0.6522 |
| MODERATE | 1.08 | 64.0 | 0.449 | 84.0 | 0.1053 | 0.3536 |
| LOW | 0.924 | 35.0 | 1.076 | 31.0 | 0.2 | 0.1538 |

## 2. Backend별 결과

### 2.1 Backend: `openai`

#### 2.1.1 v1 — `run_all_experiments.py` (agent + baseline + medabstain + pareto)

_v1 SKIP 또는 실패_

#### 2.1.2 v2 Round 7 Table 1 — Per-Stratum Coverage (Pivot A)

_Table 1 SKIP_

#### 2.1.3 v2 Round 7 Table 4 — Head-to-Head (TECP / Quach / SE / R6 / R7)

_Table 4 SKIP_

## 3. 결론 요약

- **Pivot B (FWER)**: v1 `len(triggers)>0` empirical FWER ≤ **0.132** (target 0.05 위반). v2 harmonic FWER ≤ **0.042** (✓).
- **Pivot C (Cost)**: Round 6 → Round 7 total cost **31.16× 감소** (10845 → 348).

---

_생성: `run_full_evaluation.sh` (2026-05-07T11:08:14)_
