# UASEF Full Evaluation Result

- **timestamp**: 20260507-110041
- **elapsed**: 0m 0s
- **config**:
  - backend: `openai`
  - n_cal: `200`
  - n_test: `100`
  - n_trials: `1000`
  - n_per_stratum: `200`
  - alpha: `0.1`
  - seed: `42`
  - skip_llm: `True`

## 0. 회귀 검증 (pytest)

_pytest 스킵됨_

## 1. Per-Stratum Coverage Validity (Pivot A — Stratified CRC)

각 risk stratum (CRITICAL/HIGH/MODERATE/LOW)에서 missed-escalation rate를 측정.
UASEF Round 7만이 stratum별 conformal risk control 보장을 충족해야 함.

_Table 1 SKIP (LLM 호출 필요)_

## 2. Multi-Trigger Combination FWER (Pivot B — Conformal Combination)

Null hypothesis (모든 trigger가 정상)에서 false escalation rate 측정.
v1 `len(triggers) > 0` → FWER 위반, v2 harmonic / e-value → 보장 충족.

- n_trials = 1000, α target = 0.05

| Method | Dependence | Empirical FWER | OK |
| --- | --- | --- | --- |
| v1: len(triggers) > 0 (naive OR) | independent | 0.109 | ✗ |
| v2: bonferroni | independent | 0.044 | ✓ |
| v2: harmonic | independent | 0.021 | ✓ |
| v2: e_value | independent | 0.046 | ✓ |
| v1: len(triggers) > 0 (naive OR) | correlated | 0.132 | ✗ |
| v2: bonferroni | correlated | 0.068 | ✓ |
| v2: harmonic | correlated | 0.038 | ✓ |
| v2: e_value | correlated | 0.072 | ✗ |

## 3. Cost-Weighted Performance (Pivot C — Cost-Aware Optimization)

비대칭 cost matrix (CRITICAL miss=1000× over_esc) 적용 시 total cost 비교.

- n_per_stratum = 200
- **Round 6 total cost**: `10845.0`
- **Round 7 total cost**: `348.0`
- **Cost reduction**: **31.16×** (Round 6 / Round 7)

### Per-stratum
| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over_esc |
| --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.829 | 10030.0 | -1.488 | 128.0 | 0.0 | 0.9481 |
| HIGH | 1.173 | 716.0 | -0.451 | 105.0 | 0.0 | 0.6522 |
| MODERATE | 1.08 | 64.0 | 0.449 | 84.0 | 0.1053 | 0.3536 |
| LOW | 0.924 | 35.0 | 1.076 | 31.0 | 0.2 | 0.1538 |

## 4. Head-to-Head Baseline Comparison

동일 calibration / test 풀에서 5개 method 비교:
TECP (Xu & Lu 2025), Quach 2024 CLM, Semantic Entropy (Farquhar Nature 2024), UASEF Round 6 (heuristic), **UASEF Round 7 (Stratified CRC + MTC + Cost-Aware)**.

_Table 4 SKIP (LLM 호출 필요)_

## 결론 요약

- **Pivot B (FWER)**: v1 naive OR FWER ≤ **0.132** (target 0.05 위반). v2 harmonic FWER ≤ **0.038** (충족).
- **Pivot C (Cost)**: Round 6 → Round 7 total cost **31.16× 감소** (10845 → 348).

---

_생성: `run_full_evaluation.sh` (2026-05-07T11:00:41)_
