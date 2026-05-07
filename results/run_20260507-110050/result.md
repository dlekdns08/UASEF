# UASEF Full Evaluation Result

- **timestamp**: 20260507-110050
- **elapsed**: 0m 6s
- **config**:
  - backend: `openai`
  - n_cal: `200`
  - n_test: `100`
  - n_trials: `500`
  - n_per_stratum: `150`
  - alpha: `0.1`
  - seed: `42`
  - skip_llm: `True`

## 0. 회귀 검증 (pytest)

```
tests/test_cost_aware.py ............                                    [ 29%]
tests/test_loader.py ........                                            [ 35%]
tests/test_metrics.py .............                                      [ 45%]
tests/test_round7_integration.py .........                               [ 51%]
tests/test_rtc_ede.py .............                                      [ 61%]
tests/test_runners_smoke.py .............                                [ 70%]
tests/test_stratified_crc.py .............                               [ 80%]
tests/test_uqm.py ...........................                            [100%]
============================= 137 passed in 5.62s ==============================
```

## 1. Per-Stratum Coverage Validity (Pivot A — Stratified CRC)

각 risk stratum (CRITICAL/HIGH/MODERATE/LOW)에서 missed-escalation rate를 측정.
UASEF Round 7만이 stratum별 conformal risk control 보장을 충족해야 함.

_Table 1 SKIP (LLM 호출 필요)_

## 2. Multi-Trigger Combination FWER (Pivot B — Conformal Combination)

Null hypothesis (모든 trigger가 정상)에서 false escalation rate 측정.
v1 `len(triggers) > 0` → FWER 위반, v2 harmonic / e-value → 보장 충족.

- n_trials = 500, α target = 0.05

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

## 3. Cost-Weighted Performance (Pivot C — Cost-Aware Optimization)

비대칭 cost matrix (CRITICAL miss=1000× over_esc) 적용 시 total cost 비교.

- n_per_stratum = 150
- **Round 6 total cost**: `7608.0`
- **Round 7 total cost**: `253.0`
- **Cost reduction**: **30.07×** (Round 6 / Round 7)

### Per-stratum
| Stratum | R6 thr | R6 cost | R7 thr | R7 cost | R7 miss | R7 over_esc |
| --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.829 | 7020.0 | -1.488 | 93.0 | 0.0 | 0.9588 |
| HIGH | 1.39 | 508.0 | -0.451 | 87.0 | 0.0 | 0.6905 |
| MODERATE | 0.969 | 53.0 | 0.646 | 52.0 | 0.1333 | 0.237 |
| LOW | 0.919 | 27.0 | 1.08 | 21.0 | 0.1667 | 0.1389 |

## 4. Head-to-Head Baseline Comparison

동일 calibration / test 풀에서 5개 method 비교:
TECP (Xu & Lu 2025), Quach 2024 CLM, Semantic Entropy (Farquhar Nature 2024), UASEF Round 6 (heuristic), **UASEF Round 7 (Stratified CRC + MTC + Cost-Aware)**.

_Table 4 SKIP (LLM 호출 필요)_

## 결론 요약

- **Pivot B (FWER)**: v1 naive OR FWER ≤ **0.132** (target 0.05 위반). v2 harmonic FWER ≤ **0.042** (충족).
- **Pivot C (Cost)**: Round 6 → Round 7 total cost **30.07× 감소** (7608 → 253).

---

_생성: `run_full_evaluation.sh` (2026-05-07T11:00:56)_
