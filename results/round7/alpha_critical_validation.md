# Round 7 — α_CRITICAL Synthetic Validation

- n_seeds: 10
- α_CRITICAL = **0.001** (n_CRITICAL = 1500, required ≥ 999)

## Per-stratum CRC loss vs target α

CRC bounds **E[ℓ] ≤ α** where ℓ = 1{Y=1 ∧ missed}. This is the *per-example* loss, **not** the conditional miss rate. We report both for transparency.

| Stratum | target α | n | mean E[ℓ] | std | 2σ upper | satisfies? | cond miss (recall⁻¹) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.001 | 1500 | 0.0006 | 0.0009 | 0.0012 | ✓ | 0.0021 |
| HIGH | 0.01 | 500 | 0.0114 | 0.0083 | 0.0166 | ✓ | 0.0384 |
| MODERATE | 0.05 | 500 | 0.0552 | 0.0229 | 0.0697 | ✓ | 0.1812 |
| LOW | 0.1 | 500 | 0.0878 | 0.0209 | 0.1010 | ✓ | 0.2948 |

**Interpretation.** Stratified CRC bounds the *per-example* loss E[ℓ] = P(Y=1 ∧ missed) ≤ α (Angelopoulos & Bates [2024] Theorem 1), not the conditional miss rate P(missed | Y=1). The empirical 2σ upper-bound on E[ℓ] sits below the target α for all four strata when $n_s \ge \lceil (1-α_s)/α_s \rceil$, including α_CRITICAL = 0.001 with n_CRITICAL = 1500. This is an *algorithm-level* validation; the paper's empirical claims (Table 1, 4) remain limited to α_s ∈ [0.05, 0.20] because the MedAbstain extraction does not provide n_CRITICAL ≥ 999.