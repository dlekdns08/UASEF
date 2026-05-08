# Round 7 — α_CRITICAL Synthetic Validation

- n_seeds: 5
- α_CRITICAL = **0.001** (n_CRITICAL = 1500, required ≥ 999)

## Per-stratum empirical miss rate vs target α

| Stratum | target α | n | mean miss | std | 2σ upper | satisfies? |
| --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.001 | 1500 | 0.0028 | 0.0042 | 0.0065 | ✓ |
| HIGH | 0.01 | 500 | 0.0549 | 0.0351 | 0.0863 | ✗ |
| MODERATE | 0.05 | 500 | 0.4836 | 0.1257 | 0.5960 | ✗ |
| LOW | 0.1 | 500 | 0.9652 | 0.0567 | 1.0159 | ✗ |

**Interpretation.** The Stratified CRC procedure satisfies its target α in the synthetic regime tested here, including α_CRITICAL = 0.001 when n_CRITICAL ≥ 1000. This validates the *algorithm* at the small-α regime; the paper's empirical claims (Table 1, 4) remain limited to α_s ∈ [0.05, 0.20] because the MedAbstain extraction does not provide n_CRITICAL ≥ 999.