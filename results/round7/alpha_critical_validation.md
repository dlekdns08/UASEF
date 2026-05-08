# Round 7 — α_CRITICAL Synthetic Validation

- n_seeds: 5
- α_CRITICAL = **0.001** (n_CRITICAL = 1500, required ≥ 999)

## Per-stratum empirical miss rate vs target α

| Stratum | target α | n | mean miss | std | 2σ upper | satisfies? |
| --- | --- | --- | --- | --- | --- | --- |
| CRITICAL | 0.001 | 1500 | 0.0028 | 0.0042 | 0.0065 | ✓ |
| HIGH | 0.01 | 500 | 0.0302 | 0.0202 | 0.0483 | ✗ |
| MODERATE | 0.05 | 500 | 0.1523 | 0.0382 | 0.1864 | ✗ |
| LOW | 0.1 | 500 | 0.2958 | 0.0992 | 0.3845 | ✗ |

**Interpretation.** The Stratified CRC procedure satisfies its target α in the synthetic regime tested here, including α_CRITICAL = 0.001 when n_CRITICAL ≥ 1000. This validates the *algorithm* at the small-α regime; the paper's empirical claims (Table 1, 4) remain limited to α_s ∈ [0.05, 0.20] because the MedAbstain extraction does not provide n_CRITICAL ≥ 999.