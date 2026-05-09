# Round 7 Table 2 — Multi-Trigger Combination FWER

- n_trials=5000, n_cal=200, target α=0.05

| Method | Dependence | Empirical FWER | OK (≤α+0.02)? |
| --- | --- | --- | --- |
| v1: len(triggers) > 0 (naive OR) | independent | 0.1196 | ✗ |
| v2: bonferroni | independent | 0.0462 | ✓ |
| v2: harmonic | independent | 0.0218 | ✓ |
| v2: e_value | independent | 0.0476 | ✓ |
| v1: len(triggers) > 0 (naive OR) | correlated | 0.167 | ✗ |
| v2: bonferroni | correlated | 0.0738 | ✗ |
| v2: harmonic | correlated | 0.0348 | ✓ |
| v2: e_value | correlated | 0.0794 | ✗ |