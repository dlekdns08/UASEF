# Round 7 Table 1 — Per-Stratum Coverage Validity

- backend: `openai`, n_cal=200/stratum, n_test=100/stratum

| Method | CRITICAL miss | HIGH miss | MODERATE miss | LOW miss |
| --- | --- | --- | --- | --- |
| TECP / Quach 2024 (global α) | 0.89 | 1.0 | 0.9028 | 0.0 |
| UASEF Round 6 (heuristic multiplier) | 0.16 | 0.8222 | 0.9028 | 0.0 |
| UASEF Round 7 (Stratified CRC) | 0.03 | 0.0444 | 0.0694 | 0.0 |