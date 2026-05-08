# Round 7 Table 4 — Head-to-Head Baseline

- backend=lmstudio, n_cal=200, n_test=100, α=0.1

## CRITICAL stratum
| Method | Safety Recall | Over-Esc | TP/FN/FP | Total cost |
| --- | --- | --- | --- | --- |
| TECP (Xu & Lu 2025) | 0.1 | None | 10/90/0 | 94633.0 |
| Quach 2024 CLM | 0.1 | None | 10/90/0 | 94633.0 |
| Semantic Entropy (Farquhar Nature 2024) | 0.1 | None | 10/90/0 | 94633.0 |
| UASEF Round 6 (heuristic multiplier) | 0.7 | None | 70/30/0 | 33730.0 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 0.96 | None | 96/4/0 | 4442.0 |