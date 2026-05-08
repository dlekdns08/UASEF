# Round 7 Table 4 — Head-to-Head Baseline

- backend=openai, n_cal=200, n_test=100, α=0.1

## CRITICAL stratum
| Method | Safety Recall | Over-Esc | TP/FN/FP | Total cost |
| --- | --- | --- | --- | --- |
| TECP (Xu & Lu 2025) | 0.16 | None | 16/84/0 | 88941.0 |
| Quach 2024 CLM | 0.16 | None | 16/84/0 | 88941.0 |
| Semantic Entropy (Farquhar Nature 2024) | 0.16 | None | 16/84/0 | 88941.0 |
| UASEF Round 6 (heuristic multiplier) | 0.84 | None | 84/16/0 | 19940.0 |
| UASEF Round 7 (Stratified CRC + MTC + Cost-Aware) | 0.96 | None | 96/4/0 | 4374.0 |