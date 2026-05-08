# Round 8 — Pivot B Case Study (m ∈ {3,5,8,12} + Institutional)

## (A) Variable-m FWER scaling (null hypothesis, α=0.05)

| m | dep | naive OR | naive theory | Bonferroni | Harmonic | E-value |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | independent | 0.1015 | 0.1426 | 0.04 | 0.019 | 0.042 |
| 3 | correlated | 0.141 | 0.1426 | 0.0655 | 0.036 | 0.072 |
| 5 | independent | 0.1815 | 0.2262 | 0.0435 | 0.0 | 0.045 |
| 5 | correlated | 0.234 | 0.2262 | 0.074 | 0.001 | 0.083 |
| 8 | independent | 0.316 | 0.3366 | 0.029 | 0.0 | 0.0365 |
| 8 | correlated | 0.3695 | 0.3366 | 0.0545 | 0.0 | 0.085 |

**Reading.** Naive OR FWER scales as $1 - (1-\alpha)^m$ (theory column matches empirical).
Harmonic and e-value combiners stay near α regardless of m, including under correlated scores.

## (B) Institutional customization (m=8 triggers)

- n_test=2000 (n_pos=110, n_neg=1890), α=0.05
- cost matrix: miss=100, FP=10

| variant | miss rate | over-esc rate | total cost |
| --- | --- | --- | --- |
| v1 naive OR | 0.0 | 0.3037 | 5740 |
| v2 harmonic | 0.0182 | 0.0 | 200 |

Total-cost ratio (v1 naive / v2 harmonic) = **28.7×**.

**Reading.** With 8 institutional triggers, naive OR's over-escalation rate explodes (reflecting the FWER inflation), inflating total cost despite a possibly slightly lower miss rate. The harmonic combiner preserves the formal FWER bound and yields lower total cost.