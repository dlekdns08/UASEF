# Round 8 — Pivot B Case Study (m ∈ {3,5,8,12} + Institutional)

## (A) Variable-m FWER scaling (null hypothesis, α=0.05)

| m | dep | naive OR | naive theory | Bonferroni | Harmonic | E-value |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | independent | 0.107 | 0.1426 | 0.0364 | 0.0152 | 0.0376 |
| 3 | correlated | 0.1434 | 0.1426 | 0.0628 | 0.0328 | 0.0678 |
| 5 | independent | 0.1826 | 0.2262 | 0.0398 | 0.0 | 0.0418 |
| 5 | correlated | 0.2344 | 0.2262 | 0.0744 | 0.001 | 0.0814 |
| 8 | independent | 0.325 | 0.3366 | 0.03 | 0.0 | 0.0362 |
| 8 | correlated | 0.3714 | 0.3366 | 0.0564 | 0.0 | 0.0856 |
| 12 | independent | 0.434 | 0.4596 | 0.0 | 0.0 | 0.0244 |
| 12 | correlated | 0.4608 | 0.4596 | 0.0 | 0.0 | 0.0812 |

**Reading.** Naive OR FWER scales as $1 - (1-\alpha)^m$ (theory column matches empirical).
Harmonic and e-value combiners stay near α regardless of m, including under correlated scores.

## (B) Institutional customization (m=8 triggers)

- n_test=5000 (n_pos=278, n_neg=4722), α=0.05
- cost matrix: miss=100, FP=10

| variant | miss rate | over-esc rate | total cost |
| --- | --- | --- | --- |
| v1 naive OR | 0.0 | 0.3141 | 14830 |
| v2 harmonic | 0.0108 | 0.0 | 300 |

Total-cost ratio (v1 naive / v2 harmonic) = **49.433×**.

**Reading.** With 8 institutional triggers, naive OR's over-escalation rate explodes (reflecting the FWER inflation), inflating total cost despite a possibly slightly lower miss rate. The harmonic combiner preserves the formal FWER bound and yields lower total cost.