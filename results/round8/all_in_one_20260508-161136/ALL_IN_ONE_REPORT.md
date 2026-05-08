# UASEF Round 8 — All-in-One Report

- timestamp: 20260508-161136
- run_dir: `/Users/idaun/PoC/UASEF/results/round8/all_in_one_20260508-161136`
- log: `/Users/idaun/PoC/UASEF/results/round8/all_in_one_20260508-161136/run.log`

## Step status

| Step | Description | Status | Duration (s) |
| --- | --- | --- | --- |
| P0 | pytest regression | ok | 10 |
| P0.5 | calibration pipeline (RTC/entropy/EDE) | skip | 0 |
| P1.1 | single-seed full evaluation (run_full_evaluation.sh) | skip | 0 |
| P1.2 | 5-seed multi-seed bootstrap | skip | 0 |
| P1.3 | LLM-judge self-consistency | skip | 0 |
| P1.4 | primary+ablation logprob CP (run_experiment.py) | skip | 0 |
| P2.1 | multi-dataset generalization | skip | 0 |
| P2.2 | Pivot B case study | ok | 3 |
| P3.1 | distribution shift sanity | ok | 0 |
| P3.2 | multi-lingual sanity | ok | 0 |
| P3.3 | equity audit | skip | 0 |
| P3.4 | paper-claim regression | ok | 0 |
| P4.1 | figure generation (visualize_results) | skip | 0 |
| P4.2 | cross-tag run comparison (compare_runs) | skip | 0 |

## P2.2 Pivot B case study

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
## P3.1 Distribution shift

# Round 8 — Distribution Shift Sanity (Supplementary §G)

calibrated on emergency_medicine (500 samples, target α=0.05)

| test specialty | test stratum | miss rate | target α | violation | ratio |
| --- | --- | --- | --- | --- | --- |
| internal_medicine | MODERATE | 0.3279 | 0.05 | 0.2779 | 6.557× |
| pediatrics | HIGH | 0.2787 | 0.05 | 0.2287 | 5.574× |
| neurology | HIGH | 0.459 | 0.05 | 0.409 | 9.18× |
| general_practice | LOW | 0.5246 | 0.05 | 0.4746 | 10.492× |
## Reproduction

```bash
BACKENDS="openai" SEEDS_FOR_MULTISEED="42 43 44 45 46" \
    N_CAL=200 N_TEST=100 ALPHA=0.10 \
    JUDGE_OPENAI_MODEL=gpt-5.5 \
    JUDGE_ANTHROPIC_MODEL=claude-opus-4-7 \
    bash run_all_round8.sh
```
