# UASEF Round 8 — All-in-One Report

- timestamp: 20260508-161129
- run_dir: `/Users/idaun/PoC/UASEF/results/round8/all_in_one_20260508-161129`
- log: `/Users/idaun/PoC/UASEF/results/round8/all_in_one_20260508-161129/run.log`

## Step status

| Step | Description | Status | Duration (s) |
| --- | --- | --- | --- |
| P0 | pytest regression | dry | 0 |
| P0.5 | calibration pipeline (RTC/entropy/EDE) | dry | 0 |
| P1.1 | single-seed full evaluation (run_full_evaluation.sh) | dry | 0 |
| P1.2 | 5-seed multi-seed bootstrap | dry | 0 |
| P1.3 | LLM-judge self-consistency | dry | 0 |
| P1.4 | primary+ablation logprob CP (run_experiment.py) | dry | 0 |
| P2.1 | multi-dataset generalization | dry | 0 |
| P2.2 | Pivot B case study | dry | 0 |
| P3.1 | distribution shift sanity | dry | 0 |
| P3.2 | multi-lingual sanity | dry | 0 |
| P3.3 | equity audit | skip | 0 |
| P3.4 | paper-claim regression | dry | 0 |
| P4.1 | figure generation (visualize_results) | skip | 0 |
| P4.2 | cross-tag run comparison (compare_runs) | skip | 0 |

## Reproduction

```bash
BACKENDS="openai" SEEDS_FOR_MULTISEED="42 43" \
    N_CAL=200 N_TEST=100 ALPHA=0.10 \
    JUDGE_OPENAI_MODEL=gpt-5.5 \
    JUDGE_ANTHROPIC_MODEL=claude-opus-4-7 \
    bash run_all_round8.sh
```
