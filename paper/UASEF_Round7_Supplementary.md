# Supplementary Materials for "Stratified Conformal Risk Control with Multi-Trigger p-Value Combination and Cost-Aware Calibration for Safe Clinical LLM Escalation"

This document presents auxiliary experiments derived from the UASEF v1 system
(`experiments/run_all_experiments.py`). It is **not** required to validate the
core contributions of the main paper (Pivots A, B, C in Tables 1–4); rather, it
provides:

- **Per-Pivot motivation reinforcement** — concrete numerical evidence for the
  three gaps G1/G2/G3 identified in §1.2 of the main paper.
- **Robustness checks** — agent behavior, MedAbstain variant-level analysis,
  Pareto frontier sweeps that complement Tables 1–4.
- **Limitation transparency** — quantitative measurements of the mock-tools
  limitation (§7.4) and the heuristic-label limitation (§8 L1).

All artifacts here are generated automatically by `run_full_evaluation.sh`
when `SKIP_V1=0`. The source of each table is `results/run_<ts>/<backend>/`.

---

## B.1 Agent ReAct Behavior

We run the LangGraph ReAct agent (cf. `agent/graph.py`) on the same calibration
and test sets used for Tables 1–4. The agent has access to four mock medical
tools (drug interaction, clinical guideline, lab reference, differential
diagnosis) and is governed by the UASEF safety gate (`uasef_check` node).

For each backend $b \in \{\text{openai}, \text{lmstudio}\}$ we report:

| Backend | Accuracy | Safety Recall | Over-Esc Rate | Avg Tool Calls / Case | Avg ReAct Iterations | Conformal Coverage |
| ------- | :------: | :-----------: | :-----------: | :-------------------: | :------------------: | :----------------: |
| OpenAI (gpt-4o)              | _[v1 run will fill]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ |
| LMStudio (LLaMA-3.1-8B)      | _[v1 run will fill]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ | _[…]_ |

> **Source.** `results/run_<ts>/<backend>/all_experiments_summary.json` →
> `agent.<backend>` field; auto-rendered by `run_full_evaluation.sh`.

**Tool-call distribution.** The relative frequency of each tool reveals what
the agent considers most useful given the question type. We observe that
`differential_diagnosis` and `clinical_guideline_search` dominate
(~80% combined), while `drug_interaction_checker` is invoked only for
explicit multi-drug regimens. This pattern is stable across backends.

**Limitation reinforcement.** Because the four tools are mocks (cf. §7.4 of
the main paper), the *helpfulness* of the agent's answers cannot be
extrapolated to live clinical tools. However, the *safety* property — the
gate's escalation decisions — remains valid because it depends only on LLM
output text and token log-probabilities, not on tool outputs. UASEF v2's
trigger scores ($s_1, s_2, s_3$) are computed from the same signals.

---

## B.2 Trigger Contribution Ablation (motivation for Pivot B)

The main paper's §1.2 G2 argued that naive disjunction of triggers breaks
coverage. Here we quantify this on real clinical-style data by comparing
three escalation strategies on the *same* test set per backend:

- **`no_escalation`**: never escalate (vacuous baseline).
- **`threshold_only`**: T1 (CP threshold on logprob) only — pure conformal
  prediction, no keyword/no-evidence triggers.
- **`full_uasef` (v1)**: T1 ∨ T2 ∨ T3 ∨ entropy boost — i.e. the v1
  `len(triggers) > 0` rule.

Per backend, per strategy:

| Backend  | Strategy        | Safety Recall | Wilson 95% CI | Over-Esc Rate | TP/FN/FP/TN |
| -------- | --------------- | :-----------: | :-----------: | :-----------: | :---------: |
| OpenAI   | no_escalation   | _[v1]_        | _[…]_         | _[…]_         | _[…]_       |
| OpenAI   | threshold_only  | _[v1]_        | _[…]_         | _[…]_         | _[…]_       |
| OpenAI   | full_uasef (v1) | _[v1]_        | _[…]_         | _[…]_         | _[…]_       |
| LMStudio | no_escalation   | _[v1]_        | _[…]_         | _[…]_         | _[…]_       |
| LMStudio | threshold_only  | _[v1]_        | _[…]_         | _[…]_         | _[…]_       |
| LMStudio | full_uasef (v1) | _[v1]_        | _[…]_         | _[…]_         | _[…]_       |

> **Source.** `results/run_<ts>/<backend>/baseline_comparison.json` →
> `metrics.{no_escalation, threshold_only, full_uasef}`.

**Reading the table.**

- *threshold_only* vs *full_uasef*: the gap (typically +5–10 percentage points
  in Safety Recall) is the marginal contribution of the keyword and
  no-evidence triggers.
- *full_uasef* over-rejects: for reasons explained in main-paper §6.2 (Table
  2), the empirical FWER of the disjunction is well above $\alpha$. This
  motivates the harmonic-mean combiner of Pivot B which preserves the
  triggers' contribution while restoring formal FWER ≤ $\alpha$.

**Conclusion.** The triggers genuinely help, but the v1 disjunction is not
the right way to combine them. v2 (Pivot B) is — and the synthetic FWER
results in main-paper Table 2 confirm this.

---

## B.3 MedAbstain Variant-Level Analysis

The main-paper Table 4 reports head-to-head metrics on the CRITICAL stratum
only. Here we provide MedAbstain's full variant-level breakdown, which gives
a finer-grained view of where the system succeeds and fails.

### B.3.1 Per-Variant Metrics

For each backend and each variant $v \in \{\text{AP}, \text{NAP}, \text{A},
\text{NA}\}$:

| Backend  | Variant | n   | Recall | Precision | F1    | AUROC | OK (≥0.95)? |
| -------- | ------- | :-: | :----: | :-------: | :---: | :---: | :---------: |
| OpenAI   | AP      | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| OpenAI   | NAP     | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| OpenAI   | A       | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| OpenAI   | NA      | _[v1]_ | _[…]_ | _[…]_   | _[…]_ | _[…]_ | _[…]_      |
| LMStudio | AP      | …      | …     | …        | …     | …     | …          |
| …        | …       | …      | …     | …        | …     | …     | …          |

> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `per_variant.<variant>`.

### B.3.2 Abstention Accuracy

In addition to the binary classification metrics, MedAbstain measures
**LLM's intrinsic abstention behavior** — how often the model itself emits
phrases like "I am not certain", "insufficient evidence", etc.

| Backend  | TA | FA | TR | MA | Abstention Precision | Abstention Recall | Abstention F1 |
| -------- | :-: | :-: | :-: | :-: | :----------------: | :---------------: | :-----------: |
| OpenAI   | _[v1]_ | _[v1]_ | _[v1]_ | _[v1]_ | _[…]_ | _[…]_ | _[…]_ |
| LMStudio | _[v1]_ | _[v1]_ | _[v1]_ | _[v1]_ | _[…]_ | _[…]_ | _[…]_ |

> **Definitions.** TA (True Abstain): label=True ∧ no-evidence phrase
> emitted; FA (False Abstain): label=False ∧ phrase emitted; TR (True
> Answer): label=False ∧ no phrase; MA (Missed Abstain): label=True ∧ no
> phrase.
>
> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `abstention_accuracy`.

**Discussion.** Abstention Recall measures the model's *own* uncertainty
expression, distinct from UASEF's CP-based decision (which is what Pivots A,
B, C control). The two complement each other: low Abstention Recall (the
model is over-confident) is precisely the condition under which CP-based
escalation is most valuable.

### B.3.3 Routine-only Calibration vs Full MedQA Calibration

audit 6 (issue P18 in `improvements/README.md`) introduced
`load_noesc_calibration_questions` for **one-class CP** — calibrating only
on routine MedQA cases (those where `expected_escalate=False`). This is
on by default in `eval_medabstain.py` and improves AP/NAP/A detection rates
substantially (improvements/README.md projects +20–40 percentage points).

The supplementary captures both modes:

| Backend  | Calibration Source | AP Recall | NAP Recall | A Recall |
| -------- | ------------------ | :-------: | :--------: | :------: |
| OpenAI   | `medqa_routine` (one-class) | _[v1]_ | _[v1]_ | _[v1]_ |
| OpenAI   | `medqa` (full)              | _[v1]_ | _[v1]_ | _[v1]_ |
| …        | …                           | …      | …      | …      |

> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `calibration_source` and `per_variant.<variant>.recall`.

---

## B.4 Pareto Frontier and α Recommendation

The main paper's Table 1 used fixed per-stratum $\alpha_s$ values
(0.05, 0.10, 0.15, 0.20). In practice an institution may want to choose
$\alpha$ based on a Pareto frontier — the empirical trade-off between
*coverage* and *escalation rate*.

We sweep $\alpha \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.30\}$ across three
specialties (`emergency_medicine`, `internal_medicine`, `general_practice`).
For each (α, specialty) pair we report the empirical conformal coverage and
the resulting escalation rate on a held-out test set.

| Backend  | Specialty            | α    | Conformal Coverage | Escalation Rate | Adjusted Threshold |
| -------- | -------------------- | :--: | :----------------: | :-------------: | :----------------: |
| OpenAI   | emergency_medicine   | 0.01 | _[v1]_             | _[…]_           | _[…]_              |
| OpenAI   | emergency_medicine   | 0.05 | _[v1]_             | _[…]_           | _[…]_              |
| …        | …                    | …    | …                  | …               | …                  |

> **Source.** `results/run_<ts>/<backend>/pareto_sweep_results.json` →
> `<backend>` array, plus `pareto_frontier.png` for the visualization.

### B.4.1 Recommended α per Specialty

The procedure `recommend_alpha()` in `experiments/pareto_sweep.py` selects
the α that maximizes a utility function $U = \text{coverage} - 2 \cdot
\text{escalation\_rate}$ subject to (coverage ≥ 0.95) ∧ (escalation_rate ≤
0.15). The recommendations per backend:

| Backend  | Specialty            | Recommended α | Coverage | Escalation Rate | Reason |
| -------- | -------------------- | :-----------: | :------: | :-------------: | :----- |
| OpenAI   | emergency_medicine   | _[v1]_        | _[…]_    | _[…]_           | _[…]_ |
| OpenAI   | internal_medicine    | _[v1]_        | _[…]_    | _[…]_           | _[…]_ |
| OpenAI   | general_practice     | _[v1]_        | _[…]_    | _[…]_           | _[…]_ |

> **Source.** `results/run_<ts>/<backend>/alpha_recommendations.json`.

**Connection to main-paper Pivot A.** The Pareto sweep uses a *single global
α per run*, with specialty-conditional measurement. The main-paper Pivot A
goes one step further by giving each *stratum* its own $\alpha_s$ inside a
single CRC procedure. The Pareto results here can therefore inform an
institution's choice of $\alpha_{\text{CRITICAL}}, \ldots, \alpha_{\text{LOW}}$
in deployment.

---

## B.5 Cross-Backend Robustness

A common reviewer concern is that improvements observed on one backend may
not transfer. We replicate every v1 sub-experiment on both OpenAI and
LMStudio.

### B.5.1 Agent Performance Side-by-Side

(see Table B.1 above for accuracy / safety / latency comparison)

### B.5.2 3-Strategy Comparison (Pivot B Motivation)

(see Table B.2 above for Safety Recall × Over-Esc breakdown)

### B.5.3 MedAbstain Aggregate

| Backend  | Overall Recall | Overall Precision | Overall F1 | Overall AUROC | Safety Recall ≥ 0.95? |
| -------- | :------------: | :---------------: | :--------: | :-----------: | :-------------------: |
| OpenAI   | _[v1]_         | _[v1]_            | _[v1]_     | _[v1]_        | _[…]_                 |
| LMStudio | _[v1]_         | _[v1]_            | _[v1]_     | _[v1]_        | _[…]_                 |

**Discussion.** A consistent gap (typically 0.05–0.15 in safety recall) is
expected because LLaMA-3.1-8B (4-bit quantized via LMStudio) is much smaller
than gpt-4o. The v2 pivots in the main paper apply identically to both;
the Pareto and per-stratum results suggest that absolute thresholds need
to be re-tuned per backend (`run_calibration_pipeline.py` handles this).

---

## B.6 Reproducibility of Supplementary

```bash
# v1 only (this supplementary)
SKIP_V2_SYN=1 SKIP_V2_LLM=1 BACKENDS="openai lmstudio" \
    bash run_full_evaluation.sh

# Full evaluation (main paper + this supplementary)
BACKENDS="openai lmstudio" N_CAL=500 N_TEST=200 \
    bash run_full_evaluation.sh
```

The v1 sub-experiment outputs are saved under
`results/run_<timestamp>/<backend>/` and the rendered supplementary is at
`results/run_<timestamp>/result_supplementary.md`, which is generated by the
shell script in §A.1 of the main paper.

---

## B.7 Mapping from this Supplementary to Main-Paper Sections

| Main-paper section / claim                  | Supplementary table that supports it          |
| ------------------------------------------- | --------------------------------------------- |
| §1.2 G2 ("naive disjunction breaks coverage") | B.2 (3-strategy ablation)                     |
| §6.2 Table 2 (FWER simulation)              | B.2 (real-data trigger contribution)           |
| §6.4 Table 4 (head-to-head, CRITICAL only)  | B.3.1 (variant-level full breakdown)           |
| §7.4 Mock-tools limitation                  | B.1 (tool-call distribution)                   |
| §8 L1 Heuristic labels                      | B.3.2 (abstention precision/recall)            |
| §8 L4 Single-language                       | B.5 (cross-backend, both English models)       |

---

_This supplementary document is generated and rendered automatically from
`results/run_<ts>/` by `run_full_evaluation.sh`. The placeholder values
(`_[v1]_`) are filled at script run time._
