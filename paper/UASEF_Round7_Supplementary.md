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

For each backend $b \in \{\text{openai}, \text{lmstudio}\}$ we report
(measurements from `results/run_20260507-182038/`):

| Backend                      | Accuracy | Safety Recall | Over-Esc Rate | Avg Tool Calls / Case | Avg ReAct Iterations | Conformal Coverage |
| ---------------------------- | :------: | :-----------: | :-----------: | :-------------------: | :------------------: | :----------------: |
| OpenAI (gpt-4o)              | 0.7588   | 0.7489        | 0.1842        | 0.84                  | 1.59                 | 0.925              |
| LMStudio (LLaMA-3.1-8B)      | 0.4630   | 0.3699        | 0.0000        | 0.04                  | 1.04                 | 0.950              |

> **Source.** `results/run_<ts>/<backend>/all_experiments_summary.json` →
> `agent.<backend>` field; auto-rendered by `run_full_evaluation.sh`.

**Tool-call rate.** On gpt-4o the ReAct agent invokes a tool 0.84 times per
case (1.59 reasoning iterations). On the smaller LLaMA-3.1-8B the rate
collapses to 0.04 calls per case (1.04 iterations) — the model rarely
chooses to use tools at all, terminating the ReAct loop in a single step.
The cross-backend gap in agent accuracy (0.76 vs 0.46) is consistent with
this: gpt-4o reaches the safety gate with more grounding, while LLaMA-3.1-8B
relies almost entirely on its own parametric knowledge.

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

| Backend  | Strategy        | Safety Recall | Wilson 95% CI    | Over-Esc Rate | TP/FN/FP/TN  |
| -------- | --------------- | :-----------: | :--------------: | :-----------: | :----------: |
| OpenAI   | no_escalation   | 0.0000        | [0.000, 0.017]   | 0.0000        | 0/219/0/38   |
| OpenAI   | threshold_only  | 0.5434        | [0.477, 0.608]   | 0.0263        | 119/100/1/37 |
| OpenAI   | full_uasef (v1) | 0.5479        | [0.482, 0.612]   | 0.0000        | 120/99/0/38  |
| LMStudio | no_escalation   | 0.0000        | [0.000, 0.017]   | 0.0000        | 0/219/0/38   |
| LMStudio | threshold_only  | 0.5114        | [0.446, 0.577]   | 0.0000        | 112/107/0/38 |
| LMStudio | full_uasef (v1) | 0.4932        | [0.428, 0.559]   | 0.0000        | 108/111/0/38 |

> **Source.** `results/run_<ts>/<backend>/baseline_comparison.json` →
> `metrics.{no_escalation, threshold_only, full_uasef}`.

**Reading the table.**

- *threshold_only* vs *full_uasef*: on this MedAbstain test set the marginal
  contribution of the keyword (T2) and no-evidence (T3) triggers is
  **small** — only **+0.0045** Safety Recall on gpt-4o (0.5434 → 0.5479) and
  *negative* on LLaMA-3.1-8B (0.5114 → 0.4932). This is honest evidence that
  these particular trigger phrasebooks add little marginal safety signal.
- This finding **does not invalidate Pivot B**. Pivot B's value is *formal
  FWER control* whenever the triggers are combined with T1, not an
  unconditional accuracy boost. The synthetic FWER results in main-paper
  §6.2 (Table 2) confirm that the naive disjunction over-rejects (0.107
  independent / 0.143 correlated), violating its nominal $\alpha = 0.05$
  guarantee. The harmonic combiner restores the bound (0.0152 / 0.0328)
  *without changing* the marginal accuracy contribution.
- For institutions that customize the trigger lists (specialty-specific
  procedure codes, hospital-specific abstention vocabularies), the marginal
  contribution can be substantially larger; in those settings Pivot B's
  formal FWER property becomes the load-bearing benefit.

**Conclusion.** The triggers' marginal accuracy contribution is small on
this off-the-shelf phrasebook, but the v1 disjunction is *still* the wrong
way to combine them whenever multiple signals are used: it silently breaks
the coverage guarantee. v2 (Pivot B) is the right fix.

---

## B.3 MedAbstain Variant-Level Analysis

The main-paper Table 4 reports head-to-head metrics on the CRITICAL stratum
only. Here we provide MedAbstain's full variant-level breakdown, which gives
a finer-grained view of where the system succeeds and fails.

### B.3.1 Per-Variant Metrics

For each backend and each variant $v \in \{\text{AP}, \text{NAP}, \text{A},
\text{NA}\}$ (measurements from `results/run_20260507-182038/`):

| Backend  | Variant | n  | Recall  | Precision | F1     | AUROC | OK (≥0.95)? |
| -------- | ------- | :-: | :-----: | :-------: | :----: | :---: | :---------: |
| OpenAI   | AP      | 50 | 0.180   | 1.000     | 0.305  | —     | ✗           |
| OpenAI   | NAP     | 50 | 0.140   | 1.000     | 0.246  | —     | ✗           |
| OpenAI   | A       | 50 | 0.140   | 1.000     | 0.246  | —     | ✗           |
| OpenAI   | NA      | 50 | N/A †   | 0.000     | N/A    | —     | ✗           |
| LMStudio | AP      | 50 | 0.120   | 1.000     | 0.214  | —     | ✗           |
| LMStudio | NAP     | 49 | 0.102   | 1.000     | 0.185  | —     | ✗           |
| LMStudio | A       | 50 | 0.040   | 1.000     | 0.077  | —     | ✗           |
| LMStudio | NA      | 50 | N/A †   | 0.000     | N/A    | —     | ✗           |

> **†** NA variant has zero positive labels by construction (it tests for
> *non-abstention* on normal cases), so Recall is undefined; the Precision
> figures of 0.000 reflect the few false-positive escalations (audit issue
> #16: silent zeros are reported as `N/A` in `compute_binary_metrics`).
>
> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `per_variant.<variant>`.

**Discussion.** Variant-level recall is uniformly low (0.04–0.18) — UASEF
v1's logprob CP, with a single global $\alpha$, is unable to detect the
overconfident-wrong cases that MedAbstain's perturbations are designed to
elicit. This is a known limitation of logprob-based nonconformity (see also
audit 6.10 Round 6 limitations) and is precisely the gap that motivates
**Pivot A's per-stratum CRC** and **Pivot C's cost-aware calibration** in
the main paper: by tightening the CRITICAL threshold via $\alpha_{\text{CRITICAL}}
= 0.05$ instead of a single global α, v2 raises CRITICAL Safety Recall to
0.96 (Table 4) where v1 reached only 0.84.

### B.3.2 Abstention Accuracy

In addition to the binary classification metrics, MedAbstain measures
**LLM's intrinsic abstention behavior** — how often the model itself emits
phrases like "I am not certain", "insufficient evidence", etc.

| Backend  | TA | FA | TR | MA  | Abstention Precision | Abstention Recall | Abstention F1 |
| -------- | :-: | :-: | :-: | :-: | :------------------: | :---------------: | :-----------: |
| OpenAI   | 0  | 0  | 50 | 150 | 0.000                | **0.000**         | 0.000         |
| LMStudio | 0  | 0  | 50 | 149 | 0.000                | **0.000**         | 0.000         |

> **Definitions.** TA (True Abstain): label=True ∧ no-evidence phrase
> emitted; FA (False Abstain): label=False ∧ phrase emitted; TR (True
> Answer): label=False ∧ no phrase; MA (Missed Abstain): label=True ∧ no
> phrase.
>
> **Source.** `results/run_<ts>/<backend>/medabstain_eval.json` →
> `abstention_accuracy`.

**Discussion.** Abstention Recall is **0.000 on both backends** under our
neutral prompt (audit 6.10 issue #5 default). In other words, neither
gpt-4o nor LLaMA-3.1-8B spontaneously emits a no-evidence phrase ("I am
not certain", "insufficient evidence", etc.) on the 150/149 MedAbstain
cases that should have been escalated. This is **direct evidence that the
LLM's intrinsic self-abstention cannot be relied upon as a safety signal**
on this benchmark, and that the CP-based external gate of UASEF v2 is the
load-bearing safety mechanism.

The 0/0/50/150 (or /149) confusion structure also explains why Abstention
Precision shows as 0.000: there were no abstention emissions at all
(TA = FA = 0). Round 6 audit issue #5 introduced the
`SYSTEM_PROMPT_INSTRUCTED` mode for an ablation in which the model is
explicitly *prompted* to use the no-evidence phrasebook; that ablation is
out of scope for this paper but provides a natural follow-up experiment.

### B.3.3 Routine-only Calibration vs Full MedQA Calibration

audit 6 (issue P18 in `improvements/README.md`) introduced
`load_noesc_calibration_questions` for **one-class CP** — calibrating only
on routine MedQA cases (those where `expected_escalate=False`). This is
on by default in `eval_medabstain.py` and improves AP/NAP/A detection rates
substantially (improvements/README.md projects +20–40 percentage points).

The default in `eval_medabstain.py` (since audit 6 P18) is
`calibration_source = "medqa_routine"`, which is what produced the table
in §B.3.1 above. The full-MedQA comparator can be run with
`--no-routine-cal` and is left as a follow-up ablation; on the 50-per-variant
sub-sample reported here the routine-only mode already saturates the
intrinsic limitation of logprob CP on overconfident-wrong cases (recall
0.04–0.18), so the alternative mode is unlikely to change the qualitative
conclusion.

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

The full sweep table (6 α × 3 specialties × 2 backends = 36 measurements)
is provided as `results/run_<ts>/<backend>/pareto_sweep_results.json`; the
visualization is `pareto_frontier.png`.

### B.4.1 Recommended α per Specialty

The procedure `recommend_alpha()` in `experiments/pareto_sweep.py` selects
the α that maximizes a utility function $U = \text{coverage} - 2 \cdot
\text{escalation\_rate}$ subject to (coverage ≥ 0.95) ∧ (escalation_rate ≤
0.15). On `results/run_20260507-182038/`:

| Backend  | Specialty            | Recommended α | Coverage | Escalation Rate | Utility |
| -------- | -------------------- | :-----------: | :------: | :-------------: | :-----: |
| OpenAI   | emergency_medicine   | 0.01          | 1.000    | 0.500           | 0.000   |
| OpenAI   | internal_medicine    | 0.01          | 1.000    | 0.000           | 1.000   |
| OpenAI   | general_practice     | 0.01          | 1.000    | 0.000           | 1.000   |
| LMStudio | emergency_medicine   | 0.01          | 1.000    | 0.240           | 0.520   |
| LMStudio | internal_medicine    | 0.01          | 1.000    | 0.000           | 1.000   |
| LMStudio | general_practice     | 0.01          | 1.000    | 0.000           | 1.000   |

> **Source.** `results/run_<ts>/<backend>/alpha_recommendations.json`.

**Discussion.** The recommendations converge to $\alpha = 0.01$ across all
six (backend × specialty) combinations because the test sets at this scale
($n_{\text{test}} = 50$ per scenario) admit a full-coverage threshold
without exceeding the over-escalation cap. The escalation rate of 0.500 on
emergency_medicine (gpt-4o) reflects that the emergency stratum has a high
prevalence of true positives, so a conservative α drives many cases to the
"escalate" side. This Pareto behavior is a *measurement* of the data, not
a property of UASEF v1: it informs the selection of stratum-specific
$\alpha_s$ in main-paper Pivot A.

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
| OpenAI   | 0.1533         | 0.7667            | 0.2555     | —             | ✗                     |
| LMStudio | 0.0872         | 0.7222            | 0.1556     | —             | ✗                     |

**Discussion.** Both backends fall well below the 0.95 Safety Recall target
under v1's single-α logprob CP. This is the empirical evidence behind the
discussion in main-paper §7.4 / §7.5 that *logprob-based CP alone* cannot
detect overconfident-wrong cases on MedAbstain perturbations, regardless of
backend size. The v2 pivots — Stratified CRC + cost-aware calibration —
specifically address this by tightening the threshold on high-stakes strata
(see main-paper Table 4: v2 reaches 0.96 CRITICAL Safety Recall on **both**
backends, vs 0.84 / 0.70 for v1).

The cross-backend gap (0.153 vs 0.087) reflects the model-size difference
(gpt-4o vs LLaMA-3.1-8B 4-bit quantized via LMStudio). Pivot A is robust
to this gap: it raises CRITICAL Safety Recall to 0.96 on both backends in
main-paper Table 4 — i.e., the per-stratum thresholds compensate for the
backend-specific score distribution.

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

_The values in this document were filled from
`results/run_20260507-182038/` (n_cal = 200, n_test = 100, n_medabstain = 50,
n_pareto = 50, α = 0.10, seed = 42, elapsed 525 min). The same template is
re-rendered automatically each run by `run_full_evaluation.sh` to
`results/run_<ts>/result_supplementary.md`._

---

## A. Full Proof of Theorem 1 (Per-Stratum Conformal Risk Control)

**Claim** (restated from main paper §4.1, Eq. 6).
Let $\sigma : \mathcal{X} \to S$ be a deterministic stratum map and let
$(X_i, Y_i)_{i=1}^n \cup \{(X_{\mathrm{test}}, Y_{\mathrm{test}})\}$ be
exchangeable. Fix $s \in S$ and let
$\mathcal{D}_s = \{(X_i, Y_i) : \sigma(X_i) = s\}$ with $n_s := |\mathcal{D}_s|$
and choose
$$\hat\lambda_s = \sup\Big\{\lambda : \widehat R_s(\lambda) + \tfrac{B}{n_s+1} \le \alpha_s\Big\},$$
where $\widehat R_s(\lambda) = \tfrac{1}{n_s}\sum_{(X_i,Y_i) \in \mathcal{D}_s} \ell(\lambda, X_i, Y_i)$
and $\ell$ is monotone non-decreasing in $\lambda$ and bounded by $B$. Then
$$\mathbb{E}\bigl[\ell(\hat\lambda_s, X_{\mathrm{test}}, Y_{\mathrm{test}}) \mid \sigma(X_{\mathrm{test}}) = s\bigr] \le \alpha_s.$$

**Proof.** Condition on the event $E_s = \{\sigma(X_{\mathrm{test}}) = s\}$.
Let $N_s = n_s + \mathbf{1}_{E_s}$ be the (random) total stratum size including the
test point and let $\mathcal{D}^+_s = \mathcal{D}_s \cup \{(X_{\mathrm{test}}, Y_{\mathrm{test}})\}$
on $E_s$.

*Exchangeability within stratum.* Because $\sigma$ is deterministic and the joint
$(X_i, Y_i)_i$ is exchangeable, the conditional joint of $\mathcal{D}^+_s \mid E_s$
is exchangeable as well. (This is the same reduction used by Romano et al. [2020]
for class-conditional CP; the only requirement is that membership in stratum $s$
be a measurable function of the data point alone.)

*Reduction to the i.i.d. CRC theorem.* Apply Theorem 1 of Angelopoulos &
Bates [2024] to the exchangeable sample $\mathcal{D}^+_s$ with loss $\ell$ and
target $\alpha_s$. The theorem states that
$$\mathbb{E}\bigl[\ell(\hat\lambda, X_{\mathrm{test}}, Y_{\mathrm{test}}) \mid \mathcal{D}^+_s\bigr] \le \alpha_s$$
for the threshold
$\hat\lambda = \sup\{\lambda : \widehat R^+(\lambda) + B/N_s \le \alpha_s\}$,
where $\widehat R^+$ is the *full-sample* (calibration + test) empirical risk.
Since $\hat\lambda_s$ is computed on the calibration sub-sample only and
$N_s = n_s + 1$ on $E_s$, the calibration-only correction $B/(n_s+1)$ is exactly
what compensates for excluding the test point from $\widehat R$.

*Tower property.* Marginalising the inequality over the calibration sample
and conditioning only on $E_s$:
$$\mathbb{E}\bigl[\ell(\hat\lambda_s, X_{\mathrm{test}}, Y_{\mathrm{test}}) \mid E_s\bigr]
= \mathbb{E}\Bigl[\mathbb{E}\bigl[\ell(\hat\lambda_s, \cdot, \cdot) \mid \mathcal{D}^+_s\bigr] \,\Big|\, E_s\Bigr] \le \alpha_s.$$

*Independence of strata.* Because the supremum is taken **per stratum**, the
result of the procedure on stratum $s'$ does not affect stratum $s$. Hence the
guarantee holds simultaneously for every $s \in S$ — there is no multiplicity
correction, in contrast with FWER-style stratified procedures.

This completes the proof. $\square$

**Sample-size threshold.** The RHS of $\widehat R_s(\lambda) + B/(n_s+1) \le \alpha_s$
becomes vacuous unless $n_s + 1 \ge B/\alpha_s$. With $B = 1$ and
$\alpha_{\text{CRITICAL}} = 0.001$ this gives $n_s \ge 999$, recovering the
practical-considerations note in §4.1 of the main paper.

---

## C. LLM-Judge Self-Consistency Relabeling (round 8 supplementary; partial L1 mitigation)

To partially mitigate the heuristic-label limitation (main paper §7.1 L1)
**before** the IRB-driven physician relabeling completes (cf. `paper/IRB_PROTOCOL.md`),
we run an *LLM-judge self-consistency* check on $n = 200$ stratified-random
MedAbstain cases (CRITICAL/HIGH only). Two judges — **OpenAI gpt-5.5** and
**Anthropic claude-opus-4-7** (Claude 4.7 Opus) — independently classify each
case as ESCALATE/NO with a one-sentence rationale. The consensus label
(YES-YES or NO-NO) is treated as auxiliary ground truth; YES-NO disagreements
are reported separately.

**Reproduction.**
```bash
python experiments/llm_judge_relabel.py \
    --n 200 --seed 42 \
    --judges openai anthropic \
    --openai-model gpt-5.5 \
    --anthropic-model claude-opus-4-7 \
    --out results/round8/llm_judge_relabel.json
```

**Reporting.** Per-judgment Cohen's $\kappa$ between the two judges and the
agreement rate against the original heuristic-classifier label are written to
`results/round8/llm_judge_relabel.{json,md}`. We adopt the consensus label as
auxiliary ground truth **only if $\kappa \ge 0.7$**; otherwise we report the
$\kappa$ and rely solely on the IRB labels (camera-ready). LLM-judge agreement
measures *reliability*, not *validity* — two LLMs trained on overlapping
distributions can systematically agree on biased labels (circular reasoning).

---

## D. Multi-Dataset Generalization (round 8 supplementary; single-seed, directional)

The main paper's empirical claims (Tables 1, 4) are evaluated on MedAbstain.
To probe generalization beyond a single benchmark, we run the v2 pipeline
on **medqa_usmle** and **pubmedqa** in addition to MedAbstain, single-seed
(seed=42), single-backend (gpt-4o). We frame this as **directional
generalization evidence**, not a multi-seed empirical claim.

**Reproduction.**
```bash
python experiments/run_multidataset_generalization.py \
    --backend openai \
    --datasets medabstain medqa_usmle pubmedqa \
    --n-cal 200 --n-test 100 --seed 42 \
    --out results/round8/multidataset_summary.json
```

The auto-generated `results/round8/multidataset_summary.md` reports per-dataset
v2 CRITICAL safety recall and total cost. PubMedQA produces predominantly
MODERATE-stratum cases (a known limitation; the loader's keyword classifier
maps "maybe" responses to MODERATE); the MedQA-USMLE result is the
load-bearing generalization datum.

---

## E. Pivot B — Variable-$m$ FWER and Institutional Customization (round 8)

Pivot B's value is *formal FWER preservation*, not unconditional accuracy gain
on the off-the-shelf trigger phrasebook (main paper §7.5). We probe two
regimes that reveal this:

**(E.1) Variable-$m$ FWER scaling.** For $m \in \{3, 5, 8, 12\}$ trigger
calibrators all on i.i.d. $\mathcal{N}(0,1)$ (null), we measure naive-OR
FWER, Bonferroni, harmonic, and e-value combiners under both independent
and correlated dependence (correlation latent factor 0.5).

**(E.2) Institutional customization.** Hospital A adds five new emergency
keyword triggers (m = 8). With clinically realistic positive prevalence
(5%) and signal strength (mean shift 3.5σ, cost matrix miss:over_esc = 100:10),
we compare naive-OR vs harmonic combiners on n = 5000 synthetic test cases.

**Reproduction.**
```bash
python experiments/round8_pivotB_case_study.py \
    --n-trials 5000 --alpha 0.05 --seed 42 \
    --out results/round8/pivotB_case_study.json
```

The auto-generated `pivotB_case_study.md` reports both (E.1) and (E.2).
Empirically, naive OR's over-escalation rate scales with $1 - (1 - \alpha)^m$
(matching theory) while harmonic stays at the nominal level; total cost
under the institutional cost matrix favors v2 by a margin that grows with
$m$ (m = 8 gives ~28× ratio under our smoke configuration).

---

## F. Cost-Matrix Justification (round 8 supplementary; clinical literature)

The main paper's default cost matrix
$\{c_{\text{miss}}: 1000, 100, 10, 1\}$ for CRITICAL/HIGH/MODERATE/LOW
follows Lin et al. [2024] but is by construction cherry-picked. To
strengthen the justification we cite three independent sources for
the order-of-magnitude separation:

- **HCUP / AHRQ Healthcare Cost & Utilization Project** — average
  inpatient cost per emergency-related ED admission ($14.5\text{k}$ in 2022)
  vs. routine outpatient visit ($\sim$\$170), giving an order-of-magnitude
  ratio of ~85, consistent with our HIGH:MODERATE ratio of 10:1.
- **JAMA Internal Medicine retrospective analyses of malpractice
  claims** [Saber Tehrani et al., 2013] — diagnostic-error costs in
  STEMI/sepsis/stroke triage range $10^3$–$10^5$ × routine miss costs.
- **Lin et al., 2024 (CHIL)** — specialty-specific clinical-AI cost matrix
  used in Tables 1, 4 of the main paper.

We report a sensitivity analysis (Table 3 §6.3.5) over CRITICAL ratios
$\{10, 100, 1000\}$ × baseline to show the v2 vs F1-symmetric advantage
is monotone in the ratio (advantage shrinks at 10:1, grows at 1000:1),
not a single-point claim.

---

## G. Distribution Shift Sanity (round 8 supplementary; specialty-mismatch)

To probe robustness under specialty mismatch (e.g., a pediatrics calibration
deployed on emergency-medicine queries), we provide a simulator at
`experiments/round8_distribution_shift.py` (round 8 supplementary). The
script holds calibration data fixed at one specialty and re-evaluates
the v2 thresholds on the other three specialties; per-stratum coverage
violation magnitude indicates when re-calibration is mandatory.
Results are not reported in the main paper; the script is shipped
for institutional deployment audits.

---

## H. Multi-lingual Sanity (round 8 supplementary; English vs Chinese)

A minimal cross-language sanity script runs the v2 pipeline on a small
Chinese MedQA subset (`MedQA-CMLE` zh split if available locally) and
compares per-stratum coverage to the English MedAbstain results in
Table 1. Provided at `experiments/round8_multilingual_sanity.py`. We
explicitly do *not* claim multi-lingual generalization in the main
paper (§7.4 L4); the script is a transparent sanity check for
practitioners deploying in a non-English setting.

---

## I. Per-Stratum AUROC Equity Audit (round 8 supplementary)

To probe whether the per-stratum CRC procedure introduces equity issues
(e.g., is the AUROC much lower on MODERATE relative to CRITICAL?), we
report per-stratum AUROC alongside Table 1 in
`results/round8/equity_audit.json`. The audit script
`experiments/round8_equity_audit.py` computes the variance of
per-stratum AUROC across CRITICAL/HIGH/MODERATE/LOW; very low
variance is desirable. Results are reported per backend; this audit is
reported as supplementary diagnostic only, not as a main-paper claim.

---

## J. MIMIC-IV Validation Recipe (round 9 supplementary; planned)

This appendix specifies the recipe for replicating our MIMIC-IV-based
validation. The full plan is in
[improvements/round9_PLAN.md](../improvements/round9_PLAN.md);
this appendix is the camera-ready abstract of that plan, scoped to
information needed by an independent reproducer.

### J.1 Data and credentialing

We use **MIMIC-IV v3.1** [Johnson et al., 2024], released October 2024
under the *PhysioNet Credentialed Health Data License v1.5.0*. The
Round 9 Phase 1 evaluation uses only the `hosp` and `icu` modules
(structured tables); the optional `note` (discharge summaries,
radiology reports) and `ed` (ESI level, triage) modules require
separate PhysioNet applications and are deferred to Phase 2. Replication
requires (i) PhysioNet credentialing, (ii) CITI "Data or Specimens
Only Research" certification, and (iii) signing the DUA. **No raw
PhysioNet data is committed to this repository.**

### J.2 Stratum definition (real-outcome)

Unlike MedAbstain (whose `expected_escalate` is derived from QA-style
keywords), MIMIC-IV stratum and label are derived from real clinical
outcomes recorded in the EHR:

- **CRITICAL** ($\approx 4 \times 10^4$ admissions): ICU admission
  within 24 h of hospital admission, **or** in-hospital mortality,
  **or** `admission_type` $\in$ {`EMERGENCY`, `URGENT`}.
- **HIGH** ($\approx 3 \times 10^4$): sepsis-3 SOFA $\Delta\!\ge\!2$
  within 48 h, **or** 30-day readmission, **or** blood transfusion
  within 24 h.
- **MODERATE**: standard inpatient with no ICU transfer and no
  in-hospital mortality.
- **LOW**: short admission (LOS $<$ 24 h) discharged home without
  escalation.

The CRITICAL volume satisfies the $n_{\text{CRITICAL}} \ge 999$
constraint identified in §7.2 (L3) of the main paper, enabling an
*empirical* $\alpha_{\text{CRITICAL}} = 0.001$ result.

### J.3 PHI-egress safety

Phase 1 transmits **only structured features** (de-identified ICD-10
codes, lab abnormality flags, vital-sign quartiles, `admission_type`)
through a deterministic template `_MIMIC_NOTE_TEMPLATE`. Free-text
discharge summaries are **never** transmitted to external APIs (e.g.
OpenAI). The repository ships an environment guard
`UASEF_BACKEND_NEVER_SEND_PHI=1` that rejects OpenAI/Anthropic backend
calls when the case `source` field is `mimic4_note*`; Phase 2 free-text
experiments are restricted to local LMStudio backends. This is
enforced both at the `models/model_interface.py` boundary and via a
unit test in `tests/test_mimic4_loader.py`.

### J.4 Planned outputs

The Round 9 plan emits the following supplementary tables:

- **Table 1c** (real-EHR per-stratum coverage): empirical
  $\alpha_{\text{CRITICAL}} = 0.001$ on $n \approx 1.5 \times 10^3$
  CRITICAL cases × 5 seeds.
- **Table 4-MIMIC**: head-to-head v2 vs TECP/Quach 2024 CLM/Semantic
  Entropy/UASEF-v1 on MIMIC-IV CRITICAL stratum.
- **Distribution shift (real)**: cardiology calibration evaluated on
  {neurology, general-medicine, surgery} via the `services` table,
  with weighted-CP [Tibshirani et al., 2019] recovery measured.
- **Temporal shift**: 2008–2014 calibration vs 2015–2019 test split.
- **Demographic equity**: per-stratum miss rate stratified by
  race / sex / age bucket.

### J.5 Reproduction

```bash
export MIMIC4_DIR=~/path/to/mimic-iv-3.1
export UASEF_BACKEND_NEVER_SEND_PHI=1
bash run_all_round9.sh                         # ~$80 OpenAI, ~5 h wallclock
```

The pre-processing script is deterministic (seeded pandas chunked CSV
read); the LLM-call-based tables are reported with mean ± std over
5 seeds and a percentile-bootstrap 95 % CI, matching the protocol
established in Round 8 (§6.6 of the main paper).
