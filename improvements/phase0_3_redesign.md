# Redesign — Audited Conformal Escalation for Medical LLM QA (Phase 0–3)

> Implementation blueprint for the phase plan (`phase0_3_detailed_plan.md`). It
> maps every phase to concrete modules, records the go/no-go gates, the leakage
> guards, what is **reused** from the existing diagnostic framework, what is
> **built and verified now**, and the one execution dependency (LLM backend).

## 0. Positioning (locked)

A **pre-release** conformal escalation layer: decide whether to RELEASE an LLM's
medical-QA answer or ESCALATE it to a human, *before* the answer is shown, with a
finite-sample guarantee on **benchmark-defined / objective-label** error — **not**
a deployment-ready clinical-safety claim. The LLM verifier is a *risk feature*,
never ground truth. Every phase has a pre-committed go/no-go gate so a fake
success is caught early (the discipline of the Patterns diagnostic paper, applied
forward instead of retrospectively).

**Cross-phase invariants (enforced in code):**
- *No definitional leakage* — the gold answer computes the error **label** only;
  `qa_risk_features.extract_features` never sees it.
- *Verifier-as-feature* — verifier outputs enter as risk features, audited for
  answer-key contamination in Phase 2.
- *Orientation* — every risk feature and the risk score are oriented **higher =
  riskier**; `label_conditional_conformal.check_orientation` guards it.
- *Locked test* — all final claims on a held-out test split; calibration/test
  never peeked.

## 1. Module map

| Plan piece | Module | Status |
|---|---|---|
| **Phase 1 conformal core** — `P(release\|incorrect) ≤ α` | [models/label_conditional_conformal.py](../models/label_conditional_conformal.py) | **built + verified** (11/11 Monte-Carlo tests) |
| Risk features (leakage-safe, oriented) | [models/qa_risk_features.py](../models/qa_risk_features.py) | **built** (5 minimal features) |
| MedMCQA / PubMedQA loaders → `QAItem` | [data/qa_datasets.py](../data/qa_datasets.py) | **built** (HF; subject-stratified) |
| Phase 0 gatekeeper (CV AUROC, go/no-go) | [experiments/phase0_gatekeeper.py](../experiments/phase0_gatekeeper.py) | **built + smoke** (synthetic end-to-end) |
| Draft generation (decision + k samples + logprobs) | `models/qa_drafts.py` | **pending backend decision** (§5) |
| Phase 1 Stage-A runner (split, scorer, gate, baselines) | `experiments/phase1_stage_a.py` | pending (core ready) |
| Phase 2 audit (5 detectors + verifier-contam + retrieval-ablation) | reuse [models/audit_detectors.py](../models/audit_detectors.py) + `experiments/phase2_qa_audit.py` | detectors reusable; QA adapters pending |
| Phase 3 Stage-B sanity (MIMIC/eICU objective-label contrast) | reuse `round18/19/22` + `experiments/phase3_stage_b.py` | pending |

## 2. Phase 1 conformal core (the mathematical heart — done)

Convention: risk `r(x,a)`, higher = more likely error; `release ⟺ r < τ`. τ is
calibrated on the **error cases only** (label-conditional / Mondrian; Vovk 2003):

```
k   = floor(α · (n_err + 1))
τ   = k-th smallest error-case risk        (τ = −∞, escalate-all, if k < 1)
```

giving, for a fresh exchangeable error case, `P(r < τ | incorrect) = k/(n_err+1)
≤ α`. **Verified** (`tests/test_label_conditional_conformal.py`, 200-seed
Monte-Carlo): empirical `P(release|incorrect)` = 0.0495 / 0.0985 / 0.1994 at α =
0.05 / 0.10 / 0.20 (all ≤ α, tight at the bound); holds even for a pure-noise
risk (validity ≠ informativeness); an informative risk is non-vacuous
(release-rate 0.22–0.48). **Feasibility:** `n_err ≥ ⌈(1−α)/α⌉` (α=0.10→9,
0.05→19; ~30/60 recommended). Below it → explicit `feasible=False`,
`τ=−∞` (escalate-all), never a silent collapse — the same INFEASIBLE discipline
as the endpoint core.

This is **distinct** from `models/conformal_escalation.py` (endpoint-CRC over a
nested threshold controlling the *miss* rate): here the guarantee is conditioned
on the error label, which is what "don't release wrong answers" requires.

## 3. Go/No-Go gates

| Phase | Gate | Fallback |
|---|---|---|
| **0** | pooled AUROC(risk,error) ≥ 0.70 | 0.60–0.70 → add evidence/verifier features; <0.60 → pivot to information-boundary negative |
| **1** | locked-test α met (0.10/0.05) **and** release-rate non-vacuous (≥0.40) **and** ≥ baselines at equal α | α met but escalate-all-near → "over-escalation required" audit result |
| **2** | all 5 detectors + verifier-contam + retrieval-ablation pass | any flag → remove leak/collapse, re-fit, re-audit |
| **3** | agreement + information-tension give a clear picture | gate "works" on objective labels → suspect leakage, re-audit |

## 4. Leakage guards (Phase 2, ported detectors)

- **orientation** / **escalate-all** / **definitional** / **temporal** /
  **informative-missingness**: reuse `models/audit_detectors.py` unchanged; the
  definitional detector screens each risk feature's univariate AUROC for
  answer-key contamination.
- **NEW verifier-contamination audit**: option-shuffle, paraphrase, and
  held-out-cutoff tests — if verifier-feature AUROC survives content changes but
  collapses under answer-position changes, the verifier is memorising the key →
  down-weight/remove.
- **NEW retrieval-only ablation**: full vs verifier-only vs
  retrieval-metadata-only vs coverage-indicator-only — if metadata/indicators
  alone recover ≥85% of full, the gate is an "answer-retrieval-failure" gate
  (informative-missingness recurring in the QA domain).

## 5. Execution dependency — LLM backend for drafts (decision needed)

Phase 0 needs hidden drafts for MedMCQA (~3000) + PubMedQA (~800), each = 1
decision draft + k=5 temperature samples (+ logprobs + a verbalized-confidence
prompt) ≈ **~23k generations**. The code (`qa_drafts.py`, via
`models/model_interface.py`) is backend-agnostic; only the choice differs:

| Option | Cost | Notes |
|---|---|---|
| Local **LMStudio** (`gpt-oss-120b`, configured) | **$0** | server on :1234; slow (~hours) but honours the project's $0-external / all-local stance; supports logprobs |
| **OpenAI** `gpt-4o-mini` | ~$ (small) | external API; fast; supports logprobs; breaks the 0-byte-egress stance |
| **Synthetic smoke** | $0 | pipeline verified end-to-end today (AUROC 0.86 on injected signal) — proves the code, not the science |

Until a backend is chosen, the smoke path (`--synthetic`) demonstrates the full
Phase-0 pipeline; the real go/no-go number requires real drafts.

## 6. Status (2026-07)

- **Verified now:** label-conditional conformal core (11/11), risk-feature
  extractor, MedMCQA/PubMedQA loaders, Phase-0 gatekeeper (synthetic end-to-end
  → GO on injected signal).
- **Next:** choose a backend → generate real drafts → real Phase-0 AUROC →
  Phase-1 Stage-A gate + baselines → Phase-2 QA audit → Phase-3 Stage-B contrast.

**Standing Discussion caveat (verbatim for the paper):** *This study controls
benchmark-defined error and objective-label high-risk release, not real-world
clinical harm. The LLM verifier is used only as a risk feature, not ground truth.
Before clinical deployment, the framework must be validated on open-ended
clinical QA with specialist adjudication.*
