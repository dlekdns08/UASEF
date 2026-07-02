# Round 23+ Plan — Hardening the diagnostic-framework paper (Patterns / JBI)

> Goal: move each contribution from "plausible" to "reviewer-proof." Five
> prioritized experiments (R23–R27), with feasibility, deliverable, risk, and
> fallback for each. Sequenced so compute-only high-value work runs first while
> the network-dependent, highest-leverage item (R23) is scouted in parallel.

---

## R23 (P1) — Catch a real hard-flag defect in *published original code*

**Why:** §5.3 reconstructs CPMORS and no detector hard-flags (62% < 85%);
a reviewer asks "is the reconstruction faithful? isn't it just clean?" The
paper's whole thesis ("diagnostic that catches hidden defects in the wild")
rests on at least one hard-flag on someone else's actual artifact.

**What:**
1. Scout 3–5 clinical-CP papers with **public repos** (arXiv 2024–2026 skew
   toward code release). Prefer repos that (a) use MIMIC/eICU (data we hold) and
   (b) expose scores or a runnable feature/label pipeline.
2. For each: either (i) run the repo end-to-end and apply the detector suite to
   its own scores/features, or (ii) if not runnable, do a **code-level
   reconstruction** — reimplement *their exact windowing/feature/label code*
   (following the repo, not the prose) and run detectors. Code-level is strictly
   stronger than §5.3's prose-level reconstruction.
3. Target at least one **hard-flag**: an unguarded temporal window producing
   real post-outcome features (temporal detector > 0.05) or a near-escalate-all
   over-escalation (> 0.95), or informative-missingness recover ≥ 0.85.

**Deliverable:** `experiments/round23_repo_defect_hunt.py` +
`results/round23/` per-repo detector report; a new §5.4 "Auditing original
released code."

**Risk / fallback:** repos may lack runnable data or hard-flag-worthy defects.
Fallback ladder: (a) code-level reconstruction if not runnable; (b) if all
audited repos are clean, report that honestly as *specificity in the wild*
(detectors don't false-alarm on real released code) — still a result, weaker
headline. **Neutral framing throughout:** "we identify an unreported
dependency," a practice observation, never "paper X is wrong."

**Effort:** high (network + arbitrary code). **Leverage:** highest.

---

## R24 (P3) — Semi-real detector benchmark (inject known leakage into real EHR)

**Why:** detector sensitivity/specificity (R20) is all from synthetic injection;
reviewer asks "does it catch *subtle real* leakage, not toy contamination?"

**What:** take real MIMIC-IV (and eICU) feature matrices and inject KNOWN
contamination at controlled rates on the real substrate:
- temporal: swap in labs charted after the outcome time for a fraction p of
  positives (real charttimes), sweep p, plot detection curve.
- definitional: inject an outcome-derived feature at controlled correlation.
- informative-missingness: degrade coverage to sweep the ordering-vs-value mix.
Report detection curves + real-data operating points, contrasted with R20's
synthetic curves.

**Deliverable:** `experiments/round24_semireal_benchmark.py` +
`results/round24/`; extends §3 with a "real-substrate" column.

**Risk:** low. **Effort:** moderate, compute-only (data in hand). **Leverage:** high.

---

## R25 (P4) — Information-boundary robustness

**Why:** §7's I(X;Y)/H(Y)=0.29 uses CE* over 3 tree/linear models; reviewer
asks "a stronger model would raise the bound." The MI boundary is the paper's
most scientific claim — it must be method-robust.

**What:**
1. Extend CE* to more model families (MLP / gradient-boosting variants) and show
   convergence (bound doesn't rise materially).
2. Cross-validate I(X;Y) with an **independent estimator** — Kraskov kNN MI
   (we already have a KSG path in `round11_modlow_mi`) and/or a MINE-style
   estimate — to show ~0.29 is estimator-robust.
3. Label-noise sensitivity: perturb Y by ε∈{1,3,5}% and confirm the bound holds.

**Deliverable:** `experiments/round25_mi_robustness.py` + `results/round25/`;
tightens §7 with a robustness table.

**Risk:** low. **Effort:** low–moderate, compute-only. **Leverage:** high
(hardens the core science).

---

## R26 (P2) — Audit expansion + inter-rater Cohen's κ

**Why:** audit is n=5 EHR (CP upper 0.52) and single-coder — two guaranteed
reviewer hits in a methods journal.

**What:**
1. Aggressively search MIMIC/eICU/HiRID/AmsterdamUMCdb CP cohorts (sepsis,
   mortality, AKI, deterioration); code toward 15–20 **real-EHR** papers to push
   the CP upper bound < 0.17.
2. **Dual-code:** a second independent coder pass (independent agent) on all
   papers; compute Cohen's κ per item (D1/D2/D3); adjudicate disagreements.

**Deliverable:** expanded `results/lit_audit/`; §6 with κ + tightened bound.

**Risk:** **real-EHR CP papers are scarce** — reaching 15–20 may be infeasible.
Fallback (already partly applied): keep "field-wide" out of the title, report
κ + the honest small-n bound, and frame as "a preliminary, inter-rater-reliable
audit." κ is achievable regardless; the n target may not be.

**Effort:** moderate (search-bound). **Leverage:** high for methods-journal fit.

---

## R27 (P5, optional) — Cross-institution detector threshold transfer

**Why:** strengthens the "instrument" claim — thresholds (0.85, 0.90, 0.95)
tuned on MIMIC should carry to eICU.

**What:** apply MIMIC-calibrated detector thresholds unchanged to eICU; report
that verdicts are stable across institutions (or recalibrate and quantify drift).

**Deliverable:** `experiments/round27_threshold_transfer.py` + a §5 note.

**Risk:** low. **Effort:** low. **Leverage:** medium.

---

## Sequencing

```
Wave A (compute-only, start immediately, parallel):
  R25 (MI robustness)      — hardens the core science, fast
  R24 (semi-real bench)    — real-substrate detector curves
Wave B (network, scout in parallel with Wave A):
  R23 (repo defect hunt)   — find + audit 3-5 public clinical-CP repos   [highest leverage]
Wave C (after A/B):
  R26 (audit + Cohen κ)    — dual-code existing 17 + expand real-EHR
  R27 (threshold transfer) — quick instrument-generalization note
```

**Decision gates:**
- R23: if ≥1 hard-flag on real released code → promote to a headline §5.4 and
  the paper's "so what" is answered. If all clean → report as wild-specificity.
- R26: if real-EHR reaches ~15 with upper < 0.17 → restore a measured
  reporting-gap claim. If not → keep the lowered claim + κ only.

_Plan authored 2026-07-03. Waves A/B start now._
