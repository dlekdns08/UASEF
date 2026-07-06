"""
A6 — Verifier-augmented Phase-1 gate (practicality: does the independent verifier
turn the escalation gate more USEFUL at the same safety?).

The Stage-A gate escalates ~62% at alpha=0.10 (release 0.38) — the "so what"
weak spot. Here we add the INDEPENDENT verifier's risk (gemma judging gpt-oss's
answers) as one more feature to the risk scorer, refit the label-conditional
conformal gate on the SAME 1500-item subset (base = 5 self-features; augmented =
5 + verifier), and compare on the LOCKED test:

  * both must still satisfy  P(release | incorrect) <= alpha
  * does the augmented gate RELEASE MORE at the same alpha? (higher autonomy at
    equal safety) → turns the 38% weakness into a result.

No LLM — reuses cached verifier scores + drafts. Subset = the 1495 items with a
valid verifier score (5 non-terminating verifier items excluded, reported).

Run: python experiments/phase2_verifier_augmented.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.qa_risk_features import feature_matrix
from models.label_conditional_conformal import LabelConditionalConformal, _auroc
from experiments.phase0_gatekeeper import load_drafts
from experiments.phase1_stage_a import _subject_split, _risk_scorer

VER = ROOT / "data" / "raw" / "verifier_cross.jsonl"
DRAFTS = ROOT / "data" / "raw" / "drafts_phase0_all.jsonl"


def _gate(Xf, y, tr, ca, te, seed=0):
    """Fit risk scorer on TRAIN, conformal on CAL, evaluate on locked TEST."""
    risk = _risk_scorer(Xf[tr], y[tr], Xf, seed=seed)
    au = round(max(_auroc(risk[te], y[te]), 1 - _auroc(risk[te], y[te])), 3)
    out = {}
    for alpha in (0.10, 0.05):
        g = LabelConditionalConformal(alpha).fit(risk[ca], y[ca], check_orient=False)
        r = g.evaluate(risk[te], y[te])
        out[str(alpha)] = {
            "released_given_incorrect": round(r.released_given_incorrect, 4),
            "release_rate": round(r.release_rate, 4),
            "over_escalation": round(r.over_escalation, 4),
            "feasible": g._fit.feasible,
            "alpha_satisfied": r.extra["alpha_satisfied"]}
    return out, au


def main():
    vmap = {}
    for line in open(VER):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["verifier_risk"] is not None:
                vmap[r["item_id"]] = float(r["verifier_risk"])
    drafts = [d for d in load_drafts(str(DRAFTS)) if d.item_id in vmap]
    X, y, names = feature_matrix(drafts)
    vrisk = np.array([vmap[d.item_id] for d in drafts])
    Xaug = np.column_stack([X, vrisk])
    subj = np.array([d.subject for d in drafts])
    tr, ca, te = _subject_split(subj, seed=0)

    base, base_au = _gate(X, y, tr, ca, te)
    aug, aug_au = _gate(Xaug, y, tr, ca, te)

    report = {"n": len(drafts), "note": "1495 items with valid verifier score (5 excluded)",
              "test_error_prevalence": round(float(y[te].mean()), 4),
              "risk_auroc_test": {"base": base_au, "augmented": aug_au},
              "base_gate": base, "verifier_augmented_gate": aug, "by_alpha_delta": {}}
    for a in ("0.1", "0.05"):
        b, g = base[a], aug[a]
        report["by_alpha_delta"][a] = {
            "release_rate_base": b["release_rate"], "release_rate_augmented": g["release_rate"],
            "release_gain": round(g["release_rate"] - b["release_rate"], 4),
            "both_satisfy_alpha": bool(b["alpha_satisfied"] and g["alpha_satisfied"])}
    outp = ROOT / "results" / "phase2" / "phase2_verifier_augmented"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp) + ".json").write_text(json.dumps(report, indent=2))

    print(f"[A6] n={len(drafts)} (verifier-valid) | risk AUROC(test) base={base_au} augmented={aug_au}")
    for a in ("0.1", "0.05"):
        d = report["by_alpha_delta"][a]
        print(f"  alpha={a}: release base={d['release_rate_base']} -> augmented={d['release_rate_augmented']} "
              f"(gain {d['release_gain']:+.3f}) | both satisfy alpha: {d['both_satisfy_alpha']} "
              f"| rel|inc base={base[a]['released_given_incorrect']} aug={aug[a]['released_given_incorrect']}")
    print(f"  wrote {outp}.json")


if __name__ == "__main__":
    main()
