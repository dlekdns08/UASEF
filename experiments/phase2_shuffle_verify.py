"""
Phase 2 — Option-shuffle audit of the INDEPENDENT VERIFIER (gemma).

Symmetry with phase2_shuffle_audit.py (which tests whether gpt-oss's own
confidence survives option-shuffling): here we test whether the *independent
verifier's* judgment is content-based or answer-key memorization. If gemma-4-31b
saw MedMCQA in pretraining, its high verifier AUROC could be "I recall the
correct option position" rather than genuine verification — which would make the
"independent verifier is better" result itself a contamination artifact.

We take gpt-oss's SHUFFLED-option drafts (data/raw/drafts_medmcqa_shuffled.jsonl,
which stores the shuffled options + gpt-oss's chosen answer), have gemma judge
each, and compare gemma's AUROC(verifier_risk, error) on shuffled vs the original
(from data/raw/verifier_cross.jsonl) for the SAME items.

  * gemma AUROC survives shuffle  -> content-based verification (genuine)
  * gemma AUROC collapses to ~0.5 -> the verifier also rode memorized positions

REQUIRES gemma-4-31b loaded (operator swaps models manually). Resumable.
Run:  VERIFIER_MODEL=google/gemma-4-31b \
        python experiments/phase2_shuffle_verify.py
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.qa_risk_features import QAItem
from models.label_conditional_conformal import _auroc
from experiments.phase2_cross_verifier import _prompt, _query_verifier

SHUF = ROOT / "data" / "raw" / "drafts_medmcqa_shuffled.jsonl"
OUT = ROOT / "data" / "raw" / "verifier_shuffled.jsonl"
ORIG = ROOT / "data" / "raw" / "verifier_cross.jsonl"


def main():
    vmodel = os.getenv("VERIFIER_MODEL", "google/gemma-4-31b")
    os.environ["LMSTUDIO_MODEL"] = vmodel
    if not SHUF.exists():
        sys.exit(f"missing {SHUF}; run phase2_shuffle_audit.py (gpt-oss) first")
    shuf = [json.loads(l) for l in open(SHUF) if l.strip()]
    done = set()
    if OUT.exists():
        for l in open(OUT):
            l = l.strip()
            if l:
                done.add(json.loads(l)["item_id"])
    todo = [s for s in shuf if s["item_id"] not in done]
    print(f"[shuffle-verify] model={vmodel}  {len(shuf)} shuffled drafts, {len(done)} cached, {len(todo)} to judge")
    with open(OUT, "a") as f:
        for i, s in enumerate(todo):
            it = QAItem(item_id=s["item_id"], dataset="medmcqa", question=s.get("question", ""),
                        options=s.get("shuffled_options", {}), gold_answer=s.get("new_gold_letter", ""),
                        subject=s.get("subject", ""))
            try:
                vr, vtext = _query_verifier(it, s.get("chosen_letter", ""))
                f.write(json.dumps({"item_id": s["item_id"], "verifier_risk_shuffled": vr,
                                    "error": 1 - int(s["chosen_value_correct"]), "vtext": vtext}) + "\n")
                f.flush()
            except Exception as e:
                print(f"  [skip {s['item_id']}] {type(e).__name__}: {str(e)[:60]}")
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(todo)}")

    # ── analysis: gemma AUROC shuffled vs original, same items ──
    sym = lambda a, b: max(_auroc(a, b), 1 - _auroc(a, b))
    sh = [json.loads(l) for l in open(OUT) if l.strip()]
    sh = [r for r in sh if r["verifier_risk_shuffled"] is not None]
    e_sh = np.array([r["error"] for r in sh])
    v_sh = np.array([r["verifier_risk_shuffled"] for r in sh])
    au_sh = sym(v_sh, e_sh) if (e_sh.sum() >= 5 and (e_sh == 0).sum() >= 5) else float("nan")

    orig = {json.loads(l)["item_id"]: json.loads(l) for l in open(ORIG) if l.strip()} if ORIG.exists() else {}
    ids = [r["item_id"] for r in sh if r["item_id"] in orig and orig[r["item_id"]]["verifier_risk"] is not None]
    e_or = np.array([orig[i]["error"] for i in ids])
    v_or = np.array([orig[i]["verifier_risk"] for i in ids])
    au_or = sym(v_or, e_or) if (e_or.sum() >= 5 and (e_or == 0).sum() >= 5) else float("nan")

    rep = {"model": vmodel, "n_shuffled": len(sh), "n_paired_original": len(ids),
           "error_prevalence_shuffled": round(float(e_sh.mean()), 4),
           "gemma_verifier_auroc_shuffled": round(au_sh, 3),
           "gemma_verifier_auroc_original_sameitems": round(au_or, 3),
           "delta": round(au_sh - au_or, 3) if au_sh == au_sh and au_or == au_or else None,
           "reading": ("if gemma's verifier AUROC survives option-shuffling (~ original), its "
                       "verification is content-based (genuine independent signal); a large drop "
                       "toward 0.5 would mean the verifier also memorized answer positions.")}
    (ROOT / "results" / "phase2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "phase2" / "phase2_shuffle_verify.json").write_text(json.dumps(rep, indent=2))
    print(f"[shuffle-verify] gemma AUROC: shuffled {au_sh:.3f} vs original(same items) {au_or:.3f} "
          f"(delta {rep['delta']})  n_sh={len(sh)} n_pair={len(ids)}")


if __name__ == "__main__":
    main()
