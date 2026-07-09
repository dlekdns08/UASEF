"""
Sanity-check support — build the ORIGINAL-reference file (LLM 0).

For the prompt-confound sanity check we re-judge a small sample of ORIGINAL (unshuffled)
items with the shuffle-judge prompt and compare against the existing matrix/self-answer
originals. This script emits the original items in the SAME schema as
phase2_shuffle_answer.py (so phase2_shuffle_judge.py can consume them unchanged), but with
NO shuffle: shuffled_options == original_options, identity permutation, and the answerer's
EXISTING answer (from its drafts) as the canonical answer text. No LLM call — pure reuse.

Item set = the same ~400 common-set MedMCQA items the shuffle uses (so paired by item_id).

Run:  python experiments/phase2_shuffle_reference.py --tag gptoss \
          --answerer-drafts data/raw/drafts_phase0_all.jsonl --n 400
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from experiments.phase0_gatekeeper import load_drafts
from experiments.phase2_cross_verifier import _item_map

LETTERS = ["A", "B", "C", "D"]


def _norm(s):
    return (s or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="answerer tag (matches shuffle_answer_<tag>)")
    ap.add_argument("--answerer-drafts", required=True, help="the answerer's existing drafts jsonl")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    # named so phase2_shuffle_judge.py --answerer <tag>_ref consumes it unchanged
    out = ROOT / "data" / "raw" / f"shuffle_answer_{a.tag}_ref.jsonl"

    common = ROOT / "data" / "raw" / "verifier_cross.jsonl"
    mc_ids = [json.loads(l)["item_id"] for l in open(common) if l.strip()
              if json.loads(l)["item_id"].startswith("medmcqa")][: a.n]
    imap = _item_map()
    drafts = {d.item_id: d for d in load_drafts(a.answerer_drafts)}

    n = miss = 0
    with open(out, "w") as f:
        for iid in mc_ids:
            if iid not in imap or iid not in drafts:
                miss += 1
                continue
            it = imap[iid]
            d = drafts[iid]
            gold_text = it.options.get(it.gold_answer, "")
            ans_label = (d.decision_answer or "").strip().upper()
            ans_text = it.options.get(ans_label, "")            # answerer's ORIGINAL answer text
            rec = {
                "item_id": iid, "subject": it.subject, "question": it.question,
                "original_options": it.options, "shuffled_options": it.options,   # NO shuffle
                "permutation_map": {L: L for L in it.options},                     # identity
                "gold_original_label": it.gold_answer, "gold_shuffled_label": it.gold_answer,
                "gold_answer_text": gold_text,
                "answerer_output_label": ans_label, "answerer_output_text": ans_text,
                "canonical_answer_text": ans_text,
                "correct_by_text": int(ans_text != "" and _norm(ans_text) == _norm(gold_text)),
                "correct_by_label": int(ans_label == it.gold_answer),
                "confidence": d.verbalized_confidence,
                "raw_output": (d.reasoning_text or "")[:200],
                "is_original_reference": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[shuffle-ref:{a.tag}] wrote {n} original-reference records ({miss} missing) -> {out}")
    print(f"  → sanity: VERIFIER_MODEL=... phase2_shuffle_judge.py --answerer {a.tag}_ref --tag <v>_<mode> --limit 100")


if __name__ == "__main__":
    main()
