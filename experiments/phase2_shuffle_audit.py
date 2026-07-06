"""
Phase 2 — Option-shuffle self-deception audit (is the confidence content-based
or answer-position memorization?).

The gate is confidence-dominated. If gpt-oss-120b saw MedMCQA during pretraining,
its "confidence" might track the memorized answer POSITION rather than the
content. We regenerate the decision draft with the MCQ options RANDOMLY
PERMUTED (re-mapping the gold letter) and re-measure:

  * answer content-invariance: does the model pick the same option *value* after
    shuffling? (memorized-position models flip when the letter moves)
  * confidence validity under shuffle: does AUROC(confidence, error) survive?
    If it holds, confidence is content-based; if it collapses toward 0.5, the
    confidence rode answer position (self-deception / contamination).

MCQ only (MedMCQA). Resumable. Run:
  python experiments/phase2_shuffle_audit.py --n 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.model_interface import query_model
from models.qa_drafts import _parse_answer, _parse_conf, SYS, _prompt as _draft_prompt
from models.qa_risk_features import QAItem

LETTERS = ["A", "B", "C", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/drafts_medmcqa_shuffled.jsonl")
    a = ap.parse_args()
    os.environ["LMSTUDIO_MODEL"] = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-120b")
    # NOTE: model swap (unload gemma, load gpt-oss) is done MANUALLY by the operator
    # before running this — the script does not touch model loading.
    from data.qa_datasets import load_medmcqa
    items = load_medmcqa(a.n, seed=a.seed)
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in open(out):
            line = line.strip()
            if line:
                done.add(json.loads(line)["item_id"])
    todo = [it for it in items if it.item_id not in done]
    rng = np.random.default_rng(a.seed)
    print(f"[shuffle] {len(items)} items, {len(done)} cached, {len(todo)} to regenerate (options permuted)")
    with open(out, "a") as f:
        for i, it in enumerate(todo):
            gold_value = it.options.get(it.gold_answer, "")
            vals = [it.options[L] for L in LETTERS if L in it.options]
            perm = list(range(len(vals))); rng.shuffle(perm)
            shuffled = {LETTERS[j]: vals[perm[j]] for j in range(len(vals))}
            new_gold = next((L for L, v in shuffled.items() if v == gold_value), it.gold_answer)
            sit = QAItem(item_id=it.item_id, dataset="medmcqa", question=it.question,
                         options=shuffled, gold_answer=new_gold, subject=it.subject)
            try:
                r = query_model(backend="lmstudio", system_prompt=SYS,
                                user_prompt=_draft_prompt(sit), temperature=0.0,
                                max_completion_tokens=2048, logprobs=False)
                ans = _parse_answer(r.text, mcq=True)
                chosen_value = shuffled.get(ans, "")
                rec = {"item_id": it.item_id, "subject": it.subject,
                       "question": it.question, "shuffled_options": shuffled,
                       "orig_gold_letter": it.gold_answer, "new_gold_letter": new_gold,
                       "chosen_letter": ans, "chosen_value_correct": int(chosen_value == gold_value),
                       "confidence": _parse_conf(r.text)}
                f.write(json.dumps(rec) + "\n"); f.flush()
            except Exception as e:
                print(f"  [skip {it.item_id}] {type(e).__name__}: {str(e)[:60]}")
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(todo)}")

    # analyze vs original drafts (same items)
    from models.label_conditional_conformal import _auroc
    from experiments.phase0_gatekeeper import load_drafts
    sh = [json.loads(l) for l in open(out) if l.strip()]
    err_sh = np.array([1 - r["chosen_value_correct"] for r in sh])
    conf_sh = np.array([1 - r["confidence"] if r["confidence"] is not None else np.nan for r in sh])
    conf_sh = np.nan_to_num(conf_sh, nan=np.nanmedian(conf_sh))
    au_sh = round(max(_auroc(conf_sh, err_sh), 1 - _auroc(conf_sh, err_sh)), 3)
    # original confidence-AUROC on the same items
    origmap = {d.item_id: d for d in load_drafts("data/raw/drafts_medmcqa.jsonl")}
    ids = [r["item_id"] for r in sh if r["item_id"] in origmap]
    o_err = np.array([int((origmap[i].decision_answer or "").lower() != (origmap[i].gold_answer or "").lower()) for i in ids])
    o_conf = np.array([1 - origmap[i].verbalized_confidence if origmap[i].verbalized_confidence is not None else np.nan for i in ids])
    o_conf = np.nan_to_num(o_conf, nan=np.nanmedian(o_conf))
    au_orig = round(max(_auroc(o_conf, o_err), 1 - _auroc(o_conf, o_err)), 3)
    content_inv = round(float(np.mean([r["chosen_value_correct"] for r in sh])), 3)
    rep = {"n": len(sh), "error_prevalence_shuffled": round(float(err_sh.mean()), 4),
           "answer_accuracy_shuffled": content_inv,
           "confidence_auroc_original": au_orig, "confidence_auroc_shuffled": au_sh,
           "delta": round(au_sh - au_orig, 3),
           "reading": ("if confidence AUROC survives option shuffling (~ original), the "
                       "confidence is content-based; a large drop toward 0.5 would indicate "
                       "answer-position memorization / self-deception")}
    Path("results/phase2").mkdir(parents=True, exist_ok=True)
    Path("results/phase2/phase2_shuffle_audit.json").write_text(json.dumps(rep, indent=2))
    print(f"[shuffle] n={len(sh)} acc_shuffled={content_inv} err_prev={err_sh.mean():.3f}")
    print(f"  confidence AUROC: original {au_orig} -> shuffled {au_sh} (delta {au_sh - au_orig:+.3f})")


if __name__ == "__main__":
    main()
