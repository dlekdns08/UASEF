"""
Dual-shuffle — ANSWERER side (rewritten for label/text separation + full auditability).

Option-shuffle audit: does a model's answer/confidence track answer CONTENT or the
memorized letter POSITION? We permute the MCQ options (so the gold content moves to a
different letter) and have the answerer re-answer.

CRITICAL (per methodology review): under shuffle the letter label changes but the answer
content does not, so correctness/agreement MUST be computed on the canonical answer TEXT,
never the letter. We store every raw field so any anomaly can be diagnosed as position-bias
vs parser-artifact after the fact:

  original_options, shuffled_options, permutation_map (orig_letter -> shuffled_letter),
  gold_original_label, gold_shuffled_label, gold_answer_text (canonical),
  answerer_output_label, answerer_output_text (resolved from shuffled_options[label]),
  canonical_answer_text (== answerer_output_text; the join key for agreement),
  correct_by_text, correct_by_label (diagnostic), confidence, raw_output (truncated).

MedMCQA only (PubMedQA has no options). Resumable. Reasoning mode = whatever the loaded
model does (think/no-think set by operator template). Run:
  ANSWERER_MODEL=openai/gpt-oss-120b python experiments/phase2_shuffle_answer.py \
      --tag gptoss --n 400 --max-tokens 2048
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np


def _seed(item_id: str) -> int:
    """Deterministic per-item seed (built-in hash() is per-process randomized → would
    change the permutation on resume; hashlib is stable)."""
    return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.model_interface import query_model
from models.qa_drafts import _parse_answer, _parse_conf, SYS, _prompt as _draft_prompt
from models.qa_risk_features import QAItem

LETTERS = ["A", "B", "C", "D"]


def _norm(s):
    return (s or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="answerer tag, e.g. gptoss / qwen35")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    model = os.getenv("ANSWERER_MODEL")
    if not model:
        sys.exit("set ANSWERER_MODEL")
    os.environ["LMSTUDIO_MODEL"] = model
    out = ROOT / "data" / "raw" / f"shuffle_answer_{a.tag}.jsonl"

    from data.qa_datasets import load_medmcqa
    items = load_medmcqa(a.n, seed=a.seed)
    done = set()
    if out.exists():
        for line in open(out):
            line = line.strip()
            if line:
                done.add(json.loads(line)["item_id"])
    todo = [it for it in items if it.item_id not in done]
    # deterministic per-item permutation (reproducible, independent of run order)
    print(f"[shuffle-answer:{a.tag}] model={model}  {len(items)} items, {len(done)} cached, {len(todo)} to answer")
    with open(out, "a") as f:
        for idx, it in enumerate(todo):
            letters = [L for L in LETTERS if L in it.options]
            gold_text = it.options.get(it.gold_answer, "")
            # reproducible permutation seeded by item id (stable across resumes)
            rng = np.random.default_rng(_seed(it.item_id))
            perm = list(range(len(letters)))
            rng.shuffle(perm)
            # shuffled_options[new_letter] = original value at perm position
            shuffled = {letters[j]: it.options[letters[perm[j]]] for j in range(len(letters))}
            # permutation_map: original letter -> shuffled letter (where its content went)
            perm_map = {}
            for j in range(len(letters)):
                orig_letter = letters[perm[j]]
                perm_map[orig_letter] = letters[j]
            gold_shuffled_label = perm_map.get(it.gold_answer, it.gold_answer)
            sit = QAItem(item_id=it.item_id, dataset="medmcqa", question=it.question,
                         options=shuffled, gold_answer=gold_shuffled_label, subject=it.subject)
            try:
                r = query_model(backend="lmstudio", system_prompt=SYS,
                                user_prompt=_draft_prompt(sit), temperature=0.0,
                                max_completion_tokens=a.max_tokens, logprobs=False)
                out_label = _parse_answer(r.text, mcq=True)          # letter the model emitted
                out_text = shuffled.get(out_label, "")               # resolve to content via shuffled map
                rec = {
                    "item_id": it.item_id, "subject": it.subject, "question": it.question,
                    "original_options": it.options, "shuffled_options": shuffled,
                    "permutation_map": perm_map,                      # orig_letter -> shuffled_letter
                    "gold_original_label": it.gold_answer, "gold_shuffled_label": gold_shuffled_label,
                    "gold_answer_text": gold_text,
                    "answerer_output_label": out_label, "answerer_output_text": out_text,
                    "canonical_answer_text": out_text,                # join key for agreement (text-based)
                    "correct_by_text": int(_norm(out_text) == _norm(gold_text) and out_text != ""),
                    "correct_by_label": int(out_label == gold_shuffled_label),   # diagnostic only
                    "confidence": _parse_conf(r.text),
                    "raw_output": (r.text or "")[:400],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            except Exception as e:
                print(f"  [skip {it.item_id}] {type(e).__name__}: {str(e)[:60]}")
            if (idx + 1) % 50 == 0:
                print(f"  ...{idx + 1}/{len(todo)}")

    # sanity: content-invariance + confidence AUROC under shuffle (vs original if available)
    from models.label_conditional_conformal import _auroc
    rows = [json.loads(l) for l in open(out) if l.strip()]
    err = np.array([1 - r["correct_by_text"] for r in rows])
    conf = np.array([1 - (r["confidence"] if r["confidence"] is not None else 0.5) for r in rows])
    au = round(max(_auroc(conf, err), 1 - _auroc(conf, err)), 3) if (err.sum() >= 5 and (err == 0).sum() >= 5) else None
    # parser-artifact check: how often did text resolution fail (blank)?
    blank = sum(1 for r in rows if not (r.get("answerer_output_text") or "").strip())
    label_text_mismatch = sum(1 for r in rows if r["correct_by_text"] != r["correct_by_label"])
    print(f"[shuffle-answer:{a.tag}] n={len(rows)} acc_by_text={1 - err.mean():.3f} "
          f"conf_AUROC_shuffled={au} | 빈text(parser의심) {blank} | text/label 채점불일치 {label_text_mismatch}")
    print(f"  → text/label 불일치가 크면 position-bias 신호(정상), 빈text 많으면 parser 점검 필요")


if __name__ == "__main__":
    main()
