"""
Dual-shuffle — VERIFIER side (rewritten for label/text separation + agreement-by-text).

Reads a shuffle_answer_<answerer>.jsonl (produced by phase2_shuffle_answer.py, which stores
the shuffled options + the answerer's canonical answer TEXT) and has a verifier model:
  (1) give its OWN answer to the shuffled question  -> verifier_pred_label/text
  (2) score the proposed answer's risk (0-100)      -> verifier_risk

Agreement/disagreement is computed on the canonical answer TEXT, never the letter — under
shuffle the letter label changes, so a label-based agreement would be an artifact. We store
both by_label (diagnostic) and by_text (the safe one). The proposed answer is passed to the
verifier content-unambiguously as "<letter>) <text>".

Signal for robustness = disagreement-by-text (verifier answer != answerer answer, by text),
which is exactly the A8 "model-disagreement" signal and is fully content-based; risk stored
too. Compare shuffled vs original (same items) to see if the signal is content-based
(survives) or letter-position memorization (collapses).

MedMCQA only. Works for any verifier in any reasoning mode (operator sets think/no-think).
Resumable. Run:
  VERIFIER_MODEL=google/gemma-4-31b python experiments/phase2_shuffle_judge.py \
      --answerer gptoss --tag gemma_T --max-tokens 4096
"""
from __future__ import annotations

import argparse, json, os, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.model_interface import query_model
from models.label_conditional_conformal import _auroc
from experiments.phase2_cross_verifier import reload_model

VJSYS = ("You are a careful medical expert. You are shown a multiple-choice question, its "
         "options, and another AI's proposed answer. First give YOUR OWN best answer (a "
         "letter), then rate how likely the PROPOSED answer is WRONG.")


def _norm(s):
    return (s or "").strip().lower()


def _prompt(question, options, proposed_letter, proposed_text):
    opts = "\n".join(f"{k}) {v}" for k, v in options.items())
    return (f"Question: {question}\nOptions:\n{opts}\n"
            f"Proposed answer: {proposed_letter}) {proposed_text}\n\n"
            f"Respond EXACTLY:\nYour answer: <A, B, C, or D>\n"
            f"Risk: <integer 0-100, 100 = proposed almost certainly wrong>")


def _parse(text):
    t = re.sub(r"[*#`]", "", text or "")
    ml = re.search(r"Your answer:\s*\(?([A-Da-d])\b", t, re.I)
    label = ml.group(1).upper() if ml else ""
    mr = re.search(r"Risk[:\s]*\(?(\d{1,3})", t, re.I) or re.search(r"(\d{1,3})\s*/\s*100", t)
    risk = max(0.0, min(1.0, int(mr.group(1)) / 100.0)) if mr else None
    return label, risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", required=True, help="answerer tag (reads shuffle_answer_<answerer>.jsonl)")
    ap.add_argument("--tag", required=True, help="this verifier+mode tag, e.g. gemma_T / gemma_N")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #items (e.g. 100 for the original-reference sanity re-judge; 0=all)")
    ap.add_argument("--reload-every", type=int, default=0)
    ap.add_argument("--reload-context", type=int, default=0)
    ap.add_argument("--reload-parallel", type=int, default=0)
    a = ap.parse_args()
    vmodel = os.getenv("VERIFIER_MODEL")
    if not vmodel:
        sys.exit("set VERIFIER_MODEL")
    os.environ["LMSTUDIO_MODEL"] = vmodel
    src = ROOT / "data" / "raw" / f"shuffle_answer_{a.answerer}.jsonl"
    out = ROOT / "data" / "raw" / f"shuffle_judge_{a.answerer}__{a.tag}.jsonl"
    if not src.exists():
        sys.exit(f"missing {src}; run phase2_shuffle_answer.py --tag {a.answerer} first")
    shuf = {json.loads(l)["item_id"]: json.loads(l) for l in open(src) if l.strip()}
    done = set()
    if out.exists():
        for l in open(out):
            l = l.strip()
            if l:
                done.add(json.loads(l)["item_id"])
    todo = [s for s in shuf.values() if s["item_id"] not in done]
    if a.limit:
        # deterministic subset (sorted by item_id) so the sanity sample is reproducible
        capped = sorted(shuf.values(), key=lambda s: s["item_id"])[: a.limit]
        cap_ids = {s["item_id"] for s in capped}
        todo = [s for s in todo if s["item_id"] in cap_ids]
    print(f"[shuffle-judge:{a.tag}] verifier={vmodel} on answerer={a.answerer}  "
          f"{len(shuf)} items, {len(done)} cached, {len(todo)} to judge"
          + (f" (limit {a.limit})" if a.limit else ""))
    with open(out, "a") as f:
        for i, s in enumerate(todo):
            if a.reload_every and i > 0 and i % a.reload_every == 0:
                print(f"  [reload] {i}개 후 {vmodel} 재로드...", flush=True)
                reload_model(vmodel, a.reload_context, a.reload_parallel)
            opts = s["shuffled_options"]
            prop_label = s.get("answerer_output_label", "")
            prop_text = s.get("canonical_answer_text", "")
            last = ""
            vlabel, vrisk = "", None
            for attempt in range(a.retries):
                try:
                    r = query_model(backend="lmstudio", system_prompt=VJSYS,
                                    user_prompt=_prompt(s["question"], opts, prop_label, prop_text),
                                    temperature=0.0, max_completion_tokens=a.max_tokens + 1024 * attempt,
                                    logprobs=False)
                    last = r.text or ""
                    if last.strip():
                        vlabel, vrisk = _parse(last)
                        break
                except Exception as e:
                    print(f"  [err {s['item_id']}] {type(e).__name__}: {str(e)[:50]}")
            v_text = opts.get(vlabel, "")                          # verifier's own answer, resolved to TEXT
            ans_text = s.get("canonical_answer_text", "")          # answerer's answer TEXT
            rec = {
                "item_id": s["item_id"],
                "verifier_pred_label": vlabel, "verifier_pred_text": v_text,
                "verifier_risk": vrisk,
                "answerer_output_label": prop_label, "canonical_answer_text": ans_text,
                # agreement computed on TEXT (safe under shuffle) + label (diagnostic)
                "agreement_by_text": int(v_text != "" and _norm(v_text) == _norm(ans_text)),
                "agreement_by_label": int(vlabel != "" and vlabel == prop_label),
                # NOT independent Z: the verifier chose this answer AFTER seeing the answerer's
                # proposed answer (confirmation-bias-contaminated). Independent competence Z
                # comes only from self-answer files (no proposed answer shown, original order).
                "judge_selected_correct": int(v_text != "" and _norm(v_text) == _norm(s.get("gold_answer_text", ""))),
                "error": 1 - int(s["correct_by_text"]),            # answerer error, TEXT-based
                "vtext": last[:200],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(todo)}")

    # analysis: does the signal survive shuffle? (disagreement-by-text AND risk)
    sym = lambda A, B: round(max(_auroc(A, B), 1 - _auroc(A, B)), 3)
    rows = [json.loads(l) for l in open(out) if l.strip()]
    err = np.array([r["error"] for r in rows])
    disagree = np.array([1 - r["agreement_by_text"] for r in rows])   # verifier disagrees (by text)
    valid_risk = [r for r in rows if r["verifier_risk"] is not None]
    au_dis = sym(disagree, err) if (err.sum() >= 5 and (err == 0).sum() >= 5) else None
    au_risk = None
    if len(valid_risk) >= 20:
        er = np.array([r["error"] for r in valid_risk]); ri = np.array([r["verifier_risk"] for r in valid_risk])
        if er.sum() >= 5 and (er == 0).sum() >= 5:
            au_risk = sym(ri, er)
    p_err_dis = round(float(err[disagree == 1].mean()), 3) if (disagree == 1).sum() else None
    p_err_agr = round(float(err[disagree == 0].mean()), 3) if (disagree == 0).sum() else None
    none = sum(1 for r in rows if r["verifier_risk"] is None)
    rep = {"verifier": vmodel, "tag": a.tag, "answerer": a.answerer, "n": len(rows), "none_risk": none,
           "auroc_disagreement_by_text": au_dis, "auroc_risk": au_risk,
           "P_err_given_disagree_text": p_err_dis, "P_err_given_agree_text": p_err_agr,
           "reading": "compare to original (same items): if the disagreement/risk signal survives "
                      "option-shuffling it is content-based; collapse toward 0.5 = letter-position memorization."}
    (ROOT / "results" / "phase2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "phase2" / f"phase2_shuffle_judge_{a.answerer}__{a.tag}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[shuffle-judge:{a.tag}] n={len(rows)} none={none} | AUROC(불일치-text)={au_dis} AUROC(risk)={au_risk}")
    print(f"  P(오류|불일치)={p_err_dis} vs P(오류|일치)={p_err_agr}  → 원본과 비교해 유지/붕괴 판정")


if __name__ == "__main__":
    main()
