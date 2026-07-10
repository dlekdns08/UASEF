"""
B1 — Capability vs Independence decomposition of an independent verifier.

A reviewer will ask: does the verifier (e.g. gemma) predict gpt-oss's errors
because it is INDEPENDENT (its errors are uncorrelated, so it catches gpt-oss's
mistakes) or merely because it is CAPABLE (it knows the right answer)? To
decompose, we have the VERIFIER MODEL answer the same questions ITSELF, then ask:

  does the verifier's risk still predict gpt-oss's error on the subset where the
  verifier ITSELF is wrong?

If AUROC stays well above 0.5 even where the verifier is wrong, the signal is
STRUCTURAL (independence), not just "the verifier knows the answer" (capability).

Generates the verifier model's own decision answers on the SAME item set as the
cross-verifier (matched by item_id), then runs the decomposition against
verifier_cross.jsonl. Resumable. Applies to any independent verifier (gemma,
qwen3.6-27b) — set SELFANSWER_MODEL and --tag.

Run:  SELFANSWER_MODEL=google/gemma-4-31b \
        python experiments/phase2_selfanswer.py --tag gemma --max-tokens 4096
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.qa_drafts import make_draft
from models.label_conditional_conformal import _auroc
from experiments.phase2_cross_verifier import _item_map, reload_model

def _norm(s):
    return (s or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="verifier tag, e.g. gemma / qwen27")
    ap.add_argument("--verifier-file", default="data/raw/verifier_cross.jsonl",
                    help="THIS verifier's cross-verifier output — supplies the item set AND the "
                         "verifier_risk used in the decomposition. Must be the SAME model's file "
                         "(gemma->verifier_cross.jsonl, qwen27->verifier_qwen27.jsonl).")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--k", type=int, default=0, help="self-consistency samples (>0 adds disagreement feature for Exp2)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--reload-every", type=int, default=0)
    ap.add_argument("--reload-context", type=int, default=0)
    ap.add_argument("--reload-parallel", type=int, default=0)
    a = ap.parse_args()
    model = os.getenv("SELFANSWER_MODEL")
    if not model:
        sys.exit("set SELFANSWER_MODEL (the verifier model answering the questions itself)")
    os.environ["LMSTUDIO_MODEL"] = model
    VER = ROOT / a.verifier_file
    out = ROOT / "data" / "raw" / f"selfanswer_{a.tag}.jsonl"

    # item set = this verifier's items (matched by id)
    ver = [json.loads(l) for l in open(VER) if l.strip()]
    imap = _item_map()
    items = [imap[r["item_id"]] for r in ver if r["item_id"] in imap]
    done = set()
    if out.exists():
        for line in open(out):
            line = line.strip()
            if line:
                done.add(json.loads(line)["item_id"])
    todo = [it for it in items if it.item_id not in done]
    print(f"[selfanswer:{a.tag}] model={model}  {len(items)} items, {len(done)} cached, {len(todo)} to answer")
    with open(out, "a") as f:
        for i, it in enumerate(todo):
            if a.reload_every and i > 0 and i % a.reload_every == 0:
                print(f"  [reload] {i}개 후 {model} 재로드...", flush=True)
                reload_model(model, a.reload_context, a.reload_parallel)
            try:
                d = make_draft(it, k=a.k, temp=a.temp, max_tokens=a.max_tokens)
                lp = d.token_logprobs or []
                # Exp2 competence-proxy features: verifier's own uncertainty when it self-answers
                samp = [s for s in (d.samples or []) if (s or "").strip()]
                sc_disagree = (1 - max((samp.count(x) for x in set(samp)), default=0) / len(samp)) if samp else None
                f.write(json.dumps({"item_id": it.item_id, "self_answer": d.decision_answer,
                                    "self_correct": int(_norm(d.decision_answer) == _norm(it.gold_answer)),
                                    "verbalized_confidence": d.verbalized_confidence,
                                    "neg_logprob_mean": (round(-float(np.mean(lp)), 4) if lp else None),
                                    "reasoning_len": len(d.reasoning_text or ""),
                                    "parser_ok": int(bool((d.decision_answer or "").strip())),
                                    "self_consistency_disagree": sc_disagree,
                                    "samples": d.samples}) + "\n")
                f.flush()
            except Exception as e:
                print(f"  [skip {it.item_id}] {type(e).__name__}: {str(e)[:60]}")
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(todo)}")

    # ── decomposition: verifier risk (on gpt-oss) predicts gpt-oss error, split by
    #    whether the verifier ITSELF is correct on that item ──
    self_map = {r["item_id"]: r for r in [json.loads(l) for l in open(out) if l.strip()]}
    rows = [r for r in ver if r["verifier_risk"] is not None and r["item_id"] in self_map]
    sym = lambda A, B: max(_auroc(A, B), 1 - _auroc(A, B))
    risk = np.array([r["verifier_risk"] for r in rows])
    gpt_err = np.array([r["error"] for r in rows])
    self_ok = np.array([self_map[r["item_id"]]["self_correct"] for r in rows])
    rep = {"tag": a.tag, "model": model, "n": len(rows),
           "verifier_self_accuracy": round(float(self_ok.mean()), 4),
           "auroc_all": round(sym(risk, gpt_err), 3)}
    for name, mask in [("where_verifier_CORRECT", self_ok == 1), ("where_verifier_WRONG", self_ok == 0)]:
        m = mask
        if gpt_err[m].sum() >= 5 and (gpt_err[m] == 0).sum() >= 5:
            rep[name] = {"n": int(m.sum()), "gpt_oss_errors": int(gpt_err[m].sum()),
                         "auroc": round(sym(risk[m], gpt_err[m]), 3)}
        else:
            rep[name] = {"n": int(m.sum()), "auroc": None, "note": "too few for AUROC"}
    rep["reading"] = ("if AUROC stays high on where_verifier_WRONG, the verifier's error-prediction is "
                      "STRUCTURAL (independence) not just capability (knowing the answer).")
    outp = ROOT / "results" / "phase2" / f"phase2_decomp_{a.tag}"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp) + ".json").write_text(json.dumps(rep, indent=2))
    print(f"[decomp:{a.tag}] self-acc={rep['verifier_self_accuracy']} | AUROC all={rep['auroc_all']} "
          f"| verifier-CORRECT={rep['where_verifier_CORRECT'].get('auroc')} "
          f"| verifier-WRONG={rep['where_verifier_WRONG'].get('auroc')}")
    print(f"  wrote {outp}.json")


if __name__ == "__main__":
    main()
