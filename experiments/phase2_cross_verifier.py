"""
Phase 2 — Cross-model verifier audit (decouples the escalation signal from the
answering model).

The Stage-A gate is confidence-dominated (phase2_qa_audit: confidence-only
recovers 96% of the risk signal), and that confidence comes from the SAME model
that produced the answer (gpt-oss-120b). This audit tests whether an INDEPENDENT
model — a different local LLM used ONLY as a risk feature, never as ground truth —
can predict gpt-oss's errors. If it can, escalation need not rely on the
answerer's self-report.

For each cached gpt-oss draft we ask a *different* model to judge the proposed
answer's risk (0-100, higher = more likely wrong), WITHOUT the gold answer. We
then compare AUROC(verifier_risk, error) [independent] against
AUROC(gpt-oss self-confidence, error) [same-model], and whether combining the
two beats either alone.

Resumable. Run:
  VERIFIER_MODEL=qwen/qwen3.6-35b-a3b python experiments/phase2_cross_verifier.py \
      --drafts data/raw/drafts_phase0_all.jsonl --n 1500
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.model_interface import query_model
from models.qa_risk_features import error_label
from experiments.phase0_gatekeeper import load_drafts

VSYS = ("You are a careful medical reviewer. You are shown a question and a "
        "proposed answer from another AI. Judge how likely the proposed answer "
        "is WRONG. You are NOT given the correct answer.")


def _item_map():
    """item_id -> (question, options_dict) by reloading the same samples."""
    from data.qa_datasets import load_medmcqa, load_pubmedqa
    m = {}
    for it in load_medmcqa(3000, seed=42) + load_pubmedqa(800, seed=42):
        m[it.item_id] = it
    return m


def _prompt(item, proposed):
    if item.options:
        opts = "\n".join(f"{k}) {v}" for k, v in item.options.items())
        q = f"Question: {item.question}\nOptions:\n{opts}\nProposed answer: {proposed}"
    else:
        q = f"Question: {item.question}\nProposed answer: {proposed}"
    return (f"{q}\n\nRespond EXACTLY:\nVerdict: <correct or incorrect>\n"
            f"Risk: <integer 0-100, 100 = almost certainly wrong>")


def _parse_risk(text):
    t = re.sub(r"[*#`]", "", text or "")
    m = re.search(r"Risk:\s*(\d{1,3})", t, re.I)
    if m:
        return max(0.0, min(1.0, int(m.group(1)) / 100.0))
    v = re.search(r"Verdict:\s*(correct|incorrect)", t, re.I)
    if v:
        return 0.8 if v.group(1).lower() == "incorrect" else 0.2
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="data/raw/verifier_cross.jsonl")
    a = ap.parse_args()
    vmodel = os.getenv("VERIFIER_MODEL", "qwen/qwen3.6-35b-a3b")
    os.environ["LMSTUDIO_MODEL"] = vmodel

    drafts = load_drafts(a.drafts)[: a.n]
    imap = _item_map()
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in open(out):
            line = line.strip()
            if line:
                done.add(json.loads(line)["item_id"])
    todo = [d for d in drafts if d.item_id not in done and d.item_id in imap]
    print(f"[cross-verifier] model={vmodel}  {len(drafts)} drafts, {len(done)} cached, {len(todo)} to judge")
    with open(out, "a") as f:
        for i, d in enumerate(todo):
            try:
                r = query_model(backend="lmstudio", system_prompt=VSYS,
                                user_prompt=_prompt(imap[d.item_id], d.decision_answer),
                                temperature=0.0, max_completion_tokens=400, logprobs=False)
                vr = _parse_risk(r.text)
                f.write(json.dumps({"item_id": d.item_id, "verifier_risk": vr,
                                    "gpt_oss_conf": d.verbalized_confidence,
                                    "error": error_label(d)}) + "\n"); f.flush()
            except Exception as e:
                print(f"  [skip {d.item_id}] {type(e).__name__}: {str(e)[:60]}")
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(todo)}")

    # analyze
    from models.label_conditional_conformal import _auroc
    rows = [json.loads(l) for l in open(out) if l.strip()]
    rows = [r for r in rows if r["verifier_risk"] is not None]
    err = np.array([r["error"] for r in rows])
    vrisk = np.array([r["verifier_risk"] for r in rows])
    conf = np.array([1 - r["gpt_oss_conf"] if r["gpt_oss_conf"] is not None else np.nan for r in rows])
    conf = np.nan_to_num(conf, nan=np.nanmedian(conf))
    au_v = round(max(_auroc(vrisk, err), 1 - _auroc(vrisk, err)), 3)
    au_c = round(max(_auroc(conf, err), 1 - _auroc(conf, err)), 3)
    combo = 0.5 * (vrisk - vrisk.mean()) / (vrisk.std() + 1e-9) + 0.5 * (conf - conf.mean()) / (conf.std() + 1e-9)
    au_combo = round(max(_auroc(combo, err), 1 - _auroc(combo, err)), 3)
    rep = {"verifier_model": vmodel, "n": len(rows), "error_prevalence": round(float(err.mean()), 4),
           "auroc_independent_verifier": au_v, "auroc_same_model_confidence": au_c,
           "auroc_combined": au_combo,
           "reading": ("an independent model predicts gpt-oss errors at AUROC "
                       f"{au_v} (vs same-model confidence {au_c}); escalation can be "
                       "driven by a different model, not only the answerer's self-report")}
    Path("results/phase2").mkdir(parents=True, exist_ok=True)
    Path("results/phase2/phase2_cross_verifier.json").write_text(json.dumps(rep, indent=2))
    print(f"[cross-verifier] n={len(rows)} err_prev={err.mean():.3f}")
    print(f"  AUROC independent verifier ({vmodel}) = {au_v}")
    print(f"  AUROC same-model confidence          = {au_c}")
    print(f"  AUROC combined                        = {au_combo}")


if __name__ == "__main__":
    main()
