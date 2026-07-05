"""
Generate hidden LLM drafts for Phase 0-3 (local LMStudio backend, $0).

For each QAItem we produce a DraftRecord: one decision draft (temperature 0,
with reasoning + a verbalized confidence + token logprobs if the backend exposes
them) and k temperature samples (for self-consistency / answer entropy). Output
is cached to a JSONL that the gatekeeper / Stage-A runner consumes; generation is
**resumable** (already-done item_ids are skipped).

The drafts are "hidden" — never shown to a user; they are the substrate the
risk features are computed from. The gold answer is stored on the record for
error-labelling only (never a feature).

Usage:
  export UASEF_QUERY_TIMEOUT_S=120
  python models/qa_drafts.py --dataset medmcqa --n 3000 --k 5 \
      --out data/raw/drafts_medmcqa.jsonl
  python models/qa_drafts.py --dataset pubmedqa --n 800 --k 5 \
      --out data/raw/drafts_pubmedqa.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from models.model_interface import query_model
from models.qa_risk_features import DraftRecord, QAItem

MCQ_LETTERS = ["A", "B", "C", "D"]
SYS = ("You are answering a board-style medical question. Reason briefly, then "
       "give your final answer and a calibrated confidence.")


def _prompt(item: QAItem) -> str:
    if item.options:
        opts = "\n".join(f"{k}) {v}" for k, v in item.options.items())
        choices = "A, B, C, or D"
        body = f"Question: {item.question}\nOptions:\n{opts}"
    else:
        choices = "yes, no, or maybe"
        body = f"Question: {item.question}"
    return (f"{body}\n\nRespond in EXACTLY this format:\n"
            f"Reasoning: <one or two sentences>\n"
            f"Answer: <{choices}>\n"
            f"Confidence: <integer 0-100>")


def _parse_answer(text: str, mcq: bool) -> str:
    t = text or ""
    m = re.search(r"Answer:\s*([A-Da-d]|yes|no|maybe)", t, re.I)
    if m:
        return m.group(1).strip().upper() if mcq else m.group(1).strip().lower()
    # fallback: last standalone token
    if mcq:
        letters = re.findall(r"\b([A-D])\b", t)
        return letters[-1] if letters else ""
    m2 = re.findall(r"\b(yes|no|maybe)\b", t, re.I)
    return m2[-1].lower() if m2 else ""


def _parse_conf(text: str):
    m = re.search(r"Confidence:\s*(\d{1,3})", text or "", re.I)
    if not m:
        return None
    return max(0.0, min(1.0, int(m.group(1)) / 100.0))


def _parse_reasoning(text: str) -> str:
    m = re.search(r"Reasoning:\s*(.+?)(?:\nAnswer:|$)", text or "", re.I | re.S)
    return (m.group(1).strip() if m else (text or "").strip())[:1000]


def make_draft(item: QAItem, k: int, temp: float) -> DraftRecord:
    mcq = bool(item.options)
    # decision draft (temp 0, logprobs if the backend exposes them)
    dec = query_model(backend="lmstudio", system_prompt=SYS, user_prompt=_prompt(item),
                      temperature=0.0, max_completion_tokens=512, logprobs=True)
    decision = _parse_answer(dec.text, mcq)
    # k temperature samples (answers only)
    samples = []
    for _ in range(k):
        s = query_model(backend="lmstudio", system_prompt=SYS, user_prompt=_prompt(item),
                        temperature=temp, max_completion_tokens=512, logprobs=False)
        samples.append(_parse_answer(s.text, mcq))
    return DraftRecord(
        item_id=item.item_id, dataset=item.dataset, subject=item.subject,
        decision_answer=decision, samples=samples,
        token_logprobs=dec.logprobs, verbalized_confidence=_parse_conf(dec.text),
        reasoning_text=_parse_reasoning(dec.text), gold_answer=item.gold_answer)


def _done_ids(path: Path) -> set:
    ids = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["item_id"])
                    except Exception:
                        pass
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["medmcqa", "pubmedqa"], required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    import os
    model = a.model or os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-120b")
    os.environ["LMSTUDIO_MODEL"] = model   # query_model resolves the model from env
    from data.qa_datasets import load_medmcqa, load_pubmedqa
    items = (load_medmcqa(a.n, a.seed) if a.dataset == "medmcqa"
             else load_pubmedqa(a.n, a.seed))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out)
    todo = [it for it in items if it.item_id not in done]
    print(f"[drafts] {a.dataset}: {len(items)} items, {len(done)} cached, {len(todo)} to generate "
          f"(model={model}, k={a.k})")
    n_ok = 0
    with open(out, "a") as f:
        for i, it in enumerate(todo):
            try:
                d = make_draft(it, a.k, a.temp)
                f.write(json.dumps(asdict(d)) + "\n")
                f.flush()
                n_ok += 1
            except Exception as e:
                print(f"  [skip {it.item_id}] {type(e).__name__}: {str(e)[:80]}")
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(todo)} done ({n_ok} ok)")
    print(f"[drafts] wrote {n_ok} new -> {out} (total cached {len(done) + n_ok})")


if __name__ == "__main__":
    main()
