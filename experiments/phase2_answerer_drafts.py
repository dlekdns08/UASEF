"""
길 B — 2nd answerer full drafts on the COMMON 1500 item set (by item_id).

To add a matrix row for a different answerer (e.g. Qwen3.5-122B) we need that model's
FULL drafts on the SAME items as the verifier subset: decision answer + k self-consistency
samples + logprob + verbalized-confidence + hedging (the 5 self-uncertainty features), so
its OWN self-confidence baseline AUROC is comparable to gpt-oss's. This reuses make_draft
(identical feature extraction) but targets the exact item_ids of the common set — cheap
vs re-answering all 3800.

acc(answerer) and its self-conf AUROC come straight from these drafts; verifiers then judge
this file via phase2_cross_verifier.py --drafts drafts_<tag>.jsonl.

Resumable. Run:
  ANSWERER_MODEL=qwen3.5-122b-a10b python experiments/phase2_answerer_drafts.py \
      --tag qwen35 --k 5 --temp 0.7 --max-tokens 640
"""
from __future__ import annotations

import argparse, json, os, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.qa_drafts import make_draft
from experiments.phase2_cross_verifier import _item_map

VER = ROOT / "data" / "raw" / "verifier_cross.jsonl"  # defines the common 1500 item set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="answerer tag, e.g. qwen35")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--item-source", default=str(VER),
                    help="jsonl whose item_ids define the common set (default verifier_cross.jsonl)")
    a = ap.parse_args()
    model = os.getenv("ANSWERER_MODEL")
    if not model:
        sys.exit("set ANSWERER_MODEL (the answering model)")
    os.environ["LMSTUDIO_MODEL"] = model
    out = ROOT / "data" / "raw" / f"drafts_{a.tag}.jsonl"

    ids = [json.loads(l)["item_id"] for l in open(a.item_source) if l.strip()]
    imap = _item_map()
    items = [imap[i] for i in ids if i in imap]
    done = set()
    if out.exists():
        for line in open(out):
            line = line.strip()
            if line:
                done.add(json.loads(line)["item_id"])
    todo = [it for it in items if it.item_id not in done]
    print(f"[answerer:{a.tag}] model={model}  {len(items)} items, {len(done)} cached, {len(todo)} to answer "
          f"(k={a.k}, ~{a.k + 1} calls/item)")
    with open(out, "a") as f:
        for i, it in enumerate(todo):
            try:
                d = make_draft(it, k=a.k, temp=a.temp, max_tokens=a.max_tokens)
                f.write(json.dumps(asdict(d)) + "\n"); f.flush()
            except Exception as e:
                print(f"  [skip {it.item_id}] {type(e).__name__}: {str(e)[:60]}")
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(todo)}")
    print(f"[answerer:{a.tag}] done -> {out}  (다음: 각 verifier가 --drafts {out.name} 판정)")


if __name__ == "__main__":
    main()
