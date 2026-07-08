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

import argparse, json, os, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from models.model_interface import query_model
from models.qa_drafts import make_draft
from experiments.phase2_cross_verifier import _item_map

VER = ROOT / "data" / "raw" / "verifier_cross.jsonl"  # defines the common 1500 item set
LMS = os.path.expanduser("~/.lmstudio/bin/lms")


def reload_model(model, ctx, par):
    """Unload+reload the SAME model to reset the slowdown that accrues over long runs
    (Apple-Silicon memory pressure / KV fragmentation). Memory-safe (one model at a
    time). Matches the operator's context/parallel so the reloaded config is identical;
    the thinking template lives in the model config and persists across reloads."""
    try:
        subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=120)
        cmd = [LMS, "load", model, "-y"]
        if ctx:
            cmd += ["-c", str(ctx)]
        if par:
            cmd += ["--parallel", str(par)]
        subprocess.run(cmd, capture_output=True, timeout=600)
    except Exception as e:
        print(f"  [reload] 명령 오류: {type(e).__name__}: {str(e)[:60]}")
    # readiness: retry a trivial query until the model answers again
    for _ in range(40):
        try:
            query_model(backend="lmstudio", system_prompt="", user_prompt="ok",
                        temperature=0.0, max_completion_tokens=1, logprobs=False)
            return True
        except Exception:
            time.sleep(5)
    print("  [reload] 준비 확인 실패 — 계속 시도")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="answerer tag, e.g. qwen35")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--item-source", default=str(VER),
                    help="jsonl whose item_ids define the common set (default verifier_cross.jsonl)")
    ap.add_argument("--reload-every", type=int, default=0,
                    help="unload+reload the SAME model every N items to reset long-run slowdown (0=off)")
    ap.add_argument("--reload-context", type=int, default=0, help="context length for reload (match operator's load)")
    ap.add_argument("--reload-parallel", type=int, default=0, help="parallel count for reload")
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
            if a.reload_every and i > 0 and i % a.reload_every == 0:
                print(f"  [reload] {i}개 처리 후 {model} 재로드(속도 리셋)...", flush=True)
                reload_model(model, a.reload_context, a.reload_parallel)
            try:
                d = make_draft(it, k=a.k, temp=a.temp, max_tokens=a.max_tokens)
                f.write(json.dumps(asdict(d)) + "\n"); f.flush()
            except Exception as e:
                print(f"  [skip {it.item_id}] {type(e).__name__}: {str(e)[:60]}")
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(todo)}", flush=True)
    print(f"[answerer:{a.tag}] done -> {out}  (다음: 각 verifier가 --drafts {out.name} 판정)")


if __name__ == "__main__":
    main()
