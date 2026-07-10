"""
Run-provenance capture — one JSONL record per swap session (analysis_plan §12).

The T/N distinction is a MANUAL Jinja toggle inside LM Studio's model config, invisible
to the API, so the operator must declare it via --mode; everything else is captured
automatically. Prompts (VSYS/VJSYS) are git-versioned code, so the recorded git commit
doubles as the prompt hash. SHA-256 of a 50GB+ model file takes minutes — opt-in via
--hash.

Run at the START of each session, right after loading the model:
  python analysis/run_provenance.py --session 4 --mode T [--note "gemma-T session"] [--hash]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
LMS = os.path.expanduser("~/.lmstudio/bin/lms")
OUT = ROOT / "results" / "consolidated" / "run_provenance.jsonl"


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<error {type(e).__name__}>"


def _loaded_model():
    """parse `lms ps` -> (identifier, context, parallel)."""
    txt = _cmd([LMS, "ps"])
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0] not in ("IDENTIFIER",) and not line.startswith("-"):
            # IDENTIFIER MODEL STATUS SIZE(2 tokens: "54.30 GB") CONTEXT PARALLEL ...
            try:
                return {"identifier": parts[0], "raw_ps_line": line.strip()}
            except Exception:
                pass
    return {"identifier": None, "raw_ps_line": txt[:300]}


def _model_file(identifier):
    """locate the local model file(s) under ~/.lmstudio/models for size/mtime (+optional hash)."""
    if not identifier:
        return None
    base = Path(os.path.expanduser("~/.lmstudio/models"))
    hits = []
    if base.exists():
        needle = identifier.lower().replace("/", os.sep)
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".gguf", ".safetensors", ".bin") \
                    and needle.split(os.sep)[-1][:12] in str(p).lower():
                st = p.stat()
                hits.append({"path": str(p), "bytes": st.st_size,
                             "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()})
    return hits or None


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="swap session number (1-9)")
    ap.add_argument("--mode", required=True, choices=["T", "N"],
                    help="thinking toggle state of the LOADED model (manual Jinja toggle — declare it)")
    ap.add_argument("--note", default="")
    ap.add_argument("--hash", action="store_true", help="also SHA-256 the model file(s) (slow: minutes for 50GB+)")
    a = ap.parse_args()

    loaded = _loaded_model()
    files = _model_file(loaded.get("identifier"))
    if a.hash and files:
        for f in files:
            f["sha256"] = _sha256(f["path"])

    rec = {
        "run_id": f"s{a.session}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "session": a.session,
        "declared_mode": a.mode,
        "loaded_model": loaded,
        "model_files": files,
        "lms_version": _cmd([LMS, "version"]),
        "git_commit": _cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "env": {"UASEF_QUERY_TIMEOUT_S": os.environ.get("UASEF_QUERY_TIMEOUT_S"),
                "UASEF_BACKEND_NEVER_SEND_PHI": os.environ.get("UASEF_BACKEND_NEVER_SEND_PHI")},
        "note": a.note,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[provenance] {rec['run_id']} mode={a.mode} model={loaded.get('identifier')} -> {OUT}")


if __name__ == "__main__":
    main()
