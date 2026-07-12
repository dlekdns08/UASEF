"""
Preflight / fingerprint audit — proves every judgment file judged the RIGHT answers.

Born from a real incident: a renamed 294-row cache turned out to have judged a
DEPRECATED draft version (filename said A2; contents said otherwise). Filenames lie;
contents don't. For every judgment file on disk this cross-checks, row by row, against
the answerer source that the manifest says it judged:

  matrix rows    error == recomputed answerer correctness from the source drafts
                 answerer_conf(gpt_oss_conf) == source verbalized_confidence
  shuffle rows   error == 1 - correct_by_text from shuffle_answer_<tag>
                 canonical_answer_text matches the answerer file
  all rows       item_id ⊆ source item set · no duplicates · None counts

Run BEFORE resuming/reusing any cached judgment file and AFTER each task completes:
  python analysis/preflight.py            # audit everything present
  python analysis/preflight.py --strict   # exit 1 on any mismatch (for automation)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from analysis.manifest import describe, MODELS, JUDGMENT

RAW = ROOT / "data" / "raw"

# (answerer_model_id, answerer_mode) -> source drafts file for matrix cells
DRAFTS = {
    (MODELS["gptoss"], "low"): "drafts_phase0_all",
    (MODELS["qwen35"], "T"):   "drafts_qwen35_think",
    (MODELS["qwen35"], "N"):   "drafts_qwen35_nothink",
}


def _norm(s):
    return (s or "").strip().lower()


def _load(stem):
    p = RAW / f"{stem}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def audit_matrix(stem, cond, rows):
    src_stem = DRAFTS.get((cond["answerer_model"], cond["answerer_mode"]))
    if not src_stem:
        return [f"answerer 소스 미정의: {cond['answerer_model']}[{cond['answerer_mode']}]"]
    src = {r["item_id"]: r for r in _load(src_stem) or []}
    probs = []
    not_in = sum(1 for r in rows if r["item_id"] not in src)
    if not_in:
        probs.append(f"소스({src_stem}) 밖 item {not_in}개")
    err_mis = conf_mis = conf_n = 0
    for r in rows:
        d = src.get(r["item_id"])
        if not d:
            continue
        want = int(_norm(d["decision_answer"]) != _norm(d["gold_answer"]))
        if r.get("error") is not None and r["error"] != want:
            err_mis += 1
        c = r.get("answerer_conf", r.get("gpt_oss_conf"))
        if c is not None and d.get("verbalized_confidence") is not None:
            conf_n += 1
            if abs(float(c) - float(d["verbalized_confidence"])) > 1e-9:
                conf_mis += 1
    if err_mis:
        probs.append(f"error 라벨 불일치 {err_mis} (⚠️ 다른 답변 버전 판정 의심)")
    if conf_mis:
        probs.append(f"conf 불일치 {conf_mis}/{conf_n}")
    return probs


def audit_shuffle(stem, cond, rows):
    ans_tag = stem[len("shuffle_judge_"):].split("__", 1)[0]
    src = {r["item_id"]: r for r in _load(f"shuffle_answer_{ans_tag}") or []}
    probs = []
    not_in = sum(1 for r in rows if r["item_id"] not in src)
    if not_in:
        probs.append(f"shuffle_answer_{ans_tag} 밖 item {not_in}개")
    err_mis = txt_mis = 0
    for r in rows:
        s = src.get(r["item_id"])
        if not s:
            continue
        if r.get("error") is not None and r["error"] != 1 - int(s["correct_by_text"]):
            err_mis += 1
        if _norm(r.get("canonical_answer_text")) != _norm(s.get("canonical_answer_text")):
            txt_mis += 1
    if err_mis:
        probs.append(f"error≠1-correct_by_text {err_mis} (⚠️ 다른 답변 판정 의심)")
    if txt_mis:
        probs.append(f"canonical_answer_text 불일치 {txt_mis}")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="불일치 발견 시 exit 1")
    a = ap.parse_args()
    bad = 0
    print(f"{'파일':44} {'행':>5} {'중복':>4} {'None':>5}  판정대상 정합")
    for p in sorted(glob.glob(str(RAW / "*.jsonl"))):
        stem = Path(p).stem
        conds = [c for c in describe(stem) if c["role"] == JUDGMENT]
        if not conds:
            continue
        cond = conds[0]
        rows = _load(stem)
        ids = [r["item_id"] for r in rows]
        dups = sum(1 for _, c in Counter(ids).items() if c > 1)
        nones = sum(1 for r in rows if r.get("verifier_risk") is None)
        probs = (audit_shuffle if stem.startswith("shuffle_judge_") else audit_matrix)(stem, cond, rows)
        if dups:
            probs.insert(0, f"중복 {dups}")
        status = "✅" if not probs else "❌ " + " | ".join(probs)
        if probs:
            bad += 1
        print(f"{stem:44} {len(rows):>5} {dups:>4} {nones:>5}  {status}")
    print(f"\n{'✅ 전 판정 파일이 선언된 answerer 소스와 정합' if bad == 0 else f'❌ 문제 파일 {bad}개 — 즉시 조사 필요'}")
    if a.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
