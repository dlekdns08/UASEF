"""
Consolidation — turns the per-file raw JSONLs into the mode-aware long tables that
every downstream result reads. Emits (only for files that exist):

  01_items_master.csv
  02_answerer_outputs.csv        role=answerer
  03_verifier_judgments.csv      role=verifier_judgment  (matrix + shuffle)
  04_verifier_self_answers.csv   role=verifier_self_answer
  05_accuracy_summary.csv        answerer acc + verifier self-acc, per (…,dataset)

Every emitted row carries answerer_model, answerer_mode, verifier_model,
verifier_mode, dataset, split — the mandatory keys. Provenance comes from
manifest.describe(filename); dataset comes per-row from item_id. Schema-flexible:
handles drafts, shuffle_answer, matrix-verifier (patched + legacy minimal), shuffle
judge, selfanswer (feature + legacy 3-field). Idempotent; safe to re-run anytime.

Run:  python analysis/consolidate.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from analysis.manifest import describe, dataset_of, ANSWERER, JUDGMENT, SELFANS

RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "consolidated"


def _norm(s):
    return (s or "").strip().lower()


def _rows(stem):
    p = RAW / f"{stem}.jsonl"
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def matrix_item_ids():
    """The 1500 items the matrix actually judges for the gpt-oss answerer (A1).
    Replicates cross_verifier's selection EXACTLY — Random(0).shuffle(drafts_phase0_all)
    then [:1500] — so gpt-oss's accuracy is measured on the judged subset (~1183 mc +
    317 pm), matching the verifiers' self-answer denominators. The 3800-item pool is a
    phase-0 leftover; only these 1500 are in the study. A2 (Qwen3.5) already has exactly
    1500, so only the gpt-oss pool needs this restriction."""
    import random
    from experiments.phase0_gatekeeper import load_drafts
    _all = load_drafts(str(RAW / "drafts_phase0_all.jsonl"))
    random.Random(0).shuffle(_all)
    return set(d.item_id for d in _all[:1500])


def _entropy(samples):
    samples = [s for s in (samples or []) if (s or "").strip()]
    if not samples:
        return None
    n = len(samples)
    counts = defaultdict(int)
    for s in samples:
        counts[_norm(s)] += 1
    return round(-sum((c / n) * math.log2(c / n) for c in counts.values()), 4)


# ── per-role row extractors (schema-flexible) ──
def answerer_row(r, cond):
    shuffle = "canonical_answer_text" in r
    if shuffle:
        ans_canon = r.get("canonical_answer_text"); gold_canon = r.get("gold_answer_text")
        correct = r.get("correct_by_text"); label = r.get("answerer_output_label")
        conf = r.get("confidence"); raw = r.get("raw_output") or ""
    else:  # drafts
        ans_canon = r.get("decision_answer"); gold_canon = r.get("gold_answer")
        correct = int(_norm(ans_canon) == _norm(gold_canon)) if gold_canon is not None else None
        label = r.get("decision_answer"); conf = r.get("verbalized_confidence")
        raw = r.get("reasoning_text") or ""
    conf = float(conf) if conf is not None else None
    return {"answer_label": label, "answer_canonical_id": ans_canon, "gold_canonical_id": gold_canon,
            "answer_correct": int(correct) if correct is not None else None,
            "confidence": conf, "risk_C": (round(1 - conf, 4) if conf is not None else None),
            "parser_success": int(bool((label or ans_canon or "").strip())),
            "empty_output": int(not (raw or label or "").strip()),
            "truncated_output": r.get("truncated"), "output_length": len(raw)}


def judgment_row(r, cond):
    shuffle = "verifier_pred_text" in r or "agreement_by_text" in r
    risk = r.get("verifier_risk")
    err = r.get("error")
    if shuffle:
        v_pred_canon = r.get("verifier_pred_text"); v_pred_label = r.get("verifier_pred_label")
        agree = r.get("agreement_by_text")
        # judge_selected_correct != independent Z: chosen AFTER seeing the answerer's answer.
        # Independent Z lives ONLY in 04_verifier_self_answers (join by item_id; note that
        # joined Z measures competence under the ORIGINAL option order, not shuffle-specific).
        judge_sel = r.get("judge_selected_correct")
        a_conf = None  # answerer conf lives in the shuffle_answer file (join if needed)
        raw = r.get("vtext") or ""
    else:  # matrix
        v_pred_canon = None; v_pred_label = None; agree = None; judge_sel = None
        a_conf = r.get("answerer_conf", r.get("gpt_oss_conf")); raw = r.get("vtext") or ""
    a_conf = float(a_conf) if a_conf is not None else None
    return {"Y_error": err, "answerer_correct": (1 - err if err is not None else None),
            "verifier_risk_V": risk, "risk_C": (round(1 - a_conf, 4) if a_conf is not None else None),
            "verifier_pred_label": v_pred_label, "verifier_pred_canonical_id": v_pred_canon,
            "agreement_canonical": agree,
            "judge_selected_correct": judge_sel,
            "parser_success": r.get("parser_ok", int(risk is not None)),
            "empty_output": r.get("empty", int(not raw.strip())),
            "truncated_output": r.get("truncated"),
            "output_length": r.get("output_length", len(raw))}


def selfanswer_row(r, cond):
    if "self_answer" in r:                       # selfanswer_*.jsonl (generated for this purpose)
        src = "generated"
        canon = r.get("self_answer"); z = r.get("self_correct")
        conf = r.get("verbalized_confidence"); neglp = r.get("neg_logprob_mean")
        rlen = r.get("reasoning_len"); scd = r.get("self_consistency_disagree")
        samples = r.get("samples"); gold_canon = None
    else:                                        # answerer drafts reused (dual-role): same model
        src = "reused_answerer_draft"            # solved same items, no answer shown, same prompt
        canon = r.get("decision_answer"); gold_canon = r.get("gold_answer")
        z = int(_norm(canon) == _norm(gold_canon)) if gold_canon is not None else None
        conf = r.get("verbalized_confidence")
        lp = r.get("token_logprobs")
        neglp = round(-sum(lp) / len(lp), 4) if lp else None
        rlen = len(r.get("reasoning_text") or ""); scd = None; samples = r.get("samples")
    conf = float(conf) if conf is not None else None
    return {"self_answer_source": src,
            "self_answer_canonical_id": canon, "gold_canonical_id": gold_canon,
            "verifier_self_correct_Z": z, "self_confidence": conf,
            "self_neg_logprob_mean": neglp, "self_reasoning_len": rlen,
            "self_consistency_disagreement": scd, "answer_entropy": _entropy(samples),
            "parser_success": int(bool((canon or "").strip()))}


KEYS = ["answerer_model", "answerer_mode", "verifier_model", "verifier_mode", "dataset", "split"]


def collect(role, extractor, restrict=None):
    """walk every raw jsonl whose manifest condition matches `role`, emit long rows.
    restrict = {stem: set(item_id)} limits a file to a subset (e.g. gpt-oss pool ->
    judged 1500) so accuracy denominators stay consistent across roles."""
    rows = []
    for p in sorted(glob.glob(str(RAW / "*.jsonl"))):
        stem = Path(p).stem
        lim = (restrict or {}).get(stem)
        for cond in describe(stem):
            if cond["role"] != role:
                continue
            for r in _rows(stem):
                if lim is not None and r["item_id"] not in lim:
                    continue
                base = {k: cond.get(k) for k in ("answerer_model", "answerer_mode",
                                                 "verifier_model", "verifier_mode", "split",
                                                 "verification_type")}
                # locked key rule (analysis_plan §12): a role absent for this row type is
                # "NA", never blank — and NEVER filled by duplicating the other role's model
                # (a self_answer row with answerer==verifier would masquerade as
                # self_verification). self_answer rows: answerer_* = NA; answerer rows:
                # verifier_* = NA.
                for k in ("answerer_model", "answerer_mode", "verifier_model",
                          "verifier_mode", "verification_type"):
                    if base.get(k) is None:
                        base[k] = "NA"
                base["dataset"] = dataset_of(r["item_id"])
                base["item_id"] = r["item_id"]
                base["source"] = stem
                base.update(extractor(r, cond))
                rows.append(base)
    return rows


def write_csv(name, rows, lead):
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  {name}: (소스 없음, skip)")
        return
    cols = lead + [k for k in rows[0] if k not in lead]
    allcols = list(dict.fromkeys(cols + [k for row in rows for k in row]))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allcols)
        w.writeheader()
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows, {len(allcols)} cols")


def accuracy_summary(ans_rows, self_rows):
    agg = defaultdict(lambda: [0, 0])  # key -> [n, correct]
    for r in ans_rows:
        if r["answer_correct"] is None:
            continue
        k = ("answerer_accuracy", r["answerer_model"], r["answerer_mode"], r["split"], r["dataset"])
        agg[k][0] += 1; agg[k][1] += r["answer_correct"]
    for r in self_rows:
        if r["verifier_self_correct_Z"] is None:
            continue
        k = ("verifier_self_accuracy", r["verifier_model"], r["verifier_mode"], r["split"], r["dataset"])
        agg[k][0] += 1; agg[k][1] += r["verifier_self_correct_Z"]
    out = []
    for (kind, model, mode, split, ds), (n, c) in sorted(agg.items()):
        out.append({"kind": kind, "model": model, "mode": mode, "split": split, "dataset": ds,
                    "n": n, "correct": c, "accuracy": round(c / n, 4) if n else None})
    return out


def items_master(ans_rows):
    seen = {}
    for r in ans_rows:
        iid = r["item_id"]
        if iid not in seen:
            seen[iid] = {"item_id": iid, "dataset": r["dataset"],
                         "gold_canonical_id": r.get("gold_canonical_id")}
    return list(seen.values())


def main():
    print("[consolidate] 읽는 중 (존재하는 raw만)...")
    restrict = {"drafts_phase0_all": matrix_item_ids()}   # gpt-oss pool -> judged 1500
    ans = collect(ANSWERER, answerer_row, restrict)
    jud = collect(JUDGMENT, judgment_row)                 # judgments already carry only judged items
    slf = collect(SELFANS, selfanswer_row, restrict)      # gpt-oss-T dual self-answer -> same 1500
    write_csv("01_items_master.csv", items_master(ans), ["item_id", "dataset"])
    write_csv("02_answerer_outputs.csv", ans, KEYS + ["item_id"])
    write_csv("03_verifier_judgments.csv", jud, KEYS + ["item_id"])
    write_csv("04_verifier_self_answers.csv", slf, KEYS + ["item_id"])
    write_csv("05_accuracy_summary.csv", accuracy_summary(ans, slf),
              ["kind", "model", "mode", "split", "dataset"])
    # coverage report: which (answerer×verifier×mode) judgment cells are present
    cells = sorted({(r["answerer_model"], r["verifier_model"], r["verifier_mode"], r["split"])
                    for r in jud})
    print(f"\n[consolidate] judgment 셀 {len(cells)}개 존재:")
    for a, v, vm, sp in cells:
        av = f"{a.split('/')[-1]} ← {v.split('/')[-1]}-{vm}"
        print(f"    {av:38} [{sp}]")


if __name__ == "__main__":
    main()
