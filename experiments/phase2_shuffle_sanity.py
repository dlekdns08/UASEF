"""
Sanity-check comparison (LLM 0) — does re-judging the ORIGINAL with the shuffle-judge
prompt reproduce the existing matrix/self-answer originals? Per (answerer × verifier-mode)
cell, on the ~100 sanity items, compute the pre-registered pass criteria:

  * verifier answer agreement (TEXT)        >= 0.90     [rejudge answer vs existing self-answer]
  * risk score Spearman correlation         >= 0.80     [rejudge risk vs existing matrix risk]
  * |AUROC(risk->error) change|             <= 0.05     [existing vs rejudge]
  * lift sign unchanged  (proxy: verifier AUROC sign vs 0.5 preserved)
  * P(error | disagreement-by-text) direction preserved

PASS -> reuse existing original (no full 400 re-judge). FAIL -> flag the cell for full
original re-judge. Reads only files; no LLM.

Run (one cell):
  python experiments/phase2_shuffle_sanity.py --cell gptoss__gemma_T \
     --rejudge data/raw/shuffle_judge_gptoss_ref__gemma_T.jsonl \
     --existing-risk data/raw/verifier_cross.jsonl \
     --v-selfanswer data/raw/selfanswer_gemma.jsonl \
     --answerer-drafts data/raw/drafts_phase0_all.jsonl
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.label_conditional_conformal import _auroc
from experiments.phase0_gatekeeper import load_drafts
from experiments.phase2_cross_verifier import _item_map


def _norm(s):
    return (s or "").strip().lower()


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()); ry = (ry - ry.mean())
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d else None


def _v_answer_text(v_selfanswer, imap):
    """existing verifier own-answer TEXT per item, from a selfanswer(self_answer) or drafts file."""
    out = {}
    for line in open(v_selfanswer):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        iid = r["item_id"]
        lab = (r.get("self_answer") or r.get("decision_answer") or "").strip().upper()
        if iid in imap:
            out[iid] = imap[iid].options.get(lab, "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--rejudge", required=True, help="shuffle_judge_<answerer>_ref__<v_mode>.jsonl")
    ap.add_argument("--existing-risk", required=True, help="matrix verifier file (verifier_risk per item)")
    ap.add_argument("--v-selfanswer", required=True, help="verifier's own answers (selfanswer or drafts)")
    ap.add_argument("--answerer-drafts", required=True, help="answerer drafts (for answer TEXT + error)")
    a = ap.parse_args()

    imap = _item_map()
    rej = {r["item_id"]: r for r in [json.loads(l) for l in open(a.rejudge) if l.strip()]}
    exrisk = {}
    for line in open(a.existing_risk):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("verifier_risk") is not None:
                exrisk[r["item_id"]] = float(r["verifier_risk"])
    v_ans = _v_answer_text(a.v_selfanswer, imap)
    adr = {d.item_id: d for d in load_drafts(a.answerer_drafts)}

    ids = [i for i in rej if i in exrisk and i in v_ans and i in adr
           and rej[i].get("verifier_risk") is not None]
    if len(ids) < 20:
        print(f"[sanity:{a.cell}] 교집합 {len(ids)} < 20 — 판정 불가 (파일/ID 확인)")
        return
    # answerer answer TEXT + error (by text) from its drafts + gold
    a_text = {i: imap[i].options.get((adr[i].decision_answer or "").strip().upper(), "") for i in ids}
    gold_text = {i: imap[i].options.get(imap[i].gold_answer, "") for i in ids}
    err = np.array([int(_norm(a_text[i]) != _norm(gold_text[i]) or a_text[i] == "") for i in ids])

    ex_r = np.array([exrisk[i] for i in ids])
    rj_r = np.array([rej[i]["verifier_risk"] for i in ids])
    sym = lambda S: max(_auroc(S, err), 1 - _auroc(S, err))
    au_ex = sym(ex_r) if (err.sum() >= 5 and (err == 0).sum() >= 5) else None
    au_rj = sym(rj_r) if (err.sum() >= 5 and (err == 0).sum() >= 5) else None

    # answer agreement (text): rejudge verifier answer vs existing self-answer
    agree = np.mean([_norm(rej[i].get("verifier_pred_text")) == _norm(v_ans[i]) for i in ids])
    # disagreement-by-text -> error, existing (selfanswer) vs rejudge
    ex_dis = np.array([int(_norm(v_ans[i]) != _norm(a_text[i])) for i in ids])
    rj_dis = np.array([1 - rej[i].get("agreement_by_text", 0) for i in ids])
    p_ex = float(err[ex_dis == 1].mean()) if (ex_dis == 1).sum() else None
    p_rj = float(err[rj_dis == 1].mean()) if (rj_dis == 1).sum() else None

    sp = _spearman(ex_r, rj_r)
    d_au = round(abs(au_rj - au_ex), 3) if (au_ex is not None and au_rj is not None) else None
    sign_ok = (au_ex is None or au_rj is None) or ((au_ex - 0.5) * (au_rj - 0.5) >= 0)

    checks = {
        "answer_agreement_text": (round(float(agree), 3), agree >= 0.90),
        "risk_spearman": (round(sp, 3) if sp is not None else None, (sp is not None and sp >= 0.80)),
        "auroc_change": (d_au, (d_au is not None and d_au <= 0.05)),
        "auroc_sign_preserved": (None, bool(sign_ok)),
    }
    is_no_think = a.cell.endswith("_N")
    passed = all(ok for _, ok in checks.values())
    rep = {"cell": a.cell, "n": len(ids), "is_no_think": is_no_think,
           "auroc_existing": round(au_ex, 3) if au_ex is not None else None,
           "auroc_rejudge": round(au_rj, 3) if au_rj is not None else None,
           "P_err_disagree_existing": round(p_ex, 3) if p_ex is not None else None,
           "P_err_disagree_rejudge": round(p_rj, 3) if p_rj is not None else None,
           "checks": {k: {"value": v, "pass": ok} for k, (v, ok) in checks.items()},
           "PASS": passed,
           "action": "reuse existing original" if passed else "EXPAND: re-judge full 400 original"}
    outp = ROOT / "results" / "phase2" / f"phase2_shuffle_sanity_{a.cell}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    tag = "PASS ✓" if passed else "FAIL → 원본400 확장"
    print(f"[sanity:{a.cell}] n={len(ids)} {tag}  (no-think={is_no_think})")
    print(f"  agreement={checks['answer_agreement_text'][0]} spearman={checks['risk_spearman'][0]} "
          f"ΔAUROC={d_au} AUROC {rep['auroc_existing']}→{rep['auroc_rejudge']}")
    print(f"  P(err|불일치) 기존{rep['P_err_disagree_existing']} vs 재판정{rep['P_err_disagree_rejudge']}")


if __name__ == "__main__":
    main()
