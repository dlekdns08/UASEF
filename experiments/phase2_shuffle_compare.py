"""
Dual-shuffle comparison (LLM 0, Option B) — ORIGINAL vs SHUFFLED, clean paired.

Both sides are judged by the SAME verifier with the SAME shuffle-judge prompt (no prompt
confound): original = shuffle_judge_<a>_ref__<v>.jsonl, shuffled = shuffle_judge_<a>__<v>.jsonl.
Paired by item_id. Reports whether the verifier's signal SURVIVES option shuffling
(content-based) or COLLAPSES (letter-position memorization):

  AUROC(verifier_risk -> error)                 original vs shuffled  (+Δ)
  AUROC(disagreement-by-text -> error)          original vs shuffled  (+Δ)
  P(error | disagree-by-text)                    original vs shuffled
  verifier content-invariance                    same verifier answer TEXT on orig vs shuffled?

Reading: think verifiers should be ROBUST (small Δ); if a no-think verifier COLLAPSES under
shuffle (large Δ toward 0.5) it was riding memorized answer positions -> the mechanism behind
"no-think verifier is worthless" (§ reasoning ablation).

Run:
  python experiments/phase2_shuffle_compare.py --cell gptoss__gem_T \
     --original data/raw/shuffle_judge_gptoss_ref__gem_T.jsonl \
     --shuffled data/raw/shuffle_judge_gptoss__gem_T.jsonl
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.label_conditional_conformal import _auroc

sym = lambda A, B: round(max(_auroc(A, B), 1 - _auroc(A, B)), 3)


def _norm(s):
    return (s or "").strip().lower()


def _auroc_pair(risk, err):
    risk, err = np.asarray(risk, float), np.asarray(err)
    if err.sum() < 5 or (err == 0).sum() < 5:
        return None
    return sym(risk, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="e.g. gptoss__gem_T")
    ap.add_argument("--original", required=True, help="shuffle_judge_<a>_ref__<v>.jsonl")
    ap.add_argument("--shuffled", required=True, help="shuffle_judge_<a>__<v>.jsonl")
    a = ap.parse_args()

    orig = {r["item_id"]: r for r in [json.loads(l) for l in open(a.original) if l.strip()]}
    shuf = {r["item_id"]: r for r in [json.loads(l) for l in open(a.shuffled) if l.strip()]}
    ids = sorted(set(orig) & set(shuf))
    if len(ids) < 20:
        print(f"[compare:{a.cell}] 교집합 {len(ids)} < 20 — 파일 확인")
        return

    def block(store):
        risk = np.array([store[i]["verifier_risk"] if store[i]["verifier_risk"] is not None else np.nan for i in ids])
        risk = np.nan_to_num(risk, nan=float(np.nanmedian(risk)) if np.isfinite(np.nanmedian(risk)) else 0.5)
        err = np.array([store[i]["error"] for i in ids])
        dis = np.array([1 - store[i].get("agreement_by_text", 0) for i in ids])
        return risk, err, dis

    r_o, err, dis_o = block(orig)
    r_s, err_s, dis_s = block(shuf)
    au_risk_o, au_risk_s = _auroc_pair(r_o, err), _auroc_pair(r_s, err_s)
    au_dis_o, au_dis_s = _auroc_pair(dis_o, err), _auroc_pair(dis_s, err_s)
    p_o = round(float(err[dis_o == 1].mean()), 3) if (dis_o == 1).sum() else None
    p_s = round(float(err_s[dis_s == 1].mean()), 3) if (dis_s == 1).sum() else None
    # verifier content-invariance: same verifier answer TEXT on orig vs shuffled
    inv = np.mean([_norm(orig[i].get("verifier_pred_text")) == _norm(shuf[i].get("verifier_pred_text"))
                   for i in ids])
    d_risk = round(abs(au_risk_s - au_risk_o), 3) if (au_risk_o is not None and au_risk_s is not None) else None
    d_dis = round(abs(au_dis_s - au_dis_o), 3) if (au_dis_o is not None and au_dis_s is not None) else None
    is_no_think = a.cell.endswith("_N")
    survives = ((d_risk is None or d_risk <= 0.05) and (d_dis is None or d_dis <= 0.05))
    rep = {"cell": a.cell, "n": len(ids), "is_no_think": is_no_think,
           "auroc_risk": {"original": au_risk_o, "shuffled": au_risk_s, "delta": d_risk},
           "auroc_disagree_text": {"original": au_dis_o, "shuffled": au_dis_s, "delta": d_dis},
           "P_err_given_disagree": {"original": p_o, "shuffled": p_s},
           "verifier_content_invariance": round(float(inv), 3),
           "verdict": ("ROBUST (content-based)" if survives else "COLLAPSE (position/memorization)"),
           "reading": "small Δ = signal survives shuffle (content-based); large Δ toward 0.5 = memorized positions."}
    outp = ROOT / "results" / "phase2" / f"phase2_shuffle_compare_{a.cell}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[compare:{a.cell}] n={len(ids)} no-think={is_no_think} → {rep['verdict']}")
    print(f"  AUROC(risk): 원본 {au_risk_o} → 셔플 {au_risk_s} (Δ{d_risk})")
    print(f"  AUROC(불일치-text): 원본 {au_dis_o} → 셔플 {au_dis_s} (Δ{d_dis})")
    print(f"  P(오류|불일치): 원본 {p_o} → 셔플 {p_s} | verifier 답 content 불변율 {rep['verifier_content_invariance']}")


if __name__ == "__main__":
    main()
