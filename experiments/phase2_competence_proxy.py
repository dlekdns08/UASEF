"""
Experiment 2 — deployable verifier competence proxy (gold-free ability gating).

Reviewer attack: "verifier-correct/wrong (Z) uses gold — it's oracle post-hoc analysis; in
deployment you don't know if the verifier was right." We answer with a DEPLOYABLE proxy:
estimate q = P(verifier self-answer correct | verifier's own uncertainty features), using
NO gold at inference. If the cross-model signal is strong where predicted competence q is
high and collapses where q is low, verification is competence-gated in a deployable way.

Per (answerer × verifier) cell:
  * train q = P(Z=1 | features) on calibration split (logistic)
    features: verbalized_confidence, neg_logprob_mean, reasoning_len, parser_ok, self_consistency_disagree
  * on test: split by q tertile (low/mid/high); report verifier lift, P(error|disagree) per tertile
  * interaction: Y ~ p_C + p_V + q + p_V*q  (does q modulate the verifier signal?)

Verifier self-answer features come from: gpt-oss/Qwen3.5 -> their drafts (already have features);
gemma/qwen3.6 -> selfanswer_<v>.jsonl regenerated with features (sessions 4/6). LLM 0. Run:
  python experiments/phase2_competence_proxy.py --cell qwen35N__gptoss \
     --verifier data/raw/verifier_qwen35_of_gptoss.jsonl \
     --v-features data/raw/drafts_qwen35_nothink.jsonl --answerer-drafts data/raw/drafts_phase0_all.jsonl
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.label_conditional_conformal import _auroc
from experiments.phase0_gatekeeper import load_drafts
from sklearn.linear_model import LogisticRegression

sym = lambda s, y: max(_auroc(s, y), 1 - _auroc(s, y))


def _feat_row(r):
    """extract competence features from a selfanswer-with-features row OR a drafts asdict row."""
    lp = r.get("neg_logprob_mean")
    if lp is None and r.get("token_logprobs"):
        lp = -float(np.mean(r["token_logprobs"]))
    conf = r.get("verbalized_confidence")
    z = r.get("self_correct")
    if z is None and r.get("decision_answer") is not None and r.get("gold_answer") is not None:
        z = int((r["decision_answer"] or "").strip().lower() == (r["gold_answer"] or "").strip().lower())
    return {"conf": conf if conf is not None else 0.5,
            "neglp": lp if lp is not None else 0.0,
            "rlen": r.get("reasoning_len", len(r.get("reasoning_text") or "")),
            "parser": r.get("parser_ok", int(bool((r.get("self_answer") or r.get("decision_answer") or "").strip()))),
            "scd": r.get("self_consistency_disagree") if r.get("self_consistency_disagree") is not None else 0.0,
            "Z": z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--verifier", required=True, help="cross-verifier file (verifier_risk, error)")
    ap.add_argument("--v-features", required=True, help="verifier self-answer WITH features (or its drafts)")
    ap.add_argument("--answerer-drafts", required=True)
    a = ap.parse_args()

    ver = {json.loads(l)["item_id"]: json.loads(l) for l in open(a.verifier) if l.strip()}
    vf = {json.loads(l)["item_id"]: _feat_row(json.loads(l)) for l in open(a.v_features) if l.strip()}
    dr = {d.item_id: d for d in load_drafts(a.answerer_drafts)}
    ids = [i for i in ver if ver[i].get("verifier_risk") is not None and i in vf and i in dr and vf[i]["Z"] is not None]
    if len(ids) < 60:
        print(f"[proxy:{a.cell}] 교집합 {len(ids)} < 60 — v-features에 Z/features 있나 확인")
        return
    err = np.array([int(ver[i]["error"]) for i in ids])
    V = np.array([float(ver[i]["verifier_risk"]) for i in ids])
    C = np.array([1 - (dr[i].verbalized_confidence if dr[i].verbalized_confidence is not None else 0.5) for i in ids])
    Z = np.array([vf[i]["Z"] for i in ids])
    F = np.array([[vf[i]["conf"], vf[i]["neglp"], vf[i]["rlen"], vf[i]["parser"], vf[i]["scd"]] for i in ids], float)

    # cross-fitted q on the COMMON item-grouped fold manifest (analysis_plan §5):
    # every q prediction is out-of-fold, and all rows of one canonical item share a fold
    # (no leakage across cells/analyses). Replaces the old ad-hoc 50/50 split.
    from analysis.splits import load_folds
    fmap = load_folds()
    fold = np.array([fmap[i] for i in ids])
    q = np.full(len(ids), np.nan)
    for k in sorted(set(fold)):
        tr, te_k = fold != k, fold == k
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        qm = LogisticRegression(max_iter=1000).fit((F[tr] - mu) / sd, Z[tr])
        q[te_k] = qm.predict_proba((F[te_k] - mu) / sd)[:, 1]
    # proxy quality: out-of-fold q vs Z on ALL items
    q_auroc = round(sym(q, Z), 3)

    # tertiles over all items by cross-fitted (out-of-fold) q
    te = np.arange(len(ids))
    qt = q; e = err; v = V; c = C
    order = np.argsort(qt); n = len(te); t1, t2 = n // 3, 2 * n // 3
    groups = {"low_q": order[:t1], "mid_q": order[t1:t2], "high_q": order[t2:]}
    tert = {}
    for name, g in groups.items():
        if e[g].sum() >= 5 and (e[g] == 0).sum() >= 5:
            lift = round(sym(v[g], e[g]) - sym(c[g], e[g]), 3)
            dis = np.array([int(x) for x in (v[g] > np.median(v))])  # crude disagreement proxy via risk
            tert[name] = {"n": int(len(g)), "q_mean": round(float(qt[g].mean()), 3),
                          "verifier_AUROC": round(sym(v[g], e[g]), 3), "selfconf_AUROC": round(sym(c[g], e[g]), 3),
                          "lift": lift}
        else:
            tert[name] = {"n": int(len(g)), "note": "too few"}
    # interaction: Y ~ pC + pV + q + pV*q  (does q modulate verifier signal?)
    Xi = np.c_[c, v, qt, v * qt]
    lri = LogisticRegression(max_iter=1000).fit(Xi, e)
    coef = dict(zip(["pC", "pV", "q", "pV*q"], [round(float(x), 3) for x in lri.coef_[0]]))

    rep = {"cell": a.cell, "n": len(ids), "q_predicts_Z_AUROC": q_auroc,
           "tertiles_by_predicted_competence": tert,
           "interaction_Y~pC+pV+q+pVxq_coef": coef,
           "reading": ("if lift is larger in high_q than low_q, and the pV*q interaction coef is "
                       "positive, verifier competence is deployably estimable and modulates the "
                       "cross-model signal WITHOUT gold at inference.")}
    outp = ROOT / "results" / "phase2" / f"phase2_competence_proxy_{a.cell}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[proxy:{a.cell}] n={len(ids)} | q→Z AUROC {q_auroc}")
    for k, t in tert.items():
        print(f"  {k}: q~{t.get('q_mean')} lift={t.get('lift')} (V {t.get('verifier_AUROC')} / C {t.get('selfconf_AUROC')})")
    print(f"  interaction pV*q coef = {coef['pV*q']} (양수면 q가 verifier signal 조절)")


if __name__ == "__main__":
    main()
