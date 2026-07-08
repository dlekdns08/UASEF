"""
A8 — Information-theoretic anatomy of a cross-model escalation signal (ONE matrix cell).

For a given (answerer a, verifier v) pair we quantify, in BITS, what the verifier's
risk signal actually tells us about the answerer's error Y — and crucially how much
it adds BEYOND the answerer's own self-confidence:

  I(V; Y)                — verifier risk vs error (mutual information)
  I(C; Y)                — answerer self-confidence vs error
  I(V; Y | C)            — *incremental* info the verifier adds beyond self-confidence
                           (the honest measure of "decoupling value")
  H(Y), H(Y|V), H(Y|C)   — residual uncertainty; 1 - H(Y|S)/H(Y) = reduction fraction
  kappa(v_answer, a_answer) + P(Y|agree) vs P(Y|disagree)  — model-agreement view
  I(V;Y|C) stratified by verifier-correct / verifier-wrong  — is the incremental info
                           itself capability-driven? (info-theoretic A7)

Also emits the cell's (Delta, lift) for the A9 capability-gap law:
  Delta = acc(v) - acc(a),  lift = AUROC(V) - AUROC(C).

Entropies use Miller-Madow bias correction; continuous signals are quantile-binned.
No LLM. Run e.g.:
  python experiments/phase2_infotheory.py --tag gptoss__gemma \
     --answerer-drafts data/raw/drafts_phase0_all.jsonl \
     --verifier data/raw/verifier_cross.jsonl \
     --verifier-selfanswer data/raw/selfanswer_gemma.jsonl
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.qa_risk_features import error_label
from models.label_conditional_conformal import _auroc
from experiments.phase0_gatekeeper import load_drafts


def _entropy_mm(counts):
    """Shannon entropy (bits) with Miller-Madow bias correction."""
    counts = np.asarray(counts, float)
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts[counts > 0] / n
    H = -np.sum(p * np.log2(p))
    return H + (len(p) - 1) / (2 * n * np.log(2))  # MM correction in bits


def _binned(x, bins):
    """quantile-bin a continuous signal into integer codes (unique edges)."""
    x = np.asarray(x, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:                       # near-degenerate signal
        return (x > np.median(x)).astype(int)
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


def _mi(Sb, Y):
    """I(Sb; Y) in bits (Y binary), Miller-Madow."""
    Sb = np.asarray(Sb); Y = np.asarray(Y)
    HY = _entropy_mm(np.bincount(Y, minlength=2))
    HS = _entropy_mm(np.bincount(Sb))
    joint = np.array([np.bincount(Sb[Y == y], minlength=Sb.max() + 1) for y in (0, 1)]).ravel()
    HSY = _entropy_mm(joint)
    return max(0.0, HY + HS - HSY)


def _cond_mi(Vb, Y, Cb):
    """I(V; Y | C) = sum_c p(c) I(V;Y|C=c), bits."""
    Vb = np.asarray(Vb); Y = np.asarray(Y); Cb = np.asarray(Cb)
    n = len(Y); tot = 0.0
    for c in np.unique(Cb):
        m = Cb == c
        if m.sum() < 10 or len(np.unique(Y[m])) < 2:
            continue
        tot += (m.sum() / n) * _mi(Vb[m], Y[m])
    return tot


def _Hcond(Sb, Y):
    """H(Y | Sb) bits."""
    Sb = np.asarray(Sb); Y = np.asarray(Y); n = len(Y); h = 0.0
    for s in np.unique(Sb):
        m = Sb == s
        h += (m.sum() / n) * _entropy_mm(np.bincount(Y[m], minlength=2))
    return h


def _norm(s):
    return (s or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="cell tag, e.g. gptoss__gemma")
    ap.add_argument("--answerer-drafts", required=True)
    ap.add_argument("--verifier", required=True, help="cross-verifier jsonl (verifier_risk,error)")
    ap.add_argument("--verifier-selfanswer", required=True, help="verifier self-answers (self_correct)")
    ap.add_argument("--bins", type=int, default=8)
    a = ap.parse_args()

    drafts = {d.item_id: d for d in load_drafts(a.answerer_drafts)}
    ver = {}
    for line in open(a.verifier):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("verifier_risk") is not None:
                ver[r["item_id"]] = r
    vself = {json.loads(l)["item_id"]: json.loads(l)
             for l in open(a.verifier_selfanswer) if l.strip()}

    ids = [i for i in ver if i in drafts and i in vself]
    Y = np.array([error_label(drafts[i]) for i in ids])
    V = np.array([ver[i]["verifier_risk"] for i in ids])
    # answerer self-confidence signal = 1 - verbalized_confidence (higher = riskier), matches A0
    C = np.array([1 - (drafts[i].verbalized_confidence if drafts[i].verbalized_confidence is not None else 0.5)
                  for i in ids])
    vcorrect = np.array([vself[i]["self_correct"] for i in ids])

    Vb, Cb = _binned(V, a.bins), _binned(C, a.bins)
    HY = _entropy_mm(np.bincount(Y, minlength=2))
    I_V, I_C = _mi(Vb, Y), _mi(Cb, Y)
    I_V_given_C = _cond_mi(Vb, Y, Cb)
    HY_V, HY_C = _Hcond(Vb, Y), _Hcond(Cb, Y)
    sym = lambda S: round(max(_auroc(S, Y), 1 - _auroc(S, Y)), 3)

    # kappa(verifier's own answer, answerer's answer) + error vs (dis)agreement
    a_ans = np.array([_norm(drafts[i].decision_answer) for i in ids])
    v_ans = np.array([_norm(vself[i].get("self_answer")) for i in ids])
    agree = (a_ans == v_ans)
    po = agree.mean()
    # chance agreement over the answer alphabet
    labs = set(a_ans) | set(v_ans)
    pe = sum((a_ans == L).mean() * (v_ans == L).mean() for L in labs)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    # incremental info stratified by verifier competence (info-theoretic A7)
    strat = {}
    for name, msk in [("verifier_correct", vcorrect == 1), ("verifier_wrong", vcorrect == 0)]:
        if msk.sum() >= 40 and len(np.unique(Y[msk])) == 2:
            strat[name] = {"n": int(msk.sum()),
                           "I_V_bits": round(_mi(_binned(V[msk], a.bins), Y[msk]), 4),
                           "auroc_V": round(max(_auroc(V[msk], Y[msk]), 1 - _auroc(V[msk], Y[msk])), 3)}

    acc_a = round(1 - float(Y.mean()), 4)
    acc_v = round(float(vcorrect.mean()), 4)
    rep = {
        "tag": a.tag, "n": len(ids), "H_Y_bits": round(HY, 4), "error_prevalence": round(float(Y.mean()), 4),
        "I_verifier_Y_bits": round(I_V, 4), "I_selfconf_Y_bits": round(I_C, 4),
        "I_verifier_Y_given_selfconf_bits": round(I_V_given_C, 4),
        "incremental_over_selfconf_frac": round(I_V_given_C / I_V, 3) if I_V > 0 else None,
        "uncert_reduction_verifier": round(1 - HY_V / HY, 3) if HY > 0 else None,
        "uncert_reduction_selfconf": round(1 - HY_C / HY, 3) if HY > 0 else None,
        "auroc_verifier": sym(V), "auroc_selfconf": sym(C),
        "kappa_vAnswer_aAnswer": round(kappa, 3),
        "P_err_given_agree": round(float(Y[agree].mean()), 3) if agree.sum() else None,
        "P_err_given_disagree": round(float(Y[~agree].mean()), 3) if (~agree).sum() else None,
        "acc_answerer": acc_a, "acc_verifier": acc_v,
        "Delta_cap_gap": round(acc_v - acc_a, 4),
        "lift_auroc": round(sym(V) - sym(C), 3),
        "stratified_by_verifier_competence": strat,
    }
    outp = ROOT / "results" / "phase2" / f"phase2_infotheory_{a.tag}"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp) + ".json").write_text(json.dumps(rep, indent=2))
    print(f"[A8:{a.tag}] n={len(ids)} H(Y)={HY:.3f}b | I(V;Y)={I_V:.3f}b I(C;Y)={I_C:.3f}b "
          f"I(V;Y|C)={I_V_given_C:.3f}b (증분 {rep['incremental_over_selfconf_frac']})")
    print(f"  Δ={rep['Delta_cap_gap']:+.3f} lift={rep['lift_auroc']:+.3f} | κ={rep['kappa_vAnswer_aAnswer']} "
          f"P(err|agree)={rep['P_err_given_agree']} P(err|disagree)={rep['P_err_given_disagree']}")
    if strat:
        print(f"  증분정보 층화: " + " ".join(f"{k}=I(V;Y){v['I_V_bits']}b(AUROC{v['auroc_V']})" for k, v in strat.items()))
    print(f"  wrote {outp}.json")


if __name__ == "__main__":
    main()
