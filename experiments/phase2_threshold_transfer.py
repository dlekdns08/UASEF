"""
Experiment 3 — threshold / distribution-shift transfer robustness.

Reviewer attack (Bakman et al.): threshold-free AUROC is not enough; in deployment a
release gate uses a THRESHOLD, and thresholds transfer poorly under distribution shift.
We pick a release threshold τ on a CALIBRATION split (targeting a fixed release rate) and
apply it to shifted TEST splits, measuring how well the guarantee/operating point holds:

  transfers: within-dataset · cross-dataset (Med↔Pub) · cross-answerer · original→shuffled
  scores compared: C (self-conf) · V (verifier risk) · C+V (logistic) [· +q competence, Exp2]
  metrics: Pr(release | error), Pr(error | release), release rate, τ, τ-instability across splits

Reading: if C+V holds Pr(release|error) closer to target under shift and its τ is more stable
than C alone, the cross-model signal gives a more transferable operating point — but cross-
dataset/answerer degradation quantifies where the conformal exchangeability assumption is needed.

LLM 0. Run (cross-dataset on one cell):
  python experiments/phase2_threshold_transfer.py --tag gptoss__gemma \
     --verifier data/raw/verifier_cross.jsonl --answerer-drafts data/raw/drafts_phase0_all.jsonl \
     --target-release 0.6
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from experiments.phase0_gatekeeper import load_drafts
from sklearn.linear_model import LogisticRegression


def _tau_for_release(score, target):
    """release iff score < τ (low risk); τ = target-release quantile of score."""
    return float(np.quantile(score, target))


def _eval(score, err, tau):
    rel = score < tau
    return {"release_rate": round(float(rel.mean()), 4),
            "Pr_release_given_error": round(float(rel[err == 1].mean()), 4) if (err == 1).sum() else None,
            "Pr_error_given_release": round(float(err[rel].mean()), 4) if rel.sum() else None,
            "tau": round(tau, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--answerer-drafts", required=True)
    ap.add_argument("--target-release", type=float, default=0.6)
    a = ap.parse_args()

    ver = {json.loads(l)["item_id"]: json.loads(l) for l in open(a.verifier) if l.strip()}
    dr = {d.item_id: d for d in load_drafts(a.answerer_drafts)}
    ids = [i for i in ver if ver[i].get("verifier_risk") is not None and i in dr]
    ds = np.array(["pm" if i.startswith("pubmedqa") else "mc" for i in ids])
    err = np.array([int(ver[i]["error"]) for i in ids])
    V = np.array([float(ver[i]["verifier_risk"]) for i in ids])
    C = np.array([1 - (dr[i].verbalized_confidence if dr[i].verbalized_confidence is not None else 0.5) for i in ids])

    def cv_score(mask_fit):
        """C+V combined = P(error) from logistic fit on the given split."""
        X = np.c_[C, V]
        lr = LogisticRegression(max_iter=1000).fit(X[mask_fit], err[mask_fit])
        return lr.predict_proba(X)[:, 1]
    scores = {"C": C, "V": V}

    def transfer(fit_mask, test_mask, label):
        out = {}
        cv = cv_score(fit_mask)
        allscores = {**scores, "C+V": cv}
        for name, s in allscores.items():
            tau = _tau_for_release(s[fit_mask], a.target_release)
            out[name] = {"fit": _eval(s[fit_mask], err[fit_mask], tau),
                         "test": _eval(s[test_mask], err[test_mask], tau)}
        return {"label": label, "n_fit": int(fit_mask.sum()), "n_test": int(test_mask.sum()), "scores": out}

    mc, pm = ds == "mc", ds == "pm"
    rep = {"cell": a.tag, "target_release": a.target_release, "transfers": {}}
    # within-dataset (fit and test on same ds, 50/50)
    rng = np.random.default_rng(0)
    for D, m in [("mc", mc), ("pm", pm)]:
        idx = np.where(m)[0]; rng.shuffle(idx); h = len(idx) // 2
        fit = np.zeros(len(ids), bool); fit[idx[:h]] = True
        tst = np.zeros(len(ids), bool); tst[idx[h:]] = True
        if fit.sum() >= 30 and tst.sum() >= 30:
            rep["transfers"][f"within_{D}"] = transfer(fit, tst, f"within-{D}")
    # cross-dataset
    if mc.sum() >= 30 and pm.sum() >= 30:
        rep["transfers"]["mc_to_pm"] = transfer(mc, pm, "MedMCQA→PubMedQA")
        rep["transfers"]["pm_to_mc"] = transfer(pm, mc, "PubMedQA→MedMCQA")

    # τ-instability: spread of Pr(release|error) on test across transfers, per score
    inst = {}
    for name in ["C", "V", "C+V"]:
        vals = [t["scores"][name]["test"]["Pr_release_given_error"] for t in rep["transfers"].values()
                if t["scores"][name]["test"]["Pr_release_given_error"] is not None]
        inst[name] = {"Pr_rel_given_err_range": round(max(vals) - min(vals), 4) if len(vals) > 1 else None,
                      "values": [round(v, 3) for v in vals]}
    rep["tau_instability_Pr_release_given_error"] = inst
    rep["reading"] = ("smaller Pr(release|error) spread across transfers = more stable operating point; "
                      "cross-dataset degradation shows where conformal exchangeability is required.")
    outp = ROOT / "results" / "phase2" / f"phase2_threshold_transfer_{a.tag}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[threshold-transfer:{a.tag}] target release={a.target_release}")
    for k, t in rep["transfers"].items():
        s = t["scores"]
        print(f"  {t['label']:20} Pr(rel|err) test: C={s['C']['test']['Pr_release_given_error']} "
              f"V={s['V']['test']['Pr_release_given_error']} C+V={s['C+V']['test']['Pr_release_given_error']}")
    print(f"  τ-instability(Pr_rel|err range): C={inst['C']['Pr_rel_given_err_range']} "
          f"V={inst['V']['Pr_rel_given_err_range']} C+V={inst['C+V']['Pr_rel_given_err_range']} (작을수록 안정)")


if __name__ == "__main__":
    main()
