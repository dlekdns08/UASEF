"""
Phase 1 — Stage-A MVP: label-conditional conformal escalation on a LOCKED test.

Pipeline (improvements/phase0_3_redesign.md §2):
  drafts -> features -> subject-stratified split (train 40 / cal 30 / test 30)
        -> risk scorer trained on TRAIN
        -> LabelConditionalConformal fit on CAL error cases (tau_alpha)
        -> LOCKED-TEST metrics:  P(release|incorrect) <= alpha,
           release rate, over-escalation, accepted-answer error rate
        -> baselines at a MATCHED incorrect-release target (calibrated on CAL),
           to show conformal holds the guarantee on TEST where thresholds drift.

Splits are at the ITEM level, subject-stratified for MedMCQA. All final numbers
are on the locked test; cal/test are never peeked during scorer training.

Run:
  python experiments/phase1_stage_a.py --synthetic --n 4000       # end-to-end validation
  python experiments/phase1_stage_a.py --drafts data/raw/drafts_phase0_all.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.qa_risk_features import feature_matrix
from models.label_conditional_conformal import LabelConditionalConformal, _auroc
from experiments.phase0_gatekeeper import synthetic_drafts, load_drafts


def _subject_split(subjects, seed, fracs=(0.4, 0.3, 0.3)):
    """Item-level split, stratified by subject (each subject's items divided by
    the same fractions). Returns (train_idx, cal_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    idx_by_sub = {}
    for i, s in enumerate(subjects):
        idx_by_sub.setdefault(s, []).append(i)
    tr, ca, te = [], [], []
    for s, idx in idx_by_sub.items():
        idx = list(idx); rng.shuffle(idx)
        n = len(idx); a = int(fracs[0] * n); b = int((fracs[0] + fracs[1]) * n)
        tr += idx[:a]; ca += idx[a:b]; te += idx[b:]
    return np.array(tr), np.array(ca), np.array(te)


def _risk_scorer(Xtr, ytr, Xall, seed=0):
    """Train a risk model on TRAIN, return risk scores for all rows (higher =
    riskier). HistGBDT (native NaN); falls back to prevalence if a class missing."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    if ytr.sum() == 0 or (ytr == 0).sum() == 0:
        return np.full(len(Xall), float(ytr.mean()))
    clf = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    ci = list(clf.classes_).index(1)
    return clf.predict_proba(Xall)[:, ci]


def _metrics(released, err, alpha):
    n = len(err); n_inc = int((err == 1).sum()); n_cor = int((err == 0).sum())
    n_rel = int(released.sum())
    return {
        "released_given_incorrect": round(float((released & (err == 1)).sum() / n_inc), 4) if n_inc else None,
        "release_rate": round(float(n_rel / n), 4),
        "over_escalation": round(float(((~released) & (err == 0)).sum() / n_cor), 4) if n_cor else None,
        "incorrect_given_release": round(float((released & (err == 1)).sum() / n_rel), 4) if n_rel else None,
        "alpha_satisfied": bool(n_inc == 0 or (released & (err == 1)).sum() / n_inc <= alpha),
    }


def _threshold_for_target(risk_cal, err_cal, alpha):
    """Non-conformal baseline: pick tau on CAL so the cal incorrect-release rate
    ~ alpha (release iff risk < tau). Returns tau."""
    err_risk = np.sort(risk_cal[err_cal == 1])
    if len(err_risk) == 0:
        return -np.inf
    k = int(np.floor(alpha * len(err_risk)))   # plain empirical quantile (no +1 correction)
    return err_risk[k - 1] if k >= 1 else -np.inf


def run(drafts, alphas=(0.10, 0.05), seed=0):
    X, y, names = feature_matrix(drafts)
    subj = np.array([d.subject for d in drafts])
    tr, ca, te = _subject_split(subj, seed)
    risk = _risk_scorer(X[tr], y[tr], X, seed=seed)
    scorer_auroc = round(max(_auroc(risk[te], y[te]), 1 - _auroc(risk[te], y[te])), 3)

    out = {"n": len(drafts), "prevalence": round(float(y.mean()), 4),
           "split": {"train": int(len(tr)), "cal": int(len(ca)), "test": int(len(te))},
           "test_error_prevalence": round(float(y[te].mean()), 4),
           "risk_scorer_test_auroc": scorer_auroc, "by_alpha": {}}

    for alpha in alphas:
        gate = LabelConditionalConformal(alpha).fit(risk[ca], y[ca], check_orient=False)
        conf = gate.evaluate(risk[te], y[te])
        # baselines at matched incorrect-release target (calibrated on CAL)
        tau_nc = _threshold_for_target(risk[ca], y[ca], alpha)     # B6 non-conformal
        b6 = _metrics(risk[te] < tau_nc, y[te], alpha)
        b0 = _metrics(np.ones(len(te), bool), y[te], alpha)        # B0 release-all
        out["by_alpha"][str(alpha)] = {
            "conformal_B7": {
                "feasible": gate._fit.feasible, "n_err_cal": gate._fit.n_err_cal,
                "min_n_err": gate._fit.min_n_err, "tau": None if gate._fit.tau == float("-inf") else round(gate._fit.tau, 4),
                "released_given_incorrect": round(conf.released_given_incorrect, 4),
                "release_rate": round(conf.release_rate, 4),
                "over_escalation": round(conf.over_escalation, 4),
                "incorrect_given_release": round(conf.incorrect_given_release, 4) if conf.incorrect_given_release == conf.incorrect_given_release else None,
                "alpha_satisfied": conf.extra["alpha_satisfied"]},
            "nonconformal_B6": b6,
            "release_all_B0": b0,
        }
    # go/no-go: conformal holds alpha on test AND is non-vacuous vs release-all
    a0 = out["by_alpha"][str(alphas[0])]["conformal_B7"]
    out["go_no_go"] = ("GO" if (a0["alpha_satisfied"] and a0["feasible"] and a0["release_rate"] >= 0.40)
                       else ("OVER_ESCALATION_AUDIT" if a0["feasible"]
                             else "INFEASIBLE (need more error cases / raise alpha)"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--drafts", type=str, default=None)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()
    if a.drafts:
        drafts = load_drafts(a.drafts); mode = f"real ({a.drafts})"
    elif a.synthetic:
        drafts = (synthetic_drafts(a.n, a.seed, strength=a.strength, dataset="medmcqa")
                  + synthetic_drafts(a.n // 4, a.seed + 1, strength=a.strength, dataset="pubmedqa"))
        mode = f"SYNTHETIC (strength={a.strength})"
    else:
        sys.exit("pass --synthetic or --drafts <cache.jsonl>")
    rep = run(drafts, seed=a.seed); rep["mode"] = mode
    outp = ROOT / "results" / "phase1" / "phase1_stage_a"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp) + ".json").write_text(json.dumps(rep, indent=2))
    print(f"[Phase 1 Stage-A] {mode} | n={rep['n']} test_err_prev={rep['test_error_prevalence']} "
          f"| risk AUROC(test)={rep['risk_scorer_test_auroc']}")
    for al, b in rep["by_alpha"].items():
        c = b["conformal_B7"]; nc = b["nonconformal_B6"]
        print(f"  alpha={al}: CONFORMAL rel|inc={c['released_given_incorrect']} "
              f"(<= {al}? {c['alpha_satisfied']}) release={c['release_rate']} feasible={c['feasible']} "
              f"|| non-conformal rel|inc={nc['released_given_incorrect']} (satisfied? {nc['alpha_satisfied']})")
    print(f"  ==> {rep['go_no_go']}  (wrote {outp}.json)")


if __name__ == "__main__":
    main()
