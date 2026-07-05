"""
Phase 2 — Audit the Stage-A QA escalation gate.

Ports the five diagnostic detectors (models/audit_detectors.py) to the QA risk
score, and adds the QA-specific checks the plan calls for:

  * orientation        — AUROC(risk, error) > 0.5 ?
  * escalate-all       — does the gate escalate ~everything at alpha ?
  * definitional leak  — any single risk FEATURE near-perfectly predicts error ?
  * confidence-dominance (informative-missingness analog) — does the model's own
    verbalized confidence alone recover most of the gate's above-chance signal?
    (the QA version of "the signal is which-labs-ordered, not the values":
    here, "the signal is the model's self-reported confidence, not the content")
  * per-subject confidence AUROC stability.

This runs on the EXISTING drafts (no new generation). The stronger
option-shuffle / cross-model-verifier audits regenerate drafts and live in
experiments/phase2_shuffle_audit.py and experiments/phase2_cross_verifier.py.

Run: python experiments/phase2_qa_audit.py --drafts data/raw/drafts_phase0_all.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.qa_risk_features import feature_matrix, MINIMAL_FEATURES
from models.label_conditional_conformal import _auroc, LabelConditionalConformal
from models.audit_detectors import (OrientationDetector, EscalateAllDetector,
                                    DefinitionalLeakageDetector, InformativeMissingnessDetector)
from experiments.phase0_gatekeeper import load_drafts, cv_risk_auroc


def _oof_risk(X, y, cols=None, seed=42):
    """Out-of-fold risk over the given feature columns (all if None)."""
    Xc = X if cols is None else X[:, cols]
    _, oof, _ = cv_risk_auroc(Xc, y, seed=seed)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    a = ap.parse_args()
    drafts = load_drafts(a.drafts)
    X, y, names = feature_matrix(drafts)
    subj = np.array([d.subject for d in drafts])

    # full-feature OOF risk (higher = riskier)
    risk = _oof_risk(X, y)
    au_full = max(_auroc(risk, y), 1 - _auroc(risk, y))

    # (1) orientation
    orient = OrientationDetector().detect(risk, y)

    # (2) escalate-all: fit gate on a 70/30 split, measure escalation rate on holdout
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); cut = int(0.5 * len(y))
    cal, te = idx[:cut], idx[cut:]
    gate = LabelConditionalConformal(a.alpha).fit(risk[cal], y[cal], check_orient=False)
    released = gate.predict(risk[te]); esc_rate = float((~released).mean())
    escall = EscalateAllDetector().detect(esc_rate)

    # (3) definitional leakage on the risk features
    defn = DefinitionalLeakageDetector(0.90).detect(np.nan_to_num(X, nan=0.0), y, names=names)

    # (4) confidence-dominance (informative-missingness analog): confidence-only vs full
    ci = names.index("verbalized_uncertainty")
    risk_conf = _oof_risk(X, y, cols=[ci])
    other = [j for j in range(len(names)) if j != ci]
    risk_noconf = _oof_risk(X, y, cols=other)
    au_conf = max(_auroc(risk_conf, y), 1 - _auroc(risk_conf, y))
    au_noconf = max(_auroc(risk_noconf, y), 1 - _auroc(risk_noconf, y))
    conf_dom = InformativeMissingnessDetector(0.85).detect(au_full, au_noconf, au_conf)

    # (5) per-subject confidence-feature AUROC stability
    conf_feat = np.nan_to_num(X[:, ci], nan=np.nanmedian(X[:, ci]))
    by_sub = {}
    for s in sorted(set(subj)):
        m = subj == s
        if y[m].sum() >= 5 and (y[m] == 0).sum() >= 5:
            by_sub[str(s)] = round(max(_auroc(conf_feat[m], y[m]), 1 - _auroc(conf_feat[m], y[m])), 3)

    report = {
        "n": len(drafts), "prevalence": round(float(y.mean()), 4), "alpha": a.alpha,
        "risk_auroc_full": round(au_full, 3),
        "detectors": {
            "orientation": {"flagged": orient.flagged, "auroc": round(orient.statistic, 3)},
            "escalate_all": {"flagged": escall.flagged, "escalation_rate": round(esc_rate, 3)},
            "definitional_leakage": {"flagged": defn.flagged, "max_univariate_auroc": round(defn.statistic, 3),
                                     "detail": defn.detail[:160]},
            "confidence_dominance": {"flagged": conf_dom.flagged, "recover": round(conf_dom.statistic, 3),
                                     "auroc_full": round(au_full, 3), "auroc_confidence_only": round(au_conf, 3),
                                     "auroc_without_confidence": round(au_noconf, 3)},
        },
        "univariate_feature_auroc": {n: round(max(_auroc(np.nan_to_num(X[:, j], nan=np.nanmedian(X[:, j])), y),
                                                  1 - _auroc(np.nan_to_num(X[:, j], nan=np.nanmedian(X[:, j])), y)), 3)
                                     for j, n in enumerate(names)},
        "confidence_auroc_by_subject": {"n": len(by_sub),
                                        "min": min(by_sub.values()) if by_sub else None,
                                        "median": round(float(np.median(list(by_sub.values()))), 3) if by_sub else None,
                                        "max": max(by_sub.values()) if by_sub else None},
    }
    flags = [k for k, v in report["detectors"].items() if v["flagged"]]
    report["verdict"] = "CLEAN" if not flags else f"FLAGGED: {flags}"
    outp = ROOT / "results" / "phase2" / "phase2_qa_audit"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp) + ".json").write_text(json.dumps(report, indent=2))
    print(f"[Phase 2 audit] n={len(drafts)} risk AUROC(full)={au_full:.3f}")
    for k, v in report["detectors"].items():
        print(f"  {k:22s} flagged={v['flagged']}  {({kk:vv for kk,vv in v.items() if kk!='flagged'})}")
    print(f"  confidence AUROC by subject: {report['confidence_auroc_by_subject']}")
    print(f"  ==> {report['verdict']}   (wrote {outp}.json)")


if __name__ == "__main__":
    main()
