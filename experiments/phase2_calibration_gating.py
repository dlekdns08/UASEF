"""
Experiment 1 (top priority) — is item-level ABILITY GATING reducible to calibration/sharpness?

Reviewer attack (Kiyani et al.): "the verifier-correct/wrong gap is just because the verifier
risk score is better *calibrated* and *sharper* than self-confidence, not because the verifier
can actually solve the item." We refute this: after calibrating BOTH signals to error
probabilities and controlling calibration/sharpness, does the verifier's item-level ability
Z (=1 if the verifier's own answer is correct) still add explanatory power?

Per (answerer × verifier) cell — CROSS-FITTED on the common item-grouped fold manifest
(analysis/splits.py; analysis_plan §4). Per fold: isotonic calibrators (V->p_V, C->p_C)
and nested logistics fit on the other folds, applied held-out. Every metric is out-of-fold:
  * nested models (Y = answerer error):
      M0: p_C    M1: p_C + p_V    M2: + Z    M3: + Z*disagreement
    -> does Z / Z*D improve HELD-OUT NLL/Brier/AUROC over M1? (item-bootstrap CI, no LRT)
  * ECE/Brier/sharpness reported as CELL-LEVEL DESCRIPTIVE only (never row covariates)

Z uses gold (verifier self-answer correctness) -> MECHANISM DIAGNOSIS, not an operational
feature (Experiment 2 builds a gold-free proxy). LLM 0. Run:
  python experiments/phase2_calibration_gating.py --cell gptoss__gemma \
     --verifier data/raw/verifier_cross.jsonl --v-selfanswer data/raw/selfanswer_gemma.jsonl \
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

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _ece(p, y, bins=10):
    p = np.clip(p, 0, 1); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def _nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_ll(X, y):
    """logistic; return (in-sample NLL, predicted prob)."""
    if X.shape[1] == 0:
        p = np.full(len(y), y.mean())
    else:
        lr = LogisticRegression(max_iter=1000, C=1e6).fit(X, y)
        p = lr.predict_proba(X)[:, 1]
    return _nll(p, y), p


def _norm(s):
    return (s or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--v-selfanswer", required=True)
    ap.add_argument("--answerer-drafts", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    imap = _item_map()
    ver = {json.loads(l)["item_id"]: json.loads(l) for l in open(a.verifier) if l.strip()}
    vsa = {json.loads(l)["item_id"]: json.loads(l) for l in open(a.v_selfanswer) if l.strip()}
    dr = {d.item_id: d for d in load_drafts(a.answerer_drafts)}
    ids = [i for i in ver if ver[i].get("verifier_risk") is not None and i in vsa and i in dr]
    ds = np.array(["pm" if i.startswith("pubmedqa") else "mc" for i in ids])
    Y = np.array([int(ver[i]["error"]) for i in ids])
    V = np.array([float(ver[i]["verifier_risk"]) for i in ids])
    C = np.array([1 - (dr[i].verbalized_confidence if dr[i].verbalized_confidence is not None else 0.5) for i in ids])
    Z = np.array([int(vsa[i]["self_correct"]) for i in ids])
    # disagreement (text): verifier self-answer vs answerer answer
    dis = np.array([int(_norm(vsa[i].get("self_answer")) != _norm(dr[i].decision_answer)) for i in ids])

    # ── cross-fitted design (analysis_plan §4) ──
    # Old 50/50 split + in-sample LRT on the test half retired: LRT df assumed independent
    # rows and the single split wasted half the data. Now: common item-grouped folds; per
    # fold, isotonic calibrators AND nested logistics are fit on the other folds, applied
    # to the held-out fold. Every prediction below is out-of-fold; judgment = held-out
    # NLL/Brier/AUROC improvement with item-level bootstrap CI (not in-sample LRT).
    from analysis.splits import load_folds
    from analysis.stats import grouped_bootstrap_ci
    fmap = load_folds()
    fold = np.array([fmap[i] for i in ids])
    names = ["M0_pC", "M1_pC_pV", "M2_+Z", "M3_+ZxDis"]
    pV, pC = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    preds = {n: np.full(len(ids), np.nan) for n in names}
    for k in sorted(set(fold)):
        tr, te = fold != k, fold == k
        isoV = IsotonicRegression(out_of_bounds="clip").fit(V[tr], Y[tr])
        isoC = IsotonicRegression(out_of_bounds="clip").fit(C[tr], Y[tr])
        pV_tr, pC_tr = isoV.predict(V[tr]), isoC.predict(C[tr])
        pV[te], pC[te] = isoV.predict(V[te]), isoC.predict(C[te])
        Xtr = {"M0_pC": np.c_[pC_tr], "M1_pC_pV": np.c_[pC_tr, pV_tr],
               "M2_+Z": np.c_[pC_tr, pV_tr, Z[tr]],
               "M3_+ZxDis": np.c_[pC_tr, pV_tr, Z[tr], Z[tr] * dis[tr]]}
        Xte = {"M0_pC": np.c_[pC[te]], "M1_pC_pV": np.c_[pC[te], pV[te]],
               "M2_+Z": np.c_[pC[te], pV[te], Z[te]],
               "M3_+ZxDis": np.c_[pC[te], pV[te], Z[te], Z[te] * dis[te]]}
        for n in names:
            lr = LogisticRegression(max_iter=1000, C=1e6).fit(Xtr[n], Y[tr])
            preds[n][te] = lr.predict_proba(Xte[n])[:, 1]

    base = Y.mean()
    sym = lambda s: round(max(_auroc(s, Y), 1 - _auroc(s, Y)), 3)
    res = {n: {"nll": round(_nll(preds[n], Y), 4),
               "brier": round(float(np.mean((preds[n] - Y) ** 2)), 4),
               "auroc": sym(preds[n])} for n in names}

    def d_nll_ci(reduced, full):
        """held-out per-row NLL(reduced)-NLL(full), item-bootstrap CI (>0 = full better)."""
        pr = np.clip(preds[reduced], 1e-6, 1 - 1e-6); pf = np.clip(preds[full], 1e-6, 1 - 1e-6)
        row = -(Y * np.log(pr) + (1 - Y) * np.log(1 - pr)) + (Y * np.log(pf) + (1 - Y) * np.log(1 - pf))
        lo, hi = grouped_bootstrap_ci(lambda rows: float(row[rows].mean()), ids)
        return {"dNLL": round(float(row.mean()), 4), "ci_low": lo, "ci_high": hi,
                "significant": bool(lo > 0)}
    dZ = d_nll_ci("M1_pC_pV", "M2_+Z")
    dZxD = d_nll_ci("M2_+Z", "M3_+ZxDis")

    rep = {
        "cell": a.cell, "n": len(ids), "base_error_rate": round(float(base), 4),
        "design": "cross-fitted on common item-grouped folds; all metrics held-out (analysis_plan §4)",
        "calibration_descriptive_cell_level": {
            "ECE_pV": round(_ece(pV, Y), 4), "ECE_pC": round(_ece(pC, Y), 4),
            "sharpness_pV_var": round(float(np.var(pV)), 4), "sharpness_pC_var": round(float(np.var(pC)), 4),
            "sharpness_pV_absdev": round(float(np.mean(np.abs(pV - base))), 4),
            "AUROC_V": sym(pV), "AUROC_C": sym(pC), "lift_V_over_C": round(sym(pV) - sym(pC), 3),
            "note": "cell-level descriptive only — never entered as row-level covariates"},
        "nested_models_heldout": res,
        "delta_M1_to_M2(add Z)": {**dZ, "dAUROC": round(res["M2_+Z"]["auroc"] - res["M1_pC_pV"]["auroc"], 3),
                                  "dBrier": round(res["M1_pC_pV"]["brier"] - res["M2_+Z"]["brier"], 4)},
        "delta_M2_to_M3(add ZxDisagree)": dZxD,
        "reading": ("if adding Z improves HELD-OUT NLL with item-bootstrap CI excluding 0, the "
                    "verifier signal is NOT reducible to calibration/sharpness -> ability-gating "
                    "defense holds (out-of-fold, no pseudo-replication)."),
    }
    outp = ROOT / "results" / "phase2" / f"phase2_calibration_gating_{a.cell}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    c = rep["calibration_descriptive_cell_level"]; d = rep["delta_M1_to_M2(add Z)"]
    print(f"[cal-gating:{a.cell}] n={len(ids)} base={base:.3f} (cross-fitted, held-out)")
    print(f"  ECE: pV {c['ECE_pV']} pC {c['ECE_pC']} | sharpness(Var) pV {c['sharpness_pV_var']} pC {c['sharpness_pC_var']} | AUROC V {c['AUROC_V']} C {c['AUROC_C']}")
    print(f"  M1(pC+pV) NLL {res['M1_pC_pV']['nll']} → M2(+Z) NLL {res['M2_+Z']['nll']}  ΔNLL {d['dNLL']} CI[{d['ci_low']},{d['ci_high']}] ΔAUROC {d['dAUROC']}")
    print(f"  → {'✅ Z가 calibration/sharpness 통제 후에도 held-out 개선 (ability gating 방어)' if d['significant'] else '⚠ Z 추가효과 CI가 0 포함'}")


if __name__ == "__main__":
    main()
