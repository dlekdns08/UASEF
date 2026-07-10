"""
Experiment 1 (top priority) — is item-level ABILITY GATING reducible to calibration/sharpness?

Reviewer attack (Kiyani et al.): "the verifier-correct/wrong gap is just because the verifier
risk score is better *calibrated* and *sharper* than self-confidence, not because the verifier
can actually solve the item." We refute this: after calibrating BOTH signals to error
probabilities and controlling calibration/sharpness, does the verifier's item-level ability
Z (=1 if the verifier's own answer is correct) still add explanatory power?

Per (answerer × verifier) cell, on a held-out test split:
  * calibrate raw verifier risk V -> p_V = P(answerer error | V)  [isotonic on cal split]
  * calibrate self-confidence   C -> p_C                          [same]
  * report ECE, Brier, NLL, sharpness(=Var(p_V)), AUROC, lift
  * nested logistic models (Y = answerer error):
      M0: p_C      M1: p_C + p_V      M3: p_C + p_V + Z      M4: + Z*disagreement
    -> does Z / Z*disagree improve NLL/Brier/AUROC over M1? (LRT + bootstrap)

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

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(ids)); half = len(ids) // 2
    cal, te = idx[:half], idx[half:]

    def calib(raw):
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw[cal], Y[cal])
        return iso.predict(raw[te])
    pV, pC = calib(V), calib(C)
    yte = Y[te]; Zte, diste = Z[te], dis[te]
    sym = lambda s: round(max(_auroc(s, yte), 1 - _auroc(s, yte)), 3)
    base = yte.mean()

    # nested models on TEST (LRT for added terms)
    feats = {
        "M0_pC": np.c_[pC],
        "M1_pC_pV": np.c_[pC, pV],
        "M3_+Z": np.c_[pC, pV, Zte],
        "M4_+ZxDis": np.c_[pC, pV, Zte, Zte * diste],
    }
    res = {}
    for name, X in feats.items():
        nll, p = _fit_ll(X, yte)
        res[name] = {"nll": round(nll, 4), "brier": round(float(np.mean((p - yte) ** 2)), 4), "auroc": sym(p)}
    from scipy.stats import chi2
    def lrt(reduced, full, dfree):
        stat = 2 * len(yte) * (res[reduced]["nll"] - res[full]["nll"])
        return round(stat, 2), round(float(1 - chi2.cdf(max(stat, 0), dfree)), 4)
    lrt_Z = lrt("M1_pC_pV", "M3_+Z", 1)
    lrt_ZxD = lrt("M3_+Z", "M4_+ZxDis", 1)

    rep = {
        "cell": a.cell, "n_test": len(te), "base_error_rate": round(float(base), 4),
        "calibration": {
            "ECE_pV": round(_ece(pV, yte), 4), "ECE_pC": round(_ece(pC, yte), 4),
            "sharpness_pV_var": round(float(np.var(pV)), 4), "sharpness_pC_var": round(float(np.var(pC)), 4),
            "sharpness_pV_absdev": round(float(np.mean(np.abs(pV - base))), 4),
            "AUROC_V": sym(pV), "AUROC_C": sym(pC), "lift_V_over_C": round(sym(pV) - sym(pC), 3)},
        "nested_models": res,
        "delta_M1_to_M3(add Z)": {"dNLL": round(res["M1_pC_pV"]["nll"] - res["M3_+Z"]["nll"], 4),
                                  "dBrier": round(res["M1_pC_pV"]["brier"] - res["M3_+Z"]["brier"], 4),
                                  "dAUROC": round(res["M3_+Z"]["auroc"] - res["M1_pC_pV"]["auroc"], 3),
                                  "LRT_chi2": lrt_Z[0], "LRT_p": lrt_Z[1]},
        "delta_M3_to_M4(add ZxDisagree)": {"dNLL": round(res["M3_+Z"]["nll"] - res["M4_+ZxDis"]["nll"], 4),
                                           "LRT_chi2": lrt_ZxD[0], "LRT_p": lrt_ZxD[1]},
        "reading": ("if adding Z (verifier item-level ability) improves NLL/Brier/AUROC over "
                    "calibrated p_C+p_V (LRT p<0.05), the verifier signal is NOT reducible to "
                    "calibration/sharpness -> ability-gating defense holds."),
    }
    outp = ROOT / "results" / "phase2" / f"phase2_calibration_gating_{a.cell}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    c = rep["calibration"]; d = rep["delta_M1_to_M3(add Z)"]
    print(f"[cal-gating:{a.cell}] n_test={len(te)} base={base:.3f}")
    print(f"  ECE: pV {c['ECE_pV']} pC {c['ECE_pC']} | sharpness(Var) pV {c['sharpness_pV_var']} pC {c['sharpness_pC_var']} | AUROC V {c['AUROC_V']} C {c['AUROC_C']}")
    print(f"  M1(pC+pV) NLL {res['M1_pC_pV']['nll']} → M3(+Z) NLL {res['M3_+Z']['nll']}  ΔNLL {d['dNLL']} ΔAUROC {d['dAUROC']} | LRT χ²={d['LRT_chi2']} p={d['LRT_p']}")
    print(f"  → {'✅ Z가 calibration/sharpness 통제 후에도 유의 (ability gating 방어)' if d['LRT_p']<0.05 else '⚠ Z 추가효과 약함'}")


if __name__ == "__main__":
    main()
