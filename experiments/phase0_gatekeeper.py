"""
Phase 0 — Gatekeeper (does an error-predicting signal exist at all?).

Before building the Phase-1 conformal gate, confirm that a risk score built from
pre-answer features can predict whether the LLM's own answer is WRONG. We do NOT
build the conformal guarantee here — only measure AUROC(risk, error) by
cross-validation, pooled and per-subject, plus each feature's univariate AUROC.

Go/No-Go (improvements/phase0_3_redesign.md):
  * pooled AUROC >= 0.70          -> GO (proceed to Phase 1 as designed)
  * 0.60 <= AUROC < 0.70          -> GO(conditional): add evidence/verifier features
  * AUROC < 0.60 (after features) -> NO-GO: pivot to an information-boundary negative

Run:
  # data-free smoke (synthetic drafts with a known signal, exercises real features)
  python experiments/phase0_gatekeeper.py --synthetic --n 3000
  # real drafts (produced by models/qa_drafts.py against a backend)
  python experiments/phase0_gatekeeper.py --drafts data/raw/drafts_medmcqa.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.qa_risk_features import DraftRecord, feature_matrix
from models.label_conditional_conformal import _auroc  # reuse the tie-aware AUROC


# ── synthetic drafts (data-free smoke; real feature functions are exercised) ──

def synthetic_drafts(n: int, seed: int = 0, prevalence: float = 0.35,
                     strength: float = 1.0, dataset: str = "medmcqa"):
    """Build DraftRecords whose (samples, logprobs, confidence, hedging) genuinely
    encode error via a latent difficulty s, so the real extractors see signal."""
    rng = np.random.default_rng(seed)
    subjects = [f"subj_{i}" for i in range(21)] if dataset == "medmcqa" else ["pubmedqa"]
    letters = ["A", "B", "C", "D"] if dataset == "medmcqa" else ["yes", "no", "maybe"]
    hedges = ["maybe", "perhaps", "not sure", "uncertain", "possibly", "i think"]
    drafts = []
    for i in range(n):
        err = int(rng.random() < prevalence)
        s = err * strength + rng.normal(0, 0.6)          # latent difficulty
        gold = letters[0]
        decision = gold if not err else letters[rng.integers(1, len(letters))]
        # samples: agreement with `decision` decreases in s
        p_agree = float(np.clip(0.9 - 0.35 * s, 0.25, 0.98))
        samples = [decision if rng.random() < p_agree
                   else letters[rng.integers(0, len(letters))] for _ in range(5)]
        # token logprobs: higher NLL for harder/error cases (90% present)
        logp = None
        if rng.random() < 0.9:
            logp = list(rng.normal(-(0.3 + 0.45 * s), 0.1, size=rng.integers(8, 20)))
        # verbalized confidence: lower for harder cases (70% present)
        conf = None
        if rng.random() < 0.7:
            conf = float(np.clip(0.9 - 0.4 * s + rng.normal(0, 0.05), 0.05, 0.99))
        # reasoning text with hedging proportional to s
        n_h = int(max(0, rng.poisson(max(0.1, 0.6 * s))))
        words = ["the", "patient", "with", "findings", "suggests", "diagnosis"] * 4
        words += rng.choice(hedges, size=n_h).tolist()
        rng.shuffle(words)
        drafts.append(DraftRecord(
            item_id=f"{dataset}_{i}", dataset=dataset,
            subject=str(rng.choice(subjects)),
            decision_answer=decision, samples=samples, token_logprobs=logp,
            verbalized_confidence=conf, reasoning_text=" ".join(words),
            gold_answer=gold))
    return drafts


def load_drafts(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(DraftRecord(**d))
    return out


# ── cross-validated risk scorer ──────────────────────────────────────────────

def cv_risk_auroc(X, y, seed=42, folds=5):
    """Out-of-fold risk scores from LogReg (imputed+scaled) and HistGBDT (native
    NaN); return per-model AUROC(risk, error) and the better model's OOF risk."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = {
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                        StandardScaler(),
                                        LogisticRegression(max_iter=2000)),
        "histgbdt": lambda: HistGradientBoostingClassifier(random_state=0),
    }
    oof = {m: np.full(len(y), np.nan) for m in models}
    for tr, te in skf.split(X, y):
        for m, mk in models.items():
            clf = mk()
            clf.fit(X[tr], y[tr])
            ci = list(clf.classes_).index(1)
            oof[m][te] = clf.predict_proba(X[te])[:, ci]
    aurocs = {m: _auroc(oof[m], y) for m in models}
    best = max(aurocs, key=aurocs.get)
    return aurocs, oof[best], best


def verdict(auroc: float) -> str:
    if auroc >= 0.70:
        return "GO"
    if auroc >= 0.60:
        return "GO_CONDITIONAL (add evidence/verifier features)"
    return "NO_GO (pivot to information-boundary negative)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--drafts", type=str, default=None)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="synthetic signal strength (smoke only)")
    a = ap.parse_args()

    if a.drafts:
        drafts = load_drafts(a.drafts)
        mode = f"real drafts ({a.drafts})"
    elif a.synthetic:
        drafts = (synthetic_drafts(a.n, a.seed, strength=a.strength, dataset="medmcqa")
                  + synthetic_drafts(a.n // 4, a.seed + 1, strength=a.strength, dataset="pubmedqa"))
        mode = f"SYNTHETIC smoke (strength={a.strength})"
    else:
        sys.exit("pass --synthetic or --drafts <cache.jsonl>")

    X, y, names = feature_matrix(drafts)
    aurocs, risk, best = cv_risk_auroc(X, y, seed=a.seed)
    pooled = max(aurocs.values())

    # per-dataset and per-subject AUROC of the best model's OOF risk
    ds = np.array([d.dataset for d in drafts])
    sub = np.array([d.subject for d in drafts])
    by_ds = {str(d): round(_auroc(risk[ds == d], y[ds == d]), 3)
             for d in sorted(set(ds)) if (y[ds == d].sum() and (y[ds == d] == 0).sum())}
    by_sub = {}
    for s in sorted(set(sub)):
        m = sub == s
        if y[m].sum() >= 3 and (y[m] == 0).sum() >= 3:
            by_sub[str(s)] = round(_auroc(risk[m], y[m]), 3)
    # univariate feature AUROC (direction-robust)
    uni = {names[j]: round(max(_auroc(np.nan_to_num(X[:, j], nan=np.nanmedian(X[:, j])), y),
                               1 - _auroc(np.nan_to_num(X[:, j], nan=np.nanmedian(X[:, j])), y)), 3)
           for j in range(X.shape[1])}

    report = {
        "mode": mode, "n": len(drafts), "prevalence": round(float(y.mean()), 4),
        "cv_auroc_by_model": {m: round(v, 3) for m, v in aurocs.items()},
        "pooled_auroc": round(pooled, 3), "best_model": best,
        "auroc_by_dataset": by_ds, "auroc_by_subject_summary": {
            "n_subjects": len(by_sub), "min": min(by_sub.values()) if by_sub else None,
            "median": round(float(np.median(list(by_sub.values()))), 3) if by_sub else None,
            "max": max(by_sub.values()) if by_sub else None},
        "univariate_feature_auroc": uni,
        "go_no_go": verdict(pooled),
    }
    outp = ROOT / "results" / "phase0" / "phase0_gatekeeper"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp) + ".json").write_text(json.dumps(report, indent=2))
    print(f"[Phase 0] {mode} | n={len(drafts)} prevalence={y.mean():.3f}")
    print(f"  CV AUROC(risk,error): {report['cv_auroc_by_model']} -> pooled {pooled:.3f} ({best})")
    print(f"  by dataset: {by_ds}")
    print(f"  by subject: n={len(by_sub)} "
          f"min={report['auroc_by_subject_summary']['min']} "
          f"median={report['auroc_by_subject_summary']['median']} "
          f"max={report['auroc_by_subject_summary']['max']}")
    print(f"  univariate: {uni}")
    print(f"  ==> GO/NO-GO: {report['go_no_go']}")
    print(f"  wrote {outp}.json")


if __name__ == "__main__":
    main()
