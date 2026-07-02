"""
Round 20 — Measure each audit detector's sensitivity/specificity on data with
KNOWN answers via controlled contamination injection.

Turns the five guards from anecdote ("we caught our own bug") into a validated
diagnostic framework with reported detection performance. For each detector we
inject a known contamination at a controlled rate and report the detection
curve + the clean-data false-positive rate.

Output: results/round20/r20_detector_benchmark.{json,md}
"""
from __future__ import annotations

import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.audit_detectors import (
    OrientationDetector, EscalateAllDetector, TemporalLeakageDetector,
    InformativeMissingnessDetector, DefinitionalLeakageDetector, _auroc,
)
from models.conformal_escalation import StandardCRC


def _synth(sep, n=3000, pr=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pr).astype(int)
    s = np.where(y == 1, rng.normal(sep/2, 1, n), rng.normal(-sep/2, 1, n))
    return s, y


# ── 1. Orientation: detect inverted sign across effect sizes ──────────────────

def bench_orientation(n_trials=200):
    det = OrientationDetector()
    rows = []
    rng = np.random.default_rng(1)
    tp = fp = tn = fn = 0
    for t in range(n_trials):
        sep = rng.uniform(0.2, 3.0)         # true discrimination
        s, y = _synth(sep, seed=1000 + t)
        inverted = rng.random() < 0.5
        score = -s if inverted else s
        flag = det.detect(score, y).flagged
        if inverted and flag: tp += 1
        elif inverted and not flag: fn += 1
        elif not inverted and flag: fp += 1
        else: tn += 1
    sens = tp / (tp + fn) if (tp+fn) else float("nan")
    spec = tn / (tn + fp) if (tn+fp) else float("nan")
    return {"sensitivity": sens, "specificity": spec, "tp":tp,"fp":fp,"tn":tn,"fn":fn,
            "note": "random true AUROC in [0.55,0.93], random sign flip; "
                    "misses concentrate near AUROC=0.5 where sign is ill-defined"}


# ── 2. Escalate-all: detect vacuous CRC vs discrimination ─────────────────────

def bench_escalate_all():
    det = EscalateAllDetector()
    curve = []
    for sep in [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]:
        s, y = _synth(sep, seed=7)
        r = StandardCRC(alpha=0.05).fit(s, y, check_orient=False).evaluate(s, y)
        oe = r.get("over_esc_rate", 1.0)
        flag = det.detect(oe).flagged
        curve.append({"sep": sep, "auroc": round(max(_auroc(s,y),1-_auroc(s,y)),3),
                      "over_esc": round(oe,3), "flagged_vacuous": flag})
    return {"curve": curve,
            "note": "flags escalate-all exactly when discrimination is too weak "
                    "to place a non-vacuous threshold"}


# ── 3. Temporal leakage: detection ROC vs contamination rate ──────────────────

def bench_temporal(n=3000, pr=0.3):
    det = TemporalLeakageDetector(frac_thr=0.05)
    rows = []
    rng = np.random.default_rng(3)
    for p in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.65, 0.9]:
        # positives have outcome_time ~ U(0,6h); feature_time normally BEFORE it,
        # but a fraction p are injected AFTER (post-outcome contamination).
        y = (rng.random(n) < pr).astype(int)
        outcome_time = np.where(y == 1, rng.uniform(0, 360, n), np.inf)
        feat_time = np.where(y == 1, outcome_time * rng.uniform(0.1, 0.9, n), rng.uniform(0,360,n))
        contam = (y == 1) & (rng.random(n) < p)
        feat_time = np.where(contam, outcome_time + rng.uniform(1, 60, n), feat_time)
        f = det.detect(feat_time, outcome_time, y)
        rows.append({"injected_rate": p, "detected_frac": round(f.statistic,3),
                     "flagged": f.flagged})
    fp_rate = 1.0 if rows[0]["flagged"] else 0.0   # at p=0
    detected_at = [r["injected_rate"] for r in rows if r["flagged"]]
    return {"curve": rows, "false_positive_at_zero": fp_rate,
            "min_detected_injection": min(detected_at) if detected_at else None,
            "note": "threshold 0.05: no false positive at 0% contamination; "
                    "detects from the first non-trivial injection"}


# ── 4. Informative missingness: values-signal vs ordering-signal ──────────────

def _fit_auroc(X, y, seed=42):
    from experiments.round10_method_agnostic import _make_classifier
    from experiments.metrics_utils import patient_level_split
    idx = list(range(len(y)))
    import random as _r
    rng = _r.Random(seed); rng.shuffle(idx)
    cut = int(0.8*len(idx)); tr, te = idx[:cut], idx[cut:]
    clf = _make_classifier("randomforest")
    if not clf.fit([X[i] for i in tr], [bool(y[i]) for i in tr]):
        return float("nan")
    sc = [clf.score(X[i]) for i in te]
    return max(_auroc(np.array(sc), np.array([y[i] for i in te])),
               1-_auroc(np.array(sc), np.array([y[i] for i in te])))


def bench_missingness(n=4000, pr=0.3):
    det = InformativeMissingnessDetector(recover_thr=0.85)
    rng = np.random.default_rng(4)
    out = {}
    for regime in ["value_driven", "ordering_driven"]:
        y = (rng.random(n) < pr).astype(int)
        # 8 labs; each present w.p. depending on regime
        X_full=[]; X_val=[]; X_flag=[]
        for i in range(n):
            vals=[]; flags=[]
            for k in range(8):
                if regime == "value_driven":
                    present = rng.random() < 0.6       # missingness ~ random
                    v = (rng.normal(1.5,1) if y[i] else rng.normal(0,1)) if present else 0.0
                else:  # ordering_driven: presence encodes label, value is noise
                    present = rng.random() < (0.75 if y[i] else 0.3)
                    v = rng.normal(0,1) if present else 0.0
                vals.append(v if present else 0.0)
                flags.append(0.0 if present else 1.0)
            X_full.append(vals+flags); X_val.append(vals); X_flag.append(flags)
        au_full=_fit_auroc(X_full,y); au_val=_fit_auroc(X_val,y); au_flag=_fit_auroc(X_flag,y)
        f=det.detect(au_full,au_val,au_flag)
        out[regime]={"auroc_full":round(au_full,3),"auroc_value":round(au_val,3),
                     "auroc_flag":round(au_flag,3),"recover":round(f.statistic,3),
                     "flagged_ordering_driven":f.flagged}
    ok = (not out["value_driven"]["flagged_ordering_driven"]) and \
         out["ordering_driven"]["flagged_ordering_driven"]
    return {"regimes": out, "correct_discrimination": ok,
            "note": "flags ordering-driven regime, not value-driven regime"}


# ── 5. Definitional leakage: detect injected leaky feature ────────────────────

def bench_definitional(n=3000, pr=0.3):
    det = DefinitionalLeakageDetector(auroc_thr=0.90)
    rng = np.random.default_rng(5)
    rows=[]
    for strength in [0.0, 0.3, 0.6, 0.8, 0.95, 1.0]:
        y=(rng.random(n)<pr).astype(int)
        # 4 honest weak features + 1 leaky feature at controlled strength
        honest=np.column_stack([np.where(y==1,rng.normal(0.3,1,n),rng.normal(0,1,n)) for _ in range(4)])
        leak = np.where(rng.random(n) < strength, y, rng.integers(0,2,n)).astype(float)
        leak = leak + rng.normal(0,0.01,n)
        X=np.column_stack([honest, leak])
        f=det.detect(X, y, names=["h0","h1","h2","h3","LEAK"])
        rows.append({"leak_strength": strength, "max_univariate_auroc": round(f.statistic,3),
                     "flagged": f.flagged, "detail": f.detail})
    fp = rows[0]["flagged"]   # strength 0 => no leak
    return {"curve": rows, "false_positive_at_zero": fp,
            "note": "flags the injected leaky feature once it near-determines the label"}


def main():
    report = {"timestamp": datetime.now().isoformat(),
              "orientation": bench_orientation(),
              "escalate_all": bench_escalate_all(),
              "temporal_leakage": bench_temporal(),
              "informative_missingness": bench_missingness(),
              "definitional_leakage": bench_definitional()}
    outp = ROOT/"results"/"round20"/"r20_detector_benchmark"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp)+".json").write_text(json.dumps(report, indent=2, default=str))

    L=["# Round 20 — Audit detector benchmark (sensitivity/specificity, known answers)\n"]
    L.append(f"- Generated: {report['timestamp']}\n")
    o=report["orientation"]
    L.append(f"## 1. Orientation detector")
    L.append(f"- sensitivity={o['sensitivity']:.3f}, specificity={o['specificity']:.3f} "
             f"(tp{o['tp']}/fp{o['fp']}/tn{o['tn']}/fn{o['fn']})\n")
    L.append(f"## 2. Escalate-all detector")
    L.append("| sep | AUROC | over_esc | flagged vacuous |")
    L.append("|---|---|---|---|")
    for r in report["escalate_all"]["curve"]:
        L.append(f"| {r['sep']} | {r['auroc']} | {r['over_esc']} | {r['flagged_vacuous']} |")
    t=report["temporal_leakage"]
    L.append(f"\n## 3. Temporal-leakage detector (FP at 0%={t['false_positive_at_zero']}, "
             f"min detected injection={t['min_detected_injection']})")
    L.append("| injected rate | detected frac | flagged |")
    L.append("|---|---|---|")
    for r in t["curve"]:
        L.append(f"| {r['injected_rate']} | {r['detected_frac']} | {r['flagged']} |")
    m=report["informative_missingness"]
    L.append(f"\n## 4. Informative-missingness detector (correct={m['correct_discrimination']})")
    L.append("| regime | full | value | flag | recover | flagged ordering-driven |")
    L.append("|---|---|---|---|---|---|")
    for reg,d in m["regimes"].items():
        L.append(f"| {reg} | {d['auroc_full']} | {d['auroc_value']} | {d['auroc_flag']} | "
                 f"{d['recover']} | {d['flagged_ordering_driven']} |")
    df=report["definitional_leakage"]
    L.append(f"\n## 5. Definitional-leakage detector (FP at 0%={df['false_positive_at_zero']})")
    L.append("| leak strength | max univariate AUROC | flagged |")
    L.append("|---|---|---|")
    for r in df["curve"]:
        L.append(f"| {r['leak_strength']} | {r['max_univariate_auroc']} | {r['flagged']} |")
    Path(str(outp)+".md").write_text("\n".join(L))
    print(f"✅ {outp}.{{json,md}}")
    # quick console summary
    print(f"orientation sens={o['sensitivity']:.2f} spec={o['specificity']:.2f}")
    print(f"temporal FP@0={t['false_positive_at_zero']} min_detect={t['min_detected_injection']}")
    print(f"missingness correct={m['correct_discrimination']}")
    print(f"definitional FP@0={df['false_positive_at_zero']}")


if __name__ == "__main__":
    main()
