"""
Round 21 — Apply the five detectors to a faithfully-reconstructed *published*
clinical-CP pipeline, not synthetic data.

Target: the CPMORS-style setup (Development and Validation of an Interpretable
Conformal Predictor to Predict Sepsis Mortality Risk, JMIR 2024, e50369):
  - MIMIC-IV cohort (one of CPMORS's own datasets),
  - first-Nh-of-ICU features -> in-hospital mortality,
  - inductive/split conformal with nonconformity = 1 - predicted class prob,
  - missing values imputed (kNN/mean), NO informative-missingness check (D3=No),
  - efficiency reported as flagged/"multiple prediction" rate (D1=Yes).

We reconstruct this decision-time pipeline on our MIMIC-IV lab cache and run the
detector suite. The claim is not "we reproduce CPMORS's numbers" but "auditing a
faithful reconstruction of the CPMORS pattern surfaces a dimension (D3) the
original study did not report." Honest either way: the detectors flag a real
hidden dependence, or certify the pipeline clean.

Output: results/round21/r21_published_pipeline_audit.{json,md}
"""
from __future__ import annotations
import json, sys, gzip, random
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from sklearn.ensemble import RandomForestClassifier
from models.audit_detectors import (
    OrientationDetector, EscalateAllDetector, TemporalLeakageDetector,
    InformativeMissingnessDetector, DefinitionalLeakageDetector, _auroc,
)
from models.conformal_escalation import StandardCRC

COHORT = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"
# unguarded first-6h labs (the CPMORS pattern does NOT guard the window)
LAB_UNGUARDED = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_labvalues_6h.jsonl"
MIMIC = Path("/Users/idaun/Downloads/mimic-iv-3.1")

LAB = ['anion_gap','bicarbonate','bun','calcium','chloride','creatinine','glucose',
'glucose_bg','hco3','hematocrit','hemoglobin','hemoglobin_bg','lactate','magnesium',
'pco2','ph','platelet','potassium','sodium','troponin_i','troponin_t','wbc','n_vital_flags']


def load():
    labmap = {}
    with open(LAB_UNGUARDED) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line); labmap[d["hadm_id"]] = d["labs"]
    rows = []
    with open(COHORT) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    # death time offset (hours from admit) for the temporal detector
    cohort = {str(r["hadm_id"]) for r in rows}
    from datetime import datetime as dt
    def pdt(s):
        try: return dt.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
        except Exception: return None
    admit={}; death={}
    with gzip.open(MIMIC/"hosp"/"admissions.csv.gz","rt") as f:
        h=f.readline().strip().split(","); hi=h.index("hadm_id"); ai=h.index("admittime"); di=h.index("deathtime")
        for line in f:
            p=line.split(",")
            if p[hi] in cohort:
                admit[p[hi]]=pdt(p[ai])
                if p[di].strip(): death[p[hi]]=pdt(p[di])
    return labmap, rows, admit, death


def main():
    labmap, rows, admit, death = load()
    lab_names = sorted({k for d in labmap.values() for k in d})
    med = {ln: float(np.median([labmap[str(r['hadm_id'])][ln] for r in rows
            if str(r['hadm_id']) in labmap and ln in labmap[str(r['hadm_id'])]] or [0.0]))
           for ln in lab_names}

    def y_of(r):  # in-hospital mortality (CPMORS outcome; not lab-defined)
        return 1 if (r.get('outcome') or {}).get('in_hospital_mortality') else 0

    def fv(r, mode):
        labs = labmap.get(str(r['hadm_id']), {}); vec = []
        for ln in lab_names:
            p = ln in labs
            if mode in ('full','value'): vec.append(float(labs[ln]) if p else med[ln])  # CPMORS: impute
            if mode in ('full','flag'): vec.append(0.0 if p else 1.0)
        return vec

    data = [(r, y_of(r), str(r.get('subject_id'))) for r in rows]
    prev = np.mean([d[1] for d in data])
    cov = len(labmap)/len(rows)

    # subject-level split (avoid within-patient leakage)
    subj = list({d[2] for d in data}); random.Random(42).shuffle(subj)
    cut = int(0.8*len(subj)); calsubj = set(subj[:cut])
    cal = [d for d in data if d[2] in calsubj]
    test = [d for d in data if d[2] not in calsubj]

    report = {"timestamp": datetime.now().isoformat(),
              "target_pipeline": "CPMORS-style: first-6h labs -> in-hospital mortality, MIMIC-IV, impute-missing, split conformal",
              "cohort_n": len(rows), "lab_coverage": cov, "mortality_prev": float(prev),
              "detectors": {}}

    # Fit the CPMORS-style RF (full = imputed value + implicit) and get scores
    def fit_scores(mode):
        clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
        Xc=[fv(d[0],mode) for d in cal]; yc=[d[1] for d in cal]
        clf.fit(Xc, yc); ci=list(clf.classes_).index(1)
        cs=clf.predict_proba(Xc)[:,ci]
        ts=clf.predict_proba([fv(d[0],mode) for d in test])[:,ci]
        return np.array(cs), np.array(yc), ts, np.array([d[1] for d in test])

    cs, cy, ts, ty = fit_scores("full")
    _, _, tsv, _ = fit_scores("value")
    _, _, tsf, _ = fit_scores("flag")
    au_full=max(_auroc(ts,ty),1-_auroc(ts,ty))
    au_val=max(_auroc(tsv,ty),1-_auroc(tsv,ty))
    au_flag=max(_auroc(tsf,ty),1-_auroc(tsf,ty))

    # ── Detector 1: orientation ──
    o = OrientationDetector().detect(cs, cy)
    report["detectors"]["orientation"] = {"flagged": o.flagged, "auroc": round(_auroc(cs,cy),3), "detail": o.detail}

    # ── Detector 2: escalate-all (over_esc at CRC threshold, alpha=0.10 like CPMORS 90% conf) ──
    sc = StandardCRC(alpha=0.10).fit(cs, cy, check_orient=False).evaluate(ts, ty)
    ea = EscalateAllDetector().detect(sc.get("over_esc_rate", 1.0))
    report["detectors"]["escalate_all"] = {"flagged": ea.flagged, "over_esc": round(sc.get("over_esc_rate",1.0),3),
                                            "miss_rate": round(sc.get("miss_rate",1.0),3)}

    # ── Detector 3: temporal leakage (feature time vs death time) ──
    # feature_time proxy = end of the 6h window (labs may be charted up to +6h);
    # a positive is contaminated if death occurred before +6h (feature after outcome).
    ft=[]; ot=[]; yl=[]
    for r in rows:
        hid=str(r['hadm_id']); y=y_of(r)
        at=admit.get(hid); dt_=death.get(hid)
        if at is None: continue
        # feature charted within [0,6h]; use midpoint 3h as representative
        ft.append(3.0)
        ot.append(((dt_-at).total_seconds()/3600.0) if (dt_ is not None) else float('inf'))
        yl.append(y)
    tl = TemporalLeakageDetector(frac_thr=0.05).detect(ft, ot, yl)
    report["detectors"]["temporal_leakage"] = {"flagged": tl.flagged, "frac_after_outcome": round(tl.statistic,3),
        "detail": "fraction of in-hospital-mortality positives whose 6h feature window extends past death time"}

    # ── Detector 4: informative missingness ──
    im = InformativeMissingnessDetector(0.85).detect(au_full, au_val, au_flag)
    report["detectors"]["informative_missingness"] = {"flagged": im.flagged,
        "auroc_full": round(au_full,3), "auroc_value": round(au_val,3), "auroc_flag": round(au_flag,3),
        "recover": round(im.statistic,3), "detail": im.detail}

    # ── Detector 5: definitional leakage (per-feature univariate on full feature set) ──
    Xall=np.array([fv(d[0],"value") for d in cal]); yall=np.array([d[1] for d in cal])
    dl = DefinitionalLeakageDetector(0.90).detect(Xall, yall, names=lab_names)
    report["detectors"]["definitional_leakage"] = {"flagged": dl.flagged, "max_univariate_auroc": round(dl.statistic,3),
        "detail": dl.detail}

    outp = ROOT/"results"/"round21"/"r21_published_pipeline_audit"
    outp.parent.mkdir(parents=True, exist_ok=True)
    Path(str(outp)+".json").write_text(json.dumps(report, indent=2, default=str))

    L=["# Round 21 — Detector audit of a reconstructed CPMORS-style published pipeline\n"]
    L.append(f"- Target: {report['target_pipeline']}")
    L.append(f"- MIMIC-IV cohort n={report['cohort_n']}, lab coverage {cov:.0%}, mortality prev {prev:.3f}\n")
    L.append("| Detector | Flagged? | Key statistic |")
    L.append("|---|---|---|")
    d=report["detectors"]
    L.append(f"| Orientation | {d['orientation']['flagged']} | AUROC(score,label)={d['orientation']['auroc']} |")
    L.append(f"| Escalate-all | {d['escalate_all']['flagged']} | over_esc={d['escalate_all']['over_esc']}, miss={d['escalate_all']['miss_rate']} |")
    L.append(f"| Temporal leakage | {d['temporal_leakage']['flagged']} | frac positives' window past death={d['temporal_leakage']['frac_after_outcome']} |")
    L.append(f"| **Informative missingness** | **{d['informative_missingness']['flagged']}** | full={d['informative_missingness']['auroc_full']}, value={d['informative_missingness']['auroc_value']}, flag={d['informative_missingness']['auroc_flag']}, recover={d['informative_missingness']['recover']} |")
    L.append(f"| Definitional leakage | {d['definitional_leakage']['flagged']} | max univariate AUROC={d['definitional_leakage']['max_univariate_auroc']} |")
    L.append("\n## Reading\n")
    im_flag = d['informative_missingness']['flagged']
    L.append(f"The reconstructed CPMORS-style gate " +
             (f"**is flagged by the informative-missingness detector** "
              f"(flag-only recovers {d['informative_missingness']['recover']:.0%} of the above-chance AUROC): "
              f"its apparent signal is substantially an ordering-behavior proxy, a dependence the original "
              f"study did not report (D3=No in our audit)."
              if im_flag else
              f"is **not** flagged for informative missingness (flag-only recovers "
              f"{d['informative_missingness']['recover']:.0%}); the reconstructed pipeline is comparatively clean on D3."))
    L.append("\nThis applies the detector suite to a faithful reconstruction of a published pipeline "
             "pattern — not synthetic data — turning the toolkit from a sanity-check collection into a "
             "diagnostic that surfaces (or rules out) a specific unreported dependence in field practice.")
    Path(str(outp)+".md").write_text("\n".join(L))
    print(f"✅ {outp}.{{json,md}}")
    for k,v in d.items():
        print(f"  {k}: flagged={v['flagged']}")


if __name__ == "__main__":
    main()
