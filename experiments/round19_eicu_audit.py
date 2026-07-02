"""
Round 19 — Independent-dataset validation of the audit protocol on eICU-CRD.

Applies the SAME verified core + value/flag decomposition + over-escalation
reporting to a second, independent database (eICU-CRD, 200k ICU stays, 335 US
hospitals). Purpose: show the audit tooling catches the informative-missingness
trap in data that is NOT ours — i.e. the failure modes are general, not
idiosyncratic to our MIMIC pipeline.

eICU is ICU-only, so:
  - The clean escalation-worthy label is in-hospital mortality (NOT lab-defined).
  - The MIMIC temporal-leakage-vs-ICU-transfer artifact does NOT map (patients
    are already in ICU); we do not claim to re-demonstrate it here.
  - Informative missingness (which early ICU labs were ordered) and selection
    DO map — this is the transferable general trap we test.

Features: first-6h-of-ICU lab VALUES (labresultoffset in [0,360]) for a standard
panel, value + missing-flag, plus age/emergency. Verified core, 5-seed
patient-level split (by patienthealthsystemstayid), FULL/VALUE/FLAG decomposition.

Usage:
  .venv/bin/python experiments/round19_eicu_audit.py --extract   # ~5-15 min
  .venv/bin/python experiments/round19_eicu_audit.py
"""
from __future__ import annotations

import argparse, gzip, json, sys, csv, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

EICU = ROOT / "data" / "raw" / "eicu-crd"
COHORT = ROOT / "data" / "raw" / "eicu_cases_v11_full.jsonl"
CACHE = ROOT / "data" / "raw" / "eicu_labvalues_6h.jsonl"

# eICU labname -> short name (first-6h ICU panel, leakage-safe decision-time)
PANEL = {
    "lactate":"lactate", "creatinine":"creatinine", "BUN":"bun",
    "sodium":"sodium", "potassium":"potassium", "bicarbonate":"bicarbonate",
    "chloride":"chloride", "anion gap":"anion_gap", "glucose":"glucose",
    "WBC x 1000":"wbc", "platelets x 1000":"platelet", "Hgb":"hemoglobin",
    "calcium":"calcium", "magnesium":"magnesium", "paCO2":"pco2", "pH":"ph",
    "HCO3":"hco3", "troponin - I":"troponin_i", "Hct":"hematocrit",
    "bedside glucose":"glucose_bedside",
}
LAB_NAMES = sorted(set(PANEL.values()))
WINDOW_MIN = 360.0   # first 6h of ICU


def extract():
    cohort = set()
    with open(COHORT) as f:
        for line in f:
            line=line.strip()
            if line: cohort.add(str(json.loads(line).get("hadm_id")))
    print(f"[R19] cohort {len(cohort)} eICU stays")

    vals={}; n=0; kept=0
    with gzip.open(EICU/"lab.csv.gz","rt") as f:
        h=f.readline().strip().split(",")
        si=h.index("patientunitstayid"); oi=h.index("labresultoffset")
        ni=h.index("labname"); ri=h.index("labresult")
        for p in csv.reader(f):
            n+=1
            if n%10_000_000==0: print(f"  ...{n//1_000_000}M rows, kept {kept}")
            if len(p)<=max(si,oi,ni,ri): continue
            stay=p[si]
            if stay not in cohort: continue
            name=PANEL.get(p[ni])
            if name is None: continue
            try:
                off=float(p[oi])
            except ValueError:
                continue
            if off<0 or off>WINDOW_MIN: continue   # first 6h of ICU only
            try:
                v=float(p[ri])
            except (ValueError,IndexError):
                continue
            d=vals.setdefault(stay,{})
            if name not in d: d[name]=v; kept+=1   # earliest (rows are ~time-ordered)
    print(f"[R19] scanned {n}, kept {kept} for {len(vals)} stays")
    with open(CACHE,"w") as out:
        for stay,d in vals.items():
            out.write(json.dumps({"hadm_id":stay,"labs":d})+"\n")
    print(f"[R19] cached -> {CACHE}")


def analyze():
    from models.conformal_escalation import StandardCRC
    from experiments.round14_genuine_win_feasibility import auroc
    from experiments.metrics_utils import patient_level_split
    from experiments.round10_method_agnostic import _make_classifier

    if not CACHE.exists(): sys.exit(f"missing {CACHE}; run --extract")
    labmap={}
    with open(CACHE) as f:
        for line in f:
            line=line.strip()
            if line: d=json.loads(line); labmap[d["hadm_id"]]=d["labs"]
    rows=[]
    with open(COHORT) as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    print(f"[R19] guarded lab values for {len(labmap)}/{len(rows)} stays "
          f"({100*len(labmap)/len(rows):.0f}% coverage)")

    medians={ln:0.0 for ln in LAB_NAMES}
    for ln in LAB_NAMES:
        xs=[labmap[str(r['hadm_id'])][ln] for r in rows
            if str(r['hadm_id']) in labmap and ln in labmap[str(r['hadm_id'])]]
        if xs: medians[ln]=float(np.median(xs))

    # eICU clean label: in-hospital mortality (ICU-only cohort; not lab-defined)
    def clean_y(r):
        return 1 if (r.get("outcome") or {}).get("in_hospital_mortality") else 0

    def fv(r, mode):
        labs=labmap.get(str(r['hadm_id']),{})
        vec=[]
        for ln in LAB_NAMES:
            present=ln in labs
            if mode in ("full","value"):
                vec.append(float(labs[ln]) if present else medians[ln])
            if mode in ("full","flag"):
                vec.append(0.0 if present else 1.0)
        demo=r.get("demographics") or {}
        age={"unknown":0,"<18":1,"18-34":2,"35-49":3,"50-64":4,"65-79":5,"80+":6}.get(demo.get("age_bucket","unknown"),0)
        adm=r.get("admission_type","") or ""
        vec+=[float(age),1.0 if ("EMER" in adm or "URG" in adm) else 0.0]
        return vec

    data=[(r,clean_y(r),str(r.get("subject_id")),r.get("stratum")) for r in rows]
    # overall prevalence
    ally=[d[1] for d in data]
    print(f"[R19] mortality prevalence: {np.mean(ally):.3f} ({sum(ally)}/{len(ally)})")
    ALPHAS={"CRITICAL":0.05,"HIGH":0.10,"MODERATE":0.15,"LOW":0.20}
    out={"dataset":"eICU-CRD","coverage":len(labmap)/len(rows),
         "mortality_prevalence":float(np.mean(ally)),"per_stratum":{}}

    print("\n=== R19 eICU independent audit: mortality label, FULL/VALUE/FLAG ===")
    for s in ["CRITICAL","HIGH","MODERATE","LOW"]:
        sub=[d for d in data if d[3]==s]
        if len(sub)<50: out["per_stratum"][s]={"insufficient":True}; continue
        res={}
        for mode in ["full","value","flag"]:
            CS=[];CY=[];TS=[];TY=[]
            for seed in [42,43,44,45,46]:
                cal,test=patient_level_split(sub,group_of=lambda d:d[2],cal_frac=0.8,seed=seed)
                rng=random.Random(seed); rng.shuffle(cal); rng.shuffle(test)
                cal=cal[:3000]; test=test[:3000]
                clf=_make_classifier("randomforest")
                Xc=[fv(d[0],mode) for d in cal]; yc=[bool(d[1]) for d in cal]
                if not clf.fit(Xc,yc): continue
                CS+=[clf.score(x) for x in Xc]; CY+=[d[1] for d in cal]
                TS+=[clf.score(fv(d[0],mode)) for d in test]; TY+=[d[1] for d in test]
            CS=np.array(CS);CY=np.array(CY);TS=np.array(TS);TY=np.array(TY)
            if (TY==1).sum()<5: res[mode]={"insufficient":True}; continue
            au=auroc(TS,TY); au=max(au,1-au)
            sc=StandardCRC(alpha=ALPHAS[s]).fit(CS,CY,check_orient=False).evaluate(TS,TY)
            res[mode]={"auroc":au,"miss_rate":sc.get("miss_rate"),
                       "over_esc_rate":sc.get("over_esc_rate"),"infeasible":sc.get("infeasible")}
        out["per_stratum"][s]=res
        f=res.get("full",{}); v=res.get("value",{}); fl=res.get("flag",{})
        def g(d,k):
            x=d.get(k); return f"{x:.3f}" if isinstance(x,(int,float)) else "—"
        print(f"  {s}: FULL AUROC={g(f,'auroc')} (miss {g(f,'miss_rate')}, over_esc {g(f,'over_esc_rate')}) "
              f"| VALUE={g(v,'auroc')} | FLAG={g(fl,'auroc')}")

    outp=ROOT/"results"/"round19"/"r19_eicu_audit"
    outp.parent.mkdir(parents=True,exist_ok=True)
    Path(str(outp)+".json").write_text(json.dumps(out,indent=2,default=str))
    print(f"\n✅ {outp}.json")
    # informative-missingness replication verdict
    cf=out["per_stratum"].get("CRITICAL",{})
    if isinstance(cf,dict) and "full" in cf and "flag" in cf:
        full=cf["full"].get("auroc"); flag=cf["flag"].get("auroc")
        if isinstance(full,(int,float)) and isinstance(flag,(int,float)):
            gap=full-flag
            verdict = ("REPLICATES (flag ~= full; missingness dominates)" if gap<0.03
                       else "does NOT replicate (values add signal)")
            print(f"\n★ Informative-missingness on eICU CRITICAL: FULL={full:.3f} FLAG={flag:.3f} "
                  f"(gap {gap:+.3f}) -> {verdict}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--extract",action="store_true")
    a=ap.parse_args()
    if a.extract: extract()
    analyze()


if __name__=="__main__":
    main()
