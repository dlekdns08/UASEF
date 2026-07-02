"""
Round 18 — The leakage-safe FLOOR: decision-time-guarded lab features,
full cohort, FULL/VALUE/FLAG decomposition, verified conformal core.

Fixes the R17 temporal leakage: re-extracts first-window labs with the guard
    charttime <= min(admittime + 6h, icu_intime)
so NO lab drawn after ICU transfer enters the feature vector. Evaluates on the
FULL 14000 cohort (impute missing) and reports FULL vs VALUES-ONLY vs
FLAGS-ONLY so informative-missingness is explicit. Clean label
Y = (icu_within_24h OR in_hospital_mortality).

This is the single definitive number: the honest decision-time ceiling.

Usage:
  export UASEF_BACKEND_NEVER_SEND_PHI=1
  .venv/bin/python experiments/round18_leakage_safe_floor.py --extract   # ~20-30 min
  .venv/bin/python experiments/round18_leakage_safe_floor.py             # analyze
"""
from __future__ import annotations

import argparse, gzip, json, sys, csv, random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

MIMIC = Path("/Users/idaun/Downloads/mimic-iv-3.1")
COHORT = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"
CACHE = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_labvalues_6h_guarded.jsonl"
PANEL = {50813:"lactate",50912:"creatinine",50882:"bicarbonate",50971:"potassium",
         50983:"sodium",51006:"bun",51002:"troponin_i",51003:"troponin_t",
         51222:"hemoglobin",51301:"wbc",51265:"platelet",50931:"glucose",
         50902:"chloride",50868:"anion_gap",50818:"pco2",50820:"ph",
         50893:"calcium",50960:"magnesium",50809:"glucose_bg",50811:"hemoglobin_bg"}
LAB_NAMES = sorted(set(PANEL.values()))


def _dt(s):
    try: return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
    except Exception: return None


def extract():
    cohort = set()
    with open(COHORT) as f:
        for line in f:
            line=line.strip()
            if line: cohort.add(str(json.loads(line).get("hadm_id")))
    print(f"[R18] cohort {len(cohort)}")

    admit={}
    with gzip.open(MIMIC/"hosp"/"admissions.csv.gz","rt") as f:
        h=f.readline().strip().split(","); hi=h.index("hadm_id"); ai=h.index("admittime")
        for line in f:
            p=line.split(",")
            if p[hi] in cohort: admit[p[hi]]=_dt(p[ai])

    # earliest ICU intime per hadm
    icu={}
    with gzip.open(MIMIC/"icu"/"icustays.csv.gz","rt") as f:
        h=f.readline().strip().split(","); hi=h.index("hadm_id"); ii=h.index("intime")
        for line in f:
            p=line.split(",")
            if p[hi] in cohort:
                t=_dt(p[ii])
                if t and (p[hi] not in icu or t<icu[p[hi]]): icu[p[hi]]=t
    print(f"[R18] admittime {len(admit)}, icu_intime {len(icu)}")

    # per-hadm cutoff = min(admit+6h, icu_intime)
    cutoff={}
    for hadm,at in admit.items():
        if at is None: continue
        c=at+timedelta(hours=6)
        it=icu.get(hadm)
        if it is not None and it<c: c=it
        cutoff[hadm]=(at,c)

    print("[R18] streaming labevents with icu_intime guard (~20-30 min)")
    vals={}; n=0; kept=0
    with gzip.open(MIMIC/"hosp"/"labevents.csv.gz","rt") as f:
        h=f.readline().strip().split(",")
        hi=h.index("hadm_id"); ii=h.index("itemid"); ci=h.index("charttime"); vi=h.index("valuenum")
        for p in csv.reader(f):
            n+=1
            if n%40_000_000==0: print(f"  ...{n//1_000_000}M, kept {kept}")
            if len(p)<=max(hi,ii,ci,vi): continue
            hadm=p[hi]
            if hadm not in cutoff: continue
            try: itemid=int(p[ii])
            except ValueError: continue
            if itemid not in PANEL: continue
            ct=_dt(p[ci]); at,cut=cutoff[hadm]
            if ct is None or at is None or ct<at or ct>cut: continue  # GUARD: >= admit AND <= cutoff
            try: v=float(p[vi])
            except (ValueError,IndexError): continue
            name=PANEL[itemid]; d=vals.setdefault(hadm,{})
            if name not in d: d[name]=v; kept+=1
    print(f"[R18] scanned {n}, kept {kept} for {len(vals)} admissions (guarded)")
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    with open(CACHE,"w") as out:
        for hadm,d in vals.items():
            out.write(json.dumps({"hadm_id":hadm,"labs":d})+"\n")
    print(f"[R18] cached -> {CACHE}")


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
    print(f"[R18] guarded lab values for {len(labmap)}/{len(rows)} admissions "
          f"({100*len(labmap)/len(rows):.0f}% coverage)")

    medians={ln:0.0 for ln in LAB_NAMES}
    for ln in LAB_NAMES:
        xs=[labmap[str(r['hadm_id'])][ln] for r in rows
            if str(r['hadm_id']) in labmap and ln in labmap[str(r['hadm_id'])]]
        if xs: medians[ln]=float(np.median(xs))

    def clean_y(r):
        o=r.get("outcome") or {}
        return 1 if (o.get("icu_within_24h") or o.get("in_hospital_mortality")) else 0

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
    ALPHAS={"CRITICAL":0.05,"HIGH":0.10,"MODERATE":0.15,"LOW":0.20}
    out={"coverage":len(labmap)/len(rows),"per_stratum":{}}

    print("\n=== R18 leakage-safe FLOOR: guarded labs, FULL cohort, clean label ===")
    print("    (icu_intime guard: no post-ICU-transfer labs; full-cohort impute)")
    for s in ["CRITICAL","HIGH","MODERATE","LOW"]:
        sub=[d for d in data if d[3]==s]
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
                       "over_esc_rate":sc.get("over_esc_rate"),
                       "genuine_win":sc.get("genuine_win"),
                       "infeasible":sc.get("infeasible")}
        out["per_stratum"][s]=res
        f=res.get("full",{}); v=res.get("value",{}); fl=res.get("flag",{})
        def g(d,k):
            x=d.get(k); return f"{x:.3f}" if isinstance(x,(int,float)) else "—"
        print(f"  {s}: FULL AUROC={g(f,'auroc')} (miss {g(f,'miss_rate')}, over_esc {g(f,'over_esc_rate')}) "
              f"| VALUE={g(v,'auroc')} | FLAG={g(fl,'auroc')}")

    outp=ROOT/"results"/"round18"/"r18_leakage_safe_floor"
    outp.parent.mkdir(parents=True,exist_ok=True)
    Path(str(outp)+".json").write_text(json.dumps(out,indent=2,default=str))
    print(f"\n✅ {outp}.json")
    # headline
    cf=out["per_stratum"].get("CRITICAL",{}).get("full",{})
    print(f"\n★ DECISION-TIME CEILING (CRITICAL, leakage-safe, full cohort): "
          f"AUROC={cf.get('auroc')}, over_esc={cf.get('over_esc_rate')}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--extract",action="store_true")
    a=ap.parse_args()
    if a.extract: extract()
    analyze()


if __name__=="__main__":
    main()
