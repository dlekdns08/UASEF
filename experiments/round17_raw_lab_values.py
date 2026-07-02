"""
Round 17 — Extract first-6h lab VALUES from raw MIMIC-IV and test whether a
genuine leakage-safe CRITICAL escalation gate is achievable with the verified
conformal core.

The prior rounds used only sparse abnormal-lab FLAGS (n_labs count). This round
pulls actual numeric lab values within [admittime, admittime+6h] for a standard
panel, builds value + missingness features, and evaluates the verified-core CRC
gate on the CLEAN, non-circular label Y = (ICU within 24h) OR (in-hosp mortality).

All local, credentialed data. PHI egress: 0. Extracted features cached to
data/raw/mimic-iv/mimic4_labvalues_6h.jsonl (gitignored).

Usage:
  export UASEF_BACKEND_NEVER_SEND_PHI=1
  .venv/bin/python experiments/round17_raw_lab_values.py --extract   # ~15-30 min
  .venv/bin/python experiments/round17_raw_lab_values.py             # analyze cache
"""
from __future__ import annotations

import argparse, gzip, json, sys, random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

MIMIC = Path("/Users/idaun/Downloads/mimic-iv-3.1")
COHORT = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"
LABCACHE = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_labvalues_6h.jsonl"

# Standard early-panel labs (decision-time, leakage-safe): itemid -> short name
PANEL = {
    50813: "lactate", 50912: "creatinine", 50882: "bicarbonate",
    50971: "potassium", 50983: "sodium", 51006: "bun", 51002: "troponin_i",
    51003: "troponin_t", 51222: "hemoglobin", 51301: "wbc", 51265: "platelet",
    50931: "glucose", 50902: "chloride", 50868: "anion_gap", 50818: "pco2",
    50820: "ph", 50893: "calcium", 50960: "magnesium", 50809: "glucose_bg",
    50811: "hemoglobin_bg",
}
WINDOW_H = 6.0


def _parse_dt(s):
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def extract():
    print("[R17] loading cohort hadm_ids + admittime")
    cohort_hadm = set()
    with open(COHORT) as f:
        for line in f:
            line = line.strip()
            if line:
                cohort_hadm.add(str(json.loads(line).get("hadm_id")))

    admit = {}   # hadm_id -> admittime datetime
    with gzip.open(MIMIC / "hosp" / "admissions.csv.gz", "rt") as f:
        hdr = f.readline().strip().split(",")
        hi, ai = hdr.index("hadm_id"), hdr.index("admittime")
        for line in f:
            p = line.split(",")
            if p[hi] in cohort_hadm:
                admit[p[hi]] = _parse_dt(p[ai])
    print(f"[R17] admittime for {len(admit)} admissions")

    # stream labevents, keep first value per (hadm, itemid) within window
    print("[R17] streaming labevents.csv.gz (2.4GB) — this takes ~15-30 min")
    vals: dict[str, dict[str, float]] = {}    # hadm -> {labname: valuenum}
    n = 0; kept = 0
    with gzip.open(MIMIC / "hosp" / "labevents.csv.gz", "rt") as f:
        hdr = f.readline().strip().split(",")
        hi = hdr.index("hadm_id"); ii = hdr.index("itemid")
        ci = hdr.index("charttime"); vi = hdr.index("valuenum")
        import csv
        rd = csv.reader(f)
        for p in rd:
            n += 1
            if n % 20_000_000 == 0:
                print(f"  ...{n//1_000_000}M rows, kept {kept}")
            if len(p) <= max(hi, ii, ci, vi):
                continue
            hadm = p[hi]
            if hadm not in admit:
                continue
            try:
                itemid = int(p[ii])
            except ValueError:
                continue
            if itemid not in PANEL:
                continue
            ct = _parse_dt(p[ci]); at = admit[hadm]
            if ct is None or at is None:
                continue
            if ct > at + timedelta(hours=WINDOW_H):
                continue
            try:
                v = float(p[vi])
            except (ValueError, IndexError):
                continue
            name = PANEL[itemid]
            d = vals.setdefault(hadm, {})
            # keep the EARLIEST (first) value: only set if not present
            if name not in d:
                d[name] = v; kept += 1
    print(f"[R17] scanned {n} rows, kept {kept} lab values for {len(vals)} admissions")

    LABCACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABCACHE, "w") as out:
        for hadm, d in vals.items():
            out.write(json.dumps({"hadm_id": hadm, "labs": d}) + "\n")
    print(f"[R17] cached -> {LABCACHE}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

LAB_NAMES = sorted(set(PANEL.values()))


def analyze():
    from models.conformal_escalation import StandardCRC, BoundedCRC
    from experiments.round14_genuine_win_feasibility import auroc
    from experiments.metrics_utils import patient_level_split
    from experiments.round10_method_agnostic import _make_classifier

    if not LABCACHE.exists():
        sys.exit(f"lab cache missing: {LABCACHE}. Run with --extract first.")
    labmap = {}
    with open(LABCACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line); labmap[d["hadm_id"]] = d["labs"]
    print(f"[R17] lab values for {len(labmap)} admissions")

    rows = []
    with open(COHORT) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))

    # median impute per lab (train-agnostic global median is fine for a demo)
    medians = {}
    for ln in LAB_NAMES:
        xs = [labmap[str(r["hadm_id"])].get(ln) for r in rows
              if str(r["hadm_id"]) in labmap and ln in labmap[str(r["hadm_id"])]]
        medians[ln] = float(np.median(xs)) if xs else 0.0

    def clean_y(r):
        o = r.get("outcome") or {}
        return 1 if (o.get("icu_within_24h") or o.get("in_hospital_mortality")) else 0

    def fv(r):
        labs = labmap.get(str(r["hadm_id"]), {})
        vec = []
        for ln in LAB_NAMES:
            if ln in labs:
                vec.append(float(labs[ln])); vec.append(0.0)   # value, not-missing
            else:
                vec.append(medians[ln]); vec.append(1.0)        # impute, missing-flag
        # + a few decision-time demographics
        demo = r.get("demographics") or {}
        age = {"unknown":0,"<18":1,"18-34":2,"35-49":3,"50-64":4,"65-79":5,"80+":6}.get(demo.get("age_bucket","unknown"),0)
        adm = r.get("admission_type","") or ""
        vec += [float(age), 1.0 if ("EMER" in adm or "URG" in adm) else 0.0]
        return vec

    data = [(r, clean_y(r), str(r.get("subject_id")), r.get("stratum")) for r in rows]
    ALPHAS = {"CRITICAL":0.05,"HIGH":0.10,"MODERATE":0.15,"LOW":0.20}
    out = {"timestamp": None, "n_labs_panel": len(LAB_NAMES), "per_stratum": {}}

    print("\n=== R17: raw first-6h lab VALUES, clean label (ICU/mortality), RF, verified core ===")
    for s in ["CRITICAL","HIGH","MODERATE","LOW"]:
        sub = [d for d in data if d[3] == s]
        cal_s=cal_y=test_s=test_y=None; CS=[];CY=[];TS=[];TY=[]
        for seed in [42,43,44,45,46]:
            cal,test = patient_level_split(sub, group_of=lambda d:d[2], cal_frac=0.8, seed=seed)
            rng=random.Random(seed); rng.shuffle(cal); rng.shuffle(test)
            cal=cal[:3000]; test=test[:3000]
            clf=_make_classifier("randomforest")
            Xc=[fv(d[0]) for d in cal]; yc=[bool(d[1]) for d in cal]
            if not clf.fit(Xc,yc): continue
            CS+=[clf.score(x) for x in Xc]; CY+=[d[1] for d in cal]
            TS+=[clf.score(fv(d[0])) for d in test]; TY+=[d[1] for d in test]
        CS=np.array(CS);CY=np.array(CY);TS=np.array(TS);TY=np.array(TY)
        if (TY==1).sum()<5:
            out["per_stratum"][s]={"insufficient":True}; print(f"  {s}: insufficient"); continue
        au=auroc(TS,TY); au=max(au,1-au)
        sc=StandardCRC(alpha=ALPHAS[s]).fit(CS,CY,check_orient=False).evaluate(TS,TY)
        b=BoundedCRC(alpha=ALPHAS[s],c_miss=0.9,c_over=0.1).fit(CS,CY,check_orient=False).evaluate(TS,TY)
        out["per_stratum"][s]={"auroc":au,"standard":sc,"bcrc_0.1":b}
        bstr = "INFEAS" if b.get("infeasible") else f"miss {b['miss_rate']:.3f}/oe {b['over_esc_rate']:.3f}"
        print(f"  {s}: AUROC={au:.3f} | StdCRC miss={sc['miss_rate']:.3f} over_esc={sc['over_esc_rate']:.3f} "
              f"genuine={sc['genuine_win']} highconf={sc['high_conf_coverage']} | bCRC(.1) {bstr}")

    outp = ROOT/"results"/"round17"/"r17_raw_lab_values"
    outp.parent.mkdir(parents=True, exist_ok=True)
    import datetime as _dt
    Path(str(outp)+".json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✅ {outp}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    if args.extract:
        extract()
    analyze()


if __name__ == "__main__":
    main()
