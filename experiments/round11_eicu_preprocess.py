"""
Round 11 R11.3 — eICU-CRD v2.0 preprocessing for cross-center replication.

Input: PhysioNet eICU-CRD v2.0 csv.gz files at data/raw/eicu-crd/
    - patient.csv.gz           (demographics + outcome)
    - apachePatientResult.csv  (APACHE score — leakage suspect, used in Pass A only)
    - admissionDx.csv.gz       (admission diagnosis text — decision-time)
    - pastHistory.csv.gz       (prior comorbidity — Charlson-like)
    - lab.csv.gz               (first-6h lab acuity flags)

Output: data/raw/eicu_cases_v11.jsonl
    각 row 는 MIMIC-IV 의 mimic4_cases_v10.jsonl 와 호환되는 schema:
       {hadm_id, subject_id, stratum, expected_escalate, specialty,
        admission_type, demographics, outcome, structured,
        _charlson_index, _n_vital_flags, _specialty_baseline_rate, ...}

Stratum 정의 (decision-time, leakage-safe):
    eICU 는 모두 ICU stays 이므로 MIMIC-IV 의 "ICU within 24h" outcome 을
    그대로 적용 불가. 우리는 admission diagnosis text 의 cluster 로 stratum
    을 정의:
       CRITICAL: emergency admit OR critical Dx (cardiac arrest, shock,
                 STEMI, ICH, septic shock)
       HIGH:     trauma, stroke, sepsis (non-shock), respiratory failure
       MODERATE: pneumonia, COPD/CHF exacerbation, GI bleed
       LOW:      monitoring / routine post-op / other

Outcome label Y (decision-time leakage-free):
    Y = hospital mortality (Expired) — observed only after discharge

Decision-time features (Pass B = minimal R11.1 등가):
    age_bucket, adm_emerg, spec_idx (unitType), n_labs (first-6h count)

Leakage suspect features (Pass A = full R10.4 등가):
    + charlson_index (from pastHistory comorbidity count)
    + n_vital_flags  (always 0 — eICU 의 vital 은 nurseCharting 에 있고
                       해당 파일은 다운로드 범위 밖이므로 missing)
    + specialty_baseline_rate (cohort-level mortality rate per unitType)
    + apache_predicted_mortality   (가장 강력한 leakage — APACHE 가 first 24h
                                      data 사용, decision-time 가 아님)
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
EICU_DIR = ROOT / "data" / "raw" / "eicu-crd"
OUT_JSONL = ROOT / "data" / "raw" / "eicu_cases_v11.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Stratum mapping — admission diagnosis text 기반 (decision-time)
# ─────────────────────────────────────────────────────────────────────────────

CRITICAL_TOKENS = [
    "arrest", "shock", "stemi", "intracranial hemorrhage", "ich",
    "septic shock", "respiratory arrest", "cardiac arrest",
    "massive", "acute mi", "myocardial infarction",
]
HIGH_TOKENS = [
    "stroke", "cva", "trauma", "sepsis", "respiratory failure",
    "pulmonary embolism", "aortic dissection", "hemorrhage",
    "subarachnoid", "status epilepticus", "diabetic ketoacidosis", "dka",
]
MODERATE_TOKENS = [
    "pneumonia", "copd", "chf", "exacerbation", "asthma", "gi bleed",
    "gastrointestinal bleeding", "pancreatitis", "cellulitis",
    "uti", "urinary tract infection", "delirium",
]


def classify_stratum(admit_dx_text: str) -> str:
    t = (admit_dx_text or "").lower()
    for tok in CRITICAL_TOKENS:
        if tok in t: return "CRITICAL"
    for tok in HIGH_TOKENS:
        if tok in t: return "HIGH"
    for tok in MODERATE_TOKENS:
        if tok in t: return "MODERATE"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_age(raw) -> tuple[str, int]:
    """eICU age 는 string ('> 89', '52' 등) — bucket + int 둘 다 return."""
    if raw is None or pd.isna(raw): return "unknown", -1
    s = str(raw).strip()
    if ">" in s or "89+" in s.replace(" ", ""): return "80+", 89
    try:
        v = int(float(s))
    except ValueError:
        return "unknown", -1
    if v < 18:   return "<18", v
    if v < 35:   return "18-34", v
    if v < 50:   return "35-49", v
    if v < 65:   return "50-64", v
    if v < 80:   return "65-79", v
    return "80+", v


def normalize_unit_type(unit_type: str) -> str:
    """eICU unitType → MIMIC-IV specialty 와 비슷한 normalize."""
    t = (unit_type or "").lower()
    if "cardiac" in t or "ccu" in t or "ctsicu" in t: return "cardiology"
    if "cticu" in t or "ctsicu" in t: return "cardiothoracic_surgery"
    if "neuro" in t: return "neurology"
    if "sicu" in t or "surgical" in t: return "surgery"
    if "micu" in t or "medical" in t: return "internal_medicine"
    if "obstetr" in t or "ob/" in t: return "obstetrics"
    if "psych" in t: return "psychiatry"
    if "ped" in t or "nicu" in t or "picu" in t: return "pediatrics"
    return "internal_medicine"  # default fallback


def _read_csv_gz(path: Path, usecols=None, low_memory=False) -> pd.DataFrame:
    """gzip CSV 를 pandas DataFrame 으로 robust 하게 load."""
    if not path.exists():
        print(f"  ⚠ missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, compression="gzip", usecols=usecols,
                       low_memory=low_memory)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_cases(eicu_dir: Path, n_target: int | None = None,
                seed: int = 42) -> list[dict]:
    print(f"[R11.3 preprocess] eICU dir: {eicu_dir}")

    # 1. patient.csv — primary table
    patient_cols = [
        "patientUnitStayID", "patienthealthsystemstayid", "uniquepid",
        "gender", "age", "ethnicity", "hospitalAdmitSource",
        "hospitalDischargeStatus", "hospitalDischargeOffset",
        "unitDischargeStatus", "unitType", "apacheAdmissionDx",
        "unitAdmitTime24", "unitAdmitSource",
    ]
    # eICU column naming 은 종종 case-inconsistent — fallback 로 모든 cols 읽기
    pt = _read_csv_gz(eicu_dir / "patient.csv.gz")
    if pt.empty:
        sys.exit(f"patient.csv.gz missing or empty: {eicu_dir}")
    print(f"  patient.csv: {len(pt)} rows, {len(pt.columns)} cols")

    # column normalize (case-insensitive resolve)
    pt.columns = [c.strip() for c in pt.columns]
    def col(df, name):
        for c in df.columns:
            if c.lower() == name.lower(): return c
        return None
    psid = col(pt, "patientUnitStayID") or col(pt, "patientunitstayid")
    uid_col = col(pt, "uniquepid")
    hd_status_col = col(pt, "hospitalDischargeStatus")
    has_src_col = col(pt, "hospitalAdmitSource")
    apache_dx_col = col(pt, "apacheAdmissionDx")
    age_col = col(pt, "age")
    gender_col = col(pt, "gender")
    eth_col = col(pt, "ethnicity")
    unit_type_col = col(pt, "unitType")

    # outcome label
    pt["_y_mortality"] = pt[hd_status_col].astype(str).str.lower().str.contains("expired", na=False)

    # 2. admissionDx — Dx text per stay (decision-time)
    adx = _read_csv_gz(eicu_dir / "admissionDx.csv.gz")
    adx_by_stay = {}
    if not adx.empty:
        adx.columns = [c.strip() for c in adx.columns]
        sid_col = col(adx, "patientUnitStayID")
        name_col = col(adx, "admitDxName") or col(adx, "admitdxname")
        if sid_col and name_col:
            for stay_id, group in adx.groupby(sid_col):
                names = group[name_col].dropna().astype(str).tolist()
                adx_by_stay[stay_id] = " | ".join(names[:5])
    print(f"  admissionDx grouped: {len(adx_by_stay)} stays")

    # 3. pastHistory — comorbidity count per stay (Charlson-like)
    ph = _read_csv_gz(eicu_dir / "pastHistory.csv.gz")
    charlson_by_stay = {}
    if not ph.empty:
        ph.columns = [c.strip() for c in ph.columns]
        sid_col = col(ph, "patientUnitStayID")
        val_col = col(ph, "pasthistoryvalue") or col(ph, "pastHistoryValue")
        if sid_col and val_col:
            for stay_id, group in ph.groupby(sid_col):
                charlson_by_stay[stay_id] = int(group[val_col].dropna().nunique())
    print(f"  pastHistory grouped: {len(charlson_by_stay)} stays")

    # 4. lab.csv — first-6h lab count (decision-time minimal feature)
    lab = _read_csv_gz(eicu_dir / "lab.csv.gz")
    nlabs_by_stay = {}
    if not lab.empty:
        lab.columns = [c.strip() for c in lab.columns]
        sid_col = col(lab, "patientUnitStayID")
        off_col = col(lab, "labResultOffset") or col(lab, "labresultoffset")
        if sid_col and off_col:
            early = lab[lab[off_col].fillna(99999).astype(float) <= 360.0]
            for stay_id, group in early.groupby(sid_col):
                nlabs_by_stay[stay_id] = int(len(group))
    print(f"  first-6h lab counts: {len(nlabs_by_stay)} stays")

    # 5. apachePatientResult — predicted mortality (post-decision leakage)
    ar = _read_csv_gz(eicu_dir / "apachePatientResult.csv.gz")
    apache_by_stay = {}
    if not ar.empty:
        ar.columns = [c.strip() for c in ar.columns]
        sid_col = col(ar, "patientUnitStayID")
        score_col = col(ar, "apacheScore") or col(ar, "apachescore")
        pred_col = col(ar, "predictedHospitalMortality") or col(ar, "predictedhospitalmortality")
        if sid_col:
            for _, row in ar.iterrows():
                stay_id = row.get(sid_col)
                apache_by_stay[stay_id] = {
                    "apacheScore": float(row.get(score_col, 0) or 0) if score_col else 0,
                    "predictedHospitalMortality": float(row.get(pred_col, 0) or 0) if pred_col else 0,
                }
    print(f"  APACHE per-stay: {len(apache_by_stay)}")

    # 6. Build cases
    print(f"\n[R11.3 preprocess] building case dicts")
    # cohort-level baseline mortality per unitType (specialty_baseline_rate 등가)
    unit_baseline = pt.groupby(unit_type_col)["_y_mortality"].mean().to_dict()

    cases = []
    for _, row in pt.iterrows():
        stay_id = row[psid]
        subject_id = row[uid_col]
        admit_source = str(row[has_src_col] or "").lower()
        is_emerg = ("emerg" in admit_source) or ("ed" == admit_source.strip()) \
                   or ("emergency department" in admit_source)
        apache_admit_dx = str(row[apache_dx_col] or "")
        admit_dx_text = adx_by_stay.get(stay_id, "") or apache_admit_dx
        stratum = classify_stratum(admit_dx_text + " " + apache_admit_dx)
        y_mort = bool(row["_y_mortality"])
        age_bkt, age_int = parse_age(row[age_col])
        gender = str(row[gender_col] or "")
        ethnicity = str(row[eth_col] or "")
        specialty = normalize_unit_type(row[unit_type_col])
        apache = apache_by_stay.get(stay_id, {})
        charlson = charlson_by_stay.get(stay_id, 0)
        n_labs = nlabs_by_stay.get(stay_id, 0)
        unit_type_str = str(row[unit_type_col] or "")
        spec_rate = float(unit_baseline.get(row[unit_type_col], 0.0))

        case = {
            "hadm_id": str(stay_id),
            "subject_id": str(subject_id),
            "stratum": stratum,
            "risk_group": stratum,
            "expected_escalate": y_mort,
            "y_outcome": y_mort,
            "specialty": specialty,
            "admission_type": "EMERGENCY" if is_emerg else "ELECTIVE",
            "admit_year": 2014,  # eICU collection era
            "demographics": {
                "sex": "F" if gender.startswith("F") else "M",
                "race": ethnicity, "age_bucket": age_bkt,
            },
            "outcome": {
                "icu_within_24h": True,   # eICU = all are ICU stays
                "in_hospital_mortality": y_mort,
                "deterioration_composite": y_mort,
                "sepsis": "sepsis" in admit_dx_text.lower(),
                "readmit_30d": False,
                "transfusion_24h": False,
            },
            "structured": {
                "early_lab_flags": [],   # eICU 의 lab flag 매핑 — simplified
                "early_vital_quartiles": [],
                "service": unit_type_str,
            },
            "_charlson_index": charlson,
            "_n_vital_flags": 0,
            "_specialty_baseline_rate": spec_rate,
            "_apache_score": apache.get("apacheScore", 0),
            "_apache_predicted_mort": apache.get("predictedHospitalMortality", 0),
            "_n_labs_6h": n_labs,
            "_audit_postoutcome": {
                "icd_primary": "", "icd_codes": [],
                "lab_flags_full": [], "los_days": 0,
            },
            "note_text": None,
        }
        # meta_info string (round11_method_agnostic_minimal 의 parser 와 호환)
        case["meta_info"] = (
            f"hadm_id={case['hadm_id']} subject_id={case['subject_id']} "
            f"age_bucket={age_bkt} adm_type={case['admission_type']} "
            f"labs={','.join([f'lab{i}' for i in range(n_labs)])}"
        )
        # question 도 minimal placeholder
        case["question"] = (
            f"Patient summary: {age_bkt} {gender} admitted via {admit_source}. "
            f"Admission Dx: {admit_dx_text or apache_admit_dx}. "
            f"Specialty: {specialty}."
        )
        cases.append(case)

    print(f"  → {len(cases)} cases built")

    # Stratify + sample to n_target if requested
    if n_target and n_target < len(cases):
        import random
        rng = random.Random(seed)
        by_st = defaultdict(list)
        for c in cases: by_st[c["stratum"]].append(c)
        n_per = n_target // 4
        sampled = []
        for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            pool = by_st[s]
            rng.shuffle(pool)
            sampled.extend(pool[:n_per])
        print(f"  → sampled {len(sampled)} ({n_per}/stratum)")
        return sampled

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-dir", type=Path, default=EICU_DIR)
    ap.add_argument("--out", type=Path, default=OUT_JSONL)
    ap.add_argument("--n", type=int, default=0,
                    help="0 = all available, else stratified sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cases = build_cases(args.eicu_dir, n_target=args.n or None, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for c in cases:
            f.write(json.dumps(c, default=str) + "\n")

    # Stratum distribution summary
    from collections import Counter
    dist = Counter(c["stratum"] for c in cases)
    y_dist = {s: sum(1 for c in cases if c["stratum"] == s and c["expected_escalate"])
              for s in dist}
    print(f"\n✅ {args.out} ({len(cases)} cases)")
    print(f"  Stratum distribution:")
    for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        n = dist.get(s, 0); p = y_dist.get(s, 0)
        print(f"    {s}: n={n}, positive={p} ({100*p/n if n else 0:.1f}%)")

    summary_path = args.out.parent / "eicu_preprocess_summary.json"
    summary_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "n_cases": len(cases),
        "stratum_dist": dict(dist),
        "positive_per_stratum": y_dist,
        "out_jsonl": str(args.out),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
