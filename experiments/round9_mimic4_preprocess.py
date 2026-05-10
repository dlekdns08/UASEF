"""
Round 9 — MIMIC-IV preprocessing (hosp + icu modules → MedQACase JSONL)
══════════════════════════════════════════════════════════════════════════════

입력  : $MIMIC4_DIR/{hosp,icu}/*.csv.gz   (PhysioNet credentialed)
출력  : data/raw/mimic-iv/mimic4_cases.jsonl

라벨 정의 (improvements/round9_PLAN.md §2.1):
  CRITICAL = ICU<24h ∨ in-hospital mortality ∨ admission_type∈{EMERGENCY,URGENT}
  HIGH     = sepsis-3 (SOFA Δ≥2 within 48h, proxy: lactate>2 + abx start)
             ∨ 30-day readmission ∨ blood transfusion within 24h
  MODERATE = standard inpatient (no ICU, no death)
  LOW      = LOS<24h, discharged home

CSV.gz chunked read 로 메모리 < 4 GB. CPU bound, ~2h on M-series Mac.

⚠️ DUA: 본 스크립트가 산출하는 JSONL 은 derived structured features 만 담고
free-text note 는 포함하지 않으며 hadm_id 는 보존됨 (re-identification 가능).
이 JSONL 은 **commit 금지** — .gitignore 에 data/raw/mimic-iv/ 차단됨.

Usage
-----
    python experiments/round9_mimic4_preprocess.py \\
        --mimic-dir $MIMIC4_DIR \\
        --output data/raw/mimic-iv/mimic4_cases.jsonl \\
        --n-per-stratum 1500 --seed 42

    # quick replication mode (총 400 case, ~5 min)
    python experiments/round9_mimic4_preprocess.py \\
        --mimic-dir $MIMIC4_DIR --n-per-stratum 100
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── service code → specialty (MIMIC-IV `services.curr_service` 기준) ─────────
SERVICE_TO_SPECIALTY = {
    "MED": "internal_medicine",
    "CMED": "cardiology",
    "NMED": "neurology",
    "OMED": "oncology",
    "GU": "internal_medicine",
    "GYN": "obstetrics",
    "OBS": "obstetrics",
    "ORTHO": "surgery",
    "PSURG": "surgery",
    "SURG": "surgery",
    "TSURG": "cardiothoracic_surgery",
    "VSURG": "surgery",
    "CSURG": "cardiothoracic_surgery",
    "NSURG": "surgery",
    "TRAUM": "trauma_surgery",
    "ENT": "surgery",
    "DENT": "surgery",
    "PSYCH": "psychiatry",
    "NB": "pediatrics",
    "NBB": "pediatrics",
    "OBSTETRICS": "obstetrics",
}

ICU_CARE_UNITS_INDICATING_CRITICAL = {
    "Coronary Care Unit (CCU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical Intensive Care Unit (MICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Neuro Intermediate",
    "Neuro Stepdown",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Trauma SICU (TSICU)",
}

# Lab abnormality flags (MIMIC-IV `d_labitems.itemid` 매핑은 v3.1 기준)
# 우리는 itemid 가 아니라 `label` 매칭으로 robust 하게 처리 (~ICU에서 standard)
LAB_ABNORMAL_THRESHOLDS = {
    "lactate":      (2.0,   "high",  "lactate_high"),       # > 2.0 mmol/L → sepsis suspicion
    "creatinine":   (1.5,   "high",  "creatinine_high"),
    "troponin t":   (0.04,  "high",  "troponin_high"),
    "lactate dehydrogenase (ld)": (618, "high", "ldh_high"),
    "white blood cells": (12.0, "high", "leukocytosis"),
    "wbc":          (12.0,  "high",  "leukocytosis"),
    "platelet count": (100, "low",   "thrombocytopenia"),
    "potassium":    (5.5,   "high",  "hyperkalemia"),
    "sodium":       (135,   "low",   "hyponatremia"),
    "ph":           (7.30,  "low",   "acidemia"),
    "bicarbonate":  (18,    "low",   "low_bicarb"),
}


# ── lazy pandas import ────────────────────────────────────────────────────────
def _import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise SystemExit(
            "pandas 가 필요합니다. .venv/bin/pip install pandas 로 설치하세요."
        ) from e


# ── helpers ───────────────────────────────────────────────────────────────────

def _age_bucket(age: float | int | None) -> str:
    if age is None:
        return "unknown"
    if age < 18:
        return "<18"
    if age < 35:
        return "18-34"
    if age < 50:
        return "35-49"
    if age < 65:
        return "50-64"
    if age < 80:
        return "65-79"
    return "80+"


def _read_csv(path: Path, pd, **kw):
    if not path.exists():
        raise FileNotFoundError(f"필수 파일 누락: {path}")
    return pd.read_csv(path, compression="gzip", **kw)


def _normalize_specialty(svc: str | None) -> str:
    if not svc:
        return "internal_medicine"
    return SERVICE_TO_SPECIALTY.get(svc.strip().upper(), "internal_medicine")


# ── main pipeline ─────────────────────────────────────────────────────────────

def build_cases(mimic_dir: Path, n_per_stratum: int, seed: int, verbose: bool = True):
    pd = _import_pandas()

    hosp = mimic_dir / "hosp"
    icu  = mimic_dir / "icu"

    if verbose: print(f"[1/6] Loading hosp/admissions.csv.gz ...")
    adm = _read_csv(hosp / "admissions.csv.gz", pd, parse_dates=["admittime", "dischtime"])
    if "hospital_expire_flag" not in adm.columns:
        adm["hospital_expire_flag"] = 0
    adm["los_hours"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 3600

    if verbose: print(f"[2/6] Loading hosp/patients.csv.gz ...")
    pat = _read_csv(hosp / "patients.csv.gz", pd)
    # MIMIC-IV: anchor_age 가 anchor_year 기준의 age — 그대로 age proxy 로 사용
    age_map = dict(zip(pat["subject_id"], pat["anchor_age"]))
    sex_map = dict(zip(pat["subject_id"], pat["gender"]))

    if verbose: print(f"[3/6] Loading icu/icustays.csv.gz ...")
    icustays = _read_csv(icu / "icustays.csv.gz", pd, parse_dates=["intime"])
    # CRITICAL: icustay intime ≤ admission admittime + 24h
    adm_intime = dict(zip(adm["hadm_id"], adm["admittime"]))
    icu_within_24h: set[int] = set()
    for _, row in icustays.iterrows():
        a = adm_intime.get(row["hadm_id"])
        if a is None:
            continue
        if (row["intime"] - a).total_seconds() <= 24 * 3600:
            icu_within_24h.add(int(row["hadm_id"]))

    if verbose: print(f"[4/6] Loading hosp/services.csv.gz ...")
    svc_df = _read_csv(hosp / "services.csv.gz", pd, parse_dates=["transfertime"])
    # 첫 service 만 대표값으로 (admission 시작 시점 specialty)
    svc_df = svc_df.sort_values(["hadm_id", "transfertime"])
    svc_first = svc_df.drop_duplicates("hadm_id", keep="first")
    svc_map = dict(zip(svc_first["hadm_id"], svc_first["curr_service"]))

    if verbose: print(f"[5/6] Loading hosp/diagnoses_icd.csv.gz ...")
    dx = _read_csv(hosp / "diagnoses_icd.csv.gz", pd)
    # primary diagnosis (seq_num=1) 와 active codes 분리
    dx_primary = dx[dx["seq_num"] == 1].drop_duplicates("hadm_id", keep="first")
    icd_primary_map = dict(zip(dx_primary["hadm_id"], dx_primary["icd_code"].astype(str)))
    icd_codes_map: dict[int, list[str]] = defaultdict(list)
    for hid, code in zip(dx["hadm_id"], dx["icd_code"].astype(str)):
        if len(icd_codes_map[hid]) < 5:  # cap at 5 codes
            icd_codes_map[hid].append(code)

    # 30-day readmission per subject
    readmit_30d: set[int] = set()
    by_subj = adm.sort_values(["subject_id", "admittime"])
    for subj, grp in by_subj.groupby("subject_id"):
        grp = grp.sort_values("admittime")
        times = grp["admittime"].tolist()
        hadm_ids = grp["hadm_id"].tolist()
        for i in range(len(times) - 1):
            gap = (times[i + 1] - grp.iloc[i]["dischtime"]).total_seconds() / (24 * 3600)
            if 0 < gap <= 30:
                readmit_30d.add(int(hadm_ids[i]))

    if verbose: print(f"[6/6] Sampling labevents (lactate/creatinine/etc.) ...")
    lab_flags_map: dict[int, list[str]] = defaultdict(list)
    # labevents 는 ~수억 row → chunked + filtered read.
    # d_labitems 에서 label 매칭하는 itemid 만 골라서 labevents 1차 필터.
    d_lab = _read_csv(hosp / "d_labitems.csv.gz", pd)
    d_lab["label_l"] = d_lab["label"].astype(str).str.lower()
    target_labs = {}  # itemid -> (threshold, direction, flag_name)
    for label_pat, (th, direction, fname) in LAB_ABNORMAL_THRESHOLDS.items():
        rows = d_lab[d_lab["label_l"] == label_pat]
        for _, r in rows.iterrows():
            target_labs[int(r["itemid"])] = (th, direction, fname)
    target_itemids = set(target_labs.keys())
    if verbose: print(f"      target lab itemids: {len(target_itemids)}")

    # chunked read of labevents
    lab_path = hosp / "labevents.csv.gz"
    chunk_iter = pd.read_csv(
        lab_path, compression="gzip", chunksize=2_000_000,
        usecols=["hadm_id", "itemid", "valuenum"],
    )
    for ci, chunk in enumerate(chunk_iter):
        chunk = chunk[chunk["itemid"].isin(target_itemids)]
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        for hid, iid, val in zip(chunk["hadm_id"], chunk["itemid"], chunk["valuenum"]):
            th, direction, fname = target_labs[int(iid)]
            if direction == "high" and val > th:
                if fname not in lab_flags_map[int(hid)]:
                    lab_flags_map[int(hid)].append(fname)
            elif direction == "low" and val < th:
                if fname not in lab_flags_map[int(hid)]:
                    lab_flags_map[int(hid)].append(fname)
        if verbose and ci % 5 == 0:
            print(f"      labevents chunk {ci} processed (covered hadm_ids: {len(lab_flags_map)})")

    # ── stratum classification ────────────────────────────────────────────────
    if verbose: print(f"\n[stratum] Classifying admissions ...")
    rows_out = []
    for _, a in adm.iterrows():
        hid = int(a["hadm_id"])
        sid = int(a["subject_id"])
        admit_type = str(a.get("admission_type", "")).upper()
        died = bool(a.get("hospital_expire_flag", 0))
        los_h = float(a.get("los_hours", 0))
        in_icu24 = hid in icu_within_24h
        emergency_admit = admit_type in {"EMERGENCY", "URGENT", "DIRECT EMER.", "EW EMER."}

        # sepsis proxy: lactate_high
        sepsis = "lactate_high" in lab_flags_map.get(hid, [])
        has_readmit = hid in readmit_30d

        if in_icu24 or died or emergency_admit:
            stratum = "CRITICAL"
        elif sepsis or has_readmit:
            stratum = "HIGH"
        elif los_h < 24:
            stratum = "LOW"
        else:
            stratum = "MODERATE"

        # expected_escalate: stratum != LOW 인 case 중 실제 escalation 발생
        expected_escalate = (stratum in ("CRITICAL", "HIGH"))

        age = age_map.get(sid)
        sex = sex_map.get(sid, "?")
        admit_year = a["admittime"].year if isinstance(a["admittime"], (datetime,)) else None
        if admit_year is None:
            try: admit_year = int(str(a["admittime"])[:4])
            except Exception: admit_year = 0

        svc_code = svc_map.get(hid)
        specialty = _normalize_specialty(svc_code)

        rows_out.append({
            "hadm_id":  str(hid),
            "subject_id": str(sid),
            "stratum":  stratum,
            "expected_escalate": expected_escalate,
            "specialty": specialty,
            "admission_type": admit_type or "ELECTIVE",
            "admit_year": admit_year,
            "demographics": {
                "sex": sex, "race": str(a.get("race", "UNKNOWN")),
                "age_bucket": _age_bucket(age),
            },
            "outcome": {
                "icu_within_24h": in_icu24,
                "in_hospital_mortality": died,
                "sepsis": sepsis,
                "readmit_30d": has_readmit,
                "transfusion_24h": False,  # not extracted in Phase 1; reserved for extension
            },
            "structured": {
                "icd_primary": icd_primary_map.get(hid, "unknown"),
                "icd_codes":   icd_codes_map.get(hid, []),
                "lab_flags":   sorted(lab_flags_map.get(hid, [])),
                "vital_quartiles": [],   # reserved for chartevents extension
                "los_days": round(los_h / 24, 2),
                "service":  svc_code or "MED",
            },
            "note_text": None,
        })

    # stratum-balanced sampling
    by_stratum = defaultdict(list)
    for r in rows_out:
        by_stratum[r["stratum"]].append(r)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        rng.shuffle(by_stratum[s])
        sampled.extend(by_stratum[s][:n_per_stratum])
        if verbose:
            esc = sum(1 for r in by_stratum[s][:n_per_stratum] if r["expected_escalate"])
            print(f"  {s:<9}: 전체 {len(by_stratum[s]):>7}, 샘플 {len(by_stratum[s][:n_per_stratum]):>5} (escalate={esc})")
    rng.shuffle(sampled)
    return sampled


def main():
    ap = argparse.ArgumentParser(description="Round 9 MIMIC-IV preprocessing")
    ap.add_argument("--mimic-dir", type=Path,
                    default=Path(os.environ.get("MIMIC4_DIR", "")),
                    help="$MIMIC4_DIR (compressed CSV root with hosp/ and icu/ subdirs)")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases.jsonl")
    ap.add_argument("--n-per-stratum", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.mimic_dir or not args.mimic_dir.exists():
        sys.exit(f"--mimic-dir 누락 또는 미존재: {args.mimic_dir}\n"
                 f"  $MIMIC4_DIR 환경변수 또는 --mimic-dir 인자로 지정하세요.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = build_cases(args.mimic_dir, args.n_per_stratum, args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {len(rows)} cases → {args.output}")
    print(f"   ⚠️ DUA: 본 파일은 commit 하지 마세요. .gitignore 에 차단되어 있습니다.")


if __name__ == "__main__":
    main()
