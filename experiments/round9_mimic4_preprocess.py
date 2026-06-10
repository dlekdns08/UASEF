"""
Round 9 — MIMIC-IV preprocessing (hosp + icu modules → MedQACase JSONL)
══════════════════════════════════════════════════════════════════════════════

입력  : $MIMIC4_DIR/{hosp,icu}/*.csv.gz   (PhysioNet credentialed)
출력  : data/raw/mimic-iv/mimic4_cases.jsonl

⚠️ LEAKAGE-SAFE 재설계 (REVISION_PLAN P0-1).
   decision-time 위험군 G(X_t0) 과 미래 outcome Y 를 **완전히 분리**한다.
   (이전 버전은 σ 가 미래 outcome 으로 정의되고 Y=σ∈{CRIT,HIGH} 로 유도되어
    label leakage 였음 — 더 이상 사용 금지.)

  G(X_t0) = risk_group, **입원 시점에 알 수 있는 정보만** 사용 (→ stratum, per-stratum α):
      emergency = admission_type ∈ {EMERGENCY, URGENT, DIRECT EMER., EW EMER.}
      elderly   = anchor_age ≥ 80
      early_*   = 입원 후 **첫 6h** 이내 charttime 의 lab abnormality (decision-time 가용)
      CRITICAL  = emergency ∧ (early_severe ∨ elderly)
      HIGH      = emergency ∨ early_severe
      MODERATE  = early_any ∨ elderly
      LOW       = 그 외

  Y = expected_escalate, **미래 adverse outcome** (decision 이후에만 관측, G 와 독립):
      Y = ICU transfer within 24h ∨ in-hospital mortality   (deterioration composite)

  ⚠️ 누수 방지: los_days / discharge ICD / 전체-입원 lab / 사망 / 재입원 은
     prompt 입력에서 제외하고 `_audit_postoutcome` 블록에만 보관(평가용).

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
    # anchor_year_group: 실 calendar era ("2008 - 2010" 등). R9.4 temporal split 의 ground truth.
    anchor_year_group_map = (
        dict(zip(pat["subject_id"], pat["anchor_year_group"]))
        if "anchor_year_group" in pat.columns else {}
    )

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

    # chunked read of labevents — charttime 으로 decision-time(첫 6h) 창 분리.
    #   early_lab_flags_map : charttime ≤ admittime + 6h  → G(X_t0) & prompt (누수 없음)
    #   lab_flags_map       : 전체 입원 창               → _audit 전용 (prompt 미사용)
    EARLY_WINDOW_H = 6.0
    early_lab_flags_map: dict[int, list[str]] = defaultdict(list)
    lab_path = hosp / "labevents.csv.gz"
    chunk_iter = pd.read_csv(
        lab_path, compression="gzip", chunksize=2_000_000,
        usecols=["hadm_id", "itemid", "valuenum", "charttime"],
        parse_dates=["charttime"],
    )
    for ci, chunk in enumerate(chunk_iter):
        chunk = chunk[chunk["itemid"].isin(target_itemids)]
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        for hid, iid, val, ct in zip(chunk["hadm_id"], chunk["itemid"],
                                     chunk["valuenum"], chunk["charttime"]):
            th, direction, fname = target_labs[int(iid)]
            abnormal = (direction == "high" and val > th) or (direction == "low" and val < th)
            if not abnormal:
                continue
            hid_i = int(hid)
            if fname not in lab_flags_map[hid_i]:
                lab_flags_map[hid_i].append(fname)
            # decision-time 가용 여부: charttime 이 admittime + 6h 이내인가
            admit_t = adm_intime.get(hid)
            if admit_t is not None and ct is not None:
                try:
                    dt_h = (ct - admit_t).total_seconds() / 3600.0
                except (TypeError, ValueError):
                    dt_h = None
                if dt_h is not None and -1.0 <= dt_h <= EARLY_WINDOW_H:
                    if fname not in early_lab_flags_map[hid_i]:
                        early_lab_flags_map[hid_i].append(fname)
        if verbose and ci % 5 == 0:
            print(f"      labevents chunk {ci} processed (covered hadm_ids: {len(lab_flags_map)})")

    # ── G(X_t0) risk-group + Y outcome (LEAKAGE-SAFE) ──────────────────────────
    # G 는 decision-time 정보만, Y 는 미래 outcome 만. 둘은 독립이며 Y 가 G 로부터
    # 유도되지 않는다 (이전 expected_escalate = stratum∈{CRIT,HIGH} 누수 제거).
    EARLY_SEVERE_FLAGS = {"lactate_high", "acidemia", "hyperkalemia", "leukocytosis"}
    if verbose: print(f"\n[risk_group] Classifying admissions (decision-time only) ...")
    rows_out = []
    for _, a in adm.iterrows():
        hid = int(a["hadm_id"])
        sid = int(a["subject_id"])
        admit_type = str(a.get("admission_type", "")).upper()
        age = age_map.get(sid)

        # ── decision-time 신호 (입원 시점/첫 6h 가용) ──
        emergency_admit = admit_type in {"EMERGENCY", "URGENT", "DIRECT EMER.", "EW EMER."}
        elderly = (age is not None) and (float(age) >= 80)
        early_flags = early_lab_flags_map.get(hid, [])
        early_severe = any(f in EARLY_SEVERE_FLAGS for f in early_flags)
        early_any = len(early_flags) > 0

        # G(X_t0): risk_group (→ stratum, per-stratum α 의 조건화 변수)
        if emergency_admit and (early_severe or elderly):
            risk_group = "CRITICAL"
        elif emergency_admit or early_severe:
            risk_group = "HIGH"
        elif early_any or elderly:
            risk_group = "MODERATE"
        else:
            risk_group = "LOW"

        # ── 미래 outcome (decision 이후에만 관측) ──
        died = bool(a.get("hospital_expire_flag", 0))
        los_h = float(a.get("los_hours", 0))
        in_icu24 = hid in icu_within_24h
        sepsis = "lactate_high" in lab_flags_map.get(hid, [])   # full-window (audit only)
        has_readmit = hid in readmit_30d

        # Y = expected_escalate: deterioration composite (ICU<24h ∨ in-hospital death).
        # G 와 독립 — escalation 이 사후적으로 정당했는지의 ground truth.
        y_outcome = bool(in_icu24 or died)

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
            "stratum":  risk_group,            # = G(X_t0), decision-time
            "risk_group": risk_group,
            "expected_escalate": y_outcome,    # = Y, future outcome (G 와 독립)
            "y_outcome": y_outcome,
            "specialty": specialty,
            "admission_type": admit_type or "ELECTIVE",
            "admit_year": admit_year,
            "anchor_year_group": anchor_year_group_map.get(sid),
            "demographics": {
                "sex": sex, "race": str(a.get("race", "UNKNOWN")),
                "age_bucket": _age_bucket(age),
            },
            "outcome": {
                "icu_within_24h": in_icu24,
                "in_hospital_mortality": died,
                "deterioration_composite": y_outcome,
                "sepsis": sepsis,
                "readmit_30d": has_readmit,
                "transfusion_24h": False,  # not extracted in Phase 1; reserved for extension
            },
            # prompt 입력용 — decision-time 가용 필드만.
            "structured": {
                "early_lab_flags": sorted(early_flags),
                "early_vital_quartiles": [],   # reserved for first-6h chartevents extension
                "service":  svc_code or "MED",
            },
            # 평가 전용 — prompt 로 절대 전달 금지 (미래/사후 정보).
            "_audit_postoutcome": {
                "icd_primary": icd_primary_map.get(hid, "unknown"),
                "icd_codes":   icd_codes_map.get(hid, []),
                "lab_flags_full": sorted(lab_flags_map.get(hid, [])),
                "los_days": round(los_h / 24, 2),
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
