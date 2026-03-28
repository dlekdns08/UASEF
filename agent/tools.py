"""
UASEF Agent — 의료 도구 모음 (Mock 구현)

실제 연구에서는 아래 소스로 교체:
  - drug_interaction_checker  → Drugs@FDA API / RxNorm / Lexicomp
  - clinical_guideline_search → UpToDate API / PubMed E-utilities
  - lab_reference_lookup      → LOINC / 기관 내 LIS 연동
  - differential_diagnosis    → Isabel DDx / 기관 내 CDR 시스템

4개 도구 모두 str 파라미터만 사용 — LLM tool-calling 안정성 최대화.
"""

from langchain_core.tools import tool


# ── 약물 상호작용 DB ──────────────────────────────────────────────────────────

_INTERACTIONS: dict[frozenset, str] = {
    frozenset(["warfarin", "aspirin"]):
        "HIGH: 출혈 위험 증가. INR 밀접 모니터링 필요.",
    frozenset(["warfarin", "ibuprofen"]):
        "HIGH: NSAIDs가 항응고 효과 증강. 출혈 주의.",
    frozenset(["warfarin", "amiodarone"]):
        "HIGH: Amiodarone이 warfarin 효과를 현저히 증가. 용량 50% 감량 고려.",
    frozenset(["metformin", "iv_contrast"]):
        "MODERATE: 조영제 투여 전 48h 중단. CKD에서 유산산증 위험.",
    frozenset(["lisinopril", "potassium"]):
        "MODERATE: 고칼륨혈증 위험 — CKD·당뇨 환자에서 특히 주의.",
    frozenset(["digoxin", "amiodarone"]):
        "HIGH: Amiodarone이 digoxin 혈중농도 상승. digoxin 용량 50% 감량.",
    frozenset(["ssri", "maoi"]):
        "CONTRAINDICATED: 세로토닌 증후군. 최소 14일 washout 필요.",
    frozenset(["furosemide", "aminoglycoside"]):
        "HIGH: 이독성(ototoxicity) 위험 상승. 청력 모니터링.",
    frozenset(["tacrolimus", "fluconazole"]):
        "HIGH: CYP3A4 억제로 tacrolimus 독성. 혈중농도 모니터링.",
    frozenset(["methotrexate", "nsaid"]):
        "HIGH: NSAIDs가 methotrexate 배설 억제 — 골수억제 위험.",
}


# ── 임상 가이드라인 DB ───────────────────────────────────────────────────────

_GUIDELINES: dict[str, str] = {
    "stemi": (
        "[AHA/ACC 2022 STEMI] Primary PCI: door-to-balloon ≤90 min. "
        "Dual antiplatelet: Aspirin 325mg + P2Y12 inhibitor (ticagrelor/prasugrel 선호). "
        "UFH or bivalirudin 항응고. 성공적 PCI 후 12개월 DAPT 유지."
    ),
    "nstemi": (
        "[AHA/ACC 2021 NSTEMI] GRACE 점수 계산. 고위험: 24h 이내 침습적 전략. "
        "ASA + P2Y12 + anticoagulation. 중위험: 72h 이내 invasive strategy."
    ),
    "sepsis": (
        "[Surviving Sepsis 2021 1h Bundle] ① 혈액배양 2세트, ② 광범위 항생제, "
        "③ 저혈압 or 젖산 ≥4: 30mL/kg 결정질액, ④ MAP <65 지속: 승압제(norepinephrine 1st), "
        "⑤ 젖산 재측정. 항생제 투여까지 goal ≤1h."
    ),
    "heart_failure": (
        "[AHA/ACC 2022 HF] HFrEF 4중 요법: ACEi/ARB/ARNI + β-차단제 + MRA + SGLT2i. "
        "LVEF ≤40: 모든 4가지 약제 목표. Loop diuretic으로 울혈 조절. "
        "ICD: LVEF ≤35 + NYHA II-III."
    ),
    "dm2_ckd": (
        "[ADA 2024 + KDIGO 2024] eGFR ≥30 + 단백뇨: SGLT2i 우선(신장 보호). "
        "eGFR <30: metformin 금기. GLP-1 RA: 심혈관 보호 2차 선택. "
        "HbA1c 목표: 고령/다중이환 <8.0%, 건강 노인 <7.5%."
    ),
    "copd": (
        "[GOLD 2024] 그룹 A: SABA prn. 그룹 B: LABA or LAMA 단일 기관지확장제. "
        "그룹 E: LABA+LAMA; ICS 추가 시 혈중 eosinophil ≥300. "
        "금연, 독감/폐렴구균 백신, 폐재활."
    ),
    "atrial_fibrillation": (
        "[AHA/ACC 2023 AF] 항응고: CHA₂DS₂-VASc ≥2(남)/≥3(여) → DOAC 우선(warfarin 대비 선호). "
        "rate control: β-차단제 or CCB. rhythm control: 젊은 환자·증상 심한 경우 고려. "
        "cardioversion 전 ≥3주 항응고 or TEE."
    ),
    "pneumonia_cap": (
        "[IDSA/ATS 2019 CAP] 외래: amoxicillin 1g tid or doxycycline or macrolide. "
        "입원 비-ICU: β-lactam + macrolide or 호흡기 FQ 단독. "
        "ICU: β-lactam + azithromycin or FQ. MRSA 위험: vancomycin or linezolid 추가."
    ),
    "osteoporosis": (
        "[NOF + AACE 2020] T-score ≤-2.5 or 취약성 골절: 치료 시작. "
        "1차: alendronate/zoledronate/risedronate. 심한 골감소 or 스테로이드 유발: "
        "teriparatide or denosumab. Ca 1000-1200mg/d + Vit D 800-1000IU/d."
    ),
    "ckd": (
        "[KDIGO 2024] BP: <120/80mmHg, ACEi/ARB 우선. SGLT2i: eGFR ≥20 + 단백뇨 >200mg/g 추가. "
        "단백뇨 감소 목표. GFR 급감시 투석 준비(AV fistula ≥6개월 전)."
    ),
}


# ── 검사 참고치 DB ────────────────────────────────────────────────────────────

_LAB_REFS: dict[str, dict] = {
    "hemoglobin":  {"남": "13.5–17.5 g/dL", "여": "12.0–15.5 g/dL"},
    "hgb":         {"남": "13.5–17.5 g/dL", "여": "12.0–15.5 g/dL"},
    "creatinine":  {"남": "0.74–1.35 mg/dL", "여": "0.59–1.04 mg/dL"},
    "egfr":        {"정상": ">60", "CKD_G3a": "45-59", "CKD_G3b": "30-44", "CKD_G4": "15-29", "CKD_G5": "<15", "단위": "mL/min/1.73m²"},
    "hba1c":       {"정상": "<5.7%", "전당뇨": "5.7–6.4%", "당뇨기준": "≥6.5%", "치료목표_일반": "<7.0%"},
    "bnp":         {"정상": "<100 pg/mL", "심부전의심": ">400 pg/mL"},
    "nt_probnp":   {"정상": "<300 pg/mL", "심부전의심": ">900 pg/mL (나이 보정 필요)"},
    "troponin_i":  {"정상_conventional": "<0.04 ng/mL", "note": "high-sensitivity: 성별 특이 99백분위수 사용"},
    "troponin_t":  {"정상_conventional": "<0.01 ng/mL", "note": "hsTnT: <14 ng/L (성별 무관)"},
    "inr":         {"정상": "0.8–1.2", "warfarin_af": "2.0–3.0", "warfarin_기계판막": "2.5–3.5"},
    "potassium":   {"정상": "3.5–5.0 mEq/L", "위험_저": "<3.0", "위험_고": ">6.0"},
    "sodium":      {"정상": "136–145 mEq/L"},
    "tsh":         {"정상": "0.4–4.0 mIU/L", "갑상선기능저하": ">4.0", "갑상선기능항진": "<0.1"},
    "glucose":     {"공복정상": "70–99 mg/dL", "당뇨기준": "≥126 mg/dL"},
    "ldl":         {"일반목표": "<100 mg/dL", "초고위험(ASCVD)": "<70 mg/dL", "매우초고위험": "<55 mg/dL"},
    "wbc":         {"정상": "4,500–11,000 /μL", "중성구감소": "<1,500", "패혈증의심": ">12,000 or <4,000"},
    "platelets":   {"정상": "150,000–400,000 /μL", "혈소판감소": "<100,000", "심각": "<50,000"},
    "lactate":     {"정상": "<2.0 mmol/L", "패혈증의심": "≥2.0", "패혈쇼크": "≥4.0"},
}


# ── 감별 진단 패턴 DB ────────────────────────────────────────────────────────

_DDX_PATTERNS: list[tuple[set[str], list[str]]] = [
    (
        {"chest pain", "diaphoresis", "radiation", "st elevation"},
        ["① STEMI (즉시 ECG + cath lab 활성화)", "② 대동맥박리(찢는 통증, BP 차이)", "③ 폐색전증(흉막성, DVT)"],
    ),
    (
        {"chest pain", "exertional"},
        ["① 불안정협심증/NSTEMI", "② 안정협심증 악화", "③ 심낭염(체위성)", "④ 식도연축"],
    ),
    (
        {"dyspnea", "edema", "orthopnea"},
        ["① 심부전 악화", "② 폐색전증", "③ 폐렴", "④ COPD/천식 악화"],
    ),
    (
        {"fever", "hypotension", "altered mental"},
        ["① 패혈쇼크(즉각 소생)", "② 세균성 뇌수막염(LP)", "③ 부신위기"],
    ),
    (
        {"ataxia", "areflexia", "cardiomyopathy"},
        ["① Friedreich 운동실조(유전자 검사: FXN)", "② AVED(비타민 E 결핍)", "③ 미토콘드리아 질환"],
    ),
    (
        {"weakness", "exercise", "episodic"},
        ["① 주기성 마비(periodic paralysis, K+ 확인)", "② 중증근무력증", "③ 이온통로병증"],
    ),
    (
        {"headache", "fever", "stiff neck"},
        ["① 세균성 뇌수막염(항생제 즉시)", "② 바이러스성 뇌수막염", "③ 지주막하출혈"],
    ),
]


# ── @tool 데코레이터 함수들 ───────────────────────────────────────────────────

@tool
def drug_interaction_checker(drug_a: str, drug_b: str) -> str:
    """
    두 약물 간 임상적으로 유의미한 상호작용을 확인합니다.
    drug_a: 첫 번째 약물 (예: 'warfarin', 'metformin')
    drug_b: 두 번째 약물 (예: 'aspirin', 'iv_contrast')
    """
    a_lower = drug_a.lower().strip()
    b_lower = drug_b.lower().strip()
    key = frozenset([a_lower, b_lower])
    if key in _INTERACTIONS:
        return f"[약물상호작용] {drug_a} + {drug_b}: {_INTERACTIONS[key]}"
    # 부분 일치 검색: frozenset 원소 문자열에 대해 substring 체크
    for db_key, interaction in _INTERACTIONS.items():
        db_drugs = sorted(db_key)   # 결정적 순서
        if any(a_lower in d or d in a_lower for d in db_drugs) or \
           any(b_lower in d or d in b_lower for d in db_drugs):
            pair = " + ".join(db_drugs)
            return f"[약물상호작용] {pair}: {interaction}\n(참고: '{drug_a}'/'{drug_b}' 포함 조합)"
    return (
        f"'{drug_a}'과 '{drug_b}'의 주요 상호작용이 데이터베이스에 없습니다. "
        "복잡한 병용 처방 시 임상약사 협의 및 Lexicomp/Micromedex 확인을 권장합니다."
    )


@tool
def clinical_guideline_search(query: str, specialty: str = "general") -> str:
    """
    근거 기반 임상 가이드라인을 검색합니다.
    query: 검색어 (예: 'STEMI management', 'HFrEF treatment', 'DM2 CKD')
    specialty: 전문과목 (예: 'cardiology', 'nephrology', 'general')
    """
    query_lower = query.lower()
    # 키워드 매칭
    for key, guideline in _GUIDELINES.items():
        if key in query_lower or any(word in query_lower for word in key.split("_")):
            return guideline
    # 전문과목 기반 2차 매칭
    specialty_map = {
        "cardiology": ["stemi", "nstemi", "heart_failure", "atrial_fibrillation"],
        "nephrology": ["ckd", "dm2_ckd"],
        "pulmonology": ["copd", "pneumonia_cap"],
        "endocrinology": ["dm2_ckd"],
        "neurology": [],
        "emergency": ["stemi", "sepsis"],
    }
    for spec_key in specialty_map.get(specialty.lower(), []):
        if any(word in query_lower for word in spec_key.split("_")):
            return _GUIDELINES[spec_key]
    return (
        f"'{query}'에 대한 가이드라인이 로컬 데이터베이스에 없습니다. "
        "UpToDate, PubMed, 또는 관련 학회(AHA/ACC/KDIGO/ADA) 웹사이트를 직접 확인하세요. "
        "근거가 불충분하거나 희귀 질환인 경우 전문가 에스컬레이션을 고려하세요."
    )


@tool
def lab_reference_lookup(test_name: str, patient_context: str = "") -> str:
    """
    임상 검사 수치의 정상 범위와 임상적 의미를 반환합니다.
    test_name: 검사명 (예: 'HbA1c', 'creatinine', 'troponin_i', 'INR')
    patient_context: 환자 맥락 (예: 'CKD stage 4', '75세 여성') — 맥락별 해석에 활용
    """
    key = test_name.lower().strip().replace(" ", "_").replace("-", "_")
    if key in _LAB_REFS:
        ref = _LAB_REFS[key]
        ref_str = " | ".join(f"{k}: {v}" for k, v in ref.items())
        context_note = ""
        if patient_context:
            context_note = f"\n[맥락 주의] '{patient_context}' 환자: 기준치 해석 시 신기능·연령·성별 보정 고려."
        return f"[{test_name.upper()} 참고치] {ref_str}{context_note}"
    # 유사 키 검색
    for db_key in _LAB_REFS:
        if db_key in key or key in db_key:
            ref = _LAB_REFS[db_key]
            ref_str = " | ".join(f"{k}: {v}" for k, v in ref.items())
            return f"[{db_key.upper()} 참고치] {ref_str}"
    return (
        f"'{test_name}' 검사의 참고치가 데이터베이스에 없습니다. "
        "기관 임상검사실 또는 LOINC 데이터베이스를 확인하세요."
    )


@tool
def differential_diagnosis(symptoms: str, patient_demographics: str = "") -> str:
    """
    증상 기반 감별 진단 목록을 우선순위별로 반환합니다.
    symptoms: 주요 증상 (예: 'crushing chest pain, diaphoresis, radiation to left arm')
    patient_demographics: 연령/성별 (예: '45세 남성', '82세 여성')
    """
    symptoms_lower = symptoms.lower()
    best_match: list[str] | None = None
    best_score = 0

    for keyword_set, diagnoses in _DDX_PATTERNS:
        score = sum(kw in symptoms_lower for kw in keyword_set)
        if score > best_score:
            best_score = score
            best_match = diagnoses

    demo_note = f" ({patient_demographics})" if patient_demographics else ""
    if best_match and best_score > 0:
        ddx_text = "\n".join(best_match)
        return (
            f"[감별진단{demo_note}]\n{ddx_text}\n"
            f"(매칭 증상: '{symptoms[:80]}')"
        )
    return (
        f"증상 '{symptoms[:80]}'에 대한 패턴 매칭 실패. "
        "포괄적 병력 청취, 신체 검진, 추가 검사 및 전문가 협진을 권장합니다."
    )


MEDICAL_TOOLS = [
    drug_interaction_checker,
    clinical_guideline_search,
    lab_reference_lookup,
    differential_diagnosis,
]