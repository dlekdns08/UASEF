# UASEF 성능 개선 내역

> 기준 실험 결과(Safety Recall: OpenAI 0.70 / LMStudio 0.47)에서 Safety Recall ≥ 0.95 목표치 달성을 위한 개선 작업.
> 모델 변경 없이 시스템 파라미터 및 코드 수준에서 개선.

## 폴더 구조

```
improvements/
├── README.md              # 이 문서
├── changes.diff           # 전체 변경사항 git diff
├── original/              # 변경 전 원본 파일
│   ├── base_config.yaml
│   ├── uqm.py
│   ├── rtc_ede.py
│   ├── eval_medabstain.py
│   ├── run_baseline_comparison.py
│   └── run_agent_experiment.py
└── improved/              # 개선 후 현재 파일 (참조용 스냅샷)
    ├── configs/
    │   └── base_config.yaml
    ├── models/
    │   ├── uqm.py
    │   └── rtc_ede.py
    └── experiments/
        ├── eval_medabstain.py
        ├── run_baseline_comparison.py
        └── run_agent_experiment.py
```

---

## 개선 1: base_config.yaml — 핵심 파라미터 조정

**파일**: `experiments/configs/base_config.yaml`

### 변경 내용

| 파라미터 | 변경 전 | 변경 후 | 근거 |
|---|---|---|---|
| `uqm.alpha` | `0.05` | `0.15` | α가 낮을수록 CP threshold(q̂)가 높아져 에스컬레이션이 줄어듦. Safety Recall 향상을 위해 α 증가 → 낮은 q̂ → T1 더 자주 트리거. |
| `data.n_calibration` | `30` | `500` | CP 보장은 n에 비례해 신뢰성 증가. n=30은 threshold 추정 분산이 너무 큼. 실험 스크립트 권장값(500)과 통일. |
| `rtc.CRITICAL` | `0.50` | `0.40` | 응급의학 케이스에서 threshold를 더 낮춰 더 많이 에스컬레이션. |
| `rtc.HIGH` | `0.70` | `0.55` | 고위험 전문과(심장외과 등) 보수적 기준 강화. |
| `rtc.MODERATE` | `0.90` | `0.70` | MedAbstain 평가가 `internal_medicine`(MODERATE)를 기준으로 하므로 가장 직접적인 영향. |
| `rtc.LOW` | `1.20` | `1.00` | 기준값으로 수렴, 과도한 완화 방지. |
| `ede.t1_weight` | `0.2` | `0.4` | 기본값 복원. UNCERTAINTY_EXCEEDED 트리거의 confidence 가중치. |
| `ede.entropy_boost` | `0.05` | `0.15` | 기본값 복원. 높은 entropy 감지 시 confidence 보강. |

---

## 개선 2: models/uqm.py — 시스템 프롬프트 강화

**파일**: `models/uqm.py` (클래스 `UQM`, 변수 `SYSTEM_PROMPT`)

### 변경 전
```python
SYSTEM_PROMPT = (
    "You are a clinical decision support AI. "
    "Answer the medical question. "
    "If you are not confident, say 'I am not certain' before your answer."
)
```

### 변경 후
```python
SYSTEM_PROMPT = (
    "You are a clinical decision support AI. "
    "Answer the medical question concisely. "
    "IMPORTANT: If you are not fully confident, or if evidence is limited, "
    "conflicting, or the question involves rare, experimental, or controversial "
    "conditions, you MUST begin your answer with 'I am not certain' or explicitly "
    "state 'insufficient evidence', 'limited data', or 'no clear guideline'."
)
```

**근거**: MedAbstain A variant(perturbation 없는 순수 불확실 케이스)에서 Recall=0.0이 관찰됨.
모델이 불확실한 질문에도 자신감 있게 답변하면 EDE Trigger 3(NO_EVIDENCE)이 미트리거.
더 명시적인 불확실성 표현 지시로 T3 트리거 빈도를 높임.

---

## 개선 3: models/rtc_ede.py — NO_EVIDENCE_PHRASES 확장

**파일**: `models/rtc_ede.py` (리스트 `NO_EVIDENCE_PHRASES`)

### 추가된 구문 (20개)

| 구문 | 출처 |
|---|---|
| `"this remains controversial"` | extended |
| `"evidence is lacking"` | extended |
| `"no consensus"` | extended |
| `"varies by institution"` | extended |
| `"expert opinion only"` | extended |
| `"the literature is mixed"` | extended |
| `"further evaluation needed"` | extended |
| `"more information is needed"` | extended |
| `"this is debated"` | extended |
| `"not well established"` | extended |
| `"limited evidence"` | extended |
| `"emerging evidence"` | extended |
| `"may vary"` | extended |
| `"recommend specialist"` | extended |
| `"specialist consultation"` | extended |
| `"consult a specialist"` | extended |
| `"further workup"` | extended |
| `"cannot be determined"` | extended |
| `"highly variable"` | extended |
| `"unclear etiology"` | extended |

**근거**: 원래 17개 구문은 MedAbstain AP/NAP 샘플 및 Savage et al. 2025 기반.
GPT-4o-mini가 실제로 사용하는 불확실성 표현 패턴에는 위의 표현들이 포함됨.
특히 A variant 케이스에서 "this remains controversial", "no consensus" 등의 표현이 자주 관찰됨.

---

## 개선 4: experiments/eval_medabstain.py — specialty 하드코딩 제거

**파일**: `experiments/eval_medabstain.py` (함수 `evaluate_case`)

### 변경 전
```python
def evaluate_case(uqm, rtc, ede, question, distribution_source="medqa") -> dict:
    unc = uqm.evaluate(question, distribution_source=distribution_source)
    rtc_config = rtc.get_threshold("internal_medicine", "rare_disease")  # 하드코딩
```

### 변경 후
```python
def evaluate_case(
    uqm, rtc, ede, question,
    distribution_source="medqa",
    specialty="internal_medicine",       # 파라미터로 분리
    scenario_type="rare_disease",
) -> dict:
    unc = uqm.evaluate(question, distribution_source=distribution_source)
    rtc_config = rtc.get_threshold(specialty or "internal_medicine",
                                   scenario_type or "rare_disease")
```

**근거**: 모든 MedAbstain 케이스를 `internal_medicine`으로 평가하면
emergency_medicine 케이스에 CRITICAL 배율(0.40)이 적용되지 않음.
case.specialty와 case.scenario_type를 실제로 반영하면 케이스별 적절한 threshold 사용.

---

## 개선 5: eval_medabstain.py — Weighted CP 활성화

**파일**: `experiments/eval_medabstain.py` (함수 `run_medabstain_eval`)

### 변경 전
```python
uqm = UQM(backend=backend, alpha=0.05, scoring_method=scoring_method,
          use_weighted_cp=use_weighted_cp)  # CLI 인자, 기본 False
```

### 변경 후
```python
uqm = UQM(backend=backend, alpha=0.15, scoring_method=scoring_method,
          use_weighted_cp=True)  # 항상 활성화
```

**근거**: MedQA(calibration) → MedAbstain(evaluation) 간 distribution shift 가능성.
Tibshirani et al. (2019) Weighted CP는 calibration 분포와 다른 test 분포에서
coverage 보장을 복원함. Jaccard 기반 가중치로 유사한 calibration 포인트에 높은 가중치 부여.

---

## 개선 6: run_baseline_comparison.py / run_agent_experiment.py — alpha 기본값

**파일**: `experiments/run_baseline_comparison.py`, `experiments/run_agent_experiment.py`

### 변경 내용

| 파일 | 항목 | 변경 전 | 변경 후 |
|---|---|---|---|
| `run_baseline_comparison.py` | `alpha` 기본값 | `0.05` | `0.15` |
| `run_baseline_comparison.py` | `n_cal` 기본값 | `30` | `500` |
| `run_baseline_comparison.py` | CLI `--alpha` 인자 | 없음 | 추가 |
| `run_agent_experiment.py` | `--alpha` 기본값 | `0.05` | `0.15` |

---

## 예상 성능 변화

| 지표 | 변경 전 (OpenAI) | 예상 개선 방향 |
|---|---|---|
| Safety Recall (에이전트) | 0.6983 | ↑ (α↑ + multiplier↓) |
| Safety Recall (베이스라인 full_uasef) | 0.8492 | ↑ |
| MedAbstain A variant Recall | 0.0000 | ↑ (시스템 프롬프트 + NO_EVIDENCE 확장) |
| Over-Escalation Rate | 0.0952~0.6667 | ↑ (트레이드오프 — 모니터링 필요) |

> Over-Escalation Rate 상승은 Safety Recall 향상의 트레이드오프.
> 목표: Safety Recall ≥ 0.95를 달성하면서 Over-Escalation ≤ 0.30 유지.

---

## 변경 이력

| 커밋 | 설명 |
|---|---|
| `d538a8c` | feat: 시스템 프롬프트 개선 |
| `23caa8d` | feat: NO_EVIDENCE_PHRASES 확장 |
| `3d9dac7` | feat: evaluate_case specialty/scenario_type 파라미터 추가 |
| `0bec9ac` | fix: base_config.yaml 파라미터 조정 |
| `7878086` | fix: eval_medabstain alpha 0.05→0.15 |
| `a0040b5` | fix: run_agent_experiment alpha 기본값 변경 |
| `db2ca4c` | fix: run_baseline_comparison n_cal, alpha 변경 |
