# data/ — UASEF 데이터 소스 가이드

이 디렉토리에는 UASEF 실험에 사용되는 의료 QA 데이터셋이 위치합니다.

---

## 디렉토리 구조

```text
data/
├── __init__.py
├── loader.py          # 데이터 로딩 API (load_calibration_questions, load_scenarios, load_pubmedqa)
├── README.md          # 이 파일
└── raw/               # 로컬 JSONL 파일 (git 제외, .gitignore에 등록)
    ├── medqa_train.jsonl
    ├── medqa_test.jsonl
    ├── medabstain_AP.jsonl
    ├── medabstain_NAP.jsonl
    └── pubmedqa_test.jsonl    # 선택 — load_pubmedqa() 오프라인 사용 시
```

`data/raw/` 는 `.gitignore`에 포함되어야 합니다 (라이선스 및 용량 이유).

---

## 데이터소스 1 — MedQA (USMLE)

**용도**: Calibration 질문 및 기본 테스트 시나리오

**출처**: Jin et al., 2021 — "What Disease does this Patient Have?"
**라이선스**: MIT License

### HuggingFace 자동 다운로드 (권장)

`loader.py`가 자동으로 시도합니다:

```python
from datasets import load_dataset
ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
```

인터넷 연결이 있으면 별도 설치 불필요. 첫 실행 시 HuggingFace 캐시(`~/.cache/huggingface/`)에 저장됩니다.

### 로컬 JSONL 수동 설치 (오프라인 환경)

```bash
# jind11/MedQA GitHub에서 다운로드
git clone https://github.com/jind11/MedQA.git /tmp/MedQA
cp /tmp/MedQA/data/questions/US/4_options/train.jsonl data/raw/medqa_train.jsonl
cp /tmp/MedQA/data/questions/US/4_options/test.jsonl  data/raw/medqa_test.jsonl
```

### JSONL 포맷

```json
{
  "question": "A 45-year-old man presents with...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer_idx": "A",
  "answer": "...",
  "meta_info": "step1"
}
```

---

## 데이터소스 2 — MedAbstain

**용도**: 희귀질환·불확실 시나리오 (expected_escalate=True 케이스)

**출처**: Zhu et al., 2023 — "Can LLMs Express Their Uncertainty?"
**GitHub**: https://github.com/HowieSiao/medabstain
**라이선스**: CC BY 4.0

### 4가지 변형

| 파일명 | 설명 | expected_escalate |
|--------|------|-------------------|
| `medabstain_NA.jsonl` | 정상 + 정상 | False |
| `medabstain_A.jsonl`  | Abstention (불확실) | True |
| `medabstain_NAP.jsonl`| 정상 + Perturbed (변형) | True |
| `medabstain_AP.jsonl` | Abstention + Perturbed | True |

`AP` / `NAP` 변형이 희귀질환 시나리오의 핵심 테스트 소스입니다.

### 수동 설치

```bash
git clone https://github.com/HowieSiao/medabstain.git /tmp/medabstain
cp /tmp/medabstain/data/AP.jsonl  data/raw/medabstain_AP.jsonl
cp /tmp/medabstain/data/NAP.jsonl data/raw/medabstain_NAP.jsonl
```

---

## 데이터소스 3 — MIMIC-III (선택, 분포 이동 실험)

**용도**: 실제 ICU 임상 기록으로 distribution shift 실험

**출처**: Johnson et al., 2016 — MIMIC-III Clinical Database
**라이선스**: PhysioNet Credentialed Health Data License
**필수 조건**: PhysioNet DUA(Data Use Agreement) 서명 필요

> ⚠️ **중요**: MIMIC-III 데이터는 PhysioNet 계정이 있고 DUA를 완료한 연구자만 접근할 수 있습니다.
> 무단 배포 금지. 데이터는 절대 git에 포함하지 마세요.

### 신청 절차

1. PhysioNet 계정 생성: https://physionet.org/register/
2. CITI 교육 이수 (Biomedical Research 과정)
3. MIMIC-III DUA 서명: https://physionet.org/content/mimiciii/1.4/
4. 승인 후 다운로드: `NOTEEVENTS.csv.gz` (임상 노트)

### UASEF에서의 사용

```yaml
# experiments/configs/scenario_multimorbidity.yaml
data:
  distribution_source: mimic3
```

MIMIC-III 사용 시 **반드시 재보정**이 필요합니다 (CP exchangeability):

```python
uqm.calibrate(mimic_cal_questions, distribution_source="mimic3")
uqm.evaluate(test_question, distribution_source="mimic3")
```

MedQA로 보정 후 MIMIC-III로 평가하면 CP 보장이 깨집니다 (Tibshirani et al., 2019).

---

## Fallback 데이터 (데이터셋 없는 경우)

인터넷 연결이 없고 로컬 파일도 없으면 `loader.py`는 내장 fallback 데이터를 사용합니다:
- Calibration: 20개 USMLE 스타일 질문
- 시나리오: 9개 케이스 (emergency/multimorbidity/routine 각 3개)

Fallback은 **개발·테스트 전용**입니다. 논문 품질 실험에는 실제 데이터셋이 필요합니다.

---

## 논문 권장 설정

| 파라미터 | 최솟값 | 권장값 |
|---------|--------|--------|
| `n_calibration` | 30 | 500 |
| `n_test_per_scenario` | 10 | 50–100 |
| Calibration split | train | train |
| Test split | test | test |

CP 이론 보증 (Angelopoulos & Bates, 2021):

```
q̂ = ceil((n+1)(1-α)) / n 번째 순위 비적합 점수
P(s_test ≤ q̂) ≥ 1-α  (n=500이면 실측 coverage ≈ 이론값)
```

n이 작으면 (n=30) 실측 coverage가 이론값보다 높게 측정될 수 있습니다 (보수적 특성).

---

## .gitignore 권장 항목

```
data/raw/
results/
.env
```
