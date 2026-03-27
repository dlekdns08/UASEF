# UASEF 연구 파이프라인
**Uncertainty-Aware Safe Escalation Framework for Clinical LLM Agents**

LMStudio(로컬 모델)와 OpenAI를 동일한 파이프라인으로 비교 실험합니다.

---

## 프로젝트 구조

```
uasef/
├── modules/
│   ├── model_interface.py   # LMStudio / OpenAI 통합 레이어
│   ├── uqm.py               # Uncertainty Quantification Module
│   └── rtc_ede.py           # Risk-Threshold Calibrator + Escalation Engine
├── experiments/
│   ├── run_experiment.py    # 메인 실험 실행
│   └── visualize_results.py # 결과 시각화
├── results/                 # 실험 결과 저장 (자동 생성)
├── requirements.txt
└── .env.example
```

---

## 설치

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY와 LMSTUDIO_MODEL 수정
```

---

## LMStudio 설정

1. LMStudio 앱 실행
2. 원하는 모델 다운로드 (권장: `meta-llama-3.1-8b-instruct` 또는 `qwen2.5-7b-instruct`)
3. **Local Server** 탭 → **Start Server** (기본 포트: 1234)
4. `.env`의 `LMSTUDIO_MODEL`을 로드된 모델명으로 수정

LMStudio는 OpenAI-compatible API를 제공하므로 `base_url=http://localhost:1234/v1`만 설정하면 됩니다.

---

## 실행 순서

```bash
# Step 1: 실험 실행 (LMStudio + OpenAI 순차 실행)
python experiments/run_experiment.py

# Step 2: 결과 시각화
python experiments/visualize_results.py
```

결과 파일:
- `results/experiment_results.json` — 전체 raw 결과
- `results/comparison_table.csv` — 백엔드 비교 요약표
- `results/comparison_bar.png` — Safety Recall / Over-Escalation 바차트
- `results/pareto_frontier.png` — Coverage ↔ Escalation Pareto 그래프
- `results/latency_comparison.png` — 레이턴시 비교

---

## 개별 모듈 테스트

```bash
# 모델 연결 확인
python modules/model_interface.py

# UQM 단독 테스트
python modules/uqm.py

# RTC + EDE 단독 테스트
python modules/rtc_ede.py
```

---

## 핵심 설계 결정

### Conformal Prediction 임계값 (UQM)
- `alpha=0.05` → 95% Coverage 보장
- Calibration set: MedQA에서 무작위 샘플링 (실제 연구)
- 비적합 점수: logprobs 지원 시 음수 평균 log-likelihood, 미지원 시 self-consistency

### 동적 임계값 (RTC)
| 위험도 | 배율 | 전문과목 예시 |
|--------|------|--------------|
| CRITICAL | 0.60× | 응급의학, 중환자의학 |
| HIGH | 0.75× | 심장내과, 신경과 |
| MODERATE | 1.00× | 내과, 외과 |
| LOW | 1.30× | 일반 외래, 예방의학 |

### 에스컬레이션 트리거 (EDE)
1. 비적합 점수 > 조정된 임계값
2. 고위험 임상 키워드 감지 (intubation, vasopressor 등)
3. 근거 부재 표현 감지 ("I am not certain" 등)

---

## 목표 지표

| 지표 | 목표값 |
|------|--------|
| Safety Recall | ≥ 0.95 |
| Over-Escalation Rate | ≤ 0.15 |
| Coverage Validity | 1-α 이상 |
| Abstention Accuracy | 기존 대비 +10%p |
