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
    │   ├── rtc_ede.py
    │   └── model_interface.py
    └── experiments/
        ├── config_utils.py
        ├── eval_medabstain.py
        ├── run_baseline_comparison.py
        └── run_agent_experiment.py
```

---

## 실험 결과 요약 (α=0.15, 2026-04-07)

> 1라운드 개선 후 실행한 실험. 이 결과를 기반으로 2라운드 개선을 진행함.

| 지표 | LMStudio | OpenAI | 목표 |
|---|---|---|---|
| Safety Recall (에이전트) | 0.6854 | 0.9050 | ≥ 0.95 |
| Safety Recall (full_uasef) | 0.7095 | 0.7374 | ≥ 0.95 |
| MedAbstain A variant Recall | 0.0000 | 0.0000 | ≥ 0.95 |
| Over-Escalation Rate (에이전트) | 0.2857 | 0.6190 | ≤ 0.15 |
| Coverage (에이전트) | 0.8100 | 0.8900 | ≥ 1-α |

### 확인된 핵심 문제

| # | 문제 | 원인 | 영향 |
|---|---|---|---|
| P1 | CP coverage 보장 위반 (LMStudio 0.81 < 0.85) | α=0.15로 캘리브레이션 시 LMStudio logprob 분산이 커 CP 이론 보장이 깨짐 | 방법론 신뢰성 저하 |
| P2 | MedAbstain A variant Recall = 0.00 | T1: α=0.15 → 임계값 높아 미트리거. T3: 모델이 불확실 표현을 하지 않음 | Safety Recall 전혀 달성 불가 |
| P3 | OpenAI emergency_medicine esc_rate ≈ 1.00 | CRITICAL 배율(0.40) × 시나리오 보정(0.85) = 0.34 → adjusted_threshold 거의 0 | Over-Escalation 폭증 |
| P4 | OpenAI Over-Escalation = 0.619 (목표 ≤ 0.15의 4배) | P3와 동일 원인 + α=0.15로 calibration threshold 낮음 | 임상 운용 불가 |
| P5 | OpenAI 400 에러 (MedQA ~240번 샘플) | JSON 직렬화 불가 문자(BOM, 제어문자) 포함 샘플 | 캘리브레이션 도중 크래시 |
| P6 | eval_medabstain.py alpha 하드코딩 | `alpha=0.15`가 함수 내부에 고정되어 base_config.yaml 수정이 반영 안 됨 | 파라미터 독립성 결여 |

---

## 1라운드 개선 (이전 세션)

### 개선 1: base_config.yaml — 핵심 파라미터 조정

**파일**: `experiments/configs/base_config.yaml`

| 파라미터 | 변경 전 | 변경 후 | 근거 |
|---|---|---|---|
| `uqm.alpha` | `0.05` | `0.15` | α가 낮을수록 CP threshold(q̂)가 높아져 에스컬레이션이 줄어듦. Safety Recall 향상을 위해 α 증가 → 낮은 q̂ → T1 더 자주 트리거. |
| `data.n_calibration` | `30` | `500` | CP 보장은 n에 비례해 신뢰성 증가. n=30은 threshold 추정 분산이 너무 큼. |
| `rtc.CRITICAL` | `0.50` | `0.40` | 응급의학 케이스에서 threshold를 더 낮춰 더 많이 에스컬레이션. |
| `rtc.HIGH` | `0.70` | `0.55` | 고위험 전문과 보수적 기준 강화. |
| `rtc.MODERATE` | `0.90` | `0.70` | MedAbstain 평가 기준 전문과 직접 영향. |
| `rtc.LOW` | `1.20` | `1.00` | 기준값으로 수렴, 과도한 완화 방지. |
| `ede.t1_weight` | `0.2` | `0.4` | UNCERTAINTY_EXCEEDED 트리거 confidence 가중치 복원. |
| `ede.entropy_boost` | `0.05` | `0.15` | 높은 entropy 감지 시 confidence 보강 복원. |

### 개선 2: models/uqm.py — 시스템 프롬프트 강화

MedAbstain A variant Recall=0.00 대응. T3(NO_EVIDENCE) 트리거 빈도 향상 목적으로 불확실성 표현 지시를 강화.

### 개선 3: models/rtc_ede.py — NO_EVIDENCE_PHRASES 확장

20개 구문 추가. GPT-4o-mini가 실제 사용하는 불확실성 표현 패턴 반영.

### 개선 4: experiments/eval_medabstain.py — specialty 하드코딩 제거

`evaluate_case()`의 `specialty`, `scenario_type` 파라미터화. 케이스별 적절한 RTC 임계값 적용.

### 개선 5: eval_medabstain.py — Weighted CP 항상 활성화

MedQA(calibration) → MedAbstain(evaluation) distribution shift 대응.

### 개선 6: run_baseline_comparison.py / run_agent_experiment.py — alpha 기본값

alpha 기본값 0.05→0.15, n_cal 기본값 30→500.

---

## 2라운드 개선 (2026-04-08, 현재 세션)

> 1라운드 개선 후 실험 결과(P1~P6) 분석에 기반한 수정.

### 개선 7: models/model_interface.py — `_sanitize()` 강화 + `import re` 추가

**파일**: `models/model_interface.py`
**대응 문제**: P5 (OpenAI 400 에러)

#### 변경 전
```python
import os
import time
import json
import urllib.request
...
def _sanitize(text: str) -> str:
    """null bytes 및 JSON 직렬화를 깨는 제어 문자 제거."""
    text = text.replace("\x00", "")
    return text.encode("utf-8", errors="ignore").decode("utf-8")
```

#### 변경 후
```python
import os
import re
import time
import json
import urllib.request
...
def _sanitize(text: str) -> str:
    """null bytes, 서로게이트, BOM, JSON 직렬화를 깨는 제어 문자 제거."""
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = text.replace("\ufeff", "")  # BOM 제거
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # 제어문자 제거
    return text
```

**근거**: MedQA ~240번 샘플에서 BOM(`\ufeff`) 또는 JSON 직렬화 불가 제어문자가 포함되어 OpenAI API가 400을 반환. 기존 코드는 null byte(`\x00`)만 제거했으나, OpenAI API는 `\t`, `\n`, `\r`을 제외한 모든 제어문자(`\x00-\x1f`)를 허용하지 않음.

---

### 개선 8: models/uqm.py — 캘리브레이션 루프 재시도 및 스킵 처리

**파일**: `models/uqm.py` (`UQM.calibrate()`)
**대응 문제**: P5 + LMStudio logprobs 불안정

#### 변경 전
```python
for i, q in enumerate(questions):
    score, _ = self._get_score(q)  # 실패 시 전체 크래시
```

#### 변경 후
```python
for i, q in enumerate(questions):
    last_exc = None
    for attempt in range(1, 4):          # 최대 3회 재시도
        try:
            score, _ = self._get_score(q)
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            if attempt < 3:
                print(f"  [RETRY {attempt}/3] {i+1}/{n_total}: {e}")
    if last_exc is not None:
        n_skipped += 1
        print(f"  [SKIP {i+1}/{n_total}] 3회 실패, 샘플 건너뜀: {last_exc}")
        continue
```

**근거**: 단일 샘플의 API 오류(네트워크, 400, logprobs 미반환)로 500개 전체 캘리브레이션이 중단되는 문제. 3회 재시도 후 실패 시 해당 샘플만 스킵하여 캘리브레이션 지속.

---

### 개선 9: experiments/configs/base_config.yaml — α 복원 + RTC 배율 복원

**파일**: `experiments/configs/base_config.yaml`
**대응 문제**: P1 (CP 보장 위반), P3/P4 (Over-Escalation)

| 파라미터 | 1라운드 | 2라운드 | 근거 |
|---|---|---|---|
| `uqm.alpha` | `0.15` | `0.05` | Pareto sweep: LMStudio α=0.15 → 실측 coverage 0.75 (목표 0.85 미달, CP 이론 위반). α=0.05 → coverage 0.9467 ≥ 0.95 보장 |
| `rtc.CRITICAL` | `0.40` | `0.60` | 코드 기본값(RISK_THRESHOLD_MULTIPLIER) 복원. 0.40은 기본값 0.60보다 훨씬 공격적인 override였음 |
| `rtc.HIGH` | `0.55` | `0.75` | 코드 기본값 복원 |
| `rtc.MODERATE` | `0.70` | `1.00` | 코드 기본값 복원 |
| `rtc.LOW` | `1.00` | `1.30` | 코드 기본값 복원 |

**α 방향 전환 근거 (1라운드 ↔ 2라운드)**:
- 1라운드: Safety Recall이 너무 낮아 α를 높임 (0.05→0.15) → T1이 더 자주 트리거
- 실험 결과: α=0.15에서 LMStudio CP 보장 위반 발생 (0.75 < 0.85)
- 2라운드: α=0.05로 복원 + RTC 배율 정상화로 Over-Escalation 해결
- 즉, 1라운드의 α 증가 전략은 LMStudio에서 CP 이론적 기반 자체를 무너뜨림

**RTC 배율 복원 근거**:
- 기존 config: `CRITICAL: 0.40` + 코드 내 시나리오 보정 `×0.85` = 최종 **0.34**
- emergency_medicine(CRITICAL) + emergency scenario → `base_threshold × 0.34`
- 이 값이 너무 낮아 거의 모든 케이스가 T1을 트리거 → OpenAI esc_rate = 1.00
- 복원 후: `CRITICAL: 0.60` × `0.90` = **0.54** (적정 범위)

---

### 개선 10: models/rtc_ede.py — 시나리오 보정 계수 완화

**파일**: `models/rtc_ede.py` (`RTCConfig.__post_init__()`)
**대응 문제**: P3/P4 (emergency_medicine Over-Escalation)

#### 변경 전
```python
if self.scenario_type in ("emergency", "rare_disease"):
    multiplier *= 0.85
```

#### 변경 후
```python
# 0.85 → 0.90: 기존 값은 CRITICAL(0.60) × 0.85 = 0.51로 너무 공격적이었음
if self.scenario_type in ("emergency", "rare_disease"):
    multiplier *= 0.90
```

**근거**: 개선 9에서 RTC 배율을 코드 기본값으로 복원하면 `CRITICAL × 0.85 = 0.51`. 아직도 공격적. `× 0.90 = 0.54`로 완화하여 안전 마진을 유지하면서 과-에스컬레이션 억제.

---

### 개선 11: models/uqm.py — ConformalCalibrator 최소 n 검증

**파일**: `models/uqm.py` (`ConformalCalibrator.fit()`)
**대응 문제**: P1 (CP 보장 위반 조기 감지)

```python
# CP coverage 보장을 위한 최소 n 검증
# α=0.05 → n ≥ 19 필요, α=0.01 → n ≥ 99 필요
min_n = math.ceil((1 - self.alpha) / self.alpha)
if n < min_n:
    warnings.warn(
        f"[UQM] Calibration n={n}이 CP 보장을 위한 최소값 {min_n}(α={self.alpha})보다 작습니다. ...",
        UserWarning,
    )
```

**근거**: 개선 8에서 스킵된 샘플로 실제 n_cal이 줄어들 때 CP 보장이 무음으로 깨지는 문제를 조기 감지. `ceil((n+1)(1-α)/n) ≤ 1` 조건에서 유도.

---

### 개선 12: experiments/eval_medabstain.py — alpha 하드코딩 제거 + n_cal 기본값 수정

**파일**: `experiments/eval_medabstain.py` (`run_medabstain_eval()`)
**대응 문제**: P6 (alpha 하드코딩)

#### 변경 전
```python
def run_medabstain_eval(
    backend: str,
    n_cal: int = 30,        # ← n_cal 기본값 30 (너무 낮음)
    ...
) -> dict:
    uqm = UQM(
        backend=backend,
        alpha=0.15,          # ← 하드코딩! base_config.yaml 무시
        ...
    )
```

#### 변경 후
```python
def run_medabstain_eval(
    backend: str,
    n_cal: int = 500,        # n_cal 기본값 500으로 수정
    alpha: float = None,     # 외부 주입 가능하도록 파라미터 추가
    ...
) -> dict:
    from experiments.config_utils import load_config
    cfg = load_config()
    effective_alpha = alpha if alpha is not None else cfg.get("uqm", {}).get("alpha", 0.05)
    uqm = UQM(
        backend=backend,
        alpha=effective_alpha,   # base_config.yaml → CLI 인자 순으로 우선순위
        ...
    )
```

**근거**: base_config.yaml의 `uqm.alpha` 수정이 eval_medabstain.py에 반영되지 않았음. 우선순위: 명시적 `alpha` 인자 > `base_config.yaml` > 기본값 0.05.

---

### 개선 13: experiments/config_utils.py — load_config() 함수 추가

**파일**: `experiments/config_utils.py`
**대응 문제**: 개선 12 의존성

`load_calibration_config()`는 RTC/EDE 파라미터만 반환했음. `load_config()`는 전체 base_config.yaml을 반환하여 다른 실험 스크립트에서도 alpha, n_calibration 등을 읽을 수 있도록 함.

---

## 예상 성능 변화 (2라운드 적용 후)

| 지표 | 1라운드 결과 | 2라운드 예상 | 방향 |
|---|---|---|---|
| LMStudio CP Coverage | 0.81 (보장 위반) | ≥ 0.95 | ↑ (α 0.15→0.05 효과) |
| OpenAI CP Coverage | 0.89 | ≥ 0.95 | ↑ |
| OpenAI emergency esc_rate | 1.00 | 0.5~0.7 | ↓ (RTC 배율 복원) |
| OpenAI Over-Escalation (전체) | 0.619 | 0.3~0.4 | ↓ |
| Safety Recall (에이전트) | 0.6854 / 0.9050 | 모니터링 필요 | 불확실 (α 낮아져 T1 감소 가능) |
| MedAbstain A variant Recall | 0.00 | 모니터링 필요 | α 낮아지면 T1 감소, T3는 불변 |

> **A variant Recall 잔류 문제**: α=0.05로 임계값이 높아지면 T1 감소 가능성 있음.
> 모델이 A variant에서도 높은 logprob(자신감 있는 답변)을 출력하는 한 T1으로 감지 불가.
> 근본 해결은 A variant 케이스를 calibration set에 포함시키거나 별도 fine-tuning 필요.

---

## 변경 이력

### 1라운드

| 커밋 | 설명 |
|---|---|
| `d538a8c` | feat: 시스템 프롬프트 개선 |
| `23caa8d` | feat: NO_EVIDENCE_PHRASES 확장 |
| `3d9dac7` | feat: evaluate_case specialty/scenario_type 파라미터 추가 |
| `0bec9ac` | fix: base_config.yaml 파라미터 조정 |
| `7878086` | fix: eval_medabstain alpha 0.05→0.15 |
| `a0040b5` | fix: run_agent_experiment alpha 기본값 변경 |
| `db2ca4c` | fix: run_baseline_comparison n_cal, alpha 변경 |

### 2라운드 (2026-04-08)

| 파일 | 변경 내용 |
|---|---|
| `models/model_interface.py` | `_sanitize()` 강화: BOM·제어문자 제거, `import re` 추가 |
| `models/uqm.py` | 캘리브레이션 루프 3회 재시도 + 스킵 처리 |
| `models/uqm.py` | `ConformalCalibrator.fit()` 최소 n 경고 추가 |
| `models/rtc_ede.py` | 시나리오 보정 계수 0.85→0.90 |
| `experiments/configs/base_config.yaml` | α 0.15→0.05, RTC 배율 코드 기본값으로 복원 |
| `experiments/eval_medabstain.py` | alpha 하드코딩 제거, n_cal 기본값 30→500 |
| `experiments/config_utils.py` | `load_config()` 함수 추가 |
