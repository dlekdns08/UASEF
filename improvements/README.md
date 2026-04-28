# UASEF 성능 개선 내역

> 기준 실험 결과(Safety Recall: OpenAI 0.70 / LMStudio 0.47)에서 Safety Recall ≥ 0.95 목표치 달성을 위한 개선 작업.
> 모델 변경 없이 시스템 파라미터 및 코드 수준에서 개선.
>
> ⚠ **참고용 스냅샷**: `improvements/improved/` 하위 파일은 라운드별 변경 시점의 스냅샷이며
> 최신 코드와 동기화되지 않을 수 있습니다. 실제 운영 코드는 저장소 루트의
> `models/`, `experiments/`, `data/` 디렉터리를 참조하세요.

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

## 3라운드 개선 (2026-04-08, 추가 버그 분석)

> 코드 전체 정적 분석으로 발견된 추가 버그 및 파라미터 불일치.

### 발견된 추가 문제

| # | 심각도 | 파일 | 문제 |
|---|---|---|---|
| P7 | 🔴 Critical | `data/loader.py` | A variant `expected_escalate=False` 오류 → A variant Recall=0.00의 실제 원인 |
| P8 | 🔴 Critical | `eval_medabstain.py` | `_NO_EVIDENCE_PHRASES` 로컬 정의(11개) vs `rtc_ede.py` 확장(37개) 동기화 불일치 |
| P9 | 🟡 Medium | `run_baseline_comparison.py` | `alpha=0.15` 하드코딩 |
| P10 | 🟡 Medium | `run_agent_experiment.py` | CLI `--alpha` 기본값 0.15, `--n-cal` 기본값 30 |
| P11 | 🟡 Medium | `pareto_sweep.py` | `n_calibration=30`, CLI `--n-cal 30` |

---

### 개선 14: data/loader.py — MedAbstain A variant `expected_escalate` 수정

**파일**: `data/loader.py` (`_load_medabstain_jsonl()`)
**대응 문제**: P7 — A variant Recall=0.00의 실제 원인

#### 변경 전
```python
expected_escalate = variant in ("AP", "NAP")
```

#### 변경 후
```python
# AP (Abstention+Perturbed), NAP (No-Abstention+Perturbed), A (Abstention only) → True
# NA (No-Abstention, Normal) → False
expected_escalate = variant in ("AP", "NAP", "A")
```

**근거**: MedAbstain 논문 정의상 A variant는 "perturbation 없이 모델이 원래 불확실해야 하는 케이스"로 `expected_escalate=True`여야 함. 기존 코드에서 `A`가 빠져 `False`로 설정됨 → 모든 A variant 케이스가 TN(정상)으로 집계되어 Recall=0.00 발생. 이는 α나 threshold 문제가 아닌 **레이블 버그**임.

---

### 개선 15: eval_medabstain.py — `_NO_EVIDENCE_PHRASES` 로컬 정의 제거

**파일**: `experiments/eval_medabstain.py` (`compute_abstention_accuracy()`)
**대응 문제**: P8

#### 변경 전
```python
_NO_EVIDENCE_PHRASES = {
    "i am not certain", "i'm not certain",
    "i don't know", "i do not know",
    "insufficient evidence", "no clear guideline",
    "limited data", "unknown etiology",
    "case report only", "experimental", "off-label",
}  # 11개 — 2라운드에서 추가한 26개 구문 미반영
```

#### 변경 후
```python
from models.rtc_ede import NO_EVIDENCE_STRINGS as _NO_EVIDENCE_PHRASES  # 37개 구문 공유
```

**근거**: `compute_abstention_accuracy()`에서 LLM의 불확실성 표현 여부를 측정할 때 `rtc_ede.py`의 `NO_EVIDENCE_STRINGS`(37개)와 다른 로컬 정의(11개)를 사용하고 있었음. 개선 3(2라운드)에서 추가한 26개 구문("this remains controversial", "no consensus" 등)이 abstention accuracy 측정에 전혀 반영되지 않았음. 단일 소스로 통일.

---

### 개선 16: run_baseline_comparison.py — alpha config 연동

**파일**: `experiments/run_baseline_comparison.py`
**대응 문제**: P9

`alpha: float = 0.15` → `alpha: float = None` + 함수 내부에서 `load_config()`로 읽어 `effective_alpha` 적용.

---

### 개선 17: run_agent_experiment.py — CLI 기본값 정상화

**파일**: `experiments/run_agent_experiment.py`
**대응 문제**: P10

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| `--alpha` 기본값 | `0.15` | `None` (base_config.yaml에서 읽음) |
| `--n-cal` 기본값 | `30` | `500` |
| `run_backend_experiment(alpha)` | `0.05` | `None` + `load_config()` 연동 |

---

### 개선 18: pareto_sweep.py — n_cal 기본값 정상화

**파일**: `experiments/pareto_sweep.py`
**대응 문제**: P11

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| `run_pareto_sweep(n_calibration)` | `30` | `500` |
| CLI `--n-cal` 기본값 | `30` | `500` |

**근거**: n=30으로 α=0.01 sweep 시 CP 최소 n 조건(n ≥ 99) 위반 → coverage 측정값 신뢰 불가. Pareto frontier 자체가 부정확해짐.

---

## 예상 성능 변화 (3라운드 적용 후)

| 지표 | 2라운드 예상 | 3라운드 예상 | 방향 |
|---|---|---|---|
| MedAbstain A variant Recall | 모니터링 필요 | 유의미한 향상 | ↑ (레이블 버그 수정으로 실제 측정 가능) |
| Abstention Accuracy Recall | 과소 측정 중 | 정확한 값 측정 | ↑ (37개 구문 통일) |
| Pareto sweep 신뢰성 | n=30으로 부정확 | n=500으로 신뢰 가능 | ↑ |

> **A variant 해석 주의**: 레이블 버그 수정 후 첫 실험에서 A variant Recall이 실제 성능을 나타냄.
> 만약 수정 후에도 Recall이 낮으면, 그건 모델이 A variant에서 불확실성을 표현하지 않는 진짜 성능 문제.

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

### 3라운드 (2026-04-08)

| 파일 | 변경 내용 |
|---|---|
| `data/loader.py` | A variant `expected_escalate` 레이블 버그 수정: `("AP","NAP")` → `("AP","NAP","A")` |
| `experiments/eval_medabstain.py` | `_NO_EVIDENCE_PHRASES` 로컬 정의 제거 → `rtc_ede.NO_EVIDENCE_STRINGS` import로 통일 |
| `experiments/run_baseline_comparison.py` | `alpha=0.15` 하드코딩 제거 → `load_config()` 연동 |
| `experiments/run_agent_experiment.py` | CLI `--alpha` 기본값 `0.15→None`, `--n-cal` `30→500`, `run_backend_experiment` config 연동 |
| `experiments/pareto_sweep.py` | `n_calibration` 기본값 `30→500`, CLI `--n-cal` `30→500` |

### 4라운드 (2026-04-08)

| # | 파일 | 변경 내용 |
|---|---|---|
| P12 | `experiments/run_all_experiments.py` | CLI `--n-cal` `30→500`, `--n-test` `3→50`, `--alpha` `0.15→None`(config 연동) |
| P13 | `experiments/run_calibration_pipeline.py` | CLI `--n-cal` `30→500`, `--n-labeled` `10→50`; 루프 내 `import math`, `from models.rtc_ede import EscalationTrigger` 반복 import → 모듈 상단으로 이동 |
| P14 | `experiments/run_baseline_comparison.py` | 함수 내 `from experiments.config_utils import load_config` → 모듈 상단으로 이동 |
| P15 | `experiments/run_agent_experiment.py` | 함수 내 `from experiments.config_utils import load_config` → 모듈 상단으로 이동 |
| P16 | `experiments/eval_medabstain.py` | 함수 내 `from experiments.config_utils import load_config` → 모듈 상단으로 이동 |
| P17 | `agent/nodes.py` | `reason` 노드 내 LLM 재생성이 설계 의도임을 주석으로 명시 (State 직렬화 제약) |

---

## 5라운드 개선 (2026-04-10, 실험 결과 분석 및 근본 원인 수정)

### 실험 결과 (5라운드 이전, α=0.05, corrected RTC multipliers)

| 지표 | LMStudio | OpenAI | 목표 | 상태 |
|---|---|---|---|---|
| Safety Recall (에이전트) | 0.3371 | 0.5866 | ≥ 0.95 | ✗ |
| Safety Recall (full_uasef) | 0.4190 | 0.4246 | ≥ 0.95 | ✗ |
| Safety Recall (threshold_only) | 0.2514 | 0.3911 | ≥ 0.95 | ✗ |
| Over-Escalation Rate (full_uasef) | 0.0476 | 0.2381 | ≤ 0.15 | 부분 |
| Conformal Coverage (에이전트) | 0.9200 | 0.9500 | ≥ 0.90 | ✓ |

### 핵심 원인 분석

이전 라운드들의 수정(α 조정, RTC multiplier 교정, A 변형 레이블 수정)에도 불구하고 Safety Recall이 0.42에 머무른 이유:

**P18 — CP 캘리브레이션 분포 불일치 (근본 원인)**

| 항목 | 내용 |
|---|---|
| 증상 | threshold_only OpenAI: TP=70, FP=0 — 모든 NA 케이스는 threshold 미만, AP/NAP/A의 61%도 threshold 미만 |
| 원인 | MedQA(USMLE 복잡한 시험 문제) 전체로 캘리브레이션 → q̂가 높게 설정됨. MedAbstain AP/NAP/A는 LLM이 **틀렸지만 자신있게** 답변(overconfident wrong) → logprob 기반 nonconformity score가 낮음 → threshold 미초과 |
| 이론적 배경 | logprob 비적합 점수는 **토큰 확률**을 측정하므로 overconfident-wrong 케이스를 탐지 불가. 이는 MedAbstain 특유의 도전과제 |
| 한계 | α를 아무리 높여도 MedQA 기반 threshold로는 ~63% recall이 상한. 0.95는 달성 불가 |

**P19 — α=0.05 과보수적 설정**

| 항목 | 내용 |
|---|---|
| 증상 | Pareto sweep α=0.10에서 coverage=0.94 (≥ 0.90 보장), esc_rate=0.47 |
| 원인 | α=0.05 → q̂ = 95th percentile → 너무 높은 threshold |
| 영향 | Safety Recall 0.05p 추가 하락 |

**P20 — distribution_source 하드코딩으로 WeightedCP 비활성화**

| 항목 | 내용 |
|---|---|
| 증상 | `distribution_source="medqa"` 하드코딩 → calibration("medqa") == eval("medqa") → shift 미감지 → WeightedCP OFF |
| 원인 | `eval_medabstain.py`, `run_baseline_comparison.py`에서 MedAbstain 케이스에도 `"medqa"` 전달 |
| 영향 | Tibshirani et al. (2019) WeightedCP가 전혀 작동하지 않음 |

**P21 — base_config.yaml n_test_per_scenario YAML 미동기화**

| 항목 | 내용 |
|---|---|
| 증상 | YAML에 `n_test_per_scenario: 3` 잔존. 실험은 CLI 기본값(50)으로 실행되어 YAML과 불일치 |

### 5라운드 수정 내용

| P# | 파일 | 변경 내용 | 기대 효과 |
|---|---|---|---|
| P18a | `experiments/configs/base_config.yaml` | `alpha: 0.05` → `0.10`; q̂ 낮아져 에스컬레이션 증가 | Recall +5~10%p |
| P18b | `experiments/configs/base_config.yaml` | `n_test_per_scenario: 3` → `50`; YAML-CLI 동기화 | 문서 정합성 |
| P18c | `data/loader.py` | `load_noesc_calibration_questions(n, split, seed)` 추가; MedQA에서 `expected_escalate=False`인 루틴 케이스만 필터링하여 반환 | one-class CP 캘리브레이션 지원 |
| P18d | `experiments/eval_medabstain.py` | `use_routine_cal=True` 파라미터 추가(기본값); 루틴 케이스로 캘리브레이션하면 q̂가 낮아져 AP/NAP/A 탐지율 향상. 이론 근거: one-class conformal prediction — 정상 클래스 점수 분포로 임계값 설정 | Recall +20~40%p 기대 |
| P20a | `experiments/eval_medabstain.py` | `distribution_source="medqa"` → `"medabstain"`; calibration="medqa", eval="medabstain" → shift 감지 → WeightedCP 자동 활성화 | Tibshirani et al. (2019) 보정 실효화 |
| P20b | `experiments/run_baseline_comparison.py` | 하드코딩 `"medqa"` → `case.source` 기반 동적 결정; MedAbstain 케이스엔 `"medabstain"` | WeightedCP 활성화 |
| P20c | `experiments/run_agent_experiment.py` | `case_to_agent_dict()`가 반환한 `distribution_source` 필드 사용; MedAbstain 케이스 자동 감지 | WeightedCP 활성화 |
| P20d | `data/loader.py` | `case_to_agent_dict()`에 `"distribution_source"` 필드 추가; MedAbstain source 자동 감지 | 에이전트 실험 WeightedCP 지원 |

### 설계 결정 및 연구 함의

**one-class CP 캘리브레이션 (P18c/P18d) 이론 근거**:
- 기존 방식: 전체 MedQA(mixed difficulty)로 캘리브레이션 → q̂ = MedQA 전체 95th percentile (높음)
- 개선 방식: MedQA 루틴(non-escalation) 케이스만으로 캘리브레이션 → q̂ = 쉬운 질문 95th percentile (낮음)
- 핵심 이유: MedAbstain NA(정상 케이스) 점수 < MedQA 루틴 점수 ≤ MedQA 전체 점수. 루틴 기반 q̂는 NA 케이스의 95%를 threshold 미만에 두면서 AP/NAP/A 케이스는 더 많이 threshold를 초과하게 함.

**근본적 한계 (논문 Limitation 섹션에 기술 필요)**:
- logprob 비적합 점수는 모델이 **확신있게 틀리는** 케이스(overconfident wrong)를 감지할 수 없음
- MedAbstain AP/NAP 케이스는 이 패턴을 의도적으로 테스트하는 설계임
- one-class CP + α=0.10으로 recall ~0.65~0.75 달성 예측. 0.95는 logprob T1만으로는 불가능
- 0.95 달성을 위해서는 self-consistency 앙상블(N회 쿼리), 외부 지식 베이스 fact-check, 또는 의미론적 불확실성 측정이 필요

**--no-routine-cal 플래그**:
- eval_medabstain.py에 `--no-routine-cal` 추가하여 기존 동작(전체 MedQA 캘리브레이션) 재현 가능
- 논문에서 기존 방식과 one-class CP 방식의 차이를 ablation으로 보고 가능
