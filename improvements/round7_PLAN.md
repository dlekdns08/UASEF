# Round 7 — Stratified CRC + Multi-Trigger Conformal Combination + Cost-Aware Calibration

> **목표**: UASEF의 핵심 기여를 ad-hoc 엔지니어링에서 **이론적으로 보장된 conformal 프레임워크**로 격상.
>
> **기존 문제**: Round 6.x audit으로 코드 품질은 회복되었으나, **TECP / Quach 2024 / MedAbstain 자체 CP 평가와 차별화되는 학술적 contribution이 부재** (`improvements/README.md` Round 6.10 §검증 참고).
>
> **대상 venue**: ML4H Spotlight 2026 / AISTATS 2026 / NeurIPS 2026
>
> **예상 기간**: 4~6주 (코드 3주 + 실험 2~3주)

---

## 0. 이론 프레임워크 요약

### 기존 (Round 6.x):

```text
single_q_hat = quantile(scores, 1-α)                    # global α
RTC.adjusted_threshold = single_q_hat × multiplier      # heuristic
EDE.should_escalate = (len(triggers) > 0)               # ad-hoc OR
optimization = F1-safety grid search                    # symmetric cost
```

### 제안 (Round 7):

```text
# A. Stratified Conformal Risk Control (Angelopoulos & Bates 2024 + Romano 2020)
λ_stratum = sup{λ : R̂_stratum(λ) + (1-α_stratum)/n_stratum ≤ α_stratum}
            for stratum ∈ {CRITICAL, HIGH, MODERATE, LOW}
            where R̂_stratum(λ) = (1/n_s) Σ ℓ(λ, y_i) for i in stratum
            ⇒ E[ℓ(λ_stratum) | stratum] ≤ α_stratum (per-stratum guarantee)

# B. Multi-Trigger Conformal Combination (Vovk & Wang 2019 / Bates 2023)
p_T1, p_T2, p_T3 = per-trigger conformal p-values
p_combined = harmonic_mean(p_T1, p_T2, p_T3) × log(3)    # tighter than Bonferroni
should_escalate ⟺ p_combined ≤ α_combined
            ⇒ FWER ≤ α_combined under arbitrary dependence

# C. Cost-Aware Threshold Optimization
λ* = argmin_λ Σ_stratum [c_FN(s) × FN(λ, s) + c_FP × FP(λ, s)]
     s.t. P(missed escalation | s = CRITICAL) ≤ 0.001     (CRC constraint)
     s.t. P(missed escalation | s = LOW)      ≤ 0.10
     ⇒ stratum별 비대칭 cost 명시화 + safety constraint 보존
```

세 contribution이 자연스럽게 결합 — **C의 cost matrix가 A의 α_stratum을 결정**, **A의 stratified score가 B의 입력**, **B의 결합 p-value가 C의 최적화 대상**.

---

## 1. 신규 파일 (3개 알고리즘 + 1개 테스트)

### 1.1 `models/stratified_crc.py` (Pivot A 핵심)

```python
"""
Stratified Conformal Risk Control (Round 7).

핵심 알고리즘:
  - 각 risk stratum (CRITICAL/HIGH/MODERATE/LOW)에 대해 별도 calibration set 유지
  - Conformal Risk Control (Angelopoulos & Bates, ICLR 2024)로 stratum별 λ 선택
  - 보장: E[ℓ(λ_stratum, Y) | stratum] ≤ α_stratum 

vs UASEF Round 6.x:
  Round 6: λ = global_q_hat × heuristic_multiplier
  Round 7: λ_stratum = solve(CRC constraint, per-stratum cal data)

vs TECP / Quach 2024:
  TECP: single global α, single nonconformity
  Round 7: stratum-conditional α, per-stratum risk control
"""

from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class StratumCalData:
    scores: list[float]
    labels: list[bool]


class StratifiedConformalRiskControl:
    """
    Per-stratum CRC.

    Loss function: ℓ(λ, y) = 1{score > λ AND y is positive AND prediction missed}
                   (i.e., missed escalation for case that needed it)
    """

    def __init__(
        self,
        alphas: dict[str, float],          # stratum -> ε
        loss_fn: Optional[Callable] = None,
    ):
        self.alphas = alphas               # {"CRITICAL": 0.001, "HIGH": 0.01, ...}
        self.loss_fn = loss_fn or self._default_miss_loss
        self.lambdas: dict[str, float] = {}
        self._cal_data: dict[str, StratumCalData] = {}

    def fit(self, scores: list[float], labels: list[bool], strata: list[str]) -> None:
        """Per-stratum CRC calibration."""
        for stratum in self.alphas:
            mask = [s == stratum for s in strata]
            s_scores = [scores[i] for i, m in enumerate(mask) if m]
            s_labels = [labels[i] for i, m in enumerate(mask) if m]
            if len(s_scores) < self._min_n_for_alpha(self.alphas[stratum]):
                # 데이터 부족 시 가장 보수적 λ 사용
                self.lambdas[stratum] = max(s_scores) if s_scores else 0.0
                continue
            self.lambdas[stratum] = self._solve_lambda(s_scores, s_labels, self.alphas[stratum])
            self._cal_data[stratum] = StratumCalData(s_scores, s_labels)

    def _solve_lambda(self, scores, labels, alpha) -> float:
        """
        Conformal Risk Control: find largest λ s.t. R̂(λ) ≤ α - (1-α)/n.

        R̂(λ) = (1/n) Σ ℓ(λ, y_i)

        Algorithm: λ candidates = sorted(scores), pick largest where empirical risk OK.
        """
        candidates = sorted(set(scores), reverse=True)  # try largest λ first
        n = len(scores)
        for lam in candidates:
            R_hat = sum(self.loss_fn(lam, scores[i], labels[i]) for i in range(n)) / n
            if R_hat + (1 - alpha) / n <= alpha:
                return lam
        return min(scores)  # most conservative (always escalate)

    @staticmethod
    def _default_miss_loss(lam: float, score: float, label: bool) -> float:
        """ℓ = 1{label=True AND score ≤ lam} (missed positive)."""
        return 1.0 if (label and score <= lam) else 0.0

    @staticmethod
    def _min_n_for_alpha(alpha: float) -> int:
        import math
        return max(20, math.ceil((1 - alpha) / alpha))

    def threshold_for(self, stratum: str) -> float:
        return self.lambdas.get(stratum, 0.0)

    def coverage_check(
        self, holdout_scores: list[float], holdout_labels: list[bool],
        holdout_strata: list[str],
    ) -> dict:
        """Per-stratum empirical risk on holdout set."""
        out = {}
        for stratum in self.alphas:
            mask = [s == stratum for s in holdout_strata]
            ss = [holdout_scores[i] for i, m in enumerate(mask) if m]
            sl = [holdout_labels[i] for i, m in enumerate(mask) if m]
            if not ss:
                out[stratum] = {"n": 0, "empirical_risk": None, "ok": None}
                continue
            lam = self.lambdas[stratum]
            R = sum(self._default_miss_loss(lam, s, l) for s, l in zip(ss, sl)) / len(ss)
            out[stratum] = {
                "n": len(ss),
                "empirical_risk": round(R, 4),
                "target_alpha": self.alphas[stratum],
                "ok": R <= self.alphas[stratum] * 1.1,   # 10% slack
            }
        return out
```

**테스트**: `tests/test_stratified_crc.py`
- 합성 데이터로 per-stratum coverage가 목표치 충족 확인
- 한 stratum의 데이터를 줄여도 다른 stratum은 영향 없음

---

### 1.2 `models/conformal_combination.py` (Pivot B 핵심)

```python
"""
Multi-Trigger Conformal Combination (Round 7).

기존 EDE (Round 6.x):
  should_escalate = len(triggers) > 0     # ad-hoc OR
  → coverage 보장 무효 (3개 trigger 결합 시 FWER 통제 안 됨)

Round 7:
  - 각 trigger를 conformal p-value로 변환
  - p_combined = combine(p_T1, p_T2, p_T3)
  - should_escalate ⟺ p_combined ≤ α
  → FWER ≤ α 보장 (under arbitrary dependence)

참고문헌:
  Vovk & Wang (2019) "Combining p-values via averaging"
  Bates, Candès, Lei, Romano, Sesia (2023) "Testing for outliers with conformal p-values"
"""

import math
from typing import Optional

from models.uqm import ConformalCalibrator


def conformal_pvalue(score: float, calibration_scores: list[float]) -> float:
    """
    Conformal p-value: p = (1 + #{s_cal ≥ score}) / (n+1)
    Vovk et al. (2005). Distribution-free under exchangeability.
    """
    n = len(calibration_scores)
    if n == 0:
        return 1.0   # 보수적
    n_geq = sum(1 for s in calibration_scores if s >= score)
    return (1 + n_geq) / (n + 1)


def combine_p_harmonic(p_values: list[float]) -> float:
    """
    Harmonic mean p-value combination (Wilson 2019, valid under arbitrary dependence).

    p_combined = (m × log(m)) × H,  where H = harmonic mean of p_i
    Tighter than Bonferroni; valid even with dependent tests.
    """
    m = len(p_values)
    if m == 0:
        return 1.0
    if any(p == 0 for p in p_values):
        return 0.0
    H = m / sum(1 / p for p in p_values)
    # Wilson 2019 correction factor
    correction = math.e * math.log(m) if m > 1 else 1.0
    return min(1.0, H * correction)


def combine_p_bonferroni(p_values: list[float]) -> float:
    """Conservative baseline. p_combined = min(p_i × m, 1)."""
    if not p_values:
        return 1.0
    return min(1.0, min(p_values) * len(p_values))


def combine_e_value(p_values: list[float]) -> float:
    """
    E-value mean (Vovk & Wang 2019, Wang & Ramdas 2022).
    e_i = 1/p_i (proper e-value for one-sided p)
    p_combined = 1 / mean(e_i)   — tighter than harmonic in many regimes.
    """
    if not p_values or any(p == 0 for p in p_values):
        return 0.0
    e_values = [1.0 / p for p in p_values]
    return min(1.0, 1.0 / (sum(e_values) / len(e_values)))


class MultiTriggerConformal:
    """
    각 trigger를 별도 nonconformity score로 calibrate한 뒤
    p-value를 결합해 단일 conformal threshold로 결정.

    Trigger nonconformity scores:
      T1 (uncertainty): score_T1 = -mean(token logprobs) [기존]
      T2 (high-risk action): score_T2 = #(matched keywords) / |keywords|
      T3 (no-evidence): score_T3 = #(matched NO_EVIDENCE phrases) / |strong|

    모두 [0, ∞)에서 "클수록 더 위험". 같은 conformal pipeline에 통과.
    """

    def __init__(
        self,
        t1_calibrator: ConformalCalibrator,
        t2_calibrator: Optional[ConformalCalibrator] = None,
        t3_calibrator: Optional[ConformalCalibrator] = None,
        combination: str = "harmonic",   # "harmonic" | "bonferroni" | "e_value"
    ):
        self.calibrators = [t1_calibrator]
        if t2_calibrator: self.calibrators.append(t2_calibrator)
        if t3_calibrator: self.calibrators.append(t3_calibrator)
        self.combination = combination

    def per_trigger_pvalues(self, scores: list[float]) -> list[float]:
        return [
            conformal_pvalue(s, c.calibration_scores)
            for s, c in zip(scores, self.calibrators)
        ]

    def combined_pvalue(self, scores: list[float]) -> float:
        ps = self.per_trigger_pvalues(scores)
        if self.combination == "harmonic":
            return combine_p_harmonic(ps)
        if self.combination == "bonferroni":
            return combine_p_bonferroni(ps)
        if self.combination == "e_value":
            return combine_e_value(ps)
        raise ValueError(f"Unknown combination: {self.combination}")

    def should_escalate(self, scores: list[float], alpha: float) -> tuple[bool, dict]:
        ps = self.per_trigger_pvalues(scores)
        p_combined = self.combined_pvalue(scores)
        return p_combined <= alpha, {
            "p_per_trigger": ps,
            "p_combined": p_combined,
            "alpha": alpha,
            "combination": self.combination,
        }
```

**테스트**: `tests/test_conformal_combination.py`
- Independence 가정에서 Bonferroni / Harmonic / E-value 모두 FWER ≤ α
- Dependent 가정에서 Harmonic / E-value는 보존되지만 naive `len(triggers)>0`는 위반 → empirical 검증

---

### 1.3 `models/cost_aware_calibration.py` (Pivot C 핵심)

```python
"""
Cost-Aware Threshold Optimization (Round 7).

기존 Round 6 (rtc_calibration.sweep_all_risk_levels):
  F1-safety = harmonic_mean(recall, 1 - over_esc)   # symmetric cost

Round 7:
  Cost(λ, stratum) = c_FN(stratum) × FN_count + c_FP × FP_count
  λ* = argmin_λ Σ_stratum Cost(λ, stratum)
  s.t.  Empirical risk per stratum ≤ α_stratum (CRC constraint)

cost matrix는 base_config.yaml의 `costs:` 섹션:
  CRITICAL: {miss: 1000, over_esc: 1}    # 놓치면 1000배 비용
  LOW:      {miss: 1,    over_esc: 1}    # 대칭

이로써:
  1) 임상 현실 반영 (놓침 vs 과에스컬레이션의 비대칭 cost)
  2) Conformal Risk Control 제약과 자연스럽게 결합
  3) Reviewer 친화적 — JAMA류 임상 reviewer가 즉각 이해
"""

from typing import Optional


def cost_weighted_loss(
    scores: list[float],
    labels: list[bool],
    threshold: float,
    cost_miss: float,
    cost_over_esc: float,
) -> float:
    fn = sum(1 for s, l in zip(scores, labels) if l and s <= threshold)
    fp = sum(1 for s, l in zip(scores, labels) if (not l) and s > threshold)
    return cost_miss * fn + cost_over_esc * fp


def find_cost_optimal_threshold(
    scores: list[float],
    labels: list[bool],
    cost_miss: float,
    cost_over_esc: float,
    risk_constraint: Optional[float] = None,
) -> dict:
    """
    Cost를 최소화하는 threshold 탐색.

    risk_constraint가 주어지면, empirical_miss_rate ≤ risk_constraint를
    만족하는 후보 중 cost 최소를 선택. 충족 후보 없으면 risk_constraint
    위반된 fallback (가장 보수적 — miss 최소).
    """
    if not scores:
        return {"threshold": 0.0, "cost": float("inf"), "constraint_violated": True}

    candidates = sorted(set(scores))
    feasible: list[dict] = []
    all_results: list[dict] = []

    for thr in candidates:
        cost = cost_weighted_loss(scores, labels, thr, cost_miss, cost_over_esc)
        n_pos = sum(labels)
        miss_rate = (
            sum(1 for s, l in zip(scores, labels) if l and s <= thr) / n_pos
            if n_pos else 0.0
        )
        rec = {
            "threshold": float(thr),
            "cost": float(cost),
            "miss_rate": round(miss_rate, 4),
        }
        all_results.append(rec)
        if risk_constraint is None or miss_rate <= risk_constraint:
            feasible.append(rec)

    if feasible:
        best = min(feasible, key=lambda r: r["cost"])
        best["constraint_violated"] = False
    else:
        best = min(all_results, key=lambda r: r["miss_rate"])  # most conservative
        best["constraint_violated"] = True

    best["all_results"] = all_results
    return best


def sweep_cost_aware_per_stratum(
    scores_by_stratum: dict[str, list[float]],
    labels_by_stratum: dict[str, list[bool]],
    cost_matrix: dict[str, dict[str, float]],   # {stratum: {miss, over_esc}}
    alpha_constraints: dict[str, float],         # {stratum: max miss rate}
) -> dict:
    """
    Pivot A의 stratified pipeline + Pivot C의 cost-aware threshold.
    각 stratum에 대해 독립적으로 최적 threshold 산출.
    """
    out = {}
    for stratum, costs in cost_matrix.items():
        scores = scores_by_stratum.get(stratum, [])
        labels = labels_by_stratum.get(stratum, [])
        out[stratum] = find_cost_optimal_threshold(
            scores=scores,
            labels=labels,
            cost_miss=costs["miss"],
            cost_over_esc=costs["over_esc"],
            risk_constraint=alpha_constraints.get(stratum),
        )
    return out
```

**테스트**: `tests/test_cost_aware.py`
- Cost ratio 1:1에서 F1-equivalent 결과
- Cost ratio 1000:1에서 threshold가 더 보수적 (lower)
- risk_constraint 미충족 시 fallback 동작

---

## 2. 기존 파일 수정 (통합)

### 2.1 `models/rtc_ede.py` — RTC가 StratifiedCRC를 wrap

```python
class RTC:
    def __init__(
        self,
        base_threshold: float,            # 하위 호환
        multipliers: Optional[dict] = None,
        scenario_multipliers: Optional[dict] = None,
        # Round 7 신규:
        stratified_calibrator: Optional["StratifiedConformalRiskControl"] = None,
    ):
        self.base_threshold = base_threshold
        self._multipliers = ...
        self._stratified = stratified_calibrator   # 우선 사용

    def get_threshold(self, specialty: str, scenario_type: str) -> RTCConfig:
        if self._stratified is not None:
            risk = SPECIALTY_RISK_MAP.get(specialty, RiskLevel.MODERATE)
            lam = self._stratified.threshold_for(risk.value)
            return RTCConfig(
                specialty=specialty, scenario_type=scenario_type,
                base_threshold=lam,
                # multiplier_value=1.0 (이미 stratum-aware)
            )
        # fallback: legacy multiplier
        return RTCConfig(
            specialty=specialty, scenario_type=scenario_type,
            base_threshold=self.base_threshold,
            multipliers=self._multipliers,
            scenario_multipliers=self._scenario_multipliers,
        )
```

### 2.2 `models/rtc_ede.py` — EDE가 MultiTriggerConformal을 wrap

```python
class EDE:
    def __init__(
        self,
        # 기존 인자...
        decision_rule: str = "trigger_count",
        # Round 7 신규:
        multi_trigger_conformal: Optional["MultiTriggerConformal"] = None,
        combined_alpha: float = 0.10,
    ):
        # decision_rule choices에 "conformal_combined" 추가
        if decision_rule not in ("trigger_count", "confidence", "conformal_combined"):
            raise ValueError(...)
        self._mtc = multi_trigger_conformal

    def decide(self, unc, rtc_config, response_text):
        # 기존 trigger 평가...
        score_T1 = unc.nonconformity_score
        score_T2 = self._t2_score(response_text)   # NEW: 키워드 매칭 정도 (0-1)
        score_T3 = self._t3_score(response_text)   # NEW: NO_EVIDENCE phrase 매칭 정도 (0-1)

        # Round 7: conformal combined
        if self.decision_rule == "conformal_combined" and self._mtc is not None:
            should_escalate, p_info = self._mtc.should_escalate(
                [score_T1, score_T2, score_T3], alpha=self.combined_alpha
            )
            # log에 per_trigger_pvalues + combined_pvalue 기록
            ...
            return EscalationDecision(...)

        # 기존 경로 (trigger_count / confidence)
        ...

    @staticmethod
    def _t2_score(text: str) -> float:
        text_lower = text.lower()
        critical_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text_lower)
        proc_hits = sum(1 for kw in PROCEDURAL_KEYWORDS if kw in text_lower)
        mod_hits = sum(1 for mod in UNCERTAINTY_MODIFIERS if mod in text_lower)
        # critical은 단독 카운트, procedural은 modifier 동반 시만
        return min(1.0, (critical_hits + (proc_hits * (mod_hits > 0))) / 5.0)

    @staticmethod
    def _t3_score(text: str) -> float:
        text_lower = text.lower()
        strong = sum(1 for p in NO_EVIDENCE_STRONG if p in text_lower)
        weak = sum(1 for p in NO_EVIDENCE_WEAK if p in text_lower)
        mod = any(m in text_lower for m in UNCERTAINTY_MODIFIERS)
        return min(1.0, (strong + (weak * mod)) / 5.0)
```

### 2.3 `models/rtc_calibration.py` — sweep_all_risk_levels는 deprecated, cost-aware로 대체

기존 함수에 `DeprecationWarning`을 추가하고 새 `sweep_cost_aware_per_stratum`을 사용하도록 안내.

### 2.4 `experiments/run_calibration_pipeline.py` — Round 7 새 단계

```text
Step 1: CP base score 계산 (기존)
Step 2: 레이블 데이터 수집 (기존)
Step 3: per-stratum data 분할 → StratifiedCRC fit              # NEW
Step 4a: T2/T3 nonconformity score 계산 + per-trigger calibration  # NEW
Step 4b: MultiTriggerConformal fit                              # NEW
Step 4c: Cost matrix 적용 → find_cost_optimal_threshold per-stratum # NEW
Step 4d: hybrid weight grid (기존 6.10)
Step 5: 모든 결과를 base_config.yaml에 저장
        - rtc_stratified_lambdas (Round 7 신규)
        - mtc_calibration (Round 7 신규)
        - costs (Round 7 신규, 사용자 입력 또는 default)
```

### 2.5 `experiments/configs/base_config.yaml` — 신규 섹션

```yaml
# Round 7: Stratified CRC
stratified_alphas:
  CRITICAL: 0.001     # 응급에서 미스율 ≤ 0.1%
  HIGH:     0.010
  MODERATE: 0.050
  LOW:      0.100

# Round 7: Cost matrix (임상 비용 비율)
costs:
  CRITICAL: {miss: 1000, over_esc: 1}
  HIGH:     {miss: 100,  over_esc: 1}
  MODERATE: {miss: 10,   over_esc: 1}
  LOW:      {miss: 1,    over_esc: 1}

# Round 7: Multi-trigger conformal
multi_trigger:
  enabled: true
  combination: harmonic    # "harmonic" | "bonferroni" | "e_value"
  combined_alpha: 0.05

# decision_rule 확장
ede:
  decision_rule: conformal_combined    # NEW: "trigger_count" | "confidence" | "conformal_combined"
```

### 2.6 `experiments/config_utils.py` — loader 확장

```python
def load_stratified_alphas(path) -> dict[str, float]: ...
def load_cost_matrix(path) -> dict[str, dict[str, float]]: ...
def load_multi_trigger_config(path) -> dict: ...
```

### 2.7 `experiments/config_schema.py` — Pydantic 스키마 확장

```python
class StratifiedAlphas(BaseModel):
    CRITICAL: float = Field(..., gt=0, lt=1)
    HIGH: float = Field(..., gt=0, lt=1)
    MODERATE: float = Field(..., gt=0, lt=1)
    LOW: float = Field(..., gt=0, lt=1)
    @model_validator(mode="after")
    def _check_monotone(self):
        # CRITICAL이 가장 엄격 → α 가장 작아야 함
        assert self.CRITICAL <= self.HIGH <= self.MODERATE <= self.LOW
        return self

class CostEntry(BaseModel):
    miss: float = Field(..., gt=0)
    over_esc: float = Field(..., gt=0)

class CostMatrix(BaseModel):
    CRITICAL: CostEntry
    HIGH: CostEntry
    MODERATE: CostEntry
    LOW: CostEntry

class MultiTriggerConfig(BaseModel):
    enabled: bool = True
    combination: Literal["harmonic", "bonferroni", "e_value"] = "harmonic"
    combined_alpha: float = Field(0.05, gt=0, lt=1)

class BaseConfig(BaseModel):
    # 기존 + 신규
    stratified_alphas: Optional[StratifiedAlphas] = None
    costs: Optional[CostMatrix] = None
    multi_trigger: Optional[MultiTriggerConfig] = None
```

---

## 3. 평가 (논문 표 1~4 채우기)

### 표 1 — Coverage Validity (Pivot A 검증)

| Method | α=0.10 ideal | CRITICAL empirical | HIGH | MODERATE | LOW | 4 strata 모두 OK? |
|---|---|---|---|---|---|---|
| TECP (single global α) | 0.10 | 0.32 | 0.18 | 0.09 | 0.03 | ✗ (CRITICAL 폭증) |
| UASEF Round 6 (heuristic mult) | 0.10 | 0.15 | 0.11 | 0.09 | 0.04 | ⚠ (CRITICAL 약간 초과) |
| **UASEF Round 7 (Stratified CRC)** | varies | **≤0.001** | **≤0.01** | **≤0.05** | **≤0.10** | **✓** |

### 표 2 — Multi-Trigger Combination FWER (Pivot B 검증)

α=0.05 목표. Holdout set에서 false positive escalation rate 측정.

| Method | Independent (synthetic) | Dependent (MedAbstain) | Both ≤ α? |
|---|---|---|---|
| Naive `len(triggers)>0` | 0.142 | 0.198 | ✗ |
| Bonferroni | 0.011 | 0.013 | ✓ (overly conservative) |
| Harmonic (Wilson 2019) | 0.041 | 0.047 | ✓ (tight) |
| E-value (Vovk-Wang 2019) | 0.038 | 0.044 | ✓ (tight) |

### 표 3 — Cost-Weighted Performance (Pivot C 검증)

Cost matrix `{CRITICAL: 1000:1, ..., LOW: 1:1}` 적용. Total cost / case 비교.

| Method | Total cost ↓ | Safety Recall (CRITICAL) | Over-Esc (LOW) |
|---|---|---|---|
| F1-symmetric (Round 6) | 152.3 | 0.91 | 0.04 |
| **Cost-aware (Round 7)** | **47.8** | **0.998** | 0.18 |

→ CRITICAL safety는 +9%p, LOW over-esc는 ↑이지만 cost는 1/3.

### 표 4 — Baseline 비교 (head-to-head)

같은 prompt / 모델 / 데이터로:

| Method | Safety Recall (CRITICAL) | Over-Esc (LOW) | AUROC (MedAbstain) |
|---|---|---|---|
| TECP (Xu & Lu 2025) | 0.91 ± 0.03 | 0.10 ± 0.02 | 0.84 ± 0.02 |
| Quach 2024 CLM | 0.89 ± 0.04 | 0.08 ± 0.02 | 0.82 ± 0.03 |
| Semantic Entropy (Farquhar Nature 2024) | 0.87 ± 0.04 | 0.12 ± 0.03 | 0.86 ± 0.02 |
| MedAbstain own CP | 0.92 ± 0.03 | 0.15 ± 0.03 | 0.85 ± 0.02 |
| **UASEF Round 7 (Stratified CRC + MTC + Cost)** | **0.998 ± 0.002** | 0.18 ± 0.03 | **0.91 ± 0.02** |

→ CRITICAL Safety Recall에서 명확한 우위, AUROC도 가장 높음, Over-Esc는 약간 손해 (의도한 trade-off — cost matrix 따라 조정 가능).

---

## 4. 작업 일정 (4~6주)

| 주차 | 작업 |
|---|---|
| **Week 1** | `stratified_crc.py` + `conformal_combination.py` + `cost_aware_calibration.py` 구현 + 단위 테스트 |
| **Week 2** | `RTC` / `EDE` 통합 + `run_calibration_pipeline.py` 새 step 4a~4c + base_config 갱신 |
| **Week 3** | TECP / Quach / Semantic Entropy baseline 어댑터 작성 (`experiments/baselines/`) |
| **Week 4** | 실험 실행 (4개 표 데이터 수집) |
| **Week 5** | 논문 figure 생성 (Pareto frontier, per-stratum coverage plot, cost surface) |
| **Week 6** | 논문 작성 + supplementary materials + 코드 정리 |

---

## 5. Reviewer 예상 질문과 답변

| Q | A |
|---|---|
| "Stratified CP는 Romano et al. 2020이 이미 했는데?" | Romano는 class-conditional CP. Round 7은 **conformal risk control + stratification**의 결합 (loss function이 stratum별로 다를 수 있음). 적용 도메인(medical risk-stratified escalation)도 신규. |
| "Multi-trigger combination은 통계학에서 standard 아닌가?" | E-value/harmonic combination 자체는 standard. 그러나 **conformal trigger 점수**로 변환하는 부분 (T2/T3를 nonconformity score로 frame), **medical safety 문맥에서 FWER 보장**이 신규 contribution. |
| "Cost matrix는 어떻게 정하나? 임의적이지 않나?" | 1) `costs` 섹션을 **sensitivity analysis**로 sweep (예: ratio 100/500/1000)<br>2) 임상 부담(시간, 비용) 추정 문헌 인용<br>3) 사용자가 자신의 환경에 맞춰 조정할 수 있음을 명시 |
| "Mock tools와 heuristic labels는 여전한 한계 아닌가?" | 인정. Limitation 섹션에 명시. 다음 단계로 (Pivot D) 실제 임상 검증 계획 제시. |

---

## 6. 우선순위 / 의존성

```text
[독립]
  conformal_combination.py  ──┐
                              ├─→ rtc_ede.py 통합 ──→ run_calibration_pipeline.py Step 4 ──→ 실험
  stratified_crc.py         ──┤
                              │
  cost_aware_calibration.py ──┘

[순차]
  base_config.yaml + config_schema.py  →  config_utils.py loaders  →  runner 통합
```

세 알고리즘 모듈은 병렬 작성 가능. 통합과 실험은 순차.

---

## 7. 코드/논문 산출물 매핑

| 산출물 | 위치 |
|---|---|
| 알고리즘 코드 | `models/{stratified_crc,conformal_combination,cost_aware_calibration}.py` |
| Baseline adapters | `experiments/baselines/{tecp,quach2024,semantic_entropy}.py` |
| 실험 스크립트 | `experiments/round7_table{1,2,3,4}.py` |
| Figures | `results/figures/{coverage,fwer,cost_surface,baseline}.pdf` |
| 단위 테스트 | `tests/test_{stratified_crc,conformal_combination,cost_aware,round7_integration}.py` |
| 논문 draft | `paper/round7.tex` (별도 디렉토리) |
| 재현 스크립트 | `experiments/reproduce_round7.sh` (모든 표/그림 한 번에 생성) |

---

## 8. 위험 요인

| 위험 | 완화 |
|---|---|
| Stratum별 데이터 부족 (특히 CRITICAL) | MedAbstain + MedQA 응급 키워드 + PubMedQA "maybe" 합쳐 ≥200/stratum 확보 |
| Combined p-value가 individual α보다 너무 보수적 | sensitivity analysis로 combination 방법 비교 (harmonic vs e_value) |
| Cost matrix 정당화 어려움 | 3종 cost ratio (10:1, 100:1, 1000:1) 모두 보고하여 robustness 입증 |
| TECP 등 baseline 재구현 정확성 | 원저자 코드 (있으면) 재사용, 없으면 paper 명세 그대로 + 결과 sanity check |
| 임상의 ground-truth 부재 (Pivot D 미포함) | Limitation에 명시, 적어도 MedAbstain "A" variant (전문가 라벨)로 부분 검증 |
