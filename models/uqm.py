"""
UASEF — Module 1: Uncertainty Quantification Module (UQM)

Conformal Prediction을 통해 통계적 Coverage 보장이 있는 불확실성을 측정합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
비적합 점수(Nonconformity Score) 방식 — scoring_method 파라미터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LOGPROB (Primary — 논문 주요 기여):
    s(x) = -mean(token logprobs)
    Coverage guarantee: P(s_test ≤ q̂) ≥ 1-α  (Angelopoulos & Bates, 2021)
    요건: 모델이 token-level logprobs를 지원해야 함
          지원 모델: GPT-4o, GPT-4o-mini, llama.cpp (--logprobs 플래그)

  SELF_CONSISTENCY (Ablation — ablation study 전용):
    s(x) = Jaccard_diversity(responses × N)
    Coverage guarantee: 수학적으로 동일하게 성립 (다른 비적합 함수 사용)
    요건: logprobs 불필요. 단, N회 쿼리로 비용/지연 N배 증가.
    ⚠ 논문에서 "ablation"으로 명시하지 않으면 심사 지적을 받을 수 있음:
       "CP라고 주장했지만 실제론 majority voting"

  AUTO (하위 호환 — 권장하지 않음):
    런타임에 logprobs 지원 여부를 감지하여 자동 선택.
    실험 재현성(reproducibility) 저하 위험.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distribution Shift 처리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  calibrate(distribution_source="medqa") 후 evaluate(distribution_source="mimic3")
  호출 시 UserWarning 발생 — exchangeability 가정 위반 가능성 알림.

  권고 대처 방법:
    1. 도메인별 재보정: 각 도메인에 맞는 calibration set으로 UQM 재학습
    2. Weighted CP: Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
       중요도 가중치 w_i = p_test(x_i) / p_cal(x_i) 적용
"""

import math
import random
import warnings
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from models.model_interface import query_model, ModelResponse


# ── Scoring Method 열거형 ──────────────────────────────────────────────────────

class ScoringMethod(str, Enum):
    LOGPROB          = "logprob"           # Primary: logprob-based CP (논문 주요 기여)
    SELF_CONSISTENCY = "self_consistency"  # Ablation: SC-based CP (다른 비적합 함수)
    AUTO             = "auto"              # 런타임 감지 (하위 호환, 비권장)


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class UncertaintyResult:
    nonconformity_score: float     # 클수록 불확실 (0~∞)
    prediction_set_size: int       # score/threshold 비율 기반 집합 크기
    confidence_entropy: float      # logprobs 기반 Shannon 엔트로피 (nan=미지원)
    should_escalate: bool
    threshold_used: float
    raw_response: ModelResponse
    scoring_method: str = "logprob"   # 실제 사용된 방식 기록 (재현성)


@dataclass
class CalibrationMetadata:
    """
    Calibration 이력 — distribution shift 감지 및 실험 재현성 추적에 사용됩니다.
    """
    distribution_source: str = "unknown"   # "medqa" | "mimic3" | "pubmedqa" | "custom"
    n_calibration: int = 0
    n_holdout: int = 0
    alpha: float = 0.05
    threshold: float = 0.0
    scoring_method: str = "logprob"
    timestamp: str = ""
    coverage_report: dict = field(default_factory=dict)


# ── 비적합 점수 계산 ────────────────────────────────────────────────────────────

def compute_entropy(logprobs: Optional[list[float]]) -> float:
    """토큰 수준 log probability → Shannon 엔트로피 (nats/token)."""
    if not logprobs:
        return float("nan")
    probs = [math.exp(lp) for lp in logprobs]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return entropy / len(probs)


def compute_nonconformity_score(response: ModelResponse) -> float:
    """
    LOGPROB 방식 비적합 점수: 평균 negative log-likelihood.
    logprobs 미지원 시 ValueError — 명시적 오류로 방법 혼용 방지.
    """
    if not response.logprobs:
        raise ValueError(
            "Backend이 logprobs를 반환하지 않습니다.\n"
            "  옵션 1 (권장): logprobs 지원 백엔드 사용\n"
            "                 - OpenAI: gpt-4o, gpt-4o-mini (기본 지원)\n"
            "                 - LMStudio: llama.cpp 기반 모델 + logprobs=True 설정\n"
            "  옵션 2 (Ablation): UQM(scoring_method='self_consistency')\n"
            "                 논문에서 ablation study로 명시적으로 구분 필요"
        )
    return -float(np.mean(response.logprobs))


def _answer_diversity(texts: list[str]) -> float:
    """Jaccard 기반 답변 다양성 (0=완전일치, 1=완전다양)."""
    if len(texts) < 2:
        return 0.0
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a = set(texts[i].lower().split())
            b = set(texts[j].lower().split())
            union = a | b
            sim = len(a & b) / len(union) if union else 1.0
            similarities.append(sim)
    return 1.0 - sum(similarities) / len(similarities)


def compute_self_consistency_score(
    backend: str,
    system_prompt: str,
    question: str,
    n: int = 5,
) -> float:
    """
    SELF_CONSISTENCY 방식 비적합 점수 (Ablation).
    n회 쿼리 후 Jaccard 다양성 → 0~5 범위로 정규화.
    Coverage guarantee는 성립하지만 logprob 방식과 직접 비교 불가.
    """
    texts = [
        query_model(backend, system_prompt, question, temperature=0.7, logprobs=False).text.strip()[:200]
        for _ in range(n)
    ]
    return _answer_diversity(texts) * 5.0


# ── Conformal Prediction 임계값 보정 ───────────────────────────────────────────

class ConformalCalibrator:
    """
    Angelopoulos & Bates (2021) 공식에 따른 Conformal 임계값 q̂ 계산.
    q̂ = ceil((n+1)(1-α))/n 번째 분위수
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.threshold: float = float("inf")
        self.calibration_scores: list[float] = []

    def fit(self, nonconformity_scores: list[float]) -> None:
        n = len(nonconformity_scores)
        if n == 0:
            raise ValueError("빈 calibration set입니다.")
        self.calibration_scores = sorted(nonconformity_scores)
        level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.threshold = float(np.quantile(self.calibration_scores, level))
        print(f"[UQM] Calibration 완료: n={n}, α={self.alpha}, q̂={self.threshold:.4f}")

    def check_coverage(
        self,
        test_scores: list[float],
        test_labels_correct: list[bool],
    ) -> dict:
        """Hold-out set에서 실제 coverage ≥ 1-α 검증."""
        if not test_scores:
            return {"error": "빈 test set"}
        covered = sum(s <= self.threshold for s in test_scores)
        actual = covered / len(test_scores)
        target = 1 - self.alpha
        return {
            "target_coverage": target,
            "actual_coverage": round(actual, 4),
            "coverage_valid": actual >= target,
            "n_test": len(test_scores),
        }


# ── UQM 메인 클래스 ────────────────────────────────────────────────────────────

class UQM:
    """
    Uncertainty Quantification Module.

    권장 사용법 (논문 재현):
        uqm = UQM(backend="openai", alpha=0.05, scoring_method="logprob")
        report = uqm.calibrate(cal_questions, distribution_source="medqa")
        result = uqm.evaluate(question, distribution_source="medqa")

    Ablation 비교:
        uqm_sc = UQM(backend="lmstudio", alpha=0.05, scoring_method="self_consistency")
    """

    SYSTEM_PROMPT = (
        "You are a clinical decision support AI. "
        "Answer the medical question. "
        "If you are not confident, say 'I am not certain' before your answer."
    )

    def __init__(
        self,
        backend: str,
        alpha: float = 0.05,
        consistency_n: int = 5,
        scoring_method: str = "logprob",
    ):
        self.backend = backend
        self.calibrator = ConformalCalibrator(alpha=alpha)
        self._calibrated = False
        self.consistency_n = consistency_n
        self._scoring_method = ScoringMethod(scoring_method)
        self._calibration_meta: Optional[CalibrationMetadata] = None

        # _use_self_consistency 초기화
        if self._scoring_method == ScoringMethod.LOGPROB:
            self._use_self_consistency = False
        elif self._scoring_method == ScoringMethod.SELF_CONSISTENCY:
            self._use_self_consistency = True
        else:  # AUTO
            self._use_self_consistency = None  # 런타임 감지

        # Ablation 경고
        if self._scoring_method == ScoringMethod.SELF_CONSISTENCY:
            warnings.warn(
                "\n[UQM] scoring_method='self_consistency' 선택됨.\n"
                "  CP coverage guarantee는 수학적으로 유효하지만,\n"
                "  이 논문의 primary 기여(logprob-based CP)와 다른 비적합 함수를 사용합니다.\n"
                "  논문에서 반드시 ablation study로 명시적으로 구분하여 보고하세요.\n"
                "  Primary 방법과 직접 성능 비교 시 nonconformity 함수의 차이를 서술해야 합니다.",
                UserWarning, stacklevel=2,
            )

        if self._scoring_method == ScoringMethod.AUTO:
            warnings.warn(
                "\n[UQM] scoring_method='auto' 사용 중.\n"
                "  실험 재현성(reproducibility)을 위해 방법을 명시적으로 지정하세요:\n"
                "  UQM(scoring_method='logprob') 또는 UQM(scoring_method='self_consistency')",
                UserWarning, stacklevel=2,
            )

    def _get_score(self, question: str) -> tuple[float, ModelResponse]:
        resp = query_model(self.backend, self.SYSTEM_PROMPT, question, temperature=0.0)

        # AUTO 모드: 최초 호출 시 logprobs 지원 여부 감지
        if self._use_self_consistency is None:
            self._use_self_consistency = resp.logprobs is None
            if self._use_self_consistency:
                warnings.warn(
                    "\n[UQM] AUTO: logprobs 미지원 감지 → self-consistency 모드로 전환.\n"
                    "  scoring_method='self_consistency'를 명시적으로 설정하고\n"
                    "  논문에서 ablation으로 보고하세요.",
                    UserWarning, stacklevel=3,
                )

        if self._use_self_consistency:
            score = compute_self_consistency_score(
                self.backend, self.SYSTEM_PROMPT, question, self.consistency_n
            )
        else:
            score = compute_nonconformity_score(resp)

        return score, resp

    @property
    def active_scoring_method(self) -> str:
        """실제 사용 중인 scoring method 문자열 반환 (재현성 추적용)."""
        if self._scoring_method == ScoringMethod.AUTO:
            if self._use_self_consistency is None:
                return "auto(undecided)"
            return "self_consistency" if self._use_self_consistency else "logprob"
        return self._scoring_method.value

    def calibrate(
        self,
        questions: list[str],
        holdout_fraction: float = 0.2,
        distribution_source: str = "unknown",
    ) -> dict:
        """
        Calibration set으로 임계값을 학습하고 hold-out으로 coverage를 검증합니다.

        Args:
            distribution_source: 데이터 출처. evaluate() 호출 시 다른 분포가 감지되면
                                  distribution shift 경고가 발생합니다.
                                  예: "medqa", "mimic3", "pubmedqa", "custom"
        """
        n_total = len(questions)
        n_holdout = max(1, int(n_total * holdout_fraction))
        n_cal = n_total - n_holdout

        rng = random.Random(42)
        idx = list(range(n_total))
        rng.shuffle(idx)
        holdout_set = set(idx[:n_holdout])

        print(
            f"[UQM] Calibration 시작 | "
            f"n_cal={n_cal}, n_holdout={n_holdout}, "
            f"backend={self.backend}, method={self._scoring_method.value}, "
            f"distribution={distribution_source}"
        )

        cal_scores, holdout_scores = [], []
        for i, q in enumerate(questions):
            score, _ = self._get_score(q)
            (holdout_scores if i in holdout_set else cal_scores).append(score)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_total}] score={score:.4f}")

        self.calibrator.fit(cal_scores)
        self._calibrated = True

        coverage_report = self.calibrator.check_coverage(
            holdout_scores, [True] * len(holdout_scores)
        )
        ok = "✓" if coverage_report.get("coverage_valid") else "✗"
        print(
            f"[UQM] Coverage 검증: "
            f"{coverage_report['actual_coverage']:.3f} "
            f"(목표 {coverage_report['target_coverage']:.2f}) {ok}"
        )

        self._calibration_meta = CalibrationMetadata(
            distribution_source=distribution_source,
            n_calibration=n_cal,
            n_holdout=n_holdout,
            alpha=self.calibrator.alpha,
            threshold=self.calibrator.threshold,
            scoring_method=self.active_scoring_method,
            timestamp=datetime.now().isoformat(),
            coverage_report=coverage_report,
        )
        return coverage_report

    def evaluate(
        self,
        question: str,
        distribution_source: Optional[str] = None,
    ) -> UncertaintyResult:
        """
        단일 질문의 불확실성을 측정합니다.

        Args:
            distribution_source: 이 질문의 데이터 출처.
                                  calibration 분포와 다르면 distribution shift 경고 발생.
        """
        if not self._calibrated:
            raise RuntimeError("calibrate()를 먼저 호출하세요.")

        # Distribution shift 경고
        if (
            distribution_source
            and self._calibration_meta
            and self._calibration_meta.distribution_source not in ("unknown", distribution_source)
        ):
            warnings.warn(
                f"\n[CP Warning] Distribution shift 감지!\n"
                f"  Calibration: '{self._calibration_meta.distribution_source}'\n"
                f"  Evaluation:  '{distribution_source}'\n"
                f"  CP exchangeability 가정이 위반될 수 있습니다.\n"
                f"  권고:\n"
                f"    1. 타깃 도메인 데이터로 재보정 (도메인별 calibration)\n"
                f"    2. Weighted CP 적용 (Tibshirani et al., 2019)\n"
                f"  이 경고를 논문의 limitation 섹션에 서술하세요.",
                UserWarning, stacklevel=2,
            )

        score, resp = self._get_score(question)
        entropy = compute_entropy(resp.logprobs)
        threshold = self.calibrator.threshold
        should_escalate = score > threshold
        prediction_set_size = max(1, round(score / threshold)) if threshold > 0 else 1

        return UncertaintyResult(
            nonconformity_score=score,
            prediction_set_size=prediction_set_size,
            confidence_entropy=entropy,
            should_escalate=should_escalate,
            threshold_used=threshold,
            raw_response=resp,
            scoring_method=self.active_scoring_method,
        )


# ── 빠른 확인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    CAL = ["What is the first-line treatment for type 2 diabetes?"] * 5 + \
          ["What antibiotic is used for community-acquired pneumonia?"] * 5 + \
          ["What is the mechanism of beta-blockers?"] * 5

    for sm in [ScoringMethod.LOGPROB, ScoringMethod.SELF_CONSISTENCY]:
        print(f"\n{'='*55}\nscoring_method={sm.value}")
        try:
            uqm = UQM(backend="openai", alpha=0.05, scoring_method=sm.value)
            uqm.calibrate(CAL, distribution_source="medqa")
            r = uqm.evaluate("What is aspirin used for?", distribution_source="medqa")
            print(f"Score={r.nonconformity_score:.3f}, Escalate={r.should_escalate}, Method={r.scoring_method}")
            # Distribution shift 테스트
            r2 = uqm.evaluate("Rare mitochondrial disease presentation.", distribution_source="mimic3")
        except Exception as e:
            print(f"[SKIP] {e}")
