"""
UASEF — Module 1: Uncertainty Quantification Module (UQM)
Conformal Prediction으로 통계적 Coverage 보장이 있는 불확실성을 측정합니다.

핵심 아이디어:
  1. Calibration set에서 각 샘플의 비적합 점수(nonconformity score)를 계산
  2. 원하는 coverage (1-α)에 맞는 임계 분위수 q̂ 를 구함
  3. 새 샘플의 점수가 q̂ 를 넘으면 → 불확실성이 높다 → 에스컬레이션 후보
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from modules.model_interface import query_model, ModelResponse


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class UncertaintyResult:
    nonconformity_score: float     # 클수록 불확실 (0~∞)
    prediction_set_size: int       # Conformal Prediction Set 크기 (클수록 불확실)
    confidence_entropy: float      # logprobs 기반 엔트로피 (클수록 불확실)
    should_escalate: bool          # 에스컬레이션 여부
    threshold_used: float          # 적용된 임계값
    raw_response: ModelResponse


# ── 비적합 점수 계산 ────────────────────────────────────────────────────────────

def compute_entropy(logprobs: Optional[list[float]]) -> float:
    """
    토큰 수준 log probability → Shannon 엔트로피.
    logprobs를 지원하지 않는 경우 self-consistency 기반 점수를 사용하세요.
    """
    if not logprobs:
        return float("nan")
    probs = [math.exp(lp) for lp in logprobs]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return entropy / len(probs)   # 토큰 수로 정규화


def compute_nonconformity_score(response: ModelResponse) -> float:
    """
    비적합 점수: 높을수록 모델이 불확실하다는 의미.

    전략:
    - logprobs 사용 가능 → 음수 평균 log probability (높으면 불확실)
    - logprobs 불가 → self-consistency 방식 사용 (query_selfconsistency 별도 호출)
    """
    if response.logprobs:
        # 평균 negative log-likelihood
        return -np.mean(response.logprobs)
    else:
        # logprobs 미지원 모델의 폴백: 응답 길이 기반 휴리스틱
        # 실제 연구에서는 self-consistency(동일 질문 N회 반복 후 분산)로 대체 권장
        words = len(response.text.split())
        return max(0.0, 5.0 - math.log(words + 1))   # 짧은 답변 = 불확실


# ── Conformal Prediction 임계값 보정 ───────────────────────────────────────────

class ConformalCalibrator:
    """
    Calibration set으로부터 Conformal 임계값 q̂ 를 계산합니다.

    사용법:
        cal = ConformalCalibrator(alpha=0.05)
        cal.fit(scores)          # calibration set의 비적합 점수 목록
        threshold = cal.threshold
        cal.check_coverage(new_scores, true_labels)  # coverage 검증
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha          # 허용 오류율 (0.05 → 95% coverage 보장)
        self.threshold: float = float("inf")
        self.calibration_scores: list[float] = []

    def fit(self, nonconformity_scores: list[float]) -> None:
        """
        분위수 기반 임계값 계산.
        Angelopoulos & Bates (2021) 공식: q̂ = ceil((n+1)(1-α))/n 번째 분위수
        """
        n = len(nonconformity_scores)
        if n == 0:
            raise ValueError("빈 calibration set입니다.")
        self.calibration_scores = sorted(nonconformity_scores)
        level = math.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.threshold = float(np.quantile(self.calibration_scores, level))
        print(f"[UQM] Calibration 완료: n={n}, α={self.alpha}, q̂={self.threshold:.4f}")

    def check_coverage(
        self,
        test_scores: list[float],
        test_labels_correct: list[bool],
    ) -> dict:
        """
        실제 coverage가 (1-α) 이상인지 검증합니다.
        test_labels_correct[i] = True 이면 올바른 답변을 예측한 샘플.
        """
        covered = sum(
            score <= self.threshold for score in test_scores
        )
        actual_coverage = covered / len(test_scores)
        target_coverage = 1 - self.alpha
        return {
            "target_coverage": target_coverage,
            "actual_coverage": actual_coverage,
            "coverage_valid": actual_coverage >= target_coverage,
            "n_test": len(test_scores),
        }


# ── UQM 메인 클래스 ────────────────────────────────────────────────────────────

class UQM:
    """
    Uncertainty Quantification Module.

    workflow:
        uqm = UQM(backend="lmstudio", alpha=0.05)
        uqm.calibrate(calibration_questions, calibration_answers)
        result = uqm.evaluate(question)
    """

    SYSTEM_PROMPT = (
        "You are a clinical decision support AI. "
        "Answer the medical question. "
        "If you are not confident, say 'I am not certain' before your answer."
    )

    def __init__(self, backend: str, alpha: float = 0.05):
        self.backend = backend
        self.calibrator = ConformalCalibrator(alpha=alpha)
        self._calibrated = False

    def _get_score(self, question: str) -> tuple[float, ModelResponse]:
        resp = query_model(
            self.backend,
            self.SYSTEM_PROMPT,
            question,
            temperature=0.0,
        )
        score = compute_nonconformity_score(resp)
        return score, resp

    def calibrate(
        self,
        questions: list[str],
        expected_answers: Optional[list[str]] = None,
    ) -> None:
        """
        Calibration set으로 임계값을 학습합니다.
        expected_answers가 없으면 점수 분포만으로 임계값을 추정합니다.
        """
        print(f"[UQM] Calibration 시작 ({len(questions)}개 샘플, backend={self.backend})")
        scores = []
        for i, q in enumerate(questions):
            score, _ = self._get_score(q)
            scores.append(score)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] score={score:.4f}")
        self.calibrator.fit(scores)
        self._calibrated = True

    def evaluate(self, question: str) -> UncertaintyResult:
        """
        단일 질문의 불확실성을 측정하고 에스컬레이션 여부를 반환합니다.
        """
        if not self._calibrated:
            raise RuntimeError("calibrate()를 먼저 호출하세요.")
        score, resp = self._get_score(question)
        entropy = compute_entropy(resp.logprobs)
        should_escalate = score > self.calibrator.threshold
        return UncertaintyResult(
            nonconformity_score=score,
            prediction_set_size=1 if not should_escalate else 3,  # 단순화
            confidence_entropy=entropy,
            should_escalate=should_escalate,
            threshold_used=self.calibrator.threshold,
            raw_response=resp,
        )


# ── 빠른 확인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 임시 calibration 데이터 (실제 연구에서는 MedQA 사용)
    CAL_QUESTIONS = [
        "What is the first-line treatment for type 2 diabetes?",
        "What antibiotic is used for community-acquired pneumonia?",
        "What is the mechanism of beta-blockers?",
    ] * 5   # 데모용 반복

    TEST_QUESTIONS = [
        "A 3-year-old with recurrent infections, absent lymph nodes. Diagnosis?",  # 희귀 → 어려움
        "What is aspirin used for?",                                                # 쉬움
    ]

    for backend in ["lmstudio", "openai"]:
        print(f"\n{'='*60}\nBackend: {backend.upper()}")
        try:
            uqm = UQM(backend=backend, alpha=0.05)
            uqm.calibrate(CAL_QUESTIONS)
            for q in TEST_QUESTIONS:
                result = uqm.evaluate(q)
                status = "🔴 ESCALATE" if result.should_escalate else "🟢 AUTO"
                print(f"\n  Q: {q[:60]}...")
                print(f"  → Score={result.nonconformity_score:.3f}, "
                      f"Threshold={result.threshold_used:.3f}, {status}")
        except Exception as e:
            print(f"[SKIP] {e}")
