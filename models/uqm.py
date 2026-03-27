"""
UASEF — Module 1: Uncertainty Quantification Module (UQM)
Conformal Prediction으로 통계적 Coverage 보장이 있는 불확실성을 측정합니다.

핵심 아이디어:
  1. Calibration set에서 각 샘플의 비적합 점수(nonconformity score)를 계산
  2. 원하는 coverage (1-α)에 맞는 임계 분위수 q̂ 를 구함
  3. 새 샘플의 점수가 q̂ 를 넘으면 → 불확실성이 높다 → 에스컬레이션 후보
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from models.model_interface import query_model, ModelResponse


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class UncertaintyResult:
    nonconformity_score: float     # 클수록 불확실 (0~∞)
    prediction_set_size: int       # Conformal Prediction Set 크기 (클수록 불확실)
    confidence_entropy: float      # logprobs 기반 엔트로피 (클수록 불확실, nan=미지원)
    should_escalate: bool          # 에스컬레이션 여부
    threshold_used: float          # 적용된 임계값
    raw_response: ModelResponse
    scoring_method: str = "logprob"   # "logprob" | "self_consistency"


# ── 비적합 점수 계산 ────────────────────────────────────────────────────────────

def compute_entropy(logprobs: Optional[list[float]]) -> float:
    """
    토큰 수준 log probability → Shannon 엔트로피 (nats/token).
    logprobs를 지원하지 않는 경우 nan 반환.
    """
    if not logprobs:
        return float("nan")
    probs = [math.exp(lp) for lp in logprobs]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return entropy / len(probs)   # 토큰 수로 정규화


def compute_nonconformity_score(response: ModelResponse) -> float:
    """
    logprobs 기반 비적합 점수: 평균 negative log-likelihood.
    높을수록 모델이 불확실하다는 의미.
    logprobs 미지원 시 ValueError → _get_score()에서 self-consistency로 전환.
    """
    if not response.logprobs:
        raise ValueError("logprobs not available — use self-consistency scoring")
    return -float(np.mean(response.logprobs))


def _answer_diversity(texts: list[str]) -> float:
    """
    Jaccard 유사도 기반 답변 다양성 (0=완전일치, 1=완전다양).
    n개 응답의 모든 쌍에 대해 token-level Jaccard 유사도를 계산하고
    다양성 = 1 - 평균 유사도로 반환합니다.
    """
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
    avg_sim = sum(similarities) / len(similarities)
    return 1.0 - avg_sim   # 다양성 = 1 - 유사도


def compute_self_consistency_score(
    backend: str,
    system_prompt: str,
    question: str,
    n: int = 5,
) -> float:
    """
    Self-consistency 기반 비적합 점수.
    동일 질문을 n회 (temperature=0.7) 쿼리해 답변 다양성으로 불확실성을 측정합니다.
    다양성 높음 = 모델이 확신 없음 = 높은 점수 반환 (0~5 범위).

    논문 권장: n=5 (calibration 속도와 신뢰성의 균형).
    """
    texts = []
    for _ in range(n):
        resp = query_model(
            backend, system_prompt, question,
            temperature=0.7, logprobs=False,
        )
        texts.append(resp.text.strip()[:200])   # 앞 200자만 비교
    diversity = _answer_diversity(texts)
    return diversity * 5.0   # 0~5 범위로 정규화 (logprob 점수 분포와 유사하게 스케일링)


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
        Conformal Prediction 이론에 따르면 hold-out set에서 score ≤ threshold인
        비율이 1-α 이상이어야 합니다.
        """
        if not test_scores:
            return {"error": "빈 test set"}
        covered = sum(score <= self.threshold for score in test_scores)
        actual_coverage = covered / len(test_scores)
        target_coverage = 1 - self.alpha
        return {
            "target_coverage": target_coverage,
            "actual_coverage": round(actual_coverage, 4),
            "coverage_valid": actual_coverage >= target_coverage,
            "n_test": len(test_scores),
        }


# ── UQM 메인 클래스 ────────────────────────────────────────────────────────────

class UQM:
    """
    Uncertainty Quantification Module.

    workflow:
        uqm = UQM(backend="lmstudio", alpha=0.05, consistency_n=5)
        coverage_report = uqm.calibrate(calibration_questions)
        result = uqm.evaluate(question)

    scoring_method 자동 감지:
        - logprobs 지원 모델 → 평균 negative log-likelihood
        - logprobs 미지원 모델 → self-consistency (n회 반복 후 Jaccard 다양성)
    """

    SYSTEM_PROMPT = (
        "You are a clinical decision support AI. "
        "Answer the medical question. "
        "If you are not confident, say 'I am not certain' before your answer."
    )

    def __init__(self, backend: str, alpha: float = 0.05, consistency_n: int = 5):
        self.backend = backend
        self.calibrator = ConformalCalibrator(alpha=alpha)
        self._calibrated = False
        self.consistency_n = consistency_n
        self._use_self_consistency: Optional[bool] = None   # None = 미감지

    def _get_score(self, question: str) -> tuple[float, ModelResponse]:
        resp = query_model(
            self.backend,
            self.SYSTEM_PROMPT,
            question,
            temperature=0.0,
        )

        # 최초 호출 시 logprob 지원 여부 자동 감지 후 캐싱
        if self._use_self_consistency is None:
            self._use_self_consistency = resp.logprobs is None
            mode = "self-consistency" if self._use_self_consistency else "log-probability"
            print(f"[UQM] 점수 산출 방식 감지 → {mode} 모드"
                  + (f" (n={self.consistency_n})" if self._use_self_consistency else ""))

        if self._use_self_consistency:
            score = compute_self_consistency_score(
                self.backend, self.SYSTEM_PROMPT, question, self.consistency_n
            )
        else:
            score = compute_nonconformity_score(resp)

        return score, resp

    def calibrate(
        self,
        questions: list[str],
        holdout_fraction: float = 0.2,
    ) -> dict:
        """
        Calibration set으로 임계값을 학습합니다.
        holdout_fraction만큼을 hold-out해 conformal coverage를 검증합니다.

        Returns:
            coverage_report: {"target_coverage", "actual_coverage", "coverage_valid", "n_test"}
        """
        n_total = len(questions)
        n_holdout = max(1, int(n_total * holdout_fraction))
        n_cal = n_total - n_holdout

        # 재현 가능한 셔플
        rng = random.Random(42)
        indices = list(range(n_total))
        rng.shuffle(indices)
        holdout_set = set(indices[:n_holdout])

        print(f"[UQM] Calibration 시작 "
              f"({n_cal}개 학습 / {n_holdout}개 hold-out 검증, backend={self.backend})")

        cal_scores: list[float] = []
        holdout_scores: list[float] = []

        for i, q in enumerate(questions):
            score, _ = self._get_score(q)
            if i in holdout_set:
                holdout_scores.append(score)
            else:
                cal_scores.append(score)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_total}] score={score:.4f}")

        self.calibrator.fit(cal_scores)
        self._calibrated = True

        # Hold-out으로 Coverage 검증
        coverage_report = self.calibrator.check_coverage(
            holdout_scores,
            [True] * len(holdout_scores),   # 정답 레이블 없이 점수 분포만 검증
        )
        status = "✓" if coverage_report.get("coverage_valid") else "✗"
        print(f"[UQM] Coverage 검증: "
              f"{coverage_report['actual_coverage']:.3f} "
              f"(목표 {coverage_report['target_coverage']:.2f}) {status}")

        return coverage_report

    def evaluate(self, question: str) -> UncertaintyResult:
        """
        단일 질문의 불확실성을 측정하고 에스컬레이션 여부를 반환합니다.
        """
        if not self._calibrated:
            raise RuntimeError("calibrate()를 먼저 호출하세요.")
        score, resp = self._get_score(question)
        entropy = compute_entropy(resp.logprobs)
        threshold = self.calibrator.threshold
        should_escalate = score > threshold

        # Prediction set 크기: score/threshold 비율로 추정 (클수록 불확실)
        prediction_set_size = max(1, round(score / threshold)) if threshold > 0 else 1

        return UncertaintyResult(
            nonconformity_score=score,
            prediction_set_size=prediction_set_size,
            confidence_entropy=entropy,
            should_escalate=should_escalate,
            threshold_used=threshold,
            raw_response=resp,
            scoring_method="self_consistency" if self._use_self_consistency else "logprob",
        )


# ── 빠른 확인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 임시 calibration 데이터 (실제 연구에서는 MedQA calibration split 사용)
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
            report = uqm.calibrate(CAL_QUESTIONS)
            print(f"  Coverage 검증: {report}")
            for q in TEST_QUESTIONS:
                result = uqm.evaluate(q)
                status = "ESCALATE" if result.should_escalate else "AUTO"
                print(f"\n  Q: {q[:60]}...")
                print(f"  → Score={result.nonconformity_score:.3f}, "
                      f"Threshold={result.threshold_used:.3f}, "
                      f"PredSet={result.prediction_set_size}, "
                      f"Method={result.scoring_method}, {status}")
        except Exception as e:
            print(f"[SKIP] {e}")
