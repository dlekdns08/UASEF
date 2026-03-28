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
    margin: float                  # threshold - score (양수=임계값 아래(안전), 음수=초과(에스컬레이션))
    confidence_entropy: float      # 위치별 조건부 엔트로피 추정 (nats/token); top_logprobs 없으면 nan
    should_escalate: bool
    threshold_used: float
    raw_response: ModelResponse
    scoring_method: str = "logprob"   # 실제 사용된 방식 기록 (재현성)
    weighted_cp_used: bool = False    # Weighted CP 적용 여부 (Tibshirani et al., 2019)


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

def compute_entropy(response: ModelResponse) -> float:
    """
    위치별 조건부 엔트로피 추정 (nats/token).

    top_logprobs 있음: 각 토큰 위치의 상위 k개 확률로 per-position H 계산 후 평균.
                       어휘 전체가 아닌 top-k 분포이므로 실제 엔트로피의 하한.
    top_logprobs 없음: nan 반환. 개별 토큰 logprob으로는 Shannon 엔트로피를
                       계산할 수 없음 (각 logprob이 완전한 분포를 구성하지 않음).
    """
    if not response.top_logprobs:
        return float("nan")
    entropies = []
    for pos_logprobs in response.top_logprobs:
        if not pos_logprobs:
            continue
        # top-k logprob을 정규화하여 조건부 분포 근사
        max_lp = max(pos_logprobs)
        probs = [math.exp(lp - max_lp) for lp in pos_logprobs]
        total = sum(probs)
        probs = [p / total for p in probs]
        entropies.append(-sum(p * math.log(p + 1e-12) for p in probs))
    return sum(entropies) / len(entropies) if entropies else float("nan")


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

    def check_coverage(self, test_scores: list[float]) -> dict:
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


# ── Weighted Conformal Calibrator ──────────────────────────────────────────────

class WeightedConformalCalibrator:
    """
    Tibshirani et al. (2019) Weighted Conformal Prediction.

    분포 이동(distribution shift) 상황에서 CP coverage 보장을 복원합니다.

    핵심 아이디어:
        표준 CP: 모든 calibration 포인트에 동일 가중치 → i.i.d. 가정 필요
        Weighted CP: w_i ∝ p_test(x_i) / p_cal(x_i) 로 calibration 분포를 재구성
                     → 테스트 분포와 유사한 calibration 포인트에 높은 가중치 부여

    가중치 근사 (이 구현):
        w_i = 1 + k * jaccard(cal_text_i, test_text)
        실용적 대안: TF-IDF 코사인 유사도, 임베딩 코사인 유사도

    Coverage 보장:
        P(s_test ≤ q̂_w) ≥ 1 - α  (Theorem 1, Tibshirani et al., 2019)
        단, 가중치가 실제 밀도비 p_test/p_cal를 정확히 근사할수록 보장이 타이트해짐.
        Jaccard 근사는 보수적 (over-coverage 가능) — 논문에서 limitation으로 기술.

    참고문헌:
        Tibshirani, R. J. et al. (2019). Conformal Prediction Under Covariate Shift.
        NeurIPS 2019. arXiv:1904.06019
    """

    def __init__(self, alpha: float = 0.05, similarity_scale: float = 5.0):
        self.alpha = alpha
        self.similarity_scale = similarity_scale  # Jaccard 가중치 배율
        self.threshold: float = float("inf")      # 참고용 표준 CP 임계값
        self._cal_scores: list[float] = []
        self._cal_texts: list[str] = []

    def fit(self, scores: list[float], texts: list[str]) -> None:
        """Calibration set 저장. predict()에서 테스트 포인트별 q̂_w를 계산합니다."""
        if len(scores) != len(texts):
            raise ValueError(f"scores({len(scores)})와 texts({len(texts)}) 길이가 다릅니다.")
        if not scores:
            raise ValueError("빈 calibration set입니다.")
        self._cal_scores = scores
        self._cal_texts = texts
        # 참고용: 표준 CP 임계값 (Weighted CP와 비교 기준)
        n = len(scores)
        level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.threshold = float(np.quantile(sorted(scores), level))
        print(
            f"[WeightedCP] Fit 완료: n={n}, α={self.alpha}, "
            f"표준 q̂={self.threshold:.4f} (비교 기준)"
        )

    def predict(self, test_text: str) -> float:
        """
        test_text에 대한 개인화된 weighted quantile q̂_w를 반환합니다.
        evaluate()에서 threshold 대신 이 값을 사용합니다.

        Tibshirani et al. (2019) Algorithm 1:
            test point 자신의 weight w_{n+1} = 1 + k * Jaccard(test, test) = 1 + k
            을 포함하여 총 n+1개 질량점으로 분위수 계산.
            이 항이 없으면 coverage ≥ 1-α 하한 보장이 성립하지 않음.
        """
        weights = self._compute_weights(test_text)
        # w_{n+1}: 테스트 포인트는 자기 자신과 Jaccard = 1.0 → 최대 유사도
        w_test = 1.0 + self.similarity_scale
        return self._weighted_quantile(self._cal_scores, weights, w_test, 1 - self.alpha)

    def _compute_weights(self, test_text: str) -> list[float]:
        """
        Jaccard 단어 유사도 기반 importance weight.

        논문 품질 연구에서는 다음으로 교체 권장:
          - TF-IDF 코사인 유사도 (sklearn.feature_extraction.text.TfidfVectorizer)
          - 문장 임베딩 코사인 유사도 (sentence-transformers)
          - BM25 유사도 (rank_bm25)
        """
        test_tokens = set(test_text.lower().split())
        weights = []
        for cal_text in self._cal_texts:
            cal_tokens = set(cal_text.lower().split())
            union = cal_tokens | test_tokens
            jaccard = len(cal_tokens & test_tokens) / len(union) if union else 0.0
            weights.append(1.0 + self.similarity_scale * jaccard)
        return weights

    @staticmethod
    def _weighted_quantile(
        scores: list[float],
        weights: list[float],
        w_test: float,
        level: float,
    ) -> float:
        """
        Weighted quantile (Tibshirani et al. 2019, Algorithm 1):
            inf{q : Σ_{i: s_i ≤ q} w_i / (Σ_i w_i + w_test) ≥ level}

        w_test: 테스트 포인트의 weight (w_{n+1}).
                분모에만 포함 — 테스트 점수는 미지이므로 +∞ 질량점으로 처리.
                calibration 포인트들이 level을 충족하지 못하면 +∞ 반환(에스컬레이션).
        """
        total_w = sum(weights) + w_test   # n+1개 질량점의 총 가중치
        pairs = sorted(zip(scores, weights), key=lambda x: x[0])
        cumulative = 0.0
        for score, w in pairs:
            cumulative += w / total_w
            if cumulative >= level:
                return score
        # calibration 포인트만으로 level 미달 → +∞ 질량점 포함 시 통과
        # 즉, 임계값을 +∞로 설정 = 반드시 에스컬레이션
        return float("inf")


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
        use_weighted_cp: bool = False,
    ):
        self.backend = backend
        self.calibrator = ConformalCalibrator(alpha=alpha)
        self._calibrated = False
        self.consistency_n = consistency_n
        self._scoring_method = ScoringMethod(scoring_method)
        self._calibration_meta: Optional[CalibrationMetadata] = None
        self.use_weighted_cp = use_weighted_cp
        self._weighted_calibrator: Optional[WeightedConformalCalibrator] = None
        self._cal_texts: list[str] = []

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
        cal_texts: list[str] = []
        for i, q in enumerate(questions):
            score, _ = self._get_score(q)
            if i in holdout_set:
                holdout_scores.append(score)
            else:
                cal_scores.append(score)
                cal_texts.append(q)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_total}] score={score:.4f}")

        self._cal_texts = cal_texts
        self.calibrator.fit(cal_scores)
        self._calibrated = True

        # Weighted CP 보정 (use_weighted_cp=True 또는 나중에 evaluate()에서 분포 이동 감지 시 사용)
        self._weighted_calibrator = WeightedConformalCalibrator(alpha=self.calibrator.alpha)
        self._weighted_calibrator.fit(cal_scores, cal_texts)
        if self.use_weighted_cp:
            print("[UQM] WeightedCP 활성화 — evaluate()에서 weighted q̂_w 사용")

        coverage_report = self.calibrator.check_coverage(holdout_scores)
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

        # 분포 이동 감지 여부
        is_shift = bool(
            distribution_source
            and self._calibration_meta
            and self._calibration_meta.distribution_source not in ("unknown", distribution_source)
        )

        # Weighted CP 임계값 결정:
        #   - use_weighted_cp=True: 항상 weighted q̂_w 사용
        #   - 분포 이동 감지: 자동으로 weighted q̂_w로 전환
        #   - 그 외: 표준 q̂ 사용
        use_weighted = self._weighted_calibrator is not None and (
            self.use_weighted_cp or is_shift
        )
        if use_weighted:
            threshold = self._weighted_calibrator.predict(question)
            if is_shift and not self.use_weighted_cp:
                print(
                    f"[WeightedCP] 분포 이동 감지 → weighted q̂_w={threshold:.4f} 자동 사용 "
                    f"(표준 q̂={self.calibrator.threshold:.4f})"
                )
        else:
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
            weighted_cp_used=use_weighted,
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
