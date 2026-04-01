"""
UASEF — Pareto Frontier Alpha Sweep

Coverage guarantee ↔ Escalation rate의 실제 Pareto frontier를 계산합니다.
α 값을 스윕하며 각 α에서 UQM을 재보정하고 실측 (coverage, escalation_rate)을 측정합니다.

실행:
    python experiments/pareto_sweep.py
    python experiments/pareto_sweep.py --backend openai --n-cal 500

출력:
    results/pareto_sweep_results.json
    results/pareto_frontier.png  (실측 데이터 기반)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pareto Frontier의 의미:
  X축: Escalation Rate (낮을수록 효율적 — 불필요한 에스컬레이션 없음)
  Y축: Conformal Coverage (높을수록 안전 — 1-α 보장)
  목표 영역: X < 0.15 and Y > 0.95

이론값: α = 0.05 → coverage = 0.95 (이론적 보장)
실측값: hold-out set에서 실제로 score ≤ q̂인 비율 (항상 ≥ 1-α)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from models.uqm import UQM
from models.rtc_ede import RTC, EDE
from data.loader import load_calibration_questions, load_scenarios
from experiments.config_utils import load_calibration_config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── 실험 파라미터 ─────────────────────────────────────────────────────────────

# 스윕할 α 값: 낮은 α = 높은 coverage 요구 = 더 많은 에스컬레이션
ALPHAS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]

# 전문과목별 Pareto point 비교
SPECIALTIES = [
    ("emergency_medicine", "emergency"),
    ("internal_medicine",  "multimorbidity"),
    ("general_practice",   "routine"),
]

COLORS = {
    "emergency_medicine": "#D85A30",
    "internal_medicine":  "#7F77DD",
    "general_practice":   "#1D9E75",
}


# ── 단일 (α, specialty) Pareto 포인트 측정 ────────────────────────────────────

def measure_pareto_point(
    backend: str,
    alpha: float,
    specialty: str,
    scenario_type: str,
    cal_questions: list[str],
    test_cases: list[dict],
    distribution_source: str = "medqa",
    scoring_method: str = "logprob",
) -> dict:
    """
    단일 (α, specialty) 조합에서 conformal coverage와 escalation rate를 측정합니다.

    CP Trigger만 사용 (UNCERTAINTY_EXCEEDED만):
    keyword/evidence triggers를 제외하여 순수 conformal prediction 효과를 측정.
    이 값이 Pareto frontier의 x, y 좌표가 됩니다.
    """
    uqm = UQM(backend=backend, alpha=alpha, scoring_method=scoring_method)
    coverage_report = uqm.calibrate(
        cal_questions,
        holdout_fraction=0.2,
        distribution_source=distribution_source,
    )
    actual_coverage = coverage_report.get("actual_coverage", 1 - alpha)

    rtc_multipliers, _ = load_calibration_config()
    rtc = RTC(base_threshold=uqm.calibrator.threshold, multipliers=rtc_multipliers)
    rtc_config = rtc.get_threshold(specialty, scenario_type)

    escalations = []
    for case in test_cases:
        unc = uqm.evaluate(
            case.get("question", case),
            distribution_source=distribution_source,
        )
        # 순수 CP Trigger만 사용 (Pareto 분석: keyword/evidence trigger 제외)
        escalated = unc.nonconformity_score > rtc_config.adjusted_threshold
        escalations.append(escalated)

    esc_rate = sum(escalations) / len(escalations) if escalations else 0.0

    return {
        "alpha": alpha,
        "specialty": specialty,
        "scenario_type": scenario_type,
        "target_coverage": round(1 - alpha, 4),
        "actual_coverage": actual_coverage,
        "adjusted_threshold": round(rtc_config.adjusted_threshold, 4),
        "base_threshold": round(uqm.calibrator.threshold, 4),
        "escalation_rate": round(esc_rate, 4),
        "n_cal": len(cal_questions),
        "n_test": len(test_cases),
        "scoring_method": scoring_method,
    }


# ── 전체 스윕 실행 ─────────────────────────────────────────────────────────────

def run_pareto_sweep(
    backend: str,
    n_calibration: int = 30,
    n_test: int = 20,
    seed: int = 42,
    scoring_method: str = "logprob",
) -> list[dict]:
    """
    모든 (α, specialty) 조합에 대해 Pareto point를 측정합니다.
    각 α마다 UQM을 새로 보정합니다 (독립적 실험 조건).
    """
    print(f"\n{'='*60}")
    print(f"  Pareto Sweep — backend={backend}, scoring={scoring_method}")
    print(f"  α values: {ALPHAS}")
    print(f"  Specialties: {[s[0] for s in SPECIALTIES]}")
    print(f"{'='*60}")

    # 데이터 로드 (스윕 전체에서 공유)
    print(f"\n[Data] Calibration {n_calibration}개 + Test {n_test}개/시나리오 로드...")
    cal_questions = load_calibration_questions(n=n_calibration, split="train", seed=seed)
    scenario_map = load_scenarios(n_per_scenario=n_test, split="test", seed=seed)

    results = []
    total = len(ALPHAS) * len(SPECIALTIES)
    done = 0

    for alpha in ALPHAS:
        for specialty, scenario_type in SPECIALTIES:
            done += 1
            test_cases_raw = scenario_map.get(scenario_type, [])
            test_cases = [{"question": c.question} for c in test_cases_raw]

            if not test_cases:
                print(f"  [{done}/{total}] α={alpha:.2f}, {specialty}: 테스트 케이스 없음 — 건너뜀")
                continue

            print(f"\n  [{done}/{total}] α={alpha:.2f}, specialty={specialty} "
                  f"(n_test={len(test_cases)})...")
            try:
                point = measure_pareto_point(
                    backend=backend,
                    alpha=alpha,
                    specialty=specialty,
                    scenario_type=scenario_type,
                    cal_questions=cal_questions,
                    test_cases=test_cases,
                    scoring_method=scoring_method,
                )
                results.append(point)
                print(
                    f"    coverage={point['actual_coverage']:.3f} "
                    f"(목표={point['target_coverage']:.2f}) | "
                    f"escalation_rate={point['escalation_rate']:.3f}"
                )
            except Exception as e:
                print(f"    [ERROR] {e}")

    return results


# ── Pareto Frontier 시각화 ─────────────────────────────────────────────────────

def plot_pareto_frontier(
    all_results: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """실측 데이터 기반 Pareto frontier 산점도 + 연결선."""
    if not HAS_MPL:
        print("[WARNING] matplotlib 미설치. pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        "Pareto Frontier: Conformal Coverage ↔ Escalation Rate\n"
        "(실측 데이터, α sweep 기반)",
        fontsize=12,
    )
    ax.set_xlabel("Escalation Rate (낮을수록 효율적)")
    ax.set_ylabel("Conformal Coverage (높을수록 안전, 1-α 보장)")

    # 목표 영역
    ax.axvline(0.15, color="#D85A30", linestyle="--", linewidth=1.2, alpha=0.7,
               label="목표 Escalation ≤0.15")
    ax.axhline(0.95, color="#1D9E75", linestyle="--", linewidth=1.2, alpha=0.7,
               label="목표 Coverage ≥0.95")
    ax.fill_betweenx([0.95, 1.05], 0, 0.15, alpha=0.07, color="#1D9E75", label="이상적 영역")

    markers = {"lmstudio": "o", "openai": "s"}
    linestyles = {"lmstudio": "-", "openai": "--"}

    for backend, results in all_results.items():
        for spec, st in SPECIALTIES:
            pts = [r for r in results if r["specialty"] == spec]
            if not pts:
                continue
            # α 내림차순 정렬 (coverage 오름차순 = escalation 오름차순)
            pts.sort(key=lambda r: r["alpha"], reverse=True)
            xs = [p["escalation_rate"] for p in pts]
            ys = [p["actual_coverage"] for p in pts]

            color = COLORS.get(spec, "#888")
            ax.plot(
                xs, ys,
                marker=markers.get(backend, "o"),
                linestyle=linestyles.get(backend, "-"),
                color=color,
                linewidth=1.5,
                markersize=7,
                label=f"{backend.upper()} / {spec.replace('_', ' ')}",
                alpha=0.85,
            )
            # α 레이블 (처음과 끝 포인트만)
            for p in [pts[0], pts[-1]]:
                ax.annotate(
                    f"α={p['alpha']:.2f}",
                    (p["escalation_rate"], p["actual_coverage"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7, color=color, alpha=0.8,
                )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)

    path = out_dir / "pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Pareto frontier 저장: {path}")


# ── 결과 저장 ─────────────────────────────────────────────────────────────────

def save_pareto_results(all_results: dict[str, list[dict]]) -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "alphas": ALPHAS,
        "specialties": [s[0] for s in SPECIALTIES],
        "backends": all_results,
    }
    path = out_dir / "pareto_sweep_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Pareto sweep 결과 저장: {path}")

    plot_pareto_frontier(all_results, out_dir)

    # 요약 출력
    print("\n" + "="*65)
    print("  Pareto Sweep 요약")
    print("="*65)
    print(f"{'Backend':<12} {'α':<6} {'Specialty':<22} {'Coverage':<10} {'Esc.Rate'}")
    print("-"*65)
    for backend, results in all_results.items():
        for r in sorted(results, key=lambda x: (x["specialty"], x["alpha"])):
            ok = "✓" if r["actual_coverage"] >= r["target_coverage"] else "✗"
            print(
                f"{backend:<12} {r['alpha']:<6.2f} "
                f"{r['specialty']:<22} "
                f"{r['actual_coverage']:.3f} {ok}   "
                f"{r['escalation_rate']:.3f}"
            )


# ── α 권고 분석 ───────────────────────────────────────────────────────────────

def recommend_alpha(
    results_path: Path = None,
    all_results: dict = None,
    min_coverage: float = 0.95,
    max_escalation_rate: float = 0.15,
    coverage_weight: float = 2.0,
) -> dict:
    """
    Pareto sweep 결과에서 specialty별 최적 α를 권고합니다.

    선택 기준 (우선순위 순):
      1. 안전 제약 충족: actual_coverage ≥ min_coverage (CP 보장)
      2. 효율 제약 충족: escalation_rate ≤ max_escalation_rate
      3. 유틸리티 최대화: U = coverage - coverage_weight × escalation_rate

    두 제약을 모두 충족하는 α가 없을 경우:
      - coverage 제약만 충족하는 것 중 escalation_rate 최소인 α 선택
      - 없으면 utility 최대인 α 선택

    Args:
        results_path:         pareto_sweep_results.json 경로 (파일에서 로드 시)
        all_results:          run_pareto_sweep()의 직접 반환값 (메모리에서 사용 시)
        min_coverage:         최소 요구 coverage (기본 0.95 = α=0.05 이론 보장)
        max_escalation_rate:  최대 허용 escalation_rate (기본 0.15)
        coverage_weight:      U 함수에서 coverage 가중치 (기본 2.0)

    Returns:
        dict: {backend: {specialty: {alpha, actual_coverage, escalation_rate, reason}}}
    """
    # 데이터 로드
    if all_results is None:
        path = results_path or (ROOT / "results" / "pareto_sweep_results.json")
        if not path.exists():
            raise FileNotFoundError(
                f"Pareto sweep 결과 없음: {path}\n"
                f"  먼저 실행하세요: python experiments/pareto_sweep.py"
            )
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        all_results = payload.get("backends", {})

    recommendations: dict[str, dict[str, dict]] = {}

    for backend, results in all_results.items():
        recommendations[backend] = {}

        # specialty별로 그룹화
        by_specialty: dict[str, list[dict]] = {}
        for pt in results:
            spec = pt["specialty"]
            by_specialty.setdefault(spec, []).append(pt)

        for specialty, points in by_specialty.items():
            # 유효한 포인트만 (실측 데이터 있는 것)
            valid = [p for p in points if p.get("actual_coverage") is not None
                     and p.get("escalation_rate") is not None]
            if not valid:
                recommendations[backend][specialty] = {
                    "alpha": None,
                    "reason": "실측 데이터 없음 — pareto_sweep.py 실행 필요",
                }
                continue

            def utility(p: dict) -> float:
                return (p["actual_coverage"]
                        - coverage_weight * p["escalation_rate"])

            # 케이스 1: 두 제약 모두 충족
            both_ok = [
                p for p in valid
                if p["actual_coverage"] >= min_coverage
                and p["escalation_rate"] <= max_escalation_rate
            ]
            if both_ok:
                best = max(both_ok, key=utility)
                reason = (
                    f"coverage={best['actual_coverage']:.3f}(≥{min_coverage}) & "
                    f"esc_rate={best['escalation_rate']:.3f}(≤{max_escalation_rate}) "
                    f"충족 — utility 최대"
                )
            else:
                # 케이스 2: coverage 제약만 충족
                cov_ok = [p for p in valid if p["actual_coverage"] >= min_coverage]
                if cov_ok:
                    best = min(cov_ok, key=lambda p: p["escalation_rate"])
                    reason = (
                        f"coverage 제약(≥{min_coverage}) 충족 중 "
                        f"escalation_rate 최소 선택 "
                        f"(esc_rate={best['escalation_rate']:.3f} > {max_escalation_rate} 초과)"
                    )
                else:
                    # 케이스 3: 제약 없이 utility 최대
                    best = max(valid, key=utility)
                    reason = (
                        f"coverage 제약 미충족 — utility 최대 선택 "
                        f"(coverage={best['actual_coverage']:.3f} < {min_coverage})"
                    )

            recommendations[backend][specialty] = {
                "alpha":           best["alpha"],
                "actual_coverage": best["actual_coverage"],
                "escalation_rate": best["escalation_rate"],
                "adjusted_threshold": best.get("adjusted_threshold"),
                "utility":         round(utility(best), 4),
                "reason":          reason,
            }

    return recommendations


def print_recommendations(recommendations: dict) -> None:
    """권고 α 테이블 출력."""
    print("\n" + "="*75)
    print("  specialty별 최적 α 권고")
    print("="*75)
    print(f"  {'Backend':<12} {'Specialty':<24} {'α':>5} {'Coverage':>9} "
          f"{'Esc.Rate':>9} {'Utility':>8}")
    print("  " + "-"*71)
    for backend, by_spec in recommendations.items():
        for specialty, rec in by_spec.items():
            if rec.get("alpha") is None:
                print(f"  {backend:<12} {specialty:<24} — 실측 데이터 없음")
                continue
            print(
                f"  {backend:<12} {specialty:<24} "
                f"{rec['alpha']:>5.2f} "
                f"{rec['actual_coverage']:>9.4f} "
                f"{rec['escalation_rate']:>9.4f} "
                f"{rec['utility']:>8.4f}"
            )
            print(f"  {'':>37}↳ {rec['reason']}")
    print("="*75)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UASEF Pareto Frontier Alpha Sweep")
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["lmstudio", "openai"],
        help="단일 백엔드만 실행 (기본: 양쪽 모두)",
    )
    parser.add_argument("--n-cal", type=int, default=30, help="Calibration 질문 수 (권장: 500)")
    parser.add_argument("--n-test", type=int, default=20, help="시나리오별 테스트 케이스 수 (권장: 100)")
    parser.add_argument(
        "--scoring-method", type=str, default="logprob",
        choices=["logprob", "self_consistency"],
        help="비적합 점수 방식 (logprob=primary, self_consistency=ablation)",
    )
    args = parser.parse_args()

    backends = [args.backend] if args.backend else ["lmstudio", "openai"]

    all_results = {}
    for backend in backends:
        try:
            results = run_pareto_sweep(
                backend=backend,
                n_calibration=args.n_cal,
                n_test=args.n_test,
                scoring_method=args.scoring_method,
            )
            if results:
                all_results[backend] = results
        except Exception as e:
            print(f"\n[SKIP] {backend}: {e}")

    if all_results:
        save_pareto_results(all_results)

        # α 권고 분석 (sweep 직후 자동 실행)
        print("\n\n" + "─"*65)
        print("  α 권고 분석 (min_coverage=0.95, max_esc_rate=0.15)")
        print("─"*65)
        try:
            recs = recommend_alpha(
                all_results=all_results,
                min_coverage=0.95,
                max_escalation_rate=0.15,
                coverage_weight=2.0,
            )
            print_recommendations(recs)
            # 권고 결과를 JSON에도 저장
            rec_path = ROOT / "results" / "alpha_recommendations.json"
            rec_path.write_text(
                __import__("json").dumps(recs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n✅ α 권고 저장: {rec_path}")
        except Exception as e:
            print(f"[WARNING] α 권고 계산 실패: {e}")
