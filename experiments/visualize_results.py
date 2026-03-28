"""
UASEF — 결과 시각화
experiment_results.json을 읽어 논문용 그래프를 생성합니다.

생성 파일:
    results/pareto_frontier.png  — Coverage ↔ Escalation Rate Pareto
    results/comparison_bar.png   — 백엔드별 지표 비교 바차트
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib 미설치. pip install matplotlib")


COLORS = {
    "lmstudio": "#7F77DD",    # purple
    "openai":   "#1D9E75",    # teal
    "emergency":      "#D85A30",
    "rare_disease":   "#D4537E",
    "multimorbidity": "#BA7517",
}

SCENARIO_LABELS = {
    "emergency":      "응급",
    "rare_disease":   "희귀질환",
    "multimorbidity": "다중이환",
}


def plot_comparison_bar(results: dict, out_dir: Path) -> None:
    """백엔드 × 시나리오별 Safety Recall / Over-Escalation Rate 바차트."""
    if not HAS_MPL:
        return

    scenarios = ["emergency", "rare_disease", "multimorbidity"]
    backends = list(results.keys())
    metrics_keys = ["safety_recall", "over_escalation_rate"]
    titles = ["Safety Recall (↑ 목표 ≥0.95)", "Over-Escalation Rate (↓ 목표 ≤0.15)"]
    targets = [0.95, 0.15]
    target_colors = ["#1D9E75", "#D85A30"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("UASEF: LMStudio vs OpenAI 성능 비교", fontsize=14, fontweight="bold")

    x = np.arange(len(scenarios))
    width = 0.35

    for ax_i, (metric, title, target, tcol) in enumerate(
        zip(metrics_keys, titles, targets, target_colors)
    ):
        ax = axes[ax_i]
        for b_i, backend in enumerate(backends):
            vals = []
            for sc in scenarios:
                try:
                    v = results[backend]["scenarios"][sc]["metrics"].get(metric, 0)
                    vals.append(float(v) if v != "" else 0.0)
                except (KeyError, TypeError):
                    vals.append(0.0)
            offset = (b_i - len(backends) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width * 0.9,
                label=backend.upper(),
                color=COLORS.get(backend, "#888"),
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9,
                )

        # 목표선
        ax.axhline(target, color=tcol, linestyle="--", linewidth=1.5,
                   label=f"목표 {'≥' if metric=='safety_recall' else '≤'}{target}")
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("비율")

    plt.tight_layout()
    path = out_dir / "comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 비교 바차트 저장: {path}")


def _pareto_frontier_indices(points: list[tuple[float, float]]) -> list[int]:
    """
    Non-dominated point 인덱스를 반환합니다.
    목표: escalation_rate는 낮을수록, coverage는 높을수록 좋음.
    j가 i를 지배: esc_j ≤ esc_i AND cov_j ≥ cov_i (둘 중 하나는 엄격한 부등호)
    """
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            esc_i, cov_i = points[i]
            esc_j, cov_j = points[j]
            if esc_j <= esc_i and cov_j >= cov_i and (esc_j < esc_i or cov_j > cov_i):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def plot_pareto_frontier(results: dict, out_dir: Path) -> None:
    """
    Escalation Rate ↔ Conformal Coverage Pareto frontier.

    Y축: conformal_coverage — calibration hold-out에서 검증된 실제 coverage (1-α 보장).
    X축: escalation_rate — (TP+FP)/total (낮을수록 효율적).

    conformal_coverage 데이터가 없는 백엔드는 제외합니다.
    safety_recall은 다른 지표이므로 혼용하지 않습니다.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Pareto Frontier: Conformal Coverage ↔ Escalation Rate", fontsize=13)
    ax.set_xlabel("Escalation Rate (낮을수록 효율적)")
    ax.set_ylabel("Conformal Coverage (높을수록 안전, 목표 ≥0.95)")

    ax.axvline(0.15, color="#D85A30", linestyle="--", linewidth=1, alpha=0.6,
               label="목표 Escalation ≤0.15")
    ax.axhline(0.95, color="#1D9E75", linestyle="--", linewidth=1, alpha=0.6,
               label="목표 Coverage ≥0.95")
    ax.fill_betweenx([0.95, 1.05], 0, 0.15, alpha=0.08, color="#1D9E75", label="이상적 영역")

    markers = {"lmstudio": "o", "openai": "s"}
    scenarios = ["emergency", "rare_disease", "multimorbidity"]
    all_pts: list[tuple[float, float]] = []
    skipped_backends: list[str] = []

    for backend, bdata in results.items():
        cal_coverage = bdata.get("coverage_report", {}).get("actual_coverage")
        if cal_coverage is None:
            skipped_backends.append(backend)
            continue

        esc_rates, coverages, labels = [], [], []
        for sc in scenarios:
            try:
                m = bdata["scenarios"][sc]["metrics"]
                esc_rates.append(float(m.get("escalation_rate", 0)))
                coverages.append(float(cal_coverage))
                labels.append(SCENARIO_LABELS[sc])
            except (KeyError, TypeError):
                pass

        if not esc_rates:
            continue

        ax.scatter(
            esc_rates, coverages,
            marker=markers.get(backend, "o"),
            color=COLORS.get(backend, "#888"),
            s=100, zorder=5,
            label=backend.upper(),
        )
        for x, y, lbl in zip(esc_rates, coverages, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(8, 4), fontsize=9,
                        color=COLORS.get(backend, "#888"))
        all_pts.extend(zip(esc_rates, coverages))

    # 실제 Pareto frontier: non-dominated points를 연결
    if len(all_pts) >= 2:
        frontier_idx = _pareto_frontier_indices(all_pts)
        frontier_pts = sorted([all_pts[i] for i in frontier_idx], key=lambda p: p[0])
        fx, fy = zip(*frontier_pts)
        ax.step(fx, fy, where="post", color="#444", lw=1.8, ls="-",
                label="Pareto frontier", zorder=4, alpha=0.7)

    if skipped_backends:
        print(f"[WARNING] conformal_coverage 없어 제외된 백엔드: {skipped_backends}")
        print("  coverage_report.actual_coverage 필드가 있는 실험 결과가 필요합니다.")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = out_dir / "pareto_frontier.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Pareto frontier 저장: {path}")


def plot_latency_comparison(results: dict, out_dir: Path) -> None:
    """로컬 vs 클라우드 레이턴시 비교."""
    if not HAS_MPL:
        return

    scenarios = ["emergency", "rare_disease", "multimorbidity"]
    backends = list(results.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("평균 응답 레이턴시 비교 (ms)", fontsize=12)

    x = np.arange(len(scenarios))
    width = 0.35

    for b_i, backend in enumerate(backends):
        vals = []
        for sc in scenarios:
            try:
                v = results[backend]["scenarios"][sc]["metrics"].get("avg_latency_ms", 0)
                vals.append(float(v) if v else 0.0)
            except (KeyError, TypeError):
                vals.append(0.0)
        offset = (b_i - len(backends) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=backend.upper(),
               color=COLORS.get(backend, "#888"),
               alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel("ms")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = out_dir / "latency_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 레이턴시 비교 저장: {path}")


def main():
    result_file = ROOT / "results" / "experiment_results.json"
    if not result_file.exists():
        print(f"[ERROR] 결과 파일 없음: {result_file}")
        print("먼저 experiments/run_experiment.py 를 실행하세요.")
        return

    with open(result_file, encoding="utf-8") as f:
        results = json.load(f)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    plot_comparison_bar(results, out_dir)
    plot_pareto_frontier(results, out_dir)
    plot_latency_comparison(results, out_dir)
    print("\n✅ 시각화 완료.")


if __name__ == "__main__":
    main()
