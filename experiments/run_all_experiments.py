"""
UASEF 전체 실험 통합 실행기

다음 4개 실험을 순차 실행하고 결과를 JSON + Markdown 요약으로 저장합니다:
  1. LangGraph 에이전트 실험       → results/agent_results.json
  2. 베이스라인 비교 실험           → results/baseline_comparison.json
  3. MedAbstain 분류 정확도 평가   → results/medabstain_eval.json
  4. Pareto Frontier α Sweep       → results/pareto_sweep_results.json

실험 구조:
  [Primary]  OpenAI (GPT-4o-mini) — logprob-based CP  (논문 주요 결과)
  [Ablation] 로컬 (LMStudio GGUF) — logprob-based CP
             "로컬 GGUF 모델에도 UASEF logprob CP 적용 가능함" 검증용 ablation study.

  두 결과는 Markdown 보고서에서 명시적으로 구분됩니다.
  비적합 함수가 달라 수치를 직접 비교하지 않습니다.

추가 출력:
  results/all_experiments_summary.json   — 모든 실험 핵심 지표 통합
  results/all_experiments_report.md      — Primary / Ablation 구분 Markdown 보고서

실행 예시:
    # Primary만 (OpenAI, 빠른 스모크 테스트)
    python experiments/run_all_experiments.py --backend openai

    # Primary 논문 품질
    python experiments/run_all_experiments.py --backend openai \\
        --n-cal 500 --n-test 50 --n-medabstain 100 --n-pareto-test 100

    # Primary + Ablation 전체 (논문 최종 실행)
    python experiments/run_all_experiments.py --n-cal 500 --n-test 50
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── 실험 모듈 import ──────────────────────────────────────────────────────────

from experiments.run_agent_experiment import (
    run_backend_experiment as _run_agent,
    save_results as _save_agent,
)
from experiments.run_baseline_comparison import (
    run_baseline_comparison as _run_baseline,
    save_results as _save_baseline,
    _print_comparison_table,
)
from experiments.eval_medabstain import (
    run_medabstain_eval as _run_medabstain,
    save_eval_results as _save_medabstain,
    _print_metric_table,
)
from experiments.pareto_sweep import (
    run_pareto_sweep as _run_pareto,
    save_pareto_results as _save_pareto,
    recommend_alpha,
    print_recommendations,
)
from data.loader import (
    load_calibration_questions,
    load_scenarios,
    case_to_agent_dict,
)

RESULTS_DIR = ROOT / "results"


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _elapsed(start: datetime) -> str:
    secs = (datetime.now() - start).total_seconds()
    mins, s = divmod(int(secs), 60)
    return f"{mins}m {s:02d}s" if mins else f"{s}s"


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ── 실험 1: LangGraph 에이전트 ────────────────────────────────────────────────

def run_experiment_agent(args, cal_questions, agent_scenarios) -> dict:
    _section("실험 1 / 4 — LangGraph 에이전트 실험")
    t0 = datetime.now()

    backends = [args.backend] if args.backend else ["lmstudio", "openai"]
    all_results: dict = {}

    for backend in backends:
        try:
            result = _run_agent(
                backend=backend,
                cal_questions=cal_questions,
                agent_scenarios=agent_scenarios,
                scoring_method=args.scoring_method,
                alpha=args.alpha,
                distribution_source="medqa",
            )
            if result:
                all_results[backend] = result
        except Exception:
            print(f"\n[SKIP] {backend} 에이전트 실험 실패:")
            traceback.print_exc()

    if all_results:
        _save_agent(all_results)

    print(f"\n  ✔ 소요 시간: {_elapsed(t0)}")
    return all_results


# ── 실험 2: 베이스라인 비교 ───────────────────────────────────────────────────

def run_experiment_baseline(args) -> dict:
    _section("실험 2 / 4 — 베이스라인 비교 실험")
    t0 = datetime.now()

    backends = [args.backend] if args.backend else ["lmstudio", "openai"]
    all_results: dict = {}

    for backend in backends:
        try:
            result = _run_baseline(
                backend=backend,
                n_cal=args.n_cal,
                n_test=args.n_test,
                scoring_method=args.scoring_method,
                seed=args.seed,
            )
            if result:
                all_results[backend] = result
                _print_comparison_table(result)
        except Exception:
            print(f"\n[SKIP] {backend} 베이스라인 비교 실패:")
            traceback.print_exc()

    if all_results:
        _save_baseline(all_results)

    print(f"\n  ✔ 소요 시간: {_elapsed(t0)}")
    return all_results


# ── 실험 3: MedAbstain 분류 정확도 ───────────────────────────────────────────

def run_experiment_medabstain(args) -> dict:
    _section("실험 3 / 4 — MedAbstain 분류 정확도 평가")
    t0 = datetime.now()

    backends = [args.backend] if args.backend else ["lmstudio", "openai"]
    all_results: dict = {}

    for backend in backends:
        try:
            result = _run_medabstain(
                backend=backend,
                n_cal=args.n_cal,
                n_per_variant=args.n_medabstain,
                scoring_method=args.scoring_method,
                variants=args.variants,
                use_weighted_cp=args.weighted_cp,
                seed=args.seed,
            )
            if result:
                all_results[backend] = result
                _print_metric_table(result)
        except Exception:
            print(f"\n[SKIP] {backend} MedAbstain 평가 실패:")
            traceback.print_exc()

    if all_results:
        _save_medabstain(all_results)

    print(f"\n  ✔ 소요 시간: {_elapsed(t0)}")
    return all_results


# ── 실험 4: Pareto Frontier α Sweep ──────────────────────────────────────────

def run_experiment_pareto(args) -> tuple[dict, dict]:
    _section("실험 4 / 4 — Pareto Frontier α Sweep")
    t0 = datetime.now()

    backends = [args.backend] if args.backend else ["lmstudio", "openai"]
    all_results: dict = {}

    for backend in backends:
        try:
            results = _run_pareto(
                backend=backend,
                n_calibration=args.n_cal,
                n_test=args.n_pareto_test,
                scoring_method=args.scoring_method,
            )
            if results:
                all_results[backend] = results
        except Exception:
            print(f"\n[SKIP] {backend} Pareto Sweep 실패:")
            traceback.print_exc()

    recommendations: dict = {}
    if all_results:
        _save_pareto(all_results)
        print("\n\n" + "─" * 65)
        print("  α 권고 분석 (min_coverage=0.95, max_esc_rate=0.15)")
        print("─" * 65)
        try:
            recommendations = recommend_alpha(
                all_results=all_results,
                min_coverage=0.95,
                max_escalation_rate=0.15,
                coverage_weight=2.0,
            )
            print_recommendations(recommendations)
        except Exception:
            print("[WARN] α 권고 계산 실패:")
            traceback.print_exc()

    print(f"\n  ✔ 소요 시간: {_elapsed(t0)}")
    return all_results, recommendations


# ── 통합 요약 생성 ─────────────────────────────────────────────────────────────

def _extract_agent_summary(agent_results: dict) -> dict:
    summary = {}
    for backend, data in agent_results.items():
        s = data.get("summary", {})
        summary[backend] = {
            "accuracy":             s.get("accuracy"),
            "safety_recall":        s.get("safety_recall"),
            "over_escalation_rate": s.get("over_escalation_rate"),
            "escalation_rate":      s.get("escalation_rate"),
            "avg_tool_calls":       s.get("avg_tool_calls_per_case"),
            "avg_react_iterations": s.get("avg_react_iterations"),
            "conformal_coverage":   data.get("coverage_report", {}).get("actual_coverage"),
            "scoring_method":       data.get("scoring_method"),
            "n_cases":              s.get("total"),
        }
    return summary


def _extract_baseline_summary(baseline_results: dict) -> dict:
    summary = {}
    for backend, data in baseline_results.items():
        summary[backend] = {}
        for strategy, m in data.get("metrics", {}).items():
            if "error" in m:
                summary[backend][strategy] = {"error": m["error"]}
                continue
            summary[backend][strategy] = {
                "safety_recall":        m.get("safety_recall"),
                "over_escalation_rate": m.get("over_escalation_rate"),
                "safety_recall_ok":     m.get("safety_recall_ok"),
                "tp": m.get("tp"), "fn": m.get("fn"),
                "fp": m.get("fp"), "tn": m.get("tn"),
            }
    return summary


def _extract_medabstain_summary(medabstain_results: dict) -> dict:
    summary = {}
    for backend, data in medabstain_results.items():
        overall = data.get("overall", {})
        per_variant = {}
        for v, m in data.get("per_variant", {}).items():
            if "error" in m:
                per_variant[v] = {"error": m["error"]}
                continue
            per_variant[v] = {
                "n": m.get("n"),
                "recall": m.get("recall"),
                "precision": m.get("precision"),
                "f1": m.get("f1"),
                "auroc": m.get("auroc"),
                "safety_recall_ok": m.get("safety_recall_ok"),
            }
        summary[backend] = {
            "overall": {
                "recall": overall.get("recall"),
                "precision": overall.get("precision"),
                "f1": overall.get("f1"),
                "auroc": overall.get("auroc"),
                "safety_recall_ok": overall.get("safety_recall_ok"),
            },
            "per_variant": per_variant,
            "abstention_accuracy": data.get("abstention_accuracy", {}),
        }
    return summary


def _extract_pareto_summary(pareto_results: dict, recommendations: dict) -> dict:
    summary = {}
    for backend, points in pareto_results.items():
        if not isinstance(points, list):
            continue
        # alpha별로 coverage / escalation_rate 평균
        by_alpha: dict[float, list] = {}
        for p in points:
            a = p.get("alpha")
            by_alpha.setdefault(a, []).append(p)
        alpha_stats = {}
        for a, pts in sorted(by_alpha.items()):
            coverages = [p["actual_coverage"] for p in pts if p.get("actual_coverage") is not None]
            esc_rates = [p["escalation_rate"] for p in pts if p.get("escalation_rate") is not None]
            alpha_stats[str(a)] = {
                "mean_coverage":     round(sum(coverages) / len(coverages), 4) if coverages else None,
                "mean_esc_rate":     round(sum(esc_rates) / len(esc_rates), 4) if esc_rates else None,
                "n_points":          len(pts),
            }
        summary[backend] = {
            "alpha_stats":       alpha_stats,
            "recommendations":   recommendations.get(backend, {}),
        }
    return summary


def build_summary(
    agent_results: dict,
    baseline_results: dict,
    medabstain_results: dict,
    pareto_results: dict,
    recommendations: dict,
    args: argparse.Namespace,
    total_elapsed: str,
) -> dict:
    return {
        "meta": {
            "timestamp":      datetime.now().isoformat(),
            "total_elapsed":  total_elapsed,
            "config": {
                "backend":        args.backend or "all",
                "n_cal":          args.n_cal,
                "n_test":         args.n_test,
                "n_medabstain":   args.n_medabstain,
                "n_pareto_test":  args.n_pareto_test,
                "scoring_method": args.scoring_method,
                "alpha":          args.alpha,
                "weighted_cp":    args.weighted_cp,
                "variants":       args.variants,
                "seed":           args.seed,
            },
        },
        "agent":       _extract_agent_summary(agent_results),
        "baseline":    _extract_baseline_summary(baseline_results),
        "medabstain":  _extract_medabstain_summary(medabstain_results),
        "pareto":      _extract_pareto_summary(pareto_results, recommendations),
    }


# ── Markdown 보고서 생성 ───────────────────────────────────────────────────────

def build_markdown_report(summary: dict) -> str:
    meta = summary["meta"]
    cfg  = meta["config"]
    lines = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def row(*cells) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    def sep(*widths) -> str:
        return "| " + " | ".join("-" * max(w, 3) for w in widths) + " |"

    # ── 표지 ──
    lines.append(f"# UASEF 전체 실험 보고서\n")
    lines.append(f"- **실행 시각**: {meta['timestamp']}")
    lines.append(f"- **총 소요 시간**: {meta['total_elapsed']}")
    lines.append(f"- **백엔드**: {cfg['backend']}")
    lines.append(f"- **Scoring Method**: {cfg['scoring_method']}")
    lines.append(f"- **α**: {cfg['alpha']}")
    lines.append(f"- **n_cal**: {cfg['n_cal']} | **n_test**: {cfg['n_test']} | "
                 f"**n_medabstain**: {cfg['n_medabstain']} | **n_pareto_test**: {cfg['n_pareto_test']}")
    lines.append(f"\n> **실험 구조**")
    lines.append(f"> - `[Primary]` **OpenAI** — logprob-based CP: token-level logprobs 기반 비적합 점수. **논문 주요 결과.**")
    lines.append(f"> - `[Ablation]` **LMStudio GGUF** — logprob-based CP: LM Studio OpenAI-compatible API를 통해 token-level logprobs 추출. 로컬 GGUF 모델 적용 가능성 검증.")
    lines.append(f"> - 두 백엔드 모두 동일한 logprob 비적합 함수를 사용합니다. 모델 차이에 의한 성능 비교가 가능합니다.\n")

    def _role_label(backend: str, scoring_method: str | None = None) -> str:
        """백엔드로 Primary/Ablation 레이블 결정. 두 백엔드 모두 logprob 사용."""
        return "[Primary]" if backend == "openai" else "[Ablation]"

    def _role_label(backend: str, scoring_method: str | None = None) -> str:
        """백엔드 또는 scoring_method로 Primary/Ablation 레이블 결정."""
        if scoring_method == "logprob":
            return "[Primary]"
        if scoring_method == "self_consistency":
            return "[Ablation]"
        return "[Primary]" if backend == "openai" else "[Ablation]"

    # ── 실험 1: 에이전트 ──
    h(2, "1. LangGraph 에이전트 실험")
    agent = summary.get("agent", {})
    if agent:
        lines.append(row("Backend", "Role", "Accuracy", "Safety Recall", "Over-Esc. Rate",
                         "Escalation Rate", "Avg Tool Calls", "Avg Iters", "Coverage"))
        lines.append(sep(10, 11, 10, 14, 16, 16, 16, 10, 10))
        for backend, s in agent.items():
            role = _role_label(backend, s.get("scoring_method"))
            lines.append(row(
                backend,
                role,
                _fmt(s.get("accuracy")),
                _fmt(s.get("safety_recall")),
                _fmt(s.get("over_escalation_rate")),
                _fmt(s.get("escalation_rate")),
                _fmt(s.get("avg_tool_calls")),
                _fmt(s.get("avg_react_iterations"), 1),
                _fmt(s.get("conformal_coverage")),
            ))
    else:
        lines.append("_결과 없음 (실험 실패 또는 SKIP)_")

    # ── 실험 2: 베이스라인 비교 ──
    h(2, "2. 베이스라인 비교 실험")
    baseline = summary.get("baseline", {})
    if baseline:
        lines.append(row("Backend", "Role", "전략", "Safety Recall", "Over-Esc. Rate",
                         "TP", "FN", "FP", "TN", "OK(≥0.95)"))
        lines.append(sep(10, 11, 22, 14, 16, 4, 4, 4, 4, 10))
        for backend, strategies in baseline.items():
            role = _role_label(backend)
            for strategy, m in strategies.items():
                if "error" in m:
                    lines.append(row(backend, role, strategy, "오류", "", "", "", "", "", ""))
                    continue
                ok = "✓" if m.get("safety_recall_ok") else "✗"
                lines.append(row(
                    backend, role, strategy,
                    _fmt(m.get("safety_recall")),
                    _fmt(m.get("over_escalation_rate")),
                    m.get("tp", ""), m.get("fn", ""), m.get("fp", ""), m.get("tn", ""),
                    ok,
                ))
    else:
        lines.append("_결과 없음_")

    # ── 실험 3: MedAbstain ──
    h(2, "3. MedAbstain 분류 정확도 평가")
    medabstain = summary.get("medabstain", {})
    if medabstain:
        for backend, data in medabstain.items():
            h(3, f"Backend: {backend}")

            # 전체 지표
            ov = data.get("overall", {})
            if ov:
                ok = "✓" if ov.get("safety_recall_ok") else "✗"
                lines.append(f"**전체** — Safety Recall: {_fmt(ov.get('recall'))} {ok} | "
                             f"Precision: {_fmt(ov.get('precision'))} | "
                             f"F1: {_fmt(ov.get('f1'))} | "
                             f"AUROC: {_fmt(ov.get('auroc'))}\n")

            # 변형별 테이블
            pv = data.get("per_variant", {})
            if pv:
                lines.append(row("Variant", "n", "Recall", "Precision", "F1", "AUROC", "OK(≥0.95)"))
                lines.append(sep(8, 6, 8, 10, 6, 7, 10))
                for variant, m in pv.items():
                    if "error" in m:
                        lines.append(row(variant, "오류", "", "", "", "", ""))
                        continue
                    ok = "✓" if m.get("safety_recall_ok") else "✗"
                    lines.append(row(
                        variant, m.get("n", ""),
                        _fmt(m.get("recall")),
                        _fmt(m.get("precision")),
                        _fmt(m.get("f1")),
                        _fmt(m.get("auroc")),
                        ok,
                    ))

            # 고유 Abstention Accuracy
            ab = data.get("abstention_accuracy", {})
            if ab and not ab.get("error"):
                lines.append(f"\n**LLM Abstention Accuracy** — "
                             f"Precision: {_fmt(ab.get('abstention_precision'))} | "
                             f"Recall: {_fmt(ab.get('abstention_recall'))} | "
                             f"F1: {_fmt(ab.get('abstention_f1'))}")
    else:
        lines.append("_결과 없음_")

    # ── 실험 4: Pareto Frontier ──
    h(2, "4. Pareto Frontier α Sweep")
    pareto = summary.get("pareto", {})
    if pareto:
        for backend, data in pareto.items():
            h(3, f"Backend: {backend}")

            # α별 평균 통계
            alpha_stats = data.get("alpha_stats", {})
            if alpha_stats:
                lines.append(row("α", "Mean Coverage", "Mean Esc. Rate", "# Points"))
                lines.append(sep(6, 14, 14, 8))
                for alpha_str, st in alpha_stats.items():
                    lines.append(row(
                        alpha_str,
                        _fmt(st.get("mean_coverage")),
                        _fmt(st.get("mean_esc_rate")),
                        st.get("n_points", ""),
                    ))

            # 권고
            recs = data.get("recommendations", {})
            if recs:
                lines.append(f"\n**α 권고** (min_coverage=0.95, max_esc_rate=0.15)\n")
                lines.append(row("Specialty", "α", "Coverage", "Esc. Rate", "Utility", "근거"))
                lines.append(sep(24, 5, 9, 9, 8, 40))
                for specialty, rec in recs.items():
                    if rec.get("alpha") is None:
                        lines.append(row(specialty, "—", "—", "—", "—", "실측 데이터 없음"))
                        continue
                    lines.append(row(
                        specialty,
                        _fmt(rec.get("alpha"), 2),
                        _fmt(rec.get("actual_coverage")),
                        _fmt(rec.get("escalation_rate")),
                        _fmt(rec.get("utility")),
                        rec.get("reason", ""),
                    ))
    else:
        lines.append("_결과 없음_")

    # ── 핵심 지표 요약 ──
    h(2, "핵심 지표 요약")
    lines.append("> Safety Recall ≥ 0.95 달성 여부를 중심으로 각 실험 결과를 요약합니다.\n")

    for backend in (set(agent.keys()) | set(baseline.keys()) |
                    set(medabstain.keys()) | set(pareto.keys())):
        h(3, f"Backend: {backend}")

        # 에이전트
        if backend in agent:
            a = agent[backend]
            sr = a.get("safety_recall")
            ok = "✓" if (sr is not None and sr >= 0.95) else "✗"
            lines.append(f"- **[에이전트]** Safety Recall: **{_fmt(sr)}** {ok} | "
                         f"Accuracy: {_fmt(a.get('accuracy'))} | "
                         f"Over-Esc: {_fmt(a.get('over_escalation_rate'))}")

        # 베이스라인 full_uasef 행
        if backend in baseline:
            fu = baseline[backend].get("full_uasef", {})
            if "error" not in fu:
                sr = fu.get("safety_recall")
                ok = "✓" if (sr is not None and sr >= 0.95) else "✗"
                lines.append(f"- **[베이스라인 full_uasef]** Safety Recall: **{_fmt(sr)}** {ok} | "
                             f"Over-Esc: {_fmt(fu.get('over_escalation_rate'))}")

        # MedAbstain 전체
        if backend in medabstain:
            ov = medabstain[backend].get("overall", {})
            if "error" not in ov:
                sr = ov.get("recall")
                ok = "✓" if (sr is not None and sr >= 0.95) else "✗"
                lines.append(f"- **[MedAbstain 전체]** Safety Recall: **{_fmt(sr)}** {ok} | "
                             f"AUROC: {_fmt(ov.get('auroc'))}")

        # Pareto 권고 α
        if backend in pareto:
            recs = pareto[backend].get("recommendations", {})
            if recs:
                best_alphas = ", ".join(
                    f"{spec}→α={_fmt(r.get('alpha'), 2)}"
                    for spec, r in recs.items()
                    if r.get("alpha") is not None
                )
                lines.append(f"- **[Pareto 권고 α]** {best_alphas}")

    lines.append("\n---\n_Generated by `experiments/run_all_experiments.py`_\n")
    return "\n".join(lines)


# ── 저장 ──────────────────────────────────────────────────────────────────────

def save_summary(summary: dict, report_md: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    json_path = RESULTS_DIR / "all_experiments_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 통합 요약 JSON 저장: {json_path}")

    md_path = RESULTS_DIR / "all_experiments_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"✅ Markdown 보고서 저장: {md_path}")


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UASEF 전체 실험 통합 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["lmstudio", "openai"],
        help="단일 백엔드 (기본: openai[Primary] + lmstudio[Ablation] 모두)",
    )
    parser.add_argument("--n-cal", type=int, default=30,
                        help="Calibration 질문 수 (논문 품질: 500)")
    parser.add_argument("--n-test", type=int, default=3,
                        help="에이전트·베이스라인 시나리오별 테스트 케이스 수 (논문 품질: 50)")
    parser.add_argument("--n-medabstain", type=int, default=50,
                        help="MedAbstain 변형별 케이스 수 (논문 품질: 100)")
    parser.add_argument("--n-pareto-test", type=int, default=20,
                        help="Pareto Sweep 시나리오별 테스트 케이스 수 (논문 품질: 100)")
    parser.add_argument(
        "--scoring-method", type=str, default="auto",
        choices=["logprob", "self_consistency", "auto"],
        help="비적합 점수 방식 강제 지정. 기본: auto (openai=logprob, lmstudio=logprob)",
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Conformal prediction α")
    parser.add_argument(
        "--variants", nargs="+", default=["AP", "NAP", "A", "NA"],
        choices=["AP", "NAP", "A", "NA"],
        help="MedAbstain 평가 변형",
    )
    parser.add_argument("--weighted-cp", action="store_true",
                        help="Weighted Conformal Prediction 사용")
    parser.add_argument("--include-pubmedqa", action="store_true",
                        help="에이전트 실험에 PubMedQA 케이스 추가")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip", nargs="*",
        choices=["agent", "baseline", "medabstain", "pareto"],
        default=[],
        help="건너뛸 실험 (예: --skip pareto medabstain)",
    )
    args = parser.parse_args()

    total_start = datetime.now()

    _section("UASEF 전체 실험 통합 실행")
    print(f"  Backend      : {args.backend or 'all (openai[Primary] + lmstudio[Ablation])'}")
    print(f"  Scoring      : {args.scoring_method}  "
          f"{'(auto → openai=logprob, lmstudio=logprob)' if args.scoring_method == 'auto' else ''}")
    print(f"  α            : {args.alpha}")
    print(f"  n_cal        : {args.n_cal}")
    print(f"  n_test       : {args.n_test}  (에이전트·베이스라인)")
    print(f"  n_medabstain : {args.n_medabstain}  (변형별)")
    print(f"  n_pareto_test: {args.n_pareto_test}")
    print(f"  Skip         : {args.skip or '없음'}")

    # ── 공유 데이터 로드 (에이전트 실험용) ──────────────────────────────────
    cal_questions = []
    agent_scenarios = []
    if "agent" not in args.skip:
        print("\n[Dataset] MedQA / MedAbstain 로드 중...")
        cal_questions = load_calibration_questions(n=args.n_cal, split="train", seed=args.seed)
        scenario_map = load_scenarios(
            n_per_scenario=args.n_test,
            split="test",
            seed=args.seed,
            include_pubmedqa=args.include_pubmedqa,
        )
        agent_scenarios = [
            case_to_agent_dict(case)
            for cases in scenario_map.values()
            for case in cases
        ]
        print(f"  Calibration: {len(cal_questions)}개 | 시나리오: {len(agent_scenarios)}개")

    # ── 실험 실행 ─────────────────────────────────────────────────────────────
    agent_results     = run_experiment_agent(args, cal_questions, agent_scenarios) \
                        if "agent" not in args.skip else {}
    baseline_results  = run_experiment_baseline(args) \
                        if "baseline" not in args.skip else {}
    medabstain_results = run_experiment_medabstain(args) \
                        if "medabstain" not in args.skip else {}
    pareto_results, recommendations = run_experiment_pareto(args) \
                        if "pareto" not in args.skip else ({}, {})

    # ── 통합 요약 ─────────────────────────────────────────────────────────────
    total_elapsed = _elapsed(total_start)
    summary = build_summary(
        agent_results, baseline_results, medabstain_results,
        pareto_results, recommendations, args, total_elapsed,
    )
    report_md = build_markdown_report(summary)
    save_summary(summary, report_md)

    # ── 최종 출력 ─────────────────────────────────────────────────────────────
    _section("전체 실험 완료")
    print(f"  총 소요 시간: {total_elapsed}")
    print(f"  결과 디렉토리: {RESULTS_DIR}/")
    print("  생성된 파일:")
    for fname in [
        "agent_results.json", "agent_comparison_table.csv",
        "baseline_comparison.json", "baseline_comparison.csv",
        "medabstain_eval.json", "medabstain_eval_summary.csv",
        "pareto_sweep_results.json", "pareto_frontier.png",
        "alpha_recommendations.json",
        "all_experiments_summary.json",
        "all_experiments_report.md",
    ]:
        path = RESULTS_DIR / fname
        mark = "✅" if path.exists() else "  "
        print(f"    {mark} {fname}")
    print()


if __name__ == "__main__":
    main()
