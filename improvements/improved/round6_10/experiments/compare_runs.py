"""
UASEF — Run Comparison Tool (audit 6.10)

여러 `--run-tag`로 분리된 실험 결과를 한 표로 비교합니다.

사용:
    # 1) 여러 ablation 실행 (각각 다른 --run-tag)
    python experiments/run_all_experiments.py --run-tag base       --backend openai
    python experiments/run_all_experiments.py --run-tag instructed --backend openai --prompt-mode instructed
    python experiments/run_all_experiments.py --run-tag confidence --backend openai --decision-rule confidence

    # 2) 비교
    python experiments/compare_runs.py base instructed confidence
    # → results/comparison_<timestamp>.md 생성 + 콘솔 출력

각 tag 디렉토리(`results/<tag>/`)의 `all_experiments_summary.json`을 읽어 다음 항목을 한 표로 출력:
  - 에이전트 Safety Recall / Over-Esc / Accuracy
  - 베이스라인 full_uasef Safety Recall / Over-Esc
  - MedAbstain 전체 recall
  - 사용된 prompt_mode / decision_rule / weighted_cp / strict
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load_summary(tag: str) -> dict | None:
    """results/<tag>/all_experiments_summary.json 로드."""
    p = ROOT / "results" / tag / "all_experiments_summary.json"
    if not p.exists():
        # tag 없으면 results/all_experiments_summary.json 시도 (root)
        p = ROOT / "results" / "all_experiments_summary.json"
        if not p.exists():
            return None
        if tag != "":
            print(f"  [WARN] '{tag}' 디렉토리 없음 → root summary 사용")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _row(tag: str, summary: dict) -> dict:
    """summary에서 비교용 단일 행 추출."""
    cfg = summary.get("meta", {}).get("config", {})
    out = {
        "tag": tag,
        "backend": cfg.get("backend"),
        "prompt_mode": cfg.get("prompt_mode"),
        "decision_rule": cfg.get("decision_rule"),
        "weighted_cp": cfg.get("weighted_cp"),
        "strict": cfg.get("strict"),
        "n_test": cfg.get("n_test"),
    }
    # agent (단일 backend면 그 백엔드, 여러개면 첫 개)
    agent = summary.get("agent", {})
    if agent:
        first_b = next(iter(agent))
        a = agent[first_b]
        out.update({
            "agent_backend": first_b,
            "agent_safety_recall": a.get("safety_recall"),
            "agent_over_esc": a.get("over_escalation_rate"),
            "agent_accuracy": a.get("accuracy"),
        })
    # baseline full_uasef
    baseline = summary.get("baseline", {})
    if baseline:
        first_b = next(iter(baseline))
        fu = baseline[first_b].get("full_uasef", {})
        if "error" not in fu:
            out.update({
                "baseline_safety_recall": fu.get("safety_recall"),
                "baseline_over_esc": fu.get("over_escalation_rate"),
            })
    # medabstain overall
    medab = summary.get("medabstain", {})
    if medab:
        first_b = next(iter(medab))
        ov = medab[first_b].get("overall", {})
        if "error" not in ov:
            out.update({
                "medabstain_recall": ov.get("recall"),
                "medabstain_auroc": ov.get("auroc"),
            })
    return out


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_comparison_table(rows: list[dict]) -> str:
    """Markdown 표 생성."""
    if not rows:
        return "_비교할 결과 없음_\n"

    # 1) 메타 표
    lines = [
        f"# UASEF Run Comparison ({datetime.now().isoformat(timespec='seconds')})\n",
        "## 1. 실행 메타\n",
        "| Tag | Backend | prompt_mode | decision_rule | weighted_cp | strict | n_test |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['tag']}` | {_fmt(r.get('backend'))} | {_fmt(r.get('prompt_mode'))} "
            f"| {_fmt(r.get('decision_rule'))} | {_fmt(r.get('weighted_cp'))} "
            f"| {_fmt(r.get('strict'))} | {_fmt(r.get('n_test'))} |"
        )

    # 2) 에이전트
    lines += [
        "\n## 2. LangGraph 에이전트 실험\n",
        "| Tag | backend | Safety Recall | Over-Esc Rate | Accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['tag']}` | {_fmt(r.get('agent_backend'))} "
            f"| {_fmt(r.get('agent_safety_recall'))} "
            f"| {_fmt(r.get('agent_over_esc'))} "
            f"| {_fmt(r.get('agent_accuracy'))} |"
        )

    # 3) 베이스라인 full_uasef
    lines += [
        "\n## 3. 베이스라인 (full_uasef)\n",
        "| Tag | Safety Recall | Over-Esc Rate |",
        "| --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['tag']}` | {_fmt(r.get('baseline_safety_recall'))} "
            f"| {_fmt(r.get('baseline_over_esc'))} |"
        )

    # 4) MedAbstain
    lines += [
        "\n## 4. MedAbstain (전체)\n",
        "| Tag | Recall | AUROC |",
        "| --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['tag']}` | {_fmt(r.get('medabstain_recall'))} "
            f"| {_fmt(r.get('medabstain_auroc'))} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UASEF Run Comparison — 여러 --run-tag 결과를 한 표로 비교 (audit 6.10)",
    )
    parser.add_argument(
        "tags", nargs="+",
        help="비교할 run-tag 목록 (예: base instructed confidence). "
             "각각 results/<tag>/all_experiments_summary.json이 존재해야 함. "
             "tag가 'root'면 results/ 루트의 결과를 사용.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="출력 파일 경로. 기본: results/comparison_<timestamp>.md",
    )
    args = parser.parse_args()

    rows = []
    for tag in args.tags:
        actual_tag = "" if tag == "root" else tag
        summary = _load_summary(actual_tag)
        if summary is None:
            print(f"  [SKIP] '{tag}': all_experiments_summary.json 없음")
            continue
        rows.append(_row(tag, summary))

    if not rows:
        print("비교할 데이터 없음. 먼저 --run-tag로 실험을 실행하세요.")
        sys.exit(1)

    md = build_comparison_table(rows)
    print(md)

    out_path = (
        Path(args.out) if args.out
        else ROOT / "results" / f"comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n✅ 비교 표 저장: {out_path}")


if __name__ == "__main__":
    main()
