"""
Round 10 paper sync — R10 실제 결과로 paper/UASEF_Round10.md (+KO) 갱신.

paper §4.5, §6.1, §6.3, §6.4, §6.6 의 placeholder / pending 표를 실제 R10 수치로
교체. ROUND10_FINAL_REPORT.md §13 의 "즉시 paper 업데이트 항목" 5개를 자동 적용.

산출:
  paper/UASEF_Round10.md       (in-place, R10 실제 수치)
  paper/UASEF_Round10_KO.md    (in-place, 동일)

ML4H 2026 submission 준비 — paper 의 §5 (Results) 가 진짜 수치로 채워짐.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _load_jsons(results_dir: Path) -> dict:
    """모든 R10.x JSON 로드."""
    out = {}
    for name in ["r10_1_alpha_005_empirical", "r10_2_table4_multiseed",
                  "r10_3_mitigation", "r10_4_method_agnostic",
                  "r10_6_cost_sweep_4d", "r10_7_feature_expansion",
                  "r10_rf_calibration"]:
        p = results_dir / f"{name}.json"
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except Exception as e:
                print(f"  [warn] failed to load {name}: {e}")
    return out


def _build_r10_4_table(d: dict) -> str:
    """R10.4 method-agnostic 표 (5 classifier × per-stratum)."""
    lines = ["| Classifier | CRITICAL miss/n_pos | exact 95% upper | α=0.05 satisfies? | HIGH miss/n_pos | exact upper | α=0.10? |",
             "| --- | --- | --- | :---: | --- | --- | :---: |"]
    for clf in d.get("classifiers", []):
        agg = d.get("per_classifier", {}).get(clf, {})
        if not agg:
            continue
        crit = agg.get("CRITICAL", {})
        high = agg.get("HIGH", {})
        if crit is None or high is None:
            continue
        crit_ok = "✓" if crit.get("satisfies_alpha") else "✗"
        high_ok = "✓" if high.get("satisfies_alpha") else "✗"
        lines.append(
            f"| **{clf}** | "
            f"{crit.get('pooled_misses', '?')}/{crit.get('pooled_n_pos', '?')} | "
            f"{crit.get('pooled_exact_upper95', 0):.4f} | {crit_ok} | "
            f"{high.get('pooled_misses', '?')}/{high.get('pooled_n_pos', '?')} | "
            f"{high.get('pooled_exact_upper95', 0):.4f} | {high_ok} |"
        )
    return "\n".join(lines)


def _build_r10_2_table(d: dict) -> str:
    """R10.2 multi-seed Table 4-MIMIC."""
    lines = ["| Method | CRITICAL Recall (mean ± std) | Total Cost (mean ± std) |",
             "| --- | --- | --- |"]
    backend = "lmstudio"
    agg = d.get("per_backend", {}).get(backend, {})
    for name, m in agg.items():
        cr = (f"{m['critical_recall_mean']:.4f} ± {m['critical_recall_std']:.4f}"
              if m.get('critical_recall_mean') is not None else "—")
        tc = (f"{m['total_cost_mean']:.1f} ± {m['total_cost_std']:.1f}"
              if m.get('total_cost_mean') is not None else "—")
        lines.append(f"| {name} | {cr} | {tc} |")
    return "\n".join(lines)


def _build_r10_3_table(d: dict) -> str:
    """R10.3 mitigation strategies."""
    lines = ["| Strategy | CRITICAL viol × | HIGH | MODERATE | LOW | Verdict |",
             "| --- | --- | --- | --- | --- | --- |"]
    verdict = {
        "online_recal": "CRITICAL only",
        "kmm": "**best overall**",
        "group_conditional": "**catastrophic fail**",
    }
    for strat, agg in d.get("per_strategy", {}).items():
        if isinstance(agg, dict) and agg.get("error"):
            lines.append(f"| {strat} | error | — | — | — | {agg['error'][:50]} |"); continue
        cells = []
        for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            a = (agg or {}).get(s) if isinstance(agg, dict) else None
            if a is None or a.get("violation_x_mean") is None:
                cells.append("—")
            else:
                cells.append(f"{a['violation_x_mean']:.2f}×")
        lines.append(f"| **{strat}** | {' | '.join(cells)} | {verdict.get(strat, '—')} |")
    return "\n".join(lines)


def _build_r10_6_summary(d: dict) -> str:
    """R10.6 4-D cost sweep — 81/0 v2 win."""
    return (
        f"**v2 wins: {d.get('v2_wins', 0)} / "
        f"v1-cost-aware wins: {d.get('v1_wins', 0)} / "
        f"ties: {d.get('ties', 0)} out of {d.get('n_combinations', 0)} cost-matrix combinations.**\n\n"
        f"Cost-matrix dependence quantified: R10.2 의 default cost matrix 가 "
        f"corner case 였으며, 81 개 다른 regime 에서 v2 가 일관되게 win."
    )


def _build_r10_7_table(d: dict) -> str:
    """R10.7 expanded-feature validation — honest negative."""
    lines = ["| stratum | basic miss | expanded miss | improvement |",
             "| --- | --- | --- | --- |"]
    basic = d.get("summary", {}).get("basic", {})
    expanded = d.get("summary", {}).get("expanded", {})
    for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        b = basic.get(s)
        e = expanded.get(s)
        if b is None or e is None:
            lines.append(f"| {s} | — | — | — |"); continue
        diff = b["miss_rate_mean"] - e["miss_rate_mean"]
        verdict = "**악화**" if diff < -0.001 else ("개선" if diff > 0.001 else "변화 없음")
        lines.append(f"| {s} | {b['miss_rate_mean']:.4f} ± {b['miss_rate_std']:.4f} | "
                     f"{e['miss_rate_mean']:.4f} ± {e['miss_rate_std']:.4f} | "
                     f"{diff:+.4f} ({verdict}) |")
    return "\n".join(lines)


def _build_r10_1_table(d: dict) -> str:
    """R10.1 powered α=0.05 (single classifier — LLM)."""
    lines = ["| stratum | α | pooled miss / n_pos | exact 95% upper | satisfies α? |",
             "| --- | --- | --- | --- | :---: |"]
    agg = d.get("per_classifier", {}).get("gpt_oss_120b", {})
    for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        cell = agg.get(s, {})
        if not cell:
            lines.append(f"| {s} | — | — | — | — |"); continue
        ok = "✓" if cell.get("satisfies_alpha") else "✗"
        lines.append(
            f"| {s} | {cell.get('alpha', '?')} | "
            f"{cell.get('pooled_misses', '?')}/{cell.get('pooled_n_pos', '?')} | "
            f"{cell.get('pooled_exact_upper95', 0):.4f} | {ok} |"
        )
    return "\n".join(lines)


def _build_calibration_table(d: dict) -> str:
    """RF calibration analysis (only if exists)."""
    if not d:
        return ""
    lines = ["| Classifier | ECE (10-bin) | Brier | Sharpness |",
             "| --- | --- | --- | --- |"]
    for r in d.get("per_classifier", []):
        if "error" in r:
            lines.append(f"| {r['classifier']} | error | — | — |"); continue
        lines.append(
            f"| **{r['classifier']}** | {r['ece_10bin']:.4f} | "
            f"{r['brier']:.4f} | {r['sharpness']:.4f} |"
        )
    return "\n".join(lines)


# ── Section replacers ───────────────────────────────────────────────────────

def _inject(paper_text: str, marker_start: str, marker_end: str,
            new_content: str) -> str:
    """marker_start 와 marker_end 사이를 new_content 로 교체."""
    pattern = re.compile(re.escape(marker_start) + r".*?" + re.escape(marker_end),
                          re.DOTALL)
    repl = marker_start + "\n" + new_content + "\n" + marker_end
    if pattern.search(paper_text):
        return pattern.sub(repl, paper_text)
    # marker 없으면 끝에 append
    return paper_text + "\n\n" + repl


def sync_paper(paper_path: Path, data: dict) -> None:
    """paper 의 R10 result 섹션을 실제 수치로 교체."""
    if not paper_path.exists():
        sys.exit(f"paper missing: {paper_path}")
    txt = paper_path.read_text()
    original_len = len(txt)

    # §5.1 R10.1 result
    if "r10_1_alpha_005_empirical" in data:
        txt = _inject(
            txt,
            "<!-- R10.1_RESULT_START -->",
            "<!-- R10.1_RESULT_END -->",
            f"**R10.1 result — 5-seed pooled, gpt-oss-120b on n_cal=n_test≈3000 CRITICAL:**\n\n"
            + _build_r10_1_table(data["r10_1_alpha_005_empirical"])
            + "\n\nAll four strata fail to satisfy α at exact 95% upper. LLM alone "
              "cannot achieve formal coverage proof at this scale; see R10.4 for the "
              "method-agnostic head-to-head where RandomForest is the unique success."
        )

    # §5.2 R10.2 result
    if "r10_2_table4_multiseed" in data:
        txt = _inject(
            txt,
            "<!-- R10.2_RESULT_START -->",
            "<!-- R10.2_RESULT_END -->",
            "**R10.2 result — Multi-seed Table 4-MIMIC (8 methods, 5 seeds, lmstudio):**\n\n"
            + _build_r10_2_table(data["r10_2_table4_multiseed"])
            + "\n\nv2 retains its 6.5× recall advantage over single-α published baselines."
        )

    # §5.3 R10.3 result
    if "r10_3_mitigation" in data:
        txt = _inject(
            txt,
            "<!-- R10.3_RESULT_START -->",
            "<!-- R10.3_RESULT_END -->",
            "**R10.3 result — Distribution shift mitigation (3 strategies, 5 seeds):**\n\n"
            + _build_r10_3_table(data["r10_3_mitigation"])
            + "\n\nKMM is the recommended production strategy; group-conditional CRC is not."
        )

    # §5.4 R10.4 HEADLINE
    if "r10_4_method_agnostic" in data:
        txt = _inject(
            txt,
            "<!-- R10.4_RESULT_START -->",
            "<!-- R10.4_RESULT_END -->",
            "**R10.4 result — HEADLINE: Method-agnostic CRC head-to-head (5 classifiers, 5 seeds):**\n\n"
            + _build_r10_4_table(data["r10_4_method_agnostic"])
            + "\n\n**RandomForest is the unique classifier that satisfies α at CRITICAL "
              "and HIGH strata** — 0/1293 and 0/525 misses respectively across 5 seeds. "
              "LLM (gpt-oss-120b) is the worst (CRITICAL 13.4% miss). MODERATE/LOW all fail "
              "across classifiers — diagnosed as decision-time feature limitation, not "
              "framework defect (see §7 L27)."
        )

    # §5.5 R10.6 result
    if "r10_6_cost_sweep_4d" in data:
        txt = _inject(
            txt,
            "<!-- R10.6_RESULT_START -->",
            "<!-- R10.6_RESULT_END -->",
            "**R10.6 result — 4-D cost matrix sweep:**\n\n"
            + _build_r10_6_summary(data["r10_6_cost_sweep_4d"])
        )

    # §5.6 R10.7 result
    if "r10_7_feature_expansion" in data:
        txt = _inject(
            txt,
            "<!-- R10.7_RESULT_START -->",
            "<!-- R10.7_RESULT_END -->",
            "**R10.7 result — Expanded-feature validation (HONEST NEGATIVE):**\n\n"
            + _build_r10_7_table(data["r10_7_feature_expansion"])
            + "\n\nFeature engineering alone does not solve MOD/LOW; potential leakage "
              "in Charlson (current-admission ICD) and specialty_baseline_rate "
              "(cohort-level statistic) — see §7 L25-L26 and Round 11 plan."
        )

    # §6.5 RF calibration (new section)
    if "r10_rf_calibration" in data:
        txt = _inject(
            txt,
            "<!-- RF_CALIBRATION_START -->",
            "<!-- RF_CALIBRATION_END -->",
            "**Calibration analysis (RF vs LLM):**\n\n"
            + _build_calibration_table(data["r10_rf_calibration"])
            + "\n\nRandomForest의 낮은 ECE + Brier score 가 CRC 임계값의 well-fit 을 설명. "
              "Bagging 의 자연적 score smoothing 이 sharp LLM/LogReg/XGBoost boundary 보다 "
              "CRC 의 quantile 산출에 유리."
        )

    if len(txt) == original_len:
        print("  [warn] paper unchanged — markers 없음. paper 에 <!-- R10.X_RESULT_START --> 추가 필요.")
    paper_path.write_text(txt)
    delta = len(txt) - original_len
    print(f"  ✅ synced: {paper_path.name} (Δ {delta:+d} chars)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT / "results" / "round10")
    ap.add_argument("--papers", nargs="+", type=Path,
                    default=[ROOT / "paper" / "UASEF_Round10.md",
                             ROOT / "paper" / "UASEF_Round10_KO.md"])
    args = ap.parse_args()

    data = _load_jsons(args.results_dir)
    print(f"loaded {len(data)} result files: {list(data.keys())}\n")

    for p in args.papers:
        if not p.exists():
            print(f"  [skip] {p} (missing)")
            continue
        print(f"sync {p.name} ...")
        sync_paper(p, data)


if __name__ == "__main__":
    main()
