"""
Aggregate multi-seed runs of run_full_evaluation.sh.

Reads each per-seed run directory's table1_coverage.json and
table4_baseline.json, computes mean / std / 95% bootstrap CI per metric,
and emits aggregate_seeds.{json,md}.

Usage (called by run_multiseed_evaluation.sh):
    python aggregate_multiseed.py \
        --runs results/run_X1/ results/run_X2/ ... \
        --output results/run_<ts>_aggregate/ \
        --backends openai lmstudio
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from datetime import datetime


def bootstrap_ci_95(values: list[float], n_resamples: int = 2000, seed: int = 42) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI."""
    if not values or len(values) < 2:
        return None
    import random
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return (lo, hi)


def aggregate_table1(run_dirs: list[Path], backend: str) -> dict:
    """Aggregate per-seed Table 1 (per-stratum miss rates) for one backend."""
    by_method: dict[str, dict[str, list[float]]] = {}
    for d in run_dirs:
        path = d / backend / "table1_coverage.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for m in data.get("methods", []):
            name = m["name"]
            by_method.setdefault(name, {"CRITICAL": [], "HIGH": [], "MODERATE": [], "LOW": []})
            for stratum in by_method[name]:
                rate = m.get("per_stratum", {}).get(stratum, {}).get("miss_rate")
                if rate is not None:
                    by_method[name][stratum].append(rate)

    out: dict = {}
    for name, by_stratum in by_method.items():
        out[name] = {}
        for stratum, vals in by_stratum.items():
            if not vals:
                out[name][stratum] = None
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            ci = bootstrap_ci_95(vals)
            out[name][stratum] = {
                "n_seeds": len(vals),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                "values": [round(v, 4) for v in vals],
            }
    return out


def aggregate_table4(run_dirs: list[Path], backend: str) -> dict:
    """Aggregate per-seed Table 4 (head-to-head baseline)."""
    by_method: dict[str, dict[str, list[float]]] = {}
    for d in run_dirs:
        path = d / backend / "table4_baseline.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for m in data.get("methods", []):
            name = m["name"]
            by_method.setdefault(name, {"crit_recall": [], "total_cost": []})
            crit = m.get("per_stratum", {}).get("CRITICAL", {})
            r = crit.get("safety_recall")
            if r is not None:
                by_method[name]["crit_recall"].append(r)
            c = m.get("total_cost")
            if c is not None:
                by_method[name]["total_cost"].append(c)

    out: dict = {}
    for name, metrics in by_method.items():
        out[name] = {}
        for k, vals in metrics.items():
            if not vals:
                out[name][k] = None
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            ci = bootstrap_ci_95(vals)
            out[name][k] = {
                "n_seeds": len(vals),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                "values": [round(v, 4) for v in vals],
            }
    return out


def render_md(table1: dict, table4: dict, backends: list[str], n_seeds: int) -> str:
    lines = [
        "# UASEF — Multi-Seed Bootstrap Aggregate",
        "",
        f"- Aggregated over **{n_seeds} seeds**",
        f"- Backends: {', '.join(backends)}",
        f"- Generated: {datetime.now().isoformat()}",
        "",
        "Each cell reports `mean ± std  [95% bootstrap CI]`.",
        "",
    ]
    for backend in backends:
        lines.append(f"## {backend}")
        lines.append("")
        if backend in table1 and table1[backend]:
            lines.append("### Table 1 — Per-Stratum Miss Rate")
            lines.append("")
            lines.append("| Method | CRITICAL | HIGH | MODERATE | LOW |")
            lines.append("| --- | --- | --- | --- | --- |")
            for name, by_s in table1[backend].items():
                row = [name]
                for s in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
                    cell = by_s.get(s)
                    if cell is None:
                        row.append("—")
                    else:
                        ci = cell.get("ci95")
                        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
                        row.append(f"{cell['mean']:.3f} ± {cell['std']:.3f}  {ci_str}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        if backend in table4 and table4[backend]:
            lines.append("### Table 4 — Head-to-Head (CRITICAL recall + total cost)")
            lines.append("")
            lines.append("| Method | CRITICAL Safety Recall | Total Cost |")
            lines.append("| --- | --- | --- |")
            for name, metrics in table4[backend].items():
                cells = []
                for k in ["crit_recall", "total_cost"]:
                    cell = metrics.get(k)
                    if cell is None:
                        cells.append("—")
                    else:
                        ci = cell.get("ci95")
                        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
                        cells.append(f"{cell['mean']:.3f} ± {cell['std']:.3f}  {ci_str}")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--backends", nargs="+", required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    table1: dict = {}
    table4: dict = {}
    for backend in args.backends:
        table1[backend] = aggregate_table1(args.runs, backend)
        table4[backend] = aggregate_table4(args.runs, backend)

    payload = {
        "n_seeds": len(args.runs),
        "backends": args.backends,
        "run_dirs": [str(r) for r in args.runs],
        "timestamp": datetime.now().isoformat(),
        "table1": table1,
        "table4": table4,
    }
    (args.output / "aggregate_seeds.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    md = render_md(table1, table4, args.backends, len(args.runs))
    (args.output / "aggregate_seeds.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n✅ Saved: {args.output}/aggregate_seeds.{{json,md}}")


if __name__ == "__main__":
    main()
