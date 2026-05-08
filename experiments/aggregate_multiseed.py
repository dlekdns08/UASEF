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


def _binom_two_sided_pvalue(b: int, c: int) -> float:
    """
    Exact two-sided binomial p-value for McNemar (b discordant for A, c for B).
    Under H0 each discordant pair is 50/50; statistic = min(b, c) ~ Binomial(b+c, 0.5).
    Uses normal approximation if b+c >= 25 (matches scipy default), else exact.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n < 25:
        # exact two-sided: 2 * sum_{i=0..k} C(n,i) * 0.5**n
        from math import comb
        cdf = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        return min(1.0, 2 * cdf)
    # normal approx with continuity correction
    mean = n / 2
    var = n / 4
    z = (abs(b - c) - 1) / math.sqrt(var) if var > 0 else 0.0
    # 2 * (1 - Phi(z))
    from math import erf, sqrt
    p = 1 - 0.5 * (1 + erf(z / sqrt(2)))
    return min(1.0, 2 * p)


def aggregate_mcnemar_seed_pooled(run_dirs: list[Path], backend: str) -> dict:
    """
    Pool discordant pair counts across seeds and recompute McNemar.
    Per-seed McNemar p-values are not directly poolable; the principled fix
    is to sum the (b, c) discordant counts before testing. We also report
    Fisher-style combination of per-seed p-values for reference.

    Each per-seed table4_baseline.json must contain a 'mcnemar' block with
    {pair_name: {"b": ..., "c": ..., "p_value": ...}}.
    """
    pooled: dict[str, dict] = {}
    per_seed_pvals: dict[str, list[float]] = {}
    for d in run_dirs:
        path = d / backend / "table4_baseline.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for pair, info in (data.get("mcnemar") or {}).items():
            entry = pooled.setdefault(pair, {"b_total": 0, "c_total": 0, "n_seeds": 0})
            b = int(info.get("b", 0) or 0)
            c = int(info.get("c", 0) or 0)
            entry["b_total"] += b
            entry["c_total"] += c
            entry["n_seeds"] += 1
            p = info.get("p_value")
            if p is not None:
                per_seed_pvals.setdefault(pair, []).append(float(p))

    out: dict = {}
    for pair, e in pooled.items():
        p_pooled = _binom_two_sided_pvalue(e["b_total"], e["c_total"])
        # Fisher's combined p: -2 * sum(ln p_i) ~ chi2_{2k}
        fisher_p = None
        seed_ps = per_seed_pvals.get(pair, [])
        if len(seed_ps) >= 2 and all(p > 0 for p in seed_ps):
            from math import log, exp
            stat = -2 * sum(log(p) for p in seed_ps)
            k = len(seed_ps)
            # chi2 survival = regularized upper incomplete gamma
            try:
                from math import lgamma
                # naive series for small k; acceptable for k<=10
                def chi2_sf(x: float, df: int) -> float:
                    # Use upper-incomplete-gamma series (Abramowitz 6.5.32) approx
                    # Fall back to scipy if available.
                    try:
                        from scipy.stats import chi2  # type: ignore
                        return float(chi2.sf(x, df))
                    except Exception:
                        # crude approx via wilson-hilferty
                        h = 2.0 / (9 * df)
                        z = ((x / df) ** (1 / 3) - (1 - h)) / math.sqrt(h)
                        return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
                fisher_p = chi2_sf(stat, 2 * k)
            except Exception:
                fisher_p = None
        out[pair] = {
            "n_seeds": e["n_seeds"],
            "b_total": e["b_total"],
            "c_total": e["c_total"],
            "p_pooled": round(p_pooled, 5),
            "p_fisher_combined": round(fisher_p, 5) if fisher_p is not None else None,
            "per_seed_p_values": [round(p, 5) for p in seed_ps],
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
    mcnemar: dict = {}
    for backend in args.backends:
        table1[backend] = aggregate_table1(args.runs, backend)
        table4[backend] = aggregate_table4(args.runs, backend)
        mcnemar[backend] = aggregate_mcnemar_seed_pooled(args.runs, backend)

    payload = {
        "n_seeds": len(args.runs),
        "backends": args.backends,
        "run_dirs": [str(r) for r in args.runs],
        "timestamp": datetime.now().isoformat(),
        "table1": table1,
        "table4": table4,
        "mcnemar_seed_pooled": mcnemar,
    }
    (args.output / "aggregate_seeds.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    md = render_md(table1, table4, args.backends, len(args.runs))
    # Append McNemar pooled section.
    if any(mcnemar[b] for b in args.backends):
        extra = ["", "## McNemar (seed-pooled discordant counts)", ""]
        for backend in args.backends:
            if not mcnemar[backend]:
                continue
            extra.append(f"### {backend}")
            extra.append("")
            extra.append("| Pair | n_seeds | b_total | c_total | p_pooled | p_fisher |")
            extra.append("| --- | --- | --- | --- | --- | --- |")
            for pair, info in mcnemar[backend].items():
                extra.append(
                    f"| {pair} | {info['n_seeds']} | {info['b_total']} | "
                    f"{info['c_total']} | {info['p_pooled']} | "
                    f"{info['p_fisher_combined'] if info['p_fisher_combined'] is not None else '—'} |"
                )
            extra.append("")
        md = md + "\n" + "\n".join(extra)
    (args.output / "aggregate_seeds.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n✅ Saved: {args.output}/aggregate_seeds.{{json,md}}")


if __name__ == "__main__":
    main()
