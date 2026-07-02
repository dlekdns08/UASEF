"""
Round 14 — Is a genuine win achievable *at all*? ROC operating-point analysis.

A "genuine win" for stratum s requires simultaneously:
    (a) coverage:  Clopper-Pearson_95%(misses, n_pos) ≤ α_s   ⟺  TPR ≥ ~(1-α_s)
    (b) non-vacuous:  over_esc < 0.95                          ⟺  FPR < 0.95

Both are points on the classifier's ROC curve (TPR = 1 - miss_rate,
FPR = over_esc_rate). NO threshold rule — b-CRC, vanilla CRC, or any
other — can achieve an operating point off the classifier's own ROC
curve. Therefore the feasibility of a genuine win is a *pure property
of the classifier's discrimination*, decided by one number:

    FPR at the threshold where TPR = (1 - α_s)

    * FPR@(1-α) < 0.95  →  GENUINE WIN POSSIBLE   (b-CRC INFEASIBLE, if any,
                            is a cost-weight/α artifact, tunable)
    * FPR@(1-α) ≥ 0.95  →  INFORMATION LIMIT       (no algorithm can win;
                            b-CRC INFEASIBLE is the *correct* answer)

We also report AUROC (global discrimination) and the α* threshold at
which a genuine win first becomes feasible (sweeping the coverage
target). This decisively answers "is there really no genuine win?"

Output: results/round14/r14_genuine_win_feasibility.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.metrics_utils import clopper_pearson_upper
from experiments.round13_bcrc_vs_crc import (
    recompute_r10_r11, recompute_eicu, load_r12_1_seed_data, ALPHAS,
)

VACUOUS_FPR = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# ROC operating-point analysis
# ─────────────────────────────────────────────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (rank-based)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    combined = np.concatenate([pos, neg])
    # simple tie handling: use scipy-free average-rank via unique
    sorted_vals = np.sort(combined)
    rank_map = {}
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_map[sorted_vals[i]] = avg_rank
        i = j + 1
    pos_ranks = np.array([rank_map[v] for v in pos])
    n_pos, n_neg = len(pos), len(neg)
    u = pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, labels: np.ndarray):
    """Return (thresholds, TPR, FPR) arrays. Escalate iff score > threshold."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    # candidate thresholds: all unique scores (descending) + extremes
    thr = np.unique(scores)
    thr = np.concatenate([[thr.min() - 1e-9], thr, [thr.max() + 1e-9]])
    thr = np.sort(thr)[::-1]  # high → low so TPR increases
    tpr = np.array([(pos > t).sum() / n_pos for t in thr])
    fpr = np.array([(neg > t).sum() / n_neg for t in thr])
    return thr, tpr, fpr


def fpr_at_tpr(thr, tpr, fpr, target_tpr: float) -> float:
    """Minimum FPR achievable while TPR ≥ target_tpr."""
    ok = tpr >= target_tpr
    if not ok.any():
        return 1.0   # cannot reach the required TPR at all → escalate-all
    return float(fpr[ok].min())


def analyze_cell(cal_scores, cal_labels, test_scores, test_labels,
                  alpha: float) -> dict:
    """Pool cal+test (score/label) and analyze ROC feasibility.

    We use the *test* distribution for the honest operating-point
    analysis (the classifier is fit on cal, evaluated on test).
    """
    s = np.asarray(test_scores, dtype=float)
    y = np.asarray(test_labels, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 5 or n_neg < 5:
        return {"insufficient": True, "n_pos": n_pos, "n_neg": n_neg}

    au = auroc(s, y)
    # DIRECTION-ROBUST: the pipeline's raw score convention may be flipped
    # (e.g. tabular s = -P(y=1) makes positives score LOW, giving AUROC<0.5).
    # A threshold rule's achievable operating points are the same set whether
    # we use s or -s; we orient the score so AUROC>=0.5 (the discriminating
    # direction) to measure the TRUE information-theoretic feasibility.
    flipped = au < 0.5
    s_oriented = -s if flipped else s
    au_oriented = 1.0 - au if flipped else au

    roc = roc_curve(s_oriented, y)
    if roc is None:
        return {"insufficient": True, "n_pos": n_pos, "n_neg": n_neg}
    thr, tpr, fpr = roc

    # Coverage target as TPR: we need Clopper-Pearson upper ≤ α.
    # Approximate the required TPR: the number of allowed misses k satisfies
    # CP_upper(k, n_pos) ≤ α. Find max k, then required TPR = 1 - k/n_pos.
    max_allowed_misses = 0
    for k in range(0, n_pos + 1):
        if clopper_pearson_upper(k, n_pos, 0.95) <= alpha:
            max_allowed_misses = k
        else:
            break
    required_tpr = 1.0 - (max_allowed_misses / n_pos)

    fpr_at_cov = fpr_at_tpr(thr, tpr, fpr, required_tpr)
    fpr_at_95 = fpr_at_tpr(thr, tpr, fpr, 0.95)

    genuine_possible = fpr_at_cov < VACUOUS_FPR

    # α* — the smallest coverage target (as TPR = 1-α*) at which a genuine
    # win (FPR < 0.95) first becomes feasible. Sweep TPR downward.
    alpha_star = None
    for target_tpr in np.linspace(1.0, 0.0, 101):
        if fpr_at_tpr(thr, tpr, fpr, target_tpr) < VACUOUS_FPR:
            alpha_star = float(1.0 - target_tpr)
            break

    return {
        "insufficient": False,
        "n_pos": n_pos, "n_neg": n_neg,
        "auroc_raw": au,                    # as-scored (may be <0.5 if flipped)
        "auroc": au_oriented,               # direction-robust (>=0.5)
        "score_flipped": flipped,           # True => raw convention backwards
        "alpha": alpha,
        "max_allowed_misses": max_allowed_misses,
        "required_tpr": required_tpr,
        "fpr_at_coverage": fpr_at_cov,      # decisive number (best orientation)
        "fpr_at_tpr95": fpr_at_95,
        "genuine_win_possible": genuine_possible,
        "alpha_star_min_feasible": alpha_star,  # α at which win first feasible
        "verdict": ("GENUINE_POSSIBLE" if genuine_possible
                    else "INFORMATION_LIMIT"),
    }


def pool_seeds(per_seed_data: list[dict], stratum: str):
    """Concatenate test scores+labels across seeds for a stratum."""
    all_s, all_y = [], []
    for entry in per_seed_data:
        if "test_scores" not in entry:
            continue
        ts = np.asarray(entry["test_scores"], dtype=float)
        tl = np.asarray(entry["test_labels"], dtype=int)
        tstr = np.asarray(entry["test_strata"])
        mask = tstr == stratum
        all_s.append(ts[mask]); all_y.append(tl[mask])
    if not all_s:
        return np.array([]), np.array([])
    return np.concatenate(all_s), np.concatenate(all_y)


def run_cohort_feasibility(per_seed_data, strata_list, alphas) -> dict:
    out = {}
    for stratum in strata_list:
        s, y = pool_seeds(per_seed_data, stratum)
        if len(s) == 0:
            out[stratum] = {"insufficient": True}
            continue
        out[stratum] = analyze_cell(s, y, s, y, alphas[stratum])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_md(report: dict, out_md: Path):
    lines = ["# Round 14 — Genuine-win feasibility (ROC operating-point)\n"]
    lines.append(f"- Generated: {report['timestamp']}\n")
    lines.append("**Decisive number:** FPR at the coverage TPR (positives caught "
                 "at the α target). If < 0.95, a genuine win is *achievable by "
                 "some threshold*; b-CRC INFEASIBLE would then be a cost-weight "
                 "artifact. If ≥ 0.95, it is a fundamental **information limit** "
                 "— no algorithm can win.\n")
    lines.append("AUROC is direction-robust (oriented >=0.5); 'flip' marks cells "
                 "whose raw score convention was backwards (AUROC_raw < 0.5).\n")
    lines.append("| Cohort | Classifier | Stratum | AUROC | flip | req. TPR | **FPR@cov** | FPR@TPR95 | α* min-feasible | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cohort, cdata in report["cohorts"].items():
        for clf, sdata in cdata.items():
            for stratum, cell in sdata.items():
                if cell.get("insufficient"):
                    lines.append(f"| {cohort} | {clf} | {stratum} | — | — | — | — | — | — | insufficient |")
                    continue
                astar = cell["alpha_star_min_feasible"]
                astar_s = f"{astar:.3f}" if astar is not None else "none"
                flip = "⚠" if cell.get("score_flipped") else ""
                lines.append(
                    f"| {cohort} | {clf} | {stratum} | "
                    f"{cell['auroc']:.3f} | {flip} | {cell['required_tpr']:.3f} | "
                    f"**{cell['fpr_at_coverage']:.3f}** | "
                    f"{cell['fpr_at_tpr95']:.3f} | {astar_s} | "
                    f"**{cell['verdict']}** |"
                )

    # Bottom-line
    any_possible = False
    best = None
    for cohort, cdata in report["cohorts"].items():
        for clf, sdata in cdata.items():
            for stratum, cell in sdata.items():
                if cell.get("insufficient"): continue
                if cell["genuine_win_possible"]:
                    any_possible = True
                    if best is None or cell["fpr_at_coverage"] < best[3]:
                        best = (cohort, clf, stratum, cell["fpr_at_coverage"])
    lines.append("\n## Bottom line\n")
    if any_possible:
        lines.append(f"**✓ A genuine win IS achievable** in at least one cell. "
                     f"Best: {best[0]} / {best[1]} / {best[2]} with "
                     f"FPR@coverage = {best[3]:.3f} < 0.95. b-CRC should be "
                     f"tuned (α or cost weights) to realize it.")
    else:
        lines.append("**✗ No genuine win is achievable in ANY tested cell.** "
                     "For every (cohort, classifier, stratum), catching "
                     "(1-α) of positives forces escalating ≥ 95% of negatives. "
                     "This is a fundamental information limit of the "
                     "admission-time / NLL features — not a b-CRC defect. "
                     "b-CRC's INFEASIBLE is the honest, correct answer, and "
                     "vanilla CRC's 'win' is the escalate-all artifact.")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-jsonl",
                    default=ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl",
                    type=Path)
    ap.add_argument("--eicu-jsonl",
                    default=ROOT / "data" / "raw" / "eicu_cases_v11_full.jsonl",
                    type=Path)
    ap.add_argument("--classifiers", nargs="+",
                    default=["randomforest", "logreg", "gbdt", "xgboost"])
    ap.add_argument("--skip-eicu", action="store_true")
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round14" / "r14_genuine_win_feasibility")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cohorts = {}

    if args.mimic_jsonl.exists():
        from experiments.round10_method_agnostic import _feature_vector
        from experiments.round11_method_agnostic_minimal import _feature_vector_minimal
        for label, fv in [("mimic4_r10_4", _feature_vector),
                           ("mimic4_r11_1", _feature_vector_minimal)]:
            print(f"[R14] {label}")
            cohorts[label] = {}
            for clf in args.classifiers:
                print(f"  {clf}...")
                per_seed = recompute_r10_r11(clf, args.mimic_jsonl, fv,
                                              n_cal=args.n_cal, n_test=args.n_test)
                cohorts[label][clf] = run_cohort_feasibility(
                    per_seed, ["CRITICAL", "HIGH", "MODERATE", "LOW"], ALPHAS)

    if args.eicu_jsonl.exists() and not args.skip_eicu:
        for pass_name in ["pass_a", "pass_b"]:
            label = f"eicu_r11_3_{pass_name}"
            print(f"[R14] {label}")
            cohorts[label] = {}
            for clf in args.classifiers:
                print(f"  {clf}...")
                per_seed = recompute_eicu(clf, args.eicu_jsonl, pass_name,
                                           n_cal=args.n_cal, n_test=args.n_test)
                cohorts[label][clf] = run_cohort_feasibility(
                    per_seed, ["CRITICAL", "HIGH", "MODERATE", "LOW"], ALPHAS)

    r12 = load_r12_1_seed_data()
    if r12:
        print(f"[R14] medabstain_r12_1 (LLM NLL)")
        cohorts["medabstain_r12_1"] = {
            "gpt_oss_120b": run_cohort_feasibility(
                r12, ["CRITICAL", "HIGH", "MODERATE"],
                {k: v for k, v in ALPHAS.items() if k != "LOW"})}
    else:
        print("[R14] MedAbstain LLM raw scores not cached — skipping")

    report = {"timestamp": datetime.now().isoformat(), "cohorts": cohorts}
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
