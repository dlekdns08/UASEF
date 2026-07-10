"""
Statistics utilities — the ONLY inference machinery any cell/summary analysis may use.
(Locked before data collection completes; see improvements/analysis_plan.md.)

Design rule: the unit of resampling is NEVER the row. Rows repeat canonical items across
cells, so row-level bootstrap/permutation pseudo-replicates. Resample:
  * matrix cells      -> item_id
  * shuffle audit     -> shuffle_group_id (= canonical item; original+shuffled stay together)
  * pooled/multi-cell -> item_id clusters (rows of the same item move together)

All functions are deterministic (explicit seed).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.label_conditional_conformal import _auroc


def sym_auroc(score, y):
    a = _auroc(np.asarray(score, float), np.asarray(y, int))
    return max(a, 1 - a)


def paired_bootstrap_auroc_diff(score_a, score_b, y, groups, n_boot=2000, seed=0):
    """CI for AUROC(a) - AUROC(b) on the SAME rows, resampling GROUPS (item_id /
    shuffle_group_id) with replacement. Returns dict with point estimate, 95% CI,
    and a two-sided bootstrap p (2 * min tail mass of the null value 0)."""
    score_a, score_b, y = map(np.asarray, (score_a, score_b, y))
    idx_by_g = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by_g[g].append(i)
    keys = sorted(idx_by_g)
    point = sym_auroc(score_a, y) - sym_auroc(score_b, y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        take = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([idx_by_g[keys[k]] for k in take])
        yy = y[rows]
        if yy.sum() < 2 or (1 - yy).sum() < 2:
            continue
        diffs.append(sym_auroc(score_a[rows], yy) - sym_auroc(score_b[rows], yy))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": round(float(point), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "p_boot": round(float(min(1.0, p)), 4),
            "n_boot_valid": int(len(diffs))}


def grouped_bootstrap_ci(values_fn, groups, n_boot=2000, seed=0):
    """generic group-resampled CI: values_fn(row_indices)-> scalar; groups = per-row group id."""
    idx_by_g = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by_g[g].append(i)
    keys = sorted(idx_by_g)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        take = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([idx_by_g[keys[k]] for k in take])
        v = values_fn(rows)
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def mcnemar_p(b, c):
    """exact binomial McNemar for paired accuracy change (b = only-A correct,
    c = only-B correct discordant counts)."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return round(float(binomtest(min(b, c), n, 0.5).pvalue * 1.0), 5)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    cummin = 1.0
    for rank_from_end, i in enumerate(order[::-1]):
        k = m - rank_from_end          # rank of this p (1-indexed from smallest)
        cummin = min(cummin, p[i] * m / k)
        adj[i] = cummin
    return [round(float(x), 5) for x in adj]
