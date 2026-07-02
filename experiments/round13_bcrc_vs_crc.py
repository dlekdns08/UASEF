"""
Round 13 — b-CRC vs vanilla CRC head-to-head on the four cohorts.

Loads the cached calibration/test scores from Rounds 10-12 (no LLM
re-inference required) and applies both threshold rules per stratum.

Cohorts (all cached, tabular fits are cheap so LLM/tabular symmetry
via cache is maintained):
    * MIMIC-IV R10.4 5-classifier method-agnostic pool
    * eICU R11.3 tabular Pass A + Pass B
    * MedAbstain R12.1 gpt-oss-120b (per-seed cache)

For each (cohort, classifier, stratum) triple report both:
    - Vanilla CRC (existing paper §3 numbers)
    - b-CRC with a small grid of (c_miss, c_over) ∈ {(0.95,0.05),
      (0.90,0.10), (0.80,0.20)}
    - Verdict: INFEASIBLE / GENUINE_WIN / FAIL

Output: results/round13/r13_bcrc_vs_crc.{json,md}
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

from models.bounded_crc import BoundedCRC
from experiments.metrics_utils import clopper_pearson_upper


# ─────────────────────────────────────────────────────────────────────────────
# Cost-weight grid
# ─────────────────────────────────────────────────────────────────────────────

# Unified cost sweep. c_over=0.0 IS standard "vanilla" CRC (miss-only
# loss); increasing c_over engages b-CRC's escalate-all exclusion.
# This makes vanilla a *special case* of the same code path, so the
# comparison is exact (no separate reimplementation).
COST_GRID = [
    (1.00, 0.00),   # vanilla CRC (miss-only) — reproduces paper Rounds 10-12
    (0.95, 0.05),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.50, 0.50),   # symmetric extreme (informative)
]

ALPHAS = {"CRITICAL": 0.05, "HIGH": 0.10, "MODERATE": 0.15, "LOW": 0.20}


def eval_threshold(scores: np.ndarray, labels: np.ndarray,
                    lam: float) -> dict:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    escalate = scores > lam
    misses = int(((labels == 1) & (~escalate)).sum())
    over_esc = int(((labels == 0) & escalate).sum())
    miss_rate = (misses / n_pos) if n_pos else None
    over_rate = (over_esc / n_neg) if n_neg else None
    upper = clopper_pearson_upper(misses, n_pos, 0.95) if n_pos else None
    return {
        "threshold": lam,
        "n": len(scores), "n_pos": n_pos, "n_neg": n_neg,
        "misses": misses, "miss_rate": miss_rate,
        "over_esc": over_esc, "over_esc_rate": over_rate,
        "exact_upper95": upper,
    }


def strict_verdict(row: dict, alpha: float,
                     vacuous_thr: float = 0.95) -> str:
    if row.get("infeasible"):
        return "INFEASIBLE"
    upper = row.get("exact_upper95")
    over_rate = row.get("over_esc_rate")
    if upper is None or over_rate is None:
        return "N/A"
    sat = upper <= alpha
    vac = over_rate >= vacuous_thr
    if sat and not vac:
        return "GENUINE_WIN"
    if sat and vac:
        return "VACUOUS_WIN"
    return "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Per-classifier evaluator (uses cached per-seed scores)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cell(cal_scores, cal_labels, test_scores, test_labels,
                    alpha: float) -> dict:
    """Return the cost-sweep rows for a single (seed, stratum).

    c_over=0.0 (COST_GRID[0]) is vanilla CRC; everything else is b-CRC.
    A single BoundedCRC code path fits all, so vanilla is an exact special
    case (no separate reimplementation to drift from the paper numbers).
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=int)
    test_scores = np.asarray(test_scores, dtype=float)
    test_labels = np.asarray(test_labels, dtype=int)

    def _clean(s, l):
        m = np.isfinite(s)
        return s[m], l[m]
    cal_scores, cal_labels = _clean(cal_scores, cal_labels)
    test_scores, test_labels = _clean(test_scores, test_labels)

    out = {"b_crc": {}}
    for cm, co in COST_GRID:
        key = f"cm{cm}_co{co}"
        bcrc = BoundedCRC(alpha=alpha, c_miss=cm, c_over=co)
        bcrc.fit(cal_scores, cal_labels)
        if bcrc.infeasible_:
            out["b_crc"][key] = {"infeasible": True, "verdict": "INFEASIBLE",
                                  "c_miss": cm, "c_over": co}
            continue
        lam_b = bcrc.threshold_
        row = eval_threshold(test_scores, test_labels, lam_b)
        row["c_miss"] = cm; row["c_over"] = co
        row["verdict"] = strict_verdict(row, alpha)
        out["b_crc"][key] = row
    # Alias the vanilla (c_over=0) cell for convenience
    out["vanilla"] = out["b_crc"][f"cm{COST_GRID[0][0]}_co{COST_GRID[0][1]}"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cohort loaders: reuse cached per-seed scores
# ─────────────────────────────────────────────────────────────────────────────

def load_r10_4_seed_data(classifier: str) -> list[dict]:
    """R10.4 cache: results/round10/r10_4_cache/<clf>.json.

    NOTE: r10_4_cache 는 per_seed 요약만 담고 있고 raw scores 는 없다.
    따라서 R10.4 는 recomputation 이 필요 — LLM 은 cost 크므로 tabular 만.
    """
    p = ROOT / "results" / "round10" / "r10_4_cache" / f"{classifier}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    ps = d.get("per_seed", [])
    if ps and "cal_scores" in ps[0]:
        return ps
    return []   # raw scores 없음


def recompute_r10_r11(classifier: str, jsonl_path: Path,
                        feature_vec_fn, n_cal=3000, n_test=3000,
                        seeds=(42, 43, 44, 45, 46)) -> list[dict]:
    """Recompute tabular per-seed scores for one classifier."""
    from data.loader import _load_mimic4_jsonl
    from experiments.round10_method_agnostic import (
        _make_classifier, _subject_id, _attach_r10_features,
    )
    from experiments.metrics_utils import patient_level_split
    import random as _random

    if not jsonl_path.exists():
        return []

    cases = _load_mimic4_jsonl(jsonl_path, n=10**9, seed=42)
    _attach_r10_features(cases, jsonl_path)
    per_seed = []
    for seed in seeds:
        cal, test = patient_level_split(cases, group_of=_subject_id,
                                          cal_frac=0.8, seed=seed)
        rng = _random.Random(seed); rng.shuffle(cal); rng.shuffle(test)
        cal = cal[:n_cal]; test = test[:n_test]

        clf = _make_classifier(classifier)
        X_cal = [feature_vec_fn(c) for c in cal]
        y_cal = [bool(c.expected_escalate) for c in cal]
        if not clf.fit(X_cal, y_cal):
            per_seed.append({"seed": seed, "error": "fit fail"}); continue
        cal_scores = [clf.score(x) for x in X_cal]
        X_test = [feature_vec_fn(c) for c in test]
        test_scores = [clf.score(x) for x in X_test]
        per_seed.append({
            "seed": seed,
            "cal_scores": cal_scores,
            "test_scores": test_scores,
            "cal_labels": [int(bool(c.expected_escalate)) for c in cal],
            "test_labels": [int(bool(c.expected_escalate)) for c in test],
            "cal_strata": [(c.scenario_type or "").upper() for c in cal],
            "test_strata": [(c.scenario_type or "").upper() for c in test],
        })
    return per_seed


def recompute_eicu(classifier: str, eicu_jsonl: Path, pass_name: str,
                    n_cal=3000, n_test=3000,
                    seeds=(42, 43, 44, 45, 46)) -> list[dict]:
    """Recompute tabular per-seed scores for eICU (Pass A or Pass B)."""
    import random as _random
    from experiments.round11_eicu_replication import (
        load_eicu_jsonl, fv_pass_a, fv_pass_b, EHRCaseLite,
    )
    from experiments.round10_method_agnostic import _make_classifier
    from experiments.metrics_utils import patient_level_split

    if not eicu_jsonl.exists():
        return []
    fv_fn = fv_pass_a if pass_name == "pass_a" else fv_pass_b

    def _case_to_dict(c):
        return {k: getattr(c, k) for k in EHRCaseLite.__slots__}

    def _subj(c):
        return c.subject_id

    cases = load_eicu_jsonl(eicu_jsonl)
    per_seed = []
    for seed in seeds:
        cal, test = patient_level_split(cases, group_of=_subj,
                                          cal_frac=0.8, seed=seed)
        rng = _random.Random(seed); rng.shuffle(cal); rng.shuffle(test)
        cal = cal[:n_cal]; test = test[:n_test]

        clf = _make_classifier(classifier)
        X_cal = [fv_fn(_case_to_dict(c)) for c in cal]
        y_cal = [bool(c.expected_escalate) for c in cal]
        if not clf.fit(X_cal, y_cal):
            per_seed.append({"seed": seed, "error": "fit fail"}); continue
        cal_scores = [clf.score(x) for x in X_cal]
        X_test = [fv_fn(_case_to_dict(c)) for c in test]
        test_scores = [clf.score(x) for x in X_test]
        per_seed.append({
            "seed": seed,
            "cal_scores": cal_scores, "test_scores": test_scores,
            "cal_labels": [int(bool(c.expected_escalate)) for c in cal],
            "test_labels": [int(bool(c.expected_escalate)) for c in test],
            "cal_strata": [(c.scenario_type or "").upper() for c in cal],
            "test_strata": [(c.scenario_type or "").upper() for c in test],
        })
    return per_seed


def load_r12_1_seed_data() -> list[dict]:
    """R12.1 cache: results/round12/r12_1_cache/seed_*.json.

    Contains raw LLM scores per seed. Score = mean token NLL.
    """
    cache_dir = ROOT / "results" / "round12" / "r12_1_cache"
    per_seed = []
    for p in sorted(cache_dir.glob("seed_*.json")):
        d = json.loads(p.read_text())
        # r12 cache 는 per_stratum 요약만 있음; raw scores 는 저장 안 됨.
        # 이 경우 skip (R12 재실행 필요).
        if "cal_scores" not in d:
            continue
        per_seed.append(d)
    return per_seed


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate + report
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_over_seeds(per_seed_rows: list[dict], stratum: str,
                            alpha: float, key: str) -> dict:
    misses = npos = oe = nneg = 0
    infeasible_ct = 0
    n_valid = 0
    for r in per_seed_rows:
        cell = r.get(key)
        if not cell: continue
        if cell.get("infeasible"):
            infeasible_ct += 1
            continue
        st_row = cell.get(stratum)
        if not st_row: continue
        if st_row.get("infeasible"):
            infeasible_ct += 1; continue
        if st_row.get("n_pos", 0) == 0: continue
        misses += st_row["misses"]; npos += st_row["n_pos"]
        oe += st_row["over_esc"]; nneg += st_row["n_neg"]
        n_valid += 1
    if n_valid == 0:
        return {"infeasible_all": True, "n_seeds_infeasible": infeasible_ct,
                "verdict": "INFEASIBLE"}
    upper = clopper_pearson_upper(misses, npos, 0.95) if npos else None
    over_rate = (oe / nneg) if nneg else None
    verdict = strict_verdict({
        "exact_upper95": upper, "over_esc_rate": over_rate,
    }, alpha)
    return {
        "pooled_misses": misses, "pooled_n_pos": npos,
        "pooled_miss_rate": misses / npos if npos else None,
        "pooled_over_esc": oe, "pooled_n_neg": nneg,
        "pooled_over_esc_rate": over_rate,
        "pooled_exact_upper95": upper,
        "n_seeds_infeasible": infeasible_ct,
        "n_seeds_valid": n_valid,
        "alpha": alpha,
        "verdict": verdict,
    }


def run_cohort(cohort_name: str, per_seed_data: list[dict],
                strata_list: list[str], alphas: dict) -> dict:
    """Given per-seed data (each seed with cal_/test_ scores/labels/strata),
    produce vanilla + b-CRC results for every stratum."""

    # per_seed_rows: for each seed, a dict with vanilla + b_crc sub-rows,
    # each keyed by stratum.
    per_seed_rows = []
    for entry in per_seed_data:
        if "cal_scores" not in entry:
            continue
        seed = entry.get("seed")
        row = {"seed": seed, "vanilla": {}, "b_crc": {}}
        for cm, co in COST_GRID:
            row["b_crc"][f"cm{cm}_co{co}"] = {}
        cal_scores = np.array(entry["cal_scores"], dtype=float)
        test_scores = np.array(entry["test_scores"], dtype=float)
        cal_labels = np.array(entry["cal_labels"], dtype=int)
        test_labels = np.array(entry["test_labels"], dtype=int)
        cal_strata = np.array(entry["cal_strata"])
        test_strata = np.array(entry["test_strata"])
        for stratum in strata_list:
            cal_m = cal_strata == stratum
            test_m = test_strata == stratum
            if cal_m.sum() < 10 or test_m.sum() < 5:
                row["vanilla"][stratum] = {"infeasible": True,
                                            "reason": "insufficient cal/test n"}
                for k in row["b_crc"]:
                    row["b_crc"][k][stratum] = {"infeasible": True}
                continue
            result = evaluate_cell(
                cal_scores[cal_m], cal_labels[cal_m],
                test_scores[test_m], test_labels[test_m],
                alphas[stratum])
            row["vanilla"][stratum] = result["vanilla"]
            for k, v in result["b_crc"].items():
                row["b_crc"][k][stratum] = v
        per_seed_rows.append(row)

    # Aggregate
    agg = {"vanilla": {}, "b_crc": {}}
    for stratum in strata_list:
        agg["vanilla"][stratum] = aggregate_over_seeds(
            per_seed_rows, stratum, alphas[stratum], "vanilla")
    for cm, co in COST_GRID:
        key = f"cm{cm}_co{co}"
        agg["b_crc"][key] = {}
        # per-seed rows for this cost setting
        cost_rows = [
            {"cost_key": r["b_crc"][key]} for r in per_seed_rows
        ]
        # aggregate needs the strata inside the cost dict
        for stratum in strata_list:
            packed = [
                {stratum: r["b_crc"][key].get(stratum, {})}
                for r in per_seed_rows
            ]
            # Wrap each into the format aggregate_over_seeds expects
            wrapped = [{"cell": {stratum: p[stratum]}} for p in packed]
            # simple: iterate manually
            misses = npos = oe = nneg = 0; n_valid = 0; infeasible_ct = 0
            for r in per_seed_rows:
                st_row = r["b_crc"][key].get(stratum, {})
                if st_row.get("infeasible"):
                    infeasible_ct += 1; continue
                if st_row.get("n_pos", 0) == 0: continue
                misses += st_row["misses"]; npos += st_row["n_pos"]
                oe += st_row["over_esc"]; nneg += st_row["n_neg"]
                n_valid += 1
            if n_valid == 0:
                agg["b_crc"][key][stratum] = {"infeasible_all": True,
                    "n_seeds_infeasible": infeasible_ct,
                    "verdict": "INFEASIBLE"}
                continue
            upper = clopper_pearson_upper(misses, npos, 0.95) if npos else None
            over_rate = (oe / nneg) if nneg else None
            agg["b_crc"][key][stratum] = {
                "pooled_misses": misses, "pooled_n_pos": npos,
                "pooled_miss_rate": misses / npos if npos else None,
                "pooled_over_esc": oe, "pooled_n_neg": nneg,
                "pooled_over_esc_rate": over_rate,
                "pooled_exact_upper95": upper,
                "n_seeds_infeasible": infeasible_ct,
                "n_seeds_valid": n_valid,
                "alpha": alphas[stratum],
                "c_miss": cm, "c_over": co,
                "verdict": strict_verdict({
                    "exact_upper95": upper, "over_esc_rate": over_rate,
                }, alphas[stratum]),
            }

    return {
        "cohort": cohort_name,
        "strata": strata_list,
        "alphas": {s: alphas[s] for s in strata_list},
        "cost_grid": COST_GRID,
        "n_seeds_used": len(per_seed_rows),
        "per_seed": per_seed_rows,
        "aggregate": agg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def write_md(report: dict, out_md: Path):
    lines = ["# Round 13 — b-CRC vs vanilla CRC head-to-head\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Cost grid: {report['cost_grid']}\n")
    lines.append("## Summary — Genuine wins per cohort\n")
    lines.append("(GENUINE_WIN = α-satisfy AND pooled over_esc < 0.95)\n")
    lines.append("| Cohort | Classifier | Stratum | Vanilla verdict | Best b-CRC (cm,co) | b-CRC verdict |")
    lines.append("|---|---|---|---|---|---|")
    for cohort, cohort_data in report["cohorts"].items():
        for clf_name, clf_result in cohort_data.items():
            for stratum in clf_result["strata"]:
                v = clf_result["aggregate"]["vanilla"].get(stratum, {})
                best_bcrc = None; best_verdict = "N/A"
                for cm, co in COST_GRID:
                    if co == 0.0:
                        continue   # c_o=0 is vanilla, not b-CRC
                    key = f"cm{cm}_co{co}"
                    b = clf_result["aggregate"]["b_crc"][key].get(stratum, {})
                    verdict = b.get("verdict", "N/A")
                    if verdict == "GENUINE_WIN":
                        best_bcrc = (cm, co); best_verdict = verdict; break
                    if verdict == "FAIL" and best_verdict != "FAIL":
                        best_bcrc = (cm, co); best_verdict = verdict
                    if verdict == "INFEASIBLE" and best_verdict == "N/A":
                        best_bcrc = (cm, co); best_verdict = verdict
                lines.append(
                    f"| {cohort} | {clf_name} | {stratum} | "
                    f"{v.get('verdict', 'N/A')} | "
                    f"{best_bcrc or '—'} | **{best_verdict}** |"
                )

    lines.append("\n## Full detail per cohort × classifier\n")
    for cohort, cohort_data in report["cohorts"].items():
        lines.append(f"\n### Cohort: {cohort}\n")
        for clf_name, clf_result in cohort_data.items():
            lines.append(f"\n#### {clf_name}\n")
            for stratum in clf_result["strata"]:
                lines.append(f"\n**{stratum}** (α = {clf_result['alphas'][stratum]})\n")
                lines.append("| Method | c_m | c_o | miss/n_pos | upper 95% | over_esc | verdict |")
                lines.append("|---|---|---|---|---|---|---|")
                for cm, co in COST_GRID:
                    key = f"cm{cm}_co{co}"
                    method = "Vanilla CRC" if co == 0.0 else "b-CRC"
                    b = clf_result["aggregate"]["b_crc"][key].get(stratum, {})
                    if b.get("infeasible_all"):
                        lines.append(
                            f"| {method} | {cm} | {co} | — | — | — | INFEASIBLE ({b['n_seeds_infeasible']}/5) |"
                        )
                        continue
                    lines.append(
                        f"| {method} | {cm} | {co} | "
                        f"{b['pooled_misses']}/{b['pooled_n_pos']} "
                        f"({(b['pooled_miss_rate'] or 0)*100:.2f}%) | "
                        f"{b['pooled_exact_upper95']:.4f} | "
                        f"{(b['pooled_over_esc_rate'] or 0)*100:.2f}% | "
                        f"**{b['verdict']}** |"
                    )
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
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round13" / "r13_bcrc_vs_crc")
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    cohorts = {}

    # MIMIC-IV R10.4 (7-feature vector)
    if args.mimic_jsonl.exists():
        print("[R13] MIMIC-IV R10.4 (7-feature)")
        from experiments.round10_method_agnostic import _feature_vector
        cohorts["mimic4_r10_4"] = {}
        for clf in args.classifiers:
            print(f"  fitting {clf}...")
            per_seed = recompute_r10_r11(
                clf, args.mimic_jsonl, _feature_vector,
                n_cal=args.n_cal, n_test=args.n_test)
            cohorts["mimic4_r10_4"][clf] = run_cohort(
                f"mimic4_r10_4_{clf}", per_seed,
                strata_list=["CRITICAL", "HIGH", "MODERATE", "LOW"],
                alphas=ALPHAS)

    # MIMIC-IV R11.1 (4-feature minimal)
    if args.mimic_jsonl.exists():
        print("[R13] MIMIC-IV R11.1 (4-feature minimal)")
        from experiments.round11_method_agnostic_minimal import _feature_vector_minimal
        cohorts["mimic4_r11_1"] = {}
        for clf in args.classifiers:
            print(f"  fitting {clf}...")
            per_seed = recompute_r10_r11(
                clf, args.mimic_jsonl, _feature_vector_minimal,
                n_cal=args.n_cal, n_test=args.n_test)
            cohorts["mimic4_r11_1"][clf] = run_cohort(
                f"mimic4_r11_1_{clf}", per_seed,
                strata_list=["CRITICAL", "HIGH", "MODERATE", "LOW"],
                alphas=ALPHAS)

    # eICU R11.3 Pass A (9-feature) + Pass B (4-feature minimal)
    if args.eicu_jsonl.exists():
        for pass_name in ["pass_a", "pass_b"]:
            print(f"[R13] eICU R11.3 {pass_name}")
            cohort_key = f"eicu_r11_3_{pass_name}"
            cohorts[cohort_key] = {}
            for clf in args.classifiers:
                print(f"  fitting {clf}...")
                per_seed = recompute_eicu(
                    clf, args.eicu_jsonl, pass_name,
                    n_cal=args.n_cal, n_test=args.n_test)
                cohorts[cohort_key][clf] = run_cohort(
                    f"{cohort_key}_{clf}", per_seed,
                    strata_list=["CRITICAL", "HIGH", "MODERATE", "LOW"],
                    alphas=ALPHAS)
    else:
        print(f"[R13] eICU JSONL missing ({args.eicu_jsonl}) — skipping")

    # R12.1 MedAbstain (LLM cache, if raw scores present)
    r12_data = load_r12_1_seed_data()
    if r12_data:
        print(f"[R13] MedAbstain R12.1 (LLM NLL) — {len(r12_data)} seeds")
        cohorts["medabstain_r12_1"] = {
            "gpt_oss_120b": run_cohort(
                "medabstain_r12_1_gpt_oss_120b", r12_data,
                strata_list=["CRITICAL", "HIGH", "MODERATE"],
                alphas={k: v for k, v in ALPHAS.items() if k != "LOW"}),
        }
    else:
        print("[R13] MedAbstain R12.1: raw LLM scores not cached — skipping")

    report = {
        "timestamp": datetime.now().isoformat(),
        "cost_grid": COST_GRID,
        "cohorts": cohorts,
    }

    Path(str(args.out) + ".json").write_text(
        json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")


if __name__ == "__main__":
    main()
