"""
Round 11 R11.4 — MOD/LOW failure as information-theoretic data limit.

MIMIC-IV R10 의 100% miss on MODERATE / LOW 가 framework defect 가 아니라
admission-time features 의 *information-theoretic upper bound* 임을 증명.

Per-stratum mutual information:
   I(X_t0; Y | sigma = s),    s ∈ {CRITICAL, HIGH, MODERATE, LOW}

CRITICAL/HIGH 의 MI 와 MOD/LOW 의 MI 의 order-of-magnitude gap 정량화.

Pre-registered verdict:
   - I_MOD / I_CRIT < 0.1   →  data-limit (framework 무죄)
   - I_MOD / I_CRIT > 0.5   →  framework defect (다른 분류기로 해결 가능)
   - between                →  partial limit

산출: results/round11/r11_4_modlow_mi.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from data.loader import _load_mimic4_jsonl

DEFAULT_JSONL = ROOT / "data" / "raw" / "mimic-iv" / "mimic4_cases_v10.jsonl"
STRATA = ["CRITICAL", "HIGH", "MODERATE", "LOW"]


def _all_features(case) -> dict:
    """Decision-time + leakage-suspect features (모두 dump 하여 MI 계산)."""
    meta = case.meta_info or ""
    age_bucket = {"unknown": 0, "<18": 1, "18-34": 2, "35-49": 3,
                  "50-64": 4, "65-79": 5, "80+": 6}
    age_idx = 0
    for tok in meta.split():
        if tok.startswith("age_bucket="):
            age_idx = age_bucket.get(tok.split("=", 1)[1].strip(), 0)
    adm_emerg = 1 if ("EMERG" in meta or "URG" in meta) else 0
    spec_map = {"cardiology": 1, "neurology": 2, "internal_medicine": 3,
                "surgery": 4, "obstetrics": 5, "psychiatry": 6,
                "pediatrics": 7, "cardiothoracic_surgery": 8}
    spec_idx = spec_map.get(case.specialty or "", 0)
    n_labs = 0
    for tok in meta.split():
        if tok.startswith("labs="):
            n_labs = len(tok.split("=", 1)[1].split(","))
    return {
        "age_bucket": age_idx,
        "adm_emerg": adm_emerg,
        "spec_idx": spec_idx,
        "n_labs": n_labs,
        # R10.7 leakage suspects (참고용 — MI 계산에는 어떤 features 가
        # outcome 에 대한 information 을 가지는지 확인)
        "charlson": int(getattr(case, "_charlson_index", 0)),
        "n_vital_flags": int(getattr(case, "_n_vital_flags", 0)),
        "spec_baseline_rate": float(
            getattr(case, "_specialty_baseline_rate", 0.0)),
    }


def _ksg_mi_continuous_discrete(x: np.ndarray, y: np.ndarray,
                                  k: int = 3) -> float:
    """
    Mutual information between continuous X and discrete Y via the
    Ross (2014) estimator (a KSG variant).

    Falls back to sklearn's mutual_info_classif which implements the
    Kraskov-Stögbauer-Grassberger family for mixed types.
    """
    from sklearn.feature_selection import mutual_info_classif
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return float(mutual_info_classif(
        x, y.astype(int), discrete_features="auto",
        n_neighbors=k, random_state=42, copy=True).sum())


def per_stratum_mi(cases) -> dict:
    """For each stratum, compute MI between minimal features and outcome Y."""
    out = {}
    for s in STRATA:
        sub = [c for c in cases if (c.scenario_type or "").upper() == s]
        if len(sub) < 50:
            out[s] = {"n": len(sub), "note": "insufficient samples"}
            continue
        feats = [_all_features(c) for c in sub]
        y = np.array([1 if c.expected_escalate else 0 for c in sub], dtype=int)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        cell = {"n": len(sub), "n_pos": n_pos, "n_neg": n_neg}
        if n_pos < 5 or n_neg < 5:
            cell["note"] = "degenerate y distribution"
            out[s] = cell
            continue

        # Minimal 4-feature MI (R11.1 의 features)
        X_min = np.array([[f["age_bucket"], f["adm_emerg"],
                            f["spec_idx"], f["n_labs"]] for f in feats],
                          dtype=float)
        mi_min = _ksg_mi_continuous_discrete(X_min, y)

        # Full 7-feature MI (R10.4 의 features incl. leakage suspects)
        X_full = np.array([[f["age_bucket"], f["adm_emerg"], f["spec_idx"],
                             f["n_labs"], f["charlson"], f["n_vital_flags"],
                             f["spec_baseline_rate"]] for f in feats],
                           dtype=float)
        mi_full = _ksg_mi_continuous_discrete(X_full, y)

        # Per-feature MI (1-D)
        per_feat = {}
        for i, name in enumerate(["age_bucket", "adm_emerg", "spec_idx",
                                    "n_labs", "charlson", "n_vital_flags",
                                    "spec_baseline_rate"]):
            per_feat[name] = _ksg_mi_continuous_discrete(X_full[:, i], y)

        # Outcome entropy (upper bound for any classifier)
        p = n_pos / len(y)
        h_y = -(p * math.log(p) + (1 - p) * math.log(1 - p)) if 0 < p < 1 else 0.0

        cell["H_Y"] = h_y
        cell["MI_minimal"] = mi_min
        cell["MI_full"] = mi_full
        cell["MI_minimal_ratio"] = mi_min / h_y if h_y > 0 else None
        cell["MI_full_ratio"] = mi_full / h_y if h_y > 0 else None
        cell["per_feature_MI"] = per_feat
        cell["leakage_gain"] = (mi_full - mi_min)  # leakage features 가 추가하는 information
        out[s] = cell
    return out


def verdict(stratum_mi: dict) -> dict:
    """Pre-registered verdict on MOD/LOW failure."""
    crit = stratum_mi.get("CRITICAL", {})
    high = stratum_mi.get("HIGH", {})
    mod = stratum_mi.get("MODERATE", {})
    low = stratum_mi.get("LOW", {})

    crit_mi = crit.get("MI_minimal")
    mod_mi = mod.get("MI_minimal")
    low_mi = low.get("MI_minimal")

    if crit_mi is None or mod_mi is None or low_mi is None:
        return {"verdict": "insufficient data", "reason": "missing MI cells"}

    ratio_mod = mod_mi / crit_mi if crit_mi > 1e-6 else float("inf")
    ratio_low = low_mi / crit_mi if crit_mi > 1e-6 else float("inf")

    if ratio_mod < 0.1 and ratio_low < 0.1:
        v = "DATA_LIMIT_CONFIRMED"
        msg = (f"I(X;Y|MOD)/I(X;Y|CRIT) = {ratio_mod:.3f}, "
               f"I(X;Y|LOW)/I(X;Y|CRIT) = {ratio_low:.3f}. "
               "MOD/LOW failure is an information-theoretic data limit, "
               "not a framework defect.")
    elif ratio_mod > 0.5 and ratio_low > 0.5:
        v = "FRAMEWORK_DEFECT"
        msg = (f"MOD/LOW carry comparable MI to CRIT ({ratio_mod:.3f}, "
               f"{ratio_low:.3f}); failure cannot be blamed on data — "
               "investigate classifier / threshold logic.")
    else:
        v = "PARTIAL_DATA_LIMIT"
        msg = (f"MI ratios MOD={ratio_mod:.3f}, LOW={ratio_low:.3f} — "
               "moderate gap. Data limit dominant but not absolute.")

    return {
        "verdict": v, "message": msg,
        "ratio_MOD_to_CRIT": ratio_mod,
        "ratio_LOW_to_CRIT": ratio_low,
    }


def write_md(report: dict, out_md: Path):
    lines = ["# R11.4 — MOD/LOW Information-Theoretic Analysis\n"]
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- n_cases: {report['n_cases']}")
    lines.append(f"- JSONL: `{report['jsonl_path']}`\n")
    lines.append("## Per-stratum mutual information (nats)\n")
    lines.append("| Stratum | n | H(Y) | MI (minimal 4-feature) | MI (full 7-feature) | MI/H ratio (min) | Leakage gain |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in STRATA:
        cell = report["per_stratum"].get(s, {})
        if "MI_minimal" not in cell:
            lines.append(f"| {s} | {cell.get('n', 0)} | — | — | — | — | (skipped) |")
            continue
        lines.append(
            f"| {s} | {cell['n']} | {cell['H_Y']:.4f} | "
            f"{cell['MI_minimal']:.4f} | {cell['MI_full']:.4f} | "
            f"{cell['MI_minimal_ratio']:.4f} | "
            f"{cell['leakage_gain']:.4f} |"
        )

    lines.append("\n## Per-feature MI on minimal stratum\n")
    lines.append("| Feature | CRITICAL | HIGH | MODERATE | LOW |")
    lines.append("| --- | --- | --- | --- | --- |")
    feature_names = ["age_bucket", "adm_emerg", "spec_idx", "n_labs",
                     "charlson", "n_vital_flags", "spec_baseline_rate"]
    for fn in feature_names:
        row = [fn]
        for s in STRATA:
            v = report["per_stratum"].get(s, {}).get("per_feature_MI", {}).get(fn)
            row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Verdict\n")
    v = report["verdict"]
    lines.append(f"**{v['verdict']}** — {v['message']}\n")

    lines.append("\n## Paper integration\n")
    lines.append("§5.7.1 of UASEF_FINAL.md should be added with the above")
    lines.append("verdict text. If `DATA_LIMIT_CONFIRMED`, the existing §5.7-5.8")
    lines.append("\"fundamental limit\" claim is formally backed by MI evidence.")
    out_md.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--n", type=int, default=14000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "round11" / "r11_4_modlow_mi")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"JSONL missing: {args.jsonl}")

    print(f"[R11.4] loading {args.jsonl}")
    cases = _load_mimic4_jsonl(args.jsonl, n=args.n, seed=42)
    # 누락된 leakage-suspect 속성을 raw JSONL 에서 부착
    from experiments.round10_method_agnostic import _attach_r10_features
    _attach_r10_features(cases, args.jsonl)
    print(f"[R11.4] loaded {len(cases)} cases — computing per-stratum MI")

    per_st = per_stratum_mi(cases)
    v = verdict(per_st)

    report = {
        "timestamp": datetime.now().isoformat(),
        "jsonl_path": str(args.jsonl),
        "n_cases": len(cases),
        "per_stratum": per_st,
        "verdict": v,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2, default=str))
    write_md(report, Path(str(args.out) + ".md"))
    print(f"\n✅ {args.out}.{{json,md}}")
    print(f"  Verdict: {v['verdict']}")
    print(f"  {v['message']}")


if __name__ == "__main__":
    main()
