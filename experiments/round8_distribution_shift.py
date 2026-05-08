"""
Distribution-Shift Sanity (Round 8 P2-3, Supplementary §G)
══════════════════════════════════════════════════════════════════════════════

Holds the v2 stratified CRC calibration on one specialty (e.g., 'emergency_medicine')
and re-evaluates per-stratum miss rates when the test distribution comes from a
DIFFERENT specialty. Quantifies the coverage violation magnitude — useful as a
deployment-audit signal: any institution rolling out v2 to a new specialty
should re-calibrate or expect the reported violation.

Synthetic-only (no LLM calls): scores are simulated from specialty-conditioned
distributions. The point is to exercise the *algorithm* under shift, not to
make an empirical claim. Real shift evaluation requires a multi-specialty
labeled benchmark which we do not have.

Usage
-----
    python experiments/round8_distribution_shift.py \
        --calib-specialty emergency_medicine \
        --test-specialties internal_medicine pediatrics neurology general_practice \
        --n-cal 500 --n-test 200 --seed 42 \
        --out results/round8/distribution_shift.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.stratified_crc import StratifiedConformalRiskControl


# Synthetic specialty-conditioned score distributions.
# Each specialty has (positive_mean, positive_std, negative_mean, negative_std).
SPECIALTY_DIST = {
    "emergency_medicine": (2.5, 1.0, 0.0, 1.0),       # high signal
    "internal_medicine":  (2.0, 1.0, 0.0, 1.0),
    "pediatrics":         (2.2, 1.2, 0.1, 1.0),
    "neurology":          (1.8, 1.3, -0.1, 1.1),       # rare-disease drift
    "general_practice":   (1.5, 1.0, 0.0, 0.9),
}


# Stratum mapping per specialty (mirrors data/loader.SPECIALTY_RISK_MAP).
SPECIALTY_TO_STRATUM = {
    "emergency_medicine": "CRITICAL",
    "internal_medicine":  "MODERATE",
    "pediatrics":         "HIGH",
    "neurology":          "HIGH",
    "general_practice":   "LOW",
}


def gen_data(specialty: str, n: int, prevalence: float, seed: int):
    rng = random.Random(seed)
    pos_mu, pos_sd, neg_mu, neg_sd = SPECIALTY_DIST[specialty]
    scores, labels, strata = [], [], []
    s = SPECIALTY_TO_STRATUM[specialty]
    for _ in range(n):
        is_pos = rng.random() < prevalence
        if is_pos:
            scores.append(rng.gauss(pos_mu, pos_sd))
            labels.append(True)
        else:
            scores.append(rng.gauss(neg_mu, neg_sd))
            labels.append(False)
        strata.append(s)
    return scores, labels, strata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-specialty", default="emergency_medicine",
                    choices=list(SPECIALTY_DIST.keys()))
    ap.add_argument("--test-specialties", nargs="+",
                    default=["internal_medicine", "pediatrics", "neurology", "general_practice"])
    ap.add_argument("--n-cal", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prevalence", type=float, default=0.3)
    ap.add_argument("--alphas", type=float, nargs=4, default=[0.05, 0.10, 0.15, 0.20],
                    metavar=("CRITICAL", "HIGH", "MODERATE", "LOW"))
    ap.add_argument("--out", type=Path, default=Path("results/round8/distribution_shift.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    alphas = dict(zip(["CRITICAL", "HIGH", "MODERATE", "LOW"], args.alphas))

    # Calibrate on calib_specialty.
    cs, cl, cstrata = gen_data(args.calib_specialty, args.n_cal, args.prevalence, args.seed)
    crc = StratifiedConformalRiskControl(alphas=alphas)
    crc.fit(cs, cl, cstrata)

    # Evaluate on shifted specialties.
    rows = []
    for ts in args.test_specialties:
        scores, labels, strata = gen_data(ts, args.n_test, args.prevalence, args.seed + 1)
        s = SPECIALTY_TO_STRATUM[ts]
        # Use the CRC threshold for whatever stratum we calibrated on.
        # Under shift, we deliberately use the *calibration-specialty* stratum.
        cal_s = SPECIALTY_TO_STRATUM[args.calib_specialty]
        try:
            lam = crc.threshold_for(cal_s)
        except Exception:
            lam = 0.0
        misses = sum(1 for sc, l in zip(scores, labels) if l and sc <= lam)
        n_pos = sum(labels) or 1
        miss_rate = misses / n_pos
        target = alphas[cal_s]
        rows.append({
            "test_specialty": ts,
            "test_stratum": s,
            "calib_specialty": args.calib_specialty,
            "calib_stratum": cal_s,
            "lambda": float(lam),
            "n_test": args.n_test,
            "n_pos": n_pos,
            "miss_count": misses,
            "miss_rate": round(miss_rate, 4),
            "target_alpha": target,
            "violation": round(max(0.0, miss_rate - target), 4),
            "ratio_over_target": round(miss_rate / max(1e-9, target), 3),
        })

    payload = {
        "timestamp": datetime.now().isoformat(),
        "calib_specialty": args.calib_specialty,
        "alphas": alphas,
        "n_cal": args.n_cal, "n_test": args.n_test,
        "seed": args.seed, "prevalence": args.prevalence,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = ["# Round 8 — Distribution Shift Sanity (Supplementary §G)", "",
          f"calibrated on {args.calib_specialty} ({args.n_cal} samples, target α={alphas[SPECIALTY_TO_STRATUM[args.calib_specialty]]})",
          "",
          "| test specialty | test stratum | miss rate | target α | violation | ratio |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(
            f"| {r['test_specialty']} | {r['test_stratum']} | "
            f"{r['miss_rate']} | {r['target_alpha']} | {r['violation']} | {r['ratio_over_target']}× |"
        )
    args.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n✅ saved: {args.out} (+ .md)")


if __name__ == "__main__":
    main()
