"""
Primary + key secondary result tables from the consolidated ledger (analysis_plan §1-2).

Reads the mode-aware long tables (03 judgments, 04 self-answers, 02 answerer outputs) and
emits, per (answerer_model, answerer_mode, verifier_model, verifier_mode, dataset) CROSS
cell, with item-grouped paired bootstrap CIs (stats.py) — NEVER row-level:

  06_main_auroc_lift.csv        AUROC_C / AUROC_V / lift(V-C) + CI + bootstrap p   [Primary 1]
  07_z_gating_results.csv       lift on Z=1 vs Z=0 subsets                          [Primary 2]
  08_delta_lift.csv             Δ = acc(verifier self) - acc(answerer) vs lift
  09_reasoning_mode_ablation.csv T vs N (and low vs high) lift diff, same answerer  [Primary 3 / F5]

BH-FDR (F1 = all lift tests; F3 = T/N pairs; F5 = gpt-oss effort) applied across cells,
mc+pm pooled in one family (analysis_plan §2). Complete-case on valid V (parser policy).

Run:  python analysis/make_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import sym_auroc, paired_bootstrap_auroc_diff, bh_fdr
from analysis.manifest import ALIAS

CONS = ROOT / "results" / "consolidated"


def _alias(m):
    return ALIAS.get(m, m)


def load():
    jud = pd.read_csv(CONS / "03_verifier_judgments.csv")
    slf = pd.read_csv(CONS / "04_verifier_self_answers.csv")
    jud = jud[jud["verification_type"] == "cross"].copy()
    jud = jud[jud["verifier_risk_V"].notna() & jud["risk_C"].notna()].copy()
    # Z per (verifier_model, verifier_mode, item) from self-answers
    z = slf[["verifier_model", "verifier_mode", "item_id", "verifier_self_correct_Z"]].dropna()
    z = z.drop_duplicates(["verifier_model", "verifier_mode", "item_id"])
    return jud, z


KEYS = ["answerer_model", "answerer_mode", "verifier_model", "verifier_mode", "dataset"]


def cell_iter(jud):
    for key, g in jud.groupby(KEYS):
        if len(g) >= 40 and 3 <= g["Y_error"].sum() <= len(g) - 3:
            yield dict(zip(KEYS, key)), g


def main06(jud):
    rows = []
    for k, g in cell_iter(jud):
        y = g["Y_error"].to_numpy(int); V = g["verifier_risk_V"].to_numpy(float); C = g["risk_C"].to_numpy(float)
        bs = paired_bootstrap_auroc_diff(V, C, y, g["item_id"].to_numpy())
        rows.append({**k, "n": len(g), "err_rate": round(y.mean(), 4),
                     "AUROC_C": round(sym_auroc(C, y), 4), "AUROC_V": round(sym_auroc(V, y), 4),
                     "AUPRC_V": round(_auprc(V, y), 4), "AUPRC_C": round(_auprc(C, y), 4),
                     "lift": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                     "p_boot": bs["p_boot"]})
    df = pd.DataFrame(rows)
    df["p_fdr_F1"] = bh_fdr(df["p_boot"].tolist())
    df["sig"] = df["ci_low"] > 0
    return df.sort_values("lift", ascending=False)


def _auprc(score, y):
    from sklearn.metrics import average_precision_score
    a = average_precision_score(y, score)
    return max(a, average_precision_score(y, -score))


def main07(jud, z):
    """Z-gating: does verifier lift concentrate where the verifier itself is correct?"""
    m = jud.merge(z, on=["verifier_model", "verifier_mode", "item_id"], how="inner")
    rows = []
    for key, g in m.groupby(KEYS):
        k = dict(zip(KEYS, key))
        for zname, zval in [("Z1", 1), ("Z0", 0)]:
            sub = g[g["verifier_self_correct_Z"] == zval]
            y = sub["Y_error"].to_numpy(int)
            if len(sub) >= 40 and 3 <= y.sum() <= len(sub) - 3:
                V = sub["verifier_risk_V"].to_numpy(float); C = sub["risk_C"].to_numpy(float)
                bs = paired_bootstrap_auroc_diff(V, C, y, sub["item_id"].to_numpy())
                rows.append({**k, "z_subset": zname, "n": len(sub),
                             "AUROC_V": round(sym_auroc(V, y), 4), "AUROC_C": round(sym_auroc(C, y), 4),
                             "lift": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
    return pd.DataFrame(rows)


def main08(jud, z):
    """Δ = acc(verifier self-answer) - acc(answerer) vs lift, within dataset."""
    ans = pd.read_csv(CONS / "02_answerer_outputs.csv")
    ans = ans[ans["split"] == "matrix"]
    aacc = ans.groupby(["answerer_model", "answerer_mode", "dataset"])["answer_correct"].mean().rename("answerer_acc")
    zacc = z.copy()
    zacc["dataset"] = zacc["item_id"].apply(lambda i: "pubmedqa" if str(i).startswith("pubmedqa") else "medmcqa")
    vacc = zacc.groupby(["verifier_model", "verifier_mode", "dataset"])["verifier_self_correct_Z"].mean().rename("verifier_acc")
    m06 = main06(jud)
    m06 = m06.merge(aacc, on=["answerer_model", "answerer_mode", "dataset"], how="left")
    m06 = m06.merge(vacc, on=["verifier_model", "verifier_mode", "dataset"], how="left")
    m06["delta"] = (m06["verifier_acc"] - m06["answerer_acc"]).round(4)
    return m06[KEYS + ["delta", "verifier_acc", "answerer_acc", "lift", "ci_low", "ci_high", "sig"]]


def main09(m06):
    """reasoning ablation: pair cells that differ ONLY in verifier_mode, same answerer+dataset+verifier_model."""
    rows = []
    grp = m06.groupby(["answerer_model", "answerer_mode", "verifier_model", "dataset"])
    for (am, amode, vm, ds), g in grp:
        modes = dict(zip(g["verifier_mode"], zip(g["lift"], g["AUROC_V"])))
        # T/N pair (reasoning toggle) or low/high (effort, F5)
        for hi, lo, fam in [("T", "N", "F3_TN"), ("high", "low", "F5_effort")]:
            if hi in modes and lo in modes:
                rows.append({"answerer_model": _alias(am), "answerer_mode": amode,
                             "verifier_model": _alias(vm), "dataset": ds, "family": fam,
                             f"lift_{hi}": modes[hi][0], f"lift_{lo}": modes[lo][0],
                             "lift_diff": round(modes[hi][0] - modes[lo][0], 4)})
    return pd.DataFrame(rows)


def main():
    jud, z = load()
    print(f"[results] cross judgments {len(jud)} rows | Z rows {len(z)}")
    m06 = main06(jud)
    m07 = main07(jud, z)
    m08 = main08(jud, z)
    m09 = main09(m06)
    for name, df in [("06_main_auroc_lift", m06), ("07_z_gating_results", m07),
                     ("08_delta_lift", m08), ("09_reasoning_mode_ablation", m09)]:
        df.to_csv(CONS / f"{name}.csv", index=False)
        print(f"  {name}.csv: {len(df)} rows")
    # headline summaries
    print("\n=== 06 lift (cross 셀, item-paired bootstrap) — 상위/하위 ===")
    disp = m06.copy()
    for c in ["answerer_model", "verifier_model"]:
        disp[c] = disp[c].map(_alias)
    print(disp[["answerer_model", "answerer_mode", "verifier_model", "verifier_mode",
                "dataset", "lift", "ci_low", "ci_high", "sig"]].to_string(index=False))
    print(f"\n유의(CI>0) 셀: {m06['sig'].sum()}/{len(m06)}")
    print("\n=== 09 T/N & effort ablation ===")
    print(m09.to_string(index=False))


if __name__ == "__main__":
    main()
