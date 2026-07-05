# Reproducing the UASEF diagnostic-framework results

This document ties the paper (`paper/UASEF_PATTERNS_2026.md`) to runnable code and
**captured execution logs**. One command reproduces everything:

```bash
bash reproduce.sh          # data-free + real-data (if PhysioNet caches present)
SKIP_REALDATA=1 bash reproduce.sh   # data-free only (smoke, tests, benchmark, audit)
```

All output is teed to [`results/repro/reproduce.log`](results/repro/reproduce.log).
The run below was captured at:

| | |
|---|---|
| **commit** | `d7a61081b832c40fc5595c08950618de90b7fb9b` (main) |
| **python** | 3.14.3 (`.venv`) |
| **platform** | darwin (Mac Studio M3 Ultra) |
| **PHI guard** | `UASEF_BACKEND_NEVER_SEND_PHI=1`; external egress 0 bytes |

Data-free steps (1–5) run anywhere after `pip install -e .`. Real-data steps
(6–7) require the PhysioNet-credentialed MIMIC-IV / eICU-CRD lab caches under
`data/raw/` (never redistributed); they auto-skip if absent.

---

## Install

```bash
uv pip install -e .        # or: pip install -e .
```

Only `numpy` / `scipy` are needed for the detectors, core, and audit; the
smoke test and unit tests need no dataset and no network.

## 1. Packaging / API check

The five detectors are importable as classes **and** as plain functions:

```python
from models.audit_detectors import (
    OrientationDetector, EscalateAllDetector, TemporalLeakageDetector,
    InformativeMissingnessDetector, DefinitionalLeakageDetector,   # classes
    detect_orientation, detect_escalate_all, detect_temporal_leakage,
    detect_informative_missingness, detect_definitional_leakage,   # functions
    run_all_detectors, DETECTORS, AuditFlag,
)
```

Captured:
```
detectors: ['orientation', 'escalate_all', 'temporal_leakage', 'informative_missingness', 'definitional_leakage']
functional API: ['detect_orientation', 'detect_escalate_all', 'detect_temporal_leakage', 'detect_informative_missingness', 'detect_definitional_leakage']
```

## 2. Detector smoke test (`python -m`, no data)

```bash
python -m models.audit_detectors
```
Runs each detector on a clean case (must NOT flag) and a contaminated case (must
flag), synthetic and deterministic (seed 0). Captured:

```
case                             expect  flagged     stat  result
------------------------------------------------------------------------
orientation/clean                False   False      0.800  PASS
orientation/inverted             True    True       0.200  PASS
escalate_all/clean               False   False      0.300  PASS
escalate_all/vacuous             True    True       0.990  PASS
temporal/clean                   False   False      0.000  PASS
temporal/leak                    True    True       1.000  PASS
missingness/value-driven         False   False      0.067  PASS
missingness/ordering-driven      True    True       0.967  PASS
definitional/clean               False   False      0.529  PASS
definitional/leak                True    True       1.000  PASS
------------------------------------------------------------------------
ALL PASS (10/10)                                        exit 0
```

## 3. Verified-core unit tests

```bash
python -m pytest tests/test_conformal_escalation.py -q
```
Captured: `10 passed, 3 warnings` (the warnings are the orientation guard firing
on the inverted-sign regression cases — expected). exit 0.

## 4. Synthetic contamination-injection benchmark (§3)

```bash
python experiments/round20_detector_benchmark.py       # -> results/round20/
```
Captured detector operating points on data with known answers:
```
orientation sens=1.00 spec=1.00
temporal FP@0=0.0 min_detect=0.05
missingness correct=True
definitional FP@0=False
```

## 5. Literature-audit coding table + Cohen's κ (§6)

```bash
python experiments/export_audit_csv.py          # -> results/lit_audit/codings.csv
python experiments/round26_interrater_kappa.py  # -> results/lit_audit/kappa.{json,md}
```
Captured:
```
codings.csv — 29 papers x 4 items; cell agreement 92/116 (79%)
  columns: idx, ref, paper, venue_id, url, modality, clinical_cp, A_D0..A_D3, B_D0..B_D3, agree_D0..agree_D3
kappa: D0=0.293  D1=0.835  D2=0.782  D3=0.587   POOLED=0.714 (substantial)
```
(Audited-triad {D1,D2,D3} pooled κ = 0.785; D0 is an auxiliary field. See §6 and
Appendix D. Full per-paper codings: `results/lit_audit/codings.{json,csv}` and
`codings_coderB.json`.)

## 6. MIMIC-IV leakage-safe floor (§7) — real data

```bash
export UASEF_BACKEND_NEVER_SEND_PHI=1
python experiments/round18_leakage_safe_floor.py       # -> results/round18/
```
Captured (3750/14000 admissions, 27% coverage, icu_intime-guarded window):
```
  CRITICAL: FULL AUROC=0.797 (miss 0.102, over_esc 0.528) | VALUE=0.795 | FLAG=0.791
  HIGH:     FULL AUROC=0.688 (miss 0.160, over_esc 0.582) | VALUE=0.685 | FLAG=0.670
  MODERATE: FULL AUROC=0.680 (miss 0.296, over_esc 0.551) | VALUE=0.671 | FLAG=0.667
  LOW:      FULL AUROC=0.706 (miss 0.247, over_esc 0.487) | VALUE=0.703 | FLAG=0.705
  ★ decision-time ceiling (CRITICAL): AUROC=0.797, over_esc=0.528
```
value ≈ flag on CRITICAL (0.795 vs 0.791) ⇒ the gate is ordering-driven; matches
the paper's §7 floor (AUROC ~0.80, no deployable gate).

## 7. eICU-CRD independent audit (§5) — real data

```bash
python experiments/round19_eicu_audit.py               # -> results/round19/
```
Captured (147770/200859 stays, 74% coverage, mortality prevalence 0.090):
```
  CRITICAL: FULL AUROC=0.730 (miss 0.231, over_esc 0.461) | VALUE=0.727 | FLAG=0.610
  HIGH:     FULL AUROC=0.687 | VALUE=0.685 | FLAG=0.618
  MODERATE: FULL AUROC=0.632 | VALUE=0.646 | FLAG=0.588
  LOW:      FULL AUROC=0.670 | VALUE=0.664 | FLAG=0.597
  ★ informative-missingness on eICU CRITICAL: FULL=0.730 FLAG=0.610 (gap +0.120)
    -> does NOT replicate (values add signal) — correct specificity at high coverage
```
The missingness trap flags on 27%-coverage MIMIC but **correctly stays silent**
on 74%-coverage eICU — the detector is a calibrated instrument, not an
alarm-everything heuristic (§5.5 threshold-transfer confirms the fixed 0.85 line
gives the regime-correct verdict on both institutions).

---

## What maps to what

| Paper | Command | Output |
|---|---|---|
| §3 detectors + measured sens/spec | `python -m models.audit_detectors`; `round20_detector_benchmark.py` | smoke 10/10; `results/round20/` |
| §3 threshold null-calibration | `round28_missingness_null_calibration.py` | `results/round28/` |
| §5 eICU generalization/specificity | `round19_eicu_audit.py` | `results/round19/` |
| §5.5 cross-institution transfer | `round27_threshold_transfer.py` | `results/round27/` |
| §6 audit + κ (CSV) | `export_audit_csv.py`; `round26_interrater_kappa.py` | `results/lit_audit/codings.csv`, `kappa.json` |
| §6 corpus references (Appendix D) | `build_audit_references.py` | `results/lit_audit/audit_references.md` |
| §7 leakage-safe floor | `round18_leakage_safe_floor.py` | `results/round18/` |
| §7 information ceiling + robustness | `round22_mi_boundary.py`; `round25_mi_robustness.py` | `results/round22/`, `results/round25/` |

**Compliance.** MIMIC-IV / eICU-CRD are PhysioNet Credentialed Health Data
(License v1.5.0); raw data and lab caches are `.gitignore`d and never
redistributed. All computation is local; external-API cost $0; PHI egress 0 bytes.
