#!/usr/bin/env bash
# Reproduce the UASEF diagnostic-framework results (paper: paper/UASEF_PATTERNS_2026.md).
#
# Data-free steps (smoke, core tests, synthetic benchmark, audit CSV + kappa) run
# anywhere. Real-data steps (MIMIC-IV floor, eICU audit) run only if the local
# PhysioNet-credentialed lab caches are present; they are skipped otherwise.
#
# Usage:  bash reproduce.sh            # everything available
#         SKIP_REALDATA=1 bash reproduce.sh   # data-free only
#
# All output is teed to results/repro/reproduce.log.
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
export UASEF_BACKEND_NEVER_SEND_PHI=1
mkdir -p results/repro
LOG=results/repro/reproduce.log
: > "$LOG"

say() { echo "$@" | tee -a "$LOG"; }
run() { say ""; say "### \$ $*"; "$@" 2>&1 | tee -a "$LOG"; say "  -> exit ${PIPESTATUS[0]}"; }

say "==================================================================="
say " UASEF diagnostic reproduction"
say " commit : $(git rev-parse HEAD 2>/dev/null || echo 'n/a') ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
say " python : $($PY --version 2>&1)"
say " date   : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
say " PHI-guard: UASEF_BACKEND_NEVER_SEND_PHI=$UASEF_BACKEND_NEVER_SEND_PHI"
say "==================================================================="

say ""; say "## 1. Import / packaging check"
run $PY -c "import models.audit_detectors as d; print('detectors:', list(d.DETECTORS)); print('functional API:', [f for f in d.__all__ if f.startswith('detect_')])"

say ""; say "## 2. Detector smoke test (python -m, synthetic known answers, no data)"
run $PY -m models.audit_detectors

say ""; say "## 3. Verified-core unit tests"
run $PY -m pytest tests/test_conformal_escalation.py -q

say ""; say "## 4. Synthetic contamination-injection benchmark (detector sens/spec)"
run $PY experiments/round20_detector_benchmark.py

say ""; say "## 5. Literature-audit coding table (CSV) + Cohen's kappa"
run $PY experiments/export_audit_csv.py
run $PY experiments/round26_interrater_kappa.py

if [ "${SKIP_REALDATA:-0}" = "1" ]; then
  say ""; say "## 6-7. Real-data steps SKIPPED (SKIP_REALDATA=1)"
else
  MIMIC_CACHE=data/raw/mimic-iv/mimic4_labvalues_6h_guarded.jsonl
  EICU_CACHE=data/raw/eicu_labvalues_6h.jsonl
  say ""; say "## 6. MIMIC-IV leakage-safe floor (needs $MIMIC_CACHE)"
  if [ -f "$MIMIC_CACHE" ]; then run $PY experiments/round18_leakage_safe_floor.py
  else say "  SKIP: $MIMIC_CACHE not present (PhysioNet-credentialed data)"; fi
  say ""; say "## 7. eICU-CRD independent audit (needs $EICU_CACHE)"
  if [ -f "$EICU_CACHE" ]; then run $PY experiments/round19_eicu_audit.py
  else say "  SKIP: $EICU_CACHE not present (PhysioNet-credentialed data)"; fi
fi

say ""; say "==================================================================="
say " done -> $LOG"
say "==================================================================="
