#!/usr/bin/env bash
#
# UASEF — Multi-seed bootstrap wrapper around run_full_evaluation.sh
# ════════════════════════════════════════════════════════════════════════════
#
# Why this exists:
#   The main paper reports single-seed (seed=42) results in Tables 1/4. ML4H
#   reviewers typically request bootstrap CIs over multiple seeds to verify
#   that gains are not seed-specific. This script runs the full evaluation
#   once per seed and aggregates the per-seed results into a single
#   results/run_<ts>_aggregate/aggregate_seeds.{json,md}.
#
# Usage:
#   bash run_multiseed_evaluation.sh                          # default 5 seeds (42-46)
#   SEEDS="42 43 44 45 46 47 48 49 50 51" bash run_multiseed_evaluation.sh
#   BACKENDS="openai" SEEDS="42 43 44" bash run_multiseed_evaluation.sh
#
# Cost estimate (per seed): ~$25 OpenAI + ~10 min LMStudio
#   5 seeds = ~$125 + ~50 min
#   10 seeds = ~$250 + ~100 min
#
# Honest framing:
#   - This wrapper is the *infrastructure* for the multi-seed claim in §6.6
#     and §8 L9. It is *not* run as part of the paper submission; the camera-
#     ready will report the bootstrap CIs from a 5+ seed run produced by this
#     script.
#   - For the current submission, the user runs single-seed
#     `run_full_evaluation.sh` and reports those numbers; the SEEDS=
#     argument in the main script is reserved for the multi-seed mode.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-42 43 44 45 46}"
BACKENDS="${BACKENDS:-openai lmstudio}"

AGG_TS="$(date +%Y%m%d-%H%M%S)"
AGG_DIR="${ROOT}/results/run_${AGG_TS}_aggregate"
mkdir -p "$AGG_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  UASEF — Multi-seed bootstrap"
echo "════════════════════════════════════════════════════════════════════"
echo "  Seeds      : $SEEDS"
echo "  Backends   : $BACKENDS"
echo "  Aggregate  : $AGG_DIR"
echo ""

PER_SEED_DIRS=()
for SEED in $SEEDS; do
    echo ""
    echo "── seed=$SEED ─────────────────────────────────────────────────"
    SEED="$SEED" BACKENDS="$BACKENDS" bash "${ROOT}/run_full_evaluation.sh"
    # Locate the most recent run directory.
    LATEST="$(ls -td results/run_2*/ 2>/dev/null | head -n1 | sed 's:/$::')"
    PER_SEED_DIRS+=("$LATEST")
    echo "  ✓ seed=$SEED → $LATEST"
done

# Aggregate via python helper.
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"
"$PYTHON" experiments/aggregate_multiseed.py \
    --runs "${PER_SEED_DIRS[@]}" \
    --output "$AGG_DIR" \
    --backends $BACKENDS

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ Multi-seed aggregate: $AGG_DIR/aggregate_seeds.{json,md}"
echo "════════════════════════════════════════════════════════════════════"
