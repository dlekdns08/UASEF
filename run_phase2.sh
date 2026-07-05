#!/usr/bin/env bash
# Phase 2 regeneration audits (sequential — one LMStudio model at a time).
#   1) cross-model verifier: an INDEPENDENT model judges gpt-oss answers
#   2) option-shuffle: gpt-oss regenerates with permuted MCQ options
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
export UASEF_QUERY_TIMEOUT_S="${UASEF_QUERY_TIMEOUT_S:-120}"
N_VERIFIER="${N_VERIFIER:-1500}"
N_SHUFFLE="${N_SHUFFLE:-400}"
VERIFIER_MODEL="${VERIFIER_MODEL:-qwen/qwen3.6-35b-a3b}"

echo "[phase2] cross-model verifier ($VERIFIER_MODEL) on $N_VERIFIER drafts"
VERIFIER_MODEL="$VERIFIER_MODEL" $PY experiments/phase2_cross_verifier.py \
    --drafts data/raw/drafts_phase0_all.jsonl --n "$N_VERIFIER"

echo "[phase2] option-shuffle audit (gpt-oss) on $N_SHUFFLE MedMCQA items"
LMSTUDIO_MODEL="openai/gpt-oss-120b" $PY experiments/phase2_shuffle_audit.py --n "$N_SHUFFLE"

echo "[phase2] done -> results/phase2/*.json"
