#!/usr/bin/env bash
# Phase 0 gatekeeper on REAL local-LLM drafts (LMStudio gpt-oss-120b, $0).
# Resumable: re-running continues from the cached JSONL. Scale via env:
#   N_MEDMCQA=200 N_PUBMEDQA=80 bash run_phase0.sh
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
export UASEF_QUERY_TIMEOUT_S="${UASEF_QUERY_TIMEOUT_S:-120}"
export LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-openai/gpt-oss-120b}"
N_MEDMCQA="${N_MEDMCQA:-200}"
N_PUBMEDQA="${N_PUBMEDQA:-80}"
K="${K:-5}"
MED=data/raw/drafts_medmcqa.jsonl
PUB=data/raw/drafts_pubmedqa.jsonl

echo "[phase0] model=$LMSTUDIO_MODEL  MedMCQA=$N_MEDMCQA  PubMedQA=$N_PUBMEDQA  k=$K"
$PY models/qa_drafts.py --dataset medmcqa  --n "$N_MEDMCQA"  --k "$K" --out "$MED"
$PY models/qa_drafts.py --dataset pubmedqa --n "$N_PUBMEDQA" --k "$K" --out "$PUB"

echo "[phase0] running gatekeeper on combined drafts"
cat "$MED" "$PUB" > data/raw/drafts_phase0_all.jsonl
$PY experiments/phase0_gatekeeper.py --drafts data/raw/drafts_phase0_all.jsonl
echo "[phase0] done -> results/phase0/phase0_gatekeeper.json"
