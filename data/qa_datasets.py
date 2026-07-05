"""
Dataset-agnostic QA loaders for the Phase 0-3 redesign -> list[QAItem].

MedMCQA  : subject-labelled 4-option medical MCQ (21 subjects) — needed for the
           subject-stratified split (Phase 1). HF: openlifescienceai/medmcqa.
PubMedQA : expert-labelled yes/no/maybe biomedical QA. HF: pubmed_qa/pqa_labeled.

These load the *questions* only (with the gold answer stored on the QAItem for
later error-labelling). Draft generation is a separate, backend-dependent step
(models/qa_drafts.py). If HuggingFace is unavailable the loaders raise a clear
error — we never silently fabricate benchmark items (use the gatekeeper's
`--synthetic` mode for a data-free smoke).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.qa_risk_features import QAItem

_LETTERS = ["A", "B", "C", "D"]


def _subject_stratified(items, n, seed):
    """Take up to n items keeping each subject's share (deterministic)."""
    import random
    by_sub = {}
    for it in items:
        by_sub.setdefault(it.subject, []).append(it)
    rng = random.Random(seed)
    for v in by_sub.values():
        rng.shuffle(v)
    total = len(items)
    out = []
    for sub, v in by_sub.items():
        take = max(1, round(n * len(v) / total))
        out.extend(v[:take])
    rng.shuffle(out)
    return out[:n]


def load_medmcqa(n: int = 3000, seed: int = 42, split: str = "train") -> list:
    """MedMCQA -> subject-stratified sample of QAItem (4-option MCQ)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets to load MedMCQA") from e
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    items = []
    for i, r in enumerate(ds):
        cop = r.get("cop")
        if cop is None or cop not in (0, 1, 2, 3):
            continue
        opts = {_LETTERS[j]: r.get(f"op{c}", "") for j, c in enumerate("abcd")}
        items.append(QAItem(
            item_id=f"medmcqa_{i}", dataset="medmcqa",
            question=r.get("question", ""), options=opts,
            gold_answer=_LETTERS[cop],
            subject=(r.get("subject_name") or "unknown"),
        ))
    return _subject_stratified(items, n, seed)


def load_pubmedqa(n: int = 800, seed: int = 42) -> list:
    """PubMedQA (pqa_labeled) -> QAItem (yes/no/maybe)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets to load PubMedQA") from e
    import random
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    items = []
    for i, r in enumerate(ds):
        dec = (r.get("final_decision") or "").strip().lower()
        if dec not in ("yes", "no", "maybe"):
            continue
        items.append(QAItem(
            item_id=f"pubmedqa_{i}", dataset="pubmedqa",
            question=r.get("question", ""), options={},
            gold_answer=dec, subject="pubmedqa",
        ))
    random.Random(seed).shuffle(items)
    return items[:n]
