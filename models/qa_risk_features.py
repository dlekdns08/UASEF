"""
Risk features for pre-release LLM-QA escalation (Phase 0-3 redesign).

Two invariants enforced here:
  * NO definitional leakage — the gold answer is used ONLY to compute the error
    LABEL (`error_label`), never as a feature. `extract_features` never sees it.
  * orientation — every feature is oriented so **higher = riskier / more likely
    an error**, matching models.label_conditional_conformal (release iff
    risk < tau).

A `DraftRecord` is the cached output of running the LLM on one question
(decision draft + k temperature samples + optional logprobs / verbalized
confidence). Feature extraction is a pure function of the draft, so it can be
unit-tested and recomputed without re-querying the model.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QAItem:
    """One benchmark question (dataset-agnostic)."""
    item_id: str
    dataset: str                 # "medmcqa" | "pubmedqa"
    question: str
    options: dict                # {"A": "...", ...} for MCQ; {} for yes/no/maybe
    gold_answer: str             # label ONLY — never a feature
    subject: str = "unknown"     # MedMCQA subject_name (for stratified split)


@dataclass
class DraftRecord:
    """Hidden LLM output for one QAItem (never shown to the user)."""
    item_id: str
    dataset: str
    subject: str
    decision_answer: str                       # greedy / temperature-0 answer
    samples: list = field(default_factory=list)  # k temperature samples (answers)
    token_logprobs: list | None = None         # per-token logprobs of decision draft
    verbalized_confidence: float | None = None  # parsed 0..1 (from "0-100")
    reasoning_text: str = ""                    # decision-draft rationale (for hedging)
    gold_answer: str = ""                       # label ONLY


# hedging / uncertainty phrases (source-tagged, mirrors rtc_ede convention)
HEDGING_PHRASES = [
    "maybe", "perhaps", "possibly", "might be", "could be", "not sure",
    "uncertain", "unclear", "i think", "probably", "likely", "seems",
    "hard to say", "difficult to determine", "cannot be certain",
    "insufficient", "it depends", "one of", "either", "not certain",
]


def _norm(ans: str) -> str:
    return (ans or "").strip().lower()


# ── the error LABEL (uses gold; NOT a feature) ────────────────────────────────

def error_label(draft: DraftRecord) -> int:
    """1 = the decision answer is WRONG, 0 = correct. Uses the gold answer;
    this is the target, never fed back as a feature."""
    return int(_norm(draft.decision_answer) != _norm(draft.gold_answer))


# ── the five minimal Phase-0 risk features (all: higher = riskier) ────────────

def f_self_consistency_disagreement(draft: DraftRecord) -> float:
    """1 - (modal-answer share) over the k samples. 0 = unanimous, ->1 = split."""
    s = [_norm(a) for a in draft.samples if _norm(a)]
    if not s:
        return float("nan")
    return 1.0 - Counter(s).most_common(1)[0][1] / len(s)


def f_answer_entropy(draft: DraftRecord) -> float:
    """Normalized Shannon entropy of the sampled-answer distribution (0..1)."""
    s = [_norm(a) for a in draft.samples if _norm(a)]
    if len(s) <= 1:
        return 0.0 if s else float("nan")
    counts = Counter(s)
    n = len(s)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    hmax = math.log(min(len(counts), n))
    return float(h / hmax) if hmax > 0 else 0.0


def f_neg_token_logprob_mean(draft: DraftRecord) -> float:
    """Mean negative token log-prob (NLL) of the decision draft. Higher = the
    model was less confident generating its own answer. NaN if unavailable."""
    lp = draft.token_logprobs
    if not lp:
        return float("nan")
    return float(-sum(lp) / len(lp))


def f_verbalized_uncertainty(draft: DraftRecord) -> float:
    """1 - verbalized confidence (0..1). NaN if the model gave no number."""
    c = draft.verbalized_confidence
    return float("nan") if c is None else float(1.0 - max(0.0, min(1.0, c)))


def f_hedging_rate(draft: DraftRecord) -> float:
    """Hedging-phrase hits per 100 words of the decision rationale."""
    text = _norm(draft.reasoning_text)
    if not text:
        return float("nan")
    words = max(len(text.split()), 1)
    hits = sum(text.count(p) for p in HEDGING_PHRASES)
    return 100.0 * hits / words


#: ordered feature registry (name -> function). Every value: higher = riskier.
MINIMAL_FEATURES = {
    "self_consistency_disagreement": f_self_consistency_disagreement,
    "answer_entropy": f_answer_entropy,
    "neg_token_logprob_mean": f_neg_token_logprob_mean,
    "verbalized_uncertainty": f_verbalized_uncertainty,
    "hedging_rate": f_hedging_rate,
}


def extract_features(draft: DraftRecord, registry: dict | None = None) -> dict:
    """Return {feature_name: value}. Never touches draft.gold_answer."""
    reg = registry or MINIMAL_FEATURES
    return {name: fn(draft) for name, fn in reg.items()}


def feature_matrix(drafts, registry: dict | None = None):
    """Return (X, y, names) with NaNs left in place (impute at model time).
    y = error labels. X columns follow the registry order."""
    reg = registry or MINIMAL_FEATURES
    names = list(reg)
    X = np.array([[reg[n](d) for n in names] for d in drafts], float)
    y = np.array([error_label(d) for d in drafts], int)
    return X, y, names
