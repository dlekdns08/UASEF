"""
Provenance manifest — maps each raw data file to its FIXED experimental condition.

Project rule (non-negotiable): NO result may exist without BOTH answerer_mode AND
verifier_mode. Each raw JSONL file is exactly ONE fixed condition
(answerer_model, answerer_mode, verifier_model, verifier_mode, role, split); those
are recovered here DETERMINISTICALLY from the filename — never guessed per row.

`dataset` is per-ROW (item_id prefix), not per-file: matrix files mix MedMCQA +
PubMedQA, so dataset lives in the row, not the manifest.

roles:
  answerer            — model answered the question (its output is what gets judged)
  verifier_judgment   — verifier scored an answerer's proposed answer (risk V)
  verifier_self_answer— verifier solved the item ITSELF, no answer shown (gives Z, q)

A file may carry MORE THAN ONE condition: the core answerer drafts (gpt-oss,
Qwen3.5-T) double as those models' T-mode verifier self-answers, so describe()
returns a list.
"""
from __future__ import annotations

# ── canonical model registry (SINGLE source of truth) ──
# model_id == the EXACT LM Studio runtime identifier that actually loaded the model
# (verified: `lms ps` shows qwen3.5-122b-a10b; .env LMSTUDIO_MODEL=openai/gpt-oss-120b;
# gemma/qwen3.6 strings are the ones the generation commands used successfully).
# This EXACT string must be what appears in every CSV cell AND the paper — NOT the old
# .tex forms (google/gemma-4-31b-it, Qwen3.5-122B-A10B) which are wrong and must be fixed.
MODELS = {          # alias -> model_id (== runtime_name here; LM Studio uses id as runtime)
    "gptoss": "openai/gpt-oss-120b",
    "qwen35": "qwen3.5-122b-a10b",
    "gemma":  "google/gemma-4-31b",
    "qwen36": "qwen/qwen3.6-27b",
}
ALIAS = {v: k for k, v in MODELS.items()}   # model_id -> short alias (for display)

ANSWERER = "answerer"
JUDGMENT = "verifier_judgment"
SELFANS  = "verifier_self_answer"

# ── explicit fixed-name files: stem -> (a_model, a_mode, v_model, v_mode, role, split) ──
FILES = {
    # core answerer sets (A1 = gpt-oss-T, A2 = Qwen3.5-T, A2-N = Qwen3.5-N ablation pair)
    "drafts_phase0_all":     ("gptoss", "T", None, None, ANSWERER, "matrix"),
    "drafts_qwen35_think":   ("qwen35", "T", None, None, ANSWERER, "matrix"),
    "drafts_qwen35_nothink": ("qwen35", "N", None, None, ANSWERER, "matrix"),  # answerer T/N ablation (B-lite)

    # matrix verifier judgments (verifier judges an answerer set)
    "verifier_cross":            ("gptoss", "T", "gemma",  "T", JUDGMENT, "matrix"),
    "verifier_qwen27":           ("gptoss", "T", "qwen36", "T", JUDGMENT, "matrix"),
    "verifier_qwen35_of_gptoss": ("gptoss", "T", "qwen35", "N", JUDGMENT, "matrix"),  # B-1b
    "verifier_q35T_gptoss":      ("gptoss", "T", "qwen35", "T", JUDGMENT, "matrix"),
    "verifier_gptT_q35":         ("qwen35", "T", "gptoss", "T", JUDGMENT, "matrix"),  # negative-Δ anchor
    "verifier_gptN_q35":         ("qwen35", "T", "gptoss", "N", JUDGMENT, "matrix"),
    "verifier_gemT_q35":         ("qwen35", "T", "gemma",  "T", JUDGMENT, "matrix"),
    "verifier_gemT_q35N":        ("qwen35", "N", "gemma",  "T", JUDGMENT, "matrix"),  # answerer T/N ablation
    "verifier_q36T_q35N":        ("qwen35", "N", "qwen36", "T", JUDGMENT, "matrix"),  # answerer T/N ablation
    "verifier_gemN_gptoss":      ("gptoss", "T", "gemma",  "N", JUDGMENT, "matrix"),
    "verifier_gemN_q35":         ("qwen35", "T", "gemma",  "N", JUDGMENT, "matrix"),
    "verifier_q36T_q35":         ("qwen35", "T", "qwen36", "T", JUDGMENT, "matrix"),
    "verifier_q36N_gptoss":      ("gptoss", "T", "qwen36", "N", JUDGMENT, "matrix"),
    "verifier_q36N_q35":         ("qwen35", "T", "qwen36", "N", JUDGMENT, "matrix"),

    # self-verification diagonal baselines (answerer==verifier, thinking) — reviewer
    # defense only, NOT core. verification_type resolves to "self" automatically.
    "verifier_gptT_gptoss":      ("gptoss", "T", "gptoss", "T", JUDGMENT, "matrix"),  # session 2
    "verifier_q35T_q35":         ("qwen35", "T", "qwen35", "T", JUDGMENT, "matrix"),  # session 8

    # verifier self-answers (verifier solves items itself -> Z, competence proxy q)
    "selfanswer_gemma":      (None, None, "gemma",  "T", SELFANS, "matrix"),
    "selfanswer_qwen27":     (None, None, "qwen36", "T", SELFANS, "matrix"),
    "selfanswer_gptossN":    (None, None, "gptoss", "N", SELFANS, "matrix"),  # session 3
    "selfanswer_gemmaN":     (None, None, "gemma",  "N", SELFANS, "matrix"),  # session 5
    "selfanswer_qwen27N":    (None, None, "qwen36", "N", SELFANS, "matrix"),  # session 7
}

# dual-role: answerer drafts ALSO serve as that model+mode's verifier self-answers
# (drafts are the model solving items itself, no proposed answer shown — definitionally
# a self-answer; provenance marked reused_answerer_draft in consolidation)
DUAL_SELFANS = {
    "drafts_phase0_all":     (None, None, "gptoss", "T", SELFANS, "matrix"),
    "drafts_qwen35_think":   (None, None, "qwen35", "T", SELFANS, "matrix"),
    "drafts_qwen35_nothink": (None, None, "qwen35", "N", SELFANS, "matrix"),  # B-1b symmetric
}

# shuffle answerer --tag -> (a_model, a_mode, split)
SHUF_ANS = {
    "qwen35":      ("qwen35", "T", "shuffled400"),
    "qwen35_orig": ("qwen35", "T", "original400"),
    "gptoss":      ("gptoss", "T", "shuffled400"),
    "gptoss_orig": ("gptoss", "T", "original400"),
}
# shuffle judge --tag -> (v_model, v_mode)
SHUF_VER = {
    "gpt_T": ("gptoss", "T"), "gpt_N": ("gptoss", "N"),
    "gem_T": ("gemma",  "T"), "gem_N": ("gemma",  "N"),
    "q36_T": ("qwen36", "T"), "q36_N": ("qwen36", "N"),
    "q35_T": ("qwen35", "T"), "q35_N": ("qwen35", "N"),
}


def _cond(a_m, a_mode, v_m, v_mode, role, split, stem):
    # verification_type — the THREE mutually-exclusive analysis buckets (must never mix):
    #   self_answer       verifier solved the item itself, no answer shown  (role=self-answer; Z, q, Δ)
    #   self_verification verifier judged its OWN model's answer  (diagonal baseline; a==v)
    #   cross             verifier judged a DIFFERENT model's answer  (core result)
    if role == SELFANS:
        vtype = "self_answer"
    elif role == JUDGMENT and v_m is not None:
        vtype = "self_verification" if (a_m is not None and a_m == v_m) else "cross"
    else:
        vtype = None
    return {"answerer_model": MODELS.get(a_m, a_m), "answerer_mode": a_mode,
            "verifier_model": MODELS.get(v_m, v_m), "verifier_mode": v_mode,
            "role": role, "split": split, "verification_type": vtype, "source": stem}


def describe(stem: str) -> list[dict]:
    """stem = filename without '.jsonl'. Returns a list of condition dicts (1, or 2
    for dual-role answerer drafts). Empty list = file not in the manifest (skip it)."""
    out = []
    if stem in FILES:
        out.append(_cond(*FILES[stem], stem))
    if stem in DUAL_SELFANS:
        out.append(_cond(*DUAL_SELFANS[stem], stem))
    if stem.startswith("shuffle_answer_"):
        tag = stem[len("shuffle_answer_"):]
        if tag in SHUF_ANS:
            a_m, a_mode, split = SHUF_ANS[tag]
            out.append(_cond(a_m, a_mode, None, None, ANSWERER, split, stem))
    if stem.startswith("shuffle_judge_"):
        rest = stem[len("shuffle_judge_"):]
        if "__" in rest:
            ans, tag = rest.split("__", 1)
            if ans in SHUF_ANS and tag in SHUF_VER:
                a_m, a_mode, split = SHUF_ANS[ans]
                v_m, v_mode = SHUF_VER[tag]
                out.append(_cond(a_m, a_mode, v_m, v_mode, JUDGMENT, split, stem))
    return out


def dataset_of(item_id: str) -> str:
    """per-row dataset from item_id prefix."""
    return "pubmedqa" if str(item_id).startswith("pubmedqa") else "medmcqa"
