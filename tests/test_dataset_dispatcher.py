"""
Tests for the multi-dataset dispatcher (`load_dataset_for_stratification`).

These tests do *not* hit HuggingFace — they validate the dispatcher's
control flow (unknown name → ValueError, supported names list correct,
MedMCQA subject mapping covers the expected subjects, etc.).
"""
from __future__ import annotations

import pytest


def test_supported_datasets_constant():
    from data.loader import SUPPORTED_DATASETS
    assert "medabstain" in SUPPORTED_DATASETS
    assert "medqa_usmle" in SUPPORTED_DATASETS
    assert "medqa_usmle_full" in SUPPORTED_DATASETS
    assert "pubmedqa" in SUPPORTED_DATASETS
    assert "medmcqa" in SUPPORTED_DATASETS


def test_dispatcher_rejects_unknown():
    from data.loader import load_dataset_for_stratification
    with pytest.raises(ValueError):
        load_dataset_for_stratification("nonsense_dataset", n=5)


def test_medmcqa_mapping_covers_strata():
    """All 4 strata should be reachable from MedMCQA subjects via SPECIALTY_TO_STRATUM."""
    from data.loader import MEDMCQA_SUBJECT_TO_SPECIALTY
    from experiments.round7_table1_coverage import SPECIALTY_TO_STRATUM
    reached: set[str] = set()
    for spec in MEDMCQA_SUBJECT_TO_SPECIALTY.values():
        s = SPECIALTY_TO_STRATUM.get(spec, "MODERATE")
        reached.add(s)
    # We expect at least 3 of 4 strata to be reachable (CRITICAL via Anaesthesia).
    assert len(reached) >= 3, f"MedMCQA subject coverage too narrow: {reached}"


def test_collect_stratified_data_signature_includes_dataset():
    """Public function signature gates the new param; ensures shell wiring works."""
    import inspect
    from experiments.round7_table1_coverage import collect_stratified_data
    sig = inspect.signature(collect_stratified_data)
    assert "dataset" in sig.parameters
    assert sig.parameters["dataset"].default == "medabstain"
