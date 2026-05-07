"""CLI smoke tests for all runners (audit 6.10)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


RUNNER_PATHS = [
    "experiments/run_all_experiments.py",
    "experiments/run_baseline_comparison.py",
    "experiments/run_agent_experiment.py",
    "experiments/run_experiment.py",
    "experiments/eval_medabstain.py",
    "experiments/pareto_sweep.py",
    "experiments/run_calibration_pipeline.py",
    "experiments/compare_runs.py",
]


@pytest.mark.parametrize("script", RUNNER_PATHS)
def test_runner_help(script):
    """모든 runner의 --help가 비-제로 exit code 없이 동작."""
    res = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    assert res.returncode == 0, f"{script} --help failed:\n{res.stderr}"


def test_run_all_experiments_has_audit_6_options():
    """audit 6 / 6.10 신규 CLI 옵션이 모두 노출됨."""
    res = subprocess.run(
        [sys.executable, "experiments/run_all_experiments.py", "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    for opt in ["--prompt-mode", "--decision-rule", "--strict",
                "--allow-fallback", "--run-tag", "--weighted-cp"]:
        assert opt in res.stdout, f"Missing CLI: {opt}"


def test_run_all_experiments_includes_new_backends():
    """audit 6.9: anthropic/gemini 포함."""
    res = subprocess.run(
        [sys.executable, "experiments/run_all_experiments.py", "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    assert "anthropic" in res.stdout
    assert "gemini" in res.stdout
    assert "hybrid" in res.stdout    # scoring-method


def test_run_agent_excludes_new_backends():
    """audit 6.10: agent CLI는 anthropic/gemini를 choices에서 제외 (LangGraph 미지원)."""
    res = subprocess.run(
        [sys.executable, "experiments/run_agent_experiment.py", "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    # backend choices에는 없어야 함
    line = [ln for ln in res.stdout.splitlines() if "--backend" in ln]
    backend_help = " ".join(line)
    assert "anthropic" not in backend_help or "미지원" in res.stdout


def test_build_summary_metadata():
    """audit 6.8/6.10: build_summary가 신규 필드(prompt_mode, hybrid 등)를 보고."""
    sys.path.insert(0, str(ROOT))
    import argparse
    from experiments.run_all_experiments import build_summary

    args = argparse.Namespace(
        backend="openai", n_cal=100, n_test=20, n_medabstain=20, n_pareto_test=20,
        scoring_method="logprob", alpha=0.10, weighted_cp=False, variants=["AP", "NAP"],
        seed=42, prompt_mode="neutral", decision_rule=None, strict=False,
        allow_fallback=False, run_tag="test_tag",
    )
    summary = build_summary({}, {}, {}, {}, {}, args, "0s")
    cfg = summary["meta"]["config"]
    assert cfg["prompt_mode"] == "neutral"
    assert cfg["allow_fallback"] is False
    # audit 6.8: calibration_artifacts 동봉
    assert "calibration_artifacts" in summary["meta"]


def test_compare_runs_help():
    res = subprocess.run(
        [sys.executable, "experiments/compare_runs.py", "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    assert res.returncode == 0
    assert "tags" in res.stdout
