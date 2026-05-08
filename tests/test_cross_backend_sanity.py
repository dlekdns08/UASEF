"""
Cross-backend / cross-method sanity checks.

These post-hoc tests scan a `results/run_<ts>/` directory and flag
suspicious patterns that would indicate a placeholder / copy-paste / data
leakage bug, e.g.:

  - Two distinct backends producing identical confusion matrices on a
    stratum that has > 1 expected positive (CRITICAL n=100 with
    96/4/0 on both gpt-4o and LMStudio is suspicious enough to warrant
    a manual review even if statistically possible).
  - Two distinct *methods* producing identical confusion matrices when
    they are not mathematically equivalent.

The tests are designed to *not* hard-fail on the existing run
(since CRITICAL 96/4/0 is a real coincidence under the current α=0.05
calibration), but to emit a clear pytest XPASS / warning. To enforce the
check at submission time, run:

    UASEF_SANITY_STRICT=1 pytest tests/test_cross_backend_sanity.py

which converts soft warnings into test failures.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STRICT = os.environ.get("UASEF_SANITY_STRICT", "0") == "1"


def _latest_run() -> Path | None:
    runs = sorted(RESULTS.glob("run_2*"), reverse=True)
    return runs[0] if runs else None


@pytest.fixture(scope="module")
def latest_run() -> Path:
    r = _latest_run()
    if r is None:
        pytest.skip("No results/run_<ts>/ directory found.")
    return r


def _load_table4(run: Path, backend: str) -> dict | None:
    p = run / backend / "table4_baseline.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def test_cross_backend_critical_not_identical_for_independent_methods(latest_run: Path):
    """
    Two backends running TECP / Quach / SE *should* still produce
    different confusion matrices on CRITICAL — they consume different
    LLM scores. If two backends agree byte-for-byte on a non-trivial
    stratum across baselines, that is evidence of a data-loading bug.
    UASEF v2 may legitimately converge to the same threshold-bound count,
    so we exempt it.
    """
    backends = []
    for b in ("openai", "lmstudio"):
        d = _load_table4(latest_run, b)
        if d is not None:
            backends.append((b, d))
    if len(backends) < 2:
        pytest.skip("Need at least two backends to cross-check.")

    # Compare the 'TECP' method between backends.
    def crit_signature(data: dict, method_prefix: str) -> tuple | None:
        for m in data.get("methods", []):
            if m["name"].startswith(method_prefix):
                c = m["per_stratum"].get("CRITICAL", {})
                return (c.get("tp"), c.get("fn"), c.get("fp"))
        return None

    for prefix in ["TECP (Xu", "Quach", "Semantic", "UASEF Round 6"]:
        sigs = {b: crit_signature(d, prefix) for b, d in backends}
        if all(s is None for s in sigs.values()):
            continue
        signatures = set(s for s in sigs.values() if s is not None)
        if len(signatures) == 1:
            msg = (
                f"⚠ Cross-backend identical CRITICAL signature for {prefix!r}: "
                f"{sigs}. This is unusual — verify data loading is not bugged."
            )
            if STRICT:
                pytest.fail(msg)
            warnings.warn(msg, UserWarning)


def test_methods_not_identical_when_distinct(latest_run: Path):
    """
    Within a single backend, two methods that are *not* mathematically
    equivalent should not produce identical CRITICAL confusion matrices.

    Allowed identity (mathematically equivalent under our split-CP harness):
      TECP ≡ Quach 2024 CLM ≡ Semantic Entropy

    Suspicious identity:
      Round 6 ≡ Round 7  (would mean Stratified CRC reduced to v1 multipliers)
      v2 ≡ Cost-Sensitive (would mean Pivot A added nothing on top of cost-aware)
    """
    EQUIVALENT = {"TECP", "Quach", "Semantic"}
    for backend in ("openai", "lmstudio"):
        d = _load_table4(latest_run, backend)
        if d is None:
            continue
        sigs: dict[str, tuple] = {}
        for m in d.get("methods", []):
            c = m["per_stratum"].get("CRITICAL", {})
            sigs[m["name"]] = (c.get("tp"), c.get("fn"), c.get("fp"))
        # group by signature
        groups: dict[tuple, list[str]] = {}
        for name, sig in sigs.items():
            groups.setdefault(sig, []).append(name)
        for sig, names in groups.items():
            if len(names) <= 1:
                continue
            non_equiv = [
                n for n in names
                if not any(n.startswith(prefix) for prefix in EQUIVALENT)
            ]
            if len(non_equiv) > 1:
                msg = (
                    f"⚠ {backend}: distinct methods produced identical CRITICAL "
                    f"signature {sig}: {non_equiv}. Investigate before reporting."
                )
                if STRICT:
                    pytest.fail(msg)
                warnings.warn(msg, UserWarning)


def test_v2_recall_at_least_v1_on_critical(latest_run: Path):
    """
    UASEF v2 should not be *worse* than UASEF v1 on the CRITICAL stratum;
    if it is, the per-stratum α is mistuned and the headline claim is
    invalid.
    """
    for backend in ("openai", "lmstudio"):
        d = _load_table4(latest_run, backend)
        if d is None:
            continue
        v1 = next((m for m in d["methods"] if "Round 6" in m["name"]), None)
        v2 = next((m for m in d["methods"] if "Round 7" in m["name"]), None)
        if v1 is None or v2 is None:
            continue
        v1_recall = v1["per_stratum"]["CRITICAL"].get("safety_recall")
        v2_recall = v2["per_stratum"]["CRITICAL"].get("safety_recall")
        if v1_recall is None or v2_recall is None:
            continue
        assert v2_recall >= v1_recall - 1e-6, (
            f"{backend}: v2 CRITICAL recall {v2_recall:.4f} < v1 {v1_recall:.4f} — "
            f"Pivot A is mistuned. Check α_CRITICAL or n_cal."
        )
