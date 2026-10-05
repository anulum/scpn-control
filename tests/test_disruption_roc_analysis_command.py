# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Synthetic disruption ROC real index and command tests.

"""Check observed synthetic event indices and the actual fixed ROC command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from campaign_command_observation import ROOT, observe_command

from scpn_control.benchmark_records import load_verified_latest
from scpn_control.control.disruption_predictor import simulate_tearing_mode
from validation.disruption_roc_analysis import evaluate_batch, generate_scenario_batch


@pytest.fixture(scope="module")
def generated_shots() -> list[dict[str, Any]]:
    """Produce one real 40-shot balanced cohort shared by related contract observations.

    Returns
    -------
    list[dict[str, Any]]
        Actual deterministic synthetic signals, labels and event indices.
    """
    return generate_scenario_batch(40)


def test_actual_observed_event_indices_and_perfect_low_threshold(generated_shots: list[dict[str, Any]]) -> None:
    """Actual simulator threshold crossings use absolute rather than trigger-relative indices.

    Parameters
    ----------
    generated_shots : list[dict[str, Any]]
        Unmodified real generated cohort.
    """
    assert len(generated_shots) == 40
    assert sum(x["label"] for x in generated_shots) == 20
    assert {x["mode"] for x in generated_shots} == {"safe", "density_limit", "vde"}
    for shot in generated_shots:
        signal = shot["signal"]
        if shot["label"]:
            assert shot["t_disrupt"] == len(signal) - 1
            limit = 0.5 if shot["mode"] == "vde" else 15 / (np.pi * 2**2)
            assert signal[-1] > limit and signal[-2] <= limit
        else:
            assert shot["t_disrupt"] == -1
    assert evaluate_batch(generated_shots, 0.0) == {"tpr": 1.0, "fpr": 1.0}
    assert evaluate_batch(generated_shots, 1.0) == {"tpr": 0.0, "fpr": 0.0}
    assert evaluate_batch(generated_shots, float("nan")) == {"tpr": 0.0, "fpr": 0.0}


def test_actual_batch_repeats_with_odd_quota_and_preserves_global_rng() -> None:
    """Odd counts keep floor-half disruptive and ceiling-half safe without global RNG mutation."""
    before = np.random.get_state()
    first = generate_scenario_batch(5)
    second = generate_scenario_batch(5)
    assert len(first) == 5 and sum(x["label"] for x in first) == 2
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a["signal"], b["signal"])
        assert {k: v for k, v in a.items() if k != "signal"} == {k: v for k, v in b.items() if k != "signal"}
    after = np.random.get_state()
    assert isinstance(before, tuple) and isinstance(after, tuple)
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_actual_ntm_risk_and_short_signal_boundaries() -> None:
    """Public evaluation handles genuine NTM proxy signals and windows shorter than 128 samples."""
    for steps in [64, 500]:
        signal, label, _ = simulate_tearing_mode(steps=steps, mode="ntm", rng=np.random.default_rng(42))
        assert label == 0
        shot = dict(signal=signal, label=label, t_disrupt=-1, mode="ntm")
        result = evaluate_batch([shot], 0.0)
        assert result == {"tpr": 0.0, "fpr": 0.0 if steps < 128 else 1.0}
    assert generate_scenario_batch(0) == generate_scenario_batch(-1) == []
    assert evaluate_batch([], 0.5) == {"tpr": 0.0, "fpr": 0.0}


def test_actual_complete_fixed_roc_uses_temporary_software_custody(tmp_path: Path) -> None:
    """Execute the real 100-shot/51-threshold main and bind both outputs through the public recorded runner.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated cwd with real generated reports and custody records.
    """
    argv = [
        sys.executable,
        str(ROOT / "tools/run_recorded_benchmark.py"),
        "--repository-root",
        str(tmp_path),
        "--records-root",
        "records",
        "--family",
        "synthetic-roc-software",
        "--campaign-id",
        "actual-fixed",
        "--evidence-class",
        "software_boundary_test",
        "--artifact",
        "report=validation/reports/disruption_roc.json",
        "--artifact",
        "markdown=validation/reports/disruption_roc.md",
        "--",
        sys.executable,
        str(ROOT / "validation/disruption_roc_analysis.py"),
    ]
    result = observe_command(tmp_path, argv)
    assert result.returncode == 0, result.stdout + result.stderr
    _, manifest = load_verified_latest(tmp_path / "records", "synthetic-roc-software")
    assert manifest["evidence_class"] == "software_boundary_test" and manifest["status"] == "succeeded"
    report = json.loads((tmp_path / "validation/reports/disruption_roc.json").read_text())
    assert len(report["tpr"]) == len(report["fpr"]) == 51
    assert report["tpr"][0] == report["fpr"][0] == 1.0
    assert report["tpr"][-1] == report["fpr"][-1] == 0.0
    assert all(a >= b for a, b in zip(report["tpr"], report["tpr"][1:]))
    assert all(a >= b for a, b in zip(report["fpr"], report["fpr"][1:]))
    assert 0 <= report["auc"] <= 1
    markdown = (tmp_path / "validation/reports/disruption_roc.md").read_text()
    assert f"**AUC**: {report['auc']:.4f}" in markdown
    assert ("Result: PASS" in markdown) is (report["auc"] > 0.85)
    assert f"AUC={report['auc']:.4f}" in result.stdout
