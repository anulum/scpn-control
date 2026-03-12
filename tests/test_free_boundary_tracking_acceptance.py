# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Free-Boundary Tracking Acceptance Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for deterministic free-boundary acceptance validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "free_boundary_tracking_acceptance.py"
SPEC = importlib.util.spec_from_file_location("free_boundary_tracking_acceptance", MODULE_PATH)
assert SPEC and SPEC.loader
free_boundary_tracking_acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = free_boundary_tracking_acceptance
SPEC.loader.exec_module(free_boundary_tracking_acceptance)


def test_campaign_is_deterministic() -> None:
    a = free_boundary_tracking_acceptance.run_campaign()
    b = free_boundary_tracking_acceptance.run_campaign()
    assert a["passes_thresholds"] is True
    assert b["passes_thresholds"] is True
    for scenario in (
        "nominal",
        "coil_kick",
        "measurement_fault_uncorrected",
        "measurement_fault_corrected",
    ):
        assert a["scenarios"][scenario]["summary"]["final_tracking_error_norm"] == pytest.approx(
            b["scenarios"][scenario]["summary"]["final_tracking_error_norm"]
        )
        assert a["scenarios"][scenario]["passes_thresholds"] == b["scenarios"][scenario]["passes_thresholds"]


def test_campaign_thresholds_pass() -> None:
    out = free_boundary_tracking_acceptance.run_campaign()
    assert out["passes_thresholds"] is True
    assert out["scenarios"]["nominal"]["passes_thresholds"] is True
    assert out["scenarios"]["coil_kick"]["passes_thresholds"] is True
    assert out["scenarios"]["measurement_fault_uncorrected"]["passes_thresholds"] is True
    assert out["scenarios"]["measurement_fault_corrected"]["passes_thresholds"] is True


def test_measurement_fault_scenarios_expose_and_remove_gap() -> None:
    out = free_boundary_tracking_acceptance.run_campaign()
    uncorrected = out["scenarios"]["measurement_fault_uncorrected"]
    corrected = out["scenarios"]["measurement_fault_corrected"]
    nominal = out["scenarios"]["nominal"]

    assert uncorrected["measured_true_gap"] >= uncorrected["thresholds"]["min_measured_true_gap"]
    assert uncorrected["summary"]["max_abs_measurement_offset"] >= uncorrected["thresholds"]["min_measurement_offset"]
    assert corrected["measured_true_gap"] <= corrected["thresholds"]["max_measured_true_gap"]
    assert corrected["summary"]["max_abs_measurement_offset"] <= corrected["thresholds"]["max_measurement_offset"]
    assert corrected["summary"]["final_tracking_error_norm"] == pytest.approx(
        nominal["summary"]["final_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-12,
    )


def test_render_markdown_contains_sections() -> None:
    report = free_boundary_tracking_acceptance.generate_report()
    text = free_boundary_tracking_acceptance.render_markdown(report)
    assert "# Free-Boundary Tracking Acceptance" in text
    assert "## Scenarios" in text
    assert "nominal" in text
    assert "measurement_fault_uncorrected" in text
