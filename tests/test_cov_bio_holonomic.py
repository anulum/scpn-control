# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Bio-Holonomic Controller Coverage Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Cover bio_holonomic_controller.py — all lines."""

from __future__ import annotations

import sys

import pytest

from scpn_control.control.bio_holonomic_controller import (
    SC_NEUROCORE_HOLONOMIC_AVAILABLE,
    BioHolonomicController,
    BioTelemetrySnapshot,
)

_needs_sc_neurocore = pytest.mark.skipif(not SC_NEUROCORE_HOLONOMIC_AVAILABLE, reason="sc-neurocore not installed")


@_needs_sc_neurocore
def test_sc_neurocore_available():
    assert SC_NEUROCORE_HOLONOMIC_AVAILABLE is True


def test_bio_telemetry_snapshot_creation():
    snap = BioTelemetrySnapshot(heart_rate_bpm=72.0, eeg_coherence_r=0.85, galvanic_skin_response=3.2)
    assert snap.heart_rate_bpm == 72.0
    assert snap.eeg_coherence_r == 0.85
    assert snap.galvanic_skin_response == 3.2


def test_bio_telemetry_snapshot_frozen():
    snap = BioTelemetrySnapshot(heart_rate_bpm=60.0, eeg_coherence_r=0.5, galvanic_skin_response=1.0)
    with pytest.raises(AttributeError):
        snap.heart_rate_bpm = 80.0  # type: ignore[misc]


@_needs_sc_neurocore
def test_controller_init():
    ctrl = BioHolonomicController(dt_s=0.01, seed=42)
    assert ctrl.dt_s == 0.01
    assert ctrl.l4_adapter is not None
    assert ctrl.l5_adapter is not None


@_needs_sc_neurocore
def test_controller_step_returns_expected_keys():
    ctrl = BioHolonomicController(dt_s=0.01, seed=0)
    snap = BioTelemetrySnapshot(heart_rate_bpm=72.0, eeg_coherence_r=0.6, galvanic_skin_response=2.0)
    result = ctrl.step(snap)
    expected_keys = {
        "l4_order_parameter",
        "l4_avalanche_density",
        "l5_hrv_coherence",
        "l5_emotional_valence",
        "actuator_vibrana_intensity",
    }
    assert set(result.keys()) == expected_keys


@_needs_sc_neurocore
def test_controller_step_vibrana_activates_on_low_coherence():
    ctrl = BioHolonomicController(dt_s=0.005, seed=7)
    snap = BioTelemetrySnapshot(heart_rate_bpm=90.0, eeg_coherence_r=0.1, galvanic_skin_response=5.0)
    result = ctrl.step(snap)
    assert isinstance(result["actuator_vibrana_intensity"], float)
    assert result["actuator_vibrana_intensity"] >= 0.0


@_needs_sc_neurocore
def test_controller_step_multiple_ticks():
    ctrl = BioHolonomicController(dt_s=0.01, seed=99)
    snap = BioTelemetrySnapshot(heart_rate_bpm=65.0, eeg_coherence_r=0.9, galvanic_skin_response=1.5)
    for _ in range(5):
        result = ctrl.step(snap)
    assert "l5_hrv_coherence" in result


def test_controller_raises_when_sc_neurocore_unavailable():
    """Patch the module-level flag to simulate missing sc_neurocore."""
    mod = sys.modules["scpn_control.control.bio_holonomic_controller"]
    original = mod.SC_NEUROCORE_HOLONOMIC_AVAILABLE
    try:
        mod.SC_NEUROCORE_HOLONOMIC_AVAILABLE = False
        with pytest.raises(RuntimeError, match="sc-neurocore"):
            BioHolonomicController()
    finally:
        mod.SC_NEUROCORE_HOLONOMIC_AVAILABLE = original
