# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Cov Bio Holonomic
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

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
import types

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


def test_controller_step_with_holonomic_adapter_contract(monkeypatch):
    import importlib

    module_name = "scpn_control.control.bio_holonomic_controller"
    mod = sys.modules[module_name]

    class FakeL4Params:
        def __init__(self):
            self.k_coupling = 0.0

    class FakeL5Params:
        pass

    class FakeL4Adapter:
        def __init__(self, params, seed):
            self.params = params
            self.seed = seed

        def step_jax(self, dt_s):
            assert dt_s == pytest.approx(0.02)
            return {"l4": "bitstream"}

        def get_metrics(self):
            return {"order_parameter": 0.8, "avalanche_density": 0.12}

    class FakeL5Adapter:
        def __init__(self, params, seed):
            self.params = params
            self.seed = seed
            self.inputs = None

        def step_jax(self, dt_s, inputs):
            assert dt_s == pytest.approx(0.02)
            self.inputs = inputs

        def get_metrics(self):
            return {"hrv_coherence_r5": 0.25, "emotional_valence": -0.2}

    l4_mod = types.ModuleType("sc_neurocore.adapters.holonomic.l4_cell")
    l4_mod.L4_CellularAdapter = FakeL4Adapter
    l4_mod.L4_HolonomicParameters = FakeL4Params
    l5_mod = types.ModuleType("sc_neurocore.adapters.holonomic.l5_org")
    l5_mod.L5_OrganismalAdapter = FakeL5Adapter
    l5_mod.L5_HolonomicParameters = FakeL5Params

    for name in (
        "sc_neurocore",
        "sc_neurocore.adapters",
        "sc_neurocore.adapters.holonomic",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sc_neurocore.adapters.holonomic.l4_cell", l4_mod)
    monkeypatch.setitem(sys.modules, "sc_neurocore.adapters.holonomic.l5_org", l5_mod)

    try:
        reloaded = importlib.reload(mod)
        ctrl = reloaded.BioHolonomicController(dt_s=0.02, seed=11)
        result = ctrl.step(reloaded.BioTelemetrySnapshot(72.0, 0.6, 2.5))
        assert ctrl.l4_adapter.params.k_coupling == pytest.approx(0.48)
        assert ctrl.l5_adapter.inputs == {"l4": "bitstream"}
        assert result == {
            "l4_order_parameter": 0.8,
            "l4_avalanche_density": 0.12,
            "l5_hrv_coherence": 0.25,
            "l5_emotional_valence": -0.2,
            "actuator_vibrana_intensity": pytest.approx(0.375),
        }
    finally:
        for name in (
            "sc_neurocore.adapters.holonomic.l4_cell",
            "sc_neurocore.adapters.holonomic.l5_org",
            "sc_neurocore.adapters.holonomic",
            "sc_neurocore.adapters",
            "sc_neurocore",
        ):
            sys.modules.pop(name, None)
        importlib.reload(mod)
