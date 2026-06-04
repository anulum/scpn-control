# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed-shot MPC PyO3 parity tests.
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec
from scpn_control.control.fusion_sota_mpc import ModelPredictiveController, NeuralSurrogate, PulsedShotMPCAdapter
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
    PulsedScenarioState,
)

scpn_control_rs = pytest.importorskip("scpn_control_rs", reason="optional PyO3 extension is not installed")


def _python_adapter() -> PulsedShotMPCAdapter:
    surrogate = NeuralSurrogate(n_coils=2, n_state=2, verbose=False)
    surrogate.B = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
    mpc = ModelPredictiveController(surrogate, np.array([6.0, 0.0], dtype=np.float64))
    scheduler = PulsedScenarioScheduler(
        PulsedScenarioSpec(
            min_precharge_energy_J=5.0,
            ramp_current_A=10.0,
            phase_tolerance_rad=0.1,
            spatial_tolerance_m=0.01,
            burn_temperature_eV=100.0,
            min_fusion_power_W=50.0,
            expansion_velocity_m_s=5.0,
            dump_energy_floor_J=0.5,
            recharge_voltage_fraction=0.8,
            cooldown_temperature_eV=10.0,
            cooldown_current_A=1.0,
        )
    )
    bank = CapacitorBank(
        CapacitorBankSpec(
            capacitance_F=1.0,
            inductance_H=1.0,
            series_resistance_ohm=1.0,
            voltage_max_V=20.0,
        ),
        initial_voltage_V=0.0,
    )
    return PulsedShotMPCAdapter(mpc, scheduler, bank, pulse_duration_s=1.0)


def _rust_mpc() -> object:
    rust_mpc = scpn_control_rs.PyMpcController(
        np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64),
        np.array([6.0, 0.0], dtype=np.float64),
    )
    if not hasattr(rust_mpc, "plan_pulsed"):
        pytest.skip("installed PyO3 extension was not rebuilt with plan_pulsed")
    return rust_mpc


def test_pyo3_pulsed_mpc_matches_python_non_burn_mask() -> None:
    adapter = _python_adapter()
    state = np.array([5.0, 1.0], dtype=np.float64)
    safe = np.zeros(2, dtype=np.float64)
    mask = np.array([True, True], dtype=bool)

    py_action = adapter.step(state, context=PulsedScenarioState.FLAT_TOP)
    rust_mpc = _rust_mpc()
    rust_action, rust_decision = rust_mpc.plan_pulsed(state, "flat_top", True, mask, safe, 0.0)

    assert np.allclose(np.asarray(rust_action), py_action)
    assert rust_decision["burn_components_masked"] is True
    assert rust_decision["scheduler_state"] == "flat_top"


def test_pyo3_pulsed_mpc_matches_python_infeasible_bank_safe_action() -> None:
    adapter = _python_adapter()
    state = np.array([5.0, 1.0], dtype=np.float64)
    py_action = adapter.step(state, context=PulsedScenarioState.BURN)
    decision = adapter.explain_last_decision()
    rust_mpc = _rust_mpc()
    rust_action, rust_decision = rust_mpc.plan_pulsed(
        state,
        "burn",
        False,
        np.array([True, True], dtype=bool),
        np.zeros(2, dtype=np.float64),
        float(decision["constraint_slack"]),
    )

    assert np.array_equal(np.asarray(rust_action), py_action)
    assert rust_decision["safe_action_applied"] is True


def test_pyo3_extension_spec_is_available_when_imported() -> None:
    assert importlib.util.find_spec("scpn_control_rs") is not None
