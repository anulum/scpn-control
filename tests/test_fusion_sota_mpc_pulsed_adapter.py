# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed-shot MPC adapter tests.
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec, PulseSpec
from scpn_control.control.fusion_sota_mpc import (
    ModelPredictiveController,
    NeuralSurrogate,
    PulsedShotMPCAdapter,
)
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
    PulsedScenarioState,
)


def _mpc() -> ModelPredictiveController:
    surrogate = NeuralSurrogate(n_coils=2, n_state=2, verbose=False)
    surrogate.B = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
    return ModelPredictiveController(surrogate, np.array([6.0, 0.0], dtype=np.float64))


def _scheduler(state: PulsedScenarioState = PulsedScenarioState.BURN) -> PulsedScenarioScheduler:
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
            min_burn_duration_s=0.0,
        )
    )
    scheduler.state = state
    return scheduler


def _bank(initial_voltage_V: float = 10.0, resistance_ohm: float = 0.05) -> CapacitorBank:
    return CapacitorBank(
        CapacitorBankSpec(
            capacitance_F=1.0,
            inductance_H=1.0,
            series_resistance_ohm=resistance_ohm,
            voltage_max_V=20.0,
            recharge_power_kW=1.0,
        ),
        initial_voltage_V=initial_voltage_V,
    )


@pytest.mark.parametrize("state", [s for s in PulsedScenarioState if s is not PulsedScenarioState.BURN])
def test_non_burn_states_mask_burn_components(state: PulsedScenarioState) -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(state),
        _bank(),
        burn_action_mask=np.array([True, False]),
    )

    action = adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]))
    decision = adapter.explain_last_decision()

    assert action[0] == 0.0
    assert action[1] < 0.0
    assert decision["scheduler_state"] == state.value
    assert decision["burn_components_masked"] is True
    assert decision["safe_action_applied"] is False


def test_infeasible_bank_prevents_burn_command() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=0.0, resistance_ohm=1.0),
        burn_action_mask=np.array([True, True]),
        pulse_duration_s=1.0,
    )

    action = adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]))
    decision = adapter.explain_last_decision()

    assert np.array_equal(action, np.zeros(2))
    assert decision["bank_feasible"] is False
    assert decision["safe_action_applied"] is True
    assert decision["constraint_slack"] < 0.0


def test_explicit_feasible_pulse_is_admitted_during_burn() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=10.0, resistance_ohm=0.01),
        burn_action_mask=np.array([True, True]),
    )

    action = adapter.step(
        np.array([5.0, 1.0]),
        np.array([6.0, 0.0]),
        pulse=PulseSpec(peak_current_A=0.5, duration_s=0.001),
    )
    decision = adapter.explain_last_decision()

    assert action[0] > 0.0
    assert action[1] < 0.0
    assert decision["bank_feasible"] is True
    assert decision["safe_action_applied"] is False
    assert decision["mpc_objective"] >= 0.0


def test_invalid_adapter_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="burn_action_mask"):
        PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank(), burn_action_mask=np.array([True]))
    with pytest.raises(ValueError, match="safe_action"):
        PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank(), safe_action=np.array([0.0]))
    with pytest.raises(ValueError, match="pulse_duration_s"):
        PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank(), pulse_duration_s=0.0)


def test_explain_requires_prior_decision() -> None:
    adapter = PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank())

    with pytest.raises(RuntimeError, match="no pulsed MPC decision"):
        adapter.explain_last_decision()
