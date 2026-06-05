# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed-scenario scheduler v2 tests.
"""Tests for the CONTROL-owned pulsed-scenario scheduler v2."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    CapacitorBankTelemetry,
    PulsedPlasmaTelemetry,
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
    PulsedScenarioState,
)
from scpn_control.control import pulsed_scenario_scheduler as mif_contract


def _spec() -> PulsedScenarioSpec:
    return PulsedScenarioSpec(
        min_precharge_energy_J=100.0,
        ramp_current_A=2.0e6,
        phase_tolerance_rad=0.01,
        spatial_tolerance_m=0.002,
        burn_temperature_eV=1.0e3,
        min_fusion_power_W=2.0e6,
        expansion_velocity_m_s=1.0e3,
        dump_energy_floor_J=40.0,
        recharge_voltage_fraction=0.95,
        cooldown_temperature_eV=20.0,
        cooldown_current_A=1.0e3,
    )


def _plasma(
    *,
    coil_current_A: float = 0.0,
    temperature_eV: float = 10.0,
    phase_lock_error_rad: float = 0.02,
    reference_error_m: float = 0.01,
    fusion_power_W: float = 0.0,
    radial_velocity_m_s: float = 0.0,
) -> PulsedPlasmaTelemetry:
    return PulsedPlasmaTelemetry(
        coil_current_A=coil_current_A,
        temperature_eV=temperature_eV,
        phase_lock_error_rad=phase_lock_error_rad,
        reference_error_m=reference_error_m,
        fusion_power_W=fusion_power_W,
        radial_velocity_m_s=radial_velocity_m_s,
    )


def _bank(*, voltage_V: float = 9_800.0, energy_J: float = 200.0) -> CapacitorBankTelemetry:
    return CapacitorBankTelemetry(voltage_V=voltage_V, voltage_max_V=10_000.0, energy_J=energy_J)


def test_campaign_traverses_eight_states_with_audit_log() -> None:
    scheduler = PulsedScenarioScheduler(_spec())
    samples = [
        (0.0, _plasma(), _bank()),
        (1.0e-3, _plasma(coil_current_A=2.5e6), _bank()),
        (
            2.0e-3,
            _plasma(
                coil_current_A=2.5e6,
                temperature_eV=1.2e3,
                phase_lock_error_rad=0.004,
                reference_error_m=0.001,
            ),
            _bank(),
        ),
        (
            3.0e-3,
            _plasma(
                coil_current_A=2.5e6,
                temperature_eV=1.5e3,
                phase_lock_error_rad=0.004,
                reference_error_m=0.001,
                fusion_power_W=3.0e6,
            ),
            _bank(),
        ),
        (4.0e-3, _plasma(radial_velocity_m_s=1.5e3, temperature_eV=200.0), _bank()),
        (5.0e-3, _plasma(temperature_eV=120.0), _bank(voltage_V=2_000.0, energy_J=20.0)),
        (6.0e-3, _plasma(temperature_eV=40.0), _bank(voltage_V=9_700.0, energy_J=180.0)),
        (7.0e-3, _plasma(temperature_eV=15.0, coil_current_A=100.0), _bank()),
    ]

    commands = [scheduler.step(t_s, plasma, bank) for t_s, plasma, bank in samples]

    assert [command.state for command in commands] == [
        PulsedScenarioState.RAMP_UP,
        PulsedScenarioState.FLAT_TOP,
        PulsedScenarioState.BURN,
        PulsedScenarioState.EXPANSION,
        PulsedScenarioState.DUMP,
        PulsedScenarioState.RECHARGE,
        PulsedScenarioState.COOL_DOWN,
        PulsedScenarioState.IDLE,
    ]
    assert all(command.transition for command in commands)
    assert [record.to_state for record in scheduler.audit_log] == [command.state for command in commands]
    rows = [json.loads(line) for line in scheduler.audit_log_jsonl().splitlines()]
    assert rows[-1]["to_state"] == "idle"
    assert rows[-1]["reason"] == "plasma cooled and coil current cleared"


def test_flat_top_waits_for_phase_and_spatial_lock() -> None:
    scheduler = PulsedScenarioScheduler(_spec())
    scheduler.step(0.0, _plasma(), _bank())
    scheduler.step(1.0e-3, _plasma(coil_current_A=2.5e6), _bank())

    command = scheduler.step(
        2.0e-3,
        _plasma(
            coil_current_A=2.5e6,
            temperature_eV=1.5e3,
            phase_lock_error_rad=0.02,
            reference_error_m=0.001,
        ),
        _bank(),
    )

    assert command.state is PulsedScenarioState.FLAT_TOP
    assert not command.transition
    assert command.reason == "waiting for phase and spatial lock"


def test_idle_waits_for_precharge_energy() -> None:
    scheduler = PulsedScenarioScheduler(_spec())

    command = scheduler.step(0.0, _plasma(), _bank(energy_J=20.0))

    assert command.state is PulsedScenarioState.IDLE
    assert not command.transition
    assert command.reason == "waiting for precharge energy"
    assert scheduler.audit_log == ()


def test_burn_waits_for_minimum_dwell() -> None:
    scheduler = PulsedScenarioScheduler(replace(_spec(), min_burn_duration_s=2.0e-3))
    scheduler.step(0.0, _plasma(), _bank())
    scheduler.step(1.0e-3, _plasma(coil_current_A=2.5e6), _bank())
    scheduler.step(
        2.0e-3,
        _plasma(
            coil_current_A=2.5e6,
            temperature_eV=1.2e3,
            phase_lock_error_rad=0.004,
            reference_error_m=0.001,
        ),
        _bank(),
    )

    command = scheduler.step(
        3.0e-3,
        _plasma(
            coil_current_A=2.5e6,
            temperature_eV=1.5e3,
            phase_lock_error_rad=0.004,
            reference_error_m=0.001,
            fusion_power_W=3.0e6,
        ),
        _bank(),
    )

    assert command.state is PulsedScenarioState.BURN
    assert not command.transition
    assert command.reason == "waiting for minimum burn dwell"


def test_dump_waits_for_bank_energy_floor() -> None:
    scheduler = PulsedScenarioScheduler(_spec())
    scheduler.step(0.0, _plasma(), _bank())
    scheduler.step(1.0e-3, _plasma(coil_current_A=2.5e6), _bank())
    scheduler.step(
        2.0e-3,
        _plasma(
            coil_current_A=2.5e6,
            temperature_eV=1.2e3,
            phase_lock_error_rad=0.004,
            reference_error_m=0.001,
        ),
        _bank(),
    )
    scheduler.step(
        3.0e-3,
        _plasma(
            coil_current_A=2.5e6,
            temperature_eV=1.5e3,
            phase_lock_error_rad=0.004,
            reference_error_m=0.001,
            fusion_power_W=3.0e6,
        ),
        _bank(),
    )
    scheduler.step(4.0e-3, _plasma(radial_velocity_m_s=1.5e3, temperature_eV=200.0), _bank())

    command = scheduler.step(5.0e-3, _plasma(temperature_eV=120.0), _bank(voltage_V=2_000.0, energy_J=80.0))

    assert command.state is PulsedScenarioState.DUMP
    assert not command.transition
    assert command.reason == "waiting for dump energy floor"


def test_rejects_non_monotone_time_and_non_adjacent_manual_transition() -> None:
    scheduler = PulsedScenarioScheduler(_spec())
    scheduler.step(1.0, _plasma(), _bank())

    with pytest.raises(ValueError, match="monotone"):
        scheduler.step(0.5, _plasma(), _bank())

    scheduler.reset()
    with pytest.raises(ValueError, match="invalid transition"):
        scheduler.transition_to(PulsedScenarioState.BURN, t_s=0.0, reason="skip ramp")


def test_mif_contract_import_path_reexports_v2_scheduler() -> None:
    assert mif_contract.PulsedScenarioScheduler is PulsedScenarioScheduler
    assert mif_contract.PulsedScenarioState is PulsedScenarioState


def test_all_states_return_to_idle_within_one_formal_cycle() -> None:
    ordered_states = [
        PulsedScenarioState.IDLE,
        PulsedScenarioState.RAMP_UP,
        PulsedScenarioState.FLAT_TOP,
        PulsedScenarioState.BURN,
        PulsedScenarioState.EXPANSION,
        PulsedScenarioState.DUMP,
        PulsedScenarioState.RECHARGE,
        PulsedScenarioState.COOL_DOWN,
    ]
    successor = dict(zip(ordered_states, ordered_states[1:] + [PulsedScenarioState.IDLE], strict=True))

    for start in ordered_states:
        state = start
        visited = [state]
        for _ in range(8):
            state = successor[state]
            visited.append(state)
            if state is PulsedScenarioState.IDLE:
                break

        assert state is PulsedScenarioState.IDLE
        assert len(visited) <= 9
