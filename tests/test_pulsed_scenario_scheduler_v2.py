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


class TestSpecValidation:
    def test_rejects_negative_field(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            replace(_spec(), ramp_current_A=-1.0)

    def test_rejects_nonpositive_tolerances(self) -> None:
        with pytest.raises(ValueError, match="phase_tolerance_rad must be strictly positive"):
            replace(_spec(), phase_tolerance_rad=0.0)
        with pytest.raises(ValueError, match="spatial_tolerance_m must be strictly positive"):
            replace(_spec(), spatial_tolerance_m=0.0)

    def test_rejects_out_of_range_recharge_fraction(self) -> None:
        with pytest.raises(ValueError, match=r"recharge_voltage_fraction must lie in \(0, 1\]"):
            replace(_spec(), recharge_voltage_fraction=1.5)


class TestTelemetryValidation:
    def test_plasma_rejects_negatives(self) -> None:
        with pytest.raises(ValueError, match="temperature_eV must be non-negative"):
            _plasma(temperature_eV=-1.0)
        with pytest.raises(ValueError, match="phase_lock_error_rad must be non-negative"):
            _plasma(phase_lock_error_rad=-1.0)
        with pytest.raises(ValueError, match="reference_error_m must be non-negative"):
            _plasma(reference_error_m=-1.0)
        with pytest.raises(ValueError, match="fusion_power_W must be non-negative"):
            _plasma(fusion_power_W=-1.0)

    def test_bank_validation(self) -> None:
        with pytest.raises(ValueError, match="voltage_V must be non-negative"):
            CapacitorBankTelemetry(voltage_V=-1.0, voltage_max_V=10.0, energy_J=1.0)
        with pytest.raises(ValueError, match="voltage_max_V must be strictly positive"):
            CapacitorBankTelemetry(voltage_V=1.0, voltage_max_V=0.0, energy_J=1.0)
        with pytest.raises(ValueError, match="voltage_V must not exceed voltage_max_V"):
            CapacitorBankTelemetry(voltage_V=20.0, voltage_max_V=10.0, energy_J=1.0)
        with pytest.raises(ValueError, match="energy_J must be non-negative"):
            CapacitorBankTelemetry(voltage_V=1.0, voltage_max_V=10.0, energy_J=-1.0)


class TestManualTransitionAndTimestamp:
    def test_transition_to_records_valid_adjacent_move(self) -> None:
        scheduler = PulsedScenarioScheduler(_spec())
        record = scheduler.transition_to(PulsedScenarioState.RAMP_UP, 0.5, "manual ramp")
        assert record.to_state is PulsedScenarioState.RAMP_UP
        assert scheduler.state is PulsedScenarioState.RAMP_UP
        assert scheduler.audit_log[-1] is record

    def test_transition_to_rejects_empty_reason(self) -> None:
        scheduler = PulsedScenarioScheduler(_spec())
        with pytest.raises(ValueError, match="reason must not be empty"):
            scheduler.transition_to(PulsedScenarioState.RAMP_UP, 0.5, "   ")

    def test_validate_timestamp_rejects_negative(self) -> None:
        scheduler = PulsedScenarioScheduler(_spec())
        with pytest.raises(ValueError, match="t_s must be non-negative"):
            scheduler.step(-1.0, _plasma(), _bank())


class TestGuardWaitingBranches:
    def _step_in_state(self, state: PulsedScenarioState, plasma, bank):
        scheduler = PulsedScenarioScheduler(_spec())
        scheduler.state = state
        return scheduler.step(1.0, plasma, bank)

    def test_ramp_up_waits_for_ramp_current(self) -> None:
        command = self._step_in_state(PulsedScenarioState.RAMP_UP, _plasma(coil_current_A=0.0), _bank())
        assert command.transition is False
        assert command.reason == "waiting for ramp current"

    def test_flat_top_waits_for_temperature(self) -> None:
        plasma = _plasma(phase_lock_error_rad=0.004, reference_error_m=0.001, temperature_eV=10.0)
        command = self._step_in_state(PulsedScenarioState.FLAT_TOP, plasma, _bank())
        assert command.transition is False
        assert command.reason == "waiting for burn temperature"

    def test_flat_top_waits_for_energy(self) -> None:
        plasma = _plasma(phase_lock_error_rad=0.004, reference_error_m=0.001, temperature_eV=1.5e3)
        command = self._step_in_state(PulsedScenarioState.FLAT_TOP, plasma, _bank(energy_J=50.0))
        assert command.transition is False
        assert command.reason == "waiting for burn energy"

    def test_burn_waits_for_fusion_power(self) -> None:
        command = self._step_in_state(PulsedScenarioState.BURN, _plasma(fusion_power_W=0.0), _bank())
        assert command.transition is False
        assert command.reason == "waiting for fusion power"

    def test_expansion_waits_for_velocity(self) -> None:
        command = self._step_in_state(PulsedScenarioState.EXPANSION, _plasma(radial_velocity_m_s=0.0), _bank())
        assert command.transition is False
        assert command.reason == "waiting for expansion velocity"

    def test_recharge_waits_for_voltage(self) -> None:
        command = self._step_in_state(PulsedScenarioState.RECHARGE, _plasma(), _bank(voltage_V=9_000.0))
        assert command.transition is False
        assert command.reason == "waiting for recharge voltage"

    def test_cool_down_waits_for_cooldown(self) -> None:
        command = self._step_in_state(PulsedScenarioState.COOL_DOWN, _plasma(temperature_eV=100.0), _bank())
        assert command.transition is False
        assert command.reason == "waiting for plasma cooldown"


def test_finite_rejects_non_finite() -> None:
    from scpn_control.control.pulsed_scenario_scheduler_v2 import _finite

    with pytest.raises(ValueError, match="x must be finite"):
        _finite("x", float("inf"))
