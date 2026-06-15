# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capacitor-bank state model tests.
"""Tests for the CONTROL-owned capacitor-bank RLC state model."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import pytest

from scpn_control.control.capacitor_bank_state import (
    CapacitorBank,
    CapacitorBankSpec,
    CapacitorBankState,
    ENERGY_BALANCE_REL_TOLERANCE,
    EnergyReport,
    PulseSpec,
    RLCRegime,
    WaveformName,
    free_response,
)
from scpn_control.control.pulsed_scenario_scheduler_v2 import CapacitorBankTelemetry


def _underdamped_spec() -> CapacitorBankSpec:
    return CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=0.5,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )


def _critical_spec() -> CapacitorBankSpec:
    return CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=2.0,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )


def _overdamped_spec() -> CapacitorBankSpec:
    return CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=5.0,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capacitance_F": 0.0}, "capacitance_F"),
        ({"inductance_H": 0.0}, "inductance_H"),
        ({"series_resistance_ohm": -1.0}, "series_resistance_ohm"),
        ({"voltage_max_V": math.inf}, "voltage_max_V"),
        ({"recharge_power_kW": -1.0}, "recharge_power_kW"),
    ],
)
def test_spec_rejects_invalid_parameters(kwargs: dict[str, float], match: str) -> None:
    values = {
        "capacitance_F": 100e-6,
        "inductance_H": 100e-6,
        "series_resistance_ohm": 0.5,
        "voltage_max_V": 10_000.0,
        "recharge_power_kW": 20.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CapacitorBankSpec(**values)


def test_spec_classifies_all_rlc_regimes() -> None:
    assert _underdamped_spec().regime is RLCRegime.UNDERDAMPED
    assert _critical_spec().regime is RLCRegime.CRITICAL
    assert _overdamped_spec().regime is RLCRegime.OVERDAMPED


def test_free_response_preserves_initial_state_at_zero_time() -> None:
    state = free_response(_underdamped_spec(), v0=5_000.0, i0=12.0, t=0.0)
    assert state.voltage_V == pytest.approx(5_000.0)
    assert state.current_A == pytest.approx(12.0)
    assert state.t == 0.0
    assert isinstance(state, CapacitorBankState)


@pytest.mark.parametrize("spec_factory", [_underdamped_spec, _critical_spec, _overdamped_spec])
def test_free_response_returns_finite_state_for_all_regimes(
    spec_factory: Callable[[], CapacitorBankSpec],
) -> None:
    spec = spec_factory()
    state = free_response(spec, v0=5_000.0, i0=0.0, t=2.0e-5)
    assert math.isfinite(state.voltage_V)
    assert math.isfinite(state.current_A)
    assert math.isfinite(state.di_dt_A_s)
    assert state.energy_J >= 0.0


@pytest.mark.parametrize("regime_resistance", [0.5, 2.0, 5.0])
def test_exact_step_reproduces_closed_form_under_natural_response(regime_resistance: float) -> None:
    """The zero-order-hold stepper must match the analytic free response exactly.

    Across the underdamped, critical, and overdamped regimes the stepped state
    agrees with :func:`free_response` to accumulated floating-point precision,
    not the second-order truncation the Crank-Nicolson update used to leave.
    """
    spec = CapacitorBankSpec(
        capacitance_F=100.0e-6,
        inductance_H=100.0e-6,
        series_resistance_ohm=regime_resistance,
        voltage_max_V=10_000.0,
    )
    bank = CapacitorBank(spec, initial_voltage_V=5_000.0, initial_current_A=12.0)
    dt = 1.0e-7
    n_steps = 200
    for _ in range(n_steps):
        bank.step(dt)
    expected = free_response(spec, v0=5_000.0, i0=12.0, t=n_steps * dt)
    assert bank.state.voltage_V == pytest.approx(expected.voltage_V, abs=1.0e-7)
    assert bank.state.current_A == pytest.approx(expected.current_A, abs=1.0e-7)


def test_single_exact_step_matches_closed_form_with_load() -> None:
    """One forced step must equal the closed-form zero-order-hold response."""
    spec = _underdamped_spec()
    bank = CapacitorBank(spec, initial_voltage_V=5_000.0, initial_current_A=40.0)
    dt = 1.0e-6
    load = 300.0
    state = bank.step(dt, external_load_current_A=load)
    # Zero-order-hold reference: shift to the load steady state, evolve the
    # homogeneous part, and shift back. v_ss = -R u, i_ss = -u.
    v_ss = -spec.series_resistance_ohm * load
    i_ss = -load
    homogeneous = free_response(spec, v0=5_000.0 - v_ss, i0=40.0 - i_ss, t=dt)
    assert state.voltage_V == pytest.approx(homogeneous.voltage_V + v_ss, abs=1.0e-7)
    assert state.current_A == pytest.approx(homogeneous.current_A + i_ss, abs=1.0e-7)


@pytest.mark.parametrize(
    ("regime_resistance", "waveform"),
    [(0.5, "rect"), (2.0, "exp_decay"), (5.0, "half_sine")],
)
def test_discharge_energy_ledger_closes_to_machine_precision(regime_resistance: float, waveform: str) -> None:
    """Exact dynamics plus closed-form energy integrals close the ledger exactly."""
    spec = CapacitorBankSpec(
        capacitance_F=100.0e-6,
        inductance_H=100.0e-6,
        series_resistance_ohm=regime_resistance,
        voltage_max_V=10_000.0,
    )
    bank = CapacitorBank(spec, initial_voltage_V=5_000.0, initial_current_A=20.0)
    pulse = PulseSpec(peak_current_A=300.0, duration_s=1.0e-3, waveform=waveform)
    report = bank.discharge(pulse, dt=1.0e-6, n_steps=1_000)
    assert report.energy_balance_relative_error < 1.0e-11
    assert report.energy_balance_passed is True
    assert report.resistive_loss_J >= 0.0


def test_reset_rejects_voltage_above_bank_limit() -> None:
    bank = CapacitorBank(_underdamped_spec())
    with pytest.raises(ValueError, match="exceeds bank max"):
        bank.reset(10_001.0)


def test_step_rejects_nonfinite_load_current() -> None:
    bank = CapacitorBank(_underdamped_spec(), initial_voltage_V=5_000.0)
    with pytest.raises(ValueError, match="external_load_current_A"):
        bank.step(1.0e-6, external_load_current_A=math.nan)


def test_telemetry_adapts_to_pulsed_scheduler_contract() -> None:
    bank = CapacitorBank(_underdamped_spec(), initial_voltage_V=5_000.0)
    telemetry = bank.telemetry()
    assert isinstance(telemetry, CapacitorBankTelemetry)
    assert telemetry.voltage_V == pytest.approx(5_000.0)
    assert telemetry.energy_J == pytest.approx(0.5 * _underdamped_spec().capacitance_F * 5_000.0**2)


def test_pulse_spec_rejects_unknown_waveform() -> None:
    with pytest.raises(ValueError, match="waveform"):
        PulseSpec(peak_current_A=1_000.0, duration_s=1.0e-3, waveform=cast(WaveformName, "triangle"))


def test_discharge_preserves_total_rlc_energy_bookkeeping() -> None:
    bank = CapacitorBank(_overdamped_spec(), initial_voltage_V=5_000.0)
    initial_energy = bank.state.energy_J
    pulse = PulseSpec(peak_current_A=500.0, duration_s=1.0e-3, waveform="half_sine")
    report = bank.discharge(pulse, dt=1.0e-6, n_steps=1_000)
    assert report.energy_delivered_J + report.energy_remaining_J == pytest.approx(initial_energy, rel=1.0e-12)
    assert report.capacitor_energy_remaining_J + report.inductor_energy_remaining_J == pytest.approx(
        report.energy_remaining_J,
        rel=1.0e-12,
    )
    assert report.energy_balance_residual_J == pytest.approx(
        report.energy_delivered_J - report.resistive_loss_J - report.load_energy_J,
        abs=1.0e-12,
    )
    assert report.energy_balance_relative_error <= ENERGY_BALANCE_REL_TOLERANCE
    assert report.energy_balance_passed is True
    assert report.rlc_regime is RLCRegime.OVERDAMPED
    assert report.discharge_duration_s == pytest.approx(1.0e-3)


def test_discharge_energy_balance_includes_initial_inductor_energy() -> None:
    bank = CapacitorBank(_underdamped_spec(), initial_voltage_V=4_000.0, initial_current_A=75.0)
    capacitor_only_initial = bank.state.energy_J
    pulse = PulseSpec(peak_current_A=0.1, duration_s=2.0e-5, waveform="rect")
    report = bank.discharge(pulse, dt=1.0e-7, n_steps=200)
    assert report.energy_initial_J > capacitor_only_initial
    assert report.energy_delivered_J + report.energy_remaining_J == pytest.approx(report.energy_initial_J, rel=1.0e-12)
    assert report.energy_balance_passed is True


def test_energy_report_rejects_invalid_balance_components() -> None:
    with pytest.raises(ValueError, match="resistive_loss_J"):
        EnergyReport(
            energy_delivered_J=1.0,
            energy_initial_J=1.0,
            energy_remaining_J=0.0,
            capacitor_energy_remaining_J=0.0,
            inductor_energy_remaining_J=0.0,
            resistive_loss_J=-1.0,
            load_energy_J=0.0,
            energy_balance_residual_J=0.0,
            energy_balance_relative_error=0.0,
            energy_balance_passed=True,
            peak_voltage_V=1.0,
            peak_current_A=1.0,
            discharge_duration_s=1.0,
            rlc_regime=RLCRegime.UNDERDAMPED,
        )


def test_feasibility_rejects_peak_current_above_natural_bank_limit() -> None:
    bank = CapacitorBank(_underdamped_spec(), initial_voltage_V=5_000.0)
    pulse = PulseSpec(peak_current_A=20_000.0, duration_s=1.0e-6)
    feasible, reason = bank.feasibility(pulse)
    assert feasible is False
    assert "natural peak" in reason


def test_feasibility_uses_total_rlc_energy_when_inductor_already_stores_current() -> None:
    spec = _underdamped_spec()
    bank = CapacitorBank(spec, initial_voltage_V=100.0, initial_current_A=700.0)
    pulse = PulseSpec(peak_current_A=650.0, duration_s=1.0e-6, waveform="rect")

    feasible, reason = bank.feasibility(pulse)

    assert feasible is True
    assert reason == "ok"
    total_energy = bank.state.energy_J + 0.5 * spec.inductance_H * bank.state.current_A**2
    assert pulse.peak_current_A < math.sqrt(2.0 * total_energy / spec.inductance_H)


def test_feasibility_rejects_resistive_dissipation_above_available_energy() -> None:
    bank = CapacitorBank(_underdamped_spec(), initial_voltage_V=100.0)
    pulse = PulseSpec(peak_current_A=80.0, duration_s=1.0e-3, waveform="rect")
    feasible, reason = bank.feasibility(pulse)
    assert feasible is False
    assert "exceeds available" in reason


def test_recharge_status_uses_linear_energy_growth_until_voltage_cap() -> None:
    spec = _underdamped_spec()
    bank = CapacitorBank(spec, initial_voltage_V=0.0)
    status = bank.recharge_status(0.1)
    projected_energy = 0.5 * spec.capacitance_F * status["projected_voltage_V"] ** 2
    assert projected_energy == pytest.approx(spec.recharge_power_kW * 1000.0 * 0.1)
    assert bank.recharge_status(1.0e6)["projected_voltage_V"] == spec.voltage_max_V


def test_recharge_status_zero_power_returns_infinite_time_to_full() -> None:
    spec = CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=0.5,
        voltage_max_V=10_000.0,
        recharge_power_kW=0.0,
    )
    status = CapacitorBank(spec, initial_voltage_V=2_000.0).recharge_status(1.0)
    assert status["projected_voltage_V"] == pytest.approx(2_000.0)
    assert status["time_to_full_s"] == float("inf")
