# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capacitor-bank PyO3 bridge tests.
"""Optional tests for the Rust capacitor-bank state bridge."""

from __future__ import annotations

import pytest


rust = pytest.importorskip("scpn_control_rs")


pytestmark = pytest.mark.skipif(
    not hasattr(rust, "PyCapacitorBankModel"),
    reason="installed scpn_control_rs extension does not expose capacitor-bank bridge",
)


def test_pyo3_capacitor_bank_free_response_surface() -> None:
    state = rust.capacitor_bank_free_response(
        100e-6,
        100e-6,
        0.5,
        10_000.0,
        20.0,
        5_000.0,
        0.0,
        0.0,
    )
    assert state["voltage_V"] == pytest.approx(5_000.0)
    assert state["current_A"] == pytest.approx(0.0)
    assert state["energy_J"] == pytest.approx(0.5 * 100e-6 * 5_000.0**2)


def test_pyo3_capacitor_bank_model_step_discharge_and_recharge() -> None:
    bank = rust.PyCapacitorBankModel(100e-6, 100e-6, 5.0, 10_000.0, 20.0, 5_000.0)
    state = bank.step(1.0e-6)
    assert state["t"] == pytest.approx(1.0e-6)
    telemetry = bank.telemetry()
    assert telemetry["energy_J"] >= 0.0
    feasible, reason = bank.feasibility(100.0, 1.0e-4, "half_sine")
    assert feasible is True
    assert reason == "ok"
    report = bank.discharge(100.0, 1.0e-4, "rect", 1.0e-6, 10)
    assert report["discharge_duration_s"] == pytest.approx(1.0e-5)
    assert report["energy_initial_J"] >= report["energy_remaining_J"]
    assert report["capacitor_energy_remaining_J"] + report["inductor_energy_remaining_J"] == pytest.approx(
        report["energy_remaining_J"],
    )
    assert report["energy_balance_passed"] is True
    assert report["energy_balance_relative_error"] < 1.0e-11
    recharge = bank.recharge_status(0.1)
    assert recharge["target_voltage_V"] == pytest.approx(10_000.0)


@pytest.mark.parametrize("regime_resistance", [0.5, 2.0, 5.0])
def test_pyo3_capacitor_bank_matches_python_exact_step(regime_resistance: float) -> None:
    """The Rust and Python exact steppers must agree across damping regimes."""
    from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec

    spec = CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=regime_resistance,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )
    py_bank = CapacitorBank(spec, initial_voltage_V=5_000.0, initial_current_A=15.0)
    rs_bank = rust.PyCapacitorBankModel(100e-6, 100e-6, regime_resistance, 10_000.0, 20.0, 5_000.0, 15.0)
    dt = 1.0e-6
    for k in range(50):
        load = 80.0 * (1.0 + 0.1 * k)
        py_state = py_bank.step(dt, external_load_current_A=load)
        rs_state = rs_bank.step(dt, load)
        assert py_state.voltage_V == pytest.approx(rs_state["voltage_V"], abs=1.0e-7)
        assert py_state.current_A == pytest.approx(rs_state["current_A"], abs=1.0e-7)


def test_pyo3_capacitor_bank_matches_python_exact_discharge() -> None:
    """The Rust and Python discharge ledgers must agree to machine precision."""
    from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec, PulseSpec

    spec = CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=5.0,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )
    py_bank = CapacitorBank(spec, initial_voltage_V=5_000.0, initial_current_A=20.0)
    rs_bank = rust.PyCapacitorBankModel(100e-6, 100e-6, 5.0, 10_000.0, 20.0, 5_000.0, 20.0)
    py_report = py_bank.discharge(PulseSpec(peak_current_A=300.0, duration_s=1.0e-3), dt=1.0e-6, n_steps=1_000)
    rs_report = rs_bank.discharge(300.0, 1.0e-3, "half_sine", 1.0e-6, 1_000)
    assert py_report.energy_delivered_J == pytest.approx(rs_report["energy_delivered_J"], rel=1.0e-9)
    assert py_report.resistive_loss_J == pytest.approx(rs_report["resistive_loss_J"], rel=1.0e-9)
    assert py_report.load_energy_J == pytest.approx(rs_report["load_energy_J"], rel=1.0e-9)
    assert py_report.energy_balance_relative_error < 1.0e-11
    assert rs_report["energy_balance_relative_error"] < 1.0e-11
