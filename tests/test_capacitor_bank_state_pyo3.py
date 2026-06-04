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
    assert report["energy_balance_relative_error"] <= 1.0e-8
    recharge = bank.recharge_status(0.1)
    assert recharge["target_voltage_V"] == pytest.approx(10_000.0)
