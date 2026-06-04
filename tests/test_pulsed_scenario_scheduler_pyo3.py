# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: PyO3 pulsed-scenario scheduler tests.
"""PyO3 parity tests for the pulsed-scenario scheduler."""

from __future__ import annotations

import pytest

rust = pytest.importorskip(
    "scpn_control_rs",
    reason="Rust extension not built; run maturin develop --release to enable PyO3 parity tests.",
)


def _spec() -> object:
    return rust.PyPulsedScenarioSpec(
        100.0,
        2.0e6,
        0.01,
        0.002,
        1.0e3,
        2.0e6,
        1.0e3,
        40.0,
        0.95,
        20.0,
        1.0e3,
        0.0,
    )


def _plasma(
    coil_current_a: float,
    temperature_ev: float,
    phase_lock_error_rad: float,
    reference_error_m: float,
    fusion_power_w: float,
    radial_velocity_m_s: float,
) -> object:
    return rust.PyPulsedPlasmaTelemetry(
        coil_current_a,
        temperature_ev,
        phase_lock_error_rad,
        reference_error_m,
        fusion_power_w,
        radial_velocity_m_s,
    )


def _bank(voltage_v: float, energy_j: float) -> object:
    return rust.PyCapacitorBankTelemetry(voltage_v, 10_000.0, energy_j)


def test_pyo3_scheduler_traverses_eight_state_campaign() -> None:
    scheduler = rust.PyPulsedScenarioScheduler(_spec())
    samples = [
        (0.0, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
        (1.0e-3, _plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
        (2.0e-3, _plasma(2.5e6, 1200.0, 0.004, 0.001, 0.0, 0.0), _bank(9800.0, 200.0)),
        (3.0e-3, _plasma(2.5e6, 1500.0, 0.004, 0.001, 3.0e6, 0.0), _bank(9800.0, 200.0)),
        (4.0e-3, _plasma(0.0, 200.0, 0.02, 0.01, 0.0, 1500.0), _bank(9800.0, 200.0)),
        (5.0e-3, _plasma(0.0, 120.0, 0.02, 0.01, 0.0, 0.0), _bank(2000.0, 20.0)),
        (6.0e-3, _plasma(0.0, 40.0, 0.02, 0.01, 0.0, 0.0), _bank(9700.0, 180.0)),
        (7.0e-3, _plasma(100.0, 15.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
    ]

    states = []
    for t_s, plasma, bank in samples:
        states.append(scheduler.step(t_s, plasma, bank)[1])

    assert states == [
        "ramp_up",
        "flat_top",
        "burn",
        "expansion",
        "dump",
        "recharge",
        "cool_down",
        "idle",
    ]
    assert scheduler.state == "idle"
    assert len(scheduler.audit_log()) == 8


def test_pyo3_scheduler_rejects_non_adjacent_manual_transition() -> None:
    scheduler = rust.PyPulsedScenarioScheduler(_spec())

    with pytest.raises(ValueError, match="manual transition"):
        scheduler.transition_to("burn", 0.0, "skip states")
