# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — H-infinity closed-loop integration tests.

"""Independent continuous and sampled closed-loop checks."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from scpn_control.control.h_infinity_controller import get_flight_sim_controller


def _zoh(state: np.ndarray, inputs: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    augmented = np.zeros((state.shape[0] + inputs.shape[1],) * 2)
    augmented[: state.shape[0], : state.shape[0]] = state * dt
    augmented[: state.shape[0], state.shape[0] :] = inputs * dt
    exponential = expm(augmented)
    return exponential[: state.shape[0], : state.shape[0]], exponential[: state.shape[0], state.shape[0] :]


def test_independent_frequency_sweep_is_below_admitted_gamma() -> None:
    """A dense diagnostic sweep corroborates, but does not replace, DGKF admission."""
    controller = get_flight_sim_controller()
    state, disturbance, performance, feedthrough = controller.closed_loop_realization()
    frequencies = np.concatenate(([0.0], np.logspace(-4, 6, 20_001)))
    peak = 0.0
    identity = np.eye(state.shape[0])
    for frequency in frequencies:
        transfer = performance @ np.linalg.solve(1j * frequency * identity - state, disturbance) + feedthrough
        peak = max(peak, float(np.linalg.svd(transfer, compute_uv=False)[0]))
    assert peak < controller.gamma
    assert controller.gamma - peak > 1.0e-6


def test_supported_sampled_flight_loop_converges() -> None:
    """The documented 20 Hz reduced flight-simulator interconnection converges."""
    controller = get_flight_sim_controller()
    dt = 0.05
    plant_state, plant_input = _zoh(controller.A, controller.B2, dt)
    state = np.array([0.1, 0.0])
    initial_error = abs(float((controller.C2 @ state).item()))
    for _ in range(600):
        measurement = float((controller.C2 @ state).item())
        control = float(controller.step(measurement, dt))
        state = plant_state @ state + plant_input[:, 0] * control
    final_error = abs(float((controller.C2 @ state).item()))
    assert final_error < 1.0e-6 * initial_error


def test_supported_sampled_loop_rejects_physical_disturbance_channel() -> None:
    """The documented sampled example rejects an impulse on the physical channel."""
    controller = get_flight_sim_controller()
    dt = 0.01
    combined_inputs = np.hstack((controller.B2, controller.B1[:, :1]))
    plant_state, plant_inputs = _zoh(controller.A, combined_inputs, dt)
    state = np.zeros(controller.n)
    for sample in range(2_000):
        measurement = float((controller.C2 @ state).item())
        control = float(controller.step(measurement, dt))
        disturbance = 1.0 if sample == 100 else 0.0
        state = plant_state @ state + plant_inputs[:, 0] * control + plant_inputs[:, 1] * disturbance
    assert abs(float((controller.C2 @ state).item())) < 1.0e-8


def test_independent_controller_instances_do_not_share_state() -> None:
    """Dynamic controller state remains instance-local."""
    first = get_flight_sim_controller()
    second = get_flight_sim_controller()
    first.step(1.0, 0.01)
    first.step(1.0, 0.01)
    second.step(0.0, 0.01)
    assert np.linalg.norm(first.state) > 0.0
    assert np.linalg.norm(second.state) == 0.0
