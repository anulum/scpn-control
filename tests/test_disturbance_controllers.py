# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disturbance controller contract tests.

"""Exercise the real scalar controller laws, state and compatible keywords."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from scpn_control.control.h_infinity_controller import get_radial_robust_controller
from scpn_control.control.neuro_cybernetic_controller import SC_NEUROCORE_AVAILABLE
from validation import benchmark_disturbance_rejection as owner
from validation.disturbance_controllers import HInfinityErrorController


def test_real_HInfinity_measurement_adapter_matches_defining_public_controller() -> None:
    """Correct-sign benchmark output equals the actual positive-measurement DGKF loop."""
    adapter = HInfinityErrorController()
    configuration = dict(owner.SCENARIOS["VDE"])
    configuration["duration_s"] = 0.005
    m, t = owner.run_scenario("H-infinity", adapter, "VDE", configuration)
    raw = get_radial_robust_controller()
    plant = owner.LinearPlant()
    plant.reset(np.array([0.01, 0.0]))
    positions = [plant.z]
    controls = []
    for k in range(50):
        u = float(np.asarray(raw.step(plant.z, 1e-4)).item())
        d = owner.SCENARIOS["VDE"]["disturbance"]
        assert callable(d)
        plant.step(u, float(d(k * 1e-4)), 1e-4)
        positions.append(plant.z)
        controls.append(u)
    np.testing.assert_array_equal(t.positions, positions)
    np.testing.assert_array_equal(t.controls[:-1], controls)
    assert m.stable and adapter.controller.is_stable
    assert np.max(np.linalg.eigvals(raw.closed_loop_realization()[0]).real) < 0.0


def test_actual_neural_provider_reset_recreates_independent_trajectory() -> None:
    """SC-NeuroCore cells, rates and RNG rewind via fresh public pool construction."""
    if not SC_NEUROCORE_AVAILABLE:
        with pytest.raises(RuntimeError):
            owner.SNNControllerWrapper(n_neurons=3, tau_window=4)
        return
    controller = owner.SNNControllerWrapper(n_neurons=3, tau_window=4)
    assert controller.backend == "sc_neurocore"
    first = [controller.step(0.3, 1e-4) for _ in range(50)]
    controller.reset()
    repeated = [controller.step(0.3, 1e-4) for _ in range(50)]
    fresh = owner.SNNControllerWrapper(n_neurons=3, tau_window=4)
    independent = [fresh.step(0.3, 1e-4) for _ in range(50)]
    assert first == repeated == independent
    assert all(np.isfinite(first))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1", 10**400])
def test_PID_constructor_refuses_nonfinite_nonnumeric_gains(value: object) -> None:
    """The real public constructor refuses invalid declared scalar domains."""
    with pytest.raises(ValueError):
        owner.PIDController(kp=cast(float, value))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_real_integer_dimensions_refuse_truncation(value: object) -> None:
    """Prediction/population/window dimensions reject zero, bool and fractional counts."""
    operations: list[Callable[[], object]] = [
        lambda: owner.MPCController(horizon=cast(int, value)),
        lambda: owner.MPCController(iterations=cast(int, value)),
        lambda: owner.SNNControllerWrapper(n_neurons=cast(int, value)),
        lambda: owner.SNNControllerWrapper(tau_window=cast(int, value)),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()


def test_actual_PID_law_saturation_reset_and_refusal_state() -> None:
    """Time-scaled PID keeps its original law and integral rollback on saturation."""
    p = owner.PIDController(kp=2, ki=3, kd=4, u_max=10)
    assert p.step(1, 0.1) == pytest.approx(2.3)
    assert p.step(2, 0.1) == 10.0
    assert p.step(2, 0.1) == pytest.approx(4.9)
    with pytest.raises(ValueError):
        p.step(float("nan"), 0.1)
    assert p.step(2, 0.1) == pytest.approx(5.5)
    p.reset()
    assert p.step(1, 0.1) == pytest.approx(2.3)
    with pytest.raises(ValueError):
        p.step(1, 0.0)
    with pytest.raises(ValueError):
        owner.PIDController(kp=1e308).step(1e308, 1)


def test_actual_MPC_zero_cost_history_clipping_and_overflow_refusal() -> None:
    """Real MPC handles zero position weight, derivative history and invalid predictions."""
    p = owner.MPCController(q_weight=0, horizon=2, iterations=1)
    assert p.step(0.1, 0.001) == p.step(0.2, 0.001) == 0.0
    p.reset()
    assert p.step(0.2, 0.001) == 0.0
    clipped = owner.MPCController(horizon=1, iterations=1, q_weight=1e8, learning_rate=1, u_max=0.2)
    assert clipped.step(1, 0.01) == 0.2
    with pytest.raises(ValueError):
        owner.MPCController(q_weight=-1)
    with pytest.raises(ValueError):
        owner.MPCController(gamma_growth=1e308)
    with pytest.raises(ValueError):
        owner.MPCController().step(1e308, 1)
    with pytest.raises(ValueError):
        owner.MPCController().step(0.1, 0)
    with pytest.raises(ValueError):
        owner.SNNControllerWrapper(seed=-1)


def test_actual_neural_error_and_clock_domains() -> None:
    """The real optional provider exposes either its authored absence or input refusals."""
    if not SC_NEUROCORE_AVAILABLE:
        with pytest.raises(RuntimeError):
            owner.SNNControllerWrapper()
        return
    controller = owner.SNNControllerWrapper(n_neurons=2)
    with pytest.raises(ValueError):
        controller.step(0.1, 0)
    with pytest.raises(ValueError):
        controller.step(float("nan"), 0.001)


def test_original_MPC_keywords_and_mutable_cost_attributes_drive_actual_control() -> None:
    """Original positional/keyword cost declarations and later mutations alter the real action law."""
    controller = owner.MPCController(horizon=1, q_weight=25.0, r_weight=0.1, iterations=2, learning_rate=0.5, u_max=1.0)
    assert controller.q_weight == 25.0 and controller.r_weight == 0.1 and controller.iterations == 2
    assert controller.step(1.0, 0.01) == pytest.approx(0.00475)
    positional = owner.MPCController(100.0, 1, 25.0, 0.1, 2, 0.5, 1.0)
    assert positional.step(1.0, 0.01) == controller.step(1.0, 0.01)
    controller.q_weight = 0.0
    assert controller.step(2.0, 0.01) == 0.0
    controller.q_weight, controller.iterations = 25.0, 1
    controller.reset()
    assert controller.step(1.0, 0.01) == pytest.approx(0.0025)
    controller.iterations, controller.r_weight = 2, 10.0
    controller.reset()
    assert controller.step(1.0, 0.01) == pytest.approx(-0.02)
