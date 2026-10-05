# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for FB tracking observation vectors

"""Drive production free-boundary tracking observation vector builders."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scpn_control.control.free_boundary_tracking_observation as obs
from scpn_control._typing import FloatArray
from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController
from scpn_control.control.state_estimator import ExtendedKalmanFilter
from scpn_control.core.fusion_kernel import CoilSet


def _coilset_with_shape_and_xpoint() -> CoilSet:
    return CoilSet(
        positions=[(3.0, 2.0), (5.0, -2.0)],
        currents=np.array([1.0e4, -1.0e4]),
        turns=[10, 10],
        target_flux_points=np.array([[3.5, 0.0], [4.0, 0.5], [4.5, -0.5]], dtype=float),
        target_flux_values=np.array([0.10, 0.20, 0.15], dtype=float),
        x_point_target=np.array([4.0, -1.5], dtype=float),
        x_point_flux_target=0.12,
        divertor_strike_points=np.array([[3.2, -2.5], [4.8, -2.5]], dtype=float),
        divertor_flux_values=np.array([0.11, 0.11], dtype=float),
    )


def test_build_target_vector_blocks_and_values() -> None:
    """Target vector stacks shape, X-point, and divertor blocks in order."""
    coils = _coilset_with_shape_and_xpoint()
    target, blocks = obs.build_target_vector(coils)
    names = [b.name for b in blocks]
    assert names == ["shape_flux", "x_point_position", "x_point_flux", "divertor_flux"]
    assert target.shape == (3 + 2 + 1 + 2,)
    np.testing.assert_allclose(target[:3], [0.10, 0.20, 0.15])
    np.testing.assert_allclose(target[3:5], [4.0, -1.5])
    assert target[5] == pytest.approx(0.12)
    np.testing.assert_allclose(target[6:], [0.11, 0.11])


def test_build_target_vector_fail_closed_x_point_flux_without_target() -> None:
    """x_point_flux_target without position target fails closed."""
    coils = CoilSet(x_point_flux_target=0.1)
    with pytest.raises(ValueError, match="x_point_flux_target requires x_point_target"):
        obs.build_target_vector(coils)


def test_require_finite_observation_checks_width_and_values() -> None:
    """The observation leaf rejects malformed and nonfinite objective vectors."""
    valid = np.array([1.0, 2.0], dtype=np.float64)
    np.testing.assert_array_equal(obs.require_finite_observation(valid, width=2, name="true"), valid)
    with pytest.raises(ValueError, match="width"):
        obs.require_finite_observation(valid, width=1, name="true")
    with pytest.raises(ValueError, match="nonfinite"):
        obs.require_finite_observation(np.array([float("nan")]), width=1, name="true")


def test_resolve_measurement_vector_success_and_fail_closed() -> None:
    """Measurement vector resolution supports scalar broadcast and rejects bad input."""
    coils = _coilset_with_shape_and_xpoint()
    target, blocks = obs.build_target_vector(coils)
    zero = obs.resolve_measurement_vector(None, objective_blocks=blocks, target_size=target.size, name="m")
    np.testing.assert_allclose(zero, 0.0)
    broadcast = obs.resolve_measurement_vector(
        {"shape_flux": 0.01},
        objective_blocks=blocks,
        target_size=target.size,
        name="m",
    )
    np.testing.assert_allclose(broadcast[:3], 0.01)
    with pytest.raises(ValueError, match="mapping"):
        obs.resolve_measurement_vector("bad", objective_blocks=blocks, target_size=target.size, name="m")
    with pytest.raises(ValueError, match="Unknown"):
        obs.resolve_measurement_vector({"bogus": 1.0}, objective_blocks=blocks, target_size=target.size, name="m")
    with pytest.raises(ValueError, match="finite"):
        obs.resolve_measurement_vector(
            {"shape_flux": float("inf")},
            objective_blocks=blocks,
            target_size=target.size,
            name="m",
        )


def test_control_weights_and_measurement_offset() -> None:
    """Control weights follow tolerances; measurement offset combines channels."""
    coils = _coilset_with_shape_and_xpoint()
    target, blocks = obs.build_target_vector(coils)
    weights = obs.build_control_objective_weights(
        target.size,
        blocks,
        {"shape_rms": 0.1, "x_point_position": 0.05},
    )
    assert weights.shape == target.shape
    assert float(weights[0]) == pytest.approx(1.0 / 0.1)
    assert float(weights[3]) == pytest.approx(1.0 / 0.05)
    bias = np.ones_like(target)
    drift = np.full_like(target, 0.5)
    corr_bias = np.full_like(target, 0.25)
    corr_drift = np.full_like(target, 0.1)
    offset = obs.current_measurement_offset(bias, drift, corr_bias, corr_drift)
    np.testing.assert_allclose(offset, 1.0 + 0.5 - 0.25 - 0.1)


class _ObservationKernel:
    """One-coil plant for a public delayed-observation failure path."""

    def __init__(self, config_file: str) -> None:
        del config_file
        self.cfg: dict[str, Any] = {
            "coils": [{"current": 0.0}],
            "free_boundary": {},
            "free_boundary_tracking": {"measurement_latency_steps": 1, "latency_compensation_gain": 1.0},
        }
        self.Psi = np.zeros((2, 2), dtype=np.float64)
        self.emit_nan = False
        self.emit_large = False
        self._flux = 0.0

    def build_coilset_from_config(self) -> CoilSet:
        """Declare one bounded coil and one shape-flux target."""
        return CoilSet(
            positions=[(1.0, 1.0)],
            currents=np.zeros(1, dtype=np.float64),
            turns=[1],
            current_limits=np.ones(1, dtype=np.float64),
            target_flux_points=np.array([[1.0, 0.0]], dtype=np.float64),
            target_flux_values=np.zeros(1, dtype=np.float64),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
    ) -> dict[str, bool]:
        """Update the one-coil flux and report a completed simulated solve."""
        del boundary_variant, max_outer_iter, tol, optimize_shape
        assert coils is not None
        self._flux = float(coils.currents[0])
        return {"converged": True}

    def _sample_flux_at_points(
        self, points: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Expose an injected nonfinite or large sample after a valid step."""
        assert points.shape == (1, 2)
        value = float("nan") if self.emit_nan else (1e308 if self.emit_large else self._flux)
        return np.array([value], dtype=np.float64)


@pytest.mark.parametrize("fault", ["nonfinite_true", "overflow_measured"])
def test_public_shot_refuses_invalid_late_observation_without_latency_poison(fault: str) -> None:
    """Latency cannot hide a bad true or measured observation from the shot."""
    estimator = ExtendedKalmanFilter(
        x0=np.zeros(6, dtype=np.float64),
        P0=np.eye(6, dtype=np.float64),
        Q=np.eye(6, dtype=np.float64) * 0.01,
        R_cov=np.eye(4, dtype=np.float64),
    )
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_ObservationKernel,
        verbose=False,
        response_refresh_steps=2,
        state_estimator=estimator,
    )
    before_fault: list[tuple[FloatArray, FloatArray, FloatArray]] = []

    def corrupt_second_step(kernel: Any, _coils: CoilSet, step: int) -> None:
        if step == 1:
            before_fault.append((estimator.x.copy(), estimator.P.copy(), estimator.H.copy()))
            if fault == "nonfinite_true":
                kernel.emit_nan = True
            else:
                kernel.emit_large = True
                controller.measurement_bias_vector[:] = 1e308

    with pytest.raises(ValueError, match="nonfinite"):
        controller.run_tracking_shot(shot_steps=2, disturbance_callback=corrupt_second_step)

    assert controller.history["t"] == [0]
    assert len(before_fault) == 1
    for actual, expected in zip((estimator.x, estimator.P, estimator.H), before_fault[0]):
        np.testing.assert_array_equal(actual, expected)
    assert all(np.isfinite(value).all() for value in controller._measurement_latency_buffer)
    assert np.isfinite(controller.objective_rate_estimate).all()


def test_public_shot_restores_latency_state_after_projection_overflow() -> None:
    """A failed effective observation cannot retain an invalid latency update."""
    controller = FreeBoundaryTrackingController("dummy.json", kernel_factory=_ObservationKernel, verbose=False)

    def inject_large_sample(kernel: Any, _coils: CoilSet, step: int) -> None:
        if step == 0:
            kernel.emit_large = True

    with pytest.raises(ValueError, match="effective observation contains nonfinite"):
        controller.run_tracking_shot(shot_steps=1, disturbance_callback=inject_large_sample)

    assert controller.history["t"] == []
    assert len(controller._measurement_latency_buffer) == 2
    np.testing.assert_array_equal(controller._measurement_latency_buffer[0], np.zeros(1, dtype=np.float64))
    np.testing.assert_array_equal(controller._measurement_latency_buffer[1], np.array([1e308]))
    np.testing.assert_array_equal(controller.objective_rate_estimate, np.zeros(1, dtype=np.float64))
    np.testing.assert_array_equal(controller._last_delayed_measurement, np.zeros(1, dtype=np.float64))
