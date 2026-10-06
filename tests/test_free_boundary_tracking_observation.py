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


class _LatentFaultKernel(_ObservationKernel):
    """One-coil plant with a fault inside one band of coil current.

    Response identification moves the coil by a quarter of its limit in both
    directions and the first correction moves it by less. The band lies between
    the two, so identification completes and the fault shows only when the
    gain search applies the correction.
    """

    FAULT_BAND = (-0.24, -0.01)

    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.sample_fails_in_band = False
        self.solver_fails_after_fault = False
        self.successful_samples = 0
        self._faulted = False

    @property
    def in_band(self) -> bool:
        """Report whether the coil current is inside the fault band."""
        return self.FAULT_BAND[0] < self._flux < self.FAULT_BAND[1]

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
    ) -> dict[str, bool]:
        """Follow the coil and report failed convergence once the fault has shown."""
        result = super().solve(
            boundary_variant=boundary_variant,
            coils=coils,
            max_outer_iter=max_outer_iter,
            tol=tol,
            optimize_shape=optimize_shape,
        )
        if self.solver_fails_after_fault and self._faulted:
            return {"converged": False}
        return result

    def _sample_flux_at_points(
        self, points: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Return a nonfinite sample inside the band when so configured."""
        if self.sample_fails_in_band and self.in_band:
            self._faulted = True
            return np.array([float("nan")], dtype=np.float64)
        self.successful_samples += 1
        return super()._sample_flux_at_points(points)


def _default_estimator(**models: Any) -> ExtendedKalmanFilter:
    """Build the six-state estimator the controller accepts, with optional models."""
    return ExtendedKalmanFilter(
        x0=np.zeros(6, dtype=np.float64),
        P0=np.eye(6, dtype=np.float64),
        Q=np.eye(6, dtype=np.float64) * 0.01,
        R_cov=np.eye(4, dtype=np.float64),
        **models,
    )


def _assert_estimator_saw(estimator: ExtendedKalmanFilter, *, observations: int, dt: float) -> None:
    """Require the state of an identical estimator after that many admitted observations.

    The plant has no X-point objective, so an admitted observation advances the
    estimator by one prediction and applies no measurement update.
    """
    reference = _default_estimator()
    for _ in range(observations):
        reference.predict(dt)
    for actual, expected in zip((estimator.x, estimator.P, estimator.H), (reference.x, reference.P, reference.H)):
        np.testing.assert_array_equal(actual, expected)


def test_estimator_failure_during_gain_search_restores_estimator_and_coil() -> None:
    """A process model that refuses the corrected plant leaves no trace of the failed search.

    The caller's process model raises while the coil current is inside the
    band. The search observation runs without latency bookkeeping, so only the
    estimator has state to restore there, and the step restores it again after
    the baseline recovery solve. The estimator must then hold exactly what the
    admitted observations gave it.
    """
    kernels: list[_LatentFaultKernel] = []

    def make_kernel(config_file: str) -> _LatentFaultKernel:
        """Keep the plant that the controller builds, for the process model to observe."""
        kernels.append(_LatentFaultKernel(config_file))
        return kernels[-1]

    def process_model(x: FloatArray, u: Any, dt: float) -> FloatArray:
        """Advance the default kinematics and refuse the plant inside the band."""
        del u
        if kernels[-1].in_band:
            raise RuntimeError("process model is not valid in this band of coil current")
        predicted = x.copy()
        predicted[0] += dt * x[2]
        predicted[1] += dt * x[3]
        return predicted

    def process_jacobian(x: FloatArray, u: Any, dt: float) -> FloatArray:
        """Return the Jacobian of the default kinematics."""
        del x, u
        jacobian = np.eye(6, dtype=np.float64)
        jacobian[0, 2] = dt
        jacobian[1, 3] = dt
        return jacobian

    estimator = _default_estimator(process_model=process_model, process_jacobian=process_jacobian)
    controller = FreeBoundaryTrackingController(
        "dummy.json", kernel_factory=make_kernel, verbose=False, state_estimator=estimator
    )
    controller.measurement_bias_vector[:] = 0.2
    baseline = controller.coils.currents.copy()

    with pytest.raises(RuntimeError, match="not valid in this band") as caught:
        controller.run_tracking_shot(shot_steps=1)

    assert not getattr(caught.value, "__notes__", [])
    np.testing.assert_array_equal(controller.coils.currents, baseline)
    assert not kernels[-1].in_band
    # The plant was sampled five times: before identification, at its two
    # perturbations, at the baseline the step plans from, and after the
    # correction. Identification restores the estimator when it ends, and the
    # last observation was refused, so the estimator holds two predictions.
    assert kernels[-1].successful_samples == 5
    _assert_estimator_saw(estimator, observations=2, dt=controller.control_dt_s)
    assert controller.history["t"] == []


def test_failed_recovery_solve_is_attached_to_the_original_failure() -> None:
    """The first failure is raised; the failed baseline recovery is recorded on it.

    The flux sample turns nonfinite when the correction moves the coil into the
    band, and the solver stops converging after that. The original refusal must
    reach the caller with the recovery failure as a note, and the coil and the
    estimator must be back at their state before the correction.
    """
    estimator = _default_estimator()
    kernels: list[_LatentFaultKernel] = []

    def make_kernel(config_file: str) -> _LatentFaultKernel:
        """Build the plant with both faults armed."""
        kernel = _LatentFaultKernel(config_file)
        kernel.sample_fails_in_band = True
        kernel.solver_fails_after_fault = True
        kernels.append(kernel)
        return kernel

    controller = FreeBoundaryTrackingController(
        "dummy.json", kernel_factory=make_kernel, verbose=False, state_estimator=estimator
    )
    controller.measurement_bias_vector[:] = 0.2
    baseline = controller.coils.currents.copy()

    with pytest.raises(ValueError, match="nonfinite") as caught:
        controller.run_tracking_shot(shot_steps=1)

    assert getattr(caught.value, "__notes__", []) == [
        "baseline free-boundary recovery solve failed: free-boundary tracking requires a converged solve"
    ]
    np.testing.assert_array_equal(controller.coils.currents, baseline)
    # The fifth sample, after the correction, was the nonfinite one. As above,
    # the estimator holds the two predictions made outside identification.
    assert kernels[-1].successful_samples == 4
    _assert_estimator_saw(estimator, observations=2, dt=controller.control_dt_s)
    assert controller.history["t"] == []
