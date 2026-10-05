# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary response identification tests
"""Exercise candidate matrix construction and the public failure boundary."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control._typing import FloatArray
from scpn_control.control import free_boundary_tracking_control_law as control_law
from scpn_control.control.free_boundary_response_identification import identify_coil_response
from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController
from scpn_control.control.state_estimator import ExtendedKalmanFilter
from scpn_control.core.fusion_kernel import CoilSet


def test_identify_coil_response_matches_linear_plant() -> None:
    """A complete candidate reproduces the exact linear response columns."""
    actual = np.array([[2.0, -0.5], [0.25, 3.0]], dtype=np.float64)
    currents: FloatArray = np.zeros(2, dtype=np.float64)

    def set_currents(value: FloatArray) -> None:
        nonlocal currents
        currents = value.copy()

    def observe() -> FloatArray:
        return actual @ currents

    candidate = identify_coil_response(
        original_currents=currents.copy(),
        current_limits=np.ones(2, dtype=np.float64),
        perturbation=0.2,
        n_observations=2,
        set_currents=set_currents,
        solve=lambda: None,
        observe=observe,
    )
    np.testing.assert_allclose(candidate, actual, atol=1e-14)


def test_identify_coil_response_refuses_nonfinite_observation() -> None:
    """A nonfinite observation cannot produce a published response column."""
    with pytest.raises(ValueError, match="nonfinite"):
        identify_coil_response(
            original_currents=np.zeros(1, dtype=np.float64),
            current_limits=np.ones(1, dtype=np.float64),
            perturbation=0.2,
            n_observations=1,
            set_currents=lambda _currents: None,
            solve=lambda: None,
            observe=lambda: np.array([float("nan")], dtype=np.float64),
        )


def test_identify_coil_response_refuses_finite_observation_overflow() -> None:
    """Opposite finite observations cannot overflow into an admitted column."""
    observations = [
        np.array([1e308], dtype=np.float64),
        np.array([-1e308], dtype=np.float64),
    ]
    with pytest.raises(ValueError, match="nonfinite matrix column"):
        identify_coil_response(
            original_currents=np.zeros(1, dtype=np.float64),
            current_limits=np.ones(1, dtype=np.float64),
            perturbation=0.2,
            n_observations=1,
            set_currents=lambda _currents: None,
            solve=lambda: None,
            observe=lambda: observations.pop(0),
        )


def test_identify_coil_response_zero_range_and_wrong_width() -> None:
    """A clamped coil yields zero response; a malformed objective is refused."""
    zero_range = identify_coil_response(
        original_currents=np.array([1.0], dtype=np.float64),
        current_limits=np.ones(1, dtype=np.float64),
        perturbation=1e-14,
        n_observations=1,
        set_currents=lambda _currents: None,
        solve=lambda: None,
        observe=lambda: np.array([1.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(zero_range, np.zeros((1, 1), dtype=np.float64))

    with pytest.raises(ValueError, match="unexpected objective width"):
        identify_coil_response(
            original_currents=np.zeros(1, dtype=np.float64),
            current_limits=np.ones(1, dtype=np.float64),
            perturbation=0.2,
            n_observations=1,
            set_currents=lambda _currents: None,
            solve=lambda: None,
            observe=lambda: np.array([1.0, 2.0], dtype=np.float64),
        )


def test_identify_coil_response_refuses_overflow_before_setting_current() -> None:
    """Finite but overflowing perturbations never reach the plant callback."""
    applied: list[FloatArray] = []
    with pytest.raises(ValueError, match="nonfinite current"):
        identify_coil_response(
            original_currents=np.array([1e308], dtype=np.float64),
            current_limits=np.array([float("inf")], dtype=np.float64),
            perturbation=1e308,
            n_observations=1,
            set_currents=applied.append,
            solve=lambda: None,
            observe=lambda: np.zeros(1, dtype=np.float64),
        )
    assert applied == []


class _LinearIdentificationKernel:
    """One-coil deterministic plant for public controller failure injection."""

    def __init__(self, config_file: str) -> None:
        del config_file
        self.solve_calls = 0
        self.fail_at: int | None = None
        self.report_failure_at: int | None = None
        self.report_converged = True
        self.cfg: dict[str, object] = {"coils": [{"current": 0.0}, {"current": 0.0}], "free_boundary": {}}
        self.Psi = np.zeros((2, 2), dtype=np.float64)
        self._flux = 0.0

    def build_coilset_from_config(self) -> CoilSet:
        """Build two bounded coils and one shape-flux objective."""
        return CoilSet(
            positions=[(1.0, 1.0), (2.0, 1.0)],
            currents=np.zeros(2, dtype=np.float64),
            turns=[1, 1],
            current_limits=np.ones(2, dtype=np.float64),
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
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        """Calculate flux and optionally raise after updating simulated state."""
        del boundary_variant, max_outer_iter, tol, optimize_shape, tikhonov_alpha
        self.solve_calls += 1
        assert coils is not None
        self._flux = 2.0 * float(coils.currents[0]) + 3.0 * float(coils.currents[1])
        if self.solve_calls == self.fail_at:
            raise RuntimeError("injected identification solve failure")
        return {"converged": self.report_converged and self.solve_calls != self.report_failure_at}

    def _sample_flux_at_points(self, points: FloatArray) -> FloatArray:
        """Read the single shape-flux objective from the solved plant."""
        assert points.shape == (1, 2)
        return np.array([self._flux], dtype=np.float64)


def test_public_identification_failure_restores_controller_state() -> None:
    """A failed coil perturbation cannot publish a partial response matrix."""
    estimator = ExtendedKalmanFilter(
        x0=np.zeros(6, dtype=np.float64),
        P0=np.eye(6, dtype=np.float64),
        Q=np.eye(6, dtype=np.float64) * 0.01,
        R_cov=np.eye(4, dtype=np.float64),
    )
    controller = FreeBoundaryTrackingController(
        "dummy.json", kernel_factory=_LinearIdentificationKernel, verbose=False, state_estimator=estimator
    )
    kernel = controller.kernel
    original_currents = controller.coils.currents.copy()
    original_matrix = controller.response_matrix.copy()
    original_diagnostics = (controller.response_rank, controller.response_condition_number)
    original_actuators = controller._snapshot_actuator_states()
    original_estimator = (estimator.x.copy(), estimator.P.copy(), estimator.H.copy())
    kernel.fail_at = kernel.solve_calls + 5

    with pytest.raises(RuntimeError, match="injected identification solve failure"):
        controller.identify_response_matrix()

    np.testing.assert_array_equal(controller.coils.currents, original_currents)
    np.testing.assert_array_equal(controller.response_matrix, original_matrix)
    assert (controller.response_rank, controller.response_condition_number) == original_diagnostics
    assert controller._snapshot_actuator_states() == original_actuators
    assert [item["current"] for item in kernel.cfg["coils"]] == list(original_currents)
    assert kernel._flux == 0.0
    for actual, original in zip((estimator.x, estimator.P, estimator.H), original_estimator):
        np.testing.assert_array_equal(actual, original)


def test_public_identification_refuses_failed_solver_status() -> None:
    """A solver-reported failure cannot admit a response matrix."""
    controller = FreeBoundaryTrackingController("dummy.json", kernel_factory=_LinearIdentificationKernel, verbose=False)
    controller.kernel.report_converged = False
    with pytest.raises(RuntimeError, match="requires a converged solve"):
        controller.identify_response_matrix()
    np.testing.assert_array_equal(controller.response_matrix, np.zeros_like(controller.response_matrix))


def test_public_identification_diagnostic_failure_keeps_prior_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post-identification SVD failure cannot publish an unchecked matrix."""
    controller = FreeBoundaryTrackingController("dummy.json", kernel_factory=_LinearIdentificationKernel, verbose=False)
    prior_matrix = controller.response_matrix.copy()
    prior_diagnostics = (
        controller.response_rank,
        controller.response_condition_number,
        controller.response_max_singular_value,
        controller.response_degenerate,
    )

    def fail_diagnostics(_candidate: FloatArray) -> None:
        raise np.linalg.LinAlgError("injected SVD failure")

    monkeypatch.setattr(control_law, "compute_response_diagnostics", fail_diagnostics)
    with pytest.raises(np.linalg.LinAlgError, match="injected SVD failure"):
        controller.identify_response_matrix()

    np.testing.assert_array_equal(controller.response_matrix, prior_matrix)
    assert (
        controller.response_rank,
        controller.response_condition_number,
        controller.response_max_singular_value,
        controller.response_degenerate,
    ) == prior_diagnostics
    np.testing.assert_array_equal(controller.coils.currents, np.zeros(2, dtype=np.float64))
    assert controller.kernel._flux == 0.0


def test_public_tracking_refuses_failed_initial_solve() -> None:
    """A failed initial equilibrium cannot yield a tracking summary."""
    controller = FreeBoundaryTrackingController("dummy.json", kernel_factory=_LinearIdentificationKernel, verbose=False)
    controller.kernel.report_converged = False
    with pytest.raises(RuntimeError, match="tracking requires a converged solve"):
        controller.run_tracking_shot(shot_steps=1)
    assert controller.history["t"] == []


@pytest.mark.parametrize("failure_kind", ["exception", "reported"])
def test_public_tracking_trial_failure_restores_commanded_state(failure_kind: str) -> None:
    """A failed trial solve cannot retain the trial current or record a step."""
    controller = FreeBoundaryTrackingController("dummy.json", kernel_factory=_LinearIdentificationKernel, verbose=False)
    controller.target_vector[0] = 1.0
    kernel = controller.kernel
    failure_call = kernel.solve_calls + 9  # initial, step, identification (six solves), trial
    if failure_kind == "exception":
        kernel.fail_at = failure_call
    else:
        kernel.report_failure_at = failure_call
    original_currents = controller.coils.currents.copy()
    original_actuators = controller._snapshot_actuator_states()

    with pytest.raises(RuntimeError, match="solve failure|tracking requires a converged solve"):
        controller.run_tracking_shot(shot_steps=1)

    assert kernel.solve_calls == failure_call + 1  # accepted baseline recovery
    np.testing.assert_array_equal(controller.coils.currents, original_currents)
    assert controller._snapshot_actuator_states() == original_actuators
    assert [item["current"] for item in kernel.cfg["coils"]] == list(original_currents)
    assert kernel._flux == 0.0
    assert controller.history["t"] == []
