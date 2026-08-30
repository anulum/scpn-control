# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — normalized DGKF controller tests.

"""Public-surface tests for normalized DGKF synthesis and runtime."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from scpn_control.control.h_infinity_controller import (
    HInfinityController,
    get_flight_sim_controller,
    get_radial_robust_controller,
)


def _normalized_scalar_plant() -> dict[str, np.ndarray]:
    """Return a small unstable normalized standard plant."""
    return {
        "A": np.array([[0.0, 1.0], [1.0, -1.0]]),
        "B1": np.array([[0.0, 0.0], [0.5, 0.0]]),
        "B2": np.array([[0.0], [1.0]]),
        "C1": np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "C2": np.array([[1.0, 0.0]]),
        "D12": np.array([[0.0], [0.0], [1.0]]),
        "D21": np.array([[0.0, 1.0]]),
    }


def _controller(*, gamma: float | None = None) -> HInfinityController:
    return HInfinityController(**_normalized_scalar_plant(), gamma=gamma)


def test_constructs_exact_central_dgkf_realization() -> None:
    """The public matrices match Doyle et al. Theorem 3 exactly."""
    controller = _controller(gamma=5.0)
    gamma_squared = controller.gamma**2
    expected_f = -controller.B2.T @ controller.X
    expected_l = -controller.Y @ controller.C2.T
    expected_z = np.linalg.solve(
        np.eye(controller.n) - controller.Y @ controller.X / gamma_squared,
        np.eye(controller.n),
    )
    expected_ak = (
        controller.A
        + controller.B1 @ controller.B1.T @ controller.X / gamma_squared
        + controller.B2 @ expected_f
        + expected_z @ expected_l @ controller.C2
    )
    np.testing.assert_allclose(controller.F, expected_f, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(controller.L, expected_l, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(controller.Z, expected_z, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(controller.Ak, expected_ak, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(controller.Bk, -expected_z @ expected_l, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(controller.Ck, expected_f, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(controller.Dk, np.zeros((1, 1)))


def test_normalization_and_riccati_residuals_are_small() -> None:
    """Normalized identities are exact and both ARE residuals are negligible."""
    controller = _controller(gamma=5.0)
    assert max(controller.normalization_residual_norms()) == 0.0
    residual_x, residual_y = controller.riccati_residual_norms()
    scale_x = 1.0 + np.linalg.norm(controller.C1.T @ controller.C1, ord="fro")
    scale_y = 1.0 + np.linalg.norm(controller.B1 @ controller.B1.T, ord="fro")
    assert residual_x / scale_x < 1.0e-10
    assert residual_y / scale_y < 1.0e-10


def test_closed_loop_realization_matches_independent_block_assembly() -> None:
    """The returned w-to-z interconnection matches an independent block assembly."""
    controller = _controller(gamma=5.0)
    state, disturbance, performance, feedthrough = controller.closed_loop_realization()
    expected_state = np.block(
        [
            [controller.A, controller.B2 @ controller.Ck],
            [controller.Bk @ controller.C2, controller.Ak],
        ]
    )
    np.testing.assert_allclose(state, expected_state)
    np.testing.assert_allclose(disturbance, np.vstack((controller.B1, controller.Bk @ controller.D21)))
    np.testing.assert_allclose(performance, np.hstack((controller.C1, controller.D12 @ controller.Ck)))
    np.testing.assert_array_equal(feedthrough, np.zeros((controller.q, controller.p)))
    assert np.max(np.real(np.linalg.eigvals(state))) < 0.0
    assert controller.is_stable


def test_gamma_search_returns_strictly_feasible_near_infimum() -> None:
    """Automatic gamma search returns a strict feasible point and rejects a bad one."""
    controller = _controller()
    assert controller.gamma < controller._GAMMA_SEARCH_MAX
    assert controller.robust_feasibility_margin() > 0.0
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        HInfinityController(**_normalized_scalar_plant(), gamma=0.5)


def test_feasibility_bypass_is_deprecated_and_ineffective() -> None:
    """The historical bypass flag cannot create an infeasible controller."""
    with pytest.warns(DeprecationWarning, match="always fails closed"):
        controller = HInfinityController(
            **_normalized_scalar_plant(),
            gamma=5.0,
            enforce_robust_feasibility=False,
        )
    assert controller.robust_feasible


@pytest.mark.parametrize("missing", ["D12", "D21"])
def test_explicit_feedthrough_is_required(missing: str) -> None:
    """Neither normalized feedthrough matrix may be invented silently."""
    plant = _normalized_scalar_plant()
    plant.pop(missing)
    with pytest.raises(ValueError, match="D12 and D21 are required"):
        HInfinityController(**plant)


@pytest.mark.parametrize(
    ("matrix", "replacement", "match"),
    [
        ("D12", np.array([[1.0], [0.0], [0.0]]), r"D12.T @ C1"),
        ("D12", np.array([[0.0], [0.0], [2.0]]), r"D12.T @ D12"),
        ("D21", np.array([[1.0, 0.0]]), r"B1 @ D21.T"),
        ("D21", np.array([[0.0, 2.0]]), r"D21 @ D21.T"),
    ],
)
def test_non_normalized_feedthrough_fails_closed(
    matrix: str,
    replacement: np.ndarray,
    match: str,
) -> None:
    """Every normalization identity is a fail-closed public contract."""
    plant = _normalized_scalar_plant()
    plant[matrix] = replacement
    with pytest.raises(ValueError, match=match):
        HInfinityController(**plant)


def test_shape_and_finite_domains_fail_closed() -> None:
    """Malformed and non-finite standard-plant matrices are rejected."""
    plant = _normalized_scalar_plant()
    with pytest.raises(ValueError, match="A must be a finite square"):
        HInfinityController(**{**plant, "A": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="B1 row count"):
        HInfinityController(**{**plant, "B1": np.zeros((3, 2))})
    with pytest.raises(ValueError, match="C2 column count"):
        HInfinityController(**{**plant, "C2": np.zeros((1, 3))})
    with pytest.raises(ValueError, match="D12 must have shape"):
        HInfinityController(**{**plant, "D12": np.zeros((2, 1))})
    bad_a = plant["A"].copy()
    bad_a[0, 0] = np.nan
    with pytest.raises(ValueError, match="A must contain only finite"):
        HInfinityController(**{**plant, "A": bad_a})


def test_stabilizability_and_detectability_fail_closed() -> None:
    """Unstable uncontrollable or unobservable modes cannot enter synthesis."""
    stable_channels = {
        "A": np.diag([1.0, -1.0]),
        "B1": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "B2": np.array([[0.0], [1.0]]),
        "C1": np.array([[0.0, 1.0], [0.0, 0.0]]),
        "C2": np.array([[0.0, 1.0]]),
        "D12": np.array([[0.0], [1.0]]),
        "D21": np.array([[0.0, 1.0]]),
    }
    with pytest.raises(ValueError, match=r"\(A, B1\) must be stabilizable"):
        HInfinityController(**stable_channels)

    plant = _normalized_scalar_plant()
    plant["C2"] = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match=r"\(C2, A\) must be detectable"):
        HInfinityController(**plant)


def test_one_dimensional_b_and_c_inputs_are_oriented() -> None:
    """Unambiguous one-dimensional inputs preserve the compatible convenience."""
    plant = _normalized_scalar_plant()
    controller = HInfinityController(
        A=plant["A"],
        B1=plant["B1"],
        B2=np.array([0.0, 1.0]),
        C1=plant["C1"],
        C2=np.array([1.0, 0.0]),
        D12=plant["D12"],
        D21=plant["D21"],
        gamma=5.0,
    )
    assert controller.B2.shape == (2, 1)
    assert controller.C2.shape == (1, 2)


def test_step_matches_independent_exact_zoh_and_updates_after_output() -> None:
    """Sampled runtime matches an independent exponential and has no fake delay."""
    controller = _controller(gamma=5.0)
    dt = 0.01
    measurement = np.array([0.25])
    augmented = np.zeros((controller.n + controller.l, controller.n + controller.l))
    augmented[: controller.n, : controller.n] = controller.Ak * dt
    augmented[: controller.n, controller.n :] = controller.Bk * dt
    exact = expm(augmented)
    expected_state = exact[: controller.n, controller.n :] @ measurement

    assert controller.step(measurement, dt) == 0.0
    np.testing.assert_allclose(controller.state, expected_state, rtol=1e-13, atol=1e-13)
    expected_control = float((controller.Ck @ expected_state).item())
    assert controller.step(measurement, dt) == pytest.approx(expected_control)


def test_step_validation_saturation_reset_and_dt_cache() -> None:
    """Runtime domains, clipping, cache replacement, and reset are effective."""
    controller = _controller(gamma=5.0)
    with pytest.raises(ValueError, match="error must have shape"):
        controller.step([1.0, 2.0], 0.01)
    with pytest.raises(ValueError, match="error must contain only finite"):
        controller.step(np.nan, 0.01)
    with pytest.raises(ValueError, match="dt must be > 0"):
        controller.step(0.0, 0.0)
    controller.u_max = 0.1
    controller.state = np.full(controller.n, 1.0e6)
    assert abs(float(controller.step(0.0, 0.01))) == pytest.approx(0.1)
    assert controller._cached_dt == 0.01
    controller.step(0.0, 0.02)
    assert controller._cached_dt == 0.02
    controller.reset()
    np.testing.assert_array_equal(controller.state, np.zeros(controller.n))
    controller.u_max = float("nan")
    with pytest.raises(ValueError, match="u_max"):
        controller.step(0.0, 0.01)


def test_mimo_runtime_returns_vector() -> None:
    """A normalized MIMO controller returns one command per control channel."""
    controller = HInfinityController(
        A=np.array([[-1.0]]),
        B1=np.array([[1.0, 0.0, 0.0]]),
        B2=np.array([[1.0, 1.0]]),
        C1=np.array([[1.0], [0.0], [0.0]]),
        C2=np.array([[1.0], [1.0]]),
        D12=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        D21=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        gamma=5.0,
    )
    output = controller.step(np.array([1.0, -1.0]), 0.01)
    assert isinstance(output, np.ndarray)
    assert output.shape == (2,)


def test_legacy_margin_name_warns_and_does_not_claim_bode_margin() -> None:
    """The legacy margin property is explicitly diagnostic and warning-emitting."""
    controller = _controller(gamma=5.0)
    with pytest.warns(DeprecationWarning, match="not a classical gain margin"):
        assert controller.gain_margin_db == controller.stability_margin_db
    controller.closed_loop_eigenvalues = np.array([1.0 + 0.0j])
    assert controller.is_stable is False
    assert controller.stability_margin_db == 0.0


def test_stable_open_loop_has_infinite_legacy_diagnostic() -> None:
    """The legacy pole-displacement ratio preserves its stable-plant sentinel."""
    plant = _normalized_scalar_plant()
    plant["A"] = np.array([[-2.0, 1.0], [0.0, -1.0]])
    controller = HInfinityController(**plant, gamma=5.0)
    assert controller.stability_margin_db == float("inf")


def test_factories_are_normalized_and_validate_parameters() -> None:
    """Both reduced factories satisfy normalization and validate physical inputs."""
    for controller in (get_radial_robust_controller(gamma_growth=10.0), get_flight_sim_controller()):
        assert controller.is_stable
        assert max(controller.normalization_residual_norms()) == 0.0
    with pytest.raises(ValueError, match="gamma_growth"):
        get_radial_robust_controller(gamma_growth=0.0)
    with pytest.raises(ValueError, match="damping"):
        get_radial_robust_controller(damping=0.0)
    with pytest.raises(ValueError, match="response_gain"):
        get_flight_sim_controller(response_gain=0.0)
    with pytest.raises(ValueError, match="actuator_tau"):
        get_flight_sim_controller(actuator_tau=0.0)


def test_warning_filters_do_not_hide_explicit_compatibility_warning() -> None:
    """The feasibility-bypass compatibility warning remains observable."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HInfinityController(
            **_normalized_scalar_plant(),
            gamma=5.0,
            enforce_robust_feasibility=False,
        )
    assert any(item.category is DeprecationWarning for item in caught)
