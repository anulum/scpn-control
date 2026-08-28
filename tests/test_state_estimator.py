# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — State estimator tests.

"""Tests for the Extended Kalman Filter (EKF) state estimator."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.state_estimator import ExtendedKalmanFilter


def test_ekf_zero_noise():
    """Verify that EKF matches true state when noise is zero."""
    x0 = np.array([6.2, 0.0, 0.1, 0.05, 15.0, 10.0])
    P0 = np.eye(6) * 0.1
    Q = np.zeros((6, 6))
    R = np.zeros((4, 4))

    ekf = ExtendedKalmanFilter(x0, P0, Q, R)

    # Predict and Update with perfect measurements
    dt = 0.1
    ekf.predict(dt)

    # True state after dt
    x_true = x0.copy()
    x_true[0] += x0[2] * dt
    x_true[1] += x0[3] * dt

    z = np.array([x_true[0], x_true[1], x_true[4], x_true[5]])
    ekf.update(z)

    np.testing.assert_allclose(ekf.estimate(), x_true, atol=1e-10)


def test_ekf_convergence():
    """Verify that EKF converges to true state under Gaussian noise."""
    rng = np.random.default_rng(42)

    x_true = np.array([6.2, 0.0, 0.0, 0.0, 15.0, 10.0])
    x0 = x_true + 0.5  # Offset initial guess
    P0 = np.eye(6) * 1.0
    Q = np.eye(6) * 0.01
    R_cov = np.eye(4) * 0.1

    ekf = ExtendedKalmanFilter(x0, P0, Q, R_cov)

    # Iterate many steps with noisy measurements
    for _ in range(100):
        ekf.predict(0.1)
        z_noisy = np.array([x_true[0], x_true[1], x_true[4], x_true[5]]) + rng.normal(0, 0.1, 4)
        ekf.update(z_noisy)

    # Check that error is reduced
    err = np.abs(ekf.estimate() - x_true)
    assert np.all(err < 0.2)


def test_ekf_covariance_shrinkage():
    """Verify that uncertainty decreases over time with consistent measurements."""
    x0 = np.zeros(6)
    P0 = np.eye(6) * 10.0
    Q = np.eye(6) * 0.01
    R_cov = np.eye(4) * 1.0

    ekf = ExtendedKalmanFilter(x0, P0, Q, R_cov)

    tr_before = np.trace(ekf.P)

    for _ in range(10):
        ekf.predict(0.1)
        ekf.update(np.zeros(4))

    tr_after = np.trace(ekf.P)
    assert tr_after < tr_before


def test_ekf_covariance_psd():
    """P must remain positive semi-definite after predict + update.

    Simon 2006, "Optimal State Estimation", Ch. 13: the EKF covariance
    update (I − KH)P preserves PSD if the initial P₀ is PSD and Q, R ≥ 0.
    """
    x0 = np.array([6.2, 0.0, 0.1, 0.05, 15.0, 10.0])
    P0 = np.eye(6) * 1.0
    Q = np.eye(6) * 0.01
    R_cov = np.eye(4) * 0.1

    ekf = ExtendedKalmanFilter(x0, P0, Q, R_cov)

    rng = np.random.default_rng(7)
    for _ in range(20):
        ekf.predict(0.05)
        z = np.array([6.2, 0.0, 15.0, 10.0]) + rng.normal(0, 0.1, 4)
        ekf.update(z)

    eigvals = np.linalg.eigvalsh(ekf.P)
    assert np.all(eigvals >= -1e-10), f"P has negative eigenvalue: {eigvals.min()}"


def test_ekf_estimate_converges():
    """Estimation error must decrease as more measurements are assimilated.

    Lister et al. 1997, Nucl. Fusion 37, 1633: real-time magnetic
    reconstruction converges within ~10 ms for JET-scale plasmas.
    """
    rng = np.random.default_rng(13)

    x_true = np.array([6.2, 0.05, 0.0, 0.0, 14.0, 9.5])
    x0 = x_true + np.array([0.5, 0.3, 0.0, 0.0, 1.0, 1.0])
    P0 = np.eye(6) * 2.0
    Q = np.eye(6) * 0.001
    R_cov = np.eye(4) * 0.05

    ekf = ExtendedKalmanFilter(x0, P0, Q, R_cov)

    err_initial = np.linalg.norm(ekf.estimate() - x_true)

    for _ in range(50):
        ekf.predict(0.05)
        z = np.array([x_true[0], x_true[1], x_true[4], x_true[5]]) + rng.normal(0, 0.05, 4)
        ekf.update(z)

    err_final = np.linalg.norm(ekf.estimate() - x_true)
    assert err_final < err_initial


def test_nonlinear_ekf_matches_independent_predict_and_joseph_update():
    """The public API must execute nonlinear f/h models and analytic Jacobians."""
    x0 = np.array([0.4, -0.2])
    p0 = np.array([[0.3, 0.02], [0.02, 0.4]])
    q = np.diag([0.05, 0.08])
    r = np.array([[0.1]])
    control = np.array([0.3])
    dt = 0.2

    def process(x, u, step):
        assert u is not None
        return np.array([x[0] + step * (x[1] + u[0]), x[1] + step * np.sin(x[0])])

    def process_jacobian(x, u, step):
        assert u is not None
        return np.array([[1.0, step], [step * np.cos(x[0]), 1.0]])

    def measurement(x):
        return np.array([x[0] ** 2 + x[1]])

    def measurement_jacobian(x):
        return np.array([[2.0 * x[0], 1.0]])

    def discretise_process_noise(base_q, x, u, step):
        assert u is not None
        return base_q * step**2

    ekf = ExtendedKalmanFilter(
        x0,
        p0,
        q,
        r,
        process_model=process,
        process_jacobian=process_jacobian,
        measurement_model=measurement,
        measurement_jacobian=measurement_jacobian,
        process_noise_model=discretise_process_noise,
    )

    expected_x_pred = process(x0, control, dt)
    expected_f = process_jacobian(x0, control, dt)
    expected_p_pred = expected_f @ p0 @ expected_f.T + q * dt**2
    np.testing.assert_allclose(ekf.predict(dt, control), expected_x_pred)
    np.testing.assert_allclose(ekf.P, expected_p_pred)

    z = np.array([0.25])
    expected_h = measurement_jacobian(expected_x_pred)
    innovation = z - measurement(expected_x_pred)
    innovation_cov = expected_h @ expected_p_pred @ expected_h.T + r
    gain = np.linalg.solve(innovation_cov, expected_h @ expected_p_pred).T
    expected_x = expected_x_pred + gain @ innovation
    identity_minus_kh = np.eye(2) - gain @ expected_h
    expected_p = identity_minus_kh @ expected_p_pred @ identity_minus_kh.T + gain @ r @ gain.T

    np.testing.assert_allclose(ekf.update(z), expected_x)
    np.testing.assert_allclose(ekf.P, expected_p)


def test_nonlinear_ekf_supports_wrapped_residual_and_state_addition():
    """Wrapped measurements and states must use caller-owned manifold operations."""

    def process(x, u, dt):
        assert u is not None
        return x + dt * u

    def process_jacobian(x, u, dt):
        return np.eye(1)

    def measurement(x):
        return x.copy()

    def measurement_jacobian(x):
        return np.eye(1)

    def wrap(values):
        return (values + np.pi) % (2.0 * np.pi) - np.pi

    def residual(observed, predicted):
        return wrap(observed - predicted)

    def state_addition(state, correction):
        return wrap(state + correction)

    ekf = ExtendedKalmanFilter(
        np.array([np.pi - 0.05]),
        np.array([[1.0]]),
        np.array([[0.0]]),
        np.array([[1.0]]),
        process_model=process,
        process_jacobian=process_jacobian,
        measurement_model=measurement,
        measurement_jacobian=measurement_jacobian,
        measurement_residual=residual,
        state_addition=state_addition,
    )

    ekf.predict(0.1, np.array([0.0]))
    updated = ekf.update(np.array([-np.pi + 0.05]))
    np.testing.assert_allclose(updated, np.array([-np.pi]), atol=1e-12)


def test_default_process_noise_is_continuous_time_covariance_density():
    """Without an override, prediction must discretise Q as Q * dt."""
    ekf = ExtendedKalmanFilter(
        np.zeros(6),
        np.zeros((6, 6)),
        np.eye(6) * 2.0,
        np.eye(4),
    )
    ekf.predict(0.25)
    np.testing.assert_allclose(ekf.P, np.eye(6) * 0.5)


def test_estimator_owns_input_and_output_arrays():
    """Caller mutation must not alter internal estimator state or covariance."""
    x0 = np.zeros(6)
    p0 = np.eye(6)
    q = np.eye(6)
    r = np.eye(4)
    ekf = ExtendedKalmanFilter(x0, p0, q, r)
    x0[0] = 10.0
    p0[0, 0] = 10.0
    q[0, 0] = 10.0
    r[0, 0] = 10.0
    estimate = ekf.estimate()
    estimate[0] = 20.0

    assert ekf.x[0] == 0.0
    assert ekf.P[0, 0] == 1.0
    assert ekf.Q[0, 0] == 1.0
    assert ekf.R[0, 0] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"process_model": lambda x, u, dt: x}, "process_model and process_jacobian"),
        ({"process_jacobian": lambda x, u, dt: np.eye(6)}, "process_model and process_jacobian"),
        ({"measurement_model": lambda x: x[:4]}, "measurement_model and measurement_jacobian"),
        ({"measurement_jacobian": lambda x: np.eye(4, 6)}, "measurement_model and measurement_jacobian"),
    ],
)
def test_nonlinear_model_and_jacobian_must_be_supplied_together(kwargs, message):
    """An EKF must never combine a custom nonlinear model with a stale Jacobian."""
    with pytest.raises(ValueError, match=message):
        ExtendedKalmanFilter(np.zeros(6), np.eye(6), np.eye(6), np.eye(4), **kwargs)


@pytest.mark.parametrize(
    ("x0", "p0", "q", "r", "message"),
    [
        (np.zeros((2, 1)), np.eye(2), np.eye(2), np.eye(1), "x0 must be one-dimensional"),
        (np.zeros(6), np.eye(5), np.eye(6), np.eye(4), "P0 must have shape"),
        (np.zeros(6), np.eye(6), np.eye(5), np.eye(4), "Q must have shape"),
        (np.zeros(6), np.eye(6), np.eye(6), np.ones((4, 3)), "R_cov must be square"),
        (np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0]), np.eye(6), np.eye(6), np.eye(4), "x0 must contain only finite"),
        (np.zeros(6), np.array([[1.0, 0.5], [0.0, 1.0]]), np.eye(6), np.eye(4), "P0 must have shape"),
        (np.zeros(2), np.array([[1.0, 0.5], [0.0, 1.0]]), np.eye(2), np.eye(1), "P0 must be symmetric"),
        (np.zeros(2), np.diag([1.0, -0.1]), np.eye(2), np.eye(1), "P0 must be positive semidefinite"),
        (np.zeros(2), np.eye(2), np.array([[np.inf, 0.0], [0.0, 1.0]]), np.eye(1), "Q must contain only finite"),
    ],
)
def test_initial_state_and_covariance_validation(x0, p0, q, r, message):
    """Invalid initial estimation contracts must fail before runtime."""
    custom = x0.size != 6 or p0.shape == (2, 2)
    kwargs = {}
    if custom:
        kwargs = {
            "process_model": lambda x, u, dt: x,
            "process_jacobian": lambda x, u, dt: np.eye(x.size),
            "measurement_model": lambda x: np.array([x[0]]),
            "measurement_jacobian": lambda x: np.eye(1, x.size),
        }
    with pytest.raises(ValueError, match=message):
        ExtendedKalmanFilter(x0, p0, q, r, **kwargs)


def test_empty_state_is_rejected():
    """A zero-dimensional estimator has no valid state contract."""
    with pytest.raises(ValueError, match="x0 must not be empty"):
        ExtendedKalmanFilter(
            np.array([]),
            np.empty((0, 0)),
            np.empty((0, 0)),
            np.eye(1),
            process_model=lambda x, u, dt: x,
            process_jacobian=lambda x, u, dt: np.empty((0, 0)),
            measurement_model=lambda x: np.zeros(1),
            measurement_jacobian=lambda x: np.empty((1, 0)),
        )


def test_custom_state_dimension_requires_a_custom_process_model():
    """The compatible six-state process default cannot be applied generically."""
    with pytest.raises(ValueError, match="default process model requires"):
        ExtendedKalmanFilter(
            np.zeros(1),
            np.eye(1),
            np.eye(1),
            np.eye(1),
            measurement_model=lambda x: x.copy(),
            measurement_jacobian=lambda x: np.eye(1),
        )


def test_custom_dimensions_require_a_custom_measurement_model():
    """The compatible four-diagnostic default cannot be applied generically."""
    with pytest.raises(ValueError, match="default measurement model requires"):
        ExtendedKalmanFilter(
            np.zeros(1),
            np.eye(1),
            np.eye(1),
            np.eye(1),
            process_model=lambda x, u, dt: x.copy(),
            process_jacobian=lambda x, u, dt: np.eye(1),
        )


def test_every_initial_covariance_owns_its_validation_error():
    """Q and R must each enforce symmetry, PSD, and non-empty dimensions."""
    asymmetric_q = np.eye(6)
    asymmetric_q[0, 1] = 0.2
    with pytest.raises(ValueError, match="Q must be symmetric"):
        ExtendedKalmanFilter(np.zeros(6), np.eye(6), asymmetric_q, np.eye(4))

    non_psd_r = np.eye(4)
    non_psd_r[0, 0] = -0.1
    with pytest.raises(ValueError, match="R_cov must be positive semidefinite"):
        ExtendedKalmanFilter(np.zeros(6), np.eye(6), np.eye(6), non_psd_r)

    with pytest.raises(ValueError, match="R_cov must not be empty"):
        ExtendedKalmanFilter(
            np.zeros(1),
            np.eye(1),
            np.eye(1),
            np.empty((0, 0)),
            process_model=lambda x, u, dt: x.copy(),
            process_jacobian=lambda x, u, dt: np.eye(1),
            measurement_model=lambda x: np.empty(0),
            measurement_jacobian=lambda x: np.empty((0, 1)),
        )


def _scalar_ekf(**kwargs):
    defaults = {
        "process_model": lambda x, u, dt: x.copy(),
        "process_jacobian": lambda x, u, dt: np.eye(1),
        "measurement_model": lambda x: x.copy(),
        "measurement_jacobian": lambda x: np.eye(1),
    }
    defaults.update(kwargs)
    return ExtendedKalmanFilter(
        np.array([0.2]),
        np.array([[1.0]]),
        np.array([[0.1]]),
        np.array([[0.2]]),
        **defaults,
    )


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_predict_rejects_nonpositive_or_nonfinite_timestep(dt):
    """Prediction must reject every non-positive or non-finite time step."""
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        _scalar_ekf().predict(dt)


@pytest.mark.parametrize("control", [np.array([[1.0]]), np.array([np.nan])])
def test_predict_rejects_invalid_control_vector(control):
    """Prediction must reject malformed or non-finite control vectors."""
    with pytest.raises(ValueError, match="u must"):
        _scalar_ekf().predict(0.1, control)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"process_model": lambda x, u, dt: np.zeros(2)}, "process_model output must have shape"),
        ({"process_model": lambda x, u, dt: np.array([np.nan])}, "process_model output must contain only finite"),
        ({"process_jacobian": lambda x, u, dt: np.eye(2)}, "process_jacobian output must have shape"),
        (
            {"process_jacobian": lambda x, u, dt: np.array([[np.nan]])},
            "process_jacobian output must contain only finite",
        ),
        ({"process_noise_model": lambda q, x, u, dt: np.eye(2)}, "process_noise_model output must have shape"),
        (
            {"process_noise_model": lambda q, x, u, dt: np.array([[np.nan]])},
            "process_noise_model output must contain only finite",
        ),
        (
            {"process_noise_model": lambda q, x, u, dt: np.array([[-1.0]])},
            "process_noise_model output must be positive semidefinite",
        ),
    ],
)
def test_predict_validates_every_runtime_model_output(override, message):
    """Prediction must fail closed on every invalid callback result."""
    ekf = _scalar_ekf(**override)
    with pytest.raises(ValueError, match=message):
        ekf.predict(0.1)


def test_predict_rejects_asymmetric_runtime_process_noise():
    """A nonlinear noise discretiser must return a covariance, not any matrix."""
    ekf = ExtendedKalmanFilter(
        np.zeros(2),
        np.eye(2),
        np.eye(2),
        np.eye(1),
        process_model=lambda x, u, dt: x.copy(),
        process_jacobian=lambda x, u, dt: np.eye(2),
        measurement_model=lambda x: np.array([x[0]]),
        measurement_jacobian=lambda x: np.array([[1.0, 0.0]]),
        process_noise_model=lambda q, x, u, dt: np.array([[1.0, 0.2], [0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="process_noise_model output must be symmetric"):
        ekf.predict(0.1)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"measurement_model": lambda x: np.zeros(2)}, "measurement_model output must have shape"),
        ({"measurement_model": lambda x: np.array([np.nan])}, "measurement_model output must contain only finite"),
        ({"measurement_jacobian": lambda x: np.eye(2)}, "measurement_jacobian output must have shape"),
        (
            {"measurement_jacobian": lambda x: np.array([[np.nan]])},
            "measurement_jacobian output must contain only finite",
        ),
    ],
)
def test_constructor_validates_measurement_model_outputs(override, message):
    """Measurement callbacks must satisfy dimensions and finiteness at creation."""
    with pytest.raises(ValueError, match=message):
        _scalar_ekf(**override)


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        (np.zeros(2), "z must have shape"),
        (np.array([np.nan]), "z must contain only finite"),
    ],
)
def test_update_rejects_invalid_measurement(measurement, message):
    """Correction must reject malformed or non-finite observations."""
    with pytest.raises(ValueError, match=message):
        _scalar_ekf().update(measurement)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"measurement_residual": lambda z, h: np.zeros(2)}, "measurement_residual output must have shape"),
        (
            {"measurement_residual": lambda z, h: np.array([np.nan])},
            "measurement_residual output must contain only finite",
        ),
        ({"state_addition": lambda x, dx: np.zeros(2)}, "state_addition output must have shape"),
        ({"state_addition": lambda x, dx: np.array([np.nan])}, "state_addition output must contain only finite"),
    ],
)
def test_update_validates_residual_and_state_addition_outputs(override, message):
    """Correction must validate caller-owned residual and retraction outputs."""
    with pytest.raises(ValueError, match=message):
        _scalar_ekf(**override).update(np.array([0.1]))


def test_update_rejects_non_positive_definite_innovation_covariance():
    """A singular innovation covariance must fail instead of being inverted."""
    ekf = ExtendedKalmanFilter(
        np.array([0.0]),
        np.array([[0.0]]),
        np.array([[0.0]]),
        np.array([[0.0]]),
        process_model=lambda x, u, dt: x.copy(),
        process_jacobian=lambda x, u, dt: np.eye(1),
        measurement_model=lambda x: x.copy(),
        measurement_jacobian=lambda x: np.eye(1),
    )
    with pytest.raises(ValueError, match="innovation covariance must be positive definite"):
        ekf.update(np.array([0.0]))


def test_joseph_covariance_stays_symmetric_psd_under_nonlinear_updates():
    """Repeated nonlinear Joseph updates must retain covariance invariants."""
    ekf = ExtendedKalmanFilter(
        np.array([0.7]),
        np.array([[2.0]]),
        np.array([[1e-4]]),
        np.array([[1e-8]]),
        process_model=lambda x, u, dt: np.array([x[0] + dt * np.sin(x[0])]),
        process_jacobian=lambda x, u, dt: np.array([[1.0 + dt * np.cos(x[0])]]),
        measurement_model=lambda x: np.array([x[0] ** 2]),
        measurement_jacobian=lambda x: np.array([[2.0 * x[0]]]),
    )
    for _ in range(100):
        ekf.predict(0.01)
        ekf.update(np.array([0.5]))
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-15)
        assert np.linalg.eigvalsh(ekf.P).min() >= -1e-14
