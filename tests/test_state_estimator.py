# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Tests for the Extended Kalman Filter (EKF) state estimator.
"""

from __future__ import annotations

import numpy as np

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
