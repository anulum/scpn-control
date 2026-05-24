# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear MPC Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.nmpc_controller import NMPCConfig, NonlinearMPC


def mock_tokamak_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Very simple linearish mock.
    x = [Ip, beta_N, q95, li, T_axis, n_bar]
    u = [P_aux, I_p_ref, n_gas_puff]
    """
    x_next = x.copy()
    dt = 0.1

    # Ip tracks I_p_ref
    x_next[0] += dt * (u[1] - x[0]) * 0.5

    # beta_N driven by P_aux, heavily to force controller action
    x_next[1] += dt * (u[0] - x[1]) * 0.5

    # q95 ~ 1 / Ip
    if x_next[0] > 0.1:
        x_next[2] = 15.0 / x_next[0]
    else:
        x_next[2] = 150.0

    # li relaxes
    x_next[3] += dt * (1.0 - x[3]) * 0.1

    # T_axis driven by P_aux / n_bar
    if x[5] > 0.1:
        x_next[4] += dt * (2.0 * u[0] / x[5] - x[4]) * 0.5

    # n_bar driven by gas puff
    x_next[5] += dt * (u[2] - 0.5 * x[5])

    return x_next


def test_unconstrained_nmpc():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    # Widen bounds
    cfg.u_max = np.array([1000.0, 1000.0, 1000.0])
    cfg.du_max = np.array([1000.0, 1000.0, 1000.0])

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    # Target high beta_N to force P_aux
    x_ref = np.array([5.0, 20.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = np.array([10.0, 1.0, 1.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    # Should aggressively command I_p_ref and P_aux
    assert u_opt[0] > 5.0  # P_aux
    assert u_opt[1] > 1.0  # I_p_ref


def test_input_constrained_nmpc():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    cfg.u_max = np.array([10.0, 10.0, 10.0])  # Restrict P_aux to 10
    cfg.du_max = np.array([1000.0, 1000.0, 1000.0])

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 5.0, 3.0, 1.0, 10.0, 2.0])  # High targets
    u_prev = np.array([10.0, 1.0, 1.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert u_opt[0] <= 10.0


def test_slew_rate():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    cfg.u_max = np.array([100.0, 100.0, 100.0])
    cfg.du_max = np.array([1.0, 1.0, 1.0])  # Tight slew rate

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([15.0, 5.0, 3.0, 1.0, 10.0, 2.0])
    u_prev = np.array([5.0, 5.0, 5.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert np.all(np.abs(u_opt - u_prev) <= 1.0 + 1e-6)


def test_nmpc_rejects_previous_input_outside_bounds() -> None:
    """The slew-rate projection must not admit an already unsafe actuator state."""
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 5.0, 3.0, 1.0, 10.0, 2.0])
    u_prev = np.array([cfg.u_max[0] + 20.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="u_prev"):
        nmpc.step(x0, x_ref, u_prev)


def test_nmpc_horizon_one_is_valid_receding_horizon() -> None:
    """A one-step horizon is mathematically valid and must not crash warm start."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([2.0, 1.2, 8.0, 1.0, 3.0, 1.2])
    u_prev = np.array([1.0, 1.0, 1.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert u_opt.shape == (3,)
    assert np.all(u_opt >= cfg.u_min)
    assert np.all(u_opt <= cfg.u_max)
    assert np.all(np.abs(u_opt - u_prev) <= cfg.du_max + 1e-9)


def test_infeasibility_recovery():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    # Contradictory state constraints: beta_N < 1, but we command it to go high
    cfg.x_max[1] = 0.5

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])  # Already violating beta_N
    x_ref = np.array([5.0, 5.0, 3.0, 1.0, 10.0, 2.0])
    u_prev = np.array([10.0, 1.0, 1.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    # Should run and log infeasibility
    assert nmpc.infeasibility_count > 0
    assert u_opt.shape == (3,)


def test_nmpc_cost_decreases():
    """J evaluated at the SQP-optimal trajectory must be less than J at x_ref zero-input.

    Rawlings, Mayne & Diehl 2017, Ch. 1: the optimizer minimizes J,
    so J(u*) ≤ J(u=0).
    """
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    cfg.u_max = np.array([1000.0, 1000.0, 1000.0])
    cfg.du_max = np.array([1000.0, 1000.0, 1000.0])

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = cfg.u_min.copy()

    # Cost before optimization: zero-input trajectory
    x_traj_zero = np.zeros((cfg.horizon + 1, 6))
    x_traj_zero[0] = x0
    u_zero = np.zeros((cfg.horizon, 3))
    for k in range(cfg.horizon):
        x_traj_zero[k + 1] = mock_tokamak_plant(x_traj_zero[k], u_zero[k])
    J_zero = nmpc.compute_cost(x_traj_zero, u_zero, x_ref)

    nmpc.step(x0, x_ref, u_prev)

    J_opt = nmpc.compute_cost(nmpc.x_traj, nmpc.u_traj, x_ref)
    assert J_opt <= J_zero


def test_nmpc_cost_includes_terminal_penalty() -> None:
    """The evaluated objective must match the documented finite-horizon NMPC cost."""
    cfg = NMPCConfig(horizon=2, max_sqp_iter=1)
    cfg.P = 7.0 * np.eye(6)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x_ref = np.zeros(6)
    x_traj = np.zeros((3, 6))
    u_traj = np.zeros((2, 3))
    x_traj[-1, 0] = 2.0

    assert nmpc.compute_cost(x_traj, u_traj, x_ref) == 28.0


def test_terminal_cost_scipy_fallback():
    """Exercise nmpc_controller.py lines 119-120: DARE fallback to 10*Q."""
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    # Force DARE failure by patching scipy.linalg.solve_discrete_are
    from unittest.mock import patch

    with patch("scipy.linalg.solve_discrete_are", side_effect=ValueError("singular")):
        A = np.eye(6)
        B = np.eye(6, 3)
        P = nmpc._compute_terminal_cost(A, B)
    np.testing.assert_array_equal(P, cfg.Q * 10.0)


def test_terminal_cost_rejects_invalid_dare_matrix() -> None:
    """DARE output must be finite symmetric positive-definite before use."""
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    A = np.eye(6)
    B = np.eye(6, 3)
    bad_terminal_cost = np.eye(6)
    bad_terminal_cost[0, 0] = -1.0

    from unittest.mock import patch

    with patch("scipy.linalg.solve_discrete_are", return_value=bad_terminal_cost):
        P = nmpc._compute_terminal_cost(A, B)

    np.testing.assert_array_equal(P, cfg.Q * 10.0)


def test_nmpc_linearize_matches_analytic_nonlinear_jacobian() -> None:
    """Central differencing should recover smooth nonlinear plant Jacobians."""

    def nonlinear_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.zeros(6)
        out[0] = x[0] ** 2 + 3.0 * u[0] ** 2
        out[1] = np.sin(x[1]) + np.exp(u[1])
        out[2] = x[2] * u[2]
        out[3] = x[3]
        out[4] = x[4]
        out[5] = x[5]
        return out

    nmpc = NonlinearMPC(nonlinear_plant, NMPCConfig(horizon=2, max_sqp_iter=1))
    x0 = np.array([2.0, 0.4, 1.5, 0.8, 3.0, 4.0])
    u0 = np.array([1.2, 0.3, 0.7])

    A, B = nmpc.linearize(x0, u0)

    expected_A = np.zeros((6, 6))
    expected_B = np.zeros((6, 3))
    expected_A[0, 0] = 2.0 * x0[0]
    expected_B[0, 0] = 6.0 * u0[0]
    expected_A[1, 1] = np.cos(x0[1])
    expected_B[1, 1] = np.exp(u0[1])
    expected_A[2, 2] = u0[2]
    expected_B[2, 2] = x0[2]
    expected_A[3, 3] = 1.0
    expected_A[4, 4] = 1.0
    expected_A[5, 5] = 1.0

    np.testing.assert_allclose(A, expected_A, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(B, expected_B, rtol=1e-7, atol=1e-9)


def test_nmpc_linearize_respects_state_and_input_bounds() -> None:
    """Boundary linearization must not evaluate the plant outside physics bounds."""
    cfg = NMPCConfig(horizon=2, max_sqp_iter=1)

    def bounded_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if np.any(x < cfg.x_min) or np.any(x > cfg.x_max):
            raise ValueError("state outside configured physics domain")
        if np.any(u < cfg.u_min) or np.any(u > cfg.u_max):
            raise ValueError("input outside configured actuator domain")
        out = x.copy()
        out[0] = x[0] ** 2 + u[0] ** 2
        return out

    nmpc = NonlinearMPC(bounded_plant, cfg)
    x0 = np.array([cfg.x_min[0], 1.0, 5.0, 1.0, 5.0, 2.0])
    u0 = np.array([cfg.u_min[0], 1.0, 1.0])

    A, B = nmpc.linearize(x0, u0)

    assert np.isfinite(A).all()
    assert np.isfinite(B).all()
    assert A[0, 0] == pytest.approx(2.0 * x0[0], rel=1e-3, abs=1e-6)
    assert B[0, 0] == pytest.approx(2.0 * u0[0], rel=1e-3, abs=2e-4)


def test_nmpc_step_converges_early():
    """Exercise nmpc_controller.py line 209: SQP early convergence when dU < tol."""
    cfg = NMPCConfig(horizon=3, max_sqp_iter=10, tol=1e10)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    x0 = np.array([5.0, 5.0, 3.0, 1.0, 5.0, 2.0])
    x_ref = x0.copy()
    u_prev = cfg.u_min.copy()
    u = nmpc.step(x0, x_ref, u_prev)
    assert u.shape == (3,)
    assert nmpc.last_qp_converged is True
    assert nmpc.last_qp_iterations == 1


def test_nmpc_reports_qp_iteration_budget_exhaustion() -> None:
    """The QP inner loop must expose whether projection convergence was reached."""
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1, qp_max_iter=1, tol=0.0 + 1e-30)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = cfg.u_min.copy()

    nmpc.step(x0, x_ref, u_prev)

    assert nmpc.last_qp_converged is False
    assert nmpc.last_qp_iterations == cfg.qp_max_iter


def test_nmpc_rejects_non_spd_state_weight() -> None:
    cfg = NMPCConfig(horizon=3)
    cfg.Q = np.diag([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="Q"):
        NonlinearMPC(mock_tokamak_plant, cfg)


def test_nmpc_rejects_inconsistent_input_bounds() -> None:
    cfg = NMPCConfig(horizon=3)
    cfg.u_min = np.array([0.0, 5.0, 0.0])
    cfg.u_max = np.array([73.0, 4.0, 10.0])

    with pytest.raises(ValueError, match="u_min"):
        NonlinearMPC(mock_tokamak_plant, cfg)


def test_nmpc_step_rejects_nonfinite_plant_output() -> None:
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1)

    def bad_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = mock_tokamak_plant(x, u)
        out[0] = np.nan
        return out

    nmpc = NonlinearMPC(bad_plant, cfg)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = cfg.u_min.copy()

    with pytest.raises(ValueError, match="plant_model"):
        nmpc.step(x0, x_ref, u_prev)
