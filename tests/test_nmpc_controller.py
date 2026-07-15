# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear MPC Tests
from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_control.control import nmpc_controller as nmpc_mod
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


def test_nmpc_uses_adaptive_qp_step_size() -> None:
    """QP step size should be derived from local condensed curvature, not fixed."""
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1, qp_max_iter=1)
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = cfg.u_min.copy()

    nmpc.step(x0, x_ref, u_prev)

    assert np.isfinite(nmpc.last_qp_step_size)
    assert nmpc.last_qp_step_size > 0.0
    assert nmpc.last_qp_step_size != pytest.approx(0.05)


def test_nmpc_qp_step_size_shrinks_with_control_curvature() -> None:
    """Larger R increases QP curvature and should reduce the safe gradient step."""
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])

    cfg_low = NMPCConfig(horizon=3, max_sqp_iter=1, qp_max_iter=1)
    cfg_high = NMPCConfig(horizon=3, max_sqp_iter=1, qp_max_iter=1)
    cfg_high.R = 100.0 * np.eye(3)

    nmpc_low = NonlinearMPC(mock_tokamak_plant, cfg_low)
    nmpc_high = NonlinearMPC(mock_tokamak_plant, cfg_high)

    nmpc_low.step(x0, x_ref, cfg_low.u_min.copy())
    nmpc_high.step(x0, x_ref, cfg_high.u_min.copy())

    assert nmpc_high.last_qp_step_size < nmpc_low.last_qp_step_size


def test_nmpc_supports_scipy_qp_backend() -> None:
    """Configured SciPy backend should use an established constrained optimizer."""
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1, qp_max_iter=50)
    cfg.qp_backend = "scipy"
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = cfg.u_min.copy()

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert u_opt.shape == (3,)
    assert nmpc.last_qp_backend == "scipy"
    assert nmpc.last_qp_converged is True
    assert nmpc.last_qp_iterations >= 1


def test_nmpc_supports_osqp_qp_backend() -> None:
    """Configured OSQP backend should solve the condensed sparse QP."""
    pytest.importorskip("osqp")

    cfg = NMPCConfig(horizon=3, max_sqp_iter=1, qp_max_iter=500)
    cfg.qp_backend = "osqp"
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = cfg.u_min.copy()

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert u_opt.shape == (3,)
    assert nmpc.last_qp_backend == "osqp"
    assert nmpc.last_qp_converged is True
    assert nmpc.last_qp_iterations >= 1


def test_nmpc_accepts_casadi_qp_backend_configuration() -> None:
    """CasADi backend is an explicit established-solver option when installed."""
    cfg = NMPCConfig(horizon=3)
    cfg.qp_backend = "casadi"

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    assert nmpc.config.qp_backend == "casadi"


def test_nmpc_acados_backend_fails_closed_without_runtime() -> None:
    """acados deployment must not silently fall back to an internal solver."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = cfg.u_min.copy()

    with pytest.raises((ImportError, RuntimeError), match="acados"):
        nmpc.step(x0, x_ref, u_prev)


def test_nmpc_acados_backend_solves_through_runtime_boundary() -> None:
    """acados backend should delegate the full OCP to an injected runtime."""
    cfg = NMPCConfig(horizon=2, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    cfg.P = 2.0 * np.eye(6)
    cfg.acados_json_file = "build/acados/scpn_control_nmpc.json"
    cfg.acados_generate = False
    cfg.acados_build = False
    ocp_calls: list[dict[str, object]] = []
    solver_calls: list[dict[str, object]] = []
    solvers: list[object] = []

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[0] += 0.1 * u[0]
        out[1] += 0.05 * u[1]
        return out

    def forbidden_linearization(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("acados backend must not use condensed finite-difference QP linearization")

    def ocp_factory(config: NMPCConfig, terminal_cost: np.ndarray) -> dict[str, object]:
        ocp_calls.append({"horizon": config.horizon, "terminal_cost": terminal_cost.copy()})
        return {"kind": "fake_ocp", "horizon": config.horizon}

    class FakeAcadosSolver:
        def __init__(self, ocp: object, **kwargs: object) -> None:
            solver_calls.append({"ocp": ocp, "kwargs": kwargs})
            self.set_calls: list[tuple[int, str, np.ndarray]] = []

        def set(self, stage: int, field: str, value: object) -> None:
            self.set_calls.append((stage, field, np.asarray(value, dtype=float).copy()))

        def solve(self) -> int:
            return 0

        def get(self, stage: int, field: str) -> np.ndarray:
            if field == "u":
                return np.array([2.0 + 0.1 * stage, 1.5, 0.5])
            if field == "x":
                if stage == 0:
                    physical_state = np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0])
                    previous_input = np.array([1.5, 1.0, 0.25])
                elif stage == 1:
                    physical_state = plant(np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]), np.array([2.0, 1.5, 0.5]))
                    previous_input = np.array([2.0, 1.5, 0.5])
                else:
                    physical_state = plant(
                        plant(np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]), np.array([2.0, 1.5, 0.5])),
                        np.array([2.1, 1.5, 0.5]),
                    )
                    previous_input = np.array([2.1, 1.5, 0.5])
                return np.r_[physical_state, previous_input]
            raise KeyError(field)

        def get_stats(self, field: str) -> int:
            assert field == "sqp_iter"
            return 3

    def solver_factory(ocp: object, **kwargs: object) -> FakeAcadosSolver:
        solver = FakeAcadosSolver(ocp, **kwargs)
        solvers.append(solver)
        return solver

    nmpc = NonlinearMPC(
        plant,
        cfg,
        linearization_model=forbidden_linearization,
        acados_ocp_factory=ocp_factory,
        acados_solver_factory=solver_factory,
    )
    x0 = np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 2.0, 5.0, 1.0, 5.0, 2.0])
    u_prev = np.array([1.5, 1.0, 0.25])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    np.testing.assert_allclose(u_opt, np.array([2.0, 1.5, 0.5]))
    assert nmpc.last_qp_backend == "acados"
    assert nmpc.last_qp_converged is True
    assert nmpc.last_qp_iterations == 3
    assert nmpc.last_acados_dynamics_residual <= cfg.acados_dynamics_residual_tol
    assert len(ocp_calls) == 1
    assert ocp_calls[0]["horizon"] == 2
    np.testing.assert_allclose(ocp_calls[0]["terminal_cost"], 2.0 * np.eye(6))
    terminal_stage_set_calls = [
        (stage, field) for stage, field, _ in solvers[0].set_calls if stage == nmpc.N and field in {"lbx", "ubx"}
    ]
    assert terminal_stage_set_calls == []
    assert solver_calls[0]["kwargs"] == {
        "json_file": "build/acados/scpn_control_nmpc.json",
        "build": False,
        "generate": False,
        "verbose": False,
    }

    # Re-solving reuses the cached acados OCP and solver instead of rebuilding them
    # (arc 959->962: the "_acados_ocp is None or _acados_solver is None" guard is False).
    nmpc.step(x0, x_ref, u_prev)
    assert len(ocp_calls) == 1
    assert len(solver_calls) == 1


def test_nmpc_acados_close_releases_cached_solver_once() -> None:
    """Long-running acados deployments must explicitly release native solver state."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    freed: list[int] = []

    def identity_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x.copy()

    class ReleasableAcadosSolver:
        def set(self, stage: int, field: str, value: Any) -> None:
            return None

        def solve(self) -> int:
            return 0

        def get(self, stage: int, field: str) -> np.ndarray:
            if field == "u":
                return np.array([0.0, 1.0, 0.0])
            if field == "x":
                return np.r_[np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0])]
            raise KeyError(field)

        def get_stats(self, field: str) -> int:
            return 1

        def free(self) -> None:
            freed.append(1)

    nmpc = NonlinearMPC(
        identity_plant,
        cfg,
        acados_ocp_factory=lambda config, terminal_cost: object(),
        acados_solver_factory=lambda ocp, **kwargs: ReleasableAcadosSolver(),
    )

    nmpc.step(
        np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]),
        np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    assert freed == []
    assert nmpc._acados_solver is not None
    assert nmpc._acados_ocp is not None

    nmpc.close()
    nmpc.close()

    assert freed == [1]
    assert nmpc._acados_solver is None
    assert nmpc._acados_ocp is None


def test_nmpc_acados_context_manager_closes_solver_after_exception() -> None:
    """Context-managed acados deployments must release solver state on faults."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    freed: list[int] = []

    def identity_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x.copy()

    class ContextAcadosSolver:
        def set(self, stage: int, field: str, value: Any) -> None:
            return None

        def solve(self) -> int:
            return 0

        def get(self, stage: int, field: str) -> np.ndarray:
            if field == "u":
                return np.array([0.0, 1.0, 0.0])
            if field == "x":
                return np.r_[np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0])]
            raise KeyError(field)

        def get_stats(self, field: str) -> int:
            return 1

        def free(self) -> None:
            freed.append(1)

    nmpc = NonlinearMPC(
        identity_plant,
        cfg,
        acados_ocp_factory=lambda config, terminal_cost: object(),
        acados_solver_factory=lambda ocp, **kwargs: ContextAcadosSolver(),
    )

    with pytest.raises(RuntimeError, match="fault after solve"), nmpc as managed:
        managed.step(
            np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]),
            np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        raise RuntimeError("fault after solve")

    assert freed == [1]
    assert nmpc._acados_solver is None
    assert nmpc._acados_ocp is None


def test_nmpc_acados_context_manager_preserves_control_fault_when_free_fails() -> None:
    """acados cleanup failures must not mask the controller fault being unwound."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    cfg.qp_backend = "acados"

    def identity_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x.copy()

    class FaultingFreeAcadosSolver:
        def set(self, stage: int, field: str, value: Any) -> None:
            return None

        def solve(self) -> int:
            return 0

        def get(self, stage: int, field: str) -> np.ndarray:
            if field == "u":
                return np.array([0.0, 1.0, 0.0])
            if field == "x":
                return np.r_[np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0])]
            raise KeyError(field)

        def get_stats(self, field: str) -> int:
            return 1

        def free(self) -> None:
            raise OSError("native acados free failed")

    nmpc = NonlinearMPC(
        identity_plant,
        cfg,
        acados_ocp_factory=lambda config, terminal_cost: object(),
        acados_solver_factory=lambda ocp, **kwargs: FaultingFreeAcadosSolver(),
    )

    with (
        pytest.warns(RuntimeWarning, match="acados backend cleanup failed"),
        pytest.raises(RuntimeError, match="control-loop fault"),
        nmpc as managed,
    ):
        managed.step(
            np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]),
            np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        raise RuntimeError("control-loop fault")

    assert nmpc._acados_solver is None
    assert nmpc._acados_ocp is None


def test_nmpc_acados_backend_rejects_failed_solver_status() -> None:
    """Nonzero acados status must fail closed rather than returning stale input."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    freed: list[int] = []

    class FailingAcadosSolver:
        def set(self, stage: int, field: str, value: Any) -> None:
            return None

        def solve(self) -> int:
            return 4

        def get_stats(self, field: str) -> int:
            return 1

        def free(self) -> None:
            freed.append(1)

    nmpc = NonlinearMPC(
        mock_tokamak_plant,
        cfg,
        acados_ocp_factory=lambda config, terminal_cost: object(),
        acados_solver_factory=lambda ocp, **kwargs: FailingAcadosSolver(),
    )

    with pytest.raises(RuntimeError, match="acados backend failed"):
        nmpc.step(
            np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0]),
            np.array([5.0, 2.0, 3.0, 1.0, 5.0, 2.0]),
            cfg.u_min.copy(),
        )

    assert freed == [1]
    assert nmpc._acados_solver is None
    assert nmpc._acados_ocp is None


def test_nmpc_acados_backend_rejects_symbolic_runtime_dynamics_drift() -> None:
    """acados state predictions must match the runtime plant before admission."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    cfg.acados_dynamics_residual_tol = 1.0e-9

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[0] += 0.1 * u[0]
        return out

    class DriftedAcadosSolver:
        def set(self, stage: int, field: str, value: Any) -> None:
            return None

        def solve(self) -> int:
            return 0

        def get(self, stage: int, field: str) -> np.ndarray:
            if field == "u":
                return np.array([0.0, 1.0, 0.0])
            if field == "x":
                if stage == 0:
                    return np.r_[np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0])]
                return np.r_[np.array([1.5, 1.0, 5.0, 1.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0])]
            raise KeyError(field)

        def get_stats(self, field: str) -> int:
            return 1

    nmpc = NonlinearMPC(
        plant,
        cfg,
        acados_ocp_factory=lambda config, terminal_cost: object(),
        acados_solver_factory=lambda ocp, **kwargs: DriftedAcadosSolver(),
    )

    with pytest.raises(RuntimeError, match="dynamics residual"):
        nmpc.step(
            np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]),
            np.array([5.0, 2.0, 5.0, 1.0, 5.0, 2.0]),
            np.array([0.0, 1.0, 0.0]),
        )


def test_nmpc_acados_backend_rejects_terminal_state_violation() -> None:
    """acados terminal state must satisfy the declared terminal admissible set."""
    cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    cfg.terminal_x_min = cfg.x_min.copy()
    cfg.terminal_x_max = cfg.x_max.copy()
    cfg.terminal_x_max[0] = 1.2

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[0] = 1.5
        return out

    class TerminalViolatingAcadosSolver:
        def set(self, stage: int, field: str, value: Any) -> None:
            return None

        def solve(self) -> int:
            return 0

        def get(self, stage: int, field: str) -> np.ndarray:
            if field == "u":
                return np.array([0.0, 1.0, 0.0])
            if field == "x":
                state = (
                    np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]) if stage == 0 else np.array([1.5, 1.0, 5.0, 1.0, 2.0, 1.0])
                )
                return np.r_[state, np.array([0.0, 1.0, 0.0])]
            raise KeyError(field)

        def get_stats(self, field: str) -> int:
            return 1

    nmpc = NonlinearMPC(
        plant,
        cfg,
        acados_ocp_factory=lambda config, terminal_cost: object(),
        acados_solver_factory=lambda ocp, **kwargs: TerminalViolatingAcadosSolver(),
    )

    with pytest.raises(RuntimeError, match="terminal state"):
        nmpc.step(
            np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0]),
            np.array([5.0, 2.0, 5.0, 1.0, 5.0, 2.0]),
            np.array([0.0, 1.0, 0.0]),
        )


def test_nmpc_acados_symbolic_builder_creates_augmented_slew_constrained_ocp(monkeypatch) -> None:
    """Default acados builder should encode symbolic dynamics and slew constraints."""

    class FakeSymbol:
        def __init__(self, name: str) -> None:
            self.name = name

        def __getitem__(self, item: object) -> "FakeSymbol":
            return FakeSymbol(f"{self.name}[{item!r}]")

        def __sub__(self, other: object) -> tuple[str, object, object]:
            return ("sub", self, other)

    fake_casadi = types.SimpleNamespace(
        MX=types.SimpleNamespace(sym=lambda name, size: FakeSymbol(f"{name}:{size}")),
        vertcat=lambda *args: ("vertcat", args),
    )

    class FakeAcadosModel:
        pass

    class FakeAcadosOcp:
        def __init__(self) -> None:
            self.dims = types.SimpleNamespace()
            self.solver_options = types.SimpleNamespace()
            self.cost = types.SimpleNamespace()
            self.constraints = types.SimpleNamespace()

    fake_acados_template = types.SimpleNamespace(AcadosModel=FakeAcadosModel, AcadosOcp=FakeAcadosOcp)
    monkeypatch.setitem(sys.modules, "casadi", fake_casadi)
    monkeypatch.setitem(sys.modules, "acados_template", fake_acados_template)

    cfg = NMPCConfig(horizon=4, max_sqp_iter=2, qp_max_iter=30)
    cfg.qp_backend = "acados"
    cfg.acados_nlp_solver_type = "SQP"
    cfg.acados_qp_solver = "PARTIAL_CONDENSING_HPIPM"

    def symbolic_dynamics(ca_module: object, x: object, u: object) -> object:
        assert ca_module is fake_casadi
        assert isinstance(x, FakeSymbol)
        assert isinstance(u, FakeSymbol)
        return x

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg, symbolic_dynamics_model=symbolic_dynamics)

    ocp = nmpc._build_acados_ocp(3.0 * np.eye(6))

    assert ocp.dims.N == 4
    assert ocp.solver_options.N_horizon == 4
    assert ocp.solver_options.nlp_solver_type == "SQP"
    assert ocp.solver_options.qp_solver == "PARTIAL_CONDENSING_HPIPM"
    assert ocp.solver_options.hessian_approx == "EXACT"
    assert ocp.solver_options.integrator_type == "DISCRETE"
    assert ocp.cost.W.shape == (9, 9)
    assert ocp.cost.W_e.shape == (6, 6)
    assert ocp.constraints.idxbx.shape == (9,)
    np.testing.assert_array_equal(ocp.constraints.idxbx_e, np.arange(6))
    np.testing.assert_allclose(ocp.constraints.lbx_e, cfg.x_min)
    np.testing.assert_allclose(ocp.constraints.ubx_e, cfg.x_max)
    np.testing.assert_allclose(ocp.constraints.lh, -cfg.du_max)
    np.testing.assert_allclose(ocp.constraints.uh, cfg.du_max)
    assert ocp.model.disc_dyn_expr[0] == "vertcat"


def test_nmpc_rejects_unknown_qp_backend() -> None:
    cfg = NMPCConfig(horizon=3)
    cfg.qp_backend = "unknown"

    with pytest.raises(ValueError, match="qp_backend"):
        NonlinearMPC(mock_tokamak_plant, cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("horizon", 0, "horizon"),
        ("max_sqp_iter", 0, "max_sqp_iter"),
        ("qp_max_iter", 0, "qp_max_iter"),
        ("tol", 0.0, "tol"),
        ("linearization_backend", "finite-different", "linearization_backend"),
        ("acados_model_name", "", "acados_model_name"),
        ("acados_json_file", "", "acados_json_file"),
        ("acados_generate", 1, "acados_generate"),
        ("acados_build", 0, "acados_build"),
        ("acados_dynamics_residual_tol", np.nan, "acados_dynamics_residual_tol"),
    ],
)
def test_nmpc_config_rejects_invalid_solver_domain_fields(field: str, value: object, message: str) -> None:
    cfg = NMPCConfig(horizon=3)
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=message):
        NonlinearMPC(mock_tokamak_plant, cfg)


def test_nmpc_rejects_non_spd_state_weight() -> None:
    cfg = NMPCConfig(horizon=3)
    cfg.Q = np.diag([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="Q"):
        NonlinearMPC(mock_tokamak_plant, cfg)


def test_nmpc_rejects_nearly_symmetric_state_weight() -> None:
    cfg = NMPCConfig(horizon=3)
    cfg.Q = np.eye(6)
    cfg.Q[0, 1] = 1.0e-12

    with pytest.raises(ValueError, match="Q"):
        NonlinearMPC(mock_tokamak_plant, cfg)


def test_nmpc_rejects_inconsistent_input_bounds() -> None:
    cfg = NMPCConfig(horizon=3)
    cfg.u_min = np.array([0.0, 5.0, 0.0])
    cfg.u_max = np.array([73.0, 4.0, 10.0])

    with pytest.raises(ValueError, match="u_min"):
        NonlinearMPC(mock_tokamak_plant, cfg)


def test_nmpc_rejects_partial_or_out_of_domain_terminal_set() -> None:
    cfg_partial = NMPCConfig(horizon=3)
    cfg_partial.qp_backend = "scipy"
    cfg_partial.terminal_x_min = cfg_partial.x_min.copy()
    with pytest.raises(ValueError, match="terminal_x_min and terminal_x_max"):
        NonlinearMPC(mock_tokamak_plant, cfg_partial)

    cfg_internal = NMPCConfig(horizon=3)
    cfg_internal.terminal_x_min = cfg_internal.x_min.copy()
    cfg_internal.terminal_x_max = cfg_internal.x_max.copy()
    with pytest.raises(ValueError, match="terminal_x constraints"):
        NonlinearMPC(mock_tokamak_plant, cfg_internal)

    cfg_bad_order = NMPCConfig(horizon=3)
    cfg_bad_order.qp_backend = "scipy"
    cfg_bad_order.terminal_x_min = cfg_bad_order.x_min.copy()
    cfg_bad_order.terminal_x_max = cfg_bad_order.x_max.copy()
    cfg_bad_order.terminal_x_min[0] = cfg_bad_order.terminal_x_max[0]
    with pytest.raises(ValueError, match="terminal_x_min"):
        NonlinearMPC(mock_tokamak_plant, cfg_bad_order)

    cfg_outside = NMPCConfig(horizon=3)
    cfg_outside.qp_backend = "scipy"
    cfg_outside.terminal_x_min = cfg_outside.x_min.copy()
    cfg_outside.terminal_x_max = cfg_outside.x_max.copy()
    cfg_outside.terminal_x_max[0] = cfg_outside.x_max[0] + 1.0
    with pytest.raises(ValueError, match="terminal_x bounds"):
        NonlinearMPC(mock_tokamak_plant, cfg_outside)


def test_nmpc_compute_cost_rejects_malformed_trajectories() -> None:
    nmpc = NonlinearMPC(mock_tokamak_plant, NMPCConfig(horizon=2, max_sqp_iter=1))
    x_ref = np.zeros(6)

    with pytest.raises(ValueError, match="x_traj"):
        nmpc.compute_cost(np.zeros((2, 5)), np.zeros((1, 3)), x_ref)
    with pytest.raises(ValueError, match="u_traj"):
        nmpc.compute_cost(np.zeros((2, 6)), np.zeros((1, 2)), x_ref)
    with pytest.raises(ValueError, match="at least one more row"):
        nmpc.compute_cost(np.zeros((1, 6)), np.zeros((1, 3)), x_ref)


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


def test_nmpc_uses_analytic_linearization_provider_without_plant_calls() -> None:
    """Analytic plant Jacobians avoid finite-difference plant evaluations."""

    plant_calls = {"count": 0}

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        plant_calls["count"] += 1
        return x

    A_expected = np.eye(6)
    B_expected = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
            [0.4, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.6],
        ],
        dtype=float,
    )

    def linearization(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert x.shape == (6,)
        assert u.shape == (3,)
        return A_expected, B_expected

    nmpc = NonlinearMPC(
        plant,
        NMPCConfig(horizon=2),
        linearization_model=linearization,
    )

    A, B = nmpc.linearize(np.zeros(6), np.zeros(3))

    assert plant_calls["count"] == 0
    np.testing.assert_allclose(A, A_expected)
    np.testing.assert_allclose(B, B_expected)
    assert nmpc.last_linearization_source == "analytic"


def test_nmpc_rejects_invalid_analytic_linearization_provider_output() -> None:
    """Analytic plant Jacobians must match the NMPC state-input dimensions."""

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    def bad_linearization(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.eye(5), np.full((6, 3), np.nan)

    nmpc = NonlinearMPC(
        plant,
        NMPCConfig(horizon=2),
        linearization_model=bad_linearization,
    )

    with pytest.raises(ValueError, match="linearization_model"):
        nmpc.linearize(np.zeros(6), np.zeros(3))


def test_nmpc_uses_jax_linearization_backend_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    A_expected = np.eye(6)
    B_expected = np.zeros((6, 3))
    B_expected[0, 0] = 0.25

    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.float64 = np.float64
    fake_jnp.asarray = np.asarray

    fake_jax = types.ModuleType("jax")
    fake_jax.__path__ = []  # allow `import jax.numpy` against the injected module
    fake_jax.numpy = fake_jnp

    def fake_jacfwd(fn, argnums):  # type: ignore[no-untyped-def]
        def wrapped(x_arg, u_arg):  # type: ignore[no-untyped-def]
            out = fn(x_arg, u_arg)
            assert np.asarray(out).shape == (6,)
            assert argnums == (0, 1)
            return A_expected, B_expected

        return wrapped

    fake_jax.jacfwd = fake_jacfwd
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float64).copy()
        out[0] += 0.25 * float(np.asarray(u)[0])
        return out

    cfg = NMPCConfig(horizon=2)
    cfg.linearization_backend = "jax"
    nmpc = NonlinearMPC(plant, cfg)

    A, B = nmpc.linearize(np.zeros(6), np.zeros(3))

    np.testing.assert_allclose(A, A_expected)
    np.testing.assert_allclose(B, B_expected)
    assert nmpc.last_linearization_source == "jax"


def test_nmpc_scipy_backend_enforces_terminal_state_set() -> None:
    """Terminal set constraints must enter the condensed QP, not only the cost."""

    A = np.eye(6)
    B = np.zeros((6, 3))
    B[0, 0] = 1.0

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return A @ x + B @ u

    def linearization(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return A, B

    cfg = NMPCConfig(horizon=1, max_sqp_iter=2, qp_max_iter=100)
    cfg.qp_backend = "scipy"
    cfg.R = 1.0e-3 * np.eye(3)
    cfg.terminal_x_min = cfg.x_min.copy()
    cfg.terminal_x_min[0] = 2.0
    cfg.terminal_x_max = cfg.x_max.copy()

    nmpc = NonlinearMPC(plant, cfg, linearization_model=linearization)
    x0 = np.array([1.0, 1.0, 3.0, 1.0, 5.0, 1.0])
    x_ref = x0.copy()
    u_prev = np.array([0.0, 1.0, 0.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert u_opt[0] >= 1.0 - 1.0e-7
    assert nmpc.x_traj[-1, 0] >= cfg.terminal_x_min[0] - 1.0e-7


def test_nmpc_terminal_set_requires_established_constrained_backend() -> None:
    """Coupled terminal constraints must fail closed without a capable solver."""

    cfg = NMPCConfig(horizon=1)
    cfg.terminal_x_min = cfg.x_min.copy()
    cfg.terminal_x_max = cfg.x_max.copy()

    with pytest.raises(ValueError, match="terminal_x"):
        NonlinearMPC(mock_tokamak_plant, cfg)


# ── Real-Time Iteration ──────────────────────────────────────────────


def _linear_traceable_plant(x: Any, u: Any) -> Any:
    """A smooth, JAX-traceable linear plant: x_next = x + 0.1 (M x + N u)."""
    M = np.diag([-0.5, -0.5, -0.2, -0.1, -0.5, -0.5])
    N = np.zeros((6, 3))
    N[0, 1] = 0.5
    N[1, 0] = 0.5
    N[4, 0] = 0.3
    N[5, 2] = 1.0
    return x + 0.1 * (M @ x + N @ u)


def _wide_rti_config(horizon: int = 5, **kwargs: Any) -> NMPCConfig:
    cfg = NMPCConfig(horizon=horizon, **kwargs)
    cfg.u_max = np.array([1000.0, 1000.0, 1000.0])
    cfg.du_max = np.array([1000.0, 1000.0, 1000.0])
    return cfg


def test_rti_is_single_iteration_and_reuses_warm_start() -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config())
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    u_prev = np.array([10.0, 5.0, 1.0])

    first = mpc.step_rti(x, x_ref, u_prev)
    assert first.sqp_iterations == 1
    assert first.warm_started is False
    assert first.u0.shape == (3,)
    assert first.solve_time_ms >= 0.0

    second = mpc.step_rti(x, x_ref, u_prev)
    assert second.warm_started is True
    assert second.sqp_iterations == 1


def test_rti_admits_well_posed_step_and_certifies_stationarity() -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config())
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    u_prev = np.array([10.0, 5.0, 1.0])

    for _ in range(4):
        result = mpc.step_rti(x, x_ref, u_prev)
    assert result.admitted is True
    assert result.constraint_violation is False
    assert result.stationarity_residual <= mpc.config.rti_residual_tol


def test_rti_fails_closed_when_residual_tolerance_is_tight() -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config(rti_residual_tol=1e-30))
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    u_prev = np.array([10.0, 5.0, 1.0])
    result = mpc.step_rti(x, x_ref, u_prev)
    assert result.admitted is False
    assert result.stationarity_residual > mpc.config.rti_residual_tol


def test_rti_reset_clears_warm_start() -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config())
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    u_prev = np.array([10.0, 5.0, 1.0])
    mpc.step_rti(x, x_ref, u_prev)
    mpc.reset_warm_start()
    assert mpc.step_rti(x, x_ref, u_prev).warm_started is False


def test_rti_rejects_out_of_bounds_previous_input() -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, NMPCConfig(horizon=3))
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    with pytest.raises(ValueError, match="input bounds"):
        mpc.step_rti(x, x_ref, np.array([1e6, 5.0, 1.0]))


def test_rti_latency_report_orders_percentiles() -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config())
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    u_prev = np.array([10.0, 5.0, 1.0])
    report = mpc.benchmark_rti_latency(x, x_ref, u_prev, warmup_ticks=2, timed_ticks=12)
    assert report.timed_ticks == 12
    assert report.horizon == 5
    assert 0.0 <= report.p50_ms <= report.p95_ms <= report.p99_ms <= report.max_ms
    assert 0 <= report.admitted_ticks <= 12


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"warmup_ticks": -1}, "warmup_ticks"),
        ({"timed_ticks": 0}, "timed_ticks"),
    ],
)
def test_rti_latency_rejects_invalid_tick_counts(kwargs: dict[str, int], match: str) -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config())
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=match):
        mpc.benchmark_rti_latency(x, x, np.array([10.0, 5.0, 1.0]), **kwargs)


def test_rti_supports_horizon_one_receding_step() -> None:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config(horizon=1))
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    u_prev = np.array([10.0, 5.0, 1.0])
    first = mpc.step_rti(x, x_ref, u_prev)
    assert first.sqp_iterations == 1
    assert first.u0.shape == (3,)
    # With a one-step horizon the warm start is seeded from the previous input.
    second = mpc.step_rti(x, x_ref, u_prev)
    assert second.warm_started is True


def test_rti_marks_constraint_violation_and_fails_closed() -> None:
    cfg = _wide_rti_config(horizon=4)
    cfg.x_max = np.array([1.0, 0.5, 4.0, 1.5, 1.0, 1.0])  # tighter than the rolled-out state
    mpc = NonlinearMPC(_linear_traceable_plant, cfg)
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    u_prev = np.array([10.0, 5.0, 1.0])
    result = mpc.step_rti(x, x_ref, u_prev)
    assert result.constraint_violation is True
    assert result.admitted is False
    assert mpc.infeasibility_count >= 1


# ── JAX exact cost Hessian ───────────────────────────────────────────


def _hessian_setup() -> tuple[NonlinearMPC, np.ndarray, np.ndarray, np.ndarray]:
    mpc = NonlinearMPC(_linear_traceable_plant, _wide_rti_config(horizon=3))
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    x_ref = np.array([6.0, 1.5, 3.0, 1.0, 2.5, 3.5])
    U = np.tile(np.array([10.0, 5.0, 1.0]), (3, 1))
    return mpc, x, U, x_ref


def test_cost_hessian_is_symmetric_and_positive_semidefinite() -> None:
    pytest.importorskip("jax")
    mpc, x, U, x_ref = _hessian_setup()
    hessian = mpc.cost_hessian_jax(x, U, x_ref)
    assert hessian.shape == (9, 9)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-8)
    assert float(np.min(np.linalg.eigvalsh(0.5 * (hessian + hessian.T)))) >= -1e-8


def test_cost_hessian_audit_matches_finite_difference() -> None:
    pytest.importorskip("jax")
    mpc, x, U, x_ref = _hessian_setup()
    audit = mpc.assert_cost_hessian_consistent(x, U, x_ref)
    assert audit.passed is True
    assert audit.max_abs_error <= audit.tolerance
    assert audit.is_positive_semidefinite is True
    assert audit.symmetry_error < 1e-6


def test_assert_cost_hessian_consistent_fails_closed_on_tight_tolerance() -> None:
    pytest.importorskip("jax")
    mpc, x, U, x_ref = _hessian_setup()
    with pytest.raises(ValueError, match="cost Hessian audit failed"):
        mpc.assert_cost_hessian_consistent(x, U, x_ref, tolerance=1e-30)


def test_cost_hessian_rejects_wrong_control_shape() -> None:
    pytest.importorskip("jax")
    mpc, x, _, x_ref = _hessian_setup()
    with pytest.raises(ValueError, match="U must be finite with shape"):
        mpc.cost_hessian_jax(x, np.zeros((2, 3)), x_ref)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epsilon": 0.0}, "epsilon"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"sample_indices": []}, "at least one"),
        ({"sample_indices": [(0, 99)]}, "out-of-range"),
    ],
)
def test_cost_hessian_audit_rejects_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    pytest.importorskip("jax")
    mpc, x, U, x_ref = _hessian_setup()
    with pytest.raises(ValueError, match=match):
        mpc.audit_cost_hessian_jax(x, U, x_ref, **kwargs)


def test_cost_hessian_audit_accepts_valid_sample_indices() -> None:
    """Explicit in-range sample_indices validate and the audit runs to completion (branches 1447->1446, 1446->1450)."""
    pytest.importorskip("jax")
    mpc, x, U, x_ref = _hessian_setup()
    audit = mpc.audit_cost_hessian_jax(x, U, x_ref, sample_indices=[(0, 0), (1, 1), (2, 3)])
    assert audit.max_abs_error >= 0.0
    assert isinstance(audit.passed, bool)


def test_cost_hessian_fails_closed_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jax", None)
    mpc, x, U, x_ref = _hessian_setup()
    with pytest.raises(RuntimeError, match="requires jax"):
        mpc.cost_hessian_jax(x, U, x_ref)


def test_cost_hessian_requires_traceable_plant() -> None:
    pytest.importorskip("jax")

    def untraceable_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=float)
        if float(out[0]) > 0.0:  # data-dependent Python branch breaks tracing
            out[0] = out[0] + float(u[0])
        return out

    mpc = NonlinearMPC(untraceable_plant, _wide_rti_config(horizon=3))
    x = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    U = np.tile(np.array([10.0, 5.0, 1.0]), (3, 1))
    with pytest.raises(RuntimeError, match="JAX-traceable"):
        mpc.cost_hessian_jax(x, U, x)


def test_cost_hessian_rejects_non_finite_result(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    mpc, x, U, x_ref = _hessian_setup()
    dim = mpc.N * mpc.nu
    # A pathological plant could yield a non-finite Hessian; the guard must catch
    # it even though the shape is correct.
    monkeypatch.setattr(jax, "hessian", lambda fn: lambda u: jnp.full((dim, dim), jnp.inf))
    with pytest.raises(ValueError, match="Hessian must be finite"):
        mpc.cost_hessian_jax(x, U, x_ref)


def test_percentile_ms_helper_covers_edge_branches() -> None:
    # Empty sample is rejected; a single sample returns itself; a percentile that
    # lands on an exact integer rank returns that element without interpolation.
    with pytest.raises(ValueError, match="must not be empty"):
        nmpc_mod._percentile_ms([], 0.5)
    assert nmpc_mod._percentile_ms([3.0], 0.95) == 3.0
    assert nmpc_mod._percentile_ms([1.0, 2.0, 3.0], 0.5) == 2.0
    assert nmpc_mod._percentile_ms([1.0, 2.0], 0.5) == pytest.approx(1.5)


@pytest.mark.skipif(
    importlib.util.find_spec("casadi") is not None,
    reason="casadi is installed (e.g. pulled in by acados_template); the ImportError fallback only fires when absent",
)
def test_casadi_backend_raises_without_casadi() -> None:
    # casadi is an optional QP backend; when absent, selecting it must route
    # through _solve_qp -> _solve_qp_casadi and fail closed with a clear ImportError.
    cfg = NMPCConfig(horizon=3, max_sqp_iter=1, qp_backend="casadi")
    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 20.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = np.array([10.0, 1.0, 1.0])
    with pytest.raises(ImportError, match="casadi"):
        nmpc.step(x0, x_ref, u_prev)
