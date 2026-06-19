# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear MPC error and fail-closed paths
"""Fail-closed and validator branches of the nonlinear MPC controller.

Covers the input guards, the gradient-contract checks on the JAX transport
tuning entry points, the curvature-degeneracy fallbacks, the established-solver
backends (SciPy/OSQP/CasADi) failure and terminal-set branches, the acados
native-boundary validation raises, and the static acados marshalling helpers.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_control.control import nmpc_controller as nmpc_mod
from scpn_control.control.nmpc_controller import (
    NMPCConfig,
    NonlinearMPC,
    _as_finite_vector,
    _as_spd_matrix,
    _audit_transport_rollout_source_gradients,
    _rollout_audit_indices,
)
from scpn_control.core import differentiable_transport as transport_mod


def _identity_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return x.copy()


def _linear_plant() -> tuple[np.ndarray, np.ndarray, Any, Any]:
    A = np.eye(6)
    B = np.zeros((6, 3))
    B[0, 0] = 1.0

    def plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return A @ x + B @ u

    def linearization(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return A, B

    return A, B, plant, linearization


# ── module-level vector / matrix guards ──────────────────────────────


class TestArrayGuards:
    def test_finite_vector_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"finite vector with shape \(6,\)"):
            _as_finite_vector("v", np.zeros(5), 6)

    def test_spd_matrix_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"finite matrix with shape \(2, 2\)"):
            _as_spd_matrix("M", np.zeros((2, 3)), 2)


# ── rollout audit index parsing ───────────────────────────────────────


class TestRolloutAuditIndices:
    def test_default_indices_for_valid_shape(self):
        indices = _rollout_audit_indices((4, 4, 6), None)
        assert all(0 <= s < 4 and 0 <= c < 4 and 0 <= r < 6 for s, c, r in indices)
        assert len(set(indices)) == len(indices)

    def test_rejects_non_three_dimensional_shape(self):
        with pytest.raises(ValueError, match=r"shape \(n_steps, 4, n_rho\)"):
            _rollout_audit_indices((4, 6), None)

    def test_rejects_wrong_channel_or_radius_count(self):
        with pytest.raises(ValueError, match="n_rho >= 3"):
            _rollout_audit_indices((1, 4, 2), None)

    def test_rejects_non_iterable_sample_indices(self):
        with pytest.raises(ValueError, match="iterable of three-part indices"):
            _rollout_audit_indices((2, 4, 5), 7)

    def test_rejects_non_iterable_index_entry(self):
        with pytest.raises(ValueError, match="iterable three-part indices"):
            _rollout_audit_indices((2, 4, 5), [7])

    def test_rejects_index_with_wrong_arity(self):
        with pytest.raises(ValueError, match="three-part indices"):
            _rollout_audit_indices((2, 4, 5), [(0, 1)])

    def test_rejects_out_of_range_index(self):
        with pytest.raises(ValueError, match="out-of-range rollout source index"):
            _rollout_audit_indices((2, 4, 5), [(99, 0, 0)])

    def test_rejects_empty_index_set(self):
        with pytest.raises(ValueError, match="at least one index"):
            _rollout_audit_indices((2, 4, 5), [])


class TestRolloutAuditTolerances:
    def test_rejects_nonpositive_epsilon(self):
        dummy = np.zeros((1, 4, 3))
        with pytest.raises(ValueError, match="gradient_audit_epsilon must be positive"):
            _audit_transport_rollout_source_gradients(
                np.zeros((4, 3)),
                np.zeros((4, 3)),
                dummy,
                np.zeros((1, 4, 3)),
                np.linspace(0.1, 1.0, 3),
                1.0e-3,
                np.zeros(4),
                dummy,
                weights=None,
                epsilon=0.0,
                tolerance=1.0e-3,
                sample_indices=None,
            )

    def test_rejects_nonpositive_tolerance(self):
        dummy = np.zeros((1, 4, 3))
        with pytest.raises(ValueError, match="gradient_audit_tolerance must be positive"):
            _audit_transport_rollout_source_gradients(
                np.zeros((4, 3)),
                np.zeros((4, 3)),
                dummy,
                np.zeros((1, 4, 3)),
                np.linspace(0.1, 1.0, 3),
                1.0e-3,
                np.zeros(4),
                dummy,
                weights=None,
                epsilon=1.0e-5,
                tolerance=-1.0,
                sample_indices=None,
            )


# ── JAX transport tuning entry points ─────────────────────────────────


def _coeff_case() -> tuple[np.ndarray, ...]:
    rho = np.linspace(0.05, 1.0, 24)
    profiles = np.stack(
        [
            8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2,
            6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2,
            4.0 + 0.8 * (1.0 - rho**2),
            0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02),
        ]
    )
    chi = np.stack([0.2 + 0.02 * rho, 0.16 + 0.02 * rho, 0.04 + 0.005 * rho, 0.012 + 0.001 * rho])
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    return profiles, chi, sources, target, rho, edge_values


class TestCoefficientTuningGuards:
    def test_rejects_nonpositive_learning_rate(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            nmpc_mod.tune_transport_coefficients_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.0
            )

    def test_rejects_negative_chi_min(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="chi_min must be non-negative"):
            nmpc_mod.tune_transport_coefficients_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05, chi_min=-1.0
            )

    def test_rejects_nonpositive_max_fractional_update(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="max_fractional_update must be positive"):
            nmpc_mod.tune_transport_coefficients_for_tracking(
                profiles,
                chi,
                sources,
                target,
                rho,
                1.0e-3,
                edge,
                learning_rate=0.05,
                max_fractional_update=0.0,
            )

    def test_rejects_non_two_dimensional_chi(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="chi must be a finite non-negative two-dimensional"):
            nmpc_mod.tune_transport_coefficients_for_tracking(
                profiles, np.ones(4), sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_rejects_gradient_shape_mismatch(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(nmpc_mod, "transport_loss_gradient", lambda *a, **k: (0.1, np.zeros((2, 2))))
        with pytest.raises(ValueError, match="transport gradient must be finite and match chi shape"):
            nmpc_mod.tune_transport_coefficients_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_unbounded_fractional_update_path_runs_without_clip(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        gradient = 0.01 * np.ones_like(chi)
        audit = transport_mod.TransportGradientAudit(
            loss=0.1,
            epsilon=1.0e-5,
            tolerance=5.0e-4,
            checked_indices=((0, 0),),
            chi_max_abs_error=1.0e-9,
            source_max_abs_error=1.0e-9,
            passed=True,
        )
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(nmpc_mod, "transport_loss_gradient", lambda *a, **k: (0.1, gradient))
        monkeypatch.setattr(nmpc_mod, "assert_transport_parameter_gradients_consistent", lambda *a, **k: audit)

        result = nmpc_mod.tune_transport_coefficients_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge,
            learning_rate=1.0,
            max_fractional_update=None,
        )
        np.testing.assert_allclose(result.updated_chi, np.maximum(0.0, chi - gradient))


class TestSourceScheduleTuningGuards:
    def test_rejects_nonpositive_learning_rate(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            nmpc_mod.tune_transport_sources_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.0
            )

    def test_rejects_nonpositive_max_absolute_update(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="max_absolute_update must be positive"):
            nmpc_mod.tune_transport_sources_for_tracking(
                profiles,
                chi,
                sources,
                target,
                rho,
                1.0e-3,
                edge,
                learning_rate=0.05,
                max_absolute_update=0.0,
            )

    def test_rejects_non_two_dimensional_sources(self, monkeypatch):
        profiles, chi, _sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="sources must be a finite two-dimensional"):
            nmpc_mod.tune_transport_sources_for_tracking(
                profiles, chi, np.zeros(4), target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_rejects_non_dataclass_gradient_result(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(nmpc_mod, "transport_parameter_gradients", lambda *a, **k: object())
        with pytest.raises(ValueError, match="must return TransportParameterGradients"):
            nmpc_mod.tune_transport_sources_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_rejects_source_gradient_shape_mismatch(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        result = transport_mod.TransportParameterGradients(
            loss=0.1, chi_gradient=np.zeros_like(chi), source_gradient=np.zeros((2, 2))
        )
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(nmpc_mod, "transport_parameter_gradients", lambda *a, **k: result)
        with pytest.raises(ValueError, match="source gradient must be finite and match source shape"):
            nmpc_mod.tune_transport_sources_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )


class TestSourceRolloutTuningGuards:
    def _args(self) -> dict[str, Any]:
        return {
            "initial_profiles": np.zeros((4, 5)),
            "chi": np.zeros((4, 5)),
            "source_sequence": np.zeros((3, 4, 5)),
            "target_history": np.zeros((3, 4, 5)),
            "rho": np.linspace(0.1, 1.0, 5),
            "dt": 1.0e-3,
            "edge_values": np.zeros(4),
        }

    def _call(self, args: dict[str, Any], **kw: Any) -> Any:
        return nmpc_mod.tune_transport_source_rollout_for_tracking(
            args["initial_profiles"],
            args["chi"],
            args["source_sequence"],
            args["target_history"],
            args["rho"],
            args["dt"],
            args["edge_values"],
            **kw,
        )

    def test_rejects_nonpositive_learning_rate(self, monkeypatch):
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            self._call(self._args(), learning_rate=0.0)

    def test_rejects_nonpositive_max_absolute_update(self, monkeypatch):
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="max_absolute_update must be positive"):
            self._call(self._args(), learning_rate=0.05, max_absolute_update=0.0)

    def test_rejects_inverted_source_bounds(self, monkeypatch):
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="source_min entries must be less than or equal"):
            self._call(self._args(), learning_rate=0.05, source_min=1.0, source_max=0.0)

    def test_rejects_non_dataclass_gradient_result(self, monkeypatch):
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(nmpc_mod, "transport_rollout_source_gradients", lambda *a, **k: object())
        with pytest.raises(ValueError, match="must return TransportRolloutSourceGradients"):
            self._call(self._args(), learning_rate=0.05)

    def test_rejects_source_gradient_shape_mismatch(self, monkeypatch):
        result = transport_mod.TransportRolloutSourceGradients(
            loss=0.1, source_gradient=np.zeros((2, 2)), final_profiles=np.zeros((4, 5))
        )
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(nmpc_mod, "transport_rollout_source_gradients", lambda *a, **k: result)
        with pytest.raises(ValueError, match="rollout source gradient must be finite and match"):
            self._call(self._args(), learning_rate=0.05)

    def test_rejects_final_profile_shape_mismatch(self, monkeypatch):
        result = transport_mod.TransportRolloutSourceGradients(
            loss=0.1, source_gradient=np.zeros((3, 4, 5)), final_profiles=np.zeros((2, 2))
        )
        monkeypatch.setattr(nmpc_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(nmpc_mod, "transport_rollout_source_gradients", lambda *a, **k: result)
        with pytest.raises(ValueError, match="rollout final profiles must be finite and match"):
            self._call(self._args(), learning_rate=0.05)


# ── configuration validation ──────────────────────────────────────────


class TestConfigValidation:
    def test_rejects_non_integer_max_sqp_iter(self):
        with pytest.raises(ValueError, match="max_sqp_iter must be an integer"):
            NonlinearMPC(_identity_plant, NMPCConfig(max_sqp_iter=2.5))

    def test_rejects_non_integer_qp_max_iter(self):
        with pytest.raises(ValueError, match="qp_max_iter must be an integer"):
            NonlinearMPC(_identity_plant, NMPCConfig(qp_max_iter=2.5))

    def test_rejects_collapsed_state_bounds(self):
        cfg = NMPCConfig()
        cfg.x_min = cfg.x_max.copy()
        with pytest.raises(ValueError, match="x_min entries must be strictly less"):
            NonlinearMPC(_identity_plant, cfg)

    def test_rejects_nonpositive_slew_limit(self):
        cfg = NMPCConfig()
        cfg.du_max = np.array([0.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="du_max entries must be positive"):
            NonlinearMPC(_identity_plant, cfg)


# ── condensed QP curvature degeneracy ─────────────────────────────────


class TestQPStepSizeCurvature:
    def _controller(self) -> NonlinearMPC:
        return NonlinearMPC(_identity_plant, NMPCConfig(horizon=1, max_sqp_iter=1))

    def test_falls_back_to_spectral_norm_on_eigensolver_failure(self, monkeypatch):
        nmpc = self._controller()
        A_k = [np.eye(6)]
        B_k = [np.eye(6, 3)]

        def boom(_matrix):
            raise np.linalg.LinAlgError("forced eigensolver failure")

        monkeypatch.setattr(np.linalg, "eigvalsh", boom)
        step = nmpc._estimate_qp_step_size(A_k, B_k, np.eye(6))
        assert step > 0.0 and np.isfinite(step)

    def test_rejects_nonpositive_curvature(self, monkeypatch):
        nmpc = self._controller()
        A_k = [np.eye(6)]
        B_k = [np.eye(6, 3)]
        monkeypatch.setattr(np.linalg, "eigvalsh", lambda _m: np.zeros(_m.shape[0]))
        with pytest.raises(ValueError, match="Hessian curvature must be positive finite"):
            nmpc._estimate_qp_step_size(A_k, B_k, np.eye(6))


# ── linearisation guards ──────────────────────────────────────────────


class TestLinearizationGuards:
    def test_finite_difference_column_rejects_collapsed_interval(self):
        with pytest.raises(ValueError, match="perturbation interval collapsed"):
            NonlinearMPC._finite_difference_column(None, np.zeros(6), None, 1.0e-4)

    def test_rejects_analytic_input_jacobian_shape(self):
        def bad_b(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.eye(6), np.zeros((6, 2))

        nmpc = NonlinearMPC(_identity_plant, NMPCConfig(horizon=2), linearization_model=bad_b)
        with pytest.raises(ValueError, match=r"finite B with shape \(6, 3\)"):
            nmpc.linearize(np.zeros(6), np.zeros(3))

    def _jax_controller(self, monkeypatch, jacfwd) -> NonlinearMPC:
        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.float64 = np.float64
        fake_jnp.asarray = np.asarray
        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.numpy = fake_jnp
        fake_jax.jacfwd = jacfwd
        monkeypatch.setitem(sys.modules, "jax", fake_jax)
        monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
        cfg = NMPCConfig(horizon=2)
        cfg.linearization_backend = "jax"
        return NonlinearMPC(_identity_plant, cfg)

    def test_jax_linearization_wraps_trace_failure(self, monkeypatch):
        def jacfwd(fn, argnums):
            def wrapped(x_arg, u_arg):
                raise RuntimeError("untraceable")

            return wrapped

        nmpc = self._jax_controller(monkeypatch, jacfwd)
        with pytest.raises(RuntimeError, match="must be JAX-traceable"):
            nmpc.linearize(np.zeros(6), np.zeros(3))

    def test_jax_linearization_rejects_wrong_state_jacobian_shape(self, monkeypatch):
        def jacfwd(fn, argnums):
            def wrapped(x_arg, u_arg):
                return np.zeros((5, 6)), np.zeros((6, 3))

            return wrapped

        nmpc = self._jax_controller(monkeypatch, jacfwd)
        with pytest.raises(ValueError, match=r"finite A with shape \(6, 6\)"):
            nmpc.linearize(np.zeros(6), np.zeros(3))

    def test_jax_linearization_rejects_wrong_input_jacobian_shape(self, monkeypatch):
        def jacfwd(fn, argnums):
            def wrapped(x_arg, u_arg):
                return np.eye(6), np.zeros((6, 2))

            return wrapped

        nmpc = self._jax_controller(monkeypatch, jacfwd)
        with pytest.raises(ValueError, match=r"finite B with shape \(6, 3\)"):
            nmpc.linearize(np.zeros(6), np.zeros(3))


# ── established-solver backends ───────────────────────────────────────


class TestSolverBackends:
    def test_scipy_backend_fails_closed_on_solver_failure(self, monkeypatch):
        import scipy.optimize

        cfg = NMPCConfig(horizon=2, max_sqp_iter=1, qp_max_iter=10)
        cfg.qp_backend = "scipy"
        nmpc = NonlinearMPC(_identity_plant, cfg)

        class _FailResult:
            success = False
            message = "forced SLSQP failure"
            x = np.zeros(cfg.horizon * 3)
            nit = 1

        monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: _FailResult())
        with pytest.raises(RuntimeError, match="SciPy QP backend failed"):
            nmpc.step(np.ones(6), np.ones(6), cfg.u_min.copy())

    def test_osqp_backend_fails_closed_on_nonconverged_status(self, monkeypatch):
        osqp = pytest.importorskip("osqp")

        cfg = NMPCConfig(horizon=2, max_sqp_iter=1, qp_max_iter=10)
        cfg.qp_backend = "osqp"
        nmpc = NonlinearMPC(_identity_plant, cfg)

        class _FakeInfo:
            iter = 5
            status_val = 3
            status = "primal infeasible"

        class _FakeResult:
            info = _FakeInfo()
            x = np.zeros(cfg.horizon * 3)

        class _FakeOSQP:
            def setup(self, **kwargs: Any) -> None:
                return None

            def solve(self) -> _FakeResult:
                return _FakeResult()

        monkeypatch.setattr(osqp, "OSQP", _FakeOSQP)
        with pytest.raises(RuntimeError, match="OSQP backend failed"):
            nmpc.step(np.ones(6), np.ones(6), cfg.u_min.copy())

    def test_osqp_backend_enforces_terminal_state_set(self):
        pytest.importorskip("osqp")
        _A, _B, plant, linearization = _linear_plant()
        cfg = NMPCConfig(horizon=1, max_sqp_iter=2, qp_max_iter=400)
        cfg.qp_backend = "osqp"
        cfg.R = 1.0e-3 * np.eye(3)
        cfg.terminal_x_min = cfg.x_min.copy()
        cfg.terminal_x_min[0] = 2.0
        cfg.terminal_x_max = cfg.x_max.copy()
        nmpc = NonlinearMPC(plant, cfg, linearization_model=linearization)
        x0 = np.array([1.0, 1.0, 3.0, 1.0, 5.0, 1.0])
        nmpc.step(x0, x0.copy(), np.array([0.0, 1.0, 0.0]))
        assert nmpc.x_traj[-1, 0] >= cfg.terminal_x_min[0] - 1.0e-6

    def test_casadi_backend_solves_terminal_constrained_qp(self):
        pytest.importorskip("casadi")
        _A, _B, plant, linearization = _linear_plant()
        cfg = NMPCConfig(horizon=2, max_sqp_iter=2, qp_max_iter=200)
        cfg.qp_backend = "casadi"
        cfg.R = 1.0e-3 * np.eye(3)
        cfg.terminal_x_min = cfg.x_min.copy()
        cfg.terminal_x_min[0] = 2.0
        cfg.terminal_x_max = cfg.x_max.copy()
        nmpc = NonlinearMPC(plant, cfg, linearization_model=linearization)
        x0 = np.array([1.0, 1.0, 3.0, 1.0, 5.0, 1.0])
        u_opt = nmpc.step(x0, x0.copy(), np.array([0.0, 1.0, 0.0]))
        assert u_opt.shape == (3,)
        assert nmpc.last_qp_backend == "casadi"
        assert nmpc.x_traj[-1, 0] >= cfg.terminal_x_min[0] - 1.0e-6


# ── acados native marshalling helpers ─────────────────────────────────


class TestAcadosMarshallingHelpers:
    def test_set_wraps_native_failure(self):
        class _RaisingSet:
            def set(self, *a: Any) -> None:
                raise ValueError("native set boom")

        with pytest.raises(RuntimeError, match="setting yref at stage 2"):
            NonlinearMPC._acados_set(_RaisingSet(), 2, "yref", np.zeros(3))

    def test_get_wraps_native_failure(self):
        class _RaisingGet:
            def get(self, *a: Any) -> None:
                raise ValueError("native get boom")

        with pytest.raises(RuntimeError, match="reading x at stage 1"):
            NonlinearMPC._acados_get(_RaisingGet(), 1, "x")

    def test_iterations_zero_without_stats(self):
        assert NonlinearMPC._acados_iterations(object()) == 0

    def test_iterations_zero_when_stats_raise(self):
        class _RaisingStats:
            def get_stats(self, field: str) -> None:
                raise KeyError(field)

        assert NonlinearMPC._acados_iterations(_RaisingStats()) == 0

    def test_iterations_zero_for_empty_stats(self):
        class _EmptyStats:
            def get_stats(self, field: str) -> np.ndarray:
                return np.array([])

        assert NonlinearMPC._acados_iterations(_EmptyStats()) == 0

    def test_make_solver_imports_native_runtime(self, monkeypatch):
        constructed: list[object] = []

        class _NativeSolver:
            def __init__(self, ocp: object, **kwargs: Any) -> None:
                constructed.append(ocp)

        fake_template = types.ModuleType("acados_template")
        fake_template.AcadosOcpSolver = _NativeSolver
        monkeypatch.setitem(sys.modules, "acados_template", fake_template)

        cfg = NMPCConfig(horizon=1, max_sqp_iter=1)
        cfg.qp_backend = "acados"
        nmpc = NonlinearMPC(_identity_plant, cfg, acados_ocp_factory=lambda config, tc: object())
        ocp_sentinel = object()
        solver = nmpc._make_acados_solver(ocp_sentinel)
        assert isinstance(solver, _NativeSolver)
        assert constructed == [ocp_sentinel]


# ── acados solve-boundary fail-closed validation ──────────────────────


class _BoundaryAcadosSolver:
    """Configurable acados double returning caller-supplied control/state stages."""

    def __init__(self, u_func: Any, x_func: Any) -> None:
        self._u_func = u_func
        self._x_func = x_func

    def set(self, stage: int, field: str, value: Any) -> None:
        return None

    def solve(self) -> int:
        return 0

    def get_stats(self, field: str) -> int:
        return 1

    def get(self, stage: int, field: str) -> np.ndarray:
        if field == "u":
            return np.asarray(self._u_func(stage), dtype=float)
        if field == "x":
            return np.asarray(self._x_func(stage), dtype=float)
        raise KeyError(field)


def _acados_nmpc(solver: object, *, horizon: int = 2) -> NonlinearMPC:
    cfg = NMPCConfig(horizon=horizon, max_sqp_iter=1)
    cfg.qp_backend = "acados"
    cfg.P = 2.0 * np.eye(6)
    return NonlinearMPC(
        _identity_plant,
        cfg,
        acados_ocp_factory=lambda config, tc: object(),
        acados_solver_factory=lambda ocp, **kw: solver,
    )


_X0 = np.array([1.0, 1.0, 5.0, 1.0, 2.0, 1.0])
_U_PREV = np.array([2.0, 1.5, 0.5])


def _valid_u(_stage: int) -> np.ndarray:
    return _U_PREV.copy()


def _aug_state(phys: np.ndarray) -> np.ndarray:
    return np.r_[phys, np.zeros(3)]


class TestAcadosSolveBoundary:
    def test_rejects_nonfinite_control_trajectory(self):
        solver = _BoundaryAcadosSolver(lambda s: np.array([np.nan, 1.5, 0.5]), lambda s: _aug_state(_X0))
        nmpc = _acados_nmpc(solver)
        with pytest.raises(RuntimeError, match="invalid control trajectory"):
            nmpc.step(_X0, _X0.copy(), _U_PREV)

    def test_rejects_control_outside_actuator_bounds(self):
        solver = _BoundaryAcadosSolver(lambda s: np.array([200.0, 1.5, 0.5]), lambda s: _aug_state(_X0))
        nmpc = _acados_nmpc(solver)
        with pytest.raises(RuntimeError, match="outside configured actuator bounds"):
            nmpc.step(_X0, _X0.copy(), _U_PREV)

    def test_rejects_control_outside_slew_bounds(self):
        solver = _BoundaryAcadosSolver(
            lambda s: _U_PREV.copy() if s == 0 else np.array([50.0, 1.5, 0.5]),
            lambda s: _aug_state(_X0),
        )
        nmpc = _acados_nmpc(solver)
        with pytest.raises(RuntimeError, match="outside configured slew-rate bounds"):
            nmpc.step(_X0, _X0.copy(), _U_PREV)

    def test_rejects_nonfinite_state_trajectory(self):
        solver = _BoundaryAcadosSolver(_valid_u, lambda s: _aug_state(np.full(6, np.nan)))
        nmpc = _acados_nmpc(solver)
        with pytest.raises(RuntimeError, match="invalid state trajectory"):
            nmpc.step(_X0, _X0.copy(), _U_PREV)

    def test_rejects_inconsistent_initial_state(self):
        solver = _BoundaryAcadosSolver(_valid_u, lambda s: _aug_state(_X0 + 100.0))
        nmpc = _acados_nmpc(solver)
        with pytest.raises(RuntimeError, match="invalid initial state"):
            nmpc.step(_X0, _X0.copy(), _U_PREV)

    def test_rejects_state_outside_physics_bounds(self):
        def x_func(stage: int) -> np.ndarray:
            if stage == 0:
                return _aug_state(_X0)
            out_of_bounds = _X0.copy()
            out_of_bounds[0] = 1.0e6
            return _aug_state(out_of_bounds)

        solver = _BoundaryAcadosSolver(_valid_u, x_func)
        nmpc = _acados_nmpc(solver)
        with pytest.raises(RuntimeError, match="outside configured physics bounds"):
            nmpc.step(_X0, _X0.copy(), _U_PREV)

    def test_discard_warns_when_native_release_fails(self):
        class _FailFreeStatusSolver:
            def set(self, *a: Any) -> None:
                return None

            def solve(self) -> int:
                return 7

            def get_stats(self, field: str) -> int:
                return 1

            def free(self) -> None:
                raise RuntimeError("native free boom")

        nmpc = _acados_nmpc(_FailFreeStatusSolver())
        with (
            pytest.warns(RuntimeWarning, match="cleanup failed after solver fault"),
            pytest.raises(RuntimeError, match="failed with status 7"),
        ):
            nmpc.step(_X0, _X0.copy(), _U_PREV)


class TestAcadosContextLifetime:
    def test_clean_exit_releases_solver(self):
        freed: list[int] = []

        class _ReleasableSolver:
            def free(self) -> None:
                freed.append(1)

        nmpc = _acados_nmpc(_ReleasableSolver(), horizon=1)
        nmpc._acados_solver = _ReleasableSolver()
        with nmpc:
            pass
        assert nmpc._acados_solver is None
