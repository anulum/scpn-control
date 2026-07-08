# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Jax Solvers
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — JAX Transport Solver Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for JAX-accelerated transport solver primitives.

Validates NumPy/JAX parity, boundary conditions, conservation, and
batched transport via vmap.
"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import ModuleType
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from scpn_control.core.jax_solvers import (
    _diffusion_rhs_np,
    _thomas_solve_np,
    batched_crank_nicolson,
    crank_nicolson_step,
    diffusion_rhs,
    has_jax,
    has_jax_gpu,
    thomas_solve,
)

# Skip JAX-specific tests if JAX not installed
jax_available = has_jax()


class TestThomasSolveNumpy:
    """Verify NumPy Thomas solver against known tridiagonal systems."""

    def test_vanishing_leading_pivot_is_floored(self) -> None:
        """A zero leading diagonal entry is floored so forward elimination is finite.

        The Thomas sweep divides by the running pivot ``m``; when the first
        diagonal entry is zero the solver floors ``m`` to ``1e-30`` rather than
        dividing by zero, keeping the eliminated coefficients finite.
        """
        a = np.array([1.0, 1.0])
        b = np.array([0.0, 2.0, 2.0])
        c = np.array([1.0, 1.0])
        d = np.array([1.0, 2.0, 3.0])
        x = _thomas_solve_np(a, b, c, d)
        assert x.shape == (3,)
        assert np.all(np.isfinite(x))

    def test_vanishing_inner_pivot_is_floored(self) -> None:
        """A later zero pivot is floored during forward elimination."""
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0, 2.0])
        c = np.array([1.0, 1.0])
        d = np.array([1.0, 2.0, 3.0])
        x = _thomas_solve_np(a, b, c, d)
        assert x.shape == (3,)
        assert np.all(np.isfinite(x))

    def test_identity_system(self) -> None:
        """I x = d => x = d."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        c = np.zeros(n - 1)
        d = np.arange(n, dtype=float)
        x = _thomas_solve_np(a, b, c, d)
        np.testing.assert_allclose(x, d, atol=1e-12)

    def test_simple_tridiag(self) -> None:
        """Known 4x4 system: [2,-1,0,0; -1,2,-1,0; 0,-1,2,-1; 0,0,-1,2] x = [1,0,0,1]."""
        a = np.array([-1.0, -1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0, -1.0])
        d = np.array([1.0, 0.0, 0.0, 1.0])
        x = _thomas_solve_np(a, b, c, d)
        # Verify A @ x = d
        result = b * x
        result[:-1] += c * x[1:]
        result[1:] += a * x[:-1]
        np.testing.assert_allclose(result, d, atol=1e-10)

    def test_diffusion_like_matrix(self) -> None:
        """Tridiag from 1D diffusion discretisation."""
        n = 32
        a = -0.5 * np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -0.5 * np.ones(n - 1)
        rng = np.random.default_rng(42)
        d = rng.standard_normal(n)
        x = _thomas_solve_np(a, b, c, d)
        # Verify residual
        result = b * x
        result[:-1] += c * x[1:]
        result[1:] += a * x[:-1]
        np.testing.assert_allclose(result, d, atol=1e-10)


class TestDiffusionRhsNumpy:
    """Verify NumPy diffusion operator."""

    def test_constant_profile_zero_flux(self) -> None:
        """Uniform T => L_h(T) = 0 everywhere."""
        n = 32
        rho = np.linspace(0.01, 1.0, n)
        T = 5.0 * np.ones(n)
        chi = 0.5 * np.ones(n)
        drho = rho[1] - rho[0]
        Lh = _diffusion_rhs_np(T, chi, rho, drho)
        np.testing.assert_allclose(Lh, 0.0, atol=1e-12)

    def test_linear_profile(self) -> None:
        """Linear T(r) = r => nonzero L_h due to cylindrical geometry."""
        n = 64
        rho = np.linspace(0.01, 1.0, n)
        T = rho.copy()
        chi = np.ones(n)
        drho = rho[1] - rho[0]
        Lh = _diffusion_rhs_np(T, chi, rho, drho)
        # Interior should be nonzero (1/r correction)
        assert np.any(np.abs(Lh[1:-1]) > 1e-6)

    def test_boundary_values_zero(self) -> None:
        """Boundaries (i=0, i=n-1) should be zero (not computed)."""
        n = 32
        rho = np.linspace(0.01, 1.0, n)
        T = np.sin(np.pi * rho)
        chi = np.ones(n)
        drho = rho[1] - rho[0]
        Lh = _diffusion_rhs_np(T, chi, rho, drho)
        assert Lh[0] == 0.0
        assert Lh[-1] == 0.0


class TestCrankNicolsonStep:
    """Verify Crank-Nicolson transport step."""

    def test_no_source_diffuses(self) -> None:
        """Without sources, peaked profile should flatten."""
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 10.0 * np.exp(-((rho - 0.3) ** 2) / 0.05)
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)
        T_new = crank_nicolson_step(T, chi, source, rho, drho, dt=0.01, use_jax=False)
        # Peak should decrease
        assert np.max(T_new[1:-1]) < np.max(T[1:-1])

    def test_edge_bc_applied(self) -> None:
        """Edge boundary condition should be enforced."""
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 5.0 * np.ones(n)
        chi = 0.1 * np.ones(n)
        source = np.zeros(n)
        T_new = crank_nicolson_step(T, chi, source, rho, drho, dt=0.01, T_edge=0.1, use_jax=False)
        assert T_new[-1] == pytest.approx(0.1)
        # Core Neumann: T[0] = T[1]
        assert T_new[0] == pytest.approx(T_new[1])


@pytest.mark.skipif(not jax_available, reason="JAX not installed")
class TestJaxParity:
    """Verify JAX implementations match NumPy within tolerance."""

    def test_has_jax_gpu_reports_a_boolean(self) -> None:
        """With JAX present the GPU probe queries devices and returns a bool."""
        assert isinstance(has_jax_gpu(), bool)

    def test_has_jax_gpu_returns_false_when_device_query_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backend that fails to enumerate devices is treated as no GPU.

        ``jax.devices()`` can raise when a configured accelerator backend cannot
        be initialised; the probe swallows that and reports no GPU rather than
        propagating the backend error.
        """

        def _raise() -> object:
            raise RuntimeError("device backend unavailable")

        monkeypatch.setattr("scpn_control.core.jax_solvers.jax.devices", _raise)
        assert has_jax_gpu() is False

    def test_thomas_parity(self) -> None:
        n = 32
        rng = np.random.default_rng(42)
        a = -0.3 * np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -0.3 * np.ones(n - 1)
        d = rng.standard_normal(n)

        x_np = thomas_solve(a, b, c, d, use_jax=False)
        x_jax = thomas_solve(a, b, c, d, use_jax=True)
        np.testing.assert_allclose(x_jax, x_np, atol=1e-10)

    def test_diffusion_rhs_parity(self) -> None:
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 10.0 * np.exp(-((rho - 0.5) ** 2) / 0.1)
        chi = 0.5 + 0.2 * rho

        Lh_np = diffusion_rhs(T, chi, rho, drho, use_jax=False)
        Lh_jax = diffusion_rhs(T, chi, rho, drho, use_jax=True)
        np.testing.assert_allclose(Lh_jax, Lh_np, atol=1e-8)

    def test_cn_step_parity(self) -> None:
        n = 32
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T = 10.0 * np.exp(-((rho - 0.3) ** 2) / 0.05)
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)

        T_np = crank_nicolson_step(T, chi, source, rho, drho, 0.01, use_jax=False)
        T_jax = crank_nicolson_step(T, chi, source, rho, drho, 0.01, use_jax=True)
        np.testing.assert_allclose(T_jax, T_np, atol=1e-8)

    def test_batched_cn(self) -> None:
        n = 32
        batch = 8
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        rng = np.random.default_rng(7)
        T_batch = 5.0 + rng.standard_normal((batch, n))
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)

        result = batched_crank_nicolson(T_batch, chi, source, rho, drho, 0.01)
        assert result.shape == (batch, n)
        # Each trajectory should satisfy BCs
        for i in range(batch):
            assert result[i, -1] == pytest.approx(0.1)
            assert result[i, 0] == pytest.approx(result[i, 1])

    def test_jax_grad_thomas(self) -> None:
        """JAX autodiff through Thomas solver should produce finite gradients."""
        import jax
        import jax.numpy as jnp
        from scpn_control.core.jax_solvers import _thomas_solve_jax_impl

        n = 16
        a = -0.3 * jnp.ones(n - 1)
        b = 2.0 * jnp.ones(n)
        c = -0.3 * jnp.ones(n - 1)

        def loss(d: Any) -> Any:
            x = _thomas_solve_jax_impl(a, b, c, d)
            return jnp.sum(x**2)

        d = jnp.ones(n)
        grad = jax.grad(loss)(d)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_jax_grad_cn_step(self) -> None:
        """JAX autodiff through full CN step."""
        import jax
        import jax.numpy as jnp
        from scpn_control.core.jax_solvers import _cn_step_jax

        n = 16
        rho = jnp.linspace(0.05, 1.0, n)
        drho = float(rho[1] - rho[0])
        chi = 0.5 * jnp.ones(n)
        source = jnp.zeros(n)

        def loss(T: Any) -> Any:
            T_new = _cn_step_jax(T, chi, source, rho, drho, 0.01, 0.1)
            return jnp.sum(T_new**2)

        T = 5.0 * jnp.exp(-((rho - 0.3) ** 2) / 0.1)
        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


class TestBatchedTransportFallback:
    """Batched transport with NumPy fallback."""

    def test_batched_numpy_fallback(self) -> None:
        n = 16
        batch = 4
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        rng = np.random.default_rng(42)
        T_batch = 5.0 + rng.standard_normal((batch, n))
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)

        # Force NumPy path by comparing single vs batch
        results_single = np.stack(
            [crank_nicolson_step(T_batch[i], chi, source, rho, drho, 0.01, use_jax=False) for i in range(batch)]
        )
        # batched_crank_nicolson will use JAX if available, NumPy otherwise
        # Either way, results should match
        for i in range(batch):
            assert results_single[i, -1] == pytest.approx(0.1)


class TestStrictJaxFallbackContracts:
    """Fail-closed contracts for explicit JAX requests."""

    def test_use_jax_requires_explicit_legacy_fallback_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scpn_control.core.jax_solvers._HAS_JAX", False)
        a = np.array([-1.0, -1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0, -1.0])
        d = np.array([1.0, 0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="allow_legacy_numpy_fallback=True"):
            thomas_solve(a, b, c, d, use_jax=True, allow_numpy_fallback=True)

    def test_use_jax_fails_closed_without_jax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scpn_control.core.jax_solvers._HAS_JAX", False)
        a = np.array([-1.0, -1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0, -1.0])
        d = np.array([1.0, 0.0, 0.0, 1.0])
        with pytest.raises(RuntimeError, match="thomas_solve requested use_jax=True"):
            thomas_solve(a, b, c, d, use_jax=True)

    def test_use_jax_legacy_fallback_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scpn_control.core.jax_solvers._HAS_JAX", False)
        a = np.array([-1.0, -1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0, -1.0])
        d = np.array([1.0, 0.0, 0.0, 1.0])
        out = thomas_solve(
            a,
            b,
            c,
            d,
            use_jax=True,
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
        )
        np.testing.assert_allclose(out, _thomas_solve_np(a, b, c, d))

    def test_batched_contracts_without_jax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scpn_control.core.jax_solvers._HAS_JAX", False)
        n = 16
        batch = 3
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        rng = np.random.default_rng(12)
        T_batch = 5.0 + rng.standard_normal((batch, n))
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)

        with pytest.raises(RuntimeError, match="batched_crank_nicolson requested use_jax=True"):
            batched_crank_nicolson(T_batch, chi, source, rho, drho, 0.01)

        out = batched_crank_nicolson(
            T_batch,
            chi,
            source,
            rho,
            drho,
            0.01,
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
        )
        assert out.shape == T_batch.shape


def test_optional_jax_initialisation_failure_uses_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat a broken optional JAX install as unavailable, not as an import-time crash."""
    module_path = Path(__file__).parents[1] / "src" / "scpn_control" / "core" / "jax_solvers.py"
    spec = importlib.util.spec_from_file_location("jax_solvers_broken_optional_probe", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = ModuleType("jax_solvers_broken_optional_probe")
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> Any:
        if name == "jax" or name.startswith("jax."):
            raise AttributeError("simulated incompatible JAX runtime")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    spec.loader.exec_module(module)

    assert module.has_jax() is False
    with pytest.raises(RuntimeError, match="thomas_solve requested use_jax=True"):
        module.thomas_solve(
            np.array([-1.0, -1.0, -1.0]),
            np.array([2.0, 2.0, 2.0, 2.0]),
            np.array([-1.0, -1.0, -1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            use_jax=True,
        )
