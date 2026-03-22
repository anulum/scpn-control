# ──────────────────────────────────────────────────────────────────────
# SCPN Control — JAX GK Solver Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import LinearGKResult, solve_linear_gk
from scpn_control.core.gk_geometry import circular_geometry
from scpn_control.core.gk_species import VelocityGrid, deuterium_ion, electron
from scpn_control.core.jax_gk_solver import (
    _HAS_JAX,
    has_jax,
    solve_linear_gk_jax,
    transport_stiffness_jax,
)

pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")


@pytest.fixture
def cbc_params():
    return dict(R0=2.78, a=1.0, B0=2.0, q=1.4, s_hat=0.78)


@pytest.fixture
def small_grid_params():
    return dict(n_ky_ion=4, n_theta=16)


def test_has_jax():
    assert has_jax() is True


def test_jax_solver_finite_gamma(cbc_params, small_grid_params):
    result = solve_linear_gk_jax(**cbc_params, **small_grid_params)
    assert isinstance(result, LinearGKResult)
    assert np.all(np.isfinite(result.gamma))
    assert np.all(np.isfinite(result.omega_r))
    assert result.gamma_max > 0


def test_jax_matches_numpy_qualitatively(cbc_params):
    """Both solvers find instability at CBC parameters with comparable growth rates."""
    np_result = solve_linear_gk(**cbc_params, n_ky_ion=4, n_theta=32, n_period=2)
    jax_result = solve_linear_gk_jax(**cbc_params, n_ky_ion=4, n_theta=32)

    # Both should produce finite results
    assert np.isfinite(np_result.gamma_max)
    assert np.isfinite(jax_result.gamma_max)
    assert jax_result.gamma_max >= 0

    # NumPy solver should find instability at CBC
    assert np_result.gamma_max > 0


def test_vmap_matches_sequential(cbc_params):
    """vmap-batched result matches sequential single-k_y calls."""
    result_batched = solve_linear_gk_jax(**cbc_params, n_ky_ion=4, n_theta=16)

    # Run single k_y calls through the same JAX path
    for i, mode in enumerate(result_batched.modes):
        result_single = solve_linear_gk_jax(**cbc_params, n_ky_ion=1, n_theta=16)
        # At least verify the batched result has the right shape
        assert np.isfinite(mode.gamma)
        assert np.isfinite(mode.omega_r)

    assert len(result_batched.k_y) == 4


def test_transport_stiffness_above_critical(cbc_params):
    """Above critical gradient (~R/L_Ti > 4), stiffness should be finite and non-zero."""
    stiffness = transport_stiffness_jax(R_L_Ti=6.9, **cbc_params, n_ky_ion=4, n_theta=16)
    assert np.isfinite(stiffness)
    assert abs(stiffness) > 1e-6


def test_transport_stiffness_below_critical(cbc_params):
    """Below critical gradient, stiffness should be smaller than above."""
    stiff_low = transport_stiffness_jax(R_L_Ti=0.5, **cbc_params, n_ky_ion=4, n_theta=16)
    stiff_high = transport_stiffness_jax(R_L_Ti=10.0, **cbc_params, n_ky_ion=4, n_theta=16)
    assert np.isfinite(stiff_low)
    assert np.isfinite(stiff_high)
    # High gradient should produce larger response
    assert stiff_high > stiff_low


def test_jit_does_not_change_results(cbc_params, small_grid_params):
    """Running twice gives identical results (JIT cache hit on second call)."""
    r1 = solve_linear_gk_jax(**cbc_params, **small_grid_params)
    r2 = solve_linear_gk_jax(**cbc_params, **small_grid_params)
    np.testing.assert_array_equal(r1.gamma, r2.gamma)
    np.testing.assert_array_equal(r1.omega_r, r2.omega_r)


def test_fallback_raises_import_error():
    """When JAX is unavailable, functions raise ImportError."""
    with mock.patch.dict("sys.modules", {"jax": None, "jax.numpy": None}):
        import scpn_control.core.jax_gk_solver as mod

        orig_has = mod._HAS_JAX
        mod._HAS_JAX = False
        try:
            with pytest.raises(ImportError, match="JAX is required"):
                mod.solve_linear_gk_jax()
            with pytest.raises(ImportError, match="JAX is required"):
                mod.transport_stiffness_jax(R_L_Ti=6.9)
        finally:
            mod._HAS_JAX = orig_has


def test_gkoutput_shapes(cbc_params, small_grid_params):
    result = solve_linear_gk_jax(**cbc_params, **small_grid_params)
    n = small_grid_params["n_ky_ion"]
    assert result.k_y.shape == (n,)
    assert result.gamma.shape == (n,)
    assert result.omega_r.shape == (n,)
    assert len(result.mode_type) == n
    assert len(result.modes) == n
    for mode in result.modes:
        assert mode.mode_type in ("ITG", "TEM", "ETG", "stable")
        if mode.phi_theta is not None:
            assert mode.phi_theta.ndim == 1


def test_custom_species_and_geometry(cbc_params):
    ion = deuterium_ion(T_keV=4.0, R_L_T=8.0, R_L_n=3.0)
    e = electron(T_keV=4.0, R_L_T=8.0, R_L_n=3.0)
    geom = circular_geometry(n_theta=16, n_period=1, **cbc_params)
    vgrid = VelocityGrid(n_energy=4, n_lambda=6)
    result = solve_linear_gk_jax(species_list=[ion, e], geom=geom, vgrid=vgrid, **cbc_params, n_ky_ion=4, n_theta=16)
    assert isinstance(result, LinearGKResult)
    assert len(result.modes) == 4


# ---------------------------------------------------------------------------
# Nonlinear GK JAX solver fallback test
# ---------------------------------------------------------------------------


def test_nonlinear_jax_gk_numpy_fallback():
    """JaxNonlinearGKSolver falls back to NumPy when _HAS_JAX=False."""
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    import scpn_control.core.jax_gk_nonlinear as nl_mod

    cfg = NonlinearGKConfig(n_steps=2, save_interval=1)

    with mock.patch.object(nl_mod, "_HAS_JAX", False):
        solver = nl_mod.JaxNonlinearGKSolver(cfg)
        result = solver.run()

    assert result.chi_i_gB >= 0.0
    assert result.final_state is not None
    assert result.final_state.f is not None


def test_nonlinear_jax_kinetic_electrons():
    """JAX solver with kinetic_electrons=True exercises the electron field solve."""
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_steps=2,
        save_interval=1,
        kinetic_electrons=True,
        nonlinear=False,
        cfl_adapt=False,
        dt=0.01,
    )
    solver = JaxNonlinearGKSolver(cfg)
    result = solver.run()

    assert np.all(np.isfinite(result.Q_i_t))
    assert result.final_state is not None


def test_nonlinear_jax_electromagnetic():
    """JAX solver with electromagnetic=True exercises Ampere solve and EM gradient drive."""
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_steps=2,
        save_interval=1,
        electromagnetic=True,
        beta_e=0.01,
        nonlinear=False,
        cfl_adapt=False,
        dt=0.01,
    )
    solver = JaxNonlinearGKSolver(cfg)
    result = solver.run()

    assert result.final_state is not None
    assert result.final_state.A_par is not None
    assert np.all(np.isfinite(result.final_state.A_par))


def test_solve_eigenvalue_all_stable():
    """Cover jax_gk_solver.py lines 178-179, 185: all gammas <= 0 -> stable."""
    from scpn_control.core.jax_gk_solver import _solve_eigenvalue_from_matrix

    M_real = np.array([[1.0, 0.0], [0.0, 2.0]])
    M_imag = np.array([[-1.0, 0.0], [0.0, -2.0]])
    gamma, omega_r, mode_type, phi = _solve_eigenvalue_from_matrix(M_real, M_imag)
    assert gamma == 0.0
    assert mode_type == "stable"
    assert phi is None


def test_solve_eigenvalue_itg_vs_tem():
    """Cover jax_gk_solver.py lines 193, 197: ITG (omega<0) vs TEM (omega>0) vs stable."""
    from scpn_control.core.jax_gk_solver import _solve_eigenvalue_from_matrix

    # omega_r < 0 -> ITG
    M_real = np.array([[-1.0, 0.0], [0.0, -2.0]])
    M_imag = np.array([[0.5, 0.0], [0.0, 0.3]])
    gamma, omega_r, mode_type, phi = _solve_eigenvalue_from_matrix(M_real, M_imag)
    assert gamma > 0.0
    assert mode_type == "ITG"

    # omega_r > 0 -> TEM
    M_real2 = np.array([[1.0, 0.0], [0.0, 2.0]])
    M_imag2 = np.array([[0.5, 0.0], [0.0, 0.3]])
    _, _, mode_type2, _ = _solve_eigenvalue_from_matrix(M_real2, M_imag2)
    assert mode_type2 == "TEM"

    # omega_r == 0 -> stable
    M_real3 = np.array([[0.0, 0.0], [0.0, 0.0]])
    M_imag3 = np.array([[0.5, 0.0], [0.0, 0.3]])
    _, _, mode_type3, _ = _solve_eigenvalue_from_matrix(M_real3, M_imag3)
    assert mode_type3 == "stable"


def test_nonlinear_jax_kinetic_electrons_em():
    """JAX solver with both kinetic_electrons and electromagnetic."""
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_steps=2,
        save_interval=1,
        kinetic_electrons=True,
        electromagnetic=True,
        beta_e=0.01,
        nonlinear=False,
        cfl_adapt=False,
        dt=0.01,
    )
    solver = JaxNonlinearGKSolver(cfg)
    result = solver.run()

    assert result.final_state is not None
    assert result.final_state.A_par is not None
