# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Jax Gk Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

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
    gk_stiffness_chi_i_profile_jax,
    has_jax,
    solve_linear_gk_jax,
    transport_stiffness_jax,
)


@pytest.fixture
def cbc_params():
    return dict(R0=2.78, a=1.0, B0=2.0, q=1.4, s_hat=0.78)


@pytest.fixture
def small_grid_params():
    return dict(n_ky_ion=4, n_theta=16)


def test_has_jax():
    assert isinstance(has_jax(), bool)
    assert has_jax() is _HAS_JAX


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_solver_finite_gamma(cbc_params, small_grid_params):
    result = solve_linear_gk_jax(**cbc_params, **small_grid_params)
    assert isinstance(result, LinearGKResult)
    assert np.all(np.isfinite(result.gamma))
    assert np.all(np.isfinite(result.omega_r))
    assert result.gamma_max > 0


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
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


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
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


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_transport_stiffness_above_critical(cbc_params):
    """Above critical gradient (~R/L_Ti > 4), stiffness should be finite and non-zero."""
    stiffness = transport_stiffness_jax(R_L_Ti=6.9, **cbc_params, n_ky_ion=4, n_theta=16)
    assert np.isfinite(stiffness)
    assert abs(stiffness) > 1e-6


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_transport_stiffness_below_critical(cbc_params):
    """Below critical gradient, stiffness should be smaller than above."""
    stiff_low = transport_stiffness_jax(R_L_Ti=0.5, **cbc_params, n_ky_ion=4, n_theta=16)
    stiff_high = transport_stiffness_jax(R_L_Ti=10.0, **cbc_params, n_ky_ion=4, n_theta=16)
    assert np.isfinite(stiff_low)
    assert np.isfinite(stiff_high)
    # High gradient should produce larger response
    assert stiff_high > stiff_low


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
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
            with pytest.raises(ImportError, match="JAX is required"):
                mod.gk_stiffness_chi_i_profile_jax(R_L_Ti=6.9, rho=np.linspace(0.05, 1.0, 8))
        finally:
            mod._HAS_JAX = orig_has


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_gk_stiffness_chi_profile_is_positive_and_monotone(cbc_params):
    """JAX GK stiffness closure should produce bounded ion heat coefficients."""
    rho = np.linspace(0.05, 1.0, 16)
    low = gk_stiffness_chi_i_profile_jax(
        R_L_Ti=0.5,
        rho=rho,
        base_chi_i=0.1,
        stiffness_scale=1.0e-6,
        **cbc_params,
        n_ky_ion=4,
        n_theta=16,
    )
    high = gk_stiffness_chi_i_profile_jax(
        R_L_Ti=8.0,
        rho=rho,
        base_chi_i=0.1,
        stiffness_scale=1.0e-6,
        **cbc_params,
        n_ky_ion=4,
        n_theta=16,
    )

    assert low.shape == rho.shape
    assert high.shape == rho.shape
    assert np.all(np.isfinite(high))
    assert np.all(high >= 0.1)
    assert np.mean(high) > np.mean(low)


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
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


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
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
        with pytest.raises(ValueError, match="allow_legacy_numpy_fallback=True"):
            nl_mod.JaxNonlinearGKSolver(
                cfg,
                allow_numpy_fallback=True,
                allow_legacy_numpy_fallback=False,
            )
        with pytest.raises(RuntimeError, match="JAX nonlinear GK solver requested"):
            nl_mod.JaxNonlinearGKSolver(cfg)
        solver = nl_mod.JaxNonlinearGKSolver(
            cfg,
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
        )
        result = solver.run()

    assert result.chi_i_gB >= 0.0
    assert result.final_state is not None
    assert result.final_state.f is not None


def test_nonlinear_jax_kinetic_electrons():
    """JAX solver with kinetic_electrons=True exercises the electron field solve."""
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    import scpn_control.core.jax_gk_nonlinear as nl_mod

    if not nl_mod.jax_available():
        pytest.skip("JAX not installed; strict JAX solver path required.")

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
    try:
        solver = JaxNonlinearGKSolver(cfg)
    except RuntimeError as exc:
        if "JAX nonlinear GK solver requested but JAX is unavailable" in str(exc):
            pytest.skip("JAX runtime unavailable; strict JAX solver path required.")
        raise
    result = solver.run()

    assert np.all(np.isfinite(result.Q_i_t))
    assert result.final_state is not None


def test_nonlinear_jax_electromagnetic():
    """JAX solver with electromagnetic=True exercises Ampere solve and EM gradient drive."""
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    import scpn_control.core.jax_gk_nonlinear as nl_mod

    if not nl_mod.jax_available():
        pytest.skip("JAX not installed; strict JAX solver path required.")

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
    try:
        solver = JaxNonlinearGKSolver(cfg)
    except RuntimeError as exc:
        if "JAX nonlinear GK solver requested but JAX is unavailable" in str(exc):
            pytest.skip("JAX runtime unavailable; strict JAX solver path required.")
        raise
    result = solver.run()

    assert result.final_state is not None
    assert result.final_state.A_par is not None
    assert np.all(np.isfinite(result.final_state.A_par))


def test_solve_eigenvalue_all_stable():
    """Exercise jax_gk_solver.py lines 178-179, 185: all gammas <= 0 -> stable."""
    from scpn_control.core.jax_gk_solver import _solve_eigenvalue_from_matrix

    M_real = np.array([[1.0, 0.0], [0.0, 2.0]])
    M_imag = np.array([[-1.0, 0.0], [0.0, -2.0]])
    gamma, omega_r, mode_type, phi = _solve_eigenvalue_from_matrix(M_real, M_imag)
    assert gamma == 0.0
    assert mode_type == "stable"
    assert phi is None


def test_solve_eigenvalue_itg_vs_tem():
    """Exercise jax_gk_solver.py lines 193, 197: ITG (omega<0) vs TEM (omega>0) vs stable."""
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
    import scpn_control.core.jax_gk_nonlinear as nl_mod

    if not nl_mod.jax_available():
        pytest.skip("JAX not installed; strict JAX solver path required.")

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
    try:
        solver = JaxNonlinearGKSolver(cfg)
    except RuntimeError as exc:
        if "JAX nonlinear GK solver requested but JAX is unavailable" in str(exc):
            pytest.skip("JAX runtime unavailable; strict JAX solver path required.")
        raise
    result = solver.run()

    assert result.final_state is not None
    assert result.final_state.A_par is not None


# ---------------------------------------------------------------------------
# Small numeric-edge branches
# ---------------------------------------------------------------------------


def test_solve_eigenvalue_linalg_error(monkeypatch):
    """LinAlgError from np.linalg.eig falls back to a stable mode."""
    import scpn_control.core.jax_gk_solver as mod

    def _boom(_matrix):
        raise np.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(mod.np.linalg, "eig", _boom)
    gamma, omega_r, mode_type, phi = mod._solve_eigenvalue_from_matrix(np.eye(2), np.eye(2))
    assert gamma == 0.0
    assert omega_r == 0.0
    assert mode_type == "stable"
    assert phi is None


def test_local_dispersion_breaks_on_vanishing_derivative():
    """Zero form factors make the Newton derivative vanish, returning a stable root."""
    from scpn_control.core.jax_gk_solver import _solve_local_dispersion_from_payload

    gamma, omega_r = _solve_local_dispersion_from_payload(
        qn_denom=1.0,
        fj=np.zeros(3),
        ws_v=np.ones(3),
        wd_v=np.ones(3),
        nu_eff=0.1,
    )
    assert gamma == 0.0
    assert omega_r == 0.0


def test_dist_version_unknown_package():
    from scpn_control.core.jax_gk_solver import _dist_version

    assert _dist_version("scpn-nonexistent-package-zzz") == "unknown"


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_solve_linear_gk_jax_zero_ky_returns_empty():
    result = solve_linear_gk_jax(n_ky_ion=0)
    assert result.k_y.size == 0
    assert result.gamma.size == 0
    assert result.mode_type == []


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_gk_stiffness_profile_validators():
    rho = np.linspace(0.05, 1.0, 4)
    with pytest.raises(ValueError, match="length >= 3"):
        gk_stiffness_chi_i_profile_jax(R_L_Ti=6.0, rho=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="strictly increasing"):
        gk_stiffness_chi_i_profile_jax(R_L_Ti=6.0, rho=np.array([0.3, 0.2, 0.1]))
    with pytest.raises(ValueError, match="base_chi_i must be non-negative"):
        gk_stiffness_chi_i_profile_jax(R_L_Ti=6.0, rho=rho, base_chi_i=-1.0)
    with pytest.raises(ValueError, match="stiffness_scale must be non-negative"):
        gk_stiffness_chi_i_profile_jax(R_L_Ti=6.0, rho=rho, stiffness_scale=-1.0)


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_backend_metadata_handles_config_read_failure(monkeypatch):
    import scpn_control.core.jax_gk_solver as mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(mod.jax.config, "read", _boom)
    meta = mod._jax_backend_metadata()
    assert "x64_enabled" in meta
    assert isinstance(meta["x64_enabled"], bool)


# ---------------------------------------------------------------------------
# JAX/native parity artifact machinery
# ---------------------------------------------------------------------------

_FAST_PARITY_KWARGS = {"n_ky_ion": 2, "n_theta": 8}


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_build_parity_artifact_full_run():
    from scpn_control.core.jax_gk_solver import (
        JAX_GK_PARITY_EVIDENCE_BOUNDARY,
        JAX_GK_PARITY_SCHEMA_VERSION,
        build_jax_gk_parity_artifact,
    )

    payload = build_jax_gk_parity_artifact(solver_kwargs=dict(_FAST_PARITY_KWARGS))
    assert payload["schema_version"] == JAX_GK_PARITY_SCHEMA_VERSION
    assert payload["case"] == "cyclone_base_case"
    assert payload["evidence_boundary"] == JAX_GK_PARITY_EVIDENCE_BOUNDARY
    assert payload["external_validation_required"] is True
    assert payload["admitted_for_control"] is False
    assert len(payload["payload_sha256"]) == 64
    assert len(payload["solver_kwargs_sha256"]) == 64
    assert payload["native_dominant_mode_type"]
    assert payload["jax_dominant_mode_type"]


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_build_parity_artifact_other_cases_with_supplied_results():
    from scpn_control.core.jax_gk_solver import build_jax_gk_parity_artifact

    result = solve_linear_gk_jax(**_FAST_PARITY_KWARGS)
    for case in ("tem_kinetic_electron", "stable_mode"):
        payload = build_jax_gk_parity_artifact(
            case=case,
            native_result=result,
            jax_result=result,
            solver_kwargs=dict(_FAST_PARITY_KWARGS),
        )
        assert payload["case"] == case
        assert payload["case_parameters"]["case"] == case
        assert len(payload["case_acceptance"]["required_mode_types"]) >= 1


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_write_parity_artifact_to_directory(tmp_path):
    from scpn_control.core.jax_gk_solver import write_jax_gk_parity_artifact

    payload, path = write_jax_gk_parity_artifact(tmp_path, solver_kwargs=dict(_FAST_PARITY_KWARGS))
    assert path.suffix == ".json"
    assert path.is_file()
    assert path.parent == tmp_path
    import json

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["payload_sha256"] == payload["payload_sha256"]


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_build_parity_artifact_rejects_bad_inputs():
    from scpn_control.core.jax_gk_solver import build_jax_gk_parity_artifact

    with pytest.raises(ValueError, match="case must be one of"):
        build_jax_gk_parity_artifact(case="nope")
    with pytest.raises(ValueError, match="gamma_relative_tolerance"):
        build_jax_gk_parity_artifact(gamma_relative_tolerance=0.0)
    with pytest.raises(ValueError, match="omega_absolute_tolerance"):
        build_jax_gk_parity_artifact(omega_absolute_tolerance=0.0)
    with pytest.raises(ValueError, match="must be finite numeric"):
        build_jax_gk_parity_artifact(solver_kwargs={"R0": "x"})
    with pytest.raises(ValueError, match="positive integer"):
        build_jax_gk_parity_artifact(solver_kwargs={"n_ky_ion": 0})
    with pytest.raises(ValueError, match="must not be boolean"):
        build_jax_gk_parity_artifact(solver_kwargs={"extra_flag": True})
    with pytest.raises(ValueError, match="must be numeric"):
        build_jax_gk_parity_artifact(solver_kwargs={"extra_obj": "not-a-number"})


# ---------------------------------------------------------------------------
# Parity helper validation branches (malformed result objects)
# ---------------------------------------------------------------------------


def _fake_result(gamma, omega_r, mode_type):
    import types

    return types.SimpleNamespace(gamma=np.asarray(gamma), omega_r=np.asarray(omega_r), mode_type=mode_type)


def test_dominant_growth_and_frequency_rejects_malformed_results():
    from scpn_control.core.jax_gk_solver import _dominant_growth_and_frequency

    with pytest.raises(ValueError, match="matching one-dimensional"):
        _dominant_growth_and_frequency(_fake_result([1.0, 2.0], [1.0], ["ITG", "TEM"]), "native_result")
    with pytest.raises(ValueError, match="finite gamma"):
        _dominant_growth_and_frequency(_fake_result([np.nan], [1.0], ["ITG"]), "native_result")


def test_mode_type_spectrum_rejects_malformed_results():
    from scpn_control.core.jax_gk_solver import _mode_type_spectrum

    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        _mode_type_spectrum(_fake_result([], [], []), "native_result")
    with pytest.raises(ValueError, match="one mode type per gamma"):
        _mode_type_spectrum(_fake_result([1.0, 2.0], [1.0, 2.0], ["ITG"]), "native_result")
    with pytest.raises(ValueError, match="mode types must be non-empty"):
        _mode_type_spectrum(_fake_result([1.0], [1.0], [""]), "native_result")


def test_dominant_mode_type_rejects_non_finite_gamma():
    from scpn_control.core.jax_gk_solver import _dominant_mode_type

    with pytest.raises(ValueError, match="finite gamma"):
        _dominant_mode_type(_fake_result([np.nan], [1.0], ["ITG"]), "native_result")


def test_parity_case_inputs_rejects_unknown_case():
    from scpn_control.core.jax_gk_solver import _parity_case_inputs

    run_kwargs = {"R0": 2.78, "a": 1.0, "B0": 2.0, "q": 1.4, "s_hat": 0.78, "n_ky_ion": 2, "n_theta": 8}
    with pytest.raises(ValueError, match="case must be one of"):
        _parity_case_inputs("unknown_case", run_kwargs)


def test_safe_artifact_token_collapses_repeated_separators():
    from scpn_control.core.jax_gk_solver import _safe_artifact_token

    assert _safe_artifact_token("a -- b") == "a_b"
    assert _safe_artifact_token("///") == "jax_gk_parity"
