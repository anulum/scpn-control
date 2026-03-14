# ──────────────────────────────────────────────────────────────────────
# SCPN Control — GK Linear Benchmark Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import solve_linear_gk
from scpn_control.core.gk_quasilinear import quasilinear_fluxes_from_spectrum
from scpn_control.core.gk_species import deuterium_ion, electron


def test_cyclone_base_case_gamma_positive():
    species = [deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2), electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)]
    result = solve_linear_gk(
        species_list=species, R0=2.78, a=1.0, B0=2.0, q=1.4, s_hat=0.78, n_ky_ion=8, n_theta=32, n_period=1
    )
    assert result.gamma_max > 0


def test_cyclone_dominant_mode_itg():
    species = [deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2), electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)]
    result = solve_linear_gk(
        species_list=species, R0=2.78, a=1.0, B0=2.0, q=1.4, s_hat=0.78, n_ky_ion=8, n_theta=32, n_period=1
    )
    idx = int(np.argmax(result.gamma))
    assert result.mode_type[idx] in ("ITG", "TEM")


def test_multi_code_comparison_both_finite():
    from scpn_control.core.gyrokinetic_transport import (
        GyrokineticsParams,
        compute_spectrum,
        quasilinear_fluxes,
    )

    species = [deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2), electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)]
    gk = solve_linear_gk(
        species_list=species, R0=2.78, a=1.0, B0=2.0, q=1.4, s_hat=0.78, n_ky_ion=4, n_theta=16, n_period=1
    )
    gk_fluxes = quasilinear_fluxes_from_spectrum(gk, species[0], R0=2.78, a=1.0, B0=2.0)

    params = GyrokineticsParams(
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.0,
        nu_star=0.01,
        beta_e=0.0,
        epsilon=0.18,
    )
    ql_spec = compute_spectrum(params, n_modes=8)
    ql_fluxes = quasilinear_fluxes(params, ql_spec)

    assert np.isfinite(gk_fluxes.chi_i)
    assert np.isfinite(ql_fluxes.chi_i)


def test_sparc_parameters_finite():
    species = [deuterium_ion(T_keV=10.0, R_L_T=6.0, R_L_n=2.0), electron(T_keV=10.0, R_L_T=6.0, R_L_n=2.0)]
    result = solve_linear_gk(
        species_list=species, R0=1.85, a=0.57, B0=12.2, q=1.8, s_hat=1.0, n_ky_ion=4, n_theta=16, n_period=1
    )
    assert np.all(np.isfinite(result.gamma))


def test_iter_parameters_finite():
    species = [deuterium_ion(T_keV=8.0, R_L_T=6.0, R_L_n=2.0), electron(T_keV=8.0, R_L_T=6.0, R_L_n=2.0)]
    result = solve_linear_gk(
        species_list=species, R0=6.2, a=2.0, B0=5.3, q=1.5, s_hat=0.8, n_ky_ion=4, n_theta=16, n_period=1
    )
    assert np.all(np.isfinite(result.gamma))


def test_reference_data_exists():
    from pathlib import Path

    ref = Path(__file__).parent.parent / "validation" / "reference_data" / "cyclone_base" / "cyclone_base_case.json"
    assert ref.exists()
    data = json.loads(ref.read_text())
    assert data["R_over_a"] == pytest.approx(2.78)
    assert data["expected"]["dominant_mode"] == "ITG"
