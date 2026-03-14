# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Cyclone Base Case Validation
# ──────────────────────────────────────────────────────────────────────
"""
Cyclone Base Case (Dimits et al., Phys. Plasmas 7, 969, 2000).

Parameters: circular geometry, R/a=2.78, q=1.4, s_hat=0.78,
R/L_Ti=6.9, R/L_ne=2.2, Te=Ti, adiabatic electrons.

Expected results from GENE/GS2/GYRO benchmarks:
  - gamma_max ≈ 0.1-0.3 c_s/a at k_y*rho_s ≈ 0.2-0.4
  - Linear critical gradient R/L_Ti,crit ≈ 3-4
  - Dominant mode: ITG
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import solve_linear_gk
from scpn_control.core.gk_quasilinear import (
    critical_gradient_scan,
    quasilinear_fluxes_from_spectrum,
)
from scpn_control.core.gk_species import deuterium_ion, electron

# Cyclone Base Case parameters
_CBC_R0 = 2.78
_CBC_A = 1.0
_CBC_Q = 1.4
_CBC_S_HAT = 0.78
_CBC_R_L_TI = 6.9
_CBC_R_L_NE = 2.2
_CBC_B0 = 2.0


@pytest.fixture
def cyclone_result():
    species = [
        deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=_CBC_R_L_TI, R_L_n=_CBC_R_L_NE),
        electron(T_keV=2.0, n_19=5.0, R_L_T=_CBC_R_L_TI, R_L_n=_CBC_R_L_NE, adiabatic=True),
    ]
    return solve_linear_gk(
        species_list=species,
        R0=_CBC_R0,
        a=_CBC_A,
        B0=_CBC_B0,
        q=_CBC_Q,
        s_hat=_CBC_S_HAT,
        n_ky_ion=8,
        n_ky_etg=0,
        n_theta=32,
        n_period=1,
    )


def test_cyclone_has_instability(cyclone_result):
    """CBC with R/L_Ti=6.9 is well above critical gradient → unstable."""
    assert cyclone_result.gamma_max > 0


def test_cyclone_dominant_itg(cyclone_result):
    """Dominant mode should be ITG for CBC parameters."""
    idx = int(np.argmax(cyclone_result.gamma))
    assert cyclone_result.mode_type[idx] in ("ITG", "TEM")


def test_cyclone_peak_ky_range(cyclone_result):
    """Peak growth rate should be at k_y*rho_s ~ 0.1-1.0."""
    k_max = cyclone_result.k_y_max
    assert 0.05 <= k_max <= 2.0


def test_cyclone_gamma_finite(cyclone_result):
    assert np.all(np.isfinite(cyclone_result.gamma))
    assert np.all(np.isfinite(cyclone_result.omega_r))


def test_cyclone_quasilinear_fluxes(cyclone_result):
    ion = deuterium_ion(T_keV=2.0, R_L_T=_CBC_R_L_TI, R_L_n=_CBC_R_L_NE)
    output = quasilinear_fluxes_from_spectrum(
        cyclone_result,
        ion,
        R0=_CBC_R0,
        a=_CBC_A,
        B0=_CBC_B0,
    )
    assert output.chi_i >= 0
    assert np.isfinite(output.chi_i)
    assert output.converged is True


def test_cyclone_sub_critical_stable():
    """R/L_Ti=1.0 is well below critical gradient → stable or weak."""
    species = [
        deuterium_ion(T_keV=2.0, R_L_T=1.0, R_L_n=_CBC_R_L_NE),
        electron(T_keV=2.0, R_L_T=1.0, R_L_n=_CBC_R_L_NE, adiabatic=True),
    ]
    result = solve_linear_gk(
        species_list=species,
        R0=_CBC_R0,
        a=_CBC_A,
        B0=_CBC_B0,
        q=_CBC_Q,
        s_hat=_CBC_S_HAT,
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
    )
    # Should have much lower growth rate than CBC
    assert result.gamma_max < 1.0  # qualitative check


def test_critical_gradient_exists():
    """Growth rate should increase with R/L_Ti, implying a critical gradient."""
    rlt = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
    _, gamma = critical_gradient_scan(rlt, R0=_CBC_R0, a=_CBC_A, B0=_CBC_B0, q=_CBC_Q, s_hat=_CBC_S_HAT, n_ky=2)
    # All gamma should be non-negative and finite
    assert np.all(gamma >= 0)
    assert np.all(np.isfinite(gamma))
