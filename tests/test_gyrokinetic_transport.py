# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gyrokinetic Transport Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gyrokinetic_transport import (
    GyrokineticsParams,
    GyrokineticTransportModel,
    SpectrumResult,
    compute_spectrum,
    quasilinear_fluxes,
    saturated_growth_rate,
    solve_dispersion,
)


def test_zero_gradients():
    params = GyrokineticsParams(
        R_L_Ti=0.0,
        R_L_Te=0.0,
        R_L_ne=0.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    assert g == 0.0

    spec = compute_spectrum(params)
    fluxes = quasilinear_fluxes(params, spec)
    assert fluxes.chi_i == 0.0
    assert fluxes.chi_e == 0.0
    assert fluxes.D_e == 0.0


def test_sub_critical_itg():
    params = GyrokineticsParams(
        R_L_Ti=1.0,  # low
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    # Should not be ITG, might be TEM if driven, but let's just check mt != 1
    assert mt != 1


def test_super_critical_itg():
    params = GyrokineticsParams(
        R_L_Ti=10.0,  # high
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    assert mt == 1
    assert g > 0.0


def test_tem_regime():
    params = GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=1.0,
        R_L_ne=10.0,  # high density gradient
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    assert mt == 2
    assert g > 0.0
    assert w > 0.0


def test_etg_scale():
    params = GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=20.0,  # high Te gradient
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 5.0, etg_scale=True)
    assert mt == 3
    assert g > 0.0


def test_quasilinear_fluxes():
    params = GyrokineticsParams(
        R_L_Ti=10.0,
        R_L_Te=10.0,
        R_L_ne=10.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    spec = compute_spectrum(params, n_modes=10, include_etg=True)
    fluxes = quasilinear_fluxes(params, spec)
    assert fluxes.chi_i > 0.0
    assert fluxes.chi_e > 0.0
    assert fluxes.D_e >= 0.0


def test_saturated_growth_rate_is_monotone_and_bounded_by_field_line_rate():
    q = 1.7
    gamma = np.array([0.0, 0.05, 0.2, 1.0, 5.0])
    saturated = np.array([saturated_growth_rate(value, q) for value in gamma])

    assert saturated[0] == 0.0
    assert np.all(np.diff(saturated) > 0.0)
    assert np.all(saturated < 1.0 / q)
    assert saturated_growth_rate(1.0e6, q) == pytest.approx(1.0 / q, rel=1.0e-6)


def test_saturated_growth_rate_rejects_nonphysical_inputs():
    with pytest.raises(ValueError, match="gamma_linear"):
        saturated_growth_rate(-1.0e-3, 1.5)
    with pytest.raises(ValueError, match="gamma_linear"):
        saturated_growth_rate(float("nan"), 1.5)
    with pytest.raises(ValueError, match="q"):
        saturated_growth_rate(0.1, 0.0)


def test_stiffness():
    params1 = GyrokineticsParams(
        R_L_Ti=6.0,
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    params2 = GyrokineticsParams(
        R_L_Ti=12.0,
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    spec1 = compute_spectrum(params1)
    fluxes1 = quasilinear_fluxes(params1, spec1)

    spec2 = compute_spectrum(params2)
    fluxes2 = quasilinear_fluxes(params2, spec2)

    assert fluxes2.chi_i > fluxes1.chi_i * 1.5


def test_transport_model_eval():
    model = GyrokineticTransportModel()
    rho = 0.5
    profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": 1.5,
        "s_hat": 1.0,
        "Te": 5.0,
        "Ti": 5.0,
        "ne": 5.0,
        "dTe_dr": -50.0,
        "dTi_dr": -50.0,
        "dne_dr": -50.0,
    }
    chi_i, chi_e, D_e = model.evaluate(rho, profiles)
    assert chi_i >= 0.0
    assert chi_e >= 0.0
    assert D_e >= 0.0


def test_transport_model_eval_profile():
    model = GyrokineticTransportModel()
    rho = np.linspace(0, 1, 10)
    profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": np.full(10, 1.5),
        "s_hat": np.full(10, 1.0),
        "Te": np.linspace(5.0, 0.1, 10),
        "Ti": np.linspace(5.0, 0.1, 10),
        "ne": np.linspace(5.0, 0.1, 10),
        "dTe_dr": np.full(10, -10.0),
        "dTi_dr": np.full(10, -10.0),
        "dne_dr": np.full(10, -10.0),
    }
    chi_i, chi_e, D_e = model.evaluate_profile(rho, profiles)
    assert len(chi_i) == 10
    assert len(chi_e) == 10
    assert len(D_e) == 10
    assert np.all(chi_i >= 0.0)
    assert chi_i[0] == 0.01  # boundary handling


def test_dispersion_rejects_invalid_physical_domains():
    params = GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=0.0,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    with pytest.raises(ValueError, match="q"):
        solve_dispersion(params, 0.5)

    good = GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    with pytest.raises(ValueError, match="k_theta_rho_s"):
        solve_dispersion(good, 0.0)


def test_spectrum_and_fluxes_reject_invalid_domains():
    params = GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    with pytest.raises(ValueError, match="n_modes"):
        compute_spectrum(params, n_modes=0)

    bad_spectrum = SpectrumResult(
        k_y=np.array([0.3]),
        gamma_linear=np.array([0.1, 0.2]),
        omega_r=np.array([0.1]),
        mode_type=np.array([1]),
    )
    with pytest.raises(ValueError, match="spectrum"):
        quasilinear_fluxes(params, bad_spectrum)


def test_transport_model_rejects_invalid_radius_and_profiles():
    model = GyrokineticTransportModel()
    valid_profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": 1.5,
        "s_hat": 1.0,
        "Te": 5.0,
        "Ti": 5.0,
        "ne": 5.0,
        "dTe_dr": -50.0,
        "dTi_dr": -50.0,
        "dne_dr": -50.0,
    }
    with pytest.raises(ValueError, match="rho"):
        model.evaluate(1.1, valid_profiles)
    bad_profiles = dict(valid_profiles)
    bad_profiles["Te"] = 0.0
    with pytest.raises(ValueError, match="Te"):
        model.evaluate(0.5, bad_profiles)

    bad_geometry = dict(valid_profiles)
    bad_geometry["a"] = bad_geometry["R0"]
    with pytest.raises(ValueError, match="a must be smaller"):
        model.evaluate(0.5, bad_geometry)


def test_transport_model_rejects_boolean_mode_count():
    with pytest.raises(ValueError, match="n_modes"):
        GyrokineticTransportModel(n_modes=True)


def test_transport_model_eval_profile_rejects_profile_shape_mismatch():
    model = GyrokineticTransportModel()
    rho = np.linspace(0, 1, 5)
    profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": np.full(4, 1.5),
        "s_hat": np.full(5, 1.0),
        "Te": np.linspace(5.0, 0.1, 5),
        "Ti": np.linspace(5.0, 0.1, 5),
        "ne": np.linspace(5.0, 0.1, 5),
        "dTe_dr": np.full(5, -10.0),
        "dTi_dr": np.full(5, -10.0),
        "dne_dr": np.full(5, -10.0),
    }

    with pytest.raises(ValueError, match="profile q must match rho shape"):
        model.evaluate_profile(rho, profiles)


def test_transport_model_eval_profile_rejects_nonfinite_profile_array():
    model = GyrokineticTransportModel()
    rho = np.linspace(0, 1, 5)
    profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": np.full(5, 1.5),
        "s_hat": np.full(5, 1.0),
        "Te": np.array([5.0, 4.0, np.nan, 1.0, 0.1]),
        "Ti": np.linspace(5.0, 0.1, 5),
        "ne": np.linspace(5.0, 0.1, 5),
        "dTe_dr": np.full(5, -10.0),
        "dTi_dr": np.full(5, -10.0),
        "dne_dr": np.full(5, -10.0),
    }

    with pytest.raises(ValueError, match="profile Te must contain only finite values"):
        model.evaluate_profile(rho, profiles)


def test_transport_model_eval_profile_rejects_unsorted_radius_grid():
    model = GyrokineticTransportModel()
    rho = np.array([0.0, 0.5, 0.4, 1.0])
    profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": np.full(4, 1.5),
        "s_hat": np.full(4, 1.0),
        "Te": np.linspace(5.0, 0.1, 4),
        "Ti": np.linspace(5.0, 0.1, 4),
        "ne": np.linspace(5.0, 0.1, 4),
        "dTe_dr": np.full(4, -10.0),
        "dTi_dr": np.full(4, -10.0),
        "dne_dr": np.full(4, -10.0),
    }

    with pytest.raises(ValueError, match="rho must be strictly increasing"):
        model.evaluate_profile(rho, profiles)


def _gk_params(*, R_L_Te: float = 1.0, epsilon: float = 0.1) -> GyrokineticsParams:
    """Build a valid GyrokineticsParams, varying only the fields under test."""
    return GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=R_L_Te,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=epsilon,
    )


_SCALAR_PROFILES: dict[str, float] = {
    "R0": 2.0,
    "a": 0.5,
    "B0": 5.0,
    "q": 1.5,
    "s_hat": 1.0,
    "Te": 5.0,
    "Ti": 5.0,
    "ne": 5.0,
    "dTe_dr": -50.0,
    "dTi_dr": -50.0,
    "dne_dr": -50.0,
}


def test_validate_params_rejects_inverse_aspect_ratio_above_unity() -> None:
    """An inverse aspect ratio epsilon greater than one is non-physical."""
    with pytest.raises(ValueError, match="epsilon must be <= 1"):
        solve_dispersion(_gk_params(epsilon=1.5), 0.5)


def test_etg_branch_is_stable_below_critical_gradient() -> None:
    """The ETG mode is marginally stable when R/L_Te is below its threshold."""
    gamma, omega_r, mode_type = solve_dispersion(_gk_params(R_L_Te=0.0), 0.5, etg_scale=True)
    assert (gamma, omega_r, mode_type) == (0.0, 0.0, 0)


def test_quasilinear_fluxes_reject_malformed_spectrum_arrays() -> None:
    """Spectrum arrays must align in length and carry finite, positive entries."""
    params = _gk_params()
    with pytest.raises(ValueError, match="spectrum arrays must have matching lengths"):
        quasilinear_fluxes(
            params,
            SpectrumResult(
                k_y=np.array([0.1, 0.2, 0.3]),
                gamma_linear=np.array([0.1, 0.2]),
                omega_r=np.array([0.0, 0.0]),
                mode_type=np.array([1, 1]),
            ),
        )
    with pytest.raises(ValueError, match="k_y values must be finite and positive"):
        quasilinear_fluxes(
            params,
            SpectrumResult(
                k_y=np.array([0.1, -0.2]),
                gamma_linear=np.array([0.1, 0.2]),
                omega_r=np.array([0.0, 0.0]),
                mode_type=np.array([1, 1]),
            ),
        )
    with pytest.raises(ValueError, match="gamma_linear values must be finite"):
        quasilinear_fluxes(
            params,
            SpectrumResult(
                k_y=np.array([0.1, 0.2]),
                gamma_linear=np.array([0.1, np.inf]),
                omega_r=np.array([0.0, 0.0]),
                mode_type=np.array([1, 1]),
            ),
        )
    with pytest.raises(ValueError, match="omega_r values must be finite"):
        quasilinear_fluxes(
            params,
            SpectrumResult(
                k_y=np.array([0.1, 0.2]),
                gamma_linear=np.array([0.1, 0.2]),
                omega_r=np.array([0.0, np.nan]),
                mode_type=np.array([1, 1]),
            ),
        )
    with pytest.raises(ValueError, match="mode_type values must be 0, 1, 2, or 3"):
        quasilinear_fluxes(
            params,
            SpectrumResult(
                k_y=np.array([0.1, 0.2]),
                gamma_linear=np.array([0.1, 0.2]),
                omega_r=np.array([0.0, 0.0]),
                mode_type=np.array([1, 5]),
            ),
        )


def test_transport_model_evaluate_returns_axis_floor() -> None:
    """Inside the magnetic axis boundary the model returns the small floor values."""
    model = GyrokineticTransportModel()
    chi_i, chi_e, D_e = model.evaluate(0.01, _SCALAR_PROFILES)
    assert (chi_i, chi_e, D_e) == (0.01, 0.01, 0.01)


def test_transport_model_eval_profile_rejects_out_of_range_radius_grid() -> None:
    """A radius grid leaving the unit interval is rejected before evaluation."""
    model = GyrokineticTransportModel()
    rho = np.array([0.0, 0.5, 1.5])
    profiles = {
        "R0": np.full(3, 2.0),
        "a": np.full(3, 0.5),
        "B0": np.full(3, 5.0),
        "q": np.full(3, 1.5),
        "s_hat": np.full(3, 1.0),
        "Te": np.linspace(5.0, 0.1, 3),
        "Ti": np.linspace(5.0, 0.1, 3),
        "ne": np.linspace(5.0, 0.1, 3),
        "dTe_dr": np.full(3, -10.0),
        "dTi_dr": np.full(3, -10.0),
        "dne_dr": np.full(3, -10.0),
    }
    with pytest.raises(ValueError, match="finite one-dimensional profile"):
        model.evaluate_profile(rho, profiles)
