# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_species import (
    DiamagneticFrequencies,
    GKSpecies,
    VelocityGrid,
    bessel_j0,
    collision_frequencies,
    deuterium_ion,
    diamagnetic_frequencies,
    electron,
    pitch_angle_operator,
)


def test_deuterium_defaults():
    ion = deuterium_ion()
    assert ion.mass_amu == 2.0
    assert ion.charge_e == 1.0
    assert ion.temperature_keV == 8.0


def test_electron_mass():
    e = electron()
    assert e.mass_amu < 0.001  # m_e / m_p ≈ 5.4e-4
    assert e.charge_e == -1.0
    assert e.is_adiabatic is True


def test_kinetic_electron():
    e = electron(adiabatic=False)
    assert e.is_adiabatic is False


def test_thermal_speed():
    ion = deuterium_ion(T_keV=1.0)
    v_th = ion.thermal_speed
    # v_th = sqrt(2T/m), T=1keV=1.6e-16 J, m_D=3.34e-27 kg
    # v_th ≈ sqrt(2*1.6e-16/3.34e-27) ≈ 3.1e5 m/s
    assert 2e5 < v_th < 5e5


def test_electron_thermal_speed_faster():
    ion = deuterium_ion(T_keV=1.0)
    e = electron(T_keV=1.0)
    assert e.thermal_speed > 40 * ion.thermal_speed  # sqrt(m_i/m_e) ~ 60


def test_velocity_grid_shape():
    vg = VelocityGrid(n_energy=16, n_lambda=24)
    assert len(vg.energy) == 16
    assert len(vg.energy_weights) == 16
    assert len(vg.lam) == 24
    assert len(vg.lambda_weights) == 24
    assert vg.n_total == 384


def test_velocity_grid_energy_range():
    vg = VelocityGrid(n_energy=16)
    assert vg.energy[0] > 0
    assert vg.energy[-1] < vg.E_max


def test_velocity_grid_lambda_range():
    vg = VelocityGrid(n_lambda=24)
    assert np.all(vg.lam >= 0)
    assert np.all(vg.lam <= 1)


def test_velocity_grid_weights_positive():
    vg = VelocityGrid()
    assert np.all(vg.energy_weights > 0)
    assert np.all(vg.lambda_weights > 0)


def test_bessel_j0_at_zero():
    x = np.array([0.0])
    assert bessel_j0(x)[0] == pytest.approx(1.0)


def test_bessel_j0_long_wavelength():
    """J_0(x) → 1 as x → 0 (FLR limit)."""
    x = np.array([0.001, 0.01, 0.1])
    j0 = bessel_j0(x)
    np.testing.assert_allclose(j0, 1.0, atol=0.01)


def test_bessel_j0_first_zero():
    """J_0 first zero at x ≈ 2.4048."""
    x = np.array([2.4048])
    assert abs(bessel_j0(x)[0]) < 0.01


def test_diamagnetic_frequencies_follow_species_charge_direction_and_gradients():
    ion = deuterium_ion(R_L_T=6.0, R_L_n=2.0)
    elec = electron(R_L_T=6.0, R_L_n=2.0, adiabatic=False)

    ion_omega = diamagnetic_frequencies(ion, k_y_rho_s=0.3)
    electron_omega = diamagnetic_frequencies(elec, k_y_rho_s=0.3)

    assert isinstance(ion_omega, DiamagneticFrequencies)
    assert ion_omega.density == pytest.approx(-0.6)
    assert ion_omega.temperature == pytest.approx(-1.8)
    assert ion_omega.pressure == pytest.approx(ion_omega.density + ion_omega.temperature)
    assert electron_omega.density == pytest.approx(-ion_omega.density)
    assert electron_omega.temperature == pytest.approx(-ion_omega.temperature)


def test_diamagnetic_frequencies_vanish_for_zero_drive_and_wavenumber():
    species = deuterium_ion(R_L_T=0.0, R_L_n=0.0)
    no_drive = diamagnetic_frequencies(species, k_y_rho_s=0.7)
    no_wavenumber = diamagnetic_frequencies(deuterium_ion(R_L_T=6.0, R_L_n=2.0), k_y_rho_s=0.0)

    assert no_drive == DiamagneticFrequencies(density=0.0, temperature=0.0, pressure=0.0)
    assert no_wavenumber == DiamagneticFrequencies(density=0.0, temperature=0.0, pressure=0.0)


def test_diamagnetic_frequencies_reject_invalid_contracts():
    with pytest.raises(ValueError, match="GKSpecies"):
        diamagnetic_frequencies(object(), k_y_rho_s=0.3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_y_rho_s"):
        diamagnetic_frequencies(deuterium_ion(), k_y_rho_s=-0.1)
    with pytest.raises(ValueError, match="k_y_rho_s"):
        diamagnetic_frequencies(deuterium_ion(), k_y_rho_s=float("nan"))


def test_collision_frequencies_positive():
    ion = deuterium_ion()
    nu_D, nu_E = collision_frequencies(ion, n_e_19=10.0, T_e_keV=8.0)
    assert nu_D > 0
    assert nu_E > 0


def test_collision_frequencies_scale_with_density():
    ion = deuterium_ion()
    nu_lo, _ = collision_frequencies(ion, n_e_19=1.0, T_e_keV=8.0)
    nu_hi, _ = collision_frequencies(ion, n_e_19=10.0, T_e_keV=8.0)
    assert nu_hi > nu_lo


def test_collision_energy_relaxation_is_not_pitch_angle_alias_for_ions():
    ion = deuterium_ion(T_keV=8.0)
    nu_D, nu_E = collision_frequencies(ion, n_e_19=10.0, T_e_keV=8.0)
    assert 0.0 < nu_E < nu_D


def test_collision_energy_relaxation_varies_with_field_temperature():
    ion = deuterium_ion(T_keV=8.0)
    _, nu_cold_electrons = collision_frequencies(ion, n_e_19=10.0, T_e_keV=2.0)
    _, nu_hot_electrons = collision_frequencies(ion, n_e_19=10.0, T_e_keV=16.0)
    assert nu_cold_electrons != pytest.approx(nu_hot_electrons)


def test_pitch_angle_operator_shape():
    vg = VelocityGrid(n_lambda=24)
    L = pitch_angle_operator(vg.n_lambda, vg.lam)
    assert L.shape == (24, 24)


def test_pitch_angle_operator_tridiagonal():
    """Pitch-angle operator should be tridiagonal."""
    vg = VelocityGrid(n_lambda=8)
    L = pitch_angle_operator(vg.n_lambda, vg.lam)
    # Check no non-zero entries beyond tridiagonal band
    for i in range(8):
        for j in range(8):
            if abs(i - j) > 1:
                assert L[i, j] == 0.0


def test_larmor_radius_positive():
    """Exercise gk_species.py line 84: larmor_radius returns positive for ion."""
    ion = deuterium_ion(T_keV=8.0)
    rho_over_B = ion.larmor_radius
    assert rho_over_B > 0.0


def test_pitch_angle_operator_rejects_duplicate_lambda_grid() -> None:
    """Pitch-angle diffusion requires a strictly ordered lambda grid."""
    lam = np.array([0.0, 0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        pitch_angle_operator(4, lam)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass_amu", 0.0),
        ("charge_e", 0.0),
        ("temperature_keV", 0.0),
        ("density_19", -1.0),
        ("R_L_T", float("nan")),
        ("R_L_n", float("inf")),
    ],
)
def test_gk_species_rejects_nonphysical_parameters(field: str, value: float) -> None:
    kwargs = {
        "mass_amu": 2.0,
        "charge_e": 1.0,
        "temperature_keV": 8.0,
        "density_19": 10.0,
        "R_L_T": 6.9,
        "R_L_n": 2.2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        GKSpecies(**kwargs)


@pytest.mark.parametrize(
    "n_e_19,T_e_keV,Z_eff,ln_lambda",
    [
        (0.0, 8.0, 1.0, 17.0),
        (10.0, 0.0, 1.0, 17.0),
        (10.0, 8.0, 0.0, 17.0),
        (10.0, 8.0, 1.0, 0.0),
    ],
)
def test_collision_frequencies_reject_nonphysical_inputs(
    n_e_19: float,
    T_e_keV: float,
    Z_eff: float,
    ln_lambda: float,
) -> None:
    with pytest.raises(ValueError):
        collision_frequencies(deuterium_ion(), n_e_19=n_e_19, T_e_keV=T_e_keV, Z_eff=Z_eff, ln_lambda=ln_lambda)


def test_velocity_grid_rejects_invalid_quadrature_sizes() -> None:
    """Velocity grid requires enough quadrature points for energy-lambda integration."""
    with pytest.raises(ValueError, match="n_energy"):
        VelocityGrid(n_energy=1, n_lambda=24)
    with pytest.raises(ValueError, match="n_lambda"):
        VelocityGrid(n_energy=16, n_lambda=True)


def test_pitch_angle_operator_rejects_malformed_lambda_grid() -> None:
    """Pitch-angle operator rejects mismatched, non-finite, and unordered lambda grids."""
    with pytest.raises(ValueError, match="n_lambda"):
        pitch_angle_operator(0, np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match="lam"):
        pitch_angle_operator(3, np.array([0.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        pitch_angle_operator(3, np.array([0.0, 0.5, 0.5]))
    with pytest.raises(ValueError, match="shape"):
        pitch_angle_operator(4, np.array([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="B_ratio"):
        pitch_angle_operator(3, np.array([0.0, 0.5, 1.0]), B_ratio=0.0)


def test_pitch_angle_operator_rejects_lambda_outside_local_trapped_passing_domain() -> None:
    """Lambda grid must stay below the local trapped-passing boundary."""
    with pytest.raises(ValueError, match="trapped-passing"):
        pitch_angle_operator(3, np.array([0.0, 0.4, 0.75]), B_ratio=2.0)


def test_pitch_angle_operator_rejects_lambda_outside_unit_interval() -> None:
    """Lambda values are pitch ratios and must lie within [0, 1]."""
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        pitch_angle_operator(3, np.array([-0.1, 0.5, 0.9]))
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        pitch_angle_operator(3, np.array([0.0, 0.5, 1.5]))


def test_pitch_angle_operator_skips_degenerate_grid_spacing() -> None:
    """Sub-ULP grid spacing is skipped to avoid an unstable 1/h² coefficient.

    A strictly increasing lambda grid built from sub-normal increments has an
    interior mid-cell width below the 1e-30 floor; that node is left as a zero
    row rather than producing a divergent finite-difference stencil.
    """
    lam = np.array([1e-320, 2e-320, 3e-320, 4e-320])
    assert np.all(np.diff(lam) > 0.0)

    operator = pitch_angle_operator(4, lam)

    assert operator.shape == (4, 4)
    assert np.all(operator[1] == 0.0)


def test_gauss_legendre_newton_non_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall through the Newton iteration cap when the tolerance is never met (branch 68->75).

    For genuine Legendre polynomials Newton's method converges quadratically
    within a handful of steps, so the 100-iteration bound is defensive. Forcing
    the inner polynomial evaluation to return a constant value/derivative pair
    keeps the update from ever settling, exhausting the loop without a break;
    the routine must still return finite node and weight arrays of the requested
    size rather than raising or looping unbounded.
    """
    import scpn_control.core.gk_species as gk_mod

    monkeypatch.setattr(
        gk_mod,
        "_legendre_polynomial_and_derivative",
        lambda order, x_value: (1.0, 1.0),
    )

    nodes, weights = gk_mod._gauss_legendre_nodes_weights(2)

    assert nodes.shape == (2,)
    assert weights.shape == (2,)
    assert np.all(np.isfinite(nodes))
    assert np.all(np.isfinite(weights))
