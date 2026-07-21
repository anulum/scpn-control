# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Fokker-Planck collision reference tests
"""Offline tests for :mod:`validation.gk_collision_independent_reference`."""

from __future__ import annotations

import numpy as np
import pytest

from validation.gk_collision_independent_reference import (
    IndependentCollisionRates,
    basic_collision_frequency,
    braginskii_collision_rate,
    chandrasekhar_g,
    deflection_shape,
    elastic_energy_transfer_efficiency,
    independent_collision_rates,
    maxwellian_deflection_average_factor,
    thermal_deflection_rate,
)

_ELECTRON_AMU = 9.1093837015e-31 / 1.67262192369e-27


def test_chandrasekhar_small_x_asymptote() -> None:
    # G(x) -> 2x / (3 sqrt(pi)) as x -> 0.
    x = np.array([1.0e-6, 1.0e-5])
    expected = 2.0 * x / (3.0 * np.sqrt(np.pi))
    np.testing.assert_allclose(chandrasekhar_g(x), expected, rtol=1e-9)


def test_chandrasekhar_continuous_across_series_threshold() -> None:
    # The series and general branches must agree at the switch-over x = 1e-3.
    below = float(chandrasekhar_g(np.array([0.9e-3]))[0])
    above = float(chandrasekhar_g(np.array([1.1e-3]))[0])
    # Both close to the asymptote and to each other (continuity of G).
    assert below == pytest.approx(2.0 * 0.9e-3 / (3.0 * np.sqrt(np.pi)), rel=1e-4)
    assert above == pytest.approx(2.0 * 1.1e-3 / (3.0 * np.sqrt(np.pi)), rel=1e-4)


def test_chandrasekhar_peaks_and_decays() -> None:
    # G(x) rises, peaks near x ~ 1, then decays as erf(x)/(2 x^2) ~ 1/(2 x^2).
    x = np.linspace(0.1, 6.0, 200)
    g = chandrasekhar_g(x)
    assert np.all(np.isfinite(g))
    assert g[-1] < g[np.argmax(g)]
    np.testing.assert_allclose(g[-1], 1.0 / (2.0 * x[-1] ** 2), rtol=5e-3)


def test_chandrasekhar_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        chandrasekhar_g(np.array([-0.1]))


def test_deflection_shape_positive_and_decays() -> None:
    x = np.array([0.5, 1.0, 2.0, 4.0])
    shape = deflection_shape(x)
    assert np.all(shape > 0.0)
    # [erf-G]/x^3 decays monotonically for x >= 0.5.
    assert np.all(np.diff(shape) < 0.0)


def test_deflection_shape_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        deflection_shape(np.array([0.0, 1.0]))


def test_maxwellian_average_factor_converged() -> None:
    factor = maxwellian_deflection_average_factor()
    assert factor == pytest.approx(1.3878016605, rel=1e-8)
    # Independent of quadrature resolution and truncation once well resolved.
    assert maxwellian_deflection_average_factor(n_quad=128, x_max=12.0) == pytest.approx(factor, rel=1e-10)


@pytest.mark.parametrize("bad", [1, True, 0, -3])
def test_maxwellian_average_rejects_bad_n_quad(bad: object) -> None:
    with pytest.raises(ValueError, match="n_quad must be an integer"):
        maxwellian_deflection_average_factor(n_quad=bad)  # type: ignore[arg-type]


def test_maxwellian_average_rejects_bad_x_max() -> None:
    with pytest.raises(ValueError, match="x_max must be positive"):
        maxwellian_deflection_average_factor(x_max=0.0)


def test_basic_collision_frequency_scaling() -> None:
    base = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    dense = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=20.0, ln_lambda=17.0)
    hot = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=16.0, n_field_19=10.0, ln_lambda=17.0)
    assert dense == pytest.approx(2.0 * base)  # linear in density
    assert hot == pytest.approx(base * 2.0**-1.5)  # nu ~ v_th^-3 ~ T^-1.5


def test_basic_collision_frequency_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="temperature_keV must be positive"):
        basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=0.0, n_field_19=10.0, ln_lambda=17.0)
    with pytest.raises(ValueError, match="mass_amu must be positive"):
        basic_collision_frequency(mass_amu=-1.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    with pytest.raises(ValueError, match="charge_e must be finite"):
        basic_collision_frequency(
            mass_amu=2.0, charge_e=float("nan"), temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0
        )


def test_thermal_deflection_rate_matches_basic_times_factor() -> None:
    rate = thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=1.0)
    nu_hat = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    assert rate == pytest.approx(nu_hat * maxwellian_deflection_average_factor())


def test_thermal_deflection_rate_linear_in_zeff() -> None:
    single = thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=1.0)
    triple = thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=3.0)
    assert triple == pytest.approx(3.0 * single)


def test_thermal_deflection_rate_rejects_bad_zeff() -> None:
    with pytest.raises(ValueError, match="z_eff must be positive"):
        thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=-1.0)


def test_braginskii_rate_scaling() -> None:
    base = braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0)
    hot = braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=16.0, n_field_19=10.0)
    assert hot == pytest.approx(base * 2.0**-1.5)


def test_braginskii_rate_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="n_field_19 must be positive"):
        braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=0.0)
    with pytest.raises(ValueError, match="z_eff must be positive"):
        braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=0.0)
    with pytest.raises(ValueError, match="ln_lambda must be positive"):
        braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=-1.0)


def test_elastic_efficiency_electron_self_is_half() -> None:
    # Equal masses transfer half their energy on average.
    assert elastic_energy_transfer_efficiency(_ELECTRON_AMU) == pytest.approx(0.5)


def test_elastic_efficiency_heavy_ion_mass_suppressed() -> None:
    # Deuterium against the electron field: delta ~ 2 m_e / m_D.
    delta = elastic_energy_transfer_efficiency(2.0)
    assert delta == pytest.approx(2.0 * 2.0 * _ELECTRON_AMU / (2.0 + _ELECTRON_AMU) ** 2)
    assert delta < 1.0e-3


def test_elastic_efficiency_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="mass_amu must be positive"):
        elastic_energy_transfer_efficiency(0.0)
    with pytest.raises(ValueError, match="field_mass_amu must be positive"):
        elastic_energy_transfer_efficiency(2.0, field_mass_amu=-1.0)


def test_independent_collision_rates_assembles_channels() -> None:
    rates = independent_collision_rates(
        mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_e_19=10.0, T_e_keV=4.0, z_eff=1.0
    )
    assert isinstance(rates, IndependentCollisionRates)
    assert rates.field_temperature_factor == pytest.approx(np.sqrt(8.0 / 4.0))
    assert rates.energy_relaxation_rate == pytest.approx(
        rates.thermal_deflection_rate * rates.elastic_energy_transfer_efficiency * rates.field_temperature_factor
    )
    # Deflection and Braginskii anchors differ only by an O(1) convention factor.
    assert rates.thermal_deflection_rate / rates.braginskii_rate == pytest.approx(0.5 / 0.2710231582, rel=1e-6)


def test_independent_collision_rates_rejects_bad_temperatures() -> None:
    with pytest.raises(ValueError, match="T_e_keV must be positive"):
        independent_collision_rates(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_e_19=10.0, T_e_keV=0.0)
    with pytest.raises(ValueError, match="temperature_keV must be positive"):
        independent_collision_rates(mass_amu=2.0, charge_e=1.0, temperature_keV=-8.0, n_e_19=10.0, T_e_keV=4.0)
