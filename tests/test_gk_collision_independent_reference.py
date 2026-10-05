# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Fokker-Planck collision reference tests
"""Offline tests for :mod:`validation.gk_collision_independent_reference`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

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
    """Compare the small-speed series against its leading analytic term."""
    x = np.array([1.0e-6, 1.0e-5])
    expected = 2.0 * x / (3.0 * np.sqrt(np.pi))
    np.testing.assert_allclose(chandrasekhar_g(x), expected, rtol=1e-9)


def test_chandrasekhar_continuous_across_series_threshold() -> None:
    # The series and general branches must agree at the switch-over x = 1e-3.
    """Check values on both sides of the retained series switch."""
    below = float(chandrasekhar_g(np.array([0.9e-3]))[0])
    above = float(chandrasekhar_g(np.array([1.1e-3]))[0])
    # Both close to the asymptote and to each other (continuity of G).
    assert below == pytest.approx(2.0 * 0.9e-3 / (3.0 * np.sqrt(np.pi)), rel=1e-4)
    assert above == pytest.approx(2.0 * 1.1e-3 / (3.0 * np.sqrt(np.pi)), rel=1e-4)


def test_chandrasekhar_peaks_and_decays() -> None:
    # G(x) rises, peaks near x ~ 1, then decays as erf(x)/(2 x^2) ~ 1/(2 x^2).
    """Check finite values and the retained large-speed asymptote."""
    x = np.linspace(0.1, 6.0, 200, dtype=np.float64)
    g = chandrasekhar_g(x)
    assert np.all(np.isfinite(g))
    assert g[-1] < g[np.argmax(g)]
    np.testing.assert_allclose(g[-1], 1.0 / (2.0 * x[-1] ** 2), rtol=5e-3)


def test_chandrasekhar_rejects_negative() -> None:
    """Reject negative dimensionless speed through the public API."""
    with pytest.raises(ValueError, match="non-negative"):
        chandrasekhar_g(np.array([-0.1]))


def test_deflection_shape_positive_and_decays() -> None:
    """Check positivity and decay across representative positive speeds."""
    x = np.array([0.5, 1.0, 2.0, 4.0])
    shape = deflection_shape(x)
    assert np.all(shape > 0.0)
    # [erf-G]/x^3 decays monotonically for x >= 0.5.
    assert np.all(np.diff(shape) < 0.0)


def test_deflection_shape_rejects_nonpositive() -> None:
    """Reject a zero speed before forming the inverse-cubic shape."""
    with pytest.raises(ValueError, match="strictly positive"):
        deflection_shape(np.array([0.0, 1.0]))


def test_maxwellian_average_factor_converged() -> None:
    """Compare the default average with a resolved larger quadrature."""
    factor = maxwellian_deflection_average_factor()
    assert factor == pytest.approx(1.3878016605, rel=1e-8)
    # Independent of quadrature resolution and truncation once well resolved.
    assert maxwellian_deflection_average_factor(n_quad=128, x_max=12.0) == pytest.approx(factor, rel=1e-10)


@pytest.mark.parametrize("bad", [1, True, 0, -3])
def test_maxwellian_average_rejects_bad_n_quad(bad: int) -> None:
    """Refuse nonpositive counts and bool without discarding inherited cases."""
    with pytest.raises(ValueError, match="n_quad must be an integer"):
        maxwellian_deflection_average_factor(n_quad=bad)


def test_maxwellian_average_rejects_bad_x_max() -> None:
    """Reject a zero integration limit through the public average API."""
    with pytest.raises(ValueError, match="x_max must be positive"):
        maxwellian_deflection_average_factor(x_max=0.0)


def test_basic_collision_frequency_scaling() -> None:
    """Check density linearity and the retained temperature scaling."""
    base = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    dense = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=20.0, ln_lambda=17.0)
    hot = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=16.0, n_field_19=10.0, ln_lambda=17.0)
    assert dense == pytest.approx(2.0 * base)  # linear in density
    assert hot == pytest.approx(base * 2.0**-1.5)  # nu ~ v_th^-3 ~ T^-1.5


def test_basic_collision_frequency_rejects_invalid() -> None:
    """Reject nonpositive mass/temperature and nonfinite charge."""
    with pytest.raises(ValueError, match="temperature_keV must be positive"):
        basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=0.0, n_field_19=10.0, ln_lambda=17.0)
    with pytest.raises(ValueError, match="mass_amu must be positive"):
        basic_collision_frequency(mass_amu=-1.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    with pytest.raises(ValueError, match="charge_e must be finite"):
        basic_collision_frequency(
            mass_amu=2.0, charge_e=float("nan"), temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0
        )


def test_thermal_deflection_rate_matches_basic_times_factor() -> None:
    """Check assembly of the retained normalisation and speed average."""
    rate = thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=1.0)
    nu_hat = basic_collision_frequency(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    assert rate == pytest.approx(nu_hat * maxwellian_deflection_average_factor())


def test_thermal_deflection_rate_linear_in_zeff() -> None:
    """Check effective-charge scaling through the public rate API."""
    single = thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=1.0)
    triple = thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=3.0)
    assert triple == pytest.approx(3.0 * single)


def test_thermal_deflection_rate_rejects_bad_zeff() -> None:
    """Reject negative effective charge through the public rate API."""
    with pytest.raises(ValueError, match="z_eff must be positive"):
        thermal_deflection_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=-1.0)


def test_braginskii_rate_scaling() -> None:
    """Check temperature scaling of the separate closed-form rate."""
    base = braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0)
    hot = braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=16.0, n_field_19=10.0)
    assert hot == pytest.approx(base * 2.0**-1.5)


def test_braginskii_rate_rejects_invalid() -> None:
    """Reject nonpositive density, effective charge and Coulomb logarithm."""
    with pytest.raises(ValueError, match="n_field_19 must be positive"):
        braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=0.0)
    with pytest.raises(ValueError, match="z_eff must be positive"):
        braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=0.0)
    with pytest.raises(ValueError, match="ln_lambda must be positive"):
        braginskii_collision_rate(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=-1.0)


def test_elastic_efficiency_electron_self_is_half() -> None:
    # Equal masses transfer half their energy on average.
    """Check the equal-mass value with the default electron field."""
    assert elastic_energy_transfer_efficiency(_ELECTRON_AMU) == pytest.approx(0.5)


def test_elastic_efficiency_heavy_ion_mass_suppressed() -> None:
    # Deuterium against the electron field: delta ~ 2 m_e / m_D.
    """Check the retained electron-to-ion mass suppression factor."""
    delta = elastic_energy_transfer_efficiency(2.0)
    assert delta == pytest.approx(2.0 * 2.0 * _ELECTRON_AMU / (2.0 + _ELECTRON_AMU) ** 2)
    assert delta < 1.0e-3


def test_elastic_efficiency_rejects_invalid() -> None:
    """Reject nonpositive test and field masses through the public API."""
    with pytest.raises(ValueError, match="mass_amu must be positive"):
        elastic_energy_transfer_efficiency(0.0)
    with pytest.raises(ValueError, match="field_mass_amu must be positive"):
        elastic_energy_transfer_efficiency(2.0, field_mass_amu=-1.0)


def test_independent_collision_rates_assembles_channels() -> None:
    """Check assembly of the three rates and two dimensionless factors."""
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
    """Reject nonpositive test and field temperatures during assembly."""
    with pytest.raises(ValueError, match="T_e_keV must be positive"):
        independent_collision_rates(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_e_19=10.0, T_e_keV=0.0)
    with pytest.raises(ValueError, match="temperature_keV must be positive"):
        independent_collision_rates(mass_amu=2.0, charge_e=1.0, temperature_keV=-8.0, n_e_19=10.0, T_e_keV=4.0)


@pytest.mark.parametrize("function", [chandrasekhar_g, deflection_shape])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_velocity_functions_refuse_nonfinite_without_mutation(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]], bad: float
) -> None:
    """Refuse the whole input before calculation and preserve caller bytes."""
    values = np.array([[0.5, bad], [1.0, 2.0]])
    original = values.tobytes()
    with pytest.raises(ValueError, match="requires finite arguments"):
        function(values)
    assert values.tobytes() == original


@pytest.mark.parametrize("function", [chandrasekhar_g, deflection_shape])
def test_velocity_functions_preserve_shapes_and_readonly_views(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> None:
    """Exercise scalar, empty and strided read-only public array inputs."""
    scalar = function(np.array(1.0))
    assert scalar.shape == () and scalar.dtype == np.float64 and np.isfinite(scalar)
    empty = function(np.empty((2, 0, 3), dtype=np.float64))
    assert empty.shape == (2, 0, 3) and empty.dtype == np.float64
    base = np.linspace(0.1, 3.0, 12, dtype=np.float64).reshape(3, 4)
    view = base[:, ::2]
    view.flags.writeable = False
    before = base.tobytes()
    output = function(view)
    assert output.shape == view.shape and np.all(np.isfinite(output))
    assert not np.shares_memory(output, view) and output.flags.writeable
    assert base.tobytes() == before


def test_chandrasekhar_zero_limit() -> None:
    """The removable zero singularity has the exact documented zero limit."""
    np.testing.assert_array_equal(chandrasekhar_g(np.array([0.0, -0.0])), [0.0, 0.0])


@pytest.mark.parametrize("function", [maxwellian_deflection_average_factor])
@pytest.mark.parametrize("bad", [2.5, "64", None, np.int64(64)])
def test_quadrature_count_runtime_types(function: Callable[..., float], bad: object) -> None:
    """The actual public callable refuses counts outside Python int's domain."""
    with pytest.raises(ValueError, match="n_quad must be an integer >= 2"):
        function(n_quad=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_quadrature_limit_requires_finite(bad: float) -> None:
    """Finite-limit refusal occurs before constructing any quadrature."""
    with pytest.raises(ValueError, match="x_max must be finite"):
        maxwellian_deflection_average_factor(x_max=bad)


@pytest.mark.parametrize("function", [basic_collision_frequency, thermal_deflection_rate, braginskii_collision_rate])
def test_collision_rates_signed_and_zero_charge(function: Callable[..., float]) -> None:
    """The unchanged fourth-power charge convention is sign symmetric."""
    arguments = dict(mass_amu=2.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    positive = function(charge_e=1.0, **arguments)
    assert positive > 0.0
    assert function(charge_e=-1.0, **arguments) == positive
    assert function(charge_e=0.0, **arguments) == 0.0


@pytest.mark.parametrize("function", [basic_collision_frequency, thermal_deflection_rate, braginskii_collision_rate])
@pytest.mark.parametrize("name", ["mass_amu", "temperature_keV", "n_field_19", "ln_lambda", "charge_e"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_collision_rates_nonfinite_scalar_domains(function: Callable[..., float], name: str, bad: float) -> None:
    """Every documented scalar input has an authored finite-domain refusal."""
    arguments = dict(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, ln_lambda=17.0)
    arguments[name] = bad
    with pytest.raises(ValueError, match=name + " must be finite"):
        function(**arguments)


@pytest.mark.parametrize("function", [thermal_deflection_rate, braginskii_collision_rate])
def test_effective_charge_nonfinite(function: Callable[..., float]) -> None:
    """The positive effective-charge domain also excludes nonfinite values."""
    with pytest.raises(ValueError, match="z_eff must be finite"):
        function(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_field_19=10.0, z_eff=float("nan"))


@pytest.mark.parametrize("name", ["mass_amu", "field_mass_amu"])
def test_elastic_efficiency_nonfinite_masses(name: str) -> None:
    """Both masses are checked through the public efficiency API."""
    arguments = dict(mass_amu=2.0, field_mass_amu=_ELECTRON_AMU)
    arguments[name] = float("nan")
    with pytest.raises(ValueError, match=name + " must be finite"):
        elastic_energy_transfer_efficiency(**arguments)


@pytest.mark.parametrize("name", ["temperature_keV", "T_e_keV", "n_e_19"])
def test_assembled_rates_nonfinite_fields(name: str) -> None:
    """Assembly retains both temperature checks and the density alias domain."""
    arguments = dict(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_e_19=10.0, T_e_keV=4.0)
    arguments[name] = float("nan")
    message = "n_field_19" if name == "n_e_19" else name
    with pytest.raises(ValueError, match=message + " must be finite"):
        independent_collision_rates(
            mass_amu=arguments["mass_amu"],
            charge_e=arguments["charge_e"],
            temperature_keV=arguments["temperature_keV"],
            n_e_19=arguments["n_e_19"],
            T_e_keV=arguments["T_e_keV"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "thermal_deflection_rate",
        "braginskii_rate",
        "energy_relaxation_rate",
        "elastic_energy_transfer_efficiency",
        "field_temperature_factor",
    ],
)
def test_reference_rates_observation_is_frozen_and_unchecked(field: str) -> None:
    """Construction is an observation container, with no implicit admission."""
    rates = IndependentCollisionRates(float("nan"), -1.0, 0.0, 4.0, -2.0)
    assert np.isnan(rates.thermal_deflection_rate) and rates.elastic_energy_transfer_efficiency == 4.0
    with pytest.raises(FrozenInstanceError):
        setattr(rates, field, 2.0)


def test_documented_reference_example() -> None:
    """Run the public documentation example with the declared channel units."""
    rates = independent_collision_rates(mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_e_19=10.0, T_e_keV=4.0)
    assert rates.thermal_deflection_rate > 0.0
    assert rates.braginskii_rate > 0.0
    assert rates.energy_relaxation_rate > 0.0
    assert rates.field_temperature_factor == np.sqrt(2.0)
