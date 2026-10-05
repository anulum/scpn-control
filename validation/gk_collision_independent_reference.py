#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Fokker-Planck collision-coefficient reference
"""Independent test-particle Fokker-Planck collision-coefficient reference.

This module is a deliberately *independent* second derivation of the bounded
test-particle collision coefficients used by
:mod:`scpn_control.core.gk_species`, built to satisfy the tracker #47 required
action *"validate bounded test-particle collision coefficients against a
published gyrokinetic or Fokker-Planck reference case"*.

Independence is structural. The production ``collision_frequencies`` uses a
single closed-form deflection coefficient. This reference instead reconstructs
the deflection rate from the *velocity-dependent* Fokker-Planck test-particle
deflection frequency

    nu_D(v) = nu_hat * [erf(x) - G(x)] / x^3 ,   x = v / v_th ,

where ``G`` is the Chandrasekhar function, and Maxwellian-averages it by
Gauss-Legendre quadrature to obtain the thermal deflection rate.  A completely
separate closed-form route — the Braginskii/NRL collision rate ``1/tau`` — is
provided as a second, independent anchor.  Because the two references share no
code with the production coefficient, agreement of the *scaling* (an exactly
constant production/reference ratio across density, temperature, effective
charge, and species mass) is genuine validation of the functional form, while
the *value* of that ratio quantifies the bounded O(1) prefactor by which the
production coefficient differs from each canonical convention. This is a
local structural comparison, not an externally supplied reference case or
admission of a full conserving collision operator or quantitative damping.

The energy-relaxation channel is cross-checked against the independent
elastic-collision mean energy-transfer efficiency

    delta_ab = 2 m_a m_b / (m_a + m_b)^2 ,

a textbook kinematic identity (the angle-averaged fractional energy transfer in
an elastic two-body collision), which fixes the mass-ratio suppression and the
field-temperature dependence of the energy-relaxation rate independently of the
production code.

References
----------
  - Helander & Sigmar, *Collisional Transport in Magnetized Plasmas* (2002), §3.
  - Braginskii, *Reviews of Plasma Physics* 1 (1965) 205.
  - NRL Plasma Formulary (2019), collision frequencies.
  - Sugama & Watanabe, Phys. Plasmas 13 (2006) 012501.

Input and execution contract
----------------------------
Masses are multiples of the proton mass (the historical ``mass_amu`` name),
charges are multiples of elementary charge, temperatures are in keV and
densities are in 10^19 m^-3. Scalar inputs are converted with ``float`` and
must be finite; masses, temperatures, densities, effective charge and Coulomb
logarithm must also be positive. Charge may be signed or zero; both rate
normalisations use its fourth power. Invalid domains raise authored
``ValueError``; native conversion and extreme floating arithmetic errors
propagate. Positive finite inputs alone do not guarantee representable results.

Functions allocate fresh results and do not mutate caller arrays, retain
state, read data, write reports or establish a training/facility admission.
The public API is this importable source-tree module; it has no CLI. The
separate ``validate_gk_collision_independent`` producer owns its own CLI and
report contract. Literature citations above motivate the retained equations;
they do not identify a downloaded, checksum-bound external numerical case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.special import erf

FloatArray = NDArray[np.float64]

_E_CHARGE = 1.602176634e-19  # C
_M_PROTON = 1.67262192369e-27  # kg
_M_ELECTRON = 9.1093837015e-31  # kg
_EPS0 = 8.8541878128e-12  # F/m
_ELECTRON_AMU = _M_ELECTRON / _M_PROTON


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    """Convert a scalar and refuse nonfinite or requested nonpositive domains."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def chandrasekhar_g(x: FloatArray) -> FloatArray:
    """Chandrasekhar function ``G(x) = [erf(x) - x erf'(x)] / (2 x^2)``.

    With ``erf'(x) = (2/sqrt(pi)) exp(-x^2)``. The removable singularity at
    ``x -> 0`` (where ``G(x) -> 2x / (3 sqrt(pi))``) is handled by a series
    expansion. ``x`` is a finite, non-negative dimensionless speed ratio;
    float64 conversion preserves arbitrary shape, including scalar and empty
    arrays. The returned array is fresh and has that shape; zero maps to zero.
    Negative or nonfinite elements raise authored ``ValueError`` before any
    calculation. Conversion errors propagate; extreme finite inputs retain
    NumPy's ordinary overflow/underflow behaviour.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("chandrasekhar_g requires finite arguments")
    if np.any(x_arr < 0.0):
        raise ValueError("chandrasekhar_g requires non-negative arguments")
    small = x_arr < 1.0e-3
    safe = np.where(small, 1.0, x_arr)
    erf_prime = (2.0 / np.sqrt(np.pi)) * np.exp(-safe * safe)
    general = (erf(safe) - safe * erf_prime) / (2.0 * safe * safe)
    # Leading series term: G(x) = 2x/(3 sqrt(pi)) - 2x^3/(5 sqrt(pi)) + ...
    series = (2.0 * x_arr / (3.0 * np.sqrt(np.pi))) * (1.0 - (3.0 / 5.0) * x_arr * x_arr)
    return np.asarray(np.where(small, series, general), dtype=np.float64)


def deflection_shape(x: FloatArray) -> FloatArray:
    """Velocity shape of the test-particle deflection frequency.

    Returns ``[erf(x) - G(x)] / x^3`` — the dimensionless velocity dependence
    of ``nu_D(v)`` with ``x = v / v_th``. This is the quantity Maxwellian-
    averaged to form the thermal deflection rate. ``x`` must contain finite,
    strictly positive dimensionless speeds; scalar and empty arrays are
    supported. A fresh float64 array preserves shape and caller bytes.
    Nonfinite or nonpositive elements raise authored ``ValueError``;
    conversion and extreme floating arithmetic retain native behaviour.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("deflection_shape requires finite arguments")
    if np.any(x_arr <= 0.0):
        raise ValueError("deflection_shape requires strictly positive arguments")
    return np.asarray((erf(x_arr) - chandrasekhar_g(x_arr)) / x_arr**3, dtype=np.float64)


def maxwellian_deflection_average_factor(*, n_quad: int = 64, x_max: float = 10.0) -> float:
    """Maxwellian speed-average of :func:`deflection_shape`.

    Computes ``<[erf-G]/x^3> = int f(x) x^2 e^{-x^2} dx / int x^2 e^{-x^2} dx``
    by Gauss-Legendre quadrature on ``[0, x_max]``. The result is a pure number
    (~1.3878 at the defaults), independent of plasma parameters. ``n_quad``
    must be a Python integer >= 2, excluding bool; ``x_max`` is finite and
    positive. Invalid domains raise authored ``ValueError``. There is no upper
    resource bound or adaptive error estimate. Truncation and resolution must
    be assessed together; arbitrary admitted choices are not certified
    converged. Very small/large finite limits may underflow the denominator or
    overflow intermediate values, with native arithmetic errors propagated.
    """
    if not isinstance(n_quad, int) or isinstance(n_quad, bool) or n_quad < 2:
        raise ValueError("n_quad must be an integer >= 2")
    x_max = _finite("x_max", x_max, positive=True)
    quadrature: Callable[[int], tuple[FloatArray, FloatArray]] = leggauss
    nodes, weights = quadrature(n_quad)
    x = 0.5 * x_max * (nodes + 1.0)
    w = 0.5 * x_max * weights
    maxwellian = x * x * np.exp(-x * x)
    numerator = float(np.sum(w * deflection_shape(x) * maxwellian))
    denominator = float(np.sum(w * maxwellian))
    return numerator / denominator


def basic_collision_frequency(
    *,
    mass_amu: float,
    charge_e: float,
    temperature_keV: float,
    n_field_19: float,
    ln_lambda: float,
) -> float:
    """Helander-Sigmar basic self-collision frequency ``nu_hat`` [s^-1].

    ``nu_hat = n q^4 lnL / (4 pi eps0^2 m^2 v_th^3)`` with ``v_th = sqrt(2T/m)``.
    This is the normalising rate of the velocity-dependent deflection frequency.
    Inputs follow the module's mass, charge, keV, density and dimensionless
    Coulomb-logarithm contract. The scalar return is in s^-1, without v_th/R
    normalisation; either charge sign gives the same rate and zero gives zero.
    Nonfinite/nonpositive domains raise authored ``ValueError`` as described
    above; conversion and unrepresentable arithmetic propagate unchanged.
    """
    mass_amu = _finite("mass_amu", mass_amu, positive=True)
    charge_e = _finite("charge_e", charge_e)
    temperature_keV = _finite("temperature_keV", temperature_keV, positive=True)
    n_field_19 = _finite("n_field_19", n_field_19, positive=True)
    ln_lambda = _finite("ln_lambda", ln_lambda, positive=True)

    mass = mass_amu * _M_PROTON
    charge = abs(charge_e) * _E_CHARGE
    n_field = n_field_19 * 1.0e19
    t_joule = temperature_keV * 1.0e3 * _E_CHARGE
    v_th = np.sqrt(2.0 * t_joule / mass)
    return float(n_field * charge**4 * ln_lambda / (4.0 * np.pi * _EPS0**2 * mass**2 * v_th**3))


def thermal_deflection_rate(
    *,
    mass_amu: float,
    charge_e: float,
    temperature_keV: float,
    n_field_19: float,
    z_eff: float = 1.0,
    ln_lambda: float = 17.0,
    n_quad: int = 64,
) -> float:
    """First-principles Maxwellian-averaged deflection rate ``<nu_D>`` [s^-1].

    ``<nu_D> = Z_eff * nu_hat * <[erf-G]/x^3>`` — the deflection frequency
    reconstructed from the velocity-dependent Fokker-Planck coefficient and a
    numerical Maxwellian average, sharing no code with the production closed
    form. Inputs follow the module contract, with positive dimensionless
    ``z_eff`` and Python integer ``n_quad >= 2`` (not bool). Defaults use a
    fixed speed limit of 10; the scalar return is in s^-1. Invalid domains
    raise authored ``ValueError``; quadrature has no resource or error bound,
    and conversion/extreme arithmetic errors propagate.
    """
    z_eff = _finite("z_eff", z_eff, positive=True)
    nu_hat = basic_collision_frequency(
        mass_amu=mass_amu,
        charge_e=charge_e,
        temperature_keV=temperature_keV,
        n_field_19=n_field_19,
        ln_lambda=ln_lambda,
    )
    return z_eff * nu_hat * maxwellian_deflection_average_factor(n_quad=n_quad)


def braginskii_collision_rate(
    *,
    mass_amu: float,
    charge_e: float,
    temperature_keV: float,
    n_field_19: float,
    z_eff: float = 1.0,
    ln_lambda: float = 17.0,
) -> float:
    """Canonical Braginskii/NRL collision rate ``1/tau`` [s^-1].

    ``1/tau = Z_eff * sqrt(2) n Z^4 e^4 lnL / (12 pi^{3/2} eps0^2 sqrt(m) T^{3/2})``.
    A second, closed-form independent anchor (no Chandrasekhar function, no
    quadrature) for the deflection-rate prefactor. Inputs follow the module
    contract, including positive dimensionless ``z_eff`` and ``ln_lambda``.
    The scalar return is in s^-1; signed/zero charge behaves as in
    ``basic_collision_frequency``. Invalid domains raise authored
    ``ValueError``; conversion and extreme arithmetic errors propagate.
    """
    mass_amu = _finite("mass_amu", mass_amu, positive=True)
    charge_e = _finite("charge_e", charge_e)
    temperature_keV = _finite("temperature_keV", temperature_keV, positive=True)
    n_field_19 = _finite("n_field_19", n_field_19, positive=True)
    z_eff = _finite("z_eff", z_eff, positive=True)
    ln_lambda = _finite("ln_lambda", ln_lambda, positive=True)

    mass = mass_amu * _M_PROTON
    z_charge = abs(charge_e)
    n_field = n_field_19 * 1.0e19
    t_joule = temperature_keV * 1.0e3 * _E_CHARGE
    return float(
        z_eff
        * np.sqrt(2.0)
        * n_field
        * z_charge**4
        * _E_CHARGE**4
        * ln_lambda
        / (12.0 * np.pi**1.5 * _EPS0**2 * np.sqrt(mass) * t_joule**1.5)
    )


def elastic_energy_transfer_efficiency(mass_amu: float, field_mass_amu: float = _ELECTRON_AMU) -> float:
    """Mean fractional energy transfer in an elastic collision ``2 m_a m_b/(m_a+m_b)^2``.

    The angle-averaged fractional energy exchanged when a test particle of mass
    ``m_a`` collides elastically with a field particle of mass ``m_b``. This is
    the kinematic factor that suppresses ion energy relaxation against the light
    electron field by the mass ratio, derived independently of the production
    energy-relaxation coefficient. Both masses must be finite and positive
    multiples of the proton mass; the default field mass is the electron to
    proton mass ratio. The scalar return is dimensionless (0.5 for equal
    representable masses). Invalid domains raise authored ``ValueError``;
    conversion and extreme arithmetic errors propagate.
    """
    mass_amu = _finite("mass_amu", mass_amu, positive=True)
    field_mass_amu = _finite("field_mass_amu", field_mass_amu, positive=True)
    return float(2.0 * mass_amu * field_mass_amu / (mass_amu + field_mass_amu) ** 2)


@dataclass(frozen=True)
class IndependentCollisionRates:
    """Frozen observation of three rates and two dimensionless factors.

    ``thermal_deflection_rate``, ``braginskii_rate`` and
    ``energy_relaxation_rate`` are in s^-1. The elastic efficiency and
    ``field_temperature_factor = sqrt(T_s/T_e)`` are dimensionless.
    Direct construction does not validate fields or certify their provenance;
    frozen prevents field reassignment. Use ``independent_collision_rates``
    for the module's scalar input checks and calculations.
    """

    thermal_deflection_rate: float  # Maxwellian-averaged <nu_D> [s^-1]
    braginskii_rate: float  # canonical 1/tau [s^-1]
    energy_relaxation_rate: float  # <nu_D> * elastic-efficiency * sqrt(T_s/T_e) [s^-1]
    elastic_energy_transfer_efficiency: float  # dimensionless mass-ratio factor
    field_temperature_factor: float  # sqrt(T_s / T_e), dimensionless


def independent_collision_rates(
    *,
    mass_amu: float,
    charge_e: float,
    temperature_keV: float,
    n_e_19: float,
    T_e_keV: float,
    z_eff: float = 1.0,
    ln_lambda: float = 17.0,
    n_quad: int = 64,
) -> IndependentCollisionRates:
    """Assemble the independent Fokker-Planck reference rates for one case.

    The deflection rate is the Maxwellian-averaged Fokker-Planck coefficient;
    the Braginskii rate is the closed-form canonical anchor; the energy
    relaxation rate applies the independent elastic energy-transfer efficiency
    (versus the electron field) and the field-temperature factor to the
    deflection rate — mirroring the *physics* of the production energy channel
    without reusing its code. Inputs follow the module contract; ``n_e_19``
    is density in 10^19 m^-3 and ``T_e_keV`` is a positive field temperature
    in keV. ``n_quad`` follows the fixed quadrature contract. A fresh frozen
    ``IndependentCollisionRates`` contains three rates in s^-1 and two
    dimensionless factors. Invalid domains raise authored ``ValueError``;
    conversion, resource and extreme arithmetic errors propagate. The result
    carries no external-reference, conservation or quantitative-damping proof.
    """
    temperature_keV = _finite("temperature_keV", temperature_keV, positive=True)
    T_e_keV = _finite("T_e_keV", T_e_keV, positive=True)
    deflection = thermal_deflection_rate(
        mass_amu=mass_amu,
        charge_e=charge_e,
        temperature_keV=temperature_keV,
        n_field_19=n_e_19,
        z_eff=z_eff,
        ln_lambda=ln_lambda,
        n_quad=n_quad,
    )
    braginskii = braginskii_collision_rate(
        mass_amu=mass_amu,
        charge_e=charge_e,
        temperature_keV=temperature_keV,
        n_field_19=n_e_19,
        z_eff=z_eff,
        ln_lambda=ln_lambda,
    )
    efficiency = elastic_energy_transfer_efficiency(mass_amu)
    field_factor = float(np.sqrt(temperature_keV / T_e_keV))
    return IndependentCollisionRates(
        thermal_deflection_rate=deflection,
        braginskii_rate=braginskii,
        energy_relaxation_rate=deflection * efficiency * field_factor,
        elastic_energy_transfer_efficiency=efficiency,
        field_temperature_factor=field_factor,
    )
