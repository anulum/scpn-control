# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physical local Miller inputs for standalone TGLF.

"""Map explicit nonrotating isotropic species and Miller geometry to a GYRO SAT0 deck."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import fsum, isclose, isfinite, pi, prod, sqrt

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.tglf_units import TGLFReferenceUnits

_KEV_J = 1.602176634e-16
_E = 1.602176634e-19
_MU0 = 4 * pi * 1e-7
_EPS0 = 1 / (_MU0 * 299792458.0**2)


def _finite(value: float, name: str, *, positive: bool = False) -> None:
    """Reject booleans, nonfinite scalars and nonpositive physical scales."""
    if isinstance(value, bool) or not isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))


def _product(*values: float) -> float:
    """Multiply physical factors without silently emitting infinity or losing a nonzero result."""
    result = prod(values)
    if not isfinite(result) or (result == 0 and all(value != 0 for value in values)):
        raise ValueError("Miller input product is outside the working range")
    return result


def _ratio(value: float, scale: float) -> float:
    """Normalize a scalar while refusing overflow and nonzero-to-zero underflow."""
    result = value / scale
    if not isfinite(result) or (value != 0 and result == 0):
        raise ValueError("Miller input ratio is outside the working range")
    return result


@dataclass(frozen=True)
class TGLFSpecies:
    """One nonrotating isotropic species at a physical minor-radius surface.

    Parameters
    ----------
    charge_e : float
        Signed charge in elementary-charge units. The deck requires electron
        charge -1 first, then positive integral ion charge states.
    mass_kg : float
        Declared species mass [kg], positive; no mass is inferred from charge.
    density_m3 : float
        Local number density [m^-3], positive. Omit absent species explicitly;
        a zero-density species has no logarithmic-gradient contract here.
    temperature_kev : float
        Local isotropic temperature [keV], positive.
    density_gradient_m4 : float
        Signed dn/dr [m^-4], with physical minor radius r in metres.
    temperature_gradient_kev_m : float
        Signed dT/dr [keV/m]. No R/L or normalized-coordinate gradient is inferred.
    """

    charge_e: float
    mass_kg: float
    density_m3: float
    temperature_kev: float
    density_gradient_m4: float
    temperature_gradient_kev_m: float

    def __post_init__(self) -> None:
        """Check local species scalars before logarithmic normalization."""
        for name in ("charge_e", "density_gradient_m4", "temperature_gradient_kev_m"):
            _finite(getattr(self, name), name)
        for name in ("mass_kg", "density_m3", "temperature_kev"):
            _finite(getattr(self, name), name, positive=True)


@dataclass(frozen=True)
class TGLFMillerGeometry:
    """Local up/down-symmetric Miller shape and physical radial derivatives.

    Parameters
    ----------
    minor_radius_m, major_radius_m : float
        Physical local minor radius r and surface-centre major radius R [m],
        with 0 < r < R. The axis is outside this local model.
    safety_factor : float
        Positive magnitude q; current/field orientation is supplied separately.
    safety_factor_gradient_m : float
        Signed dq/dr [m^-1], not magnetic shear or a poloidal-flux derivative.
    elongation, triangularity : float
        Shape kappa > 0 and |delta| < 1.
    elongation_gradient_m, triangularity_gradient_m : float
        Signed dkappa/dr and ddelta/dr [m^-1]. Current GACODE conventions are
        S_KAPPA = r/kappa * dkappa/dr and S_DELTA = r * ddelta/dr.
    major_radius_gradient : float
        Signed dR/dr, dimensionless. Elevation, squareness and higher harmonics
        are zero in this Miller contract, rather than inferred from equilibrium.

    Notes
    -----
    Scalar domain checks do not certify nested flux surfaces or a force-balanced
    equilibrium. Bunit must come from the caller's matching magnetic geometry.
    """

    minor_radius_m: float
    major_radius_m: float
    safety_factor: float
    safety_factor_gradient_m: float
    elongation: float = 1.0
    triangularity: float = 0.0
    elongation_gradient_m: float = 0.0
    triangularity_gradient_m: float = 0.0
    major_radius_gradient: float = 0.0

    def __post_init__(self) -> None:
        """Validate finite local shape parameters without pretending to solve an equilibrium."""
        for name in ("minor_radius_m", "major_radius_m", "safety_factor", "elongation"):
            _finite(getattr(self, name), name, positive=True)
        for name in (
            "safety_factor_gradient_m",
            "triangularity",
            "elongation_gradient_m",
            "triangularity_gradient_m",
            "major_radius_gradient",
        ):
            _finite(getattr(self, name), name)
        if self.minor_radius_m >= self.major_radius_m or abs(self.triangularity) >= 1:
            raise ValueError("Miller geometry requires r < R and |triangularity| < 1")


def miller_tglf_deck(
    reference: TGLFReferenceUnits,
    species: tuple[TGLFSpecies, ...],
    geometry: TGLFMillerGeometry,
    *,
    electron_collision_rate_s: float,
    nky: int = 12,
    modes: int = 2,
    sign_bt: int = 1,
    sign_it: int = 1,
    use_bper: bool = False,
    use_bpar: bool = False,
) -> str:
    """Build a deterministic physical-input GYRO SAT0 standalone deck without executing TGLF.

    Parameters
    ----------
    reference : TGLFReferenceUnits
        Explicit n0, T0, a0, m0 and positive Bunit. a0 is the edge minor radius;
        local gradients use physical r. References are carried into every ratio.
    species : tuple of TGLFSpecies
        Electron first, then one to six positive-charge ions, in output order.
        Both sum(Z*n) = ne and sum(Z*dn/dr) = dne/dr must hold to relative 1e-12.
        Positive-temperature Maxwellian pressure includes every supplied species.
    geometry : TGLFMillerGeometry
        Matching local shape, q and physical radial derivatives.
    electron_collision_rate_s : float
        Nonnegative electron-ion frequency [s^-1] in TGLF's convention (the NRL
        driver uses 3*sqrt(pi)/(4*tau_e)). This is an explicit model input,
        not nu_star, Z_eff times a guessed rate, or a selected collision formula.
    nky : int
        Positive high-k grid setting, not the number of emitted ky rows.
    modes : int
        SAT0 mode fit, 2 or 4. Other counts can be rewritten by provider presets.
    sign_bt, sign_it : int
        Field/current signs, +/-1 relative to counterclockwise viewed from above.
    use_bper, use_bpar : bool
        Perpendicular/parallel magnetic fluctuation switches; BPAR requires BPER.

    Returns
    -------
    str
        Key=value input with 17-digit float serialization. All species and shape
        values and zero-flow assumptions are explicit; unlisted numerical
        defaults remain provider-owned and are captured in input.tglf.gen.

    Raises
    ------
    ValueError
        Invalid composition, charge/gradient balance, geometry, switches or
        unrepresentable normalization. Tiny radii that GACODE clamps are refused.

    Notes
    -----
    RLNS=-a0*n'/n, RLTS=-a0*T'/T, Q_PRIME=q*a0^2*q'/r and
    P_PRIME=mu0/(4*pi*Bunit^2)*q*a0^2*p'/r with p in Pa. BETAE uses n0*T0,
    XNUE=nu_ei*a0/cs0 and DEBYE=lambda_D0/rho_s0. Scalar temperatures are
    isotropic; all parallel/ExB velocities and shears are zero. This builder
    supplies no rotation, MXH harmonics, anisotropic-pressure closure, radial
    sampling or equilibrium/calibration certificate. SAT1/2/3 require a separate
    qualified model/normalization contract and are not silently substituted.
    """
    _finite(electron_collision_rate_s, "electron_collision_rate_s")
    if electron_collision_rate_s < 0:
        raise ValueError("electron_collision_rate_s must be nonnegative")
    if isinstance(nky, bool) or not isinstance(nky, int) or nky <= 0:
        raise ValueError("nky must be a positive integer")
    if isinstance(modes, bool) or modes not in (2, 4) or not isinstance(modes, int):
        raise ValueError("SAT0 modes must be 2 or 4")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1) for value in (sign_bt, sign_it)
    ):
        raise ValueError("field and current signs must be +/-1")
    if not isinstance(use_bper, bool) or not isinstance(use_bpar, bool) or (use_bpar and not use_bper):
        raise ValueError("field switches must be boolean; BPAR requires BPER")
    if not 2 <= len(species) <= 7 or species[0].charge_e != -1:
        raise ValueError("Require an electron followed by one to six ions")
    ions = species[1:]
    if any(ion.charge_e <= 0 or ion.charge_e != int(ion.charge_e) for ion in ions):
        raise ValueError("Ions require positive integral charge states")
    try:
        charge_density = fsum(_product(ion.charge_e, ion.density_m3) for ion in ions)
        charge_gradients = [_product(ion.charge_e, ion.density_gradient_m4) for ion in ions]
        gradient_error = fsum([*charge_gradients, -species[0].density_gradient_m4])
        gradient_scale = fsum(abs(value) for value in [*charge_gradients, species[0].density_gradient_m4])
        if not isclose(charge_density, species[0].density_m3, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Species density must be charge neutral")
        if gradient_error != 0 and abs(_ratio(gradient_error, gradient_scale)) > 1e-12:
            raise ValueError("Species density gradients must be charge neutral")
        a = reference.length_m
        radius = _ratio(geometry.minor_radius_m, a)
        if not 1e-5 <= radius <= 1:
            raise ValueError("Local r/a must be in [1e-5, 1]; no provider axis clamp")
        t0 = _product(reference.temperature_kev, _KEV_J)
        cs = sqrt(_ratio(t0, reference.mass_kg))
        rho_s = _ratio(sqrt(_product(t0, reference.mass_kg)), _product(_E, reference.magnetic_field_t))
        pressure_gradient = fsum(
            _product(_KEV_J, value)
            for sp in species
            for value in (
                _product(sp.density_gradient_m4, sp.temperature_kev),
                _product(sp.density_m3, sp.temperature_gradient_kev_m),
            )
        )
        q_over_r = _ratio(geometry.safety_factor, radius)
        field_squared = _product(reference.magnetic_field_t, reference.magnetic_field_t)
        deck: dict[str, float | int | str] = {
            "UNITS": "GYRO",
            "SAT_RULE": 0,
            "XNU_MODEL": 2,
            "USE_TRANSPORT_MODEL": ".true.",
            "IFLUX": ".true.",
            "ADIABATIC_ELEC": ".false.",
            "GEOMETRY_FLAG": 1,
            "SIGN_BT": sign_bt,
            "SIGN_IT": sign_it,
            "NS": len(species),
            "NKY": nky,
            "NMODES": modes,
            "USE_BPER": ".true." if use_bper else ".false.",
            "USE_BPAR": ".true." if use_bpar else ".false.",
            "RMIN_LOC": radius,
            "RMAJ_LOC": _ratio(geometry.major_radius_m, a),
            "Q_LOC": geometry.safety_factor,
            "Q_PRIME_LOC": _product(q_over_r, a, geometry.safety_factor_gradient_m),
            "P_PRIME_LOC": _product(_ratio(1e-7, field_squared), q_over_r, a, pressure_gradient),
            "KAPPA_LOC": geometry.elongation,
            "DELTA_LOC": geometry.triangularity,
            "S_KAPPA_LOC": _product(
                _ratio(geometry.minor_radius_m, geometry.elongation), geometry.elongation_gradient_m
            ),
            "S_DELTA_LOC": _product(geometry.minor_radius_m, geometry.triangularity_gradient_m),
            "DRMAJDX_LOC": geometry.major_radius_gradient,
            "DRMINDX_LOC": 1.0,
            "ZMAJ_LOC": 0.0,
            "DZMAJDX_LOC": 0.0,
            "ZETA_LOC": 0.0,
            "S_ZETA_LOC": 0.0,
            "BETAE": _ratio(_product(2, _MU0, reference.density_m3, t0), field_squared),
            "XNUE": _ratio(_product(electron_collision_rate_s, a), cs),
            "ZEFF": _ratio(
                fsum(_product(sp.density_m3, sp.charge_e, sp.charge_e) for sp in ions), species[0].density_m3
            ),
            "DEBYE": _ratio(sqrt(_ratio(_product(_EPS0, t0), _product(reference.density_m3, _E, _E))), rho_s),
            "VEXB": 0.0,
            "VEXB_SHEAR": 0.0,
        }
        for index, sp in enumerate(species, 1):
            deck.update(
                {
                    f"MASS_{index}": _ratio(sp.mass_kg, reference.mass_kg),
                    f"ZS_{index}": sp.charge_e,
                    f"AS_{index}": _ratio(sp.density_m3, reference.density_m3),
                    f"TAUS_{index}": _ratio(sp.temperature_kev, reference.temperature_kev),
                    f"RLNS_{index}": -_product(a, _ratio(sp.density_gradient_m4, sp.density_m3)),
                    f"RLTS_{index}": -_product(a, _ratio(sp.temperature_gradient_kev_m, sp.temperature_kev)),
                    f"VPAR_{index}": 0.0,
                    f"VPAR_SHEAR_{index}": 0.0,
                }
            )
        for index in range(7):
            deck[f"SHAPE_COS{index}"] = deck[f"SHAPE_S_COS{index}"] = 0.0
        for index in range(3, 7):
            deck[f"SHAPE_SIN{index}"] = deck[f"SHAPE_S_SIN{index}"] = 0.0
        return "".join(
            f"{key}={value if isinstance(value, str) else format(value, '.17g')}\n" for key, value in deck.items()
        )
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Miller physical inputs are outside the working range") from exc


def miller_volume_metric(geometry: TGLFMillerGeometry) -> tuple[float, float]:
    """Integrate enclosed toroidal volume and dV/dr for a local Miller surface.

    Parameters
    ----------
    geometry : TGLFMillerGeometry
        Physical r, R, shape and their radial derivatives. The same surface
        must supply the local TGLF input; q does not enter the volume metric.

    Returns
    -------
    tuple of float
        Enclosed volume [m^3] and its physical minor-radius derivative [m^2].
        Multiply the TGLF radial flux by dV/dr to obtain an integrated rate;
        the geometric surface area is generally different.

    Raises
    ------
    ValueError
        Nonpositive sampled radial Jacobian, nonfinite arithmetic, nonpositive
        volume/derivative or failure of 32-to-256-point quadrature refinement.

    Notes
    -----
    For R(theta)=R0+r*cos(theta+asin(delta)*sin(theta)), Z=kappa*r*sin(theta),
    Green's theorem gives V=pi*integral(R^2*dZ/dtheta). Differentiating this
    integral yields dV/dr, including shape and major-radius derivatives.
    Success requires successive Gauss-Legendre estimates to agree at relative
    1e-12. This is measured quadrature convergence and sampled local nesting,
    not a rigorous global non-intersection or force-balance certificate.
    """
    g = geometry
    previous: tuple[float, float] | None = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for count in (32, 64, 128, 256):
                quadrature: Callable[[int], tuple[NDArray[np.float64], NDArray[np.float64]]] = (
                    np.polynomial.legendre.leggauss
                )
                nodes, weights = quadrature(count)
                theta = pi * (nodes + 1)
                angle = theta + np.arcsin(g.triangularity) * np.sin(theta)
                major = g.major_radius_m + g.minor_radius_m * np.cos(angle)
                major_theta = -g.minor_radius_m * np.sin(angle) * (1 + np.arcsin(g.triangularity) * np.cos(theta))
                major_radial = (
                    g.major_radius_gradient
                    + np.cos(angle)
                    - g.minor_radius_m
                    * np.sin(angle)
                    * np.sin(theta)
                    * g.triangularity_gradient_m
                    / sqrt(1 - g.triangularity**2)
                )
                height_theta = g.elongation * g.minor_radius_m * np.cos(theta)
                height_radial = (g.elongation + g.minor_radius_m * g.elongation_gradient_m) * np.sin(theta)
                jacobian = major_radial * height_theta - major_theta * height_radial
                if not np.all(np.isfinite(jacobian)) or np.any(jacobian <= 0):
                    raise ValueError("Miller surfaces fail sampled local nesting")
                displacement = g.minor_radius_m * np.cos(angle)
                displacement_radial = major_radial - g.major_radius_gradient
                reduced_square = 2 * g.major_radius_m * displacement + displacement**2
                # The constant R0^2 term integrates to zero analytically; removing
                # it avoids cancellation at small r/R without changing the metric.
                volume = float(pi**2 * np.dot(weights, reduced_square * height_theta))
                derivative = float(
                    pi**2
                    * np.dot(
                        weights,
                        2 * (g.major_radius_gradient * displacement + major * displacement_radial) * height_theta
                        + reduced_square * (g.elongation + g.minor_radius_m * g.elongation_gradient_m) * np.cos(theta),
                    )
                )
                if not all(isfinite(value) and value > 0 for value in (volume, derivative)):
                    raise ValueError("Miller volume metric is not finite and positive")
                result = (volume, derivative)
                if previous is not None and all(
                    isclose(a, b, rel_tol=1e-12, abs_tol=0) for a, b in zip(result, previous, strict=True)
                ):
                    return result
                previous = result
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Miller volume arithmetic is not representable") from exc
    raise ValueError("Miller volume quadrature did not converge")
