# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physical units for signed TGLF particle and energy flux.

"""Convert GYRO-normalized TGLF particle/energy flux to physical minor-radius flux."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from scpn_control.core.tglf_flux import TGLFFluxResult

_ELEMENTARY_CHARGE = 1.602176634e-19
_KEV_J = 1e3 * _ELEMENTARY_CHARGE


@dataclass(frozen=True)
class TGLFReferenceUnits:
    """Explicit reference units matching the GACODE input normalization.

    Parameters
    ----------
    density_m3 : float
        Reference density [m^-3], typically electron density in GYRO units.
    temperature_kev : float
        Reference temperature [keV], typically electron temperature.
    length_m : float
        Reference length [m], the minor-radius normalization used for gradients.
    mass_kg : float
        Reference ion mass [kg], not the electron mass.
    magnetic_field_t : float
        Positive GACODE Bunit [T]. It must not be silently equated with toroidal
        field for shaped geometry.

    Notes
    -----
    All references must be finite and positive. Supplying them is a caller
    assertion of agreement with the deck; this object cannot infer or certify
    that agreement. GENE-normalized outputs require their own conversion.
    """

    density_m3: float
    temperature_kev: float
    length_m: float
    mass_kg: float
    magnetic_field_t: float

    def __post_init__(self) -> None:
        """Reject invalid physical scales before normalization arithmetic."""
        for name in ("density_m3", "temperature_kev", "length_m", "mass_kg", "magnetic_field_t"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TGLFPhysicalFlux:
    """Signed species flux across surfaces of physical minor radius r.

    Attributes
    ----------
    particle_m2_s : tuple
        Flux-surface averaged radial particle moment <Gamma dot grad(r)>
        [m^-2 s^-1], electron first then ions. Its integrated rate uses dV/dr,
        which is generally different from geometric area on shaped surfaces.
    energy_w_m2 : tuple
        Provider energy moment flux [W m^-2] in the same species order.
        This is not asserted to be conductive heat alone; particle-associated
        energy must not be added again without a specified moment convention.

    Notes
    -----
    Positive means outward in r; inward values remain negative. For a different
    radial coordinate x, the contravariant flux is F_x = (dx/dr) F_r and must
    use the matching volume Jacobian. No coordinate projection is inferred.
    Momentum/exchange moments remain in the raw result; their unit contracts
    are not supplied by this particle/energy conversion.
    """

    particle_m2_s: tuple[float, ...]
    energy_w_m2: tuple[float, ...]


def physical_tglf_flux(raw: TGLFFluxResult, reference: TGLFReferenceUnits) -> TGLFPhysicalFlux:
    """Convert signed GYRO-normalized particle and energy moments to SI.

    Parameters
    ----------
    raw : TGLFFluxResult
        Validated raw output using the stated GYRO reference convention.
    reference : TGLFReferenceUnits
        Explicit physical reference scales matching that output.

    Returns
    -------
    TGLFPhysicalFlux
        Physical minor-radius flux, preserving species order and signs.

    Raises
    ------
    ValueError
        Scales, output sizes or converted fluxes are not representable as
        finite working values, or a nonzero moment rounds to zero. Exact zero
        and representable subnormal results are retained; no clipping or epsilon.

    Notes
    -----
    With T0 in joules, cs = sqrt(T0/m0), rho_s = sqrt(T0*m0)/(e*Bunit),
    Gamma_GB = n0*cs*(rho_s/a0)^2 and Q_GB = T0*Gamma_GB.
    These are the reference factors in GACODE tglf_TM_driver.f90; its extra
    drhodr factor projects F_r into its chosen flux coordinate. This function
    returns F_r, so no such factor is included. It neither divides by a profile
    gradient nor identifies a diffusion/pinch decomposition.
    """
    try:
        temperature_j = reference.temperature_kev * _KEV_J
        sound_speed = sqrt(temperature_j / reference.mass_kg)
        rho_s = sqrt(temperature_j * reference.mass_kg) / (_ELEMENTARY_CHARGE * reference.magnetic_field_t)
        particle_scale = reference.density_m3 * sound_speed * (rho_s / reference.length_m) ** 2
        energy_scale = temperature_j * particle_scale
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("TGLF reference flux scales are not representable") from exc
    if not all(isfinite(value) and value > 0 for value in (temperature_j, particle_scale, energy_scale)):
        raise ValueError("TGLF reference flux scales are not finite and positive")
    if len(raw.particle_flux_gb) < 2 or len(raw.particle_flux_gb) != len(raw.energy_flux_gb):
        raise ValueError("TGLF particle and energy flux must have the same species")
    particle = tuple(float(x * particle_scale) for x in raw.particle_flux_gb)
    energy = tuple(float(x * energy_scale) for x in raw.energy_flux_gb)
    if not all(isfinite(value) for value in (*particle, *energy)):
        raise ValueError("TGLF physical flux is not finite")
    if any(
        source != 0 and converted == 0
        for source, converted in zip((*raw.particle_flux_gb, *raw.energy_flux_gb), (*particle, *energy), strict=True)
    ):
        raise ValueError("TGLF nonzero physical flux underflows to zero")
    return TGLFPhysicalFlux(particle_m2_s=particle, energy_w_m2=energy)
