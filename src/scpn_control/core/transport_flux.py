# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Conservative signed particle and energy face transport.

"""Advance particle number and thermal energy using the same cylindrical face fluxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

_HEAT = 1.5e19 * 1.602176634e-16
_IMPURITY_CHARGE = 10.0


@dataclass(frozen=True)
class TransportFaceFlux:
    """Prescribed physical minor-radius face fluxes, positive outward.

    Attributes
    ----------
    particle_m2_s : array, shape (4, nr-1)
        Flux-surface averaged <Gamma dot grad(r)> [m^-2 s^-1], rows
        electron, D, T, He. Its integrated rate uses face dV/dr. Electron
        flux must equal D + T + 2*He on every face; impurity flux is zero.
    energy_w_m2 : array, shape (2, nr-1)
        Total transported thermal energy [W m^-2], rows electron and total
        ions. Particle-associated energy is already part of the supplied
        energy moment, not added separately. All ions share one temperature.

    Notes
    -----
    Columns lie halfway between grid nodes. The first and last faces bound
    the evolved interior, exchanging with prescribed core/edge reservoirs.
    Input arrays are read during a call; callers must not mutate them
    concurrently. No diffusion/pinch or provider-species mapping is inferred.
    """

    particle_m2_s: AnyFloatArray
    energy_w_m2: AnyFloatArray


@dataclass(frozen=True)
class TransportFaceGeometry:
    """Explicit finite-volume weights for flux averaged in physical minor radius.

    Parameters
    ----------
    cell_volume_m3 : array, shape (nr,)
        Evolved cell volumes [m^3]. Interior entries are positive. The axis
        weight is zero because its profile is copied; fixed-edge weight may
        be zero or positive and is excluded from the update.
    face_volume_derivative_m2 : array, shape (nr-1,)
        Positive dV/dr [m^2] at faces, matching the supplied radial flux
        convention. For shaped surfaces this is not their geometric area.

    Notes
    -----
    Caller owns consistency between profile nodes, face positions, enclosed
    volumes and flux coordinate. Arrays must not be mutated during a call.
    """

    cell_volume_m3: AnyFloatArray
    face_volume_derivative_m2: AnyFloatArray


@dataclass(frozen=True)
class TransportFluxBalance:
    """Immutable inventory assessment of one dedicated face-flux step.

    Attributes
    ----------
    particle_initial, particle_final, particle_boundary : tuple
        Electron,D,T,He counts; boundary contribution is positive inward.
    energy_initial_j, energy_final_j, energy_boundary_j : float
        Total ion/electron thermal energy and signed boundary energy [J].
    particle_relative_error, energy_relative_error : float
        Measured residuals, scaled by initial/final/boundary magnitudes.
    """

    particle_initial: tuple[float, ...]
    particle_final: tuple[float, ...]
    particle_boundary: tuple[float, ...]
    energy_initial_j: float
    energy_final_j: float
    energy_boundary_j: float
    particle_relative_error: float
    energy_relative_error: float


@dataclass(frozen=True)
class TransportFluxState:
    """Candidate electron,D,T,He densities, temperatures and step inventory.

    Densities are in 10^19 m^-3; temperatures are in keV. Returned arrays
    are newly allocated. Their shape/order follows the public input contract.
    """

    density: FloatArray
    temperature: FloatArray
    balance: TransportFluxBalance


def advance_face_flux(
    *,
    rho: AnyFloatArray,
    major_radius_m: float,
    minor_radius_m: float,
    density: AnyFloatArray,
    temperature: AnyFloatArray,
    impurity_density: AnyFloatArray,
    flux: TransportFaceFlux,
    dt: float,
    geometry: TransportFaceGeometry | None = None,
) -> TransportFluxState:
    """Advance the interior with prescribed radial particle and energy flux.

    Parameters
    ----------
    rho : array
        Uniform normalized grid from zero to one, at least three nodes.
    major_radius_m, minor_radius_m : float
        Positive finite physical geometry [m].
    density : array, shape (4, nr)
        Electron,D,T,He densities [10^19 m^-3], initially quasineutral.
    temperature : array, shape (2, nr)
        Electron and common-ion temperatures [keV].
    impurity_density : array, shape (nr,)
        Static nonnegative impurity nuclei density, charge fixed at ten.
    flux : TransportFaceFlux
        Explicit signed face flux, frozen across this step.
    dt : float
        Finite nonnegative time interval [s]; zero preserves input values.
    geometry : TransportFaceGeometry or None
        Explicit physical cell volumes and face dV/dr. None selects the
        circular-torus weights from major_radius_m and minor_radius_m.

    Returns
    -------
    TransportFluxState
        New state and independently measured boundary/inventory residuals.

    Raises
    ------
    ValueError
        Invalid geometry, shape, complex arrays, charge balance, nonfinite values or a step
        producing negative density/energy or zero interior thermal capacity. Nothing
        is clipped and no input is mutated. Zero cell volumes, float64 overflow
        and invalid arithmetic are refused; retry with physically valid flux
        or a smaller time interval.

    Notes
    -----
    Without explicit geometry, interior volume is 4*pi²*R*a²*rho*drho and
    face dV/dr is 4*pi²*R*a*rho_face (equal to area for circular tori).
    The update is d(n*V)/dt = F_left*(dV/dr)_left - F_right*(dV/dr)_right, likewise
    for thermal energy. The edge node is a fixed reservoir; the zero-volume
    axis copies its neighbour after positive-time steps. Reservoir exchange
    is reported, not silently called a closed boundary. This is a first-order
    frozen-flux stage without diffusion, reactions, pumping or temperature
    exchange; callers own operator composition and must avoid double counting.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            return _advance_face_flux(
                rho=rho,
                major_radius_m=major_radius_m,
                minor_radius_m=minor_radius_m,
                density=density,
                temperature=temperature,
                impurity_density=impurity_density,
                flux=flux,
                dt=dt,
                geometry=geometry,
            )
        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("Face-flux state or arithmetic is not representable") from exc


def _advance_face_flux(
    *,
    rho: AnyFloatArray,
    major_radius_m: float,
    minor_radius_m: float,
    density: AnyFloatArray,
    temperature: AnyFloatArray,
    impurity_density: AnyFloatArray,
    flux: TransportFaceFlux,
    dt: float,
    geometry: TransportFaceGeometry | None = None,
) -> TransportFluxState:
    """Compute a candidate under the public arithmetic guard without mutating any input profile."""
    if any(
        np.iscomplexobj(values)
        for values in (rho, density, temperature, impurity_density, flux.particle_m2_s, flux.energy_w_m2)
    ):
        raise ValueError("Face-flux profiles and fluxes must be real")
    rho = np.asarray(rho, dtype=np.float64)
    n = np.asarray(density, dtype=np.float64)
    t = np.asarray(temperature, dtype=np.float64)
    impurity = np.asarray(impurity_density, dtype=np.float64)
    particles = np.asarray(flux.particle_m2_s, dtype=np.float64)
    energy = np.asarray(flux.energy_w_m2, dtype=np.float64)
    if (
        rho.ndim != 1
        or rho.size < 3
        or not np.allclose(rho, np.linspace(0, 1, rho.size), rtol=0, atol=1e-12)
        or not np.isfinite(major_radius_m)
        or major_radius_m <= 0
        or not np.isfinite(minor_radius_m)
        or minor_radius_m <= 0
        or not np.isfinite(dt)
        or dt < 0
    ):
        raise ValueError("Invalid face-flux geometry or timestep")
    if (
        n.shape != (4, rho.size)
        or t.shape != (2, rho.size)
        or impurity.shape != rho.shape
        or particles.shape != (4, rho.size - 1)
        or energy.shape != (2, rho.size - 1)
        or not all(np.all(np.isfinite(x)) for x in (n, t, impurity, particles, energy))
        or np.any(n < 0)
        or np.any(t < 0)
        or np.any(impurity < 0)
    ):
        raise ValueError("Invalid face-flux state or array shape")
    charge = n[1] + n[2] + 2 * n[3] + _IMPURITY_CHARGE * impurity
    charge_flux = particles[1] + particles[2] + 2 * particles[3]
    if not np.allclose(n[0], charge, rtol=1e-12, atol=0):
        raise ValueError("Initial densities must be quasineutral")
    if not np.allclose(particles[0], charge_flux, rtol=1e-12, atol=0):
        raise ValueError("Particle face flux must be ambipolar")
    if geometry is None:
        spacing = 1.0 / (rho.size - 1)
        volume = 4 * np.pi**2 * major_radius_m * minor_radius_m**2 * rho * spacing
        area = 4 * np.pi**2 * major_radius_m * minor_radius_m * 0.5 * (rho[1:] + rho[:-1])
    else:
        raw_volume = np.asarray(geometry.cell_volume_m3)
        raw_area = np.asarray(geometry.face_volume_derivative_m2)
        if np.iscomplexobj(raw_volume) or np.iscomplexobj(raw_area):
            raise ValueError("Face-flux geometry must be real")
        volume = np.asarray(raw_volume, dtype=np.float64)
        area = np.asarray(raw_area, dtype=np.float64)
        if np.any((raw_volume != 0) & (volume == 0)) or np.any((raw_area != 0) & (area == 0)):
            raise ValueError("Face-flux geometry underflows to zero")
    if (
        volume.shape != rho.shape
        or area.shape != (rho.size - 1,)
        or not np.all(np.isfinite(volume))
        or np.any(volume < 0)
        or volume[0] != 0
        or np.any(volume[1:-1] <= 0)
        or not np.all(np.isfinite(area))
        or np.any(area <= 0)
    ):
        raise ValueError("Face-flux geometry is not representable")
    capacities = np.stack((n[0], n[1:].sum(axis=0) + impurity))
    thermal = _HEAT * capacities * t
    updated_n = n.copy()
    updated_thermal = thermal.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            updated_n[:, 1:-1] -= dt * np.diff(particles * area, axis=1) / (1e19 * volume[1:-1])
            updated_thermal[:, 1:-1] -= dt * np.diff(energy * area, axis=1) / volume[1:-1]
        except FloatingPointError as exc:
            raise ValueError("Face-flux update is not representable") from exc
    new_capacity = np.stack((updated_n[0], updated_n[1:].sum(axis=0) + impurity))
    if (
        not np.all(np.isfinite(updated_n))
        or not np.all(np.isfinite(updated_thermal))
        or np.any(updated_n < 0)
        or np.any(updated_thermal < 0)
        or np.any(new_capacity[:, 1:-1] <= 0)
    ):
        raise ValueError("Face-flux step would produce negative state or zero thermal capacity")
    updated_t = np.divide(updated_thermal, _HEAT * new_capacity, out=t.copy(), where=new_capacity > 0)
    if dt == 0:
        updated_t = t.copy()
    if dt > 0:
        updated_n[:, 0] = updated_n[:, 1]
        updated_t[:, 0] = updated_t[:, 1]
        updated_n[0, 0] = updated_n[1, 0] + updated_n[2, 0] + 2 * updated_n[3, 0] + _IMPURITY_CHARGE * impurity[0]
    final_charge = updated_n[1] + updated_n[2] + 2 * updated_n[3] + _IMPURITY_CHARGE * impurity
    if not np.allclose(updated_n[0], final_charge, rtol=1e-12, atol=0):
        raise ValueError("Final face-flux densities must be quasineutral")
    before_n = (n * volume).sum(axis=1) * 1e19
    after_n = (updated_n * volume).sum(axis=1) * 1e19
    boundary_n = dt * (particles[:, 0] * area[0] - particles[:, -1] * area[-1])
    before_e = float(np.sum(thermal * volume))
    final_capacity = np.stack((updated_n[0], updated_n[1:].sum(axis=0) + impurity))
    after_e = float(np.sum(_HEAT * final_capacity * updated_t * volume))
    boundary_e = float(dt * np.sum(energy[:, 0] * area[0] - energy[:, -1] * area[-1]))
    particle_scale = np.maximum.reduce((np.abs(before_n), np.abs(after_n), np.abs(boundary_n), np.ones(4)))
    balance = TransportFluxBalance(
        particle_initial=tuple(float(x) for x in before_n),
        particle_final=tuple(float(x) for x in after_n),
        particle_boundary=tuple(float(x) for x in boundary_n),
        energy_initial_j=before_e,
        energy_final_j=after_e,
        energy_boundary_j=boundary_e,
        particle_relative_error=float(np.max(np.abs(after_n - before_n - boundary_n) / particle_scale)),
        energy_relative_error=abs(after_e - before_e - boundary_e)
        / max(abs(before_e), abs(after_e), abs(boundary_e), 1),
    )
    if (
        not np.all(np.isfinite(updated_t))
        or not all(np.all(np.isfinite(x)) for x in (before_n, after_n, boundary_n))
        or not all(np.isfinite(x) for x in (before_e, after_e, boundary_e))
        or balance.particle_relative_error > 1e-12
        or balance.energy_relative_error > 1e-12
    ):
        raise ValueError("Face-flux inventory is nonfinite or fails conservation")
    return TransportFluxState(density=updated_n, temperature=updated_t, balance=balance)
