# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-Ion Species Evolution

"""Multi-ion (D/T/He-ash) species evolution kernel for the transport solver.

Stateless helper extracted from the integrated transport solver: one explicit
time-step of the deuterium/tritium/helium-ash densities under fusion burn, an
explicit diffusion operator with CFL sub-stepping, and helium-ash pumping, then
the derived electron density (quasineutrality), effective charge, and tungsten
line radiation. All state (species densities, temperatures, impurity density,
grid, and the pumping time) is passed explicitly and the mutated densities plus
diagnostics are returned for the caller to store.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.plasma_power_terms import bosch_hale_dt_reactivity, tungsten_radiation_rate

__all__ = ["SpeciesEvolutionResult", "SpeciesParticleBalance", "evolve_multi_ion_species"]

# Tungsten mean charge state — Pütterich et al., Nucl. Fusion 50, 025012 (2010);
# ITER edge conditions (Te ~ 1-5 keV) give Z_W ~ 8-20, 10 is a representative mid-range.
_Z_W = 10.0


@dataclass(frozen=True)
class SpeciesParticleBalance:
    """Immutable D/T/He inventory accounting for one completed species stage.

    Attributes
    ----------
    initial_count, final_count : float
        Total D, T and He nuclei on the supplied toroidal quadrature.
    fusion_count : float
        Integrated D-T reactions; each reduces the D+T+He count by one.
    pumped_count : float
        Helium nuclei removed by pumping.
    diffusive_boundary_count : float
        Signed nuclei entering interior cells through the two bounding faces.
    prescribed_boundary_count : float
        Signed nuclei supplied by resetting the imposed edge densities.
    numerical_correction_count : float
        Nuclei introduced by nonnegative clipping during diffusion. This is
        reported separately and remains an error, not a physical source.
    relative_error : float
        Absolute unexplained inventory change divided by initial count,
        floored at 1e-10 in the denominator. It includes numerical corrections.

    Notes
    -----
    Counts are dimensionless numbers of nuclei, not density units. This is a
    discrete species-stage record, not a thermal-energy or facility certificate.
    """

    initial_count: float
    final_count: float
    fusion_count: float
    pumped_count: float
    diffusive_boundary_count: float
    prescribed_boundary_count: float
    numerical_correction_count: float
    relative_error: float


@dataclass(frozen=True)
class SpeciesEvolutionResult:
    """Outputs of one multi-ion species evolution step.

    Attributes
    ----------
    n_D, n_T, n_He : array
        Updated deuterium, tritium, and helium-ash densities [10^19 m^-3].
    ne : array
        Electron density from quasineutrality [10^19 m^-3].
    Z_eff : float
        Arithmetic radial-mean effective charge, clipped to [1, 10].
    particle_balance_error : float
        Relative unexplained particle-inventory change for this step.
    particle_balance : SpeciesParticleBalance
        Frozen scalar accounting of sources, boundaries and numerical correction.
    S_He : array
        Step-average helium production rate [10^19 m^-3 / s], zero for dt=0.
    fusion_reactions : array
        Integrated reactions per cell volume [10^19 m^-3]; each consumes one D
        and one T and produces one He before boundary conditions.
    helium_pumped : array
        Integrated He removed by pumping [10^19 m^-3], before boundary reset.
    P_rad_line : array
        Tungsten line-radiation power density [W/m^3].
    """

    n_D: FloatArray
    n_T: FloatArray
    n_He: FloatArray
    ne: FloatArray
    Z_eff: float
    particle_balance_error: float
    particle_balance: SpeciesParticleBalance
    S_He: FloatArray
    fusion_reactions: FloatArray
    helium_pumped: FloatArray
    P_rad_line: FloatArray


def evolve_multi_ion_species(
    *,
    n_D: AnyFloatArray,
    n_T: AnyFloatArray,
    n_He: AnyFloatArray,
    Ti: AnyFloatArray,
    Te: AnyFloatArray,
    n_impurity: AnyFloatArray,
    dV: AnyFloatArray,
    rho: AnyFloatArray,
    drho: float,
    a_minor: float,
    D_species: float | AnyFloatArray,
    tau_He: float,
    dt: float,
) -> SpeciesEvolutionResult:
    """Advance the D/T/He-ash densities by one time-step (explicit diffusion + sources).

    Each CFL substep applies cylindrical diffusion, exact D-T burn at frozen
    temperature, then exponential helium pumping. Burn consumes equal D and T
    counts and creates exactly that many He nuclei. The diffusion stability step
    is ``0.4 (a_minor drho)^2 / max(D_species)``. Deuterium/tritium have a prescribed
    0.01 edge recycling density; helium has a zero edge density. The axis uses
    a Neumann condition. The electron density follows from
    quasineutrality (D + T + 2·He + Z_W·impurity), and the effective charge and
    tungsten line radiation are derived from the updated state.

    Parameters
    ----------
    n_D, n_T, n_He : array
        Deuterium, tritium, and helium-ash densities [10^19 m^-3].
    Ti, Te : array
        Ion and electron temperatures [keV].
    n_impurity : array
        Tungsten impurity density [10^19 m^-3].
    dV : array
        Toroidal volume element per radial cell [m^3], proportional to rho
        with zero axis weight and positive interior/edge weights.
    rho : array
        Uniform normalized radial grid from zero to one, matching the densities.
    a_minor : float
        Finite positive physical minor radius [m]; required explicitly.
    drho : float
        Normalised radial grid spacing.
    D_species : float or array
        Finite nonnegative anomalous particle diffusivity [m^2/s], shared by
        D/T/He, representable as finite float64. A vector must match rho; scalar
        values broadcast over the grid.
        Arithmetic face averages enter the conservative flux. The maximum
        cell diffusivity bounds the physical CFL step; all-zero diffusion
        uses one source substep. Inputs are not mutated.
    tau_He : float
        Positive helium-ash pumping time [s]; infinity disables pumping.
    dt : float
        Time-step [s].

    Returns
    -------
    SpeciesEvolutionResult
        Updated densities and source/boundary accounting. Numerical clipping
        remains visible in the relative inventory error.

    Raises
    ------
    ValueError
        If radial geometry, diffusivity or timestep is invalid.

    Notes
    -----
    Diffusion uses (1/a^2)/rho d/drho(rho D(rho) dn/drho). The diffusion, burn and
    pumping operators are split in that order; coupled evolution is first-order
    in time even though each local source operator is exact. Reactivity is held
    at incoming Ti and its physical calibration is unchanged. S_He now reports
    integrated production divided by dt, not the incoming instantaneous rate.
    A zero timestep leaves all three species profiles unchanged.
    Callers of the former geometry-free API must now provide rho and a_minor.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if (
        rho.ndim != 1
        or rho.size < 3
        or rho.shape != np.shape(n_D)
        or not np.isfinite(drho)
        or drho <= 0.0
        or not np.allclose(rho, np.linspace(0.0, 1.0, rho.size), rtol=0.0, atol=1e-12)
        or not np.isclose(drho, 1.0 / (rho.size - 1), rtol=1e-12, atol=0.0)
    ):
        raise ValueError("rho and drho must describe a uniform normalized radial grid")
    if not np.isfinite(a_minor) or a_minor <= 0.0:
        raise ValueError("a_minor must be finite and positive")
    raw_diffusivity = np.asarray(D_species)
    if (
        raw_diffusivity.dtype.kind not in "fiu"
        or raw_diffusivity.shape not in ((), rho.shape)
        or not np.all(np.isfinite(raw_diffusivity))
        or np.any(raw_diffusivity < 0.0)
        or np.any(raw_diffusivity > np.finfo(np.float64).max)
    ):
        raise ValueError("D_species must be a finite nonnegative real scalar or vector matching rho")
    diffusivity = np.asarray(np.broadcast_to(raw_diffusivity, rho.shape), dtype=np.float64)
    face_diffusivity = 0.5 * diffusivity[:-1] + 0.5 * diffusivity[1:]
    max_diffusivity = float(np.max(diffusivity))
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and nonnegative")
    if np.isnan(tau_He) or tau_He <= 0.0:
        raise ValueError("tau_He must be positive, or infinity to disable pumping")
    for name, values in (
        ("n_D", n_D),
        ("n_T", n_T),
        ("n_He", n_He),
        ("Ti", Ti),
        ("Te", Te),
        ("n_impurity", n_impurity),
        ("dV", dV),
    ):
        raw = np.asarray(values)
        if raw.shape != rho.shape or raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError(f"{name} must be a finite nonnegative vector matching rho")
    n_D = np.asarray(n_D, dtype=np.float64).copy()
    n_T = np.asarray(n_T, dtype=np.float64).copy()
    n_He = np.asarray(n_He, dtype=np.float64).copy()
    Ti = np.asarray(Ti, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)
    n_impurity = np.asarray(n_impurity, dtype=np.float64)
    dV = np.asarray(dV, dtype=np.float64)

    volume_scale = float(dV[1] / (rho[1] * drho))
    if (
        not np.isfinite(volume_scale)
        or volume_scale <= 0.0
        or not np.allclose(dV, volume_scale * rho * drho, rtol=1e-12, atol=0.0)
    ):
        raise ValueError("dV must be a positive cylindrical volume scale times rho * drho")

    # Particle inventory before evolution (10^19 m^-3 units × volume)
    N_before = float(np.sum((n_D + n_T + n_He) * 1e19 * dV))

    reactivity = bosch_hale_dt_reactivity(Ti) * 1e19
    fusion_reactions = np.zeros_like(n_D)
    helium_pumped = np.zeros_like(n_He)

    faces = 0.5 * (rho[1:] + rho[:-1])
    diffusive_boundary_count = 0.0
    prescribed_boundary_count = 0.0
    numerical_correction_count = 0.0

    def _diffuse(n: AnyFloatArray) -> FloatArray:
        """Advance a frozen-diffusivity substep and account for boundary exchange and positivity corrections."""
        nonlocal diffusive_boundary_count, numerical_correction_count
        flux = face_diffusivity * faces * np.diff(n) / drho
        rate = np.zeros_like(n, dtype=np.float64)
        rate[1:-1] = np.diff(flux) / (a_minor**2 * rho[1:-1] * drho)
        diffusive_boundary_count += dt_sub * 1e19 * volume_scale / a_minor**2 * float(flux[-1] - flux[0])
        candidate = n + dt_sub * rate
        updated = np.maximum(0.0, candidate)
        numerical_correction_count += float(np.sum((updated - candidate) * 1e19 * dV))
        return np.asarray(updated, dtype=np.float64)

    n_sub = 1
    if max_diffusivity > 0.0:
        dt_cfl = 0.4 * (a_minor * drho) ** 2 / max_diffusivity
        n_sub = max(1, int(np.ceil(dt / dt_cfl)))
    dt_sub = dt / n_sub

    for _ in range(n_sub if dt > 0.0 else 0):
        n_D = _diffuse(n_D)
        n_T = _diffuse(n_T)
        n_He = _diffuse(n_He)

        # D-T is invariant under burn. Integrating dL/dt=-k L(L+delta)
        # gives this bounded consumption, with the equal-fuel limit explicit.
        lower = np.minimum(n_D, n_T)
        higher = np.maximum(n_D, n_T)
        delta = higher - lower
        exposure = reactivity * dt_sub
        reacted_fraction = -np.expm1(-delta * exposure)
        equal_burn = lower * (lower * exposure / (1.0 + lower * exposure))
        burn = np.divide(
            lower * higher * reacted_fraction,
            delta + lower * reacted_fraction,
            out=equal_burn,
            where=delta > 0.0,
        )
        burn = np.minimum(lower, burn)
        lower_after = np.divide(
            lower * delta * np.exp(-delta * exposure),
            delta + lower * reacted_fraction,
            out=lower / (1.0 + lower * exposure),
            where=delta > 0.0,
        )
        deuterium_is_lower = n_D <= n_T
        n_D = np.where(deuterium_is_lower, lower_after, lower_after + delta)
        n_T = np.where(deuterium_is_lower, lower_after + delta, lower_after)
        n_He += burn
        fusion_reactions += burn

        pumped = n_He * (-np.expm1(-dt_sub / tau_He))
        n_He = n_He * np.exp(-dt_sub / tau_He)
        helium_pumped += pumped

        prescribed_boundary_count += 1e19 * dV[-1] * (0.02 - n_D[-1] - n_T[-1] - n_He[-1])
        n_D[0], n_T[0], n_He[0] = n_D[1], n_T[1], n_He[1]
        n_D[-1], n_T[-1], n_He[-1] = 0.01, 0.01, 0.0

    S_He = fusion_reactions / dt if dt > 0.0 else np.zeros_like(n_D)
    fusion_count = float(np.sum(fusion_reactions * 1e19 * dV))
    pumped_count = float(np.sum(helium_pumped * 1e19 * dV))
    N_after = float(np.sum((n_D + n_T + n_He) * 1e19 * dV))
    unexplained = (
        N_after - N_before + fusion_count + pumped_count - diffusive_boundary_count - prescribed_boundary_count
    )
    particle_balance_error = abs(unexplained) / max(abs(N_before), 1e-10)
    particle_balance = SpeciesParticleBalance(
        initial_count=N_before,
        final_count=N_after,
        fusion_count=fusion_count,
        pumped_count=pumped_count,
        diffusive_boundary_count=diffusive_boundary_count,
        prescribed_boundary_count=prescribed_boundary_count,
        numerical_correction_count=numerical_correction_count,
        relative_error=particle_balance_error,
    )

    # Recompute ne from quasineutrality: ne = n_D + n_T + 2*n_He + Z_W*n_imp
    ne: FloatArray = n_D + n_T + 2.0 * n_He + _Z_W * np.maximum(n_impurity, 0.0)

    # Z_eff
    ne_m3 = ne * 1e19
    ne_safe = np.maximum(ne_m3, 1e10)
    sum_nZ2 = n_D * 1e19 * 1.0 + n_T * 1e19 * 1.0 + n_He * 1e19 * 4.0 + np.maximum(n_impurity, 0.0) * 1e19 * _Z_W**2
    Z_eff = float(np.clip(np.mean(sum_nZ2 / ne_safe), 1.0, 10.0))

    # Tungsten line radiation [W/m^3]
    Lz = tungsten_radiation_rate(Te)
    n_W_m3 = np.maximum(n_impurity, 0.0) * 1e19
    P_rad_line: FloatArray = ne_m3 * n_W_m3 * Lz  # W/m^3

    return SpeciesEvolutionResult(
        n_D=np.asarray(n_D, dtype=np.float64),
        n_T=np.asarray(n_T, dtype=np.float64),
        n_He=np.asarray(n_He, dtype=np.float64),
        ne=ne,
        Z_eff=Z_eff,
        particle_balance_error=particle_balance_error,
        particle_balance=particle_balance,
        S_He=np.asarray(S_He, dtype=np.float64),
        fusion_reactions=fusion_reactions,
        helium_pumped=helium_pumped,
        P_rad_line=P_rad_line,
    )
