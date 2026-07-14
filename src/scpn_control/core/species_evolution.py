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

__all__ = ["SpeciesEvolutionResult", "evolve_multi_ion_species"]

# Tungsten mean charge state — Pütterich et al., Nucl. Fusion 50, 025012 (2010);
# ITER edge conditions (Te ~ 1-5 keV) give Z_W ~ 8-20, 10 is a representative mid-range.
_Z_W = 10.0


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
        Volume-mean effective charge, clipped to [1, 10].
    particle_balance_error : float
        Relative particle-inventory conservation error for this step.
    S_He : array
        Helium-ash production rate [10^19 m^-3 / s].
    P_rad_line : array
        Tungsten line-radiation power density [W/m^3].
    """

    n_D: FloatArray
    n_T: FloatArray
    n_He: FloatArray
    ne: FloatArray
    Z_eff: float
    particle_balance_error: float
    S_He: FloatArray
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
    drho: float,
    D_species: float,
    tau_He: float,
    dt: float,
) -> SpeciesEvolutionResult:
    """Advance the D/T/He-ash densities by one time-step (explicit diffusion + sources).

    The fusion, fuel-consumption, and helium-pumping source terms are evaluated once
    from the incoming densities and held constant across the CFL sub-steps
    (``dt_CFL = 0.4 drho^2 / D_species``). Deuterium/tritium are consumed by burn with
    an edge recycling floor and axis Neumann condition; helium is produced by burn and
    removed by pumping with time ``tau_He``. The electron density follows from
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
        Toroidal volume element per radial cell [m^3].
    drho : float
        Normalised radial grid spacing.
    D_species : float
        Anomalous particle diffusivity [m^2/s].
    tau_He : float
        Helium-ash pumping time [s].
    dt : float
        Time-step [s].

    Returns
    -------
    SpeciesEvolutionResult
        The updated densities and per-step diagnostics.
    """
    n_D = np.asarray(n_D, dtype=np.float64)
    n_T = np.asarray(n_T, dtype=np.float64)
    n_He = np.asarray(n_He, dtype=np.float64)
    Ti = np.asarray(Ti, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)
    n_impurity = np.asarray(n_impurity, dtype=np.float64)
    dV = np.asarray(dV, dtype=np.float64)

    # Particle inventory before evolution (10^19 m^-3 units × volume)
    N_before = float(np.sum((n_D + n_T + n_He) * 1e19 * dV))

    # Fusion source: S_fus = n_D * n_T * <sigma_v> (reactions per m^3 per s)
    # n_D, n_T are in 10^19 m^-3 => multiply by (1e19)^2
    sigmav = bosch_hale_dt_reactivity(Ti)
    S_fus = (n_D * 1e19) * (n_T * 1e19) * sigmav  # reactions/m^3/s

    # He-ash source (in 10^19 m^-3 / s) — one He per fusion reaction
    S_He = S_fus / 1e19

    # He-ash sink: pumping with tau_He
    S_He_pump = n_He / tau_He

    # D and T consumption rate (in 10^19 / s)
    S_fuel = S_fus / 1e19

    # Integrated source: D+T consumed = -2·S_fus, He produced = +S_fus, He pumped = -S_He_pump
    # Net = -S_fus - S_He_pump (in 10^19/s units, per cell)
    dN_source_expected = float(dt * np.sum((-S_fuel - S_fuel + S_He - S_He_pump) * 1e19 * dV))

    # Diffusion operator (explicit, simple Laplacian)
    def _diffuse(n: AnyFloatArray) -> FloatArray:
        d2n = np.zeros_like(n, dtype=float)
        d2n[1:-1] = (n[2:] - 2.0 * n[1:-1] + n[:-2]) / (drho**2)
        return np.asarray(D_species * d2n)

    # CFL sub-stepping for explicit diffusion stability
    dt_cfl = 0.4 * drho**2 / max(D_species, 1e-10)
    n_sub = max(1, int(np.ceil(dt / dt_cfl)))
    dt_sub = dt / n_sub

    for _ in range(n_sub):
        # Evolve D
        new_D = n_D + dt_sub * (_diffuse(n_D) - S_fuel)
        new_D[0] = new_D[1]  # Neumann at axis
        new_D[-1] = 0.01  # edge recycling floor
        n_D = np.maximum(0.001, new_D)

        # Evolve T
        new_T = n_T + dt_sub * (_diffuse(n_T) - S_fuel)
        new_T[0] = new_T[1]
        new_T[-1] = 0.01
        n_T = np.maximum(0.001, new_T)

        # Evolve He-ash
        new_He = n_He + dt_sub * (_diffuse(n_He) + S_He - S_He_pump)
        new_He[0] = new_He[1]
        new_He[-1] = 0.0
        n_He = np.maximum(0.0, new_He)

    # Particle balance diagnostic
    N_after = float(np.sum((n_D + n_T + n_He) * 1e19 * dV))
    dN_actual = N_after - N_before
    # Relative error (includes boundary flux + clipping residuals)
    particle_balance_error = abs(dN_actual - dN_source_expected) / max(abs(N_before), 1e-10)

    # Recompute ne from quasineutrality: ne = n_D + n_T + 2*n_He + Z_W*n_imp
    ne: FloatArray = n_D + n_T + 2.0 * n_He + _Z_W * np.maximum(n_impurity, 0.0)
    ne = np.maximum(ne, 0.1)

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
        S_He=S_He,
        P_rad_line=P_rad_line,
    )
