# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Species, ion-capacity and auxiliary-source steps.

"""Species, ion-capacity and auxiliary-source steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.core.aux_heating import aux_heating_source_profiles
from scpn_control.core.species_evolution import evolve_multi_ion_species
from scpn_control.core.transport_geometry import rho_volume_element
from scpn_control.core.transport_state import PhysicsError

if TYPE_CHECKING:
    from scpn_control.core.integrated_transport_solver import TransportSolver


def _rho_volume_element_impl(self: TransportSolver) -> FloatArray:
    """Toroidal volume element per radial cell [m^3]."""
    dims = self.cfg["dimensions"]
    return rho_volume_element(self.rho, self.drho, dims["R_min"], dims["R_max"])


def ion_density_impl(self: TransportSolver) -> FloatArray:
    """Return thermal ion number density in units of 10^19 m^-3.

    Multi-ion mode counts D, T, He and tungsten nuclei once each, assuming
    they share Ti. Electron charge multiplicity does not multiply ion heat
    capacity. Single-ion mode retains the ni = ne closure. The returned
    array is independent of the mutable species state.

    Raises
    ------
    ValueError
        If a multi-ion species profile is absent.
    """
    if not self.multi_ion:
        return np.array(self.ne, dtype=np.float64, copy=True)
    if self.n_D is None or self.n_T is None or self.n_He is None:
        raise ValueError("multi-ion heat capacity requires D, T and He profiles")
    return np.asarray(self.n_D + self.n_T + self.n_He + self.n_impurity, dtype=np.float64)


def _compute_aux_heating_sources_impl(self: TransportSolver, P_aux_MW: float) -> tuple[FloatArray, FloatArray]:
    """Return ion/electron auxiliary-heating sources in keV/s.

    The source is power-normalised against the radial cell volumes to ensure
    that reconstructed injected power matches ``P_aux_MW`` by construction.
    Both outputs use electron density as reference capacity; the thermal
    caller converts the ion rate to the actual ion capacity.
    """
    if self.multi_ion and (not np.all(np.isfinite(self.ne)) or np.any(self.ne <= 0.0)):
        raise PhysicsError("auxiliary heating requires positive finite electron density")
    dV = self._rho_volume_element()
    s_heat_i, s_heat_e, balance = aux_heating_source_profiles(
        P_aux_MW,
        self.rho,
        self.ne,
        dV,
        profile_width=self.aux_heating_profile_width,
        electron_fraction=self.aux_heating_electron_fraction,
    )
    # The deposition helper uses a floored reference capacity. Convert its
    # rates to the actual positive electron capacity used by this solver.
    if self.multi_ion:
        capacity_ratio = np.maximum(self.ne, 0.1) / self.ne
        s_heat_i = s_heat_i * capacity_ratio
        s_heat_e = s_heat_e * capacity_ratio
    self._last_aux_heating_balance = balance
    return s_heat_i, s_heat_e


def _evolve_species_impl(self: TransportSolver, dt: float) -> tuple[FloatArray, FloatArray]:
    """Evolve D, T, He-ash densities for one time-step (explicit diffusion + sources).

    Uses internal sub-stepping to respect the CFL stability limit of the
    explicit diffusion scheme: dt_CFL = 0.4 * (a * drho)^2 / max(D_species).
    D_species may be a scalar or a finite nonnegative profile matching rho;
    all three species share its arithmetic face coefficients.

    Returns (S_He_source, P_rad_line):
      S_He_source — step-average He-ash production rate [10^19 m^-3 / s]
      P_rad_line  — line radiation power density from tungsten [W/m^3]
    """
    if not self.multi_ion or self.n_D is None or self.n_T is None or self.n_He is None:
        self._last_particle_balance_error = 0.0
        return np.zeros(self.nr), np.zeros(self.nr)

    # He-ash pumping time (default tau_He_factor * tau_E, floored at 0.5 s)
    tau_E = self.compute_confinement_time(1.0)  # rough estimate
    tau_He = max(self.tau_He_factor * tau_E, 0.5)

    result = evolve_multi_ion_species(
        n_D=self.n_D,
        n_T=self.n_T,
        n_He=self.n_He,
        Ti=self.Ti,
        Te=self.Te,
        n_impurity=self.n_impurity,
        dV=self._rho_volume_element(),
        rho=self.rho,
        drho=self.drho,
        a_minor=self.a,
        D_species=self.D_species,
        tau_He=tau_He,
        dt=dt,
    )
    self.n_D = result.n_D
    self.n_T = result.n_T
    self.n_He = result.n_He
    self.ne = result.ne
    self._Z_eff = result.Z_eff
    self._last_particle_balance_error = result.particle_balance_error
    self._last_particle_balance = result.particle_balance
    self._last_helium_pumped = result.helium_pumped.copy()

    return result.S_He, result.P_rad_line
