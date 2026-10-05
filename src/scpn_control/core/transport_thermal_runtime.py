# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Crank-Nicolson thermal evolution and numerical recovery.

"""Crank-Nicolson thermal evolution and numerical recovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.momentum_transport import intrinsic_rotation_torque, nbi_torque
from scpn_control.core.pedestal import PedestalProfile
from scpn_control.core.plasma_power_terms import bremsstrahlung_power_density
from scpn_control.core.radial_diffusion import build_cn_tridiag, explicit_diffusion_rhs, thomas_solve
from scpn_control.core.runtime_sanitization import sanitize_with_fallback
from scpn_control.core.transport_state import PhysicsError, ThermalEnergyBalance

if TYPE_CHECKING:
    from scpn_control.core.integrated_transport_solver import TransportSolver

_logger = logging.getLogger(__name__)


def _sanitize_runtime_state_impl(self: TransportSolver) -> int:
    """Keep runtime profiles and coefficients finite during transport stepping."""
    recovered_total = 0

    ti_fb = np.where(np.isfinite(self.Ti), self.Ti, 1.0)
    self.Ti, n_ti = sanitize_with_fallback(self.Ti, ti_fb, floor=0.01, ceil=1e3)
    recovered_total += n_ti

    te_fb = np.where(np.isfinite(self.Te), self.Te, 1.0)
    self.Te, n_te = sanitize_with_fallback(self.Te, te_fb, floor=0.01, ceil=1e3)
    recovered_total += n_te

    ne_fb = np.where(np.isfinite(self.ne), self.ne, 5.0)
    self.ne, n_ne = sanitize_with_fallback(
        self.ne, ne_fb, floor=0.0 if self.multi_ion else 0.1, ceil=np.inf if self.multi_ion else 1e3
    )
    recovered_total += n_ne

    chi_i_fb = np.where(np.isfinite(self.chi_i), self.chi_i, 0.5)
    self.chi_i, n_chi_i = sanitize_with_fallback(self.chi_i, chi_i_fb, floor=0.01, ceil=1e4)
    recovered_total += n_chi_i

    chi_e_fb = np.where(np.isfinite(self.chi_e), self.chi_e, 0.5)
    self.chi_e, n_chi_e = sanitize_with_fallback(self.chi_e, chi_e_fb, floor=0.01, ceil=1e4)
    recovered_total += n_chi_e

    dn_fb = np.where(np.isfinite(self.D_n), self.D_n, 0.1)
    self.D_n, n_dn = sanitize_with_fallback(self.D_n, dn_fb, floor=0.0, ceil=1e4)
    recovered_total += n_dn

    imp_fb = np.where(np.isfinite(self.n_impurity), self.n_impurity, 0.0)
    self.n_impurity, n_imp = sanitize_with_fallback(self.n_impurity, imp_fb, floor=0.0, ceil=1e3)
    recovered_total += n_imp

    if self.n_D is not None:
        n_d_fb = np.where(np.isfinite(self.n_D), self.n_D, 0.5)
        self.n_D, n_d = sanitize_with_fallback(self.n_D, n_d_fb, floor=0.0, ceil=1e3)
        recovered_total += n_d
    if self.n_T is not None:
        n_t_fb = np.where(np.isfinite(self.n_T), self.n_T, 0.5)
        self.n_T, n_t = sanitize_with_fallback(self.n_T, n_t_fb, floor=0.0, ceil=1e3)
        recovered_total += n_t
    if self.n_He is not None:
        n_he_fb = np.where(np.isfinite(self.n_He), self.n_He, 0.0)
        self.n_He, n_he = sanitize_with_fallback(self.n_He, n_he_fb, floor=0.0, ceil=1e3)
        recovered_total += n_he

    return recovered_total


def _solve_temperature_channel(
    self: TransportSolver,
    old_temperature: AnyFloatArray,
    old_density: AnyFloatArray,
    new_density: AnyFloatArray,
    diffusivity: AnyFloatArray,
    net_source: AnyFloatArray,
    dt: float,
    edge_temperature: float,
) -> tuple[FloatArray, int]:
    """Advance one density-weighted CN channel with axis and edge constraints."""
    explicit_flux = explicit_diffusion_rhs(
        old_temperature,
        diffusivity * old_density / new_density,
        self.rho,
        self.drho,
        self.a,
        density=new_density,
    )
    explicit_flux, flux_recoveries = sanitize_with_fallback(explicit_flux, np.zeros_like(explicit_flux))
    rhs = old_temperature * old_density / new_density + 0.5 * dt * explicit_flux + dt * net_source
    rhs, rhs_recoveries = sanitize_with_fallback(rhs, old_temperature, floor=0.01, ceil=1e3)
    lower, diagonal, upper = build_cn_tridiag(diffusivity, dt, self.rho, self.drho, self.a, density=new_density)
    upper[0] = -1.0
    rhs[0] = 0.0
    rhs[-1] = edge_temperature
    result = thomas_solve(lower, diagonal, upper, rhs)
    result[0] = result[1]
    result[-1] = edge_temperature
    result, temperature_recoveries = sanitize_with_fallback(result, old_temperature, floor=0.01, ceil=1e3)
    return result, flux_recoveries + rhs_recoveries + temperature_recoveries


def _boundary_energy_terms(
    self: TransportSolver,
    old_ti: AnyFloatArray,
    old_te: AnyFloatArray,
    old_ni: AnyFloatArray,
    old_ne: AnyFloatArray,
    new_ni: AnyFloatArray,
    source_i: AnyFloatArray,
    source_e: AnyFloatArray,
    dt: float,
    e_kev_j: float,
) -> tuple[float, float]:
    """Account for conductive faces and prescribed edge temperatures after CN."""
    with np.errstate(invalid="ignore", over="ignore"):
        transport_volumes = self._rho_volume_element()
        faces = 0.5 * (self.rho[1:] + self.rho[:-1])
        boundary_flux = 0.0
        for old_temperature, new_temperature, diffusivity, old_density, new_density in (
            (old_ti, self.Ti, self.chi_i, old_ni, new_ni),
            (old_te, self.Te, self.chi_e, old_ne, self.ne),
        ):
            old_conductivity = old_density * diffusivity
            new_conductivity = new_density * diffusivity
            old_faces = 0.5 * (old_conductivity[1:] + old_conductivity[:-1])
            new_faces = 0.5 * (new_conductivity[1:] + new_conductivity[:-1])
            flux = (
                0.5 * faces / self.drho * (old_faces * np.diff(old_temperature) + new_faces * np.diff(new_temperature))
            )
            boundary_flux += float(flux[-1] - flux[0])
        volume_scale = transport_volumes[1] / (self.rho[1] * self.drho)
        diffusive = dt * 1.5 * 1e19 * e_kev_j * volume_scale / max(self.a**2, 1e-6) * boundary_flux
        edge_scale = 1.5 * 1e19 * e_kev_j * transport_volumes[-1]
        prescribed = edge_scale * (
            new_ni[-1] * (self.Ti[-1] - dt * source_i[-1])
            - old_ni[-1] * old_ti[-1]
            + self.ne[-1] * (self.Te[-1] - dt * source_e[-1])
            - old_ne[-1] * old_te[-1]
        )
    return float(diffusive), float(prescribed)


def _advance_momentum(self: TransportSolver, dt: float) -> None:
    """Advance optional rotation after thermal admission using the accepted state."""
    if self._momentum_solver is None:
        return
    p = self.neoclassical_params
    assert p is not None
    dr = self.drho * p["a"]
    grad_ti = np.gradient(self.Ti, dr)
    grad_ne = np.gradient(self.ne, dr)
    intrinsic = intrinsic_rotation_torque(grad_ti, grad_ne, p["R0"], p["a"])
    neutral_beam = nbi_torque(np.zeros(self.nr), p["R0"], 1e6, 0.0)
    self.omega_phi = self._momentum_solver.step(dt, self.chi_i, self.ne, self.Ti, neutral_beam, intrinsic)


def _record_energy_balance(
    self: TransportSolver,
    old_ti: AnyFloatArray,
    old_te: AnyFloatArray,
    old_ni: AnyFloatArray,
    old_ne: AnyFloatArray,
    new_ni: AnyFloatArray,
    source_i: AnyFloatArray,
    source_e: AnyFloatArray,
    dt: float,
    auxiliary_power: float,
    pumping_energy: float,
    diffusive_boundary: float,
    prescribed_edge: float,
    e_kev_j: float,
    enforce_conservation: bool,
) -> None:
    """Admit the final thermal state against the frozen-source energy ledger."""
    volumes = self._rho_volume_element()
    with np.errstate(invalid="ignore", over="ignore"):
        before = 1.5 * np.sum((old_ne * old_te + old_ni * old_ti) * 1e19 * e_kev_j * volumes)
        after = 1.5 * np.sum((self.ne * self.Te + self.ion_density * self.Ti) * 1e19 * e_kev_j * volumes)
        source = dt * 1.5 * np.sum((new_ni * source_i + self.ne * source_e) * 1e19 * e_kev_j * volumes)
        self._last_conservation_error = abs(after - before - source - diffusive_boundary - prescribed_edge) / max(
            abs(before), 1e-10
        )

    if not np.isfinite(self._last_conservation_error):
        self._last_conservation_error = float("inf")

    self._last_energy_balance = ThermalEnergyBalance(
        dt_s=float(dt),
        auxiliary_power_mw=float(auxiliary_power),
        initial_energy_j=float(before),
        final_energy_j=float(after),
        source_energy_j=float(source),
        helium_pumping_energy_j=pumping_energy,
        diffusive_boundary_energy_j=float(diffusive_boundary),
        prescribed_edge_energy_j=float(prescribed_edge),
        relative_error=float(self._last_conservation_error),
    )

    if self._last_conservation_error > 0.05:
        _logger.debug("Energy balance error: %.4e", self._last_conservation_error)
    if enforce_conservation and self._last_conservation_error > 0.01:
        raise PhysicsError(
            f"Energy conservation violated: relative error {self._last_conservation_error:.4e} > 1% threshold."
        )


def _ion_sources(
    self: TransportSolver,
    old_ti: AnyFloatArray,
    old_te: AnyFloatArray,
    ion_density: AnyFloatArray,
    line_radiation: AnyFloatArray,
    dt: float,
    auxiliary_power: float,
    e_kev_j: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Combine auxiliary heat, line radiation and thermalized ash removal."""
    heat_i, heat_e = self._compute_aux_heating_sources(auxiliary_power)
    heat_i = heat_i * self.ne / ion_density
    if self.multi_ion:
        electron_capacity = self.ne * 1e19
        radiation_i = line_radiation / (1.5 * ion_density * 1e19 * e_kev_j) * 0.5
        radiation_e = line_radiation / (1.5 * electron_capacity * e_kev_j) * 0.5
    else:
        cooling = 5.0 * self.ne * self.n_impurity * np.sqrt(self.Te + 0.1)
        radiation_i = 0.5 * cooling
        radiation_e = 0.5 * cooling

    pumping_i: FloatArray = np.zeros(self.nr)
    pumping_e: FloatArray = np.zeros(self.nr)
    pumping_energy = 0.0
    if self.multi_ion:
        with np.errstate(invalid="ignore", over="ignore"):
            pumping_i = self._last_helium_pumped * old_ti / (dt * ion_density)
            pumping_e = 2.0 * self._last_helium_pumped * old_te / (dt * self.ne)
            pumping_energy = float(
                -1.5
                * 1e19
                * e_kev_j
                * np.sum(self._last_helium_pumped * (old_ti + 2.0 * old_te) * self._rho_volume_element())
            )
    source_i = heat_i - radiation_i - pumping_i
    source_i, recoveries = sanitize_with_fallback(source_i, np.zeros_like(source_i))
    self._last_numerical_recovery_count += recoveries
    return source_i, heat_e - radiation_e, pumping_e, pumping_energy


def _electron_source_and_exchange_time(
    self: TransportSolver,
    old_te: AnyFloatArray,
    electron_heating_less_radiation: AnyFloatArray,
    electron_pumping: AnyFloatArray,
    e_kev_j: float,
) -> tuple[FloatArray, FloatArray]:
    """Apply bremsstrahlung loss and evaluate the frozen equilibration time."""
    bremsstrahlung = bremsstrahlung_power_density(self.ne, old_te, self._Z_eff)
    bremsstrahlung_rate = bremsstrahlung / (1.5 * self.ne * 1e19 * e_kev_j)
    coulomb_log = 17.0
    safe_te = np.maximum(old_te, 0.01)
    safe_ne = np.maximum(self.ne, 0.1)
    exchange_time = np.maximum(0.252 * safe_te**1.5 / (safe_ne * self._Z_eff * coulomb_log), 1e-4)
    source_e = electron_heating_less_radiation - bremsstrahlung_rate - electron_pumping
    source_e, recoveries = sanitize_with_fallback(source_e, np.zeros_like(source_e))
    self._last_numerical_recovery_count += recoveries
    return source_e, exchange_time


def _exchange_and_apply_pedestals(
    self: TransportSolver,
    ion_density: AnyFloatArray,
    exchange_time: AnyFloatArray,
    dt: float,
    electron_pedestal: PedestalProfile | None,
    ion_pedestal: PedestalProfile | None,
) -> None:
    """Exchange ion/electron energy at frozen rates, then impose edge profiles."""
    capacity_ratio = ion_density / self.ne
    relaxed_difference = -np.expm1(-(1.0 + capacity_ratio) * dt / exchange_time) * (self.Ti - self.Te)
    ion_change = relaxed_difference / (1.0 + capacity_ratio)
    self.Ti[1:-1] -= ion_change[1:-1]
    self.Te[1:-1] += capacity_ratio[1:-1] * ion_change[1:-1]
    self.Ti[0] = self.Ti[1]
    self.Te[0] = self.Te[1]

    if ion_pedestal is not None:
        rho_ped_top = ion_pedestal.p.x_ped - 2.0 * ion_pedestal.p.delta
        mask = self.rho >= rho_ped_top
        self.Ti[mask] = ion_pedestal.evaluate(self.rho[mask])
    if electron_pedestal is not None:
        rho_ped_top = electron_pedestal.p.x_ped - 2.0 * electron_pedestal.p.delta
        mask = self.rho >= rho_ped_top
        self.Te[mask] = electron_pedestal.evaluate(self.rho[mask])
    self._last_numerical_recovery_count += self._sanitize_runtime_state()


def evolve_profiles_impl(
    self: TransportSolver,
    dt: float,
    P_aux: float,
    enforce_conservation: bool = False,
    ped_te: PedestalProfile | None = None,
    ped_ti: PedestalProfile | None = None,
) -> tuple[float, float]:
    """Advance both temperatures with implicit diffusion and split exchange.

    For each channel, the conductive stage advances density-weighted storage:

        (n1*T1 - n0*T0)/dt = 0.5*[G(n0*chi, T0) + G(n1*chi, T1)] + Q/H

    Here G(k,T) = (1/a**2)/rho * d/drho(rho*k*dT/drho), with the
    supported cylindrical quadrature; n is in 10**19 m**-3, T in keV,
    chi in m**2/s, Q in W/m**3 and H = 1.5e19*e_keV_J. Subscript 0
    denotes the captured incoming state; n1 is the completed species-stage
    density and T1 is the pre-exchange CN solution. Multi-ion n_i counts
    D, T, He and impurity nuclei once each; n_e follows ion charge.
    Single-ion mode retains n_i = n_e. Completed channel densities must
    be strictly positive; old zero density contributes zero initial heat.

    chi is held fixed across this call, while conductive face weights use
    their respective old/new densities. In temperature-rate form, the RHS
    storage is (n0/n1)*T0 and Q/(H*n1) is in keV/s. Auxiliary power is
    volume-normalised on the completed densities. Radiation uses incoming
    temperatures and completed species/charge state. Helium pumping uses
    actual integrated removed counts and incoming temperatures, yielding
    step-average ion/electron heat sinks. These frozen source evaluations
    and split exchange limit combined temporal accuracy to first order.

    The pumping model removes mean thermal ion energy and that of two
    accompanying electrons per He, assuming thermalised ash and local
    quasineutral removal. It is not a velocity-selective pump, sheath,
    flowing-plasma enthalpy or fast-alpha model. No separate particle heat
    convection, pressure-work or fusion-product energy closure is supplied.
    Discrete heat conservation does not validate these missing physics.

    Internal exchange follows the conductive solve; pedestal overrides and
    sanitisation follow exchange. Boundary energy uses the old/new face
    fluxes and edge capacities. Admission describes the final thermal state
    and retains numerical recovery or unmodeled pedestal energy in its
    residual. Large timesteps may require recovery or fail admission.

    Parameters
    ----------
    dt : float
        Time step [s].
    P_aux : float
        Auxiliary heating power [MW].
    enforce_conservation : bool
        When True, raise :class:`PhysicsError` if the per-step energy
        conservation error of the final, postprocessed thermal state exceeds 1%.
    ped_te : PedestalProfile, optional
        Pedestal profile model for electron temperature.
    ped_ti : PedestalProfile, optional
        Pedestal profile model for ion temperature.
    """
    self._last_energy_balance = None
    self._last_particle_balance = None
    self._last_helium_pumped = np.zeros(self.nr)
    self._last_particle_balance_error = 0.0
    if (not np.isfinite(dt)) or dt < 0.0:
        raise ValueError(f"dt must be finite and >= 0, got {dt!r}")
    if dt == 0.0:
        return 0.0, 0.0
    if not np.isfinite(P_aux):
        raise ValueError(f"P_aux must be finite, got {P_aux!r}")

    self._last_numerical_recovery_count = self._ensure_valid_radial_grid()
    self._last_numerical_recovery_count += self._sanitize_runtime_state()
    Ti_old = self.Ti.copy()
    Te_old = self.Te.copy()
    ne_old = self.ne.copy()
    ni_old = self.ion_density
    e_keV_J = 1.602176634e-16

    # ── Multi-ion: evolve species and get radiation ──
    if self.multi_ion:
        _S_He, P_rad_line_Wm3 = self._evolve_species(dt)
    else:
        P_rad_line_Wm3 = np.zeros(self.nr)

    # ── Sources (ion/electron channels) ──
    ni = self.ion_density
    if np.any(ni <= 0.0):
        raise PhysicsError("thermal evolution requires positive ion density in every cell")
    net_source_i, electron_heating_less_radiation, pumping_e, pumping_energy = _ion_sources(
        self, Ti_old, Te_old, ni, P_rad_line_Wm3, dt, P_aux, e_keV_J
    )

    # Evolve n*T: old storage and explicit conductive flux use old density;
    # implicit flux and source conversion use the completed species density.
    # No particle-associated heat convection or reaction energy is inferred.
    # ── Ion temperature CN step ──
    self.Ti, ion_recoveries = _solve_temperature_channel(self, Ti_old, ni_old, ni, self.chi_i, net_source_i, dt, 0.1)
    self._last_numerical_recovery_count += ion_recoveries

    # ── Electron temperature (explicit channel, no Ti copy shortcut) ──
    net_source_e, tau_eq = _electron_source_and_exchange_time(
        self, Te_old, electron_heating_less_radiation, pumping_e, e_keV_J
    )

    self.Te, electron_recoveries = _solve_temperature_channel(
        self, Te_old, ne_old, self.ne, self.chi_e, net_source_e, dt, 0.08
    )
    self._last_numerical_recovery_count += electron_recoveries

    # Measure boundary power from the solved CN stage before exchange or
    # imposed pedestal profiles can alter the flux-carrying temperatures.
    dW_diffusive_boundary, dW_prescribed_edge = _boundary_energy_terms(
        self, Ti_old, Te_old, ni_old, ne_old, ni, net_source_i, net_source_e, dt, e_keV_J
    )

    # Preserve ni*Ti + ne*Te under frozen-rate exchange. tau_eq retains
    # the inherited ion-temperature relaxation convention; this is not
    # an independently calibrated species-resolved collision closure.
    _exchange_and_apply_pedestals(self, ni, tau_eq, dt, ped_te, ped_ti)

    # Admission describes the final post-exchange, post-pedestal thermal state.
    _record_energy_balance(
        self,
        Ti_old,
        Te_old,
        ni_old,
        ne_old,
        ni,
        net_source_i,
        net_source_e,
        dt,
        P_aux,
        pumping_energy,
        dW_diffusive_boundary,
        dW_prescribed_edge,
        e_keV_J,
        enforce_conservation,
    )

    # Momentum transport step (rotation profile evolution)
    _advance_momentum(self, dt)

    avg_ti: float = np.mean(self.Ti).item()
    core_ti: float = self.Ti[0].item()
    return avg_ti, core_ti
