# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport orchestration

"""Control-facing transport/equilibrium iteration and bounded step orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_control.core.adaptive_time_controller import AdaptiveTimeController

if TYPE_CHECKING:
    from scpn_control.core.integrated_transport_solver import TransportSolver

_logger = logging.getLogger(__name__)


def _map_profiles_to_2d(self: TransportSolver) -> None:
    """Project the 1D radial profiles back onto the 2D Grad-Shafranov grid, including neoclassical bootstrap current."""
    # 1. Get Flux Topology
    idx_max = np.argmax(self.Psi)
    iz_ax, ir_ax = np.unravel_index(idx_max, self.Psi.shape)
    Psi_axis = self.Psi[iz_ax, ir_ax]
    xp, psi_x = self.find_x_point(self.Psi)
    Psi_edge = psi_x
    if abs(Psi_edge - Psi_axis) < 1.0:
        Psi_edge = float(np.min(self.Psi))

    # 2. Calculate Rho for every 2D point
    denom = Psi_edge - Psi_axis
    if abs(denom) < 1e-9:
        denom = 1e-9
    Psi_norm = (self.Psi - Psi_axis) / denom
    Psi_norm = np.clip(Psi_norm, 0, 1)
    Rho_2D = np.sqrt(Psi_norm)

    # 3. Calculate 1D Bootstrap Current
    dims = self.cfg["dimensions"]
    R0 = (dims["R_min"] + dims["R_max"]) / 2.0
    I_target = self.cfg["physics"]["plasma_current_target"]
    a_half = 0.5 * (dims["R_max"] - dims["R_min"])
    B_pol_est = (1.256e-6 * I_target) / (2 * np.pi * a_half)
    J_bs_1d = self.calculate_bootstrap_current(R0, B_pol_est)

    # 4. Interpolate 1D profiles to 2D
    self.Pressure_2D = np.interp(Rho_2D.flatten(), self.rho, self.ion_density * self.Ti + self.ne * self.Te)
    self.Pressure_2D = self.Pressure_2D.reshape(self.Psi.shape)

    J_bs_2D = np.interp(Rho_2D.flatten(), self.rho, J_bs_1d)
    J_bs_2D = J_bs_2D.reshape(self.Psi.shape)

    # 5. Update J_phi (Pressure driven + Bootstrap)
    # J_phi = R p' + J_bs
    self.J_phi = (self.Pressure_2D * self.RR) + J_bs_2D

    # Normalize to target current
    I_curr = np.sum(self.J_phi) * self.dR * self.dZ
    if I_curr > 1e-9:
        self.J_phi *= I_target / I_curr


def _compute_confinement_time(self: TransportSolver, P_loss_MW: float) -> float:
    """Compute the energy confinement time from stored energy.

    τ_E = W_stored / P_loss, where W_stored = ∫ 3/2 (ni Ti + ne Te) dV
    and the volume element is estimated from the 1D radial profiles
    using cylindrical approximation.

    Parameters
    ----------
    P_loss_MW : float
        Total loss power [MW].  Must be > 0.

    Returns
    -------
    float
        Energy confinement time [s].
    """
    if P_loss_MW <= 0:
        return float("inf")

    # Stored energy counts each ion once, irrespective of charge state.
    # In 1D with cylindrical approx: dV ≈ 2πR₀ · 2π · r · a² · dρ
    # Units: n_e is in 10^19 m^-3, T in keV → W in MJ
    e_keV = 1.602176634e-16  # J per keV
    dims = self.cfg["dimensions"]
    R0 = (dims["R_min"] + dims["R_max"]) / 2.0
    a = (dims["R_max"] - dims["R_min"]) / 2.0

    # Volume element per rho bin: dV = 2π R₀ · 2π ρ a² dρ
    rho_mid = self.rho
    dV = 2.0 * np.pi * R0 * 2.0 * np.pi * rho_mid * a**2 * self.drho

    # Temperatures are in keV and both number densities in 10^19 m^-3.
    energy_density = 1.5 * 1e19 * (self.ion_density * self.Ti + self.ne * self.Te) * e_keV
    W_stored_J = float(np.sum(energy_density * dV))
    W_stored_MW = W_stored_J / 1e6  # J → MJ → MW·s

    return W_stored_MW / P_loss_MW


def _run_self_consistent(
    self: TransportSolver,
    P_aux: float,
    n_inner: int = 100,
    n_outer: int = 10,
    dt: float = 0.01,
    psi_tol: float = 1e-3,
) -> dict[str, Any]:
    """Run self-consistent GS <-> transport iteration.

    This implements the standard integrated-modelling loop used by
    codes such as ASTRA and JINTRAC: evolve the 1D transport for
    *n_inner* steps, project profiles onto the 2D grid, re-solve the
    Grad-Shafranov equilibrium, and repeat until the poloidal-flux
    change drops below *psi_tol*.

    Algorithm
    ---------
    1. Run transport for *n_inner* steps (evolve Ti/Te/ne).
    2. Call :meth:`map_profiles_to_2d` to update ``J_phi`` on the 2D grid.
    3. Re-solve the Grad-Shafranov equilibrium with the updated source.
    4. Check psi convergence:
       ``||Psi_new - Psi_old|| / ||Psi_old|| < psi_tol``.
    5. Repeat until converged or *n_outer* iterations exhausted.

    Parameters
    ----------
    P_aux : float
        Auxiliary heating power [MW].
    n_inner : int
        Number of transport evolution steps per outer iteration.
    n_outer : int
        Maximum number of outer (GS re-solve) iterations.
    dt : float
        Transport time step [s].
    psi_tol : float
        Relative psi convergence tolerance.

    Returns
    -------
        dict
            ``{"T_avg": float, "T_core": float, "tau_e": float,
            "n_outer_converged": int, "psi_residuals": list[float],
            "Ti_profile": ndarray, "ne_profile": ndarray,
            "converged": bool}``

    Raises
    ------
    ValueError
        If either iteration count is not positive.
    """
    if n_inner <= 0 or n_outer <= 0:
        raise ValueError("n_inner and n_outer must be positive")

    psi_residuals: list[float] = []
    converged = False
    n_outer_converged = 0

    for outer in range(n_outer):
        # Save Psi before this outer iteration
        Psi_old = self.Psi.copy()
        psi_old_norm = float(np.linalg.norm(Psi_old))
        if psi_old_norm < 1e-30:
            psi_old_norm = 1.0  # avoid division by zero on first call

        # 1. Run n_inner transport steps
        for _ in range(n_inner):
            self.update_transport_model(P_aux)
            self.evolve_profiles(dt, P_aux)

        # 2. Project 1D profiles onto 2D GS grid (updates self.J_phi)
        self.map_profiles_to_2d()

        # 3. Re-solve Grad-Shafranov equilibrium
        #    external_profile_mode=True ensures solve_equilibrium uses
        #    the J_phi we just set (no internal source update).
        self.solve_equilibrium()

        # 4. Compute psi convergence metric
        psi_residual = float(np.linalg.norm(self.Psi - Psi_old) / psi_old_norm)
        psi_residuals.append(psi_residual)
        n_outer_converged = outer + 1

        _logger.info(
            "GS-transport outer iter %d/%d: psi_residual=%.4e",
            outer + 1,
            n_outer,
            psi_residual,
        )

        # 5. Convergence check
        if psi_residual < psi_tol:
            converged = True
            _logger.info(
                "GS-transport converged after %d outer iterations (residual %.4e < tol %.4e).",
                outer + 1,
                psi_residual,
                psi_tol,
            )
            break

    T_avg = float(np.mean(self.Ti))
    T_core = float(self.Ti[0])
    tau_e = self.compute_confinement_time(P_aux)

    return {
        "T_avg": T_avg,
        "T_core": T_core,
        "tau_e": tau_e,
        "n_outer_converged": n_outer_converged,
        "psi_residuals": psi_residuals,
        "Ti_profile": self.Ti.copy(),
        "ne_profile": self.ne.copy(),
        "converged": converged,
    }


def _run_to_steady_state(
    self: TransportSolver,
    P_aux: float,
    n_steps: int = 500,
    dt: float = 0.01,
    adaptive: bool = False,
    tol: float = 1e-3,
    self_consistent: bool = False,
    sc_n_inner: int = 100,
    sc_n_outer: int = 10,
    sc_psi_tol: float = 1e-3,
) -> dict[str, Any]:
    """Run transport evolution until approximate steady state.

    Parameters
    ----------
    P_aux : float
        Auxiliary heating power [MW].
    n_steps : int
        Number of evolution steps.
    dt : float
        Time step [s] (initial value when adaptive=True).
    adaptive : bool
        Use Richardson-extrapolation adaptive time stepping.
    tol : float
        Error tolerance for adaptive stepping.
    self_consistent : bool
        When True, delegate to :meth:`run_self_consistent` which
        iterates GS <-> transport to convergence.  The remaining
        ``sc_*`` parameters are forwarded.
    sc_n_inner : int
        Transport steps per outer GS iteration (self-consistent mode).
    sc_n_outer : int
        Maximum outer GS iterations (self-consistent mode).
    sc_psi_tol : float
        Relative psi convergence tolerance (self-consistent mode).

    Returns
    -------
    dict
        ``{"T_avg": float, "T_core": float, "tau_e": float,
        "n_steps": int, "Ti_profile": ndarray,
        "ne_profile": ndarray}``
        When adaptive=True, also includes ``dt_final``,
        ``dt_history``, ``error_history``.
        When self_consistent=True, returns the
        :meth:`run_self_consistent` dict instead.

    Raises
    ------
    ValueError
        If the selected mode has a nonpositive iteration count.
    """
    # ── Self-consistent GS↔transport mode ──
    if self_consistent:
        return self.run_self_consistent(
            P_aux=P_aux,
            n_inner=sc_n_inner,
            n_outer=sc_n_outer,
            dt=dt,
            psi_tol=sc_psi_tol,
        )

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    if not adaptive:
        for _ in range(n_steps):
            self.update_transport_model(P_aux)
            T_avg, T_core = self.evolve_profiles(dt, P_aux)

        tau_e = self.compute_confinement_time(P_aux)
        return {
            "T_avg": float(T_avg),
            "T_core": float(T_core),
            "tau_e": tau_e,
            "n_steps": n_steps,
            "Ti_profile": self.Ti.copy(),
            "ne_profile": self.ne.copy(),
        }

    # ── Adaptive time stepping ──
    atc = AdaptiveTimeController(dt_init=dt, tol=tol)

    for _step in range(n_steps):
        self.update_transport_model(P_aux)
        error = atc.estimate_error(self, P_aux)
        atc.adapt_dt(error)

        # Take the accepted step (full step already applied inside estimate_error)
        T_avg = float(np.mean(self.Ti))
        T_core = float(self.Ti[0])

    tau_e = self.compute_confinement_time(P_aux)
    return {
        "T_avg": float(T_avg),
        "T_core": float(T_core),
        "tau_e": tau_e,
        "n_steps": n_steps,
        "Ti_profile": self.Ti.copy(),
        "ne_profile": self.ne.copy(),
        "dt_final": atc.dt,
        "dt_history": atc.dt_history.copy(),
        "error_history": atc.error_history.copy(),
    }
