# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Anomalous Transport Coefficient Models

"""Anomalous (turbulent) transport coefficient models for the transport solver.

Stateless helpers extracted from the integrated transport solver:

* :func:`gyro_bohm_chi_profile` — the gyro-Bohm anomalous thermal-diffusivity
  scaling ``chi_gB = c_gB * rho_s^2 * c_s / (a q R)``, evaluated cell by cell.
* :func:`gk_flux_surface_transport` — the shared per-flux-surface driver that
  runs a gyrokinetic solver (external TGLF or the native TGLF-equivalent) on
  every radial cell, validates the returned fluxes, and either fails closed or
  falls back to gyro-Bohm on unconverged output.

The radial grid and plasma profiles are passed explicitly so the models are
independent of any solver instance state; the caller assigns the returned
electron and particle diffusivities back onto its own attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase

__all__ = [
    "gk_flux_surface_transport",
    "gyro_bohm_chi_profile",
]


def gyro_bohm_chi_profile(
    rho: AnyFloatArray,
    Ti: AnyFloatArray,
    Te: AnyFloatArray,
    q: AnyFloatArray,
    R0: float,
    a: float,
    B0: float,
    A_ion: float,
    c_gB: float,
) -> FloatArray:
    """Gyro-Bohm anomalous transport diffusivity profile [m^2/s].

    ``chi_gB = c_gB * rho_s^2 * c_s / (a q R)`` where ``rho_s = sqrt(T_i m_i) /
    (e B)`` is the ion sound gyroradius and ``c_s = sqrt(T_e / m_i)`` the ion
    sound speed. Temperatures are floored at 0.01 keV and the safety factor at
    0.5 so a degenerate edge cell cannot produce a non-finite diffusivity; every
    cell is floored at 0.01 m^2/s.

    Parameters
    ----------
    rho : array
        Normalised radius [0, 1].
    Ti : array
        Ion temperature [keV].
    Te : array
        Electron temperature [keV].
    q : array
        Safety factor profile.
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    B0 : float
        Toroidal field [T].
    A_ion : float
        Ion mass number (2 = deuterium).
    c_gB : float
        Calibrated gyro-Bohm coefficient.

    Returns
    -------
    chi_gB : array
        Gyro-Bohm ion thermal diffusivity [m^2/s].
    """
    rho = np.asarray(rho, dtype=np.float64)
    Ti = np.asarray(Ti, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    e_charge = 1.602176634e-19
    m_i = A_ion * 1.672621924e-27

    chi_gB: FloatArray = np.zeros(len(rho))
    for i in range(len(rho)):
        Ti_keV = max(Ti[i], 0.01)
        Te_keV = max(Te[i], 0.01)
        qi = max(q[i], 0.5)

        T_i_J = Ti_keV * 1e3 * e_charge
        T_e_J = Te_keV * 1e3 * e_charge

        rho_s = np.sqrt(T_i_J * m_i) / (e_charge * B0)
        c_s = np.sqrt(T_e_J / m_i)

        chi_val = c_gB * rho_s**2 * c_s / max(a * qi * R0, 1e-6)
        chi_gB[i] = max(chi_val, 0.01) if np.isfinite(chi_val) else 0.01

    return chi_gB


def gk_flux_surface_transport(
    *,
    solver: GKSolverBase,
    rho: AnyFloatArray,
    Te: AnyFloatArray,
    Ti: AnyFloatArray,
    ne: AnyFloatArray,
    params: dict[str, Any],
    solver_label: str,
    catch_execution_errors: bool,
    allow_gyrobohm_fallback: bool,
    gyro_bohm_fallback: Callable[[], FloatArray],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Run a gyrokinetic solver on every flux surface and return (chi_i, chi_e, D).

    Shared driver for the ``external_gk`` and ``tglf_native`` transport models.
    On each interior cell it assembles the local gradients (``R/L_T``, ``R/L_n``,
    magnetic shear) into a :class:`GKLocalParams`, runs *solver*, and accepts the
    result only when it converged with finite non-negative fluxes. Otherwise it
    fails closed unless *allow_gyrobohm_fallback* is set, in which case the cell
    falls back to the gyro-Bohm estimate. The plasma core edge (rho <= 0.05) and
    vacuum cells (non-finite or vanishing density) are floored to 0.01.

    Parameters
    ----------
    solver : GKSolverBase
        Gyrokinetic solver exposing ``run_from_params``.
    rho, Te, Ti, ne : array
        Normalised radius and the electron/ion temperature [keV] and density
        [10^19 m^-3] profiles.
    params : dict
        Geometry and shaping parameters (``R0``, ``a``, ``B0``, ``q_profile``,
        and optional ``Z_eff``, ``kappa``, ``delta``).
    solver_label : str
        Name used in the fail-closed error messages ("external_gk" /
        "tglf_native").
    catch_execution_errors : bool
        When True (external solvers), a raised solver exception is caught and
        either re-raised as a fail-closed error or absorbed by the fallback.
    allow_gyrobohm_fallback : bool
        When True, unconverged/failed cells fall back to *gyro_bohm_fallback*
        instead of raising.
    gyro_bohm_fallback : callable
        Zero-argument callable returning the gyro-Bohm chi profile, indexed per
        cell only when a fallback is taken.

    Returns
    -------
    chi_i, chi_e, D_n : array
        Ion and electron thermal diffusivities and the particle diffusivity
        [m^2/s]; the caller assigns *chi_e* and *D_n* back onto its state.
    """
    rho = np.asarray(rho, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)
    Ti = np.asarray(Ti, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)

    R0 = params["R0"]
    a = params["a"]
    B0 = params["B0"]
    q_prof = params["q_profile"]
    Z_eff = params.get("Z_eff", 1.5)
    kappa = params.get("kappa", 1.0)
    delta = params.get("delta", 0.0)

    dTe_dr = np.gradient(Te, rho * a)
    dTi_dr = np.gradient(Ti, rho * a)
    dne_dr = np.gradient(ne, rho * a)

    # Magnetic shear: s = (rho/q) * dq/drho
    dq_drho = np.gradient(q_prof, rho)
    s_hat_profile = rho * dq_drho / np.maximum(q_prof, 0.5)

    chi_i_out: FloatArray = np.zeros(len(rho))
    chi_e_out: FloatArray = np.zeros(len(rho))
    D_e_out: FloatArray = np.zeros(len(rho))

    for i in range(len(rho)):
        if rho[i] <= 0.05 or not np.isfinite(ne[i]) or ne[i] <= 1e-6:
            chi_i_out[i] = 0.01
            chi_e_out[i] = 0.01
            D_e_out[i] = 0.01
            continue

        Te_keV = max(Te[i], 0.01)
        Ti_keV = max(Ti[i], 0.01)
        qi = max(q_prof[i], 0.5)
        eps_i = max(rho[i] * a / R0, 1e-3)
        # s_hat_profile is built from rho and always shares its length, so the
        # loop index is always in range.
        s_hat_local = float(s_hat_profile[i])

        R_L_Te = -R0 / Te_keV * dTe_dr[i] if Te_keV > 0.01 else 0.0
        R_L_Ti = -R0 / Ti_keV * dTi_dr[i] if Ti_keV > 0.01 else 0.0
        R_L_ne = -R0 / max(ne[i], 1e-3) * dne_dr[i]

        local_params = GKLocalParams(
            R_L_Ti=max(R_L_Ti, 0.0),
            R_L_Te=max(R_L_Te, 0.0),
            R_L_ne=max(R_L_ne, -5.0),
            q=qi,
            s_hat=s_hat_local,
            Te_Ti=Te_keV / Ti_keV,
            Z_eff=Z_eff,
            nu_star=0.1,
            beta_e=0.01,
            epsilon=eps_i,
            kappa=kappa,
            delta=delta,
            rho=rho[i],
            R0=R0,
            a=a,
            B0=B0,
            n_e=ne[i],
            T_e_keV=Te_keV,
            T_i_keV=Ti_keV,
        )

        result: GKOutput | None
        if catch_execution_errors:
            try:
                result = solver.run_from_params(local_params)
            except Exception as exc:
                if not allow_gyrobohm_fallback:
                    raise RuntimeError(
                        f"{solver_label} solver execution failed at rho={rho[i]:.4f}; "
                        "configure/repair external solver or select tglf_native"
                    ) from exc
                result = None
        else:
            result = solver.run_from_params(local_params)

        if result is not None and result.converged:
            fluxes_are_valid = (
                np.isfinite(result.chi_i)
                and np.isfinite(result.chi_e)
                and np.isfinite(result.D_e)
                and result.chi_i >= 0.0
                and result.chi_e >= 0.0
                and result.D_e >= 0.0
            )
            if fluxes_are_valid:
                chi_i_out[i] = max(result.chi_i, 0.01)
                chi_e_out[i] = max(result.chi_e, 0.01)
                D_e_out[i] = max(result.D_e, 0.001)
                continue

        if not allow_gyrobohm_fallback:
            raise RuntimeError(
                f"{solver_label} returned unconverged transport at rho={rho[i]:.4f}; "
                "do not continue with degraded fallback transport"
            )

        # Explicit legacy mode only
        chi_i_out[i] = max(gyro_bohm_fallback()[i], 0.01)
        chi_e_out[i] = chi_i_out[i]
        D_e_out[i] = 0.1 * chi_e_out[i]

    return chi_i_out, chi_e_out, D_e_out
