# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Vacuum vessel eddy current model using a lumped-circuit approach.

Models the induction and decay of currents in the conducting vessel walls,
providing passive stability effects and flux perturbations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import ellipe, ellipk

logger = logging.getLogger(__name__)

# ─── Physical constants (CODATA 2018) ───────────────────────────────
_MU0 = 4.0 * np.pi * 1e-7  # H/m

# ─── Halo current parameters ────────────────────────────────────────
# Toroidal peaking factor: ratio of peak-to-average halo current density.
# ITER Physics Basis 1999, Ch. 3, §3.8.3: TPF ≈ 2 for ITER-like disruptions.
TPF = 2.0  # dimensionless


@dataclass(frozen=True)
class VesselElement:
    """Discrete conducting element of the vacuum vessel.

    Parameters
    ----------
    R : float
        Major radius of the element center [m].
    Z : float
        Vertical position of the element center [m].
    resistance : float
        Electrical resistance of the toroidal loop [Ohm].
    cross_section : float
        Cross-sectional area of the element [m^2].
    inductance : float
        Self-inductance of the toroidal loop [H].
    wall_thickness : float
        Effective wall thickness for τ_vessel calculation [m].
    conductivity : float
        Electrical conductivity of wall material [S/m].
        Stainless steel 316L: σ ≈ 1.35e6 S/m (Wesson 2011, App. C).
    """

    R: float
    Z: float
    resistance: float
    cross_section: float
    inductance: float
    wall_thickness: float = 0.04
    conductivity: float = 1.35e6


def vessel_time_constant(conductivity: float, wall_thickness: float, major_radius: float) -> float:
    """Return the resistive wall time constant τ_vessel.

    τ_vessel = μ₀ σ d R

    Wesson 2011, "Tokamaks", 4th ed., Eq. 6.6.6.

    Parameters
    ----------
    conductivity : float
        Wall conductivity σ [S/m].
    wall_thickness : float
        Wall thickness d [m].
    major_radius : float
        Major radius R [m].

    Returns
    -------
    float — Time constant [s].
    """
    return _MU0 * conductivity * wall_thickness * major_radius


def halo_current(plasma_current: float, f_halo: float = 0.3) -> float:
    """Return peak halo current I_halo = f_halo × TPF × I_p.

    ITER Physics Basis 1999, Ch. 3, §3.8.3:
    f_halo is the fraction of plasma current flowing in the halo
    (ITER default 0.3), TPF ≈ 2 is the toroidal peaking factor.

    Parameters
    ----------
    plasma_current : float
        Total plasma current I_p [A].
    f_halo : float
        Halo current fraction (default 0.3, ITER Physics Basis).

    Returns
    -------
    float — Peak halo current [A].
    """
    return f_halo * TPF * plasma_current


def halo_em_force(halo_current_a: float, b_poloidal: float, path_length: float) -> float:
    """Return electromagnetic force on the vessel from halo currents.

    F = I_halo × B_pol × L

    Noll et al. 1993, Fusion Eng. Des. 22, 315 — electromagnetic loads
    on the first wall and blanket due to disruption halo currents.

    Parameters
    ----------
    halo_current_a : float
        Peak halo current [A].
    b_poloidal : float
        Poloidal field at the wall [T].
    path_length : float
        Length of the current path in the wall [m].

    Returns
    -------
    float — Electromagnetic force [N].
    """
    return halo_current_a * b_poloidal * path_length


class VesselModel:
    """Lumped-circuit model for vessel eddy currents.

    Solves M dI/dt + R I = −V_ext, where M is the mutual inductance matrix
    and V_ext is the induced loop voltage.
    """

    def __init__(self, elements: list[VesselElement]) -> None:
        self.elements = elements
        self.n_elements = len(elements)
        self.I = np.zeros(self.n_elements)

        self.M = np.zeros((self.n_elements, self.n_elements))
        for i in range(self.n_elements):
            for j in range(self.n_elements):
                if i == j:
                    self.M[i, j] = elements[i].inductance
                else:
                    self.M[i, j] = self._mutual_inductance(
                        elements[i].R,
                        elements[i].Z,
                        elements[j].R,
                        elements[j].Z,
                    )

        self.R_mat = np.diag([el.resistance for el in elements])

        try:
            self.M_inv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            logger.error("Vessel inductance matrix is singular.")
            self.M_inv = np.zeros_like(self.M)

    def _mutual_inductance(self, R1: float, Z1: float, R2: float, Z2: float) -> float:
        """Mutual inductance between two coaxial toroidal loops.

        Neumann formula evaluated via complete elliptic integrals.
        Wesson 2011, App. A; Shafranov 1966, Rev. Plasma Phys. 2.
        """
        denom = (R1 + R2) ** 2 + (Z1 - Z2) ** 2
        if denom < 1e-30:
            return 0.0
        k2 = 4.0 * R1 * R2 / denom
        k2 = np.clip(k2, 1e-9, 0.999999)

        K_val = ellipk(k2)
        E_val = ellipe(k2)

        # M = μ₀ √(R₁R₂) [(2/k − k)K(k) − (2/k)E(k)]
        # Written in k² form to match Green's function in fusion_kernel.py.
        prefactor = _MU0 * np.sqrt(R1 * R2)
        m = prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / np.sqrt(k2)
        return float(m)

    def step(self, dt: float, dphi_ext_dt: np.ndarray) -> np.ndarray:
        """Advance eddy currents by one time step (explicit Euler).

        M dI/dt + R I = −dΦ_ext/dt
        dI/dt = M⁻¹(−R I − dΦ_ext/dt)

        Parameters
        ----------
        dt : float
            Time step [s].
        dphi_ext_dt : np.ndarray
            Rate of change of external poloidal flux through each element [Wb/s].

        Returns
        -------
        np.ndarray — Updated eddy currents [A].
        """
        if dt <= 0:
            return self.I

        dI_dt = self.M_inv @ (-self.R_mat @ self.I - dphi_ext_dt)
        self.I += dI_dt * dt
        return self.I

    def psi_vessel(self, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Poloidal flux contribution from vessel currents at observation points.

        Parameters
        ----------
        R, Z : np.ndarray
            Observation points.

        Returns
        -------
        np.ndarray — Flux contribution [Wb/rad].
        """
        out_shape = R.shape
        R_flat = R.flatten()
        Z_flat = Z.flatten()
        psi = np.zeros_like(R_flat)

        for i in range(self.n_elements):
            el = self.elements[i]
            if abs(self.I[i]) < 1e-6:
                continue

            denom = (R_flat + el.R) ** 2 + (Z_flat - el.Z) ** 2
            k2 = 4.0 * R_flat * el.R / np.maximum(denom, 1e-30)
            k2 = np.clip(k2, 1e-9, 0.999999)

            K_val = ellipk(k2)
            E_val = ellipe(k2)

            prefactor = (_MU0 / (2.0 * np.pi)) * np.sqrt(R_flat * el.R)
            g = prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / np.sqrt(k2)
            psi += g * self.I[i]

        return psi.reshape(out_shape)
