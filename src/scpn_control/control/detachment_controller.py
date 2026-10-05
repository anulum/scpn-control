# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Detachment controller.

"""Radiation-front and steady-state detachment model utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control.detachment_control_runtime import (
    _T_BIFURCATION_EV,
    _T_DETACHMENT_ONSET_EV,
    _XPOINT_MARFE_THRESHOLD,
    _finite_scalar,
)
from scpn_control.control.detachment_control_runtime import (
    DetachmentController as DetachmentController,
)
from scpn_control.control.detachment_control_runtime import (
    DetachmentState as DetachmentState,
)
from scpn_control.control.detachment_control_runtime import (
    MultiImpuritySeeding as MultiImpuritySeeding,
)
from scpn_control.core.sol_model import TwoPointSOL

# Stangeby 2000, Eq. 5.67: electron Spitzer conductivity in W m^-1 eV^(-7/2).
_KAPPA_0_SPITZER = 2390.0


def _ordered_nonnegative_array(name: str, values: AnyFloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} values must be finite")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} values must be strictly increasing")
    return arr


class RadiationFrontModel:
    """Impurity radiation-front position and detachment-degree model.

    Parameters
    ----------
    impurity
        Seeding impurity species symbol.
    R0
        Major radius in metres; must be positive.
    a
        Minor radius in metres; must be positive and below ``R0``.
    q95
        Edge safety factor q95; must be positive.
    """

    def __init__(self, impurity: str, R0: float, a: float, q95: float):
        self.impurity = impurity
        self.R0 = _finite_scalar("R0", R0, positive=True)
        self.a = _finite_scalar("a", a, positive=True)
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        self.q95 = _finite_scalar("q95", q95, positive=True)

    def radiation_temperature(self, impurity: str) -> float:
        """Peak radiation temperature [eV] for each impurity species.

        N₂: ~10 eV, Ne: ~30 eV, Ar: ~100 eV.
        Kallenbach et al. 2015, Nucl. Fusion 55, 053026, §3.
        """
        temps = {"N2": 10.0, "Ne": 30.0, "Ar": 100.0}
        return temps.get(impurity, 10.0)

    def front_position(self, P_SOL_MW: float, n_u_19: float, seeding_rate: float) -> float:
        """Radiation front location: 0 = target, 1 = X-point.

        Higher seeding rate moves the front toward the X-point.
        Higher P_SOL pushes it back toward the target.
        Lipschultz et al. 1999, PPCF 41, A585: empirical front-position scaling.
        """
        P_SOL_MW = _finite_scalar("P_SOL_MW", P_SOL_MW, positive=True)
        n_u_19 = _finite_scalar("n_u_19", n_u_19, positive=True)
        seeding_rate = _finite_scalar("seeding_rate", seeding_rate, nonnegative=True)

        drive = seeding_rate * n_u_19 / P_SOL_MW
        rho_front = 1.0 - np.exp(-drive * 2.0)
        return float(np.clip(rho_front, 0.0, 1.0))

    def degree_of_detachment(self, T_target_eV: float, n_target: float, n_u: float) -> float:
        """DOD = Γ_t,attached / Γ_t,actual via T_target rollover.

        DOD = 1 when attached (T_t > 5 eV).
        Below the onset (Stangeby 2000, Ch. 16) DOD rises as T_t falls,
        reflecting reduced ion flux from volumetric recombination.
        """
        T_target_eV = _finite_scalar("T_target_eV", T_target_eV, positive=True)
        _finite_scalar("n_target", n_target, positive=True)
        _finite_scalar("n_u", n_u, positive=True)
        if T_target_eV > _T_DETACHMENT_ONSET_EV:
            return 1.0
        if T_target_eV <= 0.1:
            return 10.0
        return 1.0 + 5.0 * (1.0 - T_target_eV / _T_DETACHMENT_ONSET_EV)


def two_point_q_parallel(T_upstream_eV: float, L_parallel_m: float) -> float:
    """Parallel heat flux from the Spitzer conduction model.

    q_∥ = κ₀ T_u^(7/2) / (7 L_∥)   [W m^-2]

    Stangeby 2000, "The Plasma Boundary of Magnetic Fusion Devices", Eq. 5.69.
    κ₀ = 2390 W m^-1 eV^(-7/2) (electron Spitzer conductivity, Stangeby Eq. 5.67).
    """
    T_upstream_eV = _finite_scalar("T_upstream_eV", T_upstream_eV, nonnegative=True)
    L_parallel_m = _finite_scalar("L_parallel_m", L_parallel_m, positive=True)
    if T_upstream_eV == 0.0:
        return 0.0
    return float(_KAPPA_0_SPITZER * T_upstream_eV**3.5 / (7.0 * L_parallel_m))


@dataclass
class DetachmentPoint:
    """One steady-state point on the detachment bifurcation scan.

    Attributes
    ----------
    seeding_rate
        Impurity seeding rate at this point.
    T_target
        Divertor target temperature in eV.
    n_target
        Divertor target density in 10¹⁹ m⁻³.
    DOD
        Degree of detachment.
    P_rad_frac
        Radiated power fraction.
    state
        The detachment regime at this point.
    """

    seeding_rate: float
    T_target: float
    n_target: float
    DOD: float
    P_rad_frac: float
    state: DetachmentState


class DetachmentBifurcation:
    """Steady-state scan across seeding rates to locate the thermal bifurcation.

    Thermal bifurcation (S-curve) in T_target vs seeding_rate described by:
    Lipschultz et al. 1999, PPCF 41, A585 (Alcator C-Mod data + model).
    Two-point model (Stangeby 2000, Eq. 5.69) provides the upstream–target link.
    """

    def __init__(self, sol: TwoPointSOL, impurity: str):
        self.sol = sol
        self.impurity = impurity
        self.front_model = RadiationFrontModel(impurity, sol.R0, sol.a, sol.q95)

    def _steady_state_target(self, seeding_rate: float, P_SOL_MW: float, n_u_19: float) -> DetachmentPoint:
        seeding_rate = _finite_scalar("seeding_rate", seeding_rate, nonnegative=True)
        P_SOL_MW = _finite_scalar("P_SOL_MW", P_SOL_MW, positive=True)
        n_u_19 = _finite_scalar("n_u_19", n_u_19, positive=True)
        # f_rad scales with seeding rate; cap at 0.95 to avoid unphysical values.
        f_rad = min(0.95, seeding_rate * 0.1)

        res = self.sol.solve(P_SOL_MW, n_u_19, f_rad=f_rad)
        rho_front = self.front_model.front_position(P_SOL_MW, n_u_19, seeding_rate)

        T_t = res.T_target_eV
        if f_rad > 0.8:
            # Deep detachment: exponential T_t collapse below conduction-limit solution.
            # Lipschultz et al. 1999: rapid T_t drop on the detached branch.
            T_t = max(0.5, T_t * math.exp(-(f_rad - 0.8) * 20.0))

        dod = self.front_model.degree_of_detachment(T_t, res.n_target_19, n_u_19)

        if rho_front > _XPOINT_MARFE_THRESHOLD:
            state = DetachmentState.XPOINT_MARFE
        elif T_t > _T_BIFURCATION_EV:
            state = DetachmentState.ATTACHED
        elif T_t > _T_DETACHMENT_ONSET_EV:
            state = DetachmentState.PARTIALLY_DETACHED
        else:
            state = DetachmentState.FULLY_DETACHED

        return DetachmentPoint(seeding_rate, T_t, res.n_target_19, dod, f_rad, state)

    def scan_seeding(self, seeding_range: AnyFloatArray, P_SOL_MW: float, n_u_19: float) -> list[DetachmentPoint]:
        """Scan steady-state detachment across a range of seeding rates.

        Parameters
        ----------
        seeding_range
            Strictly increasing non-negative seeding rates.
        P_SOL_MW
            Scrape-off-layer power in MW; must be positive.
        n_u_19
            Upstream density in 10¹⁹ m⁻³; must be positive.

        Returns
        -------
        list[DetachmentPoint]
            One steady-state detachment point per seeding rate.
        """
        seeding_range = _ordered_nonnegative_array("seeding_range", seeding_range)
        return [self._steady_state_target(sr, P_SOL_MW, n_u_19) for sr in seeding_range]

    def find_rollover_point(self, P_SOL_MW: float, n_u_19: float) -> float:
        """Seeding rate where ion flux Γ ~ n_t √T_t peaks (flux rollover).

        Rollover marks the detachment onset on the bifurcation S-curve.
        Stangeby 2000, Ch. 16; Lipschultz et al. 1999, PPCF 41, A585.
        """
        P_SOL_MW = _finite_scalar("P_SOL_MW", P_SOL_MW, positive=True)
        n_u_19 = _finite_scalar("n_u_19", n_u_19, positive=True)
        sr_scan = np.linspace(0.0, 10.0, 100)
        fluxes = [
            self._steady_state_target(sr, P_SOL_MW, n_u_19).n_target
            * math.sqrt(self._steady_state_target(sr, P_SOL_MW, n_u_19).T_target)
            for sr in sr_scan
        ]
        return float(sr_scan[np.argmax(fluxes)])
