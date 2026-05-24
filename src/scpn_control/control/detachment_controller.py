# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""Divertor detachment state estimation and impurity-seeding control utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from scpn_control.core.sol_model import TwoPointSOL

# Detachment onset: T_div < 5 eV causes volumetric recombination and ion-neutral
# friction, decoupling target from upstream conditions.
# Stangeby 2000, "The Plasma Boundary of Magnetic Fusion Devices", Ch. 16.
_T_DETACHMENT_ONSET_EV = 5.0  # [eV] — Stangeby 2000, Ch. 16

# Two-point model parallel heat conduction:
# q_∥ = κ₀ T_u^(7/2) / (7 L_∥)
# where κ₀ = 2390 W m^-1 eV^(-7/2) (electron Spitzer conductivity).
# Stangeby 2000, Eq. 5.69; Wesson 2004, §4.10.
_KAPPA_0_SPITZER = 2390.0  # W m^-1 eV^(-7/2), Stangeby 2000, Eq. 5.67

# Nitrogen seeding: typically 10^21–10^22 molecules/s for ITER divertor detachment.
# Kallenbach et al. 2015, Nucl. Fusion 55, 053026, Table 2.
_N2_SEEDING_RATE_ITER_MAX = 1e22  # molecules/s — Kallenbach et al. 2015

# Thermal bifurcation threshold: below this target temperature the Greenwald-
# Stangeby rollover causes a thermal collapse to the cold, detached branch.
# Lipschultz et al. 1999, PPCF 41, A585 (thermal bifurcation in Alcator C-Mod).
_T_BIFURCATION_EV = 30.0  # [eV] — Lipschultz et al. 1999

# X-point MARFE onset: radiation front reaches > 80% of the field line length
# from the divertor target toward the X-point.
# Lipschultz et al. 1999, PPCF 41, A585.
_XPOINT_MARFE_THRESHOLD = 0.8  # dimensionless front position


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _unit_interval(name: str, value: float) -> float:
    scalar = _finite_scalar(name, value, nonnegative=True)
    if scalar > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return scalar


def _ordered_nonnegative_array(name: str, values: np.ndarray) -> np.ndarray:
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


class DetachmentState(Enum):
    ATTACHED = auto()
    PARTIALLY_DETACHED = auto()
    FULLY_DETACHED = auto()
    XPOINT_MARFE = auto()


class RadiationFrontModel:
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


class DetachmentController:
    """PI controller driving divertor detachment via impurity seeding.

    Target: T_div ≈ 3 eV (below the 5 eV onset, above the MARFE threshold).
    Stangeby 2000, Ch. 16: T_div < 5 eV criterion for detachment onset.
    Lipschultz et al. 1999, PPCF 41, A585: thermal bifurcation stability.

    Nitrogen seeding rates up to 10^22 molecules/s for ITER.
    Kallenbach et al. 2015, Nucl. Fusion 55, 053026.
    """

    def __init__(self, impurity: str = "N2", target_DOD: float = 3.0, target_T_t_eV: float = 3.0):
        self.impurity = impurity
        self.target_DOD = _finite_scalar("target_DOD", target_DOD, positive=True)
        self.target_T_t = _finite_scalar("target_T_t_eV", target_T_t_eV, positive=True)

        # PI gains [Pa m³/s per eV] — tuned for ITER divertor time scales (~0.1 s).
        self.Kp = 50.0
        self.Ki = 10.0

        self.integral_e = 0.0
        self.last_cmd = 0.0
        self.state = DetachmentState.ATTACHED

    def _determine_state(self, T_t: float, rho_front: float) -> DetachmentState:
        """Classify divertor state.

        T_t > 30 eV: attached (Lipschultz et al. 1999 bifurcation upper branch).
        5 < T_t ≤ 30 eV: partially detached (below thermal bifurcation but above onset).
        T_t ≤ 5 eV: fully detached (Stangeby 2000, Ch. 16).
        rho_front > 0.8: X-point MARFE risk (Lipschultz et al. 1999, PPCF 41, A585).
        """
        T_t = _finite_scalar("T_t", T_t, positive=True)
        rho_front = _unit_interval("rho_front", rho_front)
        if rho_front > _XPOINT_MARFE_THRESHOLD:
            return DetachmentState.XPOINT_MARFE
        if T_t > _T_BIFURCATION_EV:
            return DetachmentState.ATTACHED
        if T_t > _T_DETACHMENT_ONSET_EV:
            return DetachmentState.PARTIALLY_DETACHED
        return DetachmentState.FULLY_DETACHED

    def step(
        self, T_t_measured: float, n_t_measured: float, P_rad_measured: float, rho_front: float, dt: float
    ) -> float:
        T_t_measured = _finite_scalar("T_t_measured", T_t_measured, positive=True)
        _finite_scalar("n_t_measured", n_t_measured, positive=True)
        _finite_scalar("P_rad_measured", P_rad_measured, nonnegative=True)
        rho_front = _unit_interval("rho_front", rho_front)
        dt = _finite_scalar("dt", dt, positive=True)
        self.state = self._determine_state(T_t_measured, rho_front)

        if self.state == DetachmentState.XPOINT_MARFE:
            # Hard reduction to retreat radiation front from X-point.
            # Lipschultz et al. 1999, PPCF 41, A585: slow ramp-back required.
            self.last_cmd *= 0.5
            self.integral_e *= 0.5
            return self.last_cmd

        error = T_t_measured - self.target_T_t
        self.integral_e += error * dt
        cmd = self.Kp * error + self.Ki * self.integral_e
        cmd = max(0.0, float(cmd))
        self.last_cmd = cmd
        return cmd


@dataclass
class DetachmentPoint:
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

    def scan_seeding(self, seeding_range: np.ndarray, P_SOL_MW: float, n_u_19: float) -> list[DetachmentPoint]:
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


class MultiImpuritySeeding:
    def __init__(self, impurities: list[str], controllers: dict[str, DetachmentController]):
        self.impurities = impurities
        self.controllers = controllers

    def step(self, diagnostics: dict[str, float], dt: float) -> dict[str, float]:
        dt = _finite_scalar("dt", dt, positive=True)
        T_t = _finite_scalar("T_target_eV", diagnostics.get("T_target_eV", 20.0), positive=True)
        n_t = _finite_scalar("n_target_19", diagnostics.get("n_target_19", 10.0), positive=True)
        P_rad = _finite_scalar("P_rad_MW", diagnostics.get("P_rad_MW", 10.0), nonnegative=True)
        rho_front = _unit_interval("rho_front", diagnostics.get("rho_front", 0.1))

        rates: dict[str, float] = {}
        for imp in self.impurities:
            if imp in self.controllers:
                rates[imp] = self.controllers[imp].step(T_t, n_t, P_rad, rho_front, dt)
            else:
                rates[imp] = 0.0
        return rates
