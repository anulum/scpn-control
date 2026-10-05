# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Detachment control runtime.

"""Stateful software control and coordination for divertor detachment."""

from __future__ import annotations

import math
from enum import Enum, auto

# Stangeby 2000, Ch. 16; Lipschultz et al. 1999, PPCF 41, A585.
_T_DETACHMENT_ONSET_EV = 5.0  # eV
_T_BIFURCATION_EV = 30.0  # eV
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


class DetachmentState(Enum):
    """Divertor detachment regime classification."""

    ATTACHED = auto()
    PARTIALLY_DETACHED = auto()
    FULLY_DETACHED = auto()
    XPOINT_MARFE = auto()


class DetachmentController:
    """Software PI candidate for divertor target-temperature control.

    Target: T_div ≈ 3 eV (below the 5 eV onset, above the MARFE threshold).
    Stangeby 2000, Ch. 16: T_div < 5 eV criterion for detachment onset.
    Lipschultz et al. 1999, PPCF 41, A585: thermal bifurcation stability.

    The returned scalar has no established actuator conversion or physical
    seeding-rate unit. ``target_DOD`` is retained for API compatibility but
    does not participate in the current PI law.
    """

    def __init__(self, impurity: str = "N2", target_DOD: float = 3.0, target_T_t_eV: float = 3.0):
        self.impurity = impurity
        self.target_DOD = _finite_scalar("target_DOD", target_DOD, positive=True)
        self.target_T_t = _finite_scalar("target_T_t_eV", target_T_t_eV, positive=True)

        # Software gains; actuator conversion and physical units are unresolved.
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
        """Compute one finite, non-negative software PI candidate atomically.

        PI control on the target-temperature error, with a hard seeding cutback
        when the radiation front reaches the X-point (MARFE risk).

        Parameters
        ----------
        T_t_measured
            Measured divertor target temperature in eV; must be positive.
        n_t_measured
            Measured target density in 10¹⁹ m⁻³; must be positive.
        P_rad_measured
            Measured radiated power in MW; must be non-negative.
        rho_front
            Radiation-front position in [0, 1] (0 = target, 1 = X-point).
        dt
            Control time step in seconds; must be positive.

        Returns
        -------
        float
            Software command scalar (non-negative; no calibrated rate unit).
        """
        T_t_measured = _finite_scalar("T_t_measured", T_t_measured, positive=True)
        _finite_scalar("n_t_measured", n_t_measured, positive=True)
        _finite_scalar("P_rad_measured", P_rad_measured, nonnegative=True)
        rho_front = _unit_interval("rho_front", rho_front)
        dt = _finite_scalar("dt", dt, positive=True)
        state = self._determine_state(T_t_measured, rho_front)
        integral = _finite_scalar("integral_e", self.integral_e)
        last_cmd = _finite_scalar("last_cmd", self.last_cmd, nonnegative=True)

        if state == DetachmentState.XPOINT_MARFE:
            # Hard reduction to retreat radiation front from X-point.
            # Lipschultz et al. 1999, PPCF 41, A585: slow ramp-back required.
            cmd = _finite_scalar("seeding command", last_cmd * 0.5, nonnegative=True)
            next_integral = _finite_scalar("integral_e", integral * 0.5)
        else:
            error = _finite_scalar("temperature error", T_t_measured - self.target_T_t)
            increment = _finite_scalar("integral increment", error * dt)
            next_integral = _finite_scalar("integral_e", integral + increment)
            kp = _finite_scalar("Kp", self.Kp, nonnegative=True)
            ki = _finite_scalar("Ki", self.Ki, nonnegative=True)
            proportional = _finite_scalar("proportional command", kp * error)
            integral_command = _finite_scalar("integral command", ki * next_integral)
            cmd = max(0.0, _finite_scalar("seeding command", proportional + integral_command))

        self.state = state
        self.integral_e = next_integral
        self.last_cmd = cmd
        return cmd


class MultiImpuritySeeding:
    """Coordinate software candidates with all-species state rollback.

    Parameters
    ----------
    impurities
        Impurity species symbols to seed.
    controllers
        Per-impurity detachment controllers keyed by species symbol.
    """

    def __init__(self, impurities: list[str], controllers: dict[str, DetachmentController]):
        if len(set(impurities)) != len(impurities):
            raise ValueError("duplicate impurity species")
        self.impurities = list(impurities)
        self.controllers = dict(controllers)

    def step(self, diagnostics: dict[str, float], dt: float) -> dict[str, float]:
        """Compute per-impurity candidates from a diagnostics frame.

        Parameters
        ----------
        diagnostics
            Diagnostic values (``T_target_eV``, ``n_target_19``, ``P_rad_MW``,
            ``rho_front``); defaults are used for absent keys.
        dt
            Control time step in seconds; must be positive.

        Returns
        -------
        dict[str, float]
            Software command scalar per species (0.0 for species without a
            controller); no calibrated seeding-rate unit is implied.
        """
        dt = _finite_scalar("dt", dt, positive=True)
        T_t = _finite_scalar("T_target_eV", diagnostics.get("T_target_eV", 20.0), positive=True)
        n_t = _finite_scalar("n_target_19", diagnostics.get("n_target_19", 10.0), positive=True)
        P_rad = _finite_scalar("P_rad_MW", diagnostics.get("P_rad_MW", 10.0), nonnegative=True)
        rho_front = _unit_interval("rho_front", diagnostics.get("rho_front", 0.1))

        if len(set(self.impurities)) != len(self.impurities):
            raise ValueError("duplicate impurity species")
        active = [self.controllers[imp] for imp in self.impurities if imp in self.controllers]
        if len({id(controller) for controller in active}) != len(active):
            raise ValueError("controllers must not share mutable state across species")
        snapshots = [
            (controller, controller.integral_e, controller.last_cmd, controller.state) for controller in active
        ]
        rates: dict[str, float] = {}
        try:
            for imp in self.impurities:
                if imp in self.controllers:
                    rates[imp] = self.controllers[imp].step(T_t, n_t, P_rad, rho_front, dt)
                else:
                    rates[imp] = 0.0
        except Exception:
            for controller, integral, command, state in snapshots:
                controller.integral_e = integral
                controller.last_cmd = command
                controller.state = state
            raise
        return rates
