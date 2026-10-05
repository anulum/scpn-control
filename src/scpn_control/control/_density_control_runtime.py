# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density PI control and actuator commands

"""Density PI control and actuator commands."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control._density_transport import ParticleTransportModel

# Greenwald 2002, PPCF 44, R27, Eq. 1.
_GW_ITER_SAFETY_MARGIN = 0.85
_GW_PUMP_THRESHOLD = 0.95


@dataclass
class ActuatorCommand:
    """Fuelling and pumping set-points emitted by the density controller.

    Attributes
    ----------
    gas_puff_rate
        Gas-puff particle rate in particles/s.
    pellet_freq
        Pellet injection frequency in Hz.
    pellet_speed
        Pellet velocity in m/s.
    cryo_pump_speed
        Cryopump pumping speed in facility units.
    """

    gas_puff_rate: float
    pellet_freq: float
    pellet_speed: float
    cryo_pump_speed: float


class DensityController:
    """PI density controller with Greenwald limit enforcement.

    Greenwald limit: n_GW = I_p / (π a²) [10^20 m^-3]
    Greenwald 2002, PPCF 44, R27, Eq. 1.

    ITER operational margin: n/n_GW < 0.85.
    ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §2.3.
    """

    def __init__(self, model: ParticleTransportModel, dt_control: float = 0.001):
        if not math.isfinite(dt_control) or dt_control <= 0.0:
            raise ValueError("dt_control must be finite and positive.")
        self.model = model
        self.dt = dt_control
        self.ne_target: FloatArray = np.zeros(model.n_rho)

        # Default n_GW for ITER: I_p=15 MA, a=2.0 m → n_GW = 15/(π·4) ≈ 1.19×10^20 m^-3.
        # Greenwald 2002, PPCF 44, R27, Eq. 1.
        self.n_GW = 1.0e20  # [m^-3], updated via set_constraints
        self.gas_max = 1e22
        self.pellet_freq_max = 10.0
        self.pump_max = 10.0

        # PI gains (empirically tuned for ITER particle time scales ~0.1–1 s).
        self._Kp = 10.0
        self._Ki = 1.0
        self.integral_error = 0.0

    def set_target(self, ne_target: AnyFloatArray) -> None:
        """Set the target electron-density profile.

        Parameters
        ----------
        ne_target
            Desired density profile in m⁻³, shape ``(n_rho,)``; finite and
            non-negative.
        """
        self.ne_target = self._validate_density_profile(ne_target, "ne_target")

    def set_constraints(self, n_GW: float, gas_max: float, pellet_freq_max: float, pump_max: float) -> None:
        """Set the Greenwald density limit and actuator saturation bounds.

        Parameters
        ----------
        n_GW
            Greenwald density limit in m⁻³; must be positive.
        gas_max
            Maximum gas-puff rate in particles/s; must be non-negative.
        pellet_freq_max
            Maximum pellet frequency in Hz; must be non-negative.
        pump_max
            Maximum cryopump speed; must be non-negative.
        """
        self.n_GW = self._validate_positive_scalar(n_GW, "n_GW")
        self.gas_max = self._validate_non_negative_scalar(gas_max, "gas_max")
        self.pellet_freq_max = self._validate_non_negative_scalar(pellet_freq_max, "pellet_freq_max")
        self.pump_max = self._validate_non_negative_scalar(pump_max, "pump_max")

    @staticmethod
    def compute_greenwald_limit(I_p_MA: float, a_m: float) -> float:
        """n_GW = I_p / (π a²) [10^20 m^-3], converted to [m^-3].

        Greenwald 2002, PPCF 44, R27, Eq. 1.
        """
        if not math.isfinite(I_p_MA) or I_p_MA <= 0.0:
            raise ValueError("I_p_MA must be finite and positive.")
        if not math.isfinite(a_m) or a_m <= 0.0:
            raise ValueError("a_m must be finite and positive.")
        return I_p_MA / (math.pi * a_m**2) * 1e20

    def greenwald_fraction(self, ne: AnyFloatArray, I_p_MA: float, a: float) -> float:
        """Volume-averaged n / n_GW.

        Greenwald 2002, PPCF 44, R27, Eq. 1.
        ITER safe operating limit: fraction < 0.85.
        ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §2.3.
        """
        ne_arr = self._validate_density_profile(ne, "ne")
        vol = np.sum(self.model.V_prime * self.model.drho)
        N_tot = np.sum(ne_arr * self.model.V_prime * self.model.drho)
        n_avg = N_tot / vol

        n_GW = self.compute_greenwald_limit(I_p_MA, a)
        return float(n_avg / n_GW)

    def below_greenwald_safety_margin(self, ne: AnyFloatArray) -> bool:
        """Return True if the volume-averaged density is within the ITER safety margin.

        ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §2.3: n/n_GW < 0.85.
        """
        ne_arr = self._validate_density_profile(ne, "ne")
        vol = np.sum(self.model.V_prime * self.model.drho)
        n_avg = np.sum(ne_arr * self.model.V_prime * self.model.drho) / vol
        return bool(n_avg < _GW_ITER_SAFETY_MARGIN * self.n_GW)

    def step(self, ne_measured: AnyFloatArray) -> ActuatorCommand:
        """Compute one PI control action with Greenwald-limit enforcement.

        The volume-integrated particle error drives a PI command; gas puffing
        and pellets fuel a deficit while the cryopump removes excess. Above the
        hard Greenwald fraction (0.95) the pump saturates regardless of the PI
        term.

        Parameters
        ----------
        ne_measured
            Measured electron-density profile in m⁻³, shape ``(n_rho,)``.

        Returns
        -------
        ActuatorCommand
            Fuelling and pumping set-points for this control cycle.
        """
        ne_arr = self._validate_density_profile(ne_measured, "ne_measured")
        vol = np.sum(self.model.V_prime * self.model.drho)
        N_meas = np.sum(ne_arr * self.model.V_prime * self.model.drho)
        N_targ = np.sum(self.ne_target * self.model.V_prime * self.model.drho)

        error = N_targ - N_meas
        self.integral_error += error * self.dt

        cmd = self._Kp * error + self._Ki * self.integral_error

        f_gw = (N_meas / vol) / self.n_GW

        gas = 0.0
        pellet = 0.0
        pump = 0.0

        if f_gw > _GW_PUMP_THRESHOLD:
            pump = self.pump_max
        elif cmd > 0:
            gas = min(cmd, self.gas_max)
            if cmd > self.gas_max * 0.5:
                pellet = min(self.pellet_freq_max, (cmd - self.gas_max * 0.5) / 1e21)
        else:
            pump = min(self.pump_max, -cmd / 1e20)

        return ActuatorCommand(gas, pellet, 500.0, pump)

    def _validate_density_profile(self, values: AnyFloatArray, name: str) -> FloatArray:
        arr = self.model._validate_profile(values, name)
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative.")
        return arr

    @staticmethod
    def _validate_positive_scalar(value: float, name: str) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return float(value)

    @staticmethod
    def _validate_non_negative_scalar(value: float, name: str) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return float(value)
