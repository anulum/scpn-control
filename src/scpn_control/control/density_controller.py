# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Greenwald density limit: n_GW = I_p / (π a²) [10^20 m^-3]
# Greenwald 2002, PPCF 44, R27, Eq. 1.
# ITER operational limit: n/n_GW < 0.85 — ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §2.3.
_GW_ITER_SAFETY_MARGIN = 0.85  # ITER Physics Basis 1999, §2.3

# Greenwald fraction at which the controller switches to maximum pumping (hard limit).
_GW_PUMP_THRESHOLD = 0.95  # 5% headroom below disruption risk

# Solid D₂ number density [m^-3]: 6×10^28 — standard deuterium ice density.
_N_D2_SOLID_M3 = 6e28


class ParticleTransportModel:
    def __init__(self, n_rho: int = 50, R0: float = 6.2, a: float = 2.0):
        self.n_rho = n_rho
        self.R0 = R0
        self.a = a
        self.rho = np.linspace(0.0, 1.0, n_rho)
        self.drho = self.rho[1] - self.rho[0]

        # Anomalous diffusivity and pinch velocity: default ITER-like values.
        # D ~ 1 m²/s, V_pinch ~ -0.1 m/s (inward Ware pinch order of magnitude).
        self.D = np.ones(n_rho) * 1.0  # m²/s
        self.V_pinch = -np.ones(n_rho) * 0.1  # m/s, inward

        # Circular cross-section volume elements (simplified geometry).
        self.V = 2.0 * np.pi**2 * self.R0 * (self.a * self.rho) ** 2
        self.V_prime = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho

    def set_transport(self, D: np.ndarray, V_pinch: np.ndarray) -> None:
        self.D = D
        self.V_pinch = V_pinch

    def gas_puff_source(self, rate: float, penetration_depth: float = 0.03) -> np.ndarray:
        """Particles/s. Edge-localised source from gas injection.

        Pacher et al. 2007, Nucl. Fusion 47, 469: ITER gas-injection modelling
        places the effective source within ~3% of the minor radius from the wall.
        """
        decay = np.exp(-(1.0 - self.rho) / penetration_depth)
        decay /= np.sum(decay * self.V_prime * self.drho) + 1e-10
        return np.asarray(rate * decay)

    def pellet_source(self, speed_ms: float, radius_mm: float, launch_angle_deg: float = 0.0) -> np.ndarray:
        """Gaussian deposition profile from a single pellet.

        Penetration depth estimated via the NGS (Neutral Gas Shielding) model:
        Milora 1995, J. Vac. Sci. Technol. A 13, 534 — ablation rate scales with
        pellet radius and velocity. Rozhansky et al. 1994, Nucl. Fusion 34, 383:
        deep fueling requires high speed and large pellet radius.
        """
        if radius_mm <= 0.0:
            return np.zeros(self.n_rho)

        N_pellet = (4.0 / 3.0 * np.pi * (radius_mm * 1e-3) ** 3) * _N_D2_SOLID_M3

        # Penetration depth decreases with higher speed × radius (NGS approximation).
        pen_rho = max(0.2, 1.0 - 0.1 * (speed_ms / 1000.0) * (radius_mm / 2.0))

        dep = np.exp(-((self.rho - pen_rho) ** 2) / (0.1**2))
        dep /= np.sum(dep * self.V_prime * self.drho) + 1e-10
        return np.asarray(N_pellet * dep)

    def nbi_source(self, beam_energy_keV: float, power_MW: float) -> np.ndarray:
        """Core-peaked particle source from neutral beam injection."""
        if power_MW <= 0.0:
            return np.zeros(self.n_rho)

        I_beam = power_MW * 1e6 / (beam_energy_keV * 1e3)
        rate = I_beam / 1.6e-19  # 1.6×10^-19 C/particle (elementary charge)

        dep = np.exp(-((self.rho - 0.3) ** 2) / (0.3**2))
        dep /= np.sum(dep * self.V_prime * self.drho) + 1e-10
        return np.asarray(rate * dep)

    def cryopump_sink(self, pump_speed: float, ne_edge: float) -> np.ndarray:
        """Edge particle removal from cryopump."""
        sink = np.zeros(self.n_rho)
        sink[-1] = pump_speed * ne_edge / (self.V_prime[-1] * self.drho + 1e-10)
        return sink

    def recycling_source(self, outflux: float, recycling_coeff: float = 0.97) -> np.ndarray:
        """
        Recycling_coeff = 0.97 is the standard ITER assumption for a metal wall.
        ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §4.2.
        """
        return self.gas_puff_source(outflux * recycling_coeff, penetration_depth=0.02)

    def step(self, ne: np.ndarray, sources: np.ndarray, dt: float) -> np.ndarray:
        # Explicit forward-Euler diffusion — CFL stability: dt < drho² / (2 D_max).
        D_max = np.max(self.D)
        if D_max > 0.0:
            dt_cfl = (self.drho * self.a) ** 2 / (2.0 * D_max)
            if dt > dt_cfl:
                dt = dt_cfl

        flux = np.zeros(self.n_rho + 1)
        for i in range(1, self.n_rho):
            grad_n = (ne[i] - ne[i - 1]) / self.drho
            n_face = 0.5 * (ne[i] + ne[i - 1])
            D_face = 0.5 * (self.D[i] + self.D[i - 1])
            V_face = 0.5 * (self.V_pinch[i] + self.V_pinch[i - 1])
            flux[i] = -D_face * grad_n / self.a + V_face * n_face

        flux[0] = 0.0
        flux[-1] = -self.D[-1] * (0.0 - ne[-1]) / self.drho / self.a + self.V_pinch[-1] * ne[-1]

        dne_dt = np.zeros(self.n_rho)
        for i in range(self.n_rho):
            Vp = max(self.V_prime[i], 1e-6)
            Vp_plus = self.V_prime[i] if i == self.n_rho - 1 else 0.5 * (self.V_prime[i] + self.V_prime[i + 1])
            Vp_minus = 0.0 if i == 0 else 0.5 * (self.V_prime[i] + self.V_prime[i - 1])
            div_flux = (Vp_plus * flux[i + 1] - Vp_minus * flux[i]) / (Vp * self.drho * self.a)
            dne_dt[i] = -div_flux + sources[i]

        ne_new = ne + dne_dt * dt
        return np.asarray(np.maximum(ne_new, 1e16))


@dataclass
class ActuatorCommand:
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
        self.model = model
        self.dt = dt_control
        self.ne_target = np.zeros(model.n_rho)

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

    def set_target(self, ne_target: np.ndarray) -> None:
        self.ne_target = ne_target

    def set_constraints(self, n_GW: float, gas_max: float, pellet_freq_max: float, pump_max: float) -> None:
        self.n_GW = n_GW
        self.gas_max = gas_max
        self.pellet_freq_max = pellet_freq_max
        self.pump_max = pump_max

    @staticmethod
    def compute_greenwald_limit(I_p_MA: float, a_m: float) -> float:
        """n_GW = I_p / (π a²) [10^20 m^-3], converted to [m^-3].

        Greenwald 2002, PPCF 44, R27, Eq. 1.
        """
        return I_p_MA / (math.pi * a_m**2) * 1e20

    def greenwald_fraction(self, ne: np.ndarray, I_p_MA: float, a: float) -> float:
        """Volume-averaged n / n_GW.

        Greenwald 2002, PPCF 44, R27, Eq. 1.
        ITER safe operating limit: fraction < 0.85.
        ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §2.3.
        """
        vol = np.sum(self.model.V_prime * self.model.drho)
        N_tot = np.sum(ne * self.model.V_prime * self.model.drho)
        n_avg = N_tot / vol

        n_GW = self.compute_greenwald_limit(I_p_MA, a)
        return float(n_avg / n_GW)

    def below_greenwald_safety_margin(self, ne: np.ndarray) -> bool:
        """True if volume-averaged density is within the ITER safety margin.

        ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §2.3: n/n_GW < 0.85.
        """
        vol = np.sum(self.model.V_prime * self.model.drho)
        n_avg = np.sum(ne * self.model.V_prime * self.model.drho) / vol
        return bool(n_avg < _GW_ITER_SAFETY_MARGIN * self.n_GW)

    def step(self, ne_measured: np.ndarray) -> ActuatorCommand:
        vol = np.sum(self.model.V_prime * self.model.drho)
        N_meas = np.sum(ne_measured * self.model.V_prime * self.model.drho)
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


class KalmanDensityEstimator:
    def __init__(self, n_rho: int, n_chords: int = 8):
        self.n_rho = n_rho
        self.n_chords = n_chords
        self.x = np.zeros(n_rho)
        self.P = np.eye(n_rho) * 1e38
        self.Q = np.eye(n_rho) * 1e36  # process noise covariance
        self.R = np.eye(n_chords) * 1e34  # measurement noise covariance

    def measurement_matrix(self, chord_angles: np.ndarray) -> np.ndarray:
        """Abel-transform projection matrix for interferometry chords."""
        C = np.zeros((self.n_chords, self.n_rho))
        for i in range(self.n_chords):
            impact = i / self.n_chords
            for j in range(self.n_rho):
                rho = j / self.n_rho
                if rho > impact:
                    C[i, j] = 2.0 * rho / math.sqrt(rho**2 - impact**2 + 1e-6)
        return C

    def predict(self, ne: np.ndarray, dt: float) -> np.ndarray:
        self.x = ne
        self.P = self.P + self.Q * dt
        return self.x

    def update(self, ne_pred: np.ndarray, measurements: np.ndarray, chord_angles: np.ndarray) -> np.ndarray:
        C = self.measurement_matrix(chord_angles)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        inn = measurements - C @ ne_pred
        self.x = ne_pred + K @ inn
        self.P = (np.eye(self.n_rho) - K @ C) @ self.P
        return self.x


@dataclass
class PelletSchedule:
    times: list[float]
    speeds: list[float]
    sizes: list[float]


class FuelingOptimizer:
    def optimize_pellet_sequence(
        self, ne_current: np.ndarray, ne_target: np.ndarray, n_pellets: int, time_horizon: float
    ) -> PelletSchedule:
        """Evenly-spaced pellet schedule over the given horizon."""
        if n_pellets <= 0:
            return PelletSchedule([], [], [])

        dt = time_horizon / (n_pellets + 1)
        times = [dt * (i + 1) for i in range(n_pellets)]
        speeds = [500.0] * n_pellets
        sizes = [2.0] * n_pellets
        return PelletSchedule(times, speeds, sizes)
