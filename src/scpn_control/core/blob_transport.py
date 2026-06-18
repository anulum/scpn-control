# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL Filament Blob Transport
"""
SOL filament / blob transport.

References
----------
D'Ippolito, D. A., Myra, J. R. & Zweben, S. J., Phys. Plasmas 18 (2011) 060501.
    Comprehensive review of convective SOL transport by blobs/filaments.
Krasheninnikov, S. I., Phys. Lett. A 283 (2001) 368.
    Blob velocity scaling: v_b = 2T_e/(e B R δ_b) × (c_s/Ω_i)  [Eq. 5].
Myra, J. R. et al., Phys. Plasmas 13 (2006) 112502.
    Blob size scaling: δ_b ≈ (ρ_s L_∥)^(2/5) R^(1/5) — critical size at
    sheath-inertial transition.
Garcia, O. E. et al., Phys. Plasmas 19 (2012) 092306.
    PDF of SOL density fluctuations follows a gamma distribution;
    log-normal approximation valid for large intermittency parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

# Sheath-connected velocity prefactor — Krasheninnikov 2001, Eq. 5
_SHEATH_PREFACTOR: float = 2.0

# Critical blob size prefactor — Myra et al. 2006, Eq. 12
_CRITICAL_SIZE_PREFACTOR: float = 2.0

# Heat flux enthalpy factor (3/2 k_B T per particle)
_GAMMA_HEAT: float = 1.5


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _finite_1d_array(
    name: str,
    values: AnyFloatArray,
    *,
    nonnegative: bool = False,
    positive: bool = False,
    non_empty: bool = False,
) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if non_empty and arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} values must be finite")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} values must be positive")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    return arr


def _validate_population(population: BlobPopulation) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    sizes = _finite_1d_array("population sizes", population.sizes, positive=True)
    amplitudes = _finite_1d_array("population amplitudes", population.amplitudes, nonnegative=True)
    velocities = _finite_1d_array("population velocities", population.velocities, nonnegative=True)
    birth_times = _finite_1d_array("population birth times", population.birth_times, nonnegative=True)
    lengths = {sizes.size, amplitudes.size, velocities.size, birth_times.size}
    if len(lengths) != 1:
        raise ValueError("population arrays must have the same length")
    if birth_times.size > 1 and np.any(np.diff(birth_times) <= 0.0):
        raise ValueError("population birth times must be strictly increasing")
    return sizes, amplitudes, velocities, birth_times


class BlobDynamics:
    """Blob velocity and size in sheath-connected and inertial regimes.

    D'Ippolito et al. 2011, Phys. Plasmas 18, 060501 — two-regime picture:
      • sheath regime (small blobs):  v_b ∝ δ_b^{-1/2}
      • inertial regime (large blobs): v_b ∝ δ_b^{+1/2}
    """

    def __init__(self, R0: float, B0: float, Te_eV: float, Ti_eV: float, mi_amu: float = 2.0):
        R0 = _finite_scalar("R0", R0, positive=True)
        B0 = _finite_scalar("B0", B0, positive=True)
        Te_eV = _finite_scalar("Te_eV", Te_eV, nonnegative=True)
        Ti_eV = _finite_scalar("Ti_eV", Ti_eV, nonnegative=True)
        if Te_eV + Ti_eV <= 0.0:
            raise ValueError("Te_eV + Ti_eV must be positive")
        mi_amu = _finite_scalar("mi_amu", mi_amu, positive=True)
        self.R0 = R0
        self.B0 = B0
        self.Te = Te_eV
        self.Ti = Ti_eV
        self.mi = mi_amu * 1.67e-27  # [kg]
        self.e_charge = 1.602e-19  # [C]

        T_tot_J = (self.Te + self.Ti) * self.e_charge
        # Sound speed c_s = √(T_tot / m_i) — D'Ippolito et al. 2011, Eq. 2
        self.c_s = math.sqrt(T_tot_J / self.mi)
        # Ion sound Larmor radius ρ_s = c_s / Ω_i — Krasheninnikov 2001
        self.rho_s = self.c_s / (self.e_charge * self.B0 / self.mi)

    def critical_size(self, L_parallel: float) -> float:
        """Blob critical radius δ_b* at sheath–inertial transition [m].

        Myra et al. 2006, Phys. Plasmas 13, 112502, Eq. 12:
            δ_b* ≈ 2 ρ_s (L_∥ / (R ρ_s))^(1/5)
        """
        L_parallel = _finite_scalar("L_parallel", L_parallel, positive=True)
        return float(_CRITICAL_SIZE_PREFACTOR * self.rho_s * (L_parallel / (self.R0 * self.rho_s)) ** 0.2)

    def max_velocity(self, L_parallel: float) -> float:
        """Peak blob velocity at δ_b = δ_b* [m/s]."""
        delta_star = self.critical_size(L_parallel)
        return self.sheath_velocity(delta_star)

    def sheath_velocity(self, delta_b: float) -> float:
        """Radial blob velocity in the sheath-connected regime [m/s].

        Krasheninnikov 2001, Phys. Lett. A 283, 368, Eq. 5:
            v_b = 2 T_e / (e B R δ_b) × (c_s / Ω_i)
                ≡ 2 c_s ρ_s / (R √(δ_b / R))    [in the limit δ_b ≪ R]
        """
        delta_b = _finite_scalar("delta_b", delta_b, nonnegative=True)
        if delta_b == 0.0:
            return 0.0
        return _SHEATH_PREFACTOR * self.c_s * self.rho_s / (self.R0 * math.sqrt(delta_b / self.R0 + 1e-6))

    def inertial_velocity(self, delta_b: float) -> float:
        """Radial blob velocity in the inertia-limited regime [m/s].

        D'Ippolito et al. 2011, Phys. Plasmas 18, 060501, Eq. 4:
            v_b ∝ (δ_b / R)^{1/2} c_s
        """
        delta_b = _finite_scalar("delta_b", delta_b, positive=True)
        return self.c_s * math.sqrt(2.0 * delta_b / self.R0)

    def blob_velocity(self, delta_b: float, n_e: float, L_parallel: float) -> tuple[float, str]:
        """Select sheath or inertial velocity based on δ_b vs δ_b*.

        D'Ippolito et al. 2011, Phys. Plasmas 18, 060501, Fig. 2.
        """
        delta_b = _finite_scalar("delta_b", delta_b, positive=True)
        _finite_scalar("n_e", n_e, positive=True)
        L_parallel = _finite_scalar("L_parallel", L_parallel, positive=True)
        delta_star = self.critical_size(L_parallel)

        if delta_b < delta_star:
            v = self.sheath_velocity(delta_b)
            regime = "sheath"
        else:
            v = self.inertial_velocity(delta_b)
            regime = "inertial"

        return float(v), regime


@dataclass
class BlobPopulation:
    sizes: AnyFloatArray
    amplitudes: AnyFloatArray
    velocities: AnyFloatArray
    birth_times: AnyFloatArray


class BlobEnsemble:
    """Statistical ensemble of blobs with gamma-distributed amplitudes.

    Garcia et al. 2012, Phys. Plasmas 19, 092306: SOL density PDF is a
    gamma distribution; log-normal is the large-intermittency limit.
    Exponential waiting times produce a Poisson blob arrival process.
    """

    def __init__(self, dynamics: BlobDynamics, n_blobs: int = 1000):
        self.dynamics = dynamics
        if not isinstance(n_blobs, int) or isinstance(n_blobs, bool):
            raise ValueError("n_blobs must be an integer")
        if n_blobs < 0:
            raise ValueError("n_blobs must be non-negative")
        self.n_blobs = n_blobs

    def generate(
        self,
        delta_b_mean: float,
        delta_b_sigma: float,
        amplitude_mean: float,
        waiting_time_mean: float,
        rng: np.random.Generator,
    ) -> BlobPopulation:
        delta_b_mean = _finite_scalar("delta_b_mean", delta_b_mean, positive=True)
        delta_b_sigma = _finite_scalar("delta_b_sigma", delta_b_sigma, nonnegative=True)
        amplitude_mean = _finite_scalar("amplitude_mean", amplitude_mean, positive=True)
        waiting_time_mean = _finite_scalar("waiting_time_mean", waiting_time_mean, positive=True)
        # Log-normal amplitudes — Garcia et al. 2012, Eq. 7; σ_log ≈ 0.5
        _sigma_log: float = 0.5
        mu_amp = math.log(amplitude_mean) - 0.5 * _sigma_log**2
        amps = rng.lognormal(mu_amp, _sigma_log, self.n_blobs)

        sizes = rng.normal(delta_b_mean, delta_b_sigma, self.n_blobs)
        sizes = np.maximum(sizes, 1e-3)

        # Poisson arrival process: exponential inter-event times
        waits = rng.exponential(waiting_time_mean, self.n_blobs)
        births = np.cumsum(waits)

        vels = np.zeros(self.n_blobs)
        for i in range(self.n_blobs):
            v, _ = self.dynamics.blob_velocity(sizes[i], 1e19, L_parallel=10.0)
            vels[i] = v

        return BlobPopulation(sizes, amps, vels, births)

    def radial_flux(self, population: BlobPopulation) -> float:
        """Time-averaged blob particle flux Γ_blob [m^-2 s^-1].

        D'Ippolito et al. 2011, Phys. Plasmas 18, 060501, Eq. 12.
        """
        sizes, amplitudes, velocities, birth_times = _validate_population(population)
        if birth_times.size == 0:
            return 0.0
        tot_time = birth_times[-1] if self.n_blobs > 0 else 1.0
        if tot_time <= 0.0:
            raise ValueError("population birth times must span positive elapsed time")
        flux_sum = np.sum(amplitudes * velocities * sizes)
        return float(flux_sum / tot_time)

    def heat_flux(self, population: BlobPopulation, Te_eV: float) -> float:
        """Blob-driven heat flux q ~ (3/2) n T v [W m^-2]."""
        Te_eV = _finite_scalar("Te_eV", Te_eV, nonnegative=True)
        gamma = self.radial_flux(population)
        return gamma * Te_eV * 1.602e-19 * _GAMMA_HEAT


class SOLBlobProfile:
    @staticmethod
    def radial_density(r: AnyFloatArray, Gamma_blob: float, D_perp: float, lambda_n: float) -> FloatArray:
        """SOL density profile with blob-enhanced transport.

        Without blobs: n = n₀ exp(−r / λ_n).
        Blob transport broadens the profile via an effective λ.
        D'Ippolito et al. 2011, Phys. Plasmas 18, 060501, Sec. IV.
        """
        r = _finite_1d_array("r", r, nonnegative=True, non_empty=True)
        if r.size > 1 and np.any(np.diff(r) < 0.0):
            raise ValueError("r must be ordered from separatrix to wall")
        if r.size > 1 and np.any(np.diff(r) == 0.0):
            raise ValueError("r must be strictly ordered from separatrix to wall")
        lambda_n = _finite_scalar("lambda_n", lambda_n, positive=True)
        Gamma_blob = _finite_scalar("Gamma_blob", Gamma_blob, nonnegative=True)
        D_perp = _finite_scalar("D_perp", D_perp, nonnegative=True)
        if D_perp == 0:
            return np.asarray(np.exp(-r / lambda_n))

        blob_enhancement = 1.0 + Gamma_blob / (D_perp + 1e-6) * 1e-19
        eff_lambda = lambda_n * math.sqrt(blob_enhancement)

        return np.asarray(np.exp(-r / eff_lambda))

    @staticmethod
    def wall_flux(r_wall: float, Gamma_blob: float, lambda_n: float) -> float:
        r_wall = _finite_scalar("r_wall", r_wall, nonnegative=True)
        Gamma_blob = _finite_scalar("Gamma_blob", Gamma_blob, nonnegative=True)
        lambda_n = _finite_scalar("lambda_n", lambda_n, positive=True)
        eff_lambda = lambda_n * math.sqrt(1.0 + Gamma_blob * 1e-19)
        return float(Gamma_blob * math.exp(-r_wall / eff_lambda))


@dataclass
class BlobEvent:
    start_idx: int
    end_idx: int
    peak_amplitude: float
    duration: float
    size_estimate: float


def _validate_blob_event(event: BlobEvent, signal_size: int) -> None:
    if (
        not isinstance(event.start_idx, int)
        or isinstance(event.start_idx, bool)
        or not isinstance(event.end_idx, int)
        or isinstance(event.end_idx, bool)
    ):
        raise ValueError("event indices must be integers")
    if event.start_idx < 0 or event.end_idx > signal_size or event.end_idx <= event.start_idx:
        raise ValueError("event indices must satisfy 0 <= start_idx < end_idx <= len(signal)")
    _finite_scalar("event peak_amplitude", event.peak_amplitude, nonnegative=True)
    _finite_scalar("event duration", event.duration, positive=True)
    _finite_scalar("event size_estimate", event.size_estimate, positive=True)


class BlobDetector:
    """Threshold-crossing blob detector with conditional averaging.

    Garcia et al. 2012, Phys. Plasmas 19, 092306 — standard method for
    identifying intermittent transport events in probe signals.
    """

    def detect_blobs(self, signal: AnyFloatArray, dt: float = 1e-6, threshold: float = 2.5) -> list[BlobEvent]:
        signal = _finite_1d_array("signal", signal)
        dt = _finite_scalar("dt", dt, positive=True)
        threshold = _finite_scalar("threshold", threshold, positive=True)
        mean_sig = np.mean(signal)
        std_sig = np.std(signal)

        if std_sig == 0:
            return []

        norm_sig = (signal - mean_sig) / std_sig

        events: list[BlobEvent] = []
        in_blob = False
        start = 0
        peak = 0.0

        for i, val in enumerate(norm_sig):
            if val > threshold and not in_blob:
                in_blob = True
                start = i
                peak = val
            elif val > threshold and in_blob:
                peak = max(peak, val)
            elif val <= 0.0 and in_blob:
                in_blob = False
                dur = (i - start) * dt
                size = dur * 1000.0
                events.append(BlobEvent(start, i, float(peak * std_sig), dur, size))

        if in_blob:
            end = len(norm_sig)
            dur = (end - start) * dt
            size = dur * 1000.0
            events.append(BlobEvent(start, end, float(peak * std_sig), dur, size))

        return events

    def conditional_average(self, signal: AnyFloatArray, events: list[BlobEvent], window: int = 50) -> FloatArray:
        signal = _finite_1d_array("signal", signal)
        if window < 0:
            raise ValueError("window must be non-negative")
        if not events:
            return np.zeros(2 * window + 1)

        cond_avg = np.zeros(2 * window + 1)
        count = 0

        for ev in events:
            _validate_blob_event(ev, len(signal))
            center = ev.start_idx + (ev.end_idx - ev.start_idx) // 2
            if center - window >= 0 and center + window < len(signal):
                cond_avg += signal[center - window : center + window + 1]
                count += 1

        if count > 0:
            cond_avg /= count

        return cond_avg
