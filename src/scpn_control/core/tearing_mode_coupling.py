# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tearing-Mode Coupling Physics
"""
Coupled tearing mode dynamics including Chirikov overlap and inter-mode coupling.

Key references
--------------
Rutherford 1973     : P.H. Rutherford, Phys. Fluids 16, 1903 (1973).
                      [classical tearing: dw/dt ∝ Δ']
La Haye 2006        : R.J. La Haye, Phys. Plasmas 13, 055501 (2006).
                      [MRE coefficient fits a1 = 6.35 for bootstrap drive]
La Haye & Buttery   : R.J. La Haye & R.J. Buttery, Phys. Plasmas 16, 022107
2009                  (2009). [cross-mode coupling coefficient Eq. 8]
Chirikov 1979       : B.V. Chirikov, Phys. Rep. 52, 263 (1979).
                      [overlap criterion σ = (w1 + w2)/(2 Δr) > 1 → stochastic,
                      Eq. 3.1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from scpn_control.core.sawtooth import SawtoothCycler

# MRE bootstrap drive coefficient.
# La Haye 2006, Phys. Plasmas 13, 055501, Table I.
_A1: float = 6.35

_MU_0: float = 4.0 * np.pi * 1e-7  # H m⁻¹ (CODATA 2018)


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _mode_tuple(name: str, mode: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(mode, tuple):
        raise ValueError(f"{name} must be an (m, n) tuple")
    if len(mode) != 2:
        raise ValueError(f"{name} must be an (m, n) tuple")
    m, n = mode
    if isinstance(m, bool) or isinstance(n, bool) or int(m) != m or int(n) != n or m <= 0 or n <= 0:
        raise ValueError(f"{name} mode numbers must be positive integers")
    return int(m), int(n)


def _profile_axis(name: str, values: AnyFloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a one-dimensional non-empty scan axis")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if arr.size > 1 and np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


@dataclass
class CoupledResult:
    w1_trace: AnyFloatArray
    w2_trace: AnyFloatArray
    chirikov_trace: AnyFloatArray
    overlap_time: float
    disruption: bool


class ChirikovOverlap:
    """Overlap criterion for magnetic island stochasticity.

    σ = (w₁ + w₂) / (2 Δr)

    σ > 1 → overlapping separatrices → field-line stochasticity.
    Chirikov 1979, Phys. Rep. 52, 263, Eq. 3.1.
    """

    @staticmethod
    def parameter(w1: float, w2: float, delta_r: float) -> float:
        """Chirikov overlap parameter σ; Chirikov 1979, Eq. 3.1."""
        w1 = _finite_scalar("w1", w1, nonnegative=True)
        w2 = _finite_scalar("w2", w2, nonnegative=True)
        delta_r = _finite_scalar("delta_r", delta_r, positive=True)
        return (w1 + w2) / (2.0 * delta_r)

    @staticmethod
    def is_stochastic(sigma: float) -> bool:
        """True when σ > 1 (Chirikov 1979, Eq. 3.1)."""
        sigma = _finite_scalar("sigma", sigma, nonnegative=True)
        return sigma > 1.0

    @staticmethod
    def stochastic_region_width(w1: float, w2: float, delta_r: float) -> float:
        """Radial extent of stochastic region when σ > 1."""
        sigma = ChirikovOverlap.parameter(w1, w2, delta_r)
        if sigma > 1.0:
            return delta_r + w1 / 2.0 + w2 / 2.0
        return 0.0


class CoupledTearingModes:
    """Modified Rutherford Equation for two toroidally coupled tearing modes.

    MRE for mode k (k = 1, 2):
        dw_k/dt = (r_sk / τ_Rk) [r_sk Δ'_k + a1 (j_bs/j_φ)(r_sk/w_k)
                                  × w_k²/(w_k² + w_d²) + C_kj w_j² / (r_s1 r_s2)]

    Bootstrap drive coefficient a1 = 6.35: La Haye 2006, Phys. Plasmas 13,
    055501, Table I.
    Cross-mode coupling coefficient C_kj: La Haye & Buttery 2009, Phys.
    Plasmas 16, 022107, Eq. 8.
    Classical tearing stability Δ': Rutherford 1973, Phys. Fluids 16, 1903.
    """

    def __init__(
        self,
        mode1: tuple[int, int],
        mode2: tuple[int, int],
        r_s1: float,
        r_s2: float,
        a: float,
        R0: float,
        B0: float,
    ):
        self.m1, self.n1 = _mode_tuple("mode1", mode1)
        self.m2, self.n2 = _mode_tuple("mode2", mode2)
        self.r_s1 = _finite_scalar("r_s1", r_s1, positive=True)
        self.r_s2 = _finite_scalar("r_s2", r_s2, positive=True)
        self.a = _finite_scalar("a", a, positive=True)
        self.R0 = _finite_scalar("R0", R0, positive=True)
        self.B0 = _finite_scalar("B0", B0, positive=True)
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        if self.r_s1 >= self.a or self.r_s2 >= self.a:
            raise ValueError("rational surfaces must lie inside the minor radius")
        if self.r_s1 == self.r_s2:
            raise ValueError("rational surfaces must be radially separated")

        self.delta_r = abs(self.r_s1 - self.r_s2)

    def coupling_coefficient(self, m1: int, n1: int, m2: int, n2: int) -> float:
        """Cross-mode coupling coefficient C₁₂.

        La Haye & Buttery 2009, Phys. Plasmas 16, 022107, Eq. 8:
            C₁₂ ≈ 0.5 × (a / R₀)
        This captures the leading-order inverse-aspect-ratio toroidal coupling
        between the 3/2 and 2/1 modes via the n=1 sideband.  Modes with
        different n do not couple directly at this order.
        """
        _mode_tuple("mode1", (m1, n1))
        _mode_tuple("mode2", (m2, n2))
        return 0.5 * (self.a / self.R0)

    def delta_prime_1(self, w1: float, w2: float) -> float:
        """Effective Δ' for mode 1, modified by presence of mode 2.

        Base linear Δ'₀ = −2 m₁ / r_s1 (classically stable).
        Rutherford 1973, Phys. Fluids 16, 1903, §III.
        Nonlinear cross-modification: La Haye & Buttery 2009, §II.
        """
        w2 = _finite_scalar("w2", w2, nonnegative=True)
        dp0 = -2.0 * self.m1 / self.r_s1
        return dp0 + 0.5 * w2 / self.a

    def delta_prime_2(self, w1: float, w2: float) -> float:
        """Effective Δ' for mode 2; same references as delta_prime_1."""
        w1 = _finite_scalar("w1", w1, nonnegative=True)
        dp0 = -2.0 * self.m2 / self.r_s2
        return dp0 + 0.5 * w1 / self.a

    def evolve(
        self,
        w1_0: float,
        w2_0: float,
        j_bs: float,
        j_phi: float,
        eta: float,
        dt: float,
        n_steps: int,
        seed_time: float = -1.0,
        seed_amplitude: float = 0.0,
    ) -> CoupledResult:
        """Integrate coupled MRE for both modes.

        Bootstrap term: a1 (j_bs/j_φ) (r_s/w) w²/(w²+w_d²)
            — La Haye 2006, Phys. Plasmas 13, 055501, Eq. 5; a1 = 6.35 Table I.
        Coupling term: C₁₂ w₂²/(r_s1 r_s2)
            — La Haye & Buttery 2009, Phys. Plasmas 16, 022107, Eq. 8.
        Stochastic criterion: σ = (w1+w2)/(2Δr) > 1
            — Chirikov 1979, Phys. Rep. 52, 263, Eq. 3.1.
        """
        w1_0 = _finite_scalar("w1_0", w1_0, nonnegative=True)
        w2_0 = _finite_scalar("w2_0", w2_0, nonnegative=True)
        j_bs = _finite_scalar("j_bs", j_bs, nonnegative=True)
        j_phi = _finite_scalar("j_phi", j_phi, positive=True)
        eta = _finite_scalar("eta", eta, positive=True)
        dt = _finite_scalar("dt", dt, positive=True)
        if j_bs > j_phi:
            raise ValueError("bootstrap current density must not exceed total current density")
        if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        seed_time = _finite_scalar("seed_time", seed_time)
        if seed_time < 0.0 and seed_time != -1.0:
            raise ValueError("seed_time must be -1 or non-negative")
        seed_amplitude = _finite_scalar("seed_amplitude", seed_amplitude, nonnegative=True)

        w1 = max(w1_0, 1e-6)
        w2 = max(w2_0, 1e-6)

        w1_trace = np.zeros(n_steps)
        w2_trace = np.zeros(n_steps)
        chir_trace = np.zeros(n_steps)

        # Resistive diffusion timescales τ_R = μ₀ r_s² / η
        # (Rutherford 1973, Phys. Fluids 16, 1903, §II)
        tau_R1 = _MU_0 * self.r_s1**2 / eta
        tau_R2 = _MU_0 * self.r_s2**2 / eta

        C12 = self.coupling_coefficient(self.m1, self.n1, self.m2, self.n2)
        C21 = self.coupling_coefficient(self.m2, self.n2, self.m1, self.n1)

        j_ratio = j_bs / j_phi

        # Small seed width w_d to regularise bootstrap term near w → 0.
        # La Haye 2006, Eq. 5 uses w_d ~ ion banana width; set to 1 mm here.
        w_d2 = 1e-6  # w_d = 1e-3 m, so w_d² = 1e-6 m²

        overlap_time = -1.0
        disruption = False

        for i in range(n_steps):
            t = i * dt

            if seed_time > 0 and abs(t - seed_time) <= dt:
                w1 = max(w1, seed_amplitude)

            # MRE for mode 1 (3/2)
            # La Haye 2006, Eq. 5; La Haye & Buttery 2009, Eq. 8
            dp1 = self.delta_prime_1(w1, w2)
            bs_term1 = _A1 * j_ratio * (self.r_s1 / w1) * (w1**2 / (w1**2 + w_d2))
            c_term1 = C12 * (w2**2) / (self.r_s1 * self.r_s2)
            dw1_dt = (self.r_s1 / tau_R1) * (self.r_s1 * dp1 + bs_term1 + c_term1)

            # MRE for mode 2 (2/1)
            dp2 = self.delta_prime_2(w1, w2)
            bs_term2 = _A1 * j_ratio * (self.r_s2 / w2) * (w2**2 / (w2**2 + w_d2))
            c_term2 = C21 * (w1**2) / (self.r_s1 * self.r_s2)
            dw2_dt = (self.r_s2 / tau_R2) * (self.r_s2 * dp2 + bs_term2 + c_term2)

            w1 += dw1_dt * dt
            w2 += dw2_dt * dt

            w1 = min(max(w1, 1e-6), 2.0 * self.a)
            w2 = min(max(w2, 1e-6), 2.0 * self.a)

            w1_trace[i] = w1
            w2_trace[i] = w2

            sigma = ChirikovOverlap.parameter(w1, w2, self.delta_r)
            chir_trace[i] = sigma

            if sigma > 1.0 and not disruption:
                disruption = True
                overlap_time = t

        return CoupledResult(w1_trace, w2_trace, chir_trace, overlap_time, disruption)


class SawtoothNTMSeeding:
    def __init__(self, sawtooth_cycler: SawtoothCycler | None):
        self.st = sawtooth_cycler

    def seed_amplitude(self, crash_energy_MJ: float, r_s: float) -> float:  # noqa: ARG002
        """Seed island width ~ sqrt(δW_sawtooth).

        Scaling from Porcelli et al. 1996, Plasma Phys. Control. Fusion 38,
        2163, §4: w_seed ∝ √(δW) where δW is the sawtooth crash free energy.
        """
        crash_energy_MJ = _finite_scalar("crash_energy_MJ", crash_energy_MJ, nonnegative=True)
        _finite_scalar("r_s", r_s, positive=True)
        return 0.05 * math.sqrt(crash_energy_MJ)

    def seed_probability(self, crash_energy: float, threshold: float) -> float:
        crash_energy = _finite_scalar("crash_energy", crash_energy, nonnegative=True)
        threshold = _finite_scalar("threshold", threshold, nonnegative=True)
        if crash_energy < threshold:
            return 0.0
        prob = 1.0 - math.exp(-(crash_energy - threshold))
        return float(np.clip(prob, 0.0, 1.0))


@dataclass
class DisruptionPath:
    warning_time_ms: float
    avoidable: bool


class DisruptionTriggerAssessment:
    def __init__(self, coupled: CoupledTearingModes):
        self.coupled = coupled

    def run_scenario(
        self,
        j_bs: float,
        j_phi: float,
        omega_phi: float,
        seed_energy: float,  # noqa: ARG002
    ) -> DisruptionPath:
        j_bs = _finite_scalar("j_bs", j_bs, nonnegative=True)
        j_phi = _finite_scalar("j_phi", j_phi, positive=True)
        _finite_scalar("omega_phi", omega_phi, positive=True)
        seed_energy = _finite_scalar("seed_energy", seed_energy, nonnegative=True)
        st_seed = SawtoothNTMSeeding(None)
        amp = st_seed.seed_amplitude(seed_energy, self.coupled.r_s1)

        eta = 1e-7
        dt = 0.01
        n_steps = 1000

        res = self.coupled.evolve(1e-6, 1e-6, j_bs, j_phi, eta, dt, n_steps, seed_time=0.1, seed_amplitude=amp)

        if not res.disruption:
            return DisruptionPath(-1.0, True)

        # Check if zeroing the 3/2 bootstrap drive (ideal ECCD) prevents disruption.
        res_controlled = self.coupled.evolve(
            1e-6, 1e-6, 0.0, j_phi, eta, dt, n_steps, seed_time=0.1, seed_amplitude=amp
        )
        avoidable = not res_controlled.disruption

        return DisruptionPath(res.overlap_time * 1000.0, avoidable)


class TearingModeStabilityMap:
    def scan_beta_li(self, beta_N_range: AnyFloatArray, li_range: AnyFloatArray) -> FloatArray:
        """Heuristic stability map: beta_N × l_i > 3.0 → unstable.

        Rough guideline from ITER Physics Basis Ch.3 (Nucl. Fusion 39, 2175,
        1999): combined beta–internal inductance drives NTM onset at high
        beta_N l_i product.
        """
        beta_N_range = _profile_axis("beta_N_range", beta_N_range)
        li_range = _profile_axis("li_range", li_range)

        res = np.zeros((len(beta_N_range), len(li_range)))

        for i, b in enumerate(beta_N_range):
            for j, li in enumerate(li_range):
                res[i, j] = -1 if b * max(li, 0.1) > 3.0 else 1

        return res
