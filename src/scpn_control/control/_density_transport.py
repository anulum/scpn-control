# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Particle transport and particle-source model

"""Particle transport and particle-source model."""

from __future__ import annotations

import math

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.pellet_injection import PelletParams, PelletTrajectory


class ParticleTransportModel:
    """Radial particle-transport plant for the density-control loop.

    Integrates the flux-surface-averaged continuity equation
    ``∂_t n = -V'^{-1} ∂_ρ (V' Γ) + S`` with anomalous flux
    ``Γ = -D ∂_r n + V_pinch n`` on a uniform ``rho = r/a`` grid, using an
    explicit forward-Euler step clamped to the diffusion CFL limit. Source and
    sink terms (gas puff, pellet, NBI, recycling, cryopump) are supplied by the
    dedicated methods. A circular cross-section sets the volume elements.

    Parameters
    ----------
    n_rho
        Number of radial grid points; must be at least 2.
    R0
        Major radius in metres; must be finite and positive.
    a
        Minor radius in metres; must be finite and positive.
    """

    def __init__(self, n_rho: int = 50, R0: float = 6.2, a: float = 2.0):
        if n_rho < 2:
            raise ValueError("n_rho must be at least 2 for a physical radial grid.")
        if not math.isfinite(R0) or R0 <= 0.0:
            raise ValueError("R0 must be a finite positive physical major radius.")
        if not math.isfinite(a) or a <= 0.0:
            raise ValueError("a must be a finite positive physical minor radius.")
        self.n_rho = n_rho
        self.R0 = R0
        self.a = a
        self.rho = np.linspace(0.0, 1.0, n_rho)
        self.drho = self.rho[1] - self.rho[0]

        # Anomalous diffusivity and pinch velocity: default ITER-like values.
        # D ~ 1 m²/s, V_pinch ~ -0.1 m/s (inward Ware pinch order of magnitude).
        self.D = np.ones(n_rho) * 1.0  # m²/s
        self.V_pinch = -np.ones(n_rho) * 0.1  # m/s, inward

        # Bounded non-facility circular cross-section volume elements.
        self.V = 2.0 * np.pi**2 * self.R0 * (self.a * self.rho) ** 2
        self.V_prime = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho

    def set_transport(self, D: AnyFloatArray, V_pinch: AnyFloatArray) -> None:
        """Override the diffusivity and pinch-velocity radial profiles.

        Parameters
        ----------
        D
            Particle diffusivity in m²/s, shape ``(n_rho,)``; finite and
            non-negative.
        V_pinch
            Pinch velocity in m/s, shape ``(n_rho,)`` (negative is inward);
            finite.

        Raises
        ------
        ValueError
            If a profile has the wrong shape, is non-finite, or ``D`` is
            negative.
        """
        D_arr = self._validate_profile(D, "D")
        if np.any(D_arr < 0.0):
            raise ValueError("D must be non-negative.")
        self.D = D_arr
        self.V_pinch = self._validate_profile(V_pinch, "V_pinch")

    def _validate_profile(self, values: AnyFloatArray, name: str) -> FloatArray:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (self.n_rho,):
            raise ValueError(f"{name} must have shape ({self.n_rho},).")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")
        return arr

    def gas_puff_source(self, rate: float, penetration_depth: float = 0.03) -> FloatArray:
        """Particles/s. Edge-localised source from gas injection.

        Pacher et al. 2007, Nucl. Fusion 47, 469: ITER gas-injection modelling
        places the effective source within ~3% of the minor radius from the wall.
        """
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("Gas puff rate must be finite and non-negative.")
        if not math.isfinite(penetration_depth) or penetration_depth <= 0.0:
            raise ValueError("Gas puff penetration_depth must be finite and positive.")
        decay = np.exp(-(1.0 - self.rho) / penetration_depth)
        decay /= np.sum(decay * self.V_prime * self.drho) + 1e-10
        return np.asarray(rate * decay)

    def pellet_source(
        self,
        speed_ms: float,
        radius_mm: float,
        launch_angle_deg: float = 0.0,
        *,
        ne_profile: AnyFloatArray | None = None,
        Te_eV_profile: AnyFloatArray | None = None,
        B0_T: float = 5.3,
        injection_side: str = "HFS",
    ) -> FloatArray:
        """NGS trajectory deposition profile from a single pellet.

        The deposition path is delegated to :class:`PelletTrajectory`, which
        integrates Parks-Turnbull neutral-gas-shielding ablation along the
        radial pellet trajectory and applies the repository drift correction.
        """
        if not math.isfinite(radius_mm):
            raise ValueError("Pellet radius_mm must be finite.")
        if not math.isfinite(launch_angle_deg):
            raise ValueError("Pellet launch_angle_deg must be finite.")
        if radius_mm <= 0.0:
            return np.zeros(self.n_rho)
        if not math.isfinite(speed_ms) or speed_ms <= 0.0:
            raise ValueError("Pellet speed_ms must be finite and positive for non-zero pellets.")
        if not math.isfinite(B0_T) or B0_T <= 0.0:
            raise ValueError("Pellet B0_T must be finite and positive.")
        if injection_side not in {"HFS", "LFS"}:
            raise ValueError("Pellet injection_side must be 'HFS' or 'LFS'.")

        ne_m3 = (
            self._default_density_profile() if ne_profile is None else self._validate_profile(ne_profile, "ne_profile")
        )
        if np.any(ne_m3 <= 0.0):
            raise ValueError("Pellet ne_profile must be positive.")
        Te_eV = (
            self._default_temperature_profile()
            if Te_eV_profile is None
            else self._validate_profile(Te_eV_profile, "Te_eV_profile")
        )
        if np.any(Te_eV < 0.0):
            raise ValueError("Pellet Te_eV_profile must be non-negative.")

        trajectory = PelletTrajectory(
            PelletParams(
                r_p_mm=radius_mm,
                v_p_m_s=speed_ms,
                injection_side=injection_side,
                injection_angle_deg=launch_angle_deg,
            ),
            R0=self.R0,
            a=self.a,
            B0=B0_T,
        )
        result = trajectory.simulate(self.rho, ne_m3 / 1e19, Te_eV)
        return np.asarray(result.deposition_profile, dtype=float)

    def _default_density_profile(self) -> FloatArray:
        """ITER-like positive density profile used only when no measurement is supplied."""
        return np.asarray(np.linspace(1.1e20, 0.7e20, self.n_rho), dtype=float)

    def _default_temperature_profile(self) -> FloatArray:
        """ITER-like electron-temperature profile used only when no measurement is supplied."""
        return np.asarray(np.linspace(8000.0, 800.0, self.n_rho), dtype=float)

    def nbi_source(self, beam_energy_keV: float, power_MW: float) -> FloatArray:
        """Core-peaked particle source from neutral beam injection."""
        if not math.isfinite(beam_energy_keV) or beam_energy_keV <= 0.0:
            raise ValueError("NBI beam_energy_keV must be finite and positive.")
        if not math.isfinite(power_MW) or power_MW < 0.0:
            raise ValueError("NBI power_MW must be finite and non-negative.")
        if power_MW <= 0.0:
            return np.zeros(self.n_rho)

        I_beam = power_MW * 1e6 / (beam_energy_keV * 1e3)
        rate = I_beam / 1.6e-19  # 1.6×10^-19 C/particle (elementary charge)

        dep = np.exp(-((self.rho - 0.3) ** 2) / (0.3**2))
        dep /= np.sum(dep * self.V_prime * self.drho) + 1e-10
        return np.asarray(rate * dep)

    def cryopump_sink(self, pump_speed: float, ne_edge: float) -> FloatArray:
        """Edge particle removal from cryopump."""
        if not math.isfinite(pump_speed) or pump_speed < 0.0:
            raise ValueError("Cryopump pump_speed must be finite and non-negative.")
        if not math.isfinite(ne_edge) or ne_edge < 0.0:
            raise ValueError("Cryopump ne_edge must be finite and non-negative.")
        sink = np.zeros(self.n_rho)
        sink[-1] = pump_speed * ne_edge / (self.V_prime[-1] * self.drho + 1e-10)
        return sink

    def recycling_source(self, outflux: float, recycling_coeff: float = 0.97) -> FloatArray:
        """
        Recycling_coeff = 0.97 is the standard ITER assumption for a metal wall.

        ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §4.2.
        """
        if not math.isfinite(outflux) or outflux < 0.0:
            raise ValueError("Recycling outflux must be finite and non-negative.")
        if not math.isfinite(recycling_coeff) or not 0.0 <= recycling_coeff <= 1.0:
            raise ValueError("Recycling coefficient must be finite and between 0 and 1.")
        return self.gas_puff_source(outflux * recycling_coeff, penetration_depth=0.02)

    def step(self, ne: AnyFloatArray, sources: AnyFloatArray, dt: float) -> FloatArray:
        """Advance the density profile by one CFL-limited diffusion step.

        Parameters
        ----------
        ne
            Current electron-density profile in m⁻³, shape ``(n_rho,)``.
        sources
            Net particle source profile in m⁻³ s⁻¹, shape ``(n_rho,)``.
        dt
            Requested time step in seconds; clamped to the explicit-diffusion
            CFL limit ``(drho a)² / (2 D_max)`` when larger.

        Returns
        -------
        FloatArray
            Updated density profile in m⁻³, floored at 1e16 m⁻³, shape
            ``(n_rho,)``.

        Raises
        ------
        ValueError
            If profiles have the wrong shape or are negative, or ``dt`` is not finite and
            positive.
        """
        ne_arr = self._validate_profile(ne, "ne")
        sources_arr = self._validate_profile(sources, "sources")
        if np.any(ne_arr < 0.0):
            raise ValueError("ne must be non-negative.")
        if np.any(sources_arr < 0.0):
            raise ValueError("sources must be non-negative.")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")

        # Explicit forward-Euler diffusion — CFL stability: dt < drho² / (2 D_max).
        D_max = np.max(self.D)
        if D_max > 0.0:
            dt_cfl = (self.drho * self.a) ** 2 / (2.0 * D_max)
            if dt > dt_cfl:
                dt = dt_cfl

        flux = np.zeros(self.n_rho + 1)
        for i in range(1, self.n_rho):
            grad_n = (ne_arr[i] - ne_arr[i - 1]) / self.drho
            n_face = 0.5 * (ne_arr[i] + ne_arr[i - 1])
            D_face = 0.5 * (self.D[i] + self.D[i - 1])
            V_face = 0.5 * (self.V_pinch[i] + self.V_pinch[i - 1])
            flux[i] = -D_face * grad_n / self.a + V_face * n_face

        flux[0] = 0.0
        flux[-1] = -self.D[-1] * (0.0 - ne_arr[-1]) / self.drho / self.a + self.V_pinch[-1] * ne_arr[-1]

        dne_dt = np.zeros(self.n_rho)
        for i in range(self.n_rho):
            Vp = max(self.V_prime[i], 1e-6)
            Vp_plus = self.V_prime[i] if i == self.n_rho - 1 else 0.5 * (self.V_prime[i] + self.V_prime[i + 1])
            Vp_minus = 0.0 if i == 0 else 0.5 * (self.V_prime[i] + self.V_prime[i - 1])
            div_flux = (Vp_plus * flux[i + 1] - Vp_minus * flux[i]) / (Vp * self.drho * self.a)
            dne_dt[i] = -div_flux + sources_arr[i]

        ne_new = ne_arr + dne_dt * dt
        return np.asarray(np.maximum(ne_new, 1e16))
