# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density-control plant and particle-source model
"""Density-control plant, estimator, fuelling actuators, and optimisation utilities."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.pellet_injection import PelletParams, PelletTrajectory

# Greenwald density limit: n_GW = I_p / (π a²) [10^20 m^-3]
# Greenwald 2002, PPCF 44, R27, Eq. 1.
# ITER operational limit: n/n_GW < 0.85 — ITER Physics Basis 1999, Nucl. Fusion 39, 2175, §2.3.
_GW_ITER_SAFETY_MARGIN = 0.85  # ITER Physics Basis 1999, §2.3

# Greenwald fraction at which the controller switches to maximum pumping (hard limit).
_GW_PUMP_THRESHOLD = 0.95  # 5% headroom below disruption risk
_DENSITY_CLAIM_SCHEMA_VERSION = 1
_KALMAN_GEOMETRY_ALIAS_REMOVAL_VERSION = "0.25.0"
_FACILITY_DENSITY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "measured_discharge", "external_particle_balance", "facility_replay"}
)
_BOUNDED_DENSITY_REFERENCE_SOURCES = frozenset({"synthetic_regression_reference", *_FACILITY_DENSITY_REFERENCE_SOURCES})


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


@dataclass(frozen=True)
class DensityControlClaimEvidence:
    """Serialisable provenance and reference-comparison evidence for density-control claims."""

    schema_version: int
    source: str
    source_id: str
    geometry_source: str
    transport_source: str
    actuator_source: str
    diagnostic_source: str
    model_id: str
    n_rho: int
    R0_m: float
    a_m: float
    dt_requested_s: float
    dt_cfl_s: float | None
    cfl_limited: bool
    greenwald_fraction: float
    below_iter_greenwald_margin: bool
    gas_puff_rate: float
    pellet_frequency: float
    pellet_speed: float
    cryopump_speed: float
    total_source_particles_per_s: float
    particle_inventory_delta: float
    greenwald_fraction_abs_error: float | None
    inventory_delta_relative_error: float | None
    greenwald_fraction_abs_tolerance: float
    inventory_delta_relative_tolerance: float
    facility_density_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _relative_error(name: str, observed: float, reference: float | None) -> float | None:
    if reference is None:
        return None
    ref = DensityController._validate_non_negative_scalar(reference, name)
    obs = DensityController._validate_non_negative_scalar(observed, f"observed_{name}")
    return abs(obs - ref) / max(abs(ref), 1e-30)


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


def density_control_claim_evidence(
    model: ParticleTransportModel,
    controller: DensityController,
    *,
    source: str,
    source_id: str,
    geometry_source: str,
    transport_source: str,
    actuator_source: str,
    diagnostic_source: str,
    ne_before: AnyFloatArray,
    ne_after: AnyFloatArray,
    sources: AnyFloatArray,
    command: ActuatorCommand,
    dt_requested_s: float,
    model_id: str = "bounded_density_control",
    reference_greenwald_fraction: float | None = None,
    reference_inventory_delta: float | None = None,
    greenwald_fraction_abs_tolerance: float = 0.02,
    inventory_delta_relative_tolerance: float = 0.05,
) -> DensityControlClaimEvidence:
    """Build fail-closed density-control evidence for bounded or facility claims."""
    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_DENSITY_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_DENSITY_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")

    ne_before_arr = controller._validate_density_profile(ne_before, "ne_before")
    ne_after_arr = controller._validate_density_profile(ne_after, "ne_after")
    sources_arr = model._validate_profile(sources, "sources")
    if np.any(sources_arr < 0.0):
        raise ValueError("sources must be non-negative")

    dt_requested = DensityController._validate_positive_scalar(dt_requested_s, "dt_requested_s")
    greenwald_tol = DensityController._validate_positive_scalar(
        greenwald_fraction_abs_tolerance, "greenwald_fraction_abs_tolerance"
    )
    inventory_tol = DensityController._validate_positive_scalar(
        inventory_delta_relative_tolerance, "inventory_delta_relative_tolerance"
    )

    d_max = float(np.max(model.D))
    dt_cfl = None
    cfl_limited = False
    if d_max > 0.0:
        dt_cfl = float((model.drho * model.a) ** 2 / (2.0 * d_max))
        cfl_limited = dt_requested > dt_cfl

    greenwald_fraction = controller.greenwald_fraction(
        ne_after_arr, I_p_MA=controller.n_GW * math.pi * model.a**2 / 1e20, a=model.a
    )
    below_margin = controller.below_greenwald_safety_margin(ne_after_arr)
    volume_weights = model.V_prime * model.drho
    total_source = float(np.sum(sources_arr * volume_weights))
    inventory_delta = float(np.sum((ne_after_arr - ne_before_arr) * volume_weights))
    greenwald_error = None
    if reference_greenwald_fraction is not None:
        ref_gw = DensityController._validate_non_negative_scalar(
            reference_greenwald_fraction, "reference_greenwald_fraction"
        )
        greenwald_error = abs(greenwald_fraction - ref_gw)
    inventory_error = _relative_error(
        "reference_inventory_delta",
        abs(inventory_delta),
        None if reference_inventory_delta is None else abs(reference_inventory_delta),
    )

    references_pass = (
        greenwald_error is not None
        and inventory_error is not None
        and greenwald_error <= greenwald_tol
        and inventory_error <= inventory_tol
        and below_margin
    )
    facility_claim_allowed = source_clean in _FACILITY_DENSITY_REFERENCE_SOURCES and references_pass
    claim_status = (
        "facility_density_reference_matched" if facility_claim_allowed else "bounded_density_control_evidence"
    )

    return DensityControlClaimEvidence(
        schema_version=_DENSITY_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        geometry_source=_non_empty_text("geometry_source", geometry_source),
        transport_source=_non_empty_text("transport_source", transport_source),
        actuator_source=_non_empty_text("actuator_source", actuator_source),
        diagnostic_source=_non_empty_text("diagnostic_source", diagnostic_source),
        model_id=_non_empty_text("model_id", model_id),
        n_rho=model.n_rho,
        R0_m=float(model.R0),
        a_m=float(model.a),
        dt_requested_s=dt_requested,
        dt_cfl_s=dt_cfl,
        cfl_limited=bool(cfl_limited),
        greenwald_fraction=float(greenwald_fraction),
        below_iter_greenwald_margin=bool(below_margin),
        gas_puff_rate=DensityController._validate_non_negative_scalar(command.gas_puff_rate, "gas_puff_rate"),
        pellet_frequency=DensityController._validate_non_negative_scalar(command.pellet_freq, "pellet_freq"),
        pellet_speed=DensityController._validate_non_negative_scalar(command.pellet_speed, "pellet_speed"),
        cryopump_speed=DensityController._validate_non_negative_scalar(command.cryo_pump_speed, "cryo_pump_speed"),
        total_source_particles_per_s=total_source,
        particle_inventory_delta=inventory_delta,
        greenwald_fraction_abs_error=greenwald_error,
        inventory_delta_relative_error=inventory_error,
        greenwald_fraction_abs_tolerance=greenwald_tol,
        inventory_delta_relative_tolerance=inventory_tol,
        facility_density_claim_allowed=bool(facility_claim_allowed),
        claim_status=claim_status,
    )


def assert_density_control_facility_claim_admissible(evidence: DensityControlClaimEvidence) -> None:
    """Raise when density-control evidence is insufficient for a facility claim."""
    if not evidence.facility_density_claim_allowed:
        raise ValueError("facility density-control claim requires matched Greenwald and inventory references")


def save_density_control_claim_evidence(evidence: DensityControlClaimEvidence, path: str | Path) -> None:
    """Persist density-control claim evidence as deterministic JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class KalmanDensityEstimator:
    """Linear Kalman filter for bounded interferometric density reconstruction.

    The state contains piecewise-constant electron densities in uniform annular
    shells over normalised minor radius ``rho in [0, 1]``. The measurement model
    integrates each shell along a straight chord through a circular,
    axisymmetric plasma. Consequently, the projection matrix has units of
    metres and maps density in m⁻³ to line-integrated density in m⁻².

    This is a bounded linear-Gaussian model. It does not model shaped magnetic
    surfaces, ray refraction, diagnostic calibration, or a constrained
    non-negative posterior.

    Parameters
    ----------
    n_rho
        Number of radial state grid points.
    n_chords
        Number of interferometer chords (measurements).
    minor_radius_m
        Circular plasma minor radius in metres.

    Raises
    ------
    ValueError
        If a dimension is not a positive integer or the minor radius is not
        finite and positive.
    """

    def __init__(self, n_rho: int, n_chords: int = 8, minor_radius_m: float = 1.0):
        if isinstance(n_rho, bool) or not isinstance(n_rho, int) or n_rho < 1:
            raise ValueError("n_rho must be a positive integer.")
        if isinstance(n_chords, bool) or not isinstance(n_chords, int) or n_chords < 1:
            raise ValueError("n_chords must be a positive integer.")
        if not math.isfinite(minor_radius_m) or minor_radius_m <= 0.0:
            raise ValueError("minor_radius_m must be finite and positive.")
        self.n_rho = n_rho
        self.n_chords = n_chords
        self.minor_radius_m = minor_radius_m
        self.x: FloatArray = np.zeros(n_rho)
        self.P = np.eye(n_rho) * 1e38
        self.Q = np.eye(n_rho) * 1e36  # density-state process covariance per second
        self.R = np.eye(n_chords) * 1e34  # line-integrated measurement covariance

    def measurement_matrix(
        self,
        chord_impacts: AnyFloatArray | None = None,
        *,
        chord_angles: AnyFloatArray | None = None,
    ) -> FloatArray:
        """Return the exact annular straight-chord projection matrix.

        Parameters
        ----------
        chord_impacts
            Signed chord impact coordinates divided by ``minor_radius_m``,
            shape ``(n_chords,)`` and bounded to ``[-1, 1]``. Sign is retained
            as geometry metadata but the circular projection is even in the
            impact coordinate.
        chord_angles
            Deprecated keyword alias for ``chord_impacts``. Values use the
            same normalised impact-coordinate contract; they are not angles in
            radians.

        Returns
        -------
        FloatArray
            Matrix with shape ``(n_chords, n_rho)`` and units metres. For shell
            ``[rho_j, rho_{j+1}]`` and normalised impact ``b``, the entry is
            ``2 a (sqrt(rho_{j+1}^2-b^2)_+ - sqrt(rho_j^2-b^2)_+)``.

        Raises
        ------
        ValueError
            If the geometry is absent, ambiguous, incorrectly shaped, non-finite, or
            outside the circular plasma.
        """
        impacts = self._resolve_chord_impacts(chord_impacts, chord_angles)
        shell_edges = np.linspace(0.0, 1.0, self.n_rho + 1)
        impact_sq = np.square(impacts)[:, np.newaxis]
        outer_sq = np.maximum(np.square(shell_edges[1:])[np.newaxis, :] - impact_sq, 0.0)
        inner_sq = np.maximum(np.square(shell_edges[:-1])[np.newaxis, :] - impact_sq, 0.0)
        projection = 2.0 * self.minor_radius_m * (np.sqrt(outer_sq) - np.sqrt(inner_sq))
        return cast(FloatArray, projection)

    def _resolve_chord_impacts(
        self, chord_impacts: AnyFloatArray | None, chord_angles: AnyFloatArray | None
    ) -> FloatArray:
        if chord_impacts is not None and chord_angles is not None:
            raise ValueError("Pass chord_impacts or deprecated chord_angles, not both.")
        if chord_impacts is None:
            if chord_angles is None:
                raise ValueError("chord_impacts must be provided.")
            warnings.warn(
                "chord_angles is deprecated; pass signed normalised chord_impacts instead; "
                f"removal is scheduled for {_KALMAN_GEOMETRY_ALIAS_REMOVAL_VERSION}.",
                DeprecationWarning,
                stacklevel=3,
            )
            chord_impacts = chord_angles
        impacts = self._validate_vector(chord_impacts, "chord_impacts", self.n_chords)
        if np.any(np.abs(impacts) > 1.0):
            raise ValueError("chord_impacts must lie within [-1, 1].")
        return impacts

    @staticmethod
    def _validate_vector(values: AnyFloatArray, name: str, size: int) -> FloatArray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},).")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector.copy()

    @staticmethod
    def _validate_covariance(values: AnyFloatArray, name: str, size: int, *, positive_definite: bool) -> FloatArray:
        covariance = np.asarray(values, dtype=float)
        if covariance.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}).")
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
            raise ValueError(f"{name} must be symmetric.")
        covariance = 0.5 * (covariance + covariance.T)
        if positive_definite:
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{name} must be positive definite.") from exc
        else:
            eigenvalues = np.linalg.eigvalsh(covariance)
            tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(covariance)))) * size
            if eigenvalues[0] < -tolerance:
                raise ValueError(f"{name} must be positive semidefinite.")
        return cast(FloatArray, covariance)

    def predict(self, ne: AnyFloatArray, dt: float) -> AnyFloatArray:
        """Propagate the state and inflate its covariance by process noise.

        Parameters
        ----------
        ne
            Predicted density profile in m⁻³, shape ``(n_rho,)``, used as the
            propagated state mean.
        dt
            Time step in seconds scaling the process-noise increment.

        Returns
        -------
        AnyFloatArray
            The predicted state (density profile) in m⁻³, shape ``(n_rho,)``.

        Raises
        ------
        ValueError
            If the state, timestep, state covariance, or process covariance is
            outside its declared finite-dimensional domain.
        """
        ne_arr = self._validate_vector(ne, "ne", self.n_rho)
        if np.any(ne_arr < 0.0):
            raise ValueError("ne must be non-negative.")
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative.")
        covariance = self._validate_covariance(self.P, "P", self.n_rho, positive_definite=False)
        process_covariance = self._validate_covariance(self.Q, "Q", self.n_rho, positive_definite=False)
        self.x = ne_arr
        self.P = covariance + process_covariance * dt
        return self.x

    def update(
        self,
        ne_pred: AnyFloatArray,
        measurements: AnyFloatArray,
        chord_impacts: AnyFloatArray | None = None,
        *,
        chord_angles: AnyFloatArray | None = None,
    ) -> AnyFloatArray:
        """Correct the predicted state with chord measurements (Kalman update).

        Parameters
        ----------
        ne_pred
            Predicted density profile in m⁻³, shape ``(n_rho,)``.
        measurements
            Line-integrated chord measurements, shape ``(n_chords,)``.
        chord_impacts
            Signed normalised chord impacts passed to
            :meth:`measurement_matrix`, shape ``(n_chords,)``.
        chord_angles
            Deprecated keyword alias for ``chord_impacts``; values are still
            normalised impact coordinates, not angles in radians.

        Returns
        -------
        AnyFloatArray
            Posterior density estimate in m⁻³, shape ``(n_rho,)``.

        Raises
        ------
        ValueError
            If state, measurements, chord geometry, or covariance matrices are
            invalid, or the innovation covariance is not positive definite.
        """
        ne_arr = self._validate_vector(ne_pred, "ne_pred", self.n_rho)
        if np.any(ne_arr < 0.0):
            raise ValueError("ne_pred must be non-negative.")
        measurement_arr = self._validate_vector(measurements, "measurements", self.n_chords)
        covariance = self._validate_covariance(self.P, "P", self.n_rho, positive_definite=False)
        measurement_covariance = self._validate_covariance(self.R, "R", self.n_chords, positive_definite=True)
        C = self.measurement_matrix(chord_impacts, chord_angles=chord_angles)
        covariance_measurement = covariance @ C.T
        innovation_covariance = C @ covariance_measurement + measurement_covariance
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        try:
            factor = np.linalg.cholesky(innovation_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance must be positive definite.") from exc
        gain = np.linalg.solve(factor.T, np.linalg.solve(factor, covariance_measurement.T)).T
        innovation = measurement_arr - C @ ne_arr
        self.x = ne_arr + gain @ innovation
        correction = np.eye(self.n_rho) - gain @ C
        joseph_covariance = correction @ covariance @ correction.T + gain @ measurement_covariance @ gain.T
        self.P = 0.5 * (joseph_covariance + joseph_covariance.T)
        return self.x


@dataclass
class PelletSchedule:
    """Timed pellet-injection sequence.

    Attributes
    ----------
    times
        Injection times in seconds.
    speeds
        Pellet velocities in m/s, one per injection.
    sizes
        Pellet radii in mm, one per injection.
    """

    times: list[float]
    speeds: list[float]
    sizes: list[float]


class FuelingOptimizer:
    """Plan evenly-spaced pellet-injection sequences toward a target density."""

    def optimize_pellet_sequence(
        self, ne_current: AnyFloatArray, ne_target: AnyFloatArray, n_pellets: int, time_horizon: float
    ) -> PelletSchedule:
        """Evenly-spaced pellet schedule over the given horizon."""
        if n_pellets <= 0:
            return PelletSchedule([], [], [])

        dt = time_horizon / (n_pellets + 1)
        times = [dt * (i + 1) for i in range(n_pellets)]
        speeds = [500.0] * n_pellets
        sizes = [2.0] * n_pellets
        return PelletSchedule(times, speeds, sizes)
