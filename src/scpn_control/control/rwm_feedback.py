# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Resistive-wall-mode feedback
"""Resistive-wall-mode feedback controller and state update utilities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ── named constants ─────────────────────────────────────────────────────────
# Sentinel for the ideal-kink (β > β_wall) regime; a finite resistive-wall-mode
# growth rate is physically undefined beyond the ideal-wall limit.
_IDEAL_KINK_RATE: float = math.inf
_RWM_CLAIM_SCHEMA_VERSION = 1
_FACILITY_REFERENCE_SOURCES = frozenset({"documented_public_reference", "external_mhd_benchmark", "measured_discharge"})
_BOUNDED_REFERENCE_SOURCES = frozenset({"local_regression_reference", *_FACILITY_REFERENCE_SOURCES})


@dataclass(frozen=True)
class RWMClaimEvidence:
    """Serialisable admission evidence for RWM feedback claims."""

    schema_version: int
    source: str
    source_id: str
    model_id: str
    beta_n: float
    beta_n_nowall: float
    beta_n_wall: float
    tau_wall_s: float
    tau_eff_s: float
    omega_phi_rad_s: float
    wall_radius_m: float | None
    plasma_radius_m: float | None
    n_sensors: int
    n_coils: int
    proportional_gain: float
    derivative_gain: float
    controller_latency_s: float
    coil_coupling: float
    open_loop_growth_rate_s_inv: float
    closed_loop_growth_rate_s_inv: float
    required_proportional_gain: float
    reference_closed_loop_growth_rate_s_inv: float | None
    closed_loop_growth_rate_abs_error: float | None
    closed_loop_growth_rate_abs_tolerance: float
    facility_claim_allowed: bool
    claim_status: str


def _finite_scalar(name: str, value: float, *, nonnegative: bool = False, positive: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _optional_positive_scalar(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _finite_scalar(name, value, positive=True)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def rwm_claim_evidence(
    rwm: "RWMPhysics",
    controller: "RWMFeedbackController",
    *,
    source: str,
    source_id: str,
    model_id: str = "bounded_rwm_feedback",
    reference_closed_loop_growth_rate_s_inv: float | None = None,
    closed_loop_growth_rate_abs_tolerance: float = 5.0,
) -> RWMClaimEvidence:
    """Build fail-closed evidence for RWM feedback claim admission.

    Local regression evidence is admissible only for bounded repository claims.
    Facility claims require documented public, external MHD, or measured-shot
    references plus a finite closed-loop growth-rate comparison within the
    declared tolerance.
    """

    if source not in _BOUNDED_REFERENCE_SOURCES:
        raise ValueError("source must be a declared RWM reference source")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    tolerance = _finite_scalar(
        "closed_loop_growth_rate_abs_tolerance", closed_loop_growth_rate_abs_tolerance, positive=True
    )
    open_loop = float(rwm.growth_rate())
    closed_loop = float(controller.effective_growth_rate(rwm))
    if not math.isfinite(open_loop) or not math.isfinite(closed_loop):
        raise ValueError("RWM claim evidence excludes the ideal-kink regime")

    reference: float | None = None
    abs_error: float | None = None
    if reference_closed_loop_growth_rate_s_inv is not None:
        reference = _finite_scalar("reference_closed_loop_growth_rate_s_inv", reference_closed_loop_growth_rate_s_inv)
        abs_error = abs(closed_loop - reference)

    facility_source = source in _FACILITY_REFERENCE_SOURCES
    facility_allowed = bool(facility_source and abs_error is not None and abs_error <= tolerance)
    if source == "local_regression_reference":
        claim_status = "bounded local RWM regression evidence only; external MHD or measured-shot reference required for facility claims"
    elif abs_error is None:
        claim_status = "external RWM reference source declared but quantitative closed-loop comparison is missing"
    elif facility_allowed:
        claim_status = "external RWM reference admission passed for declared tolerance"
    else:
        claim_status = "external RWM reference admission failed declared tolerance"

    required_gain = RWMStabilityAnalysis.required_feedback_gain(
        rwm.beta_n,
        rwm.beta_n_nowall,
        rwm.beta_n_wall,
        rwm.tau_wall,
        controller.tau_controller,
        M_coil=controller.M_coil,
        omega_phi=rwm.omega_phi,
        wall_radius=rwm.wall_radius,
        plasma_radius=rwm.plasma_radius,
    )
    return RWMClaimEvidence(
        schema_version=_RWM_CLAIM_SCHEMA_VERSION,
        source=source,
        source_id=source_id.strip(),
        model_id=model_id.strip(),
        beta_n=float(rwm.beta_n),
        beta_n_nowall=float(rwm.beta_n_nowall),
        beta_n_wall=float(rwm.beta_n_wall),
        tau_wall_s=float(rwm.tau_wall),
        tau_eff_s=float(rwm.tau_eff()),
        omega_phi_rad_s=float(rwm.omega_phi),
        wall_radius_m=None if rwm.wall_radius is None else float(rwm.wall_radius),
        plasma_radius_m=None if rwm.plasma_radius is None else float(rwm.plasma_radius),
        n_sensors=int(controller.n_sensors),
        n_coils=int(controller.n_coils),
        proportional_gain=float(controller.G_p),
        derivative_gain=float(controller.G_d),
        controller_latency_s=float(controller.tau_controller),
        coil_coupling=float(controller.M_coil),
        open_loop_growth_rate_s_inv=open_loop,
        closed_loop_growth_rate_s_inv=closed_loop,
        required_proportional_gain=float(required_gain),
        reference_closed_loop_growth_rate_s_inv=reference,
        closed_loop_growth_rate_abs_error=None if abs_error is None else float(abs_error),
        closed_loop_growth_rate_abs_tolerance=tolerance,
        facility_claim_allowed=facility_allowed,
        claim_status=claim_status,
    )


def assert_rwm_facility_claim_admissible(evidence: RWMClaimEvidence) -> RWMClaimEvidence:
    """Return evidence or fail closed before a RWM facility-control claim."""

    if not isinstance(evidence, RWMClaimEvidence):
        raise ValueError("evidence must be RWMClaimEvidence")
    if evidence.schema_version != _RWM_CLAIM_SCHEMA_VERSION:
        raise ValueError("RWM claim evidence schema_version is unsupported")
    if not evidence.facility_claim_allowed:
        raise ValueError(f"RWM facility claim is not admissible: {evidence.claim_status}")
    return evidence


def save_rwm_claim_evidence(evidence: RWMClaimEvidence, path: str | Path) -> None:
    """Persist RWM claim evidence as deterministic JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RWMPhysics:
    """
    Resistive Wall Mode growth rate with rotation stabilization.

    References
    ----------
    Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709  (γ_wall formula, Ω_crit)
    Fitzpatrick 2001, Phys. Plasmas 8, 4489           (rotation stabilization γ_rot)
    Strait et al. 2003, Nucl. Fusion 43, 430          (wall dissipation, τ_eff)
    Garofalo et al. 2002, Phys. Plasmas 9, 1997       (experimental validation)
    """

    def __init__(
        self,
        beta_n: float,
        beta_n_nowall: float,
        beta_n_wall: float,
        tau_wall: float,
        omega_phi: float = 0.0,
        wall_radius: float | None = None,
        plasma_radius: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        beta_n         : normalised beta (dimensionless)
        beta_n_nowall  : no-wall stability limit (dimensionless)
        beta_n_wall    : ideal-wall stability limit (dimensionless)
        tau_wall       : resistive wall L/R time, s
        omega_phi      : toroidal rotation frequency, rad s⁻¹ (default 0)
        wall_radius    : conducting-wall minor radius b, m (optional)
        plasma_radius  : plasma-edge minor radius d, m (optional)
        """
        self.beta_n = _finite_scalar("beta_n", beta_n)
        self.beta_n_nowall = _finite_scalar("beta_n_nowall", beta_n_nowall)
        self.beta_n_wall = _finite_scalar("beta_n_wall", beta_n_wall)
        if self.beta_n_nowall >= self.beta_n_wall:
            raise ValueError("beta_n_nowall must be less than beta_n_wall")
        self.tau_wall = _finite_scalar("tau_wall", tau_wall, nonnegative=True)
        self.omega_phi = _finite_scalar("omega_phi", omega_phi)
        self.wall_radius = _optional_positive_scalar("wall_radius", wall_radius)
        self.plasma_radius = _optional_positive_scalar("plasma_radius", plasma_radius)
        if (self.wall_radius is None) != (self.plasma_radius is None):
            raise ValueError("wall_radius and plasma_radius must be provided together")

    # ── effective wall time ──────────────────────────────────────────────────

    def tau_eff(self) -> float:
        """
        Effective wall time corrected for wall–plasma gap.

        τ_eff = τ_wall × (b/d)²

        Strait et al. 2003, Nucl. Fusion 43, 430, Eq. (3).
        When b == d the expression recovers τ_wall.
        """
        if self.wall_radius is None or self.plasma_radius is None:
            return self.tau_wall
        return self.tau_wall * (self.wall_radius / self.plasma_radius) ** 2

    # ── stability flag ───────────────────────────────────────────────────────

    def is_unstable(self) -> bool:
        """True when β_N lies between the no-wall and ideal-wall limits."""
        return self.beta_n_nowall < self.beta_n < self.beta_n_wall

    # ── growth rate components ───────────────────────────────────────────────

    def _gamma_wall(self, tau: float) -> float:
        """
        RWM growth rate in the absence of rotation.

        γ_wall = (1/τ_eff) × (β_N − β_N,nowall) / (β_N,wall − β_N)

        Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709, Eq. (4).
        """
        if tau <= 0.0:
            return _IDEAL_KINK_RATE if self.beta_n > self.beta_n_nowall else 0.0
        if self.beta_n >= self.beta_n_wall:
            return _IDEAL_KINK_RATE
        if self.beta_n <= self.beta_n_nowall:
            return 0.0
        return (1.0 / tau) * (self.beta_n - self.beta_n_nowall) / (self.beta_n_wall - self.beta_n)

    def _gamma_rot(self, tau: float) -> float:
        """
        Rotation-stabilization contribution to the growth rate.

        γ_rot = −Ω_φ² τ_eff / (1 + (Ω_φ τ_eff)²)

        Fitzpatrick 2001, Phys. Plasmas 8, 4489, Eq. (14).
        γ_rot ≤ 0 always (stabilizing); vanishes when Ω_φ = 0.
        """
        if self.omega_phi == 0.0:
            return 0.0
        x = self.omega_phi * tau  # dimensionless: Ω_φ τ_eff
        return -(x**2 / tau) / (1.0 + x**2)

    def growth_rate(self) -> float:
        """
        Total RWM growth rate including rotation stabilization.

        γ_total = γ_wall + γ_rot

        Bondeson & Ward 1994, Eq. (4); Fitzpatrick 2001, Eq. (14).
        γ_total < 0 indicates rotation-stabilized regime.
        """
        tau = self.tau_eff()
        gw = self._gamma_wall(tau)
        if gw >= _IDEAL_KINK_RATE:
            return gw  # ideal kink; rotation cannot stabilize
        return gw + self._gamma_rot(tau)

    # ── critical rotation ────────────────────────────────────────────────────

    def critical_rotation(self) -> float:
        """
        Minimum toroidal rotation for passive RWM stabilization (no coil feedback).

        Derived by setting γ_wall + γ_rot = 0 with the Fitzpatrick (2001) γ_rot:

            γ_wall = A/τ_eff,  γ_rot = −(Ωτ)²/(τ(1+(Ωτ)²))
            A = (β_N − β_N,nowall)/(β_N,wall − β_N)

        Solving:  (Ω_crit τ_eff)² = A/(1−A)
            → Ω_crit = (1/τ_eff) × √[A / (1−A)]

        Fitzpatrick 2001, Phys. Plasmas 8, 4489, Eq. (14) (γ_rot source).
        Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709 (rotation-stabilization concept).

        Returns 0 when β_N ≤ β_N,nowall (already stable).
        Returns ∞ when β_N ≥ β_N,wall (ideal kink; rotation cannot stabilize).
        """
        tau = self.tau_eff()
        if tau <= 0.0:
            return math.inf
        if self.beta_n <= self.beta_n_nowall:
            return 0.0
        if self.beta_n >= self.beta_n_wall:
            return math.inf
        # A = normalised distance into the unstable window
        a = (self.beta_n - self.beta_n_nowall) / (self.beta_n_wall - self.beta_n)
        if a >= 1.0:
            # γ_wall ≥ 1/τ; γ_rot saturates at −1/τ; rotation cannot fully cancel
            return math.inf
        return math.sqrt(a / (1.0 - a)) / tau


class RWMFeedbackController:
    """
    Sensor-coil PD feedback for RWM stabilization.

    The closed-loop effective growth rate combines wall physics (including
    rotation) with proportional feedback suppression:

        γ_eff = γ_total − G_p M_coil γ_total / (1 + γ_total τ_ctrl)

    Garofalo et al. 2002, Phys. Plasmas 9, 1997, Sec. III.
    """

    def __init__(
        self,
        n_sensors: int,
        n_coils: int,
        G_p: float,
        G_d: float,
        tau_controller: float = 1e-4,
        M_coil: float = 1.0,
    ) -> None:
        self.n_sensors = _positive_int("n_sensors", n_sensors)
        self.n_coils = _positive_int("n_coils", n_coils)
        self.G_p = _finite_scalar("G_p", G_p)
        self.G_d = _finite_scalar("G_d", G_d)
        self.tau_controller = _finite_scalar("tau_controller", tau_controller, nonnegative=True)
        self.M_coil = _finite_scalar("M_coil", M_coil, positive=True)
        self.prev_B_r: NDArray[np.float64] = np.zeros(self.n_sensors, dtype=np.float64)

    def step(self, B_r_sensors: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """
        Compute feedback coil currents from sensor measurements.

        I = G_p B_r + G_d dB_r/dt
        """
        B_r: NDArray[np.float64] = np.asarray(B_r_sensors, dtype=np.float64)
        if B_r.shape != (self.n_sensors,):
            raise ValueError("B_r_sensors must be a one-dimensional sensor vector")
        if not np.all(np.isfinite(B_r)):
            raise ValueError("B_r_sensors must be finite")
        dt_value = _finite_scalar("dt", dt, positive=True)

        dB_dt = (B_r - self.prev_B_r) / dt_value
        self.prev_B_r = B_r.copy()

        signal = self.G_p * B_r + self.G_d * dB_dt
        if self.n_sensors == self.n_coils:
            return signal
        return np.full(self.n_coils, float(np.mean(signal)), dtype=np.float64)

    def effective_growth_rate(self, rwm: RWMPhysics) -> float:
        """
        Closed-loop growth rate.

        γ_eff = γ_total − G_p M_coil γ_total / (1 + γ_total τ_ctrl)

        Garofalo et al. 2002, Phys. Plasmas 9, 1997, Eq. (5).
        Uses γ_total from RWMPhysics (includes rotation if Ω_φ ≠ 0).
        """
        gamma = rwm.growth_rate()
        if gamma == 0.0:
            return 0.0
        if gamma >= _IDEAL_KINK_RATE:
            return gamma
        stabilization = self.G_p * self.M_coil * gamma / (1.0 + gamma * self.tau_controller)
        return gamma - stabilization

    def is_stabilized(self, rwm: RWMPhysics) -> bool:
        """True when the closed-loop growth rate is negative."""
        return self.effective_growth_rate(rwm) < 0.0


class RWMStabilityAnalysis:
    """Utility methods for stability-window and gain analysis."""

    @staticmethod
    def required_feedback_gain(
        beta_n: float,
        beta_n_nowall: float,
        beta_n_wall: float,
        tau_wall: float,
        tau_controller: float,
        M_coil: float = 1.0,
        omega_phi: float = 0.0,
        wall_radius: float | None = None,
        plasma_radius: float | None = None,
    ) -> float:
        """
        Minimum G_p such that γ_eff < 0.

        From γ_eff = γ_total (1 − G_p M_coil / (1 + γ_total τ_ctrl)) < 0 :
            G_p > (1 + γ_total τ_ctrl) / M_coil

        Garofalo et al. 2002, Phys. Plasmas 9, 1997, Sec. III.
        Returns 0 when already stable; ∞ for ideal kink.
        """
        tau_controller = _finite_scalar("tau_controller", tau_controller, nonnegative=True)
        M_coil = _finite_scalar("M_coil", M_coil, positive=True)
        rwm = RWMPhysics(
            beta_n,
            beta_n_nowall,
            beta_n_wall,
            tau_wall,
            omega_phi=omega_phi,
            wall_radius=wall_radius,
            plasma_radius=plasma_radius,
        )
        gamma = rwm.growth_rate()
        if gamma <= 0.0:
            return 0.0
        if gamma >= _IDEAL_KINK_RATE:
            return float("inf")
        return (1.0 + gamma * tau_controller) / M_coil
