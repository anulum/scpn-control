# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second management
"""Volt-second flux budgeting, consumption monitoring, and scenario feasibility utilities."""

from __future__ import annotations

# Volt-second balance: ∫ V_loop dt = L_p dI_p + R_p I_p dt
# (inductive + resistive consumption).
# Reference: Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4.
#
# Ejima coefficient: ΔΨ_startup = C_Ejima · μ₀ · R₀ · I_p.
# C_Ejima ≈ 0.4 for ITER.
# Reference: Ejima et al. 1982, Nucl. Fusion 22, 1313.
#
# Flat-top duration: τ_flat = (Ψ_avail − Ψ_startup) / (R_p I_p).
# Reference: ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# Ejima et al. 1982, Nucl. Fusion 22, 1313 — startup flux coefficient.
# C_Ejima ≈ 0.4 is the ITER design value.
C_EJIMA: float = 0.4  # dimensionless

# Vacuum permeability, SI.
MU_0: float = 4.0 * math.pi * 1e-7  # H/m
_VOLT_SECOND_CLAIM_SCHEMA_VERSION = 1
_FACILITY_VOLT_SECOND_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "measured_loop_voltage_replay", "external_scenario_benchmark"}
)
_BOUNDED_VOLT_SECOND_REFERENCE_SOURCES = frozenset(
    {"repository_volt_second_regression", *_FACILITY_VOLT_SECOND_REFERENCE_SOURCES}
)


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite_profile(
    name: str, values: NDArray[np.float64], *, positive: bool = False, nonnegative: bool = False
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a one-dimensional non-empty profile")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _strict_rho(values: NDArray[np.float64]) -> NDArray[np.float64]:
    rho = _finite_profile("rho", values, nonnegative=True)
    if rho.size < 2:
        raise ValueError("rho must contain at least two points")
    if rho[0] < 0.0 or rho[-1] > 1.0:
        raise ValueError("rho must stay within the normalised interval [0, 1]")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    return rho


@dataclass
class FluxStatus:
    flux_consumed_Vs: float
    flux_remaining_Vs: float
    estimated_remaining_time_s: float
    fraction_consumed: float


@dataclass
class FluxReport:
    ramp_flux: float
    flat_top_flux: float
    ramp_down_flux: float
    total_flux: float
    within_budget: bool
    margin_Vs: float


@dataclass(frozen=True)
class VoltSecondClaimEvidence:
    """Serialisable evidence for bounded or facility volt-second claims."""

    schema_version: int
    source: str
    source_id: str
    model_id: str
    Phi_CS_Vs: float
    L_plasma_H: float
    R_plasma_Ohm: float
    Ip_MA: float
    I_bs_MA: float
    ramp_duration_s: float
    flat_duration_s: float
    ramp_down_duration_s: float
    ramp_flux_Vs: float
    flat_top_flux_Vs: float
    ramp_down_flux_Vs: float
    total_flux_Vs: float
    margin_Vs: float
    within_budget: bool
    ejima_startup_flux_Vs: float
    max_flattop_duration_s: float
    bootstrap_source: str
    reference_source: str | None
    reference_dataset_id: str | None
    reference_artifact_sha256: str | None
    reference_case_count: int | None
    total_flux_relative_error: float | None
    flat_top_duration_relative_error: float | None
    ejima_flux_relative_error: float | None
    bootstrap_current_abs_error_MA: float | None
    margin_abs_error_Vs: float | None
    total_flux_relative_tolerance: float
    flat_top_duration_relative_tolerance: float
    ejima_flux_relative_tolerance: float
    bootstrap_current_abs_tolerance_MA: float
    margin_abs_tolerance_Vs: float
    facility_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and nonnegative")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _sha256_text(name: str, value: object) -> str:
    text = _non_empty_text(name, str(value))
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return text


def _extract_volt_second_reference_artifact(
    reference_artifact: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    if reference_artifact is None:
        return None, False
    if not isinstance(reference_artifact, dict):
        raise ValueError("reference_artifact must be a dictionary")
    source = _non_empty_text("reference_artifact.source", str(reference_artifact.get("source", "")))
    if source not in _FACILITY_VOLT_SECOND_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_FACILITY_VOLT_SECOND_REFERENCE_SOURCES))
        raise ValueError(f"reference_artifact.source must be one of: {allowed}")
    units = reference_artifact.get("units")
    expected_units = {
        "flux": "V s",
        "voltage": "V",
        "current": "A",
        "current_MA": "MA",
        "time": "s",
        "resistance": "ohm",
        "inductance": "H",
        "radius": "m",
        "dimensionless": "1",
    }
    if not isinstance(units, dict) or any(units.get(key) != unit for key, unit in expected_units.items()):
        raise ValueError("reference_artifact.units must declare volt-second unit contracts")
    _sha256_text("reference_artifact.reference_artifact_sha256", reference_artifact.get("reference_artifact_sha256"))
    case_count = reference_artifact.get("reference_case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise ValueError("reference_artifact.reference_case_count must be a positive integer")
    metrics = reference_artifact.get("metrics")
    tolerances = reference_artifact.get("tolerances")
    if not isinstance(metrics, dict) or not isinstance(tolerances, dict):
        raise ValueError("reference_artifact metrics and tolerances must be dictionaries")
    for metric in (
        "total_flux_relative_error",
        "flat_top_duration_relative_error",
        "ejima_flux_relative_error",
        "bootstrap_current_abs_error_MA",
        "margin_abs_error_Vs",
    ):
        observed = _nonnegative_reference_scalar(f"reference_artifact.metrics.{metric}", metrics.get(metric))
        tolerance = _positive_reference_scalar(f"reference_artifact.tolerances.{metric}", tolerances.get(metric))
        if observed > tolerance:
            raise ValueError(f"reference_artifact metric {metric} exceeds declared tolerance")
    return reference_artifact, True


def volt_second_claim_evidence(
    budget: FluxBudget,
    report: FluxReport,
    *,
    Ip_MA: float,
    I_bs_MA: float,
    ramp_duration_s: float,
    flat_duration_s: float,
    ramp_down_duration_s: float,
    R0_m: float,
    ramp_flux_for_flattop_Vs: float,
    source: str,
    source_id: str,
    bootstrap_source: str = "repository bootstrap-current proxy",
    model_id: str = "bounded_volt_second_manager",
    reference_artifact: dict[str, Any] | None = None,
    total_flux_relative_tolerance: float = 0.03,
    flat_top_duration_relative_tolerance: float = 0.05,
    ejima_flux_relative_tolerance: float = 0.05,
    bootstrap_current_abs_tolerance_MA: float = 0.25,
    margin_abs_tolerance_Vs: float = 0.5,
) -> VoltSecondClaimEvidence:
    """Build fail-closed evidence for scenario volt-second claims."""

    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_VOLT_SECOND_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_VOLT_SECOND_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    Ip = _finite_scalar("Ip_MA", Ip_MA, positive=True)
    I_bs = _finite_scalar("I_bs_MA", I_bs_MA, nonnegative=True)
    ramp_dur = _finite_scalar("ramp_duration_s", ramp_duration_s, nonnegative=True)
    flat_dur = _finite_scalar("flat_duration_s", flat_duration_s, nonnegative=True)
    down_dur = _finite_scalar("ramp_down_duration_s", ramp_down_duration_s, nonnegative=True)
    R0 = _finite_scalar("R0_m", R0_m, positive=True)
    ramp_flux = _finite_scalar("ramp_flux_for_flattop_Vs", ramp_flux_for_flattop_Vs, nonnegative=True)
    total_tol = _positive_reference_scalar("total_flux_relative_tolerance", total_flux_relative_tolerance)
    flat_tol = _positive_reference_scalar("flat_top_duration_relative_tolerance", flat_top_duration_relative_tolerance)
    ejima_tol = _positive_reference_scalar("ejima_flux_relative_tolerance", ejima_flux_relative_tolerance)
    bootstrap_tol = _positive_reference_scalar("bootstrap_current_abs_tolerance_MA", bootstrap_current_abs_tolerance_MA)
    margin_tol = _positive_reference_scalar("margin_abs_tolerance_Vs", margin_abs_tolerance_Vs)
    artifact, artifact_passed = _extract_volt_second_reference_artifact(reference_artifact)
    facility_claim_allowed = bool(source_clean in _FACILITY_VOLT_SECOND_REFERENCE_SOURCES and artifact_passed)
    claim_status = (
        "facility_volt_second_reference_matched" if facility_claim_allowed else "bounded_volt_second_evidence"
    )
    metrics = artifact.get("metrics", {}) if artifact else {}

    return VoltSecondClaimEvidence(
        schema_version=_VOLT_SECOND_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        model_id=_non_empty_text("model_id", model_id),
        Phi_CS_Vs=float(budget.Phi_CS_Vs),
        L_plasma_H=float(budget.L_plasma_H),
        R_plasma_Ohm=float(budget.R_plasma_Ohm),
        Ip_MA=Ip,
        I_bs_MA=I_bs,
        ramp_duration_s=ramp_dur,
        flat_duration_s=flat_dur,
        ramp_down_duration_s=down_dur,
        ramp_flux_Vs=float(report.ramp_flux),
        flat_top_flux_Vs=float(report.flat_top_flux),
        ramp_down_flux_Vs=float(report.ramp_down_flux),
        total_flux_Vs=float(report.total_flux),
        margin_Vs=float(report.margin_Vs),
        within_budget=bool(report.within_budget),
        ejima_startup_flux_Vs=float(budget.ejima_startup_flux(R0, Ip)),
        max_flattop_duration_s=float(budget.max_flattop_duration(Ip, I_bs, ramp_flux)),
        bootstrap_source=_non_empty_text("bootstrap_source", bootstrap_source),
        reference_source=None if artifact is None else str(artifact["source"]),
        reference_dataset_id=None if artifact is None else str(artifact["reference_dataset_id"]),
        reference_artifact_sha256=None if artifact is None else str(artifact["reference_artifact_sha256"]),
        reference_case_count=None if artifact is None else int(artifact["reference_case_count"]),
        total_flux_relative_error=None if artifact is None else float(metrics["total_flux_relative_error"]),
        flat_top_duration_relative_error=None
        if artifact is None
        else float(metrics["flat_top_duration_relative_error"]),
        ejima_flux_relative_error=None if artifact is None else float(metrics["ejima_flux_relative_error"]),
        bootstrap_current_abs_error_MA=None if artifact is None else float(metrics["bootstrap_current_abs_error_MA"]),
        margin_abs_error_Vs=None if artifact is None else float(metrics["margin_abs_error_Vs"]),
        total_flux_relative_tolerance=total_tol,
        flat_top_duration_relative_tolerance=flat_tol,
        ejima_flux_relative_tolerance=ejima_tol,
        bootstrap_current_abs_tolerance_MA=bootstrap_tol,
        margin_abs_tolerance_Vs=margin_tol,
        facility_claim_allowed=facility_claim_allowed,
        claim_status=claim_status,
    )


def assert_volt_second_facility_claim_admissible(evidence: VoltSecondClaimEvidence) -> VoltSecondClaimEvidence:
    """Raise when volt-second evidence is insufficient for facility scenario claims."""

    if not isinstance(evidence, VoltSecondClaimEvidence):
        raise ValueError("evidence must be VoltSecondClaimEvidence")
    if not evidence.facility_claim_allowed:
        raise ValueError("facility volt-second claim requires matched loop-voltage or scenario reference evidence")
    return evidence


def save_volt_second_claim_evidence(evidence: VoltSecondClaimEvidence, path: str | Path) -> None:
    """Persist volt-second claim evidence as deterministic JSON."""

    if not isinstance(evidence, VoltSecondClaimEvidence):
        raise ValueError("evidence must be VoltSecondClaimEvidence")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class FluxBudget:
    """Volt-second budget tracker.

    Implements the balance:
        ∫ V_loop dt = L_p dI_p + R_p I_p dt
    Reference: Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4.
    """

    def __init__(self, Phi_CS_Vs: float, L_plasma_uH: float, R_plasma_uOhm: float):
        self.Phi_CS_Vs = _finite_scalar("Phi_CS_Vs", Phi_CS_Vs, positive=True)
        self.L_plasma_H = _finite_scalar("L_plasma_uH", L_plasma_uH, positive=True) * 1e-6
        self.R_plasma_Ohm = _finite_scalar("R_plasma_uOhm", R_plasma_uOhm, positive=True) * 1e-6

    def inductive_flux(self, Ip_MA: float) -> float:
        """L_p · I_p — inductive volt-second consumption.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 (first term).
        """
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, nonnegative=True)
        return self.L_plasma_H * (Ip_MA * 1e6)

    def resistive_flux_ramp(self, Ip_trace: NDArray[np.float64], dt: float) -> float:
        """∫ R_p I_p dt — resistive volt-second consumption during ramp.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 (second term).
        """
        Ip_trace = _finite_profile("Ip_trace", Ip_trace, nonnegative=True)
        dt = _finite_scalar("dt", dt, positive=True)
        return float(np.sum(self.R_plasma_Ohm * (Ip_trace * 1e6) * dt))

    def ejima_startup_flux(self, R0_m: float, Ip_MA: float) -> float:
        """Startup flux via Ejima coefficient: ΔΨ = C_Ejima · μ₀ · R₀ · I_p.

        Ejima et al. 1982, Nucl. Fusion 22, 1313, Eq. 2.
        C_EJIMA = 0.4 is the ITER design value.
        """
        R0_m = _finite_scalar("R0_m", R0_m, positive=True)
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, nonnegative=True)
        return C_EJIMA * MU_0 * R0_m * (Ip_MA * 1e6)

    def remaining_flux(self, Ip_MA: float, ramp_flux: float) -> float:
        ramp_flux = _finite_scalar("ramp_flux", ramp_flux, nonnegative=True)
        ind = self.inductive_flux(Ip_MA)
        consumed = ind + ramp_flux
        return max(0.0, self.Phi_CS_Vs - consumed)

    def max_flattop_duration(self, Ip_MA: float, I_bs_MA: float, ramp_flux: float) -> float:
        """τ_flat = (Ψ_avail − Ψ_startup) / (R_p I_p).

        ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.
        """
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, positive=True)
        I_bs_MA = _finite_scalar("I_bs_MA", I_bs_MA, nonnegative=True)
        ramp_flux = _finite_scalar("ramp_flux", ramp_flux, nonnegative=True)
        rem = self.remaining_flux(Ip_MA, ramp_flux)
        I_driven = max((Ip_MA - I_bs_MA) * 1e6, 1e-6)
        return float(rem / (self.R_plasma_Ohm * I_driven))


class VoltSecondOptimizer:
    def __init__(self, flux_budget: FluxBudget, transport_model: Callable[..., Any] | None = None):
        self.budget = flux_budget
        self.transport_model = transport_model

    def optimize_ramp(self, Ip_target_MA: float, t_ramp_max: float, n_segments: int = 10) -> NDArray[np.float64]:
        Ip_target_MA = _finite_scalar("Ip_target_MA", Ip_target_MA, nonnegative=True)
        t_ramp_max = _finite_scalar("t_ramp_max", t_ramp_max, positive=True)
        n_segments = _positive_int("n_segments", n_segments, minimum=2)
        t_arr = np.linspace(0, t_ramp_max, n_segments)
        Ip_trace = Ip_target_MA * (t_arr / t_ramp_max)
        return np.asarray(Ip_trace, dtype=np.float64)


class BootstrapCurrentEstimate:
    @staticmethod
    def from_profiles(
        ne: NDArray[np.float64],
        Te: NDArray[np.float64],
        Ti: NDArray[np.float64],
        q: NDArray[np.float64],
        rho: NDArray[np.float64],
        R0: float,
        a: float,
    ) -> float:
        """Simplified bootstrap current proxy: I_bs ~ ε^{1/2} · ∫ dp/dr dr.

        Full neoclassical expression in Wesson 2011, Ch. 4.9.
        ε = a/R₀ is the inverse aspect ratio.
        """
        ne = _finite_profile("ne", ne, positive=True)
        Te = _finite_profile("Te", Te, positive=True)
        Ti = _finite_profile("Ti", Ti, positive=True)
        q = _finite_profile("q", q, positive=True)
        rho = _strict_rho(rho)
        if not (len(ne) == len(Te) == len(Ti) == len(q) == len(rho)):
            raise ValueError("ne, Te, Ti, q, and rho profiles must have the same length")
        R0 = _finite_scalar("R0", R0, positive=True)
        a = _finite_scalar("a", a, positive=True)
        if a >= R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        epsilon = a / R0
        p = 2.0 * ne * 1e19 * Te * 1e3 * 1.6e-19
        grad_p = np.gradient(p, rho)

        # J_bs ~ −ε^{1/2} / B_pol · dp/dr  (Wesson 2011, Eq. 4.9.4, rough scaling)
        J_bs_integral = np.sum(-grad_p * math.sqrt(epsilon)) * 1e-5

        I_bs_MA = max(0.0, J_bs_integral * 0.1)
        return float(I_bs_MA)


class FluxConsumptionMonitor:
    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget
        self.consumed = 0.0

    def step(self, Ip: float, V_loop: float, dt: float) -> FluxStatus:
        """Integrate V_loop dt to track consumed volt-seconds.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 — V_loop drives both
        inductive and resistive flux consumption.
        """
        _finite_scalar("Ip", Ip, nonnegative=True)
        V_loop = _finite_scalar("V_loop", V_loop, nonnegative=True)
        dt = _finite_scalar("dt", dt, positive=True)
        self.consumed += V_loop * dt
        rem = self.budget.Phi_CS_Vs - self.consumed

        est_time = rem / max(V_loop, 1e-3) if rem > 0 else 0.0
        frac = self.consumed / self.budget.Phi_CS_Vs

        return FluxStatus(
            flux_consumed_Vs=self.consumed,
            flux_remaining_Vs=max(0.0, rem),
            estimated_remaining_time_s=est_time,
            fraction_consumed=frac,
        )


class ScenarioFluxAnalysis:
    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget

    def analyze(self, ramp_dur: float, flat_dur: float, down_dur: float, Ip_MA: float, I_bs_MA: float) -> FluxReport:
        """Decompose total flux consumption into ramp / flat-top / ramp-down.

        Ramp: L_p I_p (inductive) + R_p · 0.5 I_p · t_ramp (resistive at mean current).
        Flat-top: R_p · (I_p − I_bs) · t_flat.
        Ramp-down: resistive loss minus partial inductive recovery.
        Reference: ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.
        """
        ramp_dur = _finite_scalar("ramp_dur", ramp_dur, nonnegative=True)
        flat_dur = _finite_scalar("flat_dur", flat_dur, nonnegative=True)
        down_dur = _finite_scalar("down_dur", down_dur, nonnegative=True)
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, positive=True)
        I_bs_MA = _finite_scalar("I_bs_MA", I_bs_MA, nonnegative=True)
        L_term = self.budget.inductive_flux(Ip_MA)

        R_term_ramp = self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * ramp_dur
        ramp_flux = L_term + R_term_ramp

        flat_flux = self.budget.R_plasma_Ohm * max((Ip_MA - I_bs_MA) * 1e6, 0.0) * flat_dur

        down_flux = self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * down_dur - L_term * 0.5

        tot = ramp_flux + flat_flux + down_flux

        return FluxReport(
            ramp_flux=ramp_flux,
            flat_top_flux=flat_flux,
            ramp_down_flux=down_flux,
            total_flux=tot,
            within_budget=tot <= self.budget.Phi_CS_Vs,
            margin_Vs=self.budget.Phi_CS_Vs - tot,
        )


__all__ = [
    "BootstrapCurrentEstimate",
    "C_EJIMA",
    "FluxBudget",
    "FluxConsumptionMonitor",
    "FluxReport",
    "FluxStatus",
    "MU_0",
    "ScenarioFluxAnalysis",
    "VoltSecondClaimEvidence",
    "VoltSecondOptimizer",
    "assert_volt_second_facility_claim_admissible",
    "save_volt_second_claim_evidence",
    "volt_second_claim_evidence",
]
