# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second claim evidence.

"""Claim-evidence admission and persistence for volt-second scenarios."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scpn_control.control.volt_second_core import FluxBudget, FluxReport, _finite_scalar

_VOLT_SECOND_CLAIM_SCHEMA_VERSION = 1
_FACILITY_VOLT_SECOND_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "measured_loop_voltage_replay", "external_scenario_benchmark"}
)
_BOUNDED_VOLT_SECOND_REFERENCE_SOURCES = frozenset(
    {"repository_volt_second_regression", *_FACILITY_VOLT_SECOND_REFERENCE_SOURCES}
)


@dataclass(frozen=True)
class VoltSecondClaimEvidence:
    """Serializable bounded evidence with an explicit facility-admission flag."""

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


def _non_empty_text(name: str, value: object) -> str:
    """Normalize a required, nonblank text field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_reference_scalar(name: str, value: object) -> float:
    """Require a finite positive reference tolerance."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_reference_scalar(name: str, value: object) -> float:
    """Require a finite nonnegative declared reference error."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and nonnegative")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _sha256_text(name: str, value: object) -> str:
    """Check digest syntax without claiming the referenced bytes were hashed."""
    text = _non_empty_text(name, str(value))
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return text


def _extract_volt_second_reference_artifact(
    reference_artifact: object,
) -> tuple[dict[str, Any] | None, bool]:
    """Check caller-declared reference metadata shape without admission."""
    if reference_artifact is None:
        return None, False
    if not isinstance(reference_artifact, dict):
        raise ValueError("reference_artifact must be a dictionary")
    source = _non_empty_text("reference_artifact.source", str(reference_artifact.get("source", "")))
    if source not in _FACILITY_VOLT_SECOND_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_FACILITY_VOLT_SECOND_REFERENCE_SOURCES))
        raise ValueError(f"reference_artifact.source must be one of: {allowed}")
    _non_empty_text("reference_artifact.reference_dataset_id", reference_artifact.get("reference_dataset_id"))
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
    """Build bounded evidence without admitting caller-declared facility references.

    Reference metadata is retained for traceability, but its digest and error
    values are not recomputed from independent source bytes here.
    """
    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_VOLT_SECOND_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_VOLT_SECOND_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    _finite_scalar("Phi_CS_Vs", budget.Phi_CS_Vs, positive=True)
    _finite_scalar("L_plasma_H", budget.L_plasma_H, positive=True)
    _finite_scalar("R_plasma_Ohm", budget.R_plasma_Ohm, positive=True)
    for field in ("ramp_flux", "flat_top_flux", "ramp_down_flux", "total_flux", "margin_Vs"):
        _finite_scalar(field, getattr(report, field))
    if not isinstance(report.within_budget, bool):
        raise ValueError("within_budget must be boolean")
    phase_total = _finite_scalar("phase flux total", report.ramp_flux + report.flat_top_flux + report.ramp_down_flux)
    if not math.isclose(report.total_flux, phase_total, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("total_flux must equal the phase flux sum")
    expected_margin = _finite_scalar("expected margin", budget.Phi_CS_Vs - report.total_flux)
    if not math.isclose(report.margin_Vs, expected_margin, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("margin_Vs must equal budget minus total_flux")
    if report.within_budget != (report.total_flux <= budget.Phi_CS_Vs):
        raise ValueError("within_budget must agree with total_flux")
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
    artifact, artifact_shape_passed = _extract_volt_second_reference_artifact(reference_artifact)
    facility_claim_allowed = False
    claim_status = "reference_metadata_unverified" if artifact_shape_passed else "bounded_volt_second_evidence"
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


def assert_volt_second_facility_claim_admissible(evidence: object) -> VoltSecondClaimEvidence:
    """Refuse facility admission until independent reference bytes are bound."""
    if not isinstance(evidence, VoltSecondClaimEvidence):
        raise ValueError("evidence must be VoltSecondClaimEvidence")
    raise ValueError("facility volt-second claim requires matched independent reference bytes and metrics")


def save_volt_second_claim_evidence(evidence: object, path: str | Path) -> None:
    """Persist finite bounded evidence while rejecting forged facility status."""
    if not isinstance(evidence, VoltSecondClaimEvidence):
        raise ValueError("evidence must be VoltSecondClaimEvidence")
    if evidence.facility_claim_allowed or evidence.claim_status not in {
        "bounded_volt_second_evidence",
        "reference_metadata_unverified",
    }:
        raise ValueError("unverified facility volt-second evidence cannot be persisted")
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
