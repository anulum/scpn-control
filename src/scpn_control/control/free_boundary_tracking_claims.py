# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking claim evidence
"""Fail-closed claim evidence for free-boundary tracking runs.

This module turns a free-boundary tracking run summary (the plain ``dict`` emitted
by :func:`scpn_control.control.free_boundary_tracking.run_free_boundary_tracking`)
into serialisable, admission-gated claim evidence. It is deliberately independent
of the :class:`~scpn_control.control.free_boundary_tracking.FreeBoundaryTrackingController`
engine: it consumes a summary and an optional reference artifact, never a live
controller, so evidence construction and the control loop stay separable.

A claim is admitted as a *facility* free-boundary reference match only when the run
was scored against a documented public reference, a measured free-boundary replay,
or an external equilibrium benchmark **and** the supplied reference artifact passes
its unit, digest, case-count, and per-metric tolerance checks; otherwise the
evidence downgrades to a bounded free-boundary tracking claim. Every numeric field
is validated finite (fail-closed) before it enters the evidence record.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_FREE_BOUNDARY_CLAIM_SCHEMA_VERSION = 1
_FACILITY_FREE_BOUNDARY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "measured_free_boundary_replay", "external_equilibrium_benchmark"}
)
_BOUNDED_FREE_BOUNDARY_REFERENCE_SOURCES = frozenset(
    {"repository_free_boundary_regression", *_FACILITY_FREE_BOUNDARY_REFERENCE_SOURCES}
)


@dataclass(frozen=True)
class FreeBoundaryTrackingClaimEvidence:
    """Serialisable evidence for bounded or facility free-boundary claims."""

    schema_version: int
    source: str
    source_id: str
    model_id: str
    config_path: str
    steps: int
    boundary_variant: str
    final_tracking_error_norm: float
    final_true_tracking_error_norm: float
    true_shape_rms: float
    true_x_point_position_error_m: float
    true_x_point_flux_error: float
    true_divertor_rms: float
    max_abs_coil_current: float
    max_abs_delta_i: float
    min_response_rank: int
    max_response_condition_number: float | None
    response_degenerate_count: int
    supervisor_intervention_count: int
    fallback_active_steps: int
    objective_converged: bool
    measurement_distortion_enabled: bool
    measurement_compensation_enabled: bool
    measurement_latency_enabled: bool
    latency_compensation_enabled: bool
    reference_source: str | None
    reference_dataset_id: str | None
    reference_artifact_sha256: str | None
    reference_case_count: int | None
    shape_rms_abs_error: float | None
    x_point_position_abs_error_m: float | None
    x_point_flux_abs_error: float | None
    divertor_rms_abs_error: float | None
    coil_current_relative_error: float | None
    shape_rms_abs_tolerance: float
    x_point_position_abs_tolerance_m: float
    x_point_flux_abs_tolerance: float
    divertor_rms_abs_tolerance: float
    coil_current_relative_tolerance: float
    facility_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_summary_scalar(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"summary[{key!r}] must be a finite numeric value")
    return float(value)


def _finite_summary_scalar_or_none(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"summary[{key!r}] must be numeric when supplied")
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _summary_int(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"summary[{key!r}] must be an integer")
    return int(value)


def _positive_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _extract_free_boundary_reference_artifact(
    reference_artifact: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    if reference_artifact is None:
        return None, False
    if not isinstance(reference_artifact, dict):
        raise ValueError("reference_artifact must be a dictionary")
    source = _non_empty_text("reference_artifact.source", str(reference_artifact.get("source", "")))
    if source not in _FACILITY_FREE_BOUNDARY_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_FACILITY_FREE_BOUNDARY_REFERENCE_SOURCES))
        raise ValueError(f"reference_artifact.source must be one of: {allowed}")
    units = reference_artifact.get("units")
    expected_units = {
        "position": "m",
        "flux": "Wb/rad",
        "current": "MA",
        "time": "s",
        "tracking_error": "1",
    }
    if not isinstance(units, dict) or any(units.get(key) != unit for key, unit in expected_units.items()):
        raise ValueError("reference_artifact.units must declare free-boundary SI units")
    digest = _non_empty_text(
        "reference_artifact.reference_artifact_sha256", str(reference_artifact.get("reference_artifact_sha256", ""))
    )
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("reference_artifact.reference_artifact_sha256 must be a SHA-256 hex digest")
    case_count = reference_artifact.get("reference_case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise ValueError("reference_artifact.reference_case_count must be a positive integer")
    metrics = reference_artifact.get("metrics")
    tolerances = reference_artifact.get("tolerances")
    if not isinstance(metrics, dict) or not isinstance(tolerances, dict):
        raise ValueError("reference_artifact metrics and tolerances must be dictionaries")
    for metric in (
        "shape_rms_abs_error",
        "x_point_position_abs_error_m",
        "x_point_flux_abs_error",
        "divertor_rms_abs_error",
        "coil_current_relative_error",
    ):
        observed = _nonnegative_reference_scalar(f"reference_artifact.metrics.{metric}", metrics.get(metric))
        tolerance = _positive_reference_scalar(f"reference_artifact.tolerances.{metric}", tolerances.get(metric))
        if observed > tolerance:
            raise ValueError(f"reference_artifact metric {metric} exceeds declared tolerance")
    return reference_artifact, True


def free_boundary_tracking_claim_evidence(
    summary: dict[str, Any],
    *,
    source: str,
    source_id: str,
    model_id: str = "bounded_free_boundary_tracking",
    reference_artifact: dict[str, Any] | None = None,
    shape_rms_abs_tolerance: float = 0.02,
    x_point_position_abs_tolerance_m: float = 0.05,
    x_point_flux_abs_tolerance: float = 0.02,
    divertor_rms_abs_tolerance: float = 0.02,
    coil_current_relative_tolerance: float = 0.05,
) -> FreeBoundaryTrackingClaimEvidence:
    """Build fail-closed free-boundary evidence from a run summary."""
    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_FREE_BOUNDARY_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_FREE_BOUNDARY_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    shape_tol = _positive_reference_scalar("shape_rms_abs_tolerance", shape_rms_abs_tolerance)
    x_tol = _positive_reference_scalar("x_point_position_abs_tolerance_m", x_point_position_abs_tolerance_m)
    flux_tol = _positive_reference_scalar("x_point_flux_abs_tolerance", x_point_flux_abs_tolerance)
    divertor_tol = _positive_reference_scalar("divertor_rms_abs_tolerance", divertor_rms_abs_tolerance)
    coil_tol = _positive_reference_scalar("coil_current_relative_tolerance", coil_current_relative_tolerance)
    artifact, artifact_passed = _extract_free_boundary_reference_artifact(reference_artifact)
    facility_claim_allowed = bool(source_clean in _FACILITY_FREE_BOUNDARY_REFERENCE_SOURCES and artifact_passed)
    claim_status = (
        "facility_free_boundary_reference_matched"
        if facility_claim_allowed
        else "bounded_free_boundary_tracking_evidence"
    )
    metrics = artifact.get("metrics", {}) if artifact else {}

    return FreeBoundaryTrackingClaimEvidence(
        schema_version=_FREE_BOUNDARY_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        model_id=_non_empty_text("model_id", model_id),
        config_path=_non_empty_text("config_path", str(summary.get("config_path", ""))),
        steps=_summary_int(summary, "steps"),
        boundary_variant=_non_empty_text("boundary_variant", str(summary.get("boundary_variant", ""))),
        final_tracking_error_norm=_finite_summary_scalar(summary, "final_tracking_error_norm"),
        final_true_tracking_error_norm=_finite_summary_scalar(summary, "final_true_tracking_error_norm"),
        true_shape_rms=_finite_summary_scalar(summary, "true_shape_rms"),
        true_x_point_position_error_m=_finite_summary_scalar(summary, "true_x_point_position_error"),
        true_x_point_flux_error=_finite_summary_scalar(summary, "true_x_point_flux_error"),
        true_divertor_rms=_finite_summary_scalar(summary, "true_divertor_rms"),
        max_abs_coil_current=_finite_summary_scalar(summary, "max_abs_coil_current"),
        max_abs_delta_i=_finite_summary_scalar(summary, "max_abs_delta_i"),
        min_response_rank=_summary_int(summary, "min_response_rank"),
        max_response_condition_number=_finite_summary_scalar_or_none(summary, "max_response_condition_number"),
        response_degenerate_count=_summary_int(summary, "response_degenerate_count"),
        supervisor_intervention_count=_summary_int(summary, "supervisor_intervention_count"),
        fallback_active_steps=_summary_int(summary, "fallback_active_steps"),
        objective_converged=bool(summary.get("objective_converged")),
        measurement_distortion_enabled=bool(summary.get("measurement_distortion_enabled")),
        measurement_compensation_enabled=bool(summary.get("measurement_compensation_enabled")),
        measurement_latency_enabled=bool(summary.get("measurement_latency_enabled")),
        latency_compensation_enabled=bool(summary.get("latency_compensation_enabled")),
        reference_source=None if artifact is None else str(artifact["source"]),
        reference_dataset_id=None if artifact is None else str(artifact["reference_dataset_id"]),
        reference_artifact_sha256=None if artifact is None else str(artifact["reference_artifact_sha256"]),
        reference_case_count=None if artifact is None else int(artifact["reference_case_count"]),
        shape_rms_abs_error=None if artifact is None else float(metrics["shape_rms_abs_error"]),
        x_point_position_abs_error_m=None if artifact is None else float(metrics["x_point_position_abs_error_m"]),
        x_point_flux_abs_error=None if artifact is None else float(metrics["x_point_flux_abs_error"]),
        divertor_rms_abs_error=None if artifact is None else float(metrics["divertor_rms_abs_error"]),
        coil_current_relative_error=None if artifact is None else float(metrics["coil_current_relative_error"]),
        shape_rms_abs_tolerance=shape_tol,
        x_point_position_abs_tolerance_m=x_tol,
        x_point_flux_abs_tolerance=flux_tol,
        divertor_rms_abs_tolerance=divertor_tol,
        coil_current_relative_tolerance=coil_tol,
        facility_claim_allowed=facility_claim_allowed,
        claim_status=claim_status,
    )


def assert_free_boundary_tracking_facility_claim_admissible(evidence: FreeBoundaryTrackingClaimEvidence) -> None:
    """Raise when free-boundary evidence is insufficient for facility-control claims."""
    if not evidence.facility_claim_allowed:
        raise ValueError("facility free-boundary tracking claim requires matched reference artifact evidence")


def save_free_boundary_tracking_claim_evidence(evidence: FreeBoundaryTrackingClaimEvidence, path: str | Path) -> None:
    """Persist free-boundary claim evidence as deterministic JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "FreeBoundaryTrackingClaimEvidence",
    "assert_free_boundary_tracking_facility_claim_admissible",
    "free_boundary_tracking_claim_evidence",
    "save_free_boundary_tracking_claim_evidence",
]
