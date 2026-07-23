# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport evidence and claim contracts

"""Campaign metadata, gradient-audit certificates, and readiness evidence.

This module owns the serialisable provenance and fail-closed claim surfaces for
differentiable transport campaigns. The numerical step/rollout/AD facade remains
in :mod:`scpn_control.core.differentiable_transport`, which constructs
:class:`TransportCampaignMetadata` after validating transport arrays and re-exports
these symbols for a stable public import path.

Public claim builders never invent physics: they bind digests of campaign
metadata, finite-difference audits, and optional latency reports, and they fail
closed when digests, backends, grids, or external-admission flags disagree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CHANNEL_COUNT = 4
CHANNELS = (
    "electron_temperature",
    "ion_temperature",
    "electron_density",
    "impurity_density",
)
_TRANSPORT_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TransportRolloutGradientAudit:
    """Finite-difference audit of multi-step source-rollout gradients."""

    loss: float
    epsilon: float
    tolerance: float
    checked_indices: tuple[tuple[int, int, int], ...]
    source_max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class TransportGradientAudit:
    """Finite-difference audit of differentiable transport tuning gradients."""

    loss: float
    epsilon: float
    tolerance: float
    checked_indices: tuple[tuple[int, int], ...]
    chi_max_abs_error: float
    source_max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class TransportRuntimeMetadata:
    """Runtime provenance for differentiable transport latency evidence."""

    schema_version: int
    measured_at_unix_s: float
    python_version: str
    platform: str
    machine: str
    processor: str
    jax_version: str
    jaxlib_version: str
    jax_default_backend: str
    jax_devices: tuple[str, ...]
    jax_enable_x64: bool


@dataclass(frozen=True)
class TransportGradientLatencyReport:
    """Latency evidence for audited differentiable transport tuning gradients."""

    schema_version: int
    backend: str
    dtype: str
    n_rho: int
    channel_count: int
    warmup_runs: int
    timed_runs: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    runtime_metadata: TransportRuntimeMetadata
    audit: TransportGradientAudit
    claim_status: str


@dataclass(frozen=True)
class TransportRolloutGradientLatencyReport:
    """Latency evidence for audited multi-step source-rollout gradients."""

    schema_version: int
    backend: str
    dtype: str
    n_rho: int
    n_steps: int
    channel_count: int
    warmup_runs: int
    timed_runs: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    runtime_metadata: TransportRuntimeMetadata
    audit: TransportRolloutGradientAudit
    claim_status: str


@dataclass(frozen=True)
class TransportCampaignMetadata:
    """Validated provenance for a differentiable transport tuning campaign."""

    backend: str
    dtype: str
    channel_order: tuple[str, ...]
    n_rho: int
    rho_min: float
    rho_max: float
    rho_spacing: float
    dt: float
    core_boundary: str
    edge_boundary: str
    edge_values: tuple[float, ...]
    closure_source: str | None
    closure_weights_checksum: str | None
    gradient_tolerance: float | None
    equilibrium_grid_shape: tuple[int, int] | None


@dataclass(frozen=True)
class TransportDifferentiabilityEvidence:
    """Tamper-evident admission evidence for differentiable transport gradients."""

    schema_version: int
    backend: str
    campaign_sha256: str
    gradient_audit_sha256: str
    gradient_tolerance: float
    audit_kind: str
    audit_passed: bool
    n_rho: int
    channel_order: tuple[str, ...]
    equilibrium_coupled: bool
    controller_formal_artifact_sha256: str | None
    claim_status: str


@dataclass(frozen=True)
class TransportFullFidelityReadinessEvidence:
    """Fail-closed readiness evidence for full-fidelity transport claims."""

    schema_version: int
    backend: str
    campaign_sha256: str
    gradient_latency_report_sha256: str
    gradient_audit_sha256: str
    rollout_latency_report_sha256: str | None
    rollout_audit_sha256: str | None
    external_reference_artifact_sha256: str | None
    external_reference_admitted: bool
    controller_formal_artifact_sha256: str | None
    n_rho: int
    rollout_steps: int | None
    channel_order: tuple[str, ...]
    equilibrium_coupled: bool
    full_fidelity_claim_admissible: bool
    blocked_reasons: tuple[str, ...]
    claim_status: str


def _canonical_sha256(value: Any) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_optional_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _is_sha256_hex(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _finite_float_field(name: str, value: Any, *, positive: bool = False) -> float:
    field = float(value)
    if not np.isfinite(field):
        raise ValueError(f"metadata field {name} must be finite")
    if positive and field <= 0.0:
        raise ValueError(f"metadata field {name} must be positive")
    return field


def _optional_positive_float_field(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float_field(name, value, positive=True)


def _transport_campaign_metadata_from_mapping(payload: dict[str, Any]) -> TransportCampaignMetadata:
    try:
        backend = str(payload["backend"]).strip().lower()
        dtype = str(payload["dtype"])
        channel_order = tuple(str(channel) for channel in payload["channel_order"])
        n_rho = int(payload["n_rho"])
        rho_min = _finite_float_field("rho_min", payload["rho_min"])
        rho_max = _finite_float_field("rho_max", payload["rho_max"])
        rho_spacing = _finite_float_field("rho_spacing", payload["rho_spacing"], positive=True)
        dt_value = _finite_float_field("dt", payload["dt"], positive=True)
        core_boundary = str(payload["core_boundary"])
        edge_boundary = str(payload["edge_boundary"])
        edge_values = tuple(_finite_float_field("edge_values", value) for value in payload["edge_values"])
        closure_source_value = payload["closure_source"]
        closure_checksum_value = payload["closure_weights_checksum"]
        tolerance = _optional_positive_float_field("gradient_tolerance", payload["gradient_tolerance"])
        equilibrium_shape_value = payload["equilibrium_grid_shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transport campaign metadata payload is malformed") from exc

    if backend not in {"numpy", "jax"}:
        raise ValueError("transport campaign metadata backend is invalid")
    if channel_order != CHANNELS:
        raise ValueError("transport campaign metadata channel_order is invalid")
    if n_rho < 3:
        raise ValueError("transport campaign metadata n_rho must be >= 3")
    if rho_max <= rho_min:
        raise ValueError("transport campaign metadata rho bounds are invalid")
    if len(edge_values) != CHANNEL_COUNT:
        raise ValueError("transport campaign metadata edge_values length is invalid")
    if core_boundary != "zero_gradient" or edge_boundary != "dirichlet":
        raise ValueError("transport campaign metadata boundary contract is invalid")
    closure_source = None if closure_source_value is None else str(closure_source_value)
    closure_weights_checksum = None if closure_checksum_value is None else str(closure_checksum_value)
    equilibrium_grid_shape: tuple[int, int] | None = None
    if equilibrium_shape_value is not None:
        if not isinstance(equilibrium_shape_value, list | tuple) or len(equilibrium_shape_value) != 2:
            raise ValueError("transport campaign metadata equilibrium_grid_shape is invalid")
        equilibrium_grid_shape = (int(equilibrium_shape_value[0]), int(equilibrium_shape_value[1]))
        if min(equilibrium_grid_shape) < 3:
            raise ValueError("transport campaign metadata equilibrium_grid_shape must be >= 3 in both dimensions")

    return TransportCampaignMetadata(
        backend=backend,
        dtype=dtype,
        channel_order=channel_order,
        n_rho=n_rho,
        rho_min=rho_min,
        rho_max=rho_max,
        rho_spacing=rho_spacing,
        dt=dt_value,
        core_boundary=core_boundary,
        edge_boundary=edge_boundary,
        edge_values=edge_values,
        closure_source=closure_source,
        closure_weights_checksum=closure_weights_checksum,
        gradient_tolerance=tolerance,
        equilibrium_grid_shape=equilibrium_grid_shape,
    )


def save_transport_campaign_metadata(metadata: TransportCampaignMetadata, path: str | Path) -> None:
    """Persist transport campaign metadata as schema-versioned JSON."""
    destination = Path(path)
    payload = {
        "schema_version": _TRANSPORT_METADATA_SCHEMA_VERSION,
        "metadata": asdict(metadata),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_transport_campaign_metadata(path: str | Path) -> TransportCampaignMetadata:
    """Load and validate schema-versioned transport campaign metadata JSON."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("transport campaign metadata file is not readable JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _TRANSPORT_METADATA_SCHEMA_VERSION:
        raise ValueError("transport campaign metadata schema_version is unsupported")
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError("transport campaign metadata payload is malformed")
    return _transport_campaign_metadata_from_mapping(metadata_payload)


def _metadata_field_matches(archived: Any, current: Any) -> bool:
    if archived is None or current is None:
        return archived is current
    if isinstance(archived, float | int) and isinstance(current, float | int):
        return bool(np.isclose(float(archived), float(current), rtol=1.0e-12, atol=1.0e-15))
    if isinstance(archived, tuple) and isinstance(current, tuple):
        if len(archived) != len(current):
            return False
        return all(_metadata_field_matches(left, right) for left, right in zip(archived, current, strict=True))
    return bool(archived == current)


def transport_differentiability_evidence(
    metadata: TransportCampaignMetadata,
    audit: TransportGradientAudit | TransportRolloutGradientAudit,
    *,
    controller_formal_artifact_sha256: str | None = None,
) -> TransportDifferentiabilityEvidence:
    """Build tamper-evident evidence for audited differentiable transport."""
    if not isinstance(metadata, TransportCampaignMetadata):
        raise ValueError("metadata must be TransportCampaignMetadata")
    if not isinstance(audit, TransportGradientAudit | TransportRolloutGradientAudit):
        raise ValueError("audit must be a transport gradient audit result")
    if metadata.gradient_tolerance is None:
        raise ValueError("metadata.gradient_tolerance is required for differentiability evidence")
    _validate_transport_gradient_audit(metadata, audit)
    proof_digest = _validate_optional_sha256(
        "controller_formal_artifact_sha256",
        controller_formal_artifact_sha256,
    )
    audit_kind = "rollout_source_gradient" if isinstance(audit, TransportRolloutGradientAudit) else "parameter_gradient"
    return TransportDifferentiabilityEvidence(
        schema_version=1,
        backend=metadata.backend,
        campaign_sha256=_canonical_sha256(metadata),
        gradient_audit_sha256=_canonical_sha256(audit),
        gradient_tolerance=float(metadata.gradient_tolerance),
        audit_kind=audit_kind,
        audit_passed=bool(audit.passed),
        n_rho=int(metadata.n_rho),
        channel_order=metadata.channel_order,
        equilibrium_coupled=metadata.equilibrium_grid_shape is not None,
        controller_formal_artifact_sha256=proof_digest,
        claim_status=(
            "bounded audited differentiable transport evidence only; "
            "not external transport validation or hardware timing evidence"
        ),
    )


def assert_transport_differentiability_claim_admissible(
    evidence: TransportDifferentiabilityEvidence,
    metadata: TransportCampaignMetadata,
    audit: TransportGradientAudit | TransportRolloutGradientAudit,
) -> TransportDifferentiabilityEvidence:
    """Fail closed unless differentiable transport evidence matches inputs."""
    if not isinstance(evidence, TransportDifferentiabilityEvidence):
        raise ValueError("evidence must be TransportDifferentiabilityEvidence")
    if evidence.schema_version != 1:
        raise ValueError("transport differentiability evidence schema_version is unsupported")
    if metadata.backend != "jax" or evidence.backend != "jax":
        raise ValueError("transport differentiability evidence requires JAX backend")
    if not audit.passed or not evidence.audit_passed:
        raise ValueError("transport differentiability evidence requires a passed audit")
    if metadata.gradient_tolerance is None:
        raise ValueError("transport differentiability evidence requires metadata.gradient_tolerance")
    _validate_transport_gradient_audit(metadata, audit)
    if evidence.campaign_sha256 != _canonical_sha256(metadata):
        raise ValueError("transport differentiability evidence campaign_sha256 mismatch")
    if evidence.gradient_audit_sha256 != _canonical_sha256(audit):
        raise ValueError("transport differentiability evidence gradient_audit_sha256 mismatch")
    if evidence.channel_order != metadata.channel_order or evidence.channel_order != CHANNELS:
        raise ValueError("transport differentiability evidence channel_order mismatch")
    if evidence.n_rho != metadata.n_rho:
        raise ValueError("transport differentiability evidence n_rho mismatch")
    if evidence.equilibrium_coupled != (metadata.equilibrium_grid_shape is not None):
        raise ValueError("transport differentiability evidence equilibrium_coupled mismatch")
    if not np.isclose(evidence.gradient_tolerance, metadata.gradient_tolerance, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("transport differentiability evidence gradient_tolerance mismatch")
    _validate_optional_sha256("controller_formal_artifact_sha256", evidence.controller_formal_artifact_sha256)
    return evidence


def transport_full_fidelity_readiness_evidence(
    metadata: TransportCampaignMetadata,
    gradient_report: TransportGradientLatencyReport,
    *,
    rollout_report: TransportRolloutGradientLatencyReport | None = None,
    external_reference_artifact_sha256: str | None = None,
    external_reference_admitted: bool = False,
    controller_formal_artifact_sha256: str | None = None,
) -> TransportFullFidelityReadinessEvidence:
    """Build fail-closed readiness evidence for full-fidelity transport claims.

    Local gradient audits and timing reports are necessary but not sufficient
    for a full-fidelity claim. This certificate requires a JAX campaign,
    equilibrium coupling, one-step and rollout audit reports, a controller proof
    digest, and an independently admitted external reference artefact.
    """
    if not isinstance(metadata, TransportCampaignMetadata):
        raise ValueError("metadata must be TransportCampaignMetadata")
    if not isinstance(gradient_report, TransportGradientLatencyReport):
        raise ValueError("gradient_report must be TransportGradientLatencyReport")
    if rollout_report is not None and not isinstance(rollout_report, TransportRolloutGradientLatencyReport):
        raise ValueError("rollout_report must be TransportRolloutGradientLatencyReport")
    if not isinstance(external_reference_admitted, bool):
        raise ValueError("external_reference_admitted must be boolean")

    _validate_transport_gradient_latency_report(gradient_report)
    _assert_latency_report_matches_campaign(metadata, gradient_report, report_name="gradient latency report")
    _validate_transport_gradient_audit(metadata, gradient_report.audit)

    rollout_steps: int | None = None
    rollout_report_sha256: str | None = None
    rollout_audit_sha256: str | None = None
    if rollout_report is not None:
        _validate_transport_rollout_gradient_latency_report(rollout_report)
        _assert_latency_report_matches_campaign(metadata, rollout_report, report_name="rollout latency report")
        _validate_transport_gradient_audit(metadata, rollout_report.audit)
        for step, _, rho_index in rollout_report.audit.checked_indices:
            if step < 0 or step >= rollout_report.n_steps or rho_index < 0 or rho_index >= metadata.n_rho:
                raise ValueError("rollout latency report audit indices exceed campaign metadata bounds")
        rollout_steps = int(rollout_report.n_steps)
        rollout_report_sha256 = _canonical_sha256(rollout_report)
        rollout_audit_sha256 = _canonical_sha256(rollout_report.audit)

    external_digest = _validate_optional_sha256(
        "external_reference_artifact_sha256",
        external_reference_artifact_sha256,
    )
    proof_digest = _validate_optional_sha256(
        "controller_formal_artifact_sha256",
        controller_formal_artifact_sha256,
    )

    blocked_reasons: list[str] = []
    if metadata.equilibrium_grid_shape is None:
        blocked_reasons.append("equilibrium_coupled_campaign")
    if not gradient_report.audit.passed:
        blocked_reasons.append("gradient_latency_audit")
    if rollout_report is None:
        blocked_reasons.append("rollout_latency_report")
    elif not rollout_report.audit.passed:
        blocked_reasons.append("rollout_latency_audit")
    if proof_digest is None:
        blocked_reasons.append("controller_formal_artifact_sha256")
    if external_digest is None:
        blocked_reasons.append("external_reference_artifact_sha256")
    elif not external_reference_admitted:
        blocked_reasons.append("external_reference_admission")

    full_fidelity_admissible = len(blocked_reasons) == 0
    return TransportFullFidelityReadinessEvidence(
        schema_version=1,
        backend=metadata.backend,
        campaign_sha256=_canonical_sha256(metadata),
        gradient_latency_report_sha256=_canonical_sha256(gradient_report),
        gradient_audit_sha256=_canonical_sha256(gradient_report.audit),
        rollout_latency_report_sha256=rollout_report_sha256,
        rollout_audit_sha256=rollout_audit_sha256,
        external_reference_artifact_sha256=external_digest,
        external_reference_admitted=external_reference_admitted,
        controller_formal_artifact_sha256=proof_digest,
        n_rho=int(metadata.n_rho),
        rollout_steps=rollout_steps,
        channel_order=metadata.channel_order,
        equilibrium_coupled=metadata.equilibrium_grid_shape is not None,
        full_fidelity_claim_admissible=full_fidelity_admissible,
        blocked_reasons=tuple(blocked_reasons),
        claim_status=(
            "full-fidelity differentiable transport claim admitted"
            if full_fidelity_admissible
            else "bounded differentiable transport readiness only; full-fidelity claim remains blocked"
        ),
    )


def assert_transport_full_fidelity_claim_ready(
    evidence: TransportFullFidelityReadinessEvidence,
    metadata: TransportCampaignMetadata,
    gradient_report: TransportGradientLatencyReport,
    *,
    rollout_report: TransportRolloutGradientLatencyReport | None = None,
) -> TransportFullFidelityReadinessEvidence:
    """Fail closed unless readiness evidence admits a full-fidelity claim."""
    if not isinstance(evidence, TransportFullFidelityReadinessEvidence):
        raise ValueError("evidence must be TransportFullFidelityReadinessEvidence")
    if evidence.schema_version != 1:
        raise ValueError("transport full-fidelity readiness schema_version is unsupported")
    expected = transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=rollout_report,
        external_reference_artifact_sha256=evidence.external_reference_artifact_sha256,
        external_reference_admitted=evidence.external_reference_admitted,
        controller_formal_artifact_sha256=evidence.controller_formal_artifact_sha256,
    )
    if evidence != expected:
        raise ValueError("transport full-fidelity readiness evidence digest mismatch")
    if not evidence.full_fidelity_claim_admissible:
        reasons = ", ".join(evidence.blocked_reasons)
        if "external_reference_artifact_sha256" in evidence.blocked_reasons:
            raise ValueError(f"transport full-fidelity claim requires external reference evidence: {reasons}")
        if "external_reference_admission" in evidence.blocked_reasons:
            raise ValueError(f"transport full-fidelity claim requires external reference admission: {reasons}")
        raise ValueError(f"transport full-fidelity claim is not ready: {reasons}")
    return evidence


def _assert_latency_report_matches_campaign(
    metadata: TransportCampaignMetadata,
    report: TransportGradientLatencyReport | TransportRolloutGradientLatencyReport,
    *,
    report_name: str,
) -> None:
    if report.backend != metadata.backend:
        raise ValueError(f"campaign metadata and {report_name} backend mismatch")
    if report.dtype != metadata.dtype:
        raise ValueError(f"campaign metadata and {report_name} dtype mismatch")
    if report.n_rho != metadata.n_rho:
        raise ValueError(f"campaign metadata and {report_name} n_rho mismatch")
    if report.channel_count != len(metadata.channel_order) or metadata.channel_order != CHANNELS:
        raise ValueError(f"campaign metadata and {report_name} channel contract mismatch")
    if metadata.gradient_tolerance is None:
        raise ValueError("campaign metadata gradient_tolerance is required for latency evidence")
    if not np.isclose(report.audit.tolerance, metadata.gradient_tolerance, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(f"campaign metadata and {report_name} audit tolerance mismatch")


def _validate_transport_gradient_audit(
    metadata: TransportCampaignMetadata,
    audit: TransportGradientAudit | TransportRolloutGradientAudit,
) -> None:
    if metadata.gradient_tolerance is None:
        raise ValueError("transport differentiability evidence requires metadata.gradient_tolerance")
    loss = float(audit.loss)
    epsilon = float(audit.epsilon)
    tolerance = float(audit.tolerance)
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError("transport gradient audit loss must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("transport gradient audit epsilon must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("transport gradient audit tolerance must be positive and finite")
    if not np.isclose(tolerance, metadata.gradient_tolerance, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("transport gradient audit tolerance must match campaign metadata")
    if not isinstance(audit.passed, bool):
        raise ValueError("transport gradient audit passed flag must be boolean")
    if isinstance(audit, TransportRolloutGradientAudit):
        _validate_rollout_audit_indices(audit.checked_indices, metadata.n_rho)
        max_error = float(audit.source_max_abs_error)
    else:
        _validate_parameter_audit_indices(audit.checked_indices, metadata.n_rho)
        chi_error = float(audit.chi_max_abs_error)
        source_error = float(audit.source_max_abs_error)
        if not np.isfinite(chi_error) or chi_error < 0.0:
            raise ValueError("transport gradient audit chi_max_abs_error must be finite and non-negative")
        max_error = max(chi_error, source_error)
    if not np.isfinite(max_error) or max_error < 0.0:
        raise ValueError("transport gradient audit source_max_abs_error must be finite and non-negative")
    if audit.passed != bool(max_error <= tolerance):
        raise ValueError("transport gradient audit passed flag is inconsistent with tolerance")


def _validate_parameter_audit_indices(indices: tuple[tuple[int, int], ...], n_rho: int) -> None:
    if not indices:
        raise ValueError("transport gradient audit checked_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("transport gradient audit checked_indices must be unique")
    for channel, radius in indices:
        if not (0 <= int(channel) < CHANNEL_COUNT and 0 <= int(radius) < int(n_rho)):
            raise ValueError("transport gradient audit checked_indices out of campaign bounds")


def _validate_rollout_audit_indices(indices: tuple[tuple[int, int, int], ...], n_rho: int) -> None:
    if not indices:
        raise ValueError("transport rollout gradient audit checked_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("transport rollout gradient audit checked_indices must be unique")
    for step, channel, radius in indices:
        if int(step) < 0 or not (0 <= int(channel) < CHANNEL_COUNT and 0 <= int(radius) < int(n_rho)):
            raise ValueError("transport rollout gradient audit checked_indices out of campaign bounds")


def save_transport_gradient_latency_report(report: TransportGradientLatencyReport, path: str | Path) -> None:
    """Persist differentiable transport gradient-latency evidence as JSON."""
    _validate_transport_gradient_latency_report(report)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_transport_rollout_gradient_latency_report(
    report: TransportRolloutGradientLatencyReport,
    path: str | Path,
) -> None:
    """Persist rollout source-gradient latency evidence as JSON."""
    _validate_transport_rollout_gradient_latency_report(report)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_transport_gradient_latency_report(report: TransportGradientLatencyReport) -> None:
    if not isinstance(report, TransportGradientLatencyReport):
        raise ValueError("transport gradient latency report must be TransportGradientLatencyReport")
    _validate_latency_common(
        schema_version=report.schema_version,
        backend=report.backend,
        n_rho=report.n_rho,
        channel_count=report.channel_count,
        warmup_runs=report.warmup_runs,
        timed_runs=report.timed_runs,
        p50_ms=report.p50_ms,
        p95_ms=report.p95_ms,
        max_ms=report.max_ms,
    )
    _validate_transport_runtime_metadata(report.runtime_metadata)
    metadata = TransportCampaignMetadata(
        backend=report.backend,
        dtype=report.dtype,
        channel_order=CHANNELS,
        n_rho=report.n_rho,
        rho_min=0.0,
        rho_max=1.0,
        rho_spacing=1.0 / float(report.n_rho - 1),
        dt=1.0,
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=(0.0, 0.0, 0.0, 0.0),
        closure_source=None,
        closure_weights_checksum=None,
        gradient_tolerance=report.audit.tolerance,
        equilibrium_grid_shape=None,
    )
    _validate_transport_gradient_audit(metadata, report.audit)


def _validate_transport_rollout_gradient_latency_report(report: TransportRolloutGradientLatencyReport) -> None:
    if not isinstance(report, TransportRolloutGradientLatencyReport):
        raise ValueError("transport rollout gradient latency report must be TransportRolloutGradientLatencyReport")
    _validate_latency_common(
        schema_version=report.schema_version,
        backend=report.backend,
        n_rho=report.n_rho,
        channel_count=report.channel_count,
        warmup_runs=report.warmup_runs,
        timed_runs=report.timed_runs,
        p50_ms=report.p50_ms,
        p95_ms=report.p95_ms,
        max_ms=report.max_ms,
    )
    _validate_transport_runtime_metadata(report.runtime_metadata)
    _require_int("n_steps", report.n_steps, minimum=1)
    metadata = TransportCampaignMetadata(
        backend=report.backend,
        dtype=report.dtype,
        channel_order=CHANNELS,
        n_rho=report.n_rho,
        rho_min=0.0,
        rho_max=1.0,
        rho_spacing=1.0 / float(report.n_rho - 1),
        dt=1.0,
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=(0.0, 0.0, 0.0, 0.0),
        closure_source=None,
        closure_weights_checksum=None,
        gradient_tolerance=report.audit.tolerance,
        equilibrium_grid_shape=None,
    )
    _validate_transport_gradient_audit(metadata, report.audit)


def _validate_latency_common(
    *,
    schema_version: int,
    backend: str,
    n_rho: int,
    channel_count: int,
    warmup_runs: int,
    timed_runs: int,
    p50_ms: float,
    p95_ms: float,
    max_ms: float,
) -> None:
    if schema_version != 1:
        raise ValueError("transport latency report schema_version is unsupported")
    if backend != "jax":
        raise ValueError("transport latency report requires JAX backend")
    _require_int("n_rho", n_rho, minimum=3)
    if _require_int("channel_count", channel_count, minimum=1) != CHANNEL_COUNT:
        raise ValueError("transport latency report channel_count is invalid")
    _require_int("warmup_runs", warmup_runs, minimum=0)
    _require_int("timed_runs", timed_runs, minimum=1)
    p50 = _require_nonnegative_finite("p50_ms", p50_ms)
    p95 = _require_nonnegative_finite("p95_ms", p95_ms)
    maximum = _require_nonnegative_finite("max_ms", max_ms)
    if not (p50 <= p95 <= maximum):
        raise ValueError("transport latency report percentiles must satisfy p50 <= p95 <= max")


def _validate_transport_runtime_metadata(metadata: TransportRuntimeMetadata) -> None:
    if not isinstance(metadata, TransportRuntimeMetadata):
        raise ValueError("transport latency report runtime_metadata is invalid")
    if metadata.schema_version != 1:
        raise ValueError("transport runtime metadata schema_version is unsupported")
    _require_nonnegative_finite("measured_at_unix_s", metadata.measured_at_unix_s)
    for name in (
        "python_version",
        "platform",
        "machine",
        "jax_version",
        "jaxlib_version",
        "jax_default_backend",
    ):
        value = getattr(metadata, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"transport runtime metadata {name} must be a non-empty string")
    if not isinstance(metadata.processor, str):
        raise ValueError("transport runtime metadata processor must be a string")
    if not isinstance(metadata.jax_devices, tuple) or not metadata.jax_devices:
        raise ValueError("transport runtime metadata jax_devices must be a non-empty tuple")
    if not all(isinstance(device, str) and device for device in metadata.jax_devices):
        raise ValueError("transport runtime metadata jax_devices must contain non-empty strings")
    if not isinstance(metadata.jax_enable_x64, bool):
        raise ValueError("transport runtime metadata jax_enable_x64 must be boolean")


def _require_nonnegative_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_int(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value
