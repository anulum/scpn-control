# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Safety Case Evidence
"""Bounded controller safety-case evidence chaining.

This module defines the named workflow contract that binds a safety-critical
compiled controller artifact, audited differentiable-transport evidence, and
bounded digital-twin online-update evidence into one tamper-evident admission
record. It is a repository safety-case package boundary, not an external
certification claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from scpn_control.control.codac_interface import load_codac_runtime_evidence
from scpn_control.control.digital_twin_online_update import DigitalTwinUpdateEvidence
from scpn_control.core.differentiable_transport import TransportDifferentiabilityEvidence
from scpn_control.scpn.artifact import (
    Artifact,
    ArtifactValidationError,
    compute_artifact_payload_sha256,
    validate_safety_critical_artifact,
)
from scpn_control.scpn.fpga_export import load_hdl_export_evidence
from scpn_control.phase.ws_phase_stream import load_websocket_runtime_evidence

_SAFETY_CASE_MANIFEST_SCHEMA_VERSION = 1
_READINESS_ARTIFACT_KINDS = (
    "external_physics_validation",
    "target_hardware_timing",
    "hil_replay_evidence",
    "hdl_export_evidence",
    "codac_runtime_evidence",
    "websocket_runtime_evidence",
    "independent_safety_review",
)
_READINESS_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class ControllerSafetyCaseEvidence:
    """Tamper-evident bounded controller safety-case evidence bundle."""

    schema_version: int
    controller_artifact_sha256: str
    formal_report_sha256: str
    formal_backend: str
    formal_max_depth: int
    transport_evidence_sha256: str
    digital_twin_evidence_sha256: str
    claim_status: str


@dataclass(frozen=True)
class SafetyCaseReadinessEvidence:
    """Promotion-readiness gate for a bounded controller safety-case bundle."""

    schema_version: int
    safety_case_sha256: str
    status: str
    external_physics_validation_sha256: str | None
    target_hardware_timing_sha256: str | None
    hil_replay_evidence_sha256: str | None
    hdl_export_evidence_sha256: str | None
    codac_runtime_evidence_sha256: str | None
    websocket_runtime_evidence_sha256: str | None
    independent_safety_review_sha256: str | None
    blocking_reasons: tuple[str, ...]
    claim_status: str


@dataclass(frozen=True)
class ReadinessArtifactEvidence:
    """Typed artifact evidence required for safety-case promotion readiness."""

    kind: str
    artifact_sha256: str
    artifact_uri: str
    producer: str
    generated_utc: str


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


def _optional_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _is_sha256_hex(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _safe_relative_artifact_uri(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty safe relative path")
    if "\\" in value or "://" in value or "//" in value or value.startswith(("file:", "/", "~")):
        raise ValueError(f"{name} must be a safe relative path")
    rel = PurePosixPath(value)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"{name} must be a safe relative path")
    return value


def _validate_readiness_artifact(artifact: ReadinessArtifactEvidence) -> ReadinessArtifactEvidence:
    if not isinstance(artifact, ReadinessArtifactEvidence):
        raise ValueError("readiness artifact must be ReadinessArtifactEvidence")
    if artifact.kind not in _READINESS_ARTIFACT_KINDS:
        raise ValueError("readiness artifact kind is unsupported")
    if not _is_sha256_hex(artifact.artifact_sha256):
        raise ValueError("readiness artifact artifact_sha256 must be a SHA-256 digest")
    _safe_relative_artifact_uri("readiness artifact artifact_uri", artifact.artifact_uri)
    if not isinstance(artifact.producer, str) or not artifact.producer:
        raise ValueError("readiness artifact producer must be a non-empty string")
    if not isinstance(artifact.generated_utc, str) or not artifact.generated_utc:
        raise ValueError("readiness artifact generated_utc must be a non-empty string")
    return artifact


def _resolve_readiness_artifact_path(
    artifact: ReadinessArtifactEvidence,
    artifact_root: str | Path,
) -> Path:
    root = Path(artifact_root).resolve()
    rel = Path(artifact.artifact_uri)
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("readiness artifact artifact_uri escapes artifact_root") from exc
    if not candidate.is_file():
        raise ValueError("readiness artifact artifact_uri does not resolve to a file")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != artifact.artifact_sha256.lower():
        raise ValueError("readiness artifact artifact_sha256 does not match artifact file")
    return candidate


def _validate_target_hardware_timing_artifact(
    artifact: ReadinessArtifactEvidence,
    artifact_root: str | Path,
    *,
    max_e2e_p95_us: float,
) -> None:
    report_path = _resolve_readiness_artifact_path(artifact, artifact_root)
    from validation.validate_e2e_latency_evidence import validate_e2e_latency_evidence

    result = validate_e2e_latency_evidence(
        report_path,
        require_target_hardware=True,
        max_e2e_p95_us=max_e2e_p95_us,
    )
    if result.status != "pass":
        raise ValueError("target hardware timing artifact is not admissible: " + "; ".join(result.errors))


def _validate_hil_replay_artifact(
    artifact: ReadinessArtifactEvidence,
    artifact_root: str | Path,
) -> None:
    report_path = _resolve_readiness_artifact_path(artifact, artifact_root)
    from scpn_control.control.hil_harness import load_hil_replay_evidence

    try:
        load_hil_replay_evidence(report_path, require_target_hardware=True)
    except ValueError as exc:
        raise ValueError(f"HIL replay artifact is not admissible: {exc}") from exc


def _validate_codac_runtime_artifact(
    artifact: ReadinessArtifactEvidence,
    artifact_root: str | Path,
) -> None:
    report_path = _resolve_readiness_artifact_path(artifact, artifact_root)
    try:
        load_codac_runtime_evidence(report_path, require_facility_claim=True)
    except ValueError as exc:
        raise ValueError(f"CODAC runtime artifact is not admissible: {exc}") from exc


def _validate_websocket_runtime_artifact(
    artifact: ReadinessArtifactEvidence,
    artifact_root: str | Path,
) -> None:
    report_path = _resolve_readiness_artifact_path(artifact, artifact_root)
    try:
        load_websocket_runtime_evidence(report_path, require_facility_claim=True)
    except ValueError as exc:
        raise ValueError(f"WebSocket runtime artifact is not admissible: {exc}") from exc


def _validate_hdl_export_artifact(
    artifact: ReadinessArtifactEvidence,
    artifact_root: str | Path,
    *,
    controller_artifact_sha256: str,
) -> None:
    report_path = _resolve_readiness_artifact_path(artifact, artifact_root)
    try:
        evidence = load_hdl_export_evidence(report_path, require_facility_claim=True, artifact_root=artifact_root)
    except ValueError as exc:
        raise ValueError(f"HDL export artifact is not admissible: {exc}") from exc
    if evidence.controller_artifact_sha256 != controller_artifact_sha256:
        raise ValueError("HDL export artifact is not bound to the safety-case controller artifact")


def _controller_safety_case_evidence_from_mapping(payload: dict[str, Any]) -> ControllerSafetyCaseEvidence:
    try:
        evidence = ControllerSafetyCaseEvidence(
            schema_version=int(payload["schema_version"]),
            controller_artifact_sha256=str(payload["controller_artifact_sha256"]),
            formal_report_sha256=str(payload["formal_report_sha256"]),
            formal_backend=str(payload["formal_backend"]),
            formal_max_depth=int(payload["formal_max_depth"]),
            transport_evidence_sha256=str(payload["transport_evidence_sha256"]),
            digital_twin_evidence_sha256=str(payload["digital_twin_evidence_sha256"]),
            claim_status=str(payload["claim_status"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("controller safety-case evidence payload is malformed") from exc
    if evidence.schema_version != 1:
        raise ValueError("controller safety-case evidence schema_version is unsupported")
    for field_name in (
        "controller_artifact_sha256",
        "formal_report_sha256",
        "transport_evidence_sha256",
        "digital_twin_evidence_sha256",
    ):
        value = getattr(evidence, field_name)
        if len(value) != 64:
            raise ValueError(f"controller safety-case evidence {field_name} must be a SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"controller safety-case evidence {field_name} must be a SHA-256 digest") from exc
    if evidence.formal_backend not in {"explicit-state", "z3"}:
        raise ValueError("controller safety-case evidence formal_backend is unsupported")
    if evidence.formal_max_depth < 0:
        raise ValueError("controller safety-case evidence formal_max_depth must be >= 0")
    if not evidence.claim_status or "bounded" not in evidence.claim_status.lower():
        raise ValueError("controller safety-case evidence claim_status must state a bounded boundary")
    return evidence


def _safety_case_readiness_from_mapping(payload: dict[str, Any]) -> SafetyCaseReadinessEvidence:
    try:
        readiness = SafetyCaseReadinessEvidence(
            schema_version=int(payload["schema_version"]),
            safety_case_sha256=str(payload["safety_case_sha256"]),
            status=str(payload["status"]),
            external_physics_validation_sha256=(
                None
                if payload["external_physics_validation_sha256"] is None
                else str(payload["external_physics_validation_sha256"])
            ),
            target_hardware_timing_sha256=(
                None
                if payload["target_hardware_timing_sha256"] is None
                else str(payload["target_hardware_timing_sha256"])
            ),
            hil_replay_evidence_sha256=(
                None
                if payload.get("hil_replay_evidence_sha256") is None
                else str(payload["hil_replay_evidence_sha256"])
            ),
            hdl_export_evidence_sha256=(
                None if payload["hdl_export_evidence_sha256"] is None else str(payload["hdl_export_evidence_sha256"])
            ),
            codac_runtime_evidence_sha256=(
                None
                if payload["codac_runtime_evidence_sha256"] is None
                else str(payload["codac_runtime_evidence_sha256"])
            ),
            websocket_runtime_evidence_sha256=(
                None
                if payload["websocket_runtime_evidence_sha256"] is None
                else str(payload["websocket_runtime_evidence_sha256"])
            ),
            independent_safety_review_sha256=(
                None
                if payload["independent_safety_review_sha256"] is None
                else str(payload["independent_safety_review_sha256"])
            ),
            blocking_reasons=tuple(str(reason) for reason in payload["blocking_reasons"]),
            claim_status=str(payload["claim_status"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("controller safety-case readiness payload is malformed") from exc
    if readiness.schema_version != _READINESS_SCHEMA_VERSION:
        raise ValueError("controller safety-case readiness schema_version is unsupported")
    if not _is_sha256_hex(readiness.safety_case_sha256):
        raise ValueError("controller safety-case readiness safety_case_sha256 must be a SHA-256 digest")
    _optional_sha256("external_physics_validation_sha256", readiness.external_physics_validation_sha256)
    _optional_sha256("target_hardware_timing_sha256", readiness.target_hardware_timing_sha256)
    _optional_sha256("hil_replay_evidence_sha256", readiness.hil_replay_evidence_sha256)
    _optional_sha256("hdl_export_evidence_sha256", readiness.hdl_export_evidence_sha256)
    _optional_sha256("codac_runtime_evidence_sha256", readiness.codac_runtime_evidence_sha256)
    _optional_sha256("websocket_runtime_evidence_sha256", readiness.websocket_runtime_evidence_sha256)
    _optional_sha256("independent_safety_review_sha256", readiness.independent_safety_review_sha256)
    if readiness.status not in {"blocked", "promotion_ready"}:
        raise ValueError("controller safety-case readiness status is unsupported")
    required_digests = {
        "external_physics_validation_sha256": readiness.external_physics_validation_sha256,
        "target_hardware_timing_sha256": readiness.target_hardware_timing_sha256,
        "hil_replay_evidence_sha256": readiness.hil_replay_evidence_sha256,
        "hdl_export_evidence_sha256": readiness.hdl_export_evidence_sha256,
        "codac_runtime_evidence_sha256": readiness.codac_runtime_evidence_sha256,
        "websocket_runtime_evidence_sha256": readiness.websocket_runtime_evidence_sha256,
        "independent_safety_review_sha256": readiness.independent_safety_review_sha256,
    }
    missing = tuple(name for name, value in required_digests.items() if value is None)
    if readiness.status == "promotion_ready" and readiness.blocking_reasons:
        raise ValueError("controller safety-case readiness promotion_ready cannot have blocking reasons")
    if readiness.status == "promotion_ready" and missing:
        raise ValueError("controller safety-case readiness promotion_ready is missing required evidence")
    if readiness.status == "blocked" and tuple(readiness.blocking_reasons) != missing:
        raise ValueError("controller safety-case readiness blocking_reasons must match missing evidence")
    if not readiness.claim_status or "bounded" not in readiness.claim_status.lower():
        raise ValueError("controller safety-case readiness claim_status must state a bounded boundary")
    return readiness


def evaluate_controller_safety_case_readiness(
    safety_case: ControllerSafetyCaseEvidence,
    *,
    external_physics_validation_sha256: str | None = None,
    target_hardware_timing_sha256: str | None = None,
    hil_replay_evidence_sha256: str | None = None,
    hdl_export_evidence_sha256: str | None = None,
    codac_runtime_evidence_sha256: str | None = None,
    websocket_runtime_evidence_sha256: str | None = None,
    independent_safety_review_sha256: str | None = None,
) -> SafetyCaseReadinessEvidence:
    """Evaluate whether a bounded safety-case bundle is promotion-ready.

    The linked internal evidence chain is necessary but not sufficient for
    promotion readiness. This gate requires external physics validation,
    target-hardware timing evidence, HIL replay evidence, HDL export evidence,
    CODAC/EPICS runtime evidence, WebSocket runtime evidence, and an
    independent safety review digest.
    """
    if not isinstance(safety_case, ControllerSafetyCaseEvidence):
        raise ValueError("safety_case must be ControllerSafetyCaseEvidence")
    external_digest = _optional_sha256("external_physics_validation_sha256", external_physics_validation_sha256)
    hardware_digest = _optional_sha256("target_hardware_timing_sha256", target_hardware_timing_sha256)
    hil_digest = _optional_sha256("hil_replay_evidence_sha256", hil_replay_evidence_sha256)
    hdl_digest = _optional_sha256("hdl_export_evidence_sha256", hdl_export_evidence_sha256)
    codac_digest = _optional_sha256("codac_runtime_evidence_sha256", codac_runtime_evidence_sha256)
    websocket_digest = _optional_sha256("websocket_runtime_evidence_sha256", websocket_runtime_evidence_sha256)
    review_digest = _optional_sha256("independent_safety_review_sha256", independent_safety_review_sha256)
    blocking: list[str] = []
    if external_digest is None:
        blocking.append("external_physics_validation_sha256")
    if hardware_digest is None:
        blocking.append("target_hardware_timing_sha256")
    if hil_digest is None:
        blocking.append("hil_replay_evidence_sha256")
    if hdl_digest is None:
        blocking.append("hdl_export_evidence_sha256")
    if codac_digest is None:
        blocking.append("codac_runtime_evidence_sha256")
    if websocket_digest is None:
        blocking.append("websocket_runtime_evidence_sha256")
    if review_digest is None:
        blocking.append("independent_safety_review_sha256")
    status = "promotion_ready" if not blocking else "blocked"
    return SafetyCaseReadinessEvidence(
        schema_version=_READINESS_SCHEMA_VERSION,
        safety_case_sha256=_canonical_sha256(safety_case),
        status=status,
        external_physics_validation_sha256=external_digest,
        target_hardware_timing_sha256=hardware_digest,
        hil_replay_evidence_sha256=hil_digest,
        hdl_export_evidence_sha256=hdl_digest,
        codac_runtime_evidence_sha256=codac_digest,
        websocket_runtime_evidence_sha256=websocket_digest,
        independent_safety_review_sha256=review_digest,
        blocking_reasons=tuple(blocking),
        claim_status=(
            "bounded safety-case promotion gate; external regulator certification "
            "and facility authority approval remain separate"
        ),
    )


def evaluate_controller_safety_case_readiness_from_artifacts(
    safety_case: ControllerSafetyCaseEvidence,
    artifacts: tuple[ReadinessArtifactEvidence, ...],
    *,
    artifact_root: str | Path,
    max_target_hardware_e2e_p95_us: float = 1000.0,
) -> SafetyCaseReadinessEvidence:
    """Evaluate promotion readiness from typed external evidence artifacts."""
    if not isinstance(artifacts, tuple) or not artifacts:
        raise ValueError("readiness artifacts must be a non-empty tuple")
    by_kind: dict[str, ReadinessArtifactEvidence] = {}
    for artifact in artifacts:
        parsed = _validate_readiness_artifact(artifact)
        if parsed.kind in by_kind:
            raise ValueError(f"duplicate readiness artifact kind: {parsed.kind}")
        by_kind[parsed.kind] = parsed
    missing = tuple(kind for kind in _READINESS_ARTIFACT_KINDS if kind not in by_kind)
    if missing:
        raise ValueError("missing readiness artifact kind(s): " + ", ".join(missing))
    for kind, artifact in by_kind.items():
        if kind == "target_hardware_timing":
            _validate_target_hardware_timing_artifact(
                artifact,
                artifact_root,
                max_e2e_p95_us=max_target_hardware_e2e_p95_us,
            )
        elif kind == "hil_replay_evidence":
            _validate_hil_replay_artifact(artifact, artifact_root)
        elif kind == "hdl_export_evidence":
            _validate_hdl_export_artifact(
                artifact,
                artifact_root,
                controller_artifact_sha256=safety_case.controller_artifact_sha256,
            )
        elif kind == "codac_runtime_evidence":
            _validate_codac_runtime_artifact(artifact, artifact_root)
        elif kind == "websocket_runtime_evidence":
            _validate_websocket_runtime_artifact(artifact, artifact_root)
        else:
            _resolve_readiness_artifact_path(artifact, artifact_root)
    return evaluate_controller_safety_case_readiness(
        safety_case,
        external_physics_validation_sha256=by_kind["external_physics_validation"].artifact_sha256,
        target_hardware_timing_sha256=by_kind["target_hardware_timing"].artifact_sha256,
        hil_replay_evidence_sha256=by_kind["hil_replay_evidence"].artifact_sha256,
        hdl_export_evidence_sha256=by_kind["hdl_export_evidence"].artifact_sha256,
        codac_runtime_evidence_sha256=by_kind["codac_runtime_evidence"].artifact_sha256,
        websocket_runtime_evidence_sha256=by_kind["websocket_runtime_evidence"].artifact_sha256,
        independent_safety_review_sha256=by_kind["independent_safety_review"].artifact_sha256,
    )


def save_controller_safety_case_readiness(readiness: SafetyCaseReadinessEvidence, path: str | Path) -> None:
    """Persist controller safety-case readiness with an integrity digest."""
    if not isinstance(readiness, SafetyCaseReadinessEvidence):
        raise ValueError("readiness must be SafetyCaseReadinessEvidence")
    parsed = _safety_case_readiness_from_mapping(asdict(readiness))
    manifest = {
        "schema_version": _SAFETY_CASE_MANIFEST_SCHEMA_VERSION,
        "readiness": asdict(parsed),
        "integrity_sha256": _canonical_sha256(parsed),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_controller_safety_case_readiness(path: str | Path) -> SafetyCaseReadinessEvidence:
    """Load controller safety-case readiness and verify manifest integrity."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("controller safety-case readiness manifest is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("controller safety-case readiness manifest must be a JSON object")
    if payload.get("schema_version") != _SAFETY_CASE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("controller safety-case readiness manifest schema_version is unsupported")
    readiness_payload = payload.get("readiness")
    if not isinstance(readiness_payload, dict):
        raise ValueError("controller safety-case readiness manifest payload is malformed")
    readiness = _safety_case_readiness_from_mapping(readiness_payload)
    integrity = payload.get("integrity_sha256")
    if not isinstance(integrity, str) or integrity != _canonical_sha256(readiness):
        raise ValueError("controller safety-case readiness manifest integrity digest mismatch")
    return readiness


def assert_controller_safety_case_readiness_admissible(
    readiness: SafetyCaseReadinessEvidence,
    safety_case: ControllerSafetyCaseEvidence,
) -> SafetyCaseReadinessEvidence:
    """Fail closed unless readiness evidence matches the safety case and is complete."""
    if not isinstance(readiness, SafetyCaseReadinessEvidence):
        raise ValueError("readiness must be SafetyCaseReadinessEvidence")
    if not isinstance(safety_case, ControllerSafetyCaseEvidence):
        raise ValueError("safety_case must be ControllerSafetyCaseEvidence")
    if readiness.schema_version != _READINESS_SCHEMA_VERSION:
        raise ValueError("controller safety-case readiness schema_version is unsupported")
    if readiness.safety_case_sha256 != _canonical_sha256(safety_case):
        raise ValueError("controller safety-case readiness safety_case_sha256 mismatch")
    if readiness.status not in {"blocked", "promotion_ready"}:
        raise ValueError("controller safety-case readiness status is unsupported")
    if readiness.status == "blocked" or readiness.blocking_reasons:
        raise ValueError("controller safety-case readiness is blocked: " + ", ".join(readiness.blocking_reasons))
    recomputed = evaluate_controller_safety_case_readiness(
        safety_case,
        external_physics_validation_sha256=readiness.external_physics_validation_sha256,
        target_hardware_timing_sha256=readiness.target_hardware_timing_sha256,
        hil_replay_evidence_sha256=readiness.hil_replay_evidence_sha256,
        hdl_export_evidence_sha256=readiness.hdl_export_evidence_sha256,
        codac_runtime_evidence_sha256=readiness.codac_runtime_evidence_sha256,
        websocket_runtime_evidence_sha256=readiness.websocket_runtime_evidence_sha256,
        independent_safety_review_sha256=readiness.independent_safety_review_sha256,
    )
    if readiness != recomputed:
        raise ValueError("controller safety-case readiness evidence mismatch")
    return readiness


def _require_same_controller_binding(
    controller_sha256: str,
    transport_evidence: TransportDifferentiabilityEvidence,
    digital_twin_evidence: DigitalTwinUpdateEvidence,
) -> None:
    if transport_evidence.controller_formal_artifact_sha256 != controller_sha256:
        raise ValueError("transport evidence is not bound to the controller artifact digest")
    if digital_twin_evidence.controller_formal_artifact_sha256 != controller_sha256:
        raise ValueError("digital twin evidence is not bound to the controller artifact digest")


def controller_safety_case_evidence(
    controller_artifact: Artifact,
    transport_evidence: TransportDifferentiabilityEvidence,
    digital_twin_evidence: DigitalTwinUpdateEvidence,
) -> ControllerSafetyCaseEvidence:
    """Build a bounded safety-case evidence bundle for one controller artifact."""
    if not isinstance(controller_artifact, Artifact):
        raise ValueError("controller_artifact must be an Artifact")
    if not isinstance(transport_evidence, TransportDifferentiabilityEvidence):
        raise ValueError("transport_evidence must be TransportDifferentiabilityEvidence")
    if not isinstance(digital_twin_evidence, DigitalTwinUpdateEvidence):
        raise ValueError("digital_twin_evidence must be DigitalTwinUpdateEvidence")
    try:
        validate_safety_critical_artifact(controller_artifact)
    except ArtifactValidationError as exc:
        raise ValueError(f"safety-critical controller artifact is not admissible: {exc}") from exc
    formal = controller_artifact.formal_verification
    if formal is None:
        raise ValueError("safety-critical controller artifact is missing formal verification evidence")
    controller_sha256 = compute_artifact_payload_sha256(controller_artifact)
    _require_same_controller_binding(controller_sha256, transport_evidence, digital_twin_evidence)
    if not transport_evidence.audit_passed:
        raise ValueError("transport evidence must carry a passed gradient audit")
    if transport_evidence.backend != "jax":
        raise ValueError("transport evidence must use the JAX backend")
    if not digital_twin_evidence.improved_over_baseline:
        raise ValueError("digital twin evidence must improve over baseline")
    if tuple(sorted(digital_twin_evidence.simulator_codes)) != ("TRANSP", "TSC"):
        raise ValueError("digital twin evidence must include TRANSP and TSC simulator artifacts")
    return ControllerSafetyCaseEvidence(
        schema_version=1,
        controller_artifact_sha256=controller_sha256,
        formal_report_sha256=formal.report_sha256.lower(),
        formal_backend=formal.backend,
        formal_max_depth=int(formal.max_depth),
        transport_evidence_sha256=_canonical_sha256(transport_evidence),
        digital_twin_evidence_sha256=_canonical_sha256(digital_twin_evidence),
        claim_status=(
            "bounded linked safety-case evidence only; external certification, "
            "facility validation, and hardware-control claims remain blocked"
        ),
    )


def save_controller_safety_case_evidence(evidence: ControllerSafetyCaseEvidence, path: str | Path) -> None:
    """Persist controller safety-case evidence with an integrity digest."""
    if not isinstance(evidence, ControllerSafetyCaseEvidence):
        raise ValueError("evidence must be ControllerSafetyCaseEvidence")
    parsed = _controller_safety_case_evidence_from_mapping(asdict(evidence))
    manifest = {
        "schema_version": _SAFETY_CASE_MANIFEST_SCHEMA_VERSION,
        "evidence": asdict(parsed),
        "integrity_sha256": _canonical_sha256(parsed),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_controller_safety_case_evidence(path: str | Path) -> ControllerSafetyCaseEvidence:
    """Load controller safety-case evidence and verify manifest integrity."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("controller safety-case manifest is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("controller safety-case manifest must be a JSON object")
    if payload.get("schema_version") != _SAFETY_CASE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("controller safety-case manifest schema_version is unsupported")
    evidence_payload = payload.get("evidence")
    if not isinstance(evidence_payload, dict):
        raise ValueError("controller safety-case manifest evidence payload is malformed")
    evidence = _controller_safety_case_evidence_from_mapping(evidence_payload)
    integrity = payload.get("integrity_sha256")
    if not isinstance(integrity, str) or integrity != _canonical_sha256(evidence):
        raise ValueError("controller safety-case manifest integrity digest mismatch")
    return evidence


def assert_controller_safety_case_admissible(
    evidence: ControllerSafetyCaseEvidence,
    controller_artifact: Artifact,
    transport_evidence: TransportDifferentiabilityEvidence,
    digital_twin_evidence: DigitalTwinUpdateEvidence,
) -> ControllerSafetyCaseEvidence:
    """Fail closed unless the safety-case evidence matches all supplied inputs."""
    if not isinstance(evidence, ControllerSafetyCaseEvidence):
        raise ValueError("evidence must be ControllerSafetyCaseEvidence")
    if evidence.schema_version != 1:
        raise ValueError("controller safety-case evidence schema_version is unsupported")
    recomputed = controller_safety_case_evidence(controller_artifact, transport_evidence, digital_twin_evidence)
    if evidence.controller_artifact_sha256 != recomputed.controller_artifact_sha256:
        raise ValueError("controller safety-case evidence controller_artifact_sha256 mismatch")
    if evidence.formal_report_sha256 != recomputed.formal_report_sha256:
        raise ValueError("controller safety-case evidence formal_report_sha256 mismatch")
    if evidence.formal_backend != recomputed.formal_backend:
        raise ValueError("controller safety-case evidence formal_backend mismatch")
    if evidence.formal_max_depth != recomputed.formal_max_depth:
        raise ValueError("controller safety-case evidence formal_max_depth mismatch")
    if evidence.transport_evidence_sha256 != recomputed.transport_evidence_sha256:
        raise ValueError("controller safety-case evidence transport_evidence_sha256 mismatch")
    if evidence.digital_twin_evidence_sha256 != recomputed.digital_twin_evidence_sha256:
        raise ValueError("controller safety-case evidence digital_twin_evidence_sha256 mismatch")
    if evidence.claim_status != recomputed.claim_status:
        raise ValueError("controller safety-case evidence claim_status mismatch")
    return evidence


__all__ = [
    "ControllerSafetyCaseEvidence",
    "ReadinessArtifactEvidence",
    "SafetyCaseReadinessEvidence",
    "assert_controller_safety_case_admissible",
    "assert_controller_safety_case_readiness_admissible",
    "controller_safety_case_evidence",
    "evaluate_controller_safety_case_readiness",
    "evaluate_controller_safety_case_readiness_from_artifacts",
    "load_controller_safety_case_evidence",
    "load_controller_safety_case_readiness",
    "save_controller_safety_case_evidence",
    "save_controller_safety_case_readiness",
]
