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
from typing import Any

from scpn_control.control.digital_twin_online_update import DigitalTwinUpdateEvidence
from scpn_control.core.differentiable_transport import TransportDifferentiabilityEvidence
from scpn_control.scpn.artifact import (
    Artifact,
    ArtifactValidationError,
    compute_artifact_payload_sha256,
    validate_safety_critical_artifact,
)


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


def _canonical_sha256(value: Any) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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
    "assert_controller_safety_case_admissible",
    "controller_safety_case_evidence",
]
