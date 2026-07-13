# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal Safety Certificate I/O

"""Bounded formal safety certificate I/O for compiled SCPN control logic.

This module builds, writes, validates, and admits the tamper-evident safety
certificates (and certificate bundles) produced from the bounded verifier in
:mod:`scpn_control.scpn.formal_verification`. A certificate captures a formal
verification report together with its admission policy and an optional artifact
binding; a bundle aggregates independent certificates for a controller release
gate. Every payload is schema-versioned, self-digested, and fail-closed: any
structural or semantic tampering is rejected on load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    CTLFormula,
    EventuallyFires,
    FireLeadsToMarking,
    FormalBackend,
    FormalPetriNetVerifier,
    FormalPropertyReport,
    FormalVerificationReport,
    LTLFormula,
    NeverCoMarked,
)
from scpn_control.scpn.structure import StochasticPetriNet

SAFETY_CERTIFICATE_SCHEMA_VERSION = "scpn-control.safety-certificate.v1"
SAFETY_CERTIFICATE_SCOPE = "bounded formal safety certificate for compiled SCPN Petri-net control logic"
SAFETY_CERTIFICATE_CLAIM_BOUNDARY = "not a facility safety approval, hardware timing certificate, or unbounded proof"
SAFETY_CERTIFICATE_BUNDLE_SCHEMA_VERSION = "scpn-control.safety-certificate-bundle.v1"
SAFETY_CERTIFICATE_BUNDLE_SCOPE = "bounded formal safety certificate bundle for SCPN controller release gates"
SAFETY_CERTIFICATE_BUNDLE_CLAIM_BOUNDARY = "not a facility safety approval or independent regulatory certification"
SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCHEMA_VERSION = "scpn-control.safety-certificate-bundle-artifact.v1"
SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCOPE = "hash-addressed formal safety certificate bundle artifact"
SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_CLAIM_BOUNDARY = (
    "not a facility safety approval or independent regulatory certification"
)
SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_FUTURE_SKEW_SECONDS = 300.0


@dataclass(frozen=True)
class SafetyCertificatePolicy:
    """Admission policy for generated formal safety certificates."""

    name: str
    min_depth: int = 0
    require_artifact_binding: bool = False
    require_ctl: bool = False
    require_ltl: bool = False
    required_checked_specs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("safety certificate policy name must be a non-empty string")
        if isinstance(self.min_depth, bool) or int(self.min_depth) != self.min_depth or self.min_depth < 0:
            raise ValueError("safety certificate policy min_depth must be an integer >= 0")
        if not isinstance(self.require_artifact_binding, bool):
            raise ValueError("safety certificate policy require_artifact_binding must be boolean")
        if not isinstance(self.require_ctl, bool):
            raise ValueError("safety certificate policy require_ctl must be boolean")
        if not isinstance(self.require_ltl, bool):
            raise ValueError("safety certificate policy require_ltl must be boolean")
        if not isinstance(self.required_checked_specs, tuple):
            raise ValueError("safety certificate policy required_checked_specs must be a tuple")
        if any(not isinstance(spec, str) or not spec for spec in self.required_checked_specs):
            raise ValueError("safety certificate policy required_checked_specs must contain non-empty strings")
        if len(self.required_checked_specs) != len(set(self.required_checked_specs)):
            raise ValueError("safety certificate policy required_checked_specs must be unique")

    @classmethod
    def certification_gate(
        cls,
        *,
        name: str = "bounded-ctl-ltl-artifact-gate",
        min_depth: int = 1,
        required_checked_specs: tuple[str, ...] = (),
    ) -> SafetyCertificatePolicy:
        """Create a fail-closed policy requiring artifact binding plus CTL and LTL evidence."""

        return cls(
            name=name,
            min_depth=min_depth,
            require_artifact_binding=True,
            require_ctl=True,
            require_ltl=True,
            required_checked_specs=required_checked_specs,
        )


@dataclass(frozen=True)
class SafetyCertificateBundlePolicy:
    """Admission policy for a bundle of independent safety certificates."""

    name: str
    min_certificates: int = 1
    required_policy_name: str | None = None
    require_same_artifact: bool = False
    require_same_backend: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("safety certificate bundle policy name must be a non-empty string")
        if (
            isinstance(self.min_certificates, bool)
            or int(self.min_certificates) != self.min_certificates
            or self.min_certificates < 1
        ):
            raise ValueError("safety certificate bundle policy min_certificates must be an integer >= 1")
        if self.required_policy_name is not None and (
            not isinstance(self.required_policy_name, str) or not self.required_policy_name
        ):
            raise ValueError("safety certificate bundle policy required_policy_name must be non-empty or None")
        if not isinstance(self.require_same_artifact, bool):
            raise ValueError("safety certificate bundle policy require_same_artifact must be boolean")
        if not isinstance(self.require_same_backend, bool):
            raise ValueError("safety certificate bundle policy require_same_backend must be boolean")


def build_safety_certificate_payload(
    report: FormalVerificationReport,
    *,
    ctl_report: FormalPropertyReport | None = None,
    ltl_report: FormalPropertyReport | None = None,
    artifact_sha256: str | None = None,
    issuer: str = "scpn-control",
    policy: SafetyCertificatePolicy | None = None,
) -> dict[str, Any]:
    """Build a schema-versioned tamper-evident formal safety certificate."""

    if artifact_sha256 is not None and not _is_sha256(artifact_sha256):
        raise ValueError("artifact_sha256 must be a SHA-256 hex digest")
    if not isinstance(issuer, str) or not issuer:
        raise ValueError("safety certificate issuer must be a non-empty string")
    sections = {
        "reachability": _jsonable(asdict(report.reachability)),
        "safety": _jsonable(asdict(report.safety)),
        "liveness": _jsonable(asdict(report.liveness)),
        "temporal": _jsonable(asdict(report.temporal)),
        "ctl": _jsonable(asdict(ctl_report)) if ctl_report is not None else None,
        "ltl": _jsonable(asdict(ltl_report)) if ltl_report is not None else None,
    }
    holds = report.holds
    if ctl_report is not None:
        holds = holds and ctl_report.holds
    if ltl_report is not None:
        holds = holds and ltl_report.holds
    payload: dict[str, Any] = {
        "schema_version": SAFETY_CERTIFICATE_SCHEMA_VERSION,
        "status": "pass" if holds else "fail",
        "holds": holds,
        "backend": report.reachability.backend,
        "max_depth": report.reachability.max_depth,
        "issuer": issuer,
        "artifact_sha256": artifact_sha256,
        "checked_specs": _certificate_checked_specs(report, ctl_report, ltl_report),
        "scope": SAFETY_CERTIFICATE_SCOPE,
        "claim_boundary": SAFETY_CERTIFICATE_CLAIM_BOUNDARY,
        "policy": _policy_payload(policy) if policy is not None else None,
        "sections": sections,
    }
    if policy is not None:
        _enforce_safety_certificate_policy(payload, policy)
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_safety_certificate_payload(payload)


def write_safety_certificate(
    report: FormalVerificationReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    ctl_report: FormalPropertyReport | None = None,
    ltl_report: FormalPropertyReport | None = None,
    artifact_sha256: str | None = None,
    issuer: str = "scpn-control",
    policy: SafetyCertificatePolicy | None = None,
) -> dict[str, Any]:
    """Persist a formal safety certificate as JSON and Markdown artifacts."""

    payload = build_safety_certificate_payload(
        report,
        ctl_report=ctl_report,
        ltl_report=ltl_report,
        artifact_sha256=artifact_sha256,
        issuer=issuer,
        policy=policy,
    )
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(_render_safety_certificate_markdown(payload), encoding="utf-8")
    return payload


def generate_safety_certificate(
    net: StochasticPetriNet,
    *,
    max_depth: int,
    marking_bounds: dict[str, tuple[float, float]],
    json_path: str | Path,
    markdown_path: str | Path,
    temporal_specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking]
    | None = None,
    ctl_specs: list[CTLFormula] | None = None,
    ltl_specs: list[LTLFormula] | None = None,
    artifact_path: str | Path | None = None,
    artifact_sha256: str | None = None,
    issuer: str = "scpn-control",
    backend: FormalBackend = "explicit-state",
    policy: SafetyCertificatePolicy | None = None,
) -> dict[str, Any]:
    """Run proof obligations and persist a bound formal safety certificate.

    The workflow resolves one verifier backend, runs base Petri-net safety,
    liveness, optional CTL, and optional LTL obligations through that same
    transition relation, binds the certificate to optional controller artifact
    bytes, and writes JSON/Markdown evidence in one call.
    """

    bound_artifact_sha256 = _resolve_artifact_sha256(artifact_path, artifact_sha256)
    verifier = FormalPetriNetVerifier(net, backend=backend)
    reachability = verifier.analyze_reachability(max_depth=max_depth)
    safety = verifier.prove_marking_bounds(marking_bounds, max_depth=max_depth)
    liveness = verifier.prove_transition_liveness(max_depth=max_depth)
    temporal = verifier.verify_temporal_specs(temporal_specs or [], max_depth=max_depth)
    report = FormalVerificationReport(
        holds=safety.holds and liveness.holds and temporal.holds,
        reachability=reachability,
        safety=safety,
        liveness=liveness,
        temporal=temporal,
    )
    ctl_report = verifier.verify_ctl_specs(ctl_specs, max_depth=max_depth) if ctl_specs is not None else None
    ltl_report = verifier.verify_ltl_specs(ltl_specs, max_depth=max_depth) if ltl_specs is not None else None
    return write_safety_certificate(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        ctl_report=ctl_report,
        ltl_report=ltl_report,
        artifact_sha256=bound_artifact_sha256,
        issuer=issuer,
        policy=policy,
    )


def validate_safety_certificate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a schema-versioned formal safety certificate payload."""

    if payload.get("schema_version") != SAFETY_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("safety certificate schema_version is unsupported")
    if payload.get("status") not in {"pass", "fail"}:
        raise ValueError("safety certificate status must be pass or fail")
    if not isinstance(payload.get("holds"), bool):
        raise ValueError("safety certificate holds must be a boolean")
    if payload["status"] == "pass" and not payload["holds"]:
        raise ValueError("passing safety certificate must hold")
    if payload["status"] == "fail" and payload["holds"]:
        raise ValueError("failed safety certificate must not hold")
    if payload.get("backend") not in {"explicit-state", "z3"}:
        raise ValueError("safety certificate backend is unsupported")
    if isinstance(payload.get("max_depth"), bool) or not isinstance(payload.get("max_depth"), int):
        raise ValueError("safety certificate max_depth must be an integer")
    if payload["max_depth"] < 0:
        raise ValueError("safety certificate max_depth must be non-negative")
    if not isinstance(payload.get("issuer"), str) or not payload["issuer"]:
        raise ValueError("safety certificate issuer must be a non-empty string")
    artifact_sha256 = payload.get("artifact_sha256")
    if artifact_sha256 is not None and (not isinstance(artifact_sha256, str) or not _is_sha256(artifact_sha256)):
        raise ValueError("safety certificate artifact_sha256 must be a SHA-256 hex digest or null")
    if not isinstance(payload.get("checked_specs"), list) or not payload["checked_specs"]:
        raise ValueError("safety certificate checked_specs must be a non-empty list")
    if any(not isinstance(spec, str) or not spec for spec in payload["checked_specs"]):
        raise ValueError("safety certificate checked_specs must contain non-empty strings")
    if len(payload["checked_specs"]) != len(set(payload["checked_specs"])):
        raise ValueError("safety certificate checked_specs must be unique")
    if payload.get("scope") != SAFETY_CERTIFICATE_SCOPE:
        raise ValueError("safety certificate scope is unsupported")
    if payload.get("claim_boundary") != SAFETY_CERTIFICATE_CLAIM_BOUNDARY:
        raise ValueError("safety certificate claim_boundary is unsupported")
    policy = _policy_from_payload(payload.get("policy"))
    if not isinstance(payload.get("sections"), dict):
        raise ValueError("safety certificate sections must be an object")
    sections = payload["sections"]
    required_sections = ("reachability", "safety", "liveness", "temporal")
    section_holds = True
    for section_name in (*required_sections, "ctl", "ltl"):
        section = sections.get(section_name)
        if section is None:
            if section_name in required_sections:
                raise ValueError(f"safety certificate {section_name} section must be an object")
            continue
        if not isinstance(section, dict):
            raise ValueError(f"safety certificate {section_name} section must be an object")
        if not isinstance(section.get("holds"), bool):
            raise ValueError(f"safety certificate {section_name} section holds must be a boolean")
        if section.get("backend") != payload["backend"]:
            raise ValueError(f"safety certificate {section_name} section backend must match certificate backend")
        if section.get("max_depth") != payload["max_depth"]:
            raise ValueError(f"safety certificate {section_name} section depth must match certificate depth")
        checked = section.get("checked_specs", [])
        if not isinstance(checked, list):
            raise ValueError(f"safety certificate {section_name} section checked_specs must be a list")
        if any(not isinstance(spec, str) or not spec for spec in checked):
            raise ValueError(f"safety certificate {section_name} section checked_specs must contain non-empty strings")
        section_holds = section_holds and section["holds"]
    if payload["holds"] != section_holds:
        raise ValueError("safety certificate section holds must match certificate holds")
    if payload["checked_specs"] != _certificate_checked_specs_from_sections(sections):
        raise ValueError("safety certificate checked_specs must match certificate sections")
    if policy is not None:
        _enforce_safety_certificate_policy(payload, policy)
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("safety certificate payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("safety certificate payload_sha256 does not match payload")
    return payload


def build_safety_certificate_bundle_payload(
    certificates: list[dict[str, Any]],
    *,
    policy: SafetyCertificateBundlePolicy | None = None,
) -> dict[str, Any]:
    """Build a schema-versioned tamper-evident safety certificate bundle."""

    validated = [validate_safety_certificate_payload(dict(certificate)) for certificate in certificates]
    _enforce_unique_certificate_digests(validated)
    holds = all(certificate["holds"] for certificate in validated)
    payload: dict[str, Any] = {
        "schema_version": SAFETY_CERTIFICATE_BUNDLE_SCHEMA_VERSION,
        "status": "pass" if holds else "fail",
        "holds": holds,
        "certificate_count": len(validated),
        "artifact_sha256": _common_certificate_value(validated, "artifact_sha256"),
        "backend": _common_certificate_value(validated, "backend") or "mixed",
        "scope": SAFETY_CERTIFICATE_BUNDLE_SCOPE,
        "claim_boundary": SAFETY_CERTIFICATE_BUNDLE_CLAIM_BOUNDARY,
        "policy": _bundle_policy_payload(policy) if policy is not None else None,
        "certificates": validated,
    }
    if policy is not None:
        _enforce_safety_certificate_bundle_policy(payload, policy)
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_safety_certificate_bundle_payload(payload)


def write_safety_certificate_bundle(
    certificates: list[dict[str, Any]],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    policy: SafetyCertificateBundlePolicy | None = None,
) -> dict[str, Any]:
    """Persist a formal safety certificate bundle as JSON and Markdown artifacts."""

    payload = build_safety_certificate_bundle_payload(certificates, policy=policy)
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(_render_safety_certificate_bundle_markdown(payload), encoding="utf-8")
    return payload


def validate_safety_certificate_bundle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a schema-versioned formal safety certificate bundle."""

    if payload.get("schema_version") != SAFETY_CERTIFICATE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("safety certificate bundle schema_version is unsupported")
    if payload.get("status") not in {"pass", "fail"}:
        raise ValueError("safety certificate bundle status must be pass or fail")
    if not isinstance(payload.get("holds"), bool):
        raise ValueError("safety certificate bundle holds must be a boolean")
    if isinstance(payload.get("certificate_count"), bool) or not isinstance(payload.get("certificate_count"), int):
        raise ValueError("safety certificate bundle certificate_count must be an integer")
    if payload["certificate_count"] < 1:
        raise ValueError("safety certificate bundle certificate_count must be positive")
    if payload.get("scope") != SAFETY_CERTIFICATE_BUNDLE_SCOPE:
        raise ValueError("safety certificate bundle scope is unsupported")
    if payload.get("claim_boundary") != SAFETY_CERTIFICATE_BUNDLE_CLAIM_BOUNDARY:
        raise ValueError("safety certificate bundle claim_boundary is unsupported")
    if payload.get("backend") not in {"explicit-state", "z3", "mixed"}:
        raise ValueError("safety certificate bundle backend is unsupported")
    artifact_sha256 = payload.get("artifact_sha256")
    if artifact_sha256 is not None and (not isinstance(artifact_sha256, str) or not _is_sha256(artifact_sha256)):
        raise ValueError("safety certificate bundle artifact_sha256 must be a SHA-256 hex digest or null")
    certificates = payload.get("certificates")
    if not isinstance(certificates, list) or len(certificates) != payload["certificate_count"]:
        raise ValueError("safety certificate bundle certificates must match certificate_count")
    validated = [validate_safety_certificate_payload(dict(certificate)) for certificate in certificates]
    _enforce_unique_certificate_digests(validated)
    if payload["holds"] != all(certificate["holds"] for certificate in validated):
        raise ValueError("safety certificate bundle holds must match certificate holds")
    if payload["status"] == "pass" and not payload["holds"]:
        raise ValueError("passing safety certificate bundle must hold")
    if payload["status"] == "fail" and payload["holds"]:
        raise ValueError("failed safety certificate bundle must not hold")
    if payload["artifact_sha256"] != _common_certificate_value(validated, "artifact_sha256"):
        raise ValueError("safety certificate bundle artifact_sha256 must match certificate artifact bindings")
    expected_backend = _common_certificate_value(validated, "backend") or "mixed"
    if payload["backend"] != expected_backend:
        raise ValueError("safety certificate bundle backend must match certificate backends")
    policy = _bundle_policy_from_payload(payload.get("policy"))
    if policy is not None:
        _enforce_safety_certificate_bundle_policy(payload, policy)
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("safety certificate bundle payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("safety certificate bundle payload_sha256 does not match payload")
    return payload


def build_safety_certificate_bundle_artifact(
    *,
    bundle_uri: str,
    bundle_sha256: str,
    producer: str,
    created_at: str,
) -> dict[str, Any]:
    """Build a schema-versioned reference to a persisted certificate bundle."""

    uri = _safe_relative_uri("bundle_uri", bundle_uri)
    if not isinstance(bundle_sha256, str) or not _is_sha256(bundle_sha256):
        raise ValueError("safety certificate bundle artifact bundle_sha256 must be a SHA-256 hex digest")
    if not isinstance(producer, str) or not producer:
        raise ValueError("safety certificate bundle artifact producer must be a non-empty string")
    created_at = _require_utc_timestamp("created_at", created_at)
    artifact = {
        "schema_version": SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCHEMA_VERSION,
        "kind": "safety_certificate_bundle",
        "bundle_uri": uri,
        "bundle_sha256": bundle_sha256.lower(),
        "producer": producer,
        "created_at": created_at,
        "scope": SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCOPE,
        "claim_boundary": SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_CLAIM_BOUNDARY,
    }
    artifact["artifact_sha256"] = _bundle_artifact_digest(artifact)
    return validate_safety_certificate_bundle_artifact(artifact)


def validate_safety_certificate_bundle_artifact(
    artifact: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a persisted certificate-bundle artifact reference."""

    if not isinstance(artifact, dict):
        raise ValueError("safety certificate bundle artifact must be an object")
    if artifact.get("schema_version") != SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("safety certificate bundle artifact schema_version is unsupported")
    if artifact.get("kind") != "safety_certificate_bundle":
        raise ValueError("safety certificate bundle artifact kind is unsupported")
    uri = _safe_relative_uri("bundle_uri", artifact.get("bundle_uri"))
    bundle_sha256 = artifact.get("bundle_sha256")
    if not isinstance(bundle_sha256, str) or not _is_sha256(bundle_sha256):
        raise ValueError("safety certificate bundle artifact bundle_sha256 must be a SHA-256 hex digest")
    if not isinstance(artifact.get("producer"), str) or not artifact["producer"]:
        raise ValueError("safety certificate bundle artifact producer must be a non-empty string")
    created_at = _require_utc_timestamp("created_at", artifact.get("created_at"))
    if artifact.get("scope") != SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCOPE:
        raise ValueError("safety certificate bundle artifact scope is unsupported")
    if artifact.get("claim_boundary") != SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_CLAIM_BOUNDARY:
        raise ValueError("safety certificate bundle artifact claim_boundary is unsupported")
    expected_digest_payload = {
        "schema_version": SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCHEMA_VERSION,
        "kind": "safety_certificate_bundle",
        "bundle_uri": uri,
        "bundle_sha256": bundle_sha256.lower(),
        "producer": artifact["producer"],
        "created_at": created_at,
        "scope": SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_SCOPE,
        "claim_boundary": SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_CLAIM_BOUNDARY,
    }
    artifact_sha256 = artifact.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or not _is_sha256(artifact_sha256):
        raise ValueError("safety certificate bundle artifact artifact_sha256 must be a SHA-256 hex digest")
    expected_digest = _bundle_artifact_digest(expected_digest_payload)
    if artifact_sha256.lower() != expected_digest:
        raise ValueError("safety certificate bundle artifact artifact_sha256 does not match artifact metadata")
    validated = {**expected_digest_payload, "artifact_sha256": expected_digest}
    if artifact_root is not None:
        target = _resolve_under_root(artifact_root, uri)
        if not target.is_file():
            raise ValueError("safety certificate bundle artifact target must be a file")
        digest = _file_sha256(target)
        if digest != validated["bundle_sha256"]:
            raise ValueError("safety certificate bundle artifact bundle_sha256 does not match bundle bytes")
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("safety certificate bundle artifact target must contain a bundle object")
        validate_safety_certificate_bundle_payload(raw)
    return validated


def admit_safety_certificate_bundle_artifact(artifact: dict[str, Any], *, artifact_root: str | Path) -> dict[str, Any]:
    """Load and validate the bundle referenced by a certificate-bundle artifact."""

    validated = validate_safety_certificate_bundle_artifact(artifact, artifact_root=artifact_root)
    target = _resolve_under_root(artifact_root, validated["bundle_uri"])
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("safety certificate bundle artifact target must contain a bundle object")
    return validate_safety_certificate_bundle_payload(raw)


def _certificate_checked_specs(
    report: FormalVerificationReport,
    ctl_report: FormalPropertyReport | None,
    ltl_report: FormalPropertyReport | None,
) -> list[str]:
    specs = ["marking_bounds", "transition_liveness"]
    for section in (report.temporal, ctl_report, ltl_report):
        if section is None:
            continue
        for spec in section.checked_specs:
            if spec not in specs:
                specs.append(spec)
    return specs


def _certificate_checked_specs_from_sections(sections: dict[str, Any]) -> list[str]:
    specs = ["marking_bounds", "transition_liveness"]
    for section_name in ("temporal", "ctl", "ltl"):
        section = sections.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            continue
        for spec in section.get("checked_specs", []):
            if spec not in specs:
                specs.append(spec)
    return specs


def _policy_payload(policy: SafetyCertificatePolicy) -> dict[str, Any]:
    return {
        "name": policy.name,
        "min_depth": policy.min_depth,
        "require_artifact_binding": policy.require_artifact_binding,
        "require_ctl": policy.require_ctl,
        "require_ltl": policy.require_ltl,
        "required_checked_specs": list(policy.required_checked_specs),
    }


def _bundle_policy_payload(policy: SafetyCertificateBundlePolicy) -> dict[str, Any]:
    return {
        "name": policy.name,
        "min_certificates": policy.min_certificates,
        "required_policy_name": policy.required_policy_name,
        "require_same_artifact": policy.require_same_artifact,
        "require_same_backend": policy.require_same_backend,
    }


def _policy_from_payload(value: Any) -> SafetyCertificatePolicy | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("safety certificate policy must be an object or null")
    required_specs = value.get("required_checked_specs", [])
    if not isinstance(required_specs, list):
        raise ValueError("safety certificate policy required_checked_specs must be a list")
    return SafetyCertificatePolicy(
        name=value.get("name", ""),
        min_depth=value.get("min_depth", 0),
        require_artifact_binding=value.get("require_artifact_binding", False),
        require_ctl=value.get("require_ctl", False),
        require_ltl=value.get("require_ltl", False),
        required_checked_specs=tuple(required_specs),
    )


def _bundle_policy_from_payload(value: Any) -> SafetyCertificateBundlePolicy | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("safety certificate bundle policy must be an object or null")
    return SafetyCertificateBundlePolicy(
        name=value.get("name", ""),
        min_certificates=value.get("min_certificates", 1),
        required_policy_name=value.get("required_policy_name"),
        require_same_artifact=value.get("require_same_artifact", False),
        require_same_backend=value.get("require_same_backend", False),
    )


def _enforce_safety_certificate_policy(payload: dict[str, Any], policy: SafetyCertificatePolicy) -> None:
    if payload["max_depth"] < policy.min_depth:
        raise ValueError("safety certificate policy requires max_depth to meet min_depth")
    if policy.require_artifact_binding and payload.get("artifact_sha256") is None:
        raise ValueError("safety certificate policy requires artifact binding")
    sections = payload.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("safety certificate policy requires certificate sections")
    if policy.require_ctl and sections.get("ctl") is None:
        raise ValueError("safety certificate policy requires CTL evidence")
    if policy.require_ltl and sections.get("ltl") is None:
        raise ValueError("safety certificate policy requires LTL evidence")
    checked_specs = payload.get("checked_specs")
    if not isinstance(checked_specs, list):
        raise ValueError("safety certificate policy requires checked specs")
    missing = [spec for spec in policy.required_checked_specs if spec not in checked_specs]
    if missing:
        raise ValueError(f"safety certificate policy missing required checked spec: {missing[0]}")


def _enforce_safety_certificate_bundle_policy(payload: dict[str, Any], policy: SafetyCertificateBundlePolicy) -> None:
    certificates = payload.get("certificates")
    if not isinstance(certificates, list):
        raise ValueError("safety certificate bundle policy requires certificates")
    if len(certificates) < policy.min_certificates:
        raise ValueError("safety certificate bundle policy requires more certificates")
    if policy.require_same_artifact:
        artifact = _common_certificate_value(certificates, "artifact_sha256")
        if artifact is None:
            raise ValueError("safety certificate bundle policy requires a shared artifact binding")
    if policy.require_same_backend and _common_certificate_value(certificates, "backend") is None:
        raise ValueError("safety certificate bundle policy requires a shared backend")
    if policy.required_policy_name is not None:
        for certificate in certificates:
            cert_policy = certificate.get("policy")
            if not isinstance(cert_policy, dict) or cert_policy.get("name") != policy.required_policy_name:
                raise ValueError("safety certificate bundle policy requires matching certificate policy")


def _enforce_unique_certificate_digests(certificates: list[dict[str, Any]]) -> None:
    digests = [certificate.get("payload_sha256") for certificate in certificates]
    if not digests:
        raise ValueError("safety certificate bundle must include at least one certificate")
    if any(not isinstance(digest, str) or not _is_sha256(digest) for digest in digests):
        raise ValueError("safety certificate bundle certificates must carry payload digests")
    if len(digests) != len(set(digests)):
        raise ValueError("safety certificate bundle certificates must be unique")


def _common_certificate_value(certificates: list[dict[str, Any]], key: str) -> Any:
    values = [certificate.get(key) for certificate in certificates]
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return None


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _resolve_artifact_sha256(artifact_path: str | Path | None, artifact_sha256: str | None) -> str | None:
    if artifact_sha256 is not None and not _is_sha256(artifact_sha256):
        raise ValueError("artifact_sha256 must be a SHA-256 hex digest")
    if artifact_path is None:
        return artifact_sha256
    digest = _file_sha256(artifact_path)
    if artifact_sha256 is not None and digest != artifact_sha256.lower():
        raise ValueError("artifact_sha256 does not match artifact_path bytes")
    return digest


def _file_sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        raise ValueError("artifact_path must be an existing file")
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_uri(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    if "\x00" in value or "\\" in value or "://" in value or value.startswith(("/", "~")):
        raise ValueError(f"{name} must be a safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe relative path")
    return value


def _resolve_under_root(root: str | Path, relative_uri: str) -> Path:
    relative_uri = _safe_relative_uri("bundle_uri", relative_uri)
    base = Path(root).resolve()
    target = (base / relative_uri).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("safety certificate bundle artifact path escapes artifact_root") from exc
    return target


def _require_utc_timestamp(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"safety certificate bundle artifact {name} must be a non-empty UTC timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"safety certificate bundle artifact {name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"safety certificate bundle artifact {name} must include a UTC offset")
    parsed = parsed.astimezone(timezone.utc)
    max_created_at = datetime.now(timezone.utc) + timedelta(
        seconds=SAFETY_CERTIFICATE_BUNDLE_ARTIFACT_FUTURE_SKEW_SECONDS
    )
    if parsed > max_created_at:
        raise ValueError(f"safety certificate bundle artifact {name} must not be future-dated")
    return value


def _bundle_artifact_digest(artifact: dict[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _render_safety_certificate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SCPN Formal Safety Certificate",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Status: `{payload['status']}`",
        f"- Backend: `{payload['backend']}`",
        f"- Max depth: `{payload['max_depth']}`",
        f"- Issuer: `{payload['issuer']}`",
        f"- Payload SHA-256: `{payload['payload_sha256']}`",
        f"- Scope: {payload['scope']}.",
        f"- Claim boundary: {payload['claim_boundary']}.",
        "",
        "## Checked specifications",
        "",
    ]
    if payload.get("policy") is not None:
        lines.extend(
            [
                "## Policy",
                "",
                f"- Name: `{payload['policy']['name']}`",
                f"- Min depth: `{payload['policy']['min_depth']}`",
                f"- Requires artifact binding: `{payload['policy']['require_artifact_binding']}`",
                f"- Requires CTL evidence: `{payload['policy']['require_ctl']}`",
                f"- Requires LTL evidence: `{payload['policy']['require_ltl']}`",
                "",
            ]
        )
    lines.extend(f"- `{spec}`" for spec in payload["checked_specs"])
    lines.extend(["", "## Section status", ""])
    for section_name, section in payload["sections"].items():
        if section is None:
            lines.append(f"- `{section_name}`: not supplied")
        else:
            lines.append(f"- `{section_name}`: holds=`{section['holds']}`")
    violations: list[dict[str, Any]] = []
    for section in payload["sections"].values():
        if isinstance(section, dict):
            violations.extend(section.get("violations", []))
    if violations:
        lines.extend(["", "## Counterexamples", ""])
        for violation in violations:
            lines.append(f"- `{violation['property_name']}` path={violation['path']} message={violation['message']}")
    lines.append("")
    return "\n".join(lines)


def _render_safety_certificate_bundle_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SCPN Formal Safety Certificate Bundle",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Status: `{payload['status']}`",
        f"- Certificate count: `{payload['certificate_count']}`",
        f"- Backend: `{payload['backend']}`",
        f"- Artifact SHA-256: `{payload['artifact_sha256']}`",
        f"- Payload SHA-256: `{payload['payload_sha256']}`",
        f"- Scope: {payload['scope']}.",
        f"- Claim boundary: {payload['claim_boundary']}.",
        "",
    ]
    if payload.get("policy") is not None:
        lines.extend(
            [
                "## Bundle policy",
                "",
                f"- Name: `{payload['policy']['name']}`",
                f"- Min certificates: `{payload['policy']['min_certificates']}`",
                f"- Required certificate policy: `{payload['policy']['required_policy_name']}`",
                f"- Requires shared artifact: `{payload['policy']['require_same_artifact']}`",
                f"- Requires shared backend: `{payload['policy']['require_same_backend']}`",
                "",
            ]
        )
    lines.extend(["## Certificates", ""])
    for certificate in payload["certificates"]:
        lines.append(
            f"- `{certificate['payload_sha256']}` status=`{certificate['status']}` "
            f"issuer=`{certificate['issuer']}` depth=`{certificate['max_depth']}`"
        )
    lines.append("")
    return "\n".join(lines)
