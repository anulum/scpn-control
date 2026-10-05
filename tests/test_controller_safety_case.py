# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Safety Case Tests
"""Workflow contract tests for controller safety-case evidence chaining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from safety_case_support import (
    _controller_artifact,
    _digital_twin_evidence,
    _transport_evidence,
)

from scpn_control.control.digital_twin_online_update import (
    DigitalTwinUpdateEvidence,
)
from scpn_control.control.safety_case import (
    ControllerSafetyCaseEvidence,
    SafetyCaseReadinessEvidence,
    assert_controller_safety_case_admissible,
    assert_controller_safety_case_readiness_admissible,
    controller_safety_case_evidence,
    evaluate_controller_safety_case_readiness,
    load_controller_safety_case_evidence,
    save_controller_safety_case_evidence,
    save_controller_safety_case_readiness,
)
from scpn_control.core.differentiable_transport import (
    TransportDifferentiabilityEvidence,
)
from scpn_control.scpn.artifact import (
    Artifact,
    compute_artifact_payload_sha256,
)


def test_controller_safety_case_binds_formal_transport_and_twin_evidence() -> None:
    """Check that controller safety case binds formal transport and twin evidence."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)

    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)

    assert evidence.controller_artifact_sha256 == controller_sha256
    assert evidence.formal_backend == "z3"
    assert evidence.transport_evidence_sha256
    assert evidence.digital_twin_evidence_sha256
    assert_controller_safety_case_admissible(evidence, artifact, transport, digital_twin)


def test_controller_safety_case_manifest_round_trips_with_integrity_digest(tmp_path: Path) -> None:
    """Round-trip the safety-case manifest with its integrity digest intact."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)
    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)
    path = tmp_path / "controller_safety_case.json"

    save_controller_safety_case_evidence(evidence, path)
    loaded = load_controller_safety_case_evidence(path)

    assert loaded == evidence
    assert_controller_safety_case_admissible(loaded, artifact, transport, digital_twin)


def test_controller_safety_case_manifest_rejects_tampering(tmp_path: Path) -> None:
    """Exercise controller safety case manifest validation against tampering."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    path = tmp_path / "controller_safety_case.json"
    save_controller_safety_case_evidence(evidence, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["formal_max_depth"] = 99
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        load_controller_safety_case_evidence(path)


def test_controller_safety_case_manifest_rejects_malformed_schema(tmp_path: Path) -> None:
    """Exercise controller safety case manifest validation against malformed schema."""
    path = tmp_path / "bad_controller_safety_case.json"
    path.write_text(json.dumps({"schema_version": 99, "evidence": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_controller_safety_case_evidence(path)


def test_controller_safety_case_manifest_rejects_unreadable_and_malformed_payloads(tmp_path: Path) -> None:
    """Exercise controller safety case manifest validation against unreadable and malformed payloads."""
    unreadable_json = tmp_path / "not_json.json"
    unreadable_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="readable JSON"):
        load_controller_safety_case_evidence(unreadable_json)

    non_object = tmp_path / "non_object.json"
    non_object.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_controller_safety_case_evidence(non_object)

    missing_payload = tmp_path / "missing_payload.json"
    missing_payload.write_text(json.dumps({"schema_version": 1, "evidence": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence payload"):
        load_controller_safety_case_evidence(missing_payload)


def test_controller_safety_case_manifest_rejects_invalid_evidence_fields(tmp_path: Path) -> None:
    """Exercise controller safety case manifest validation against invalid evidence fields."""
    evidence = ControllerSafetyCaseEvidence(
        schema_version=1,
        controller_artifact_sha256="1" * 64,
        formal_report_sha256="2" * 64,
        formal_backend="z3",
        formal_max_depth=4,
        transport_evidence_sha256="3" * 64,
        digital_twin_evidence_sha256="4" * 64,
        claim_status="bounded safety-case evidence only",
    )
    path = tmp_path / "controller_safety_case.json"
    save_controller_safety_case_evidence(evidence, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["formal_backend"] = "unchecked"
    payload["integrity_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="formal_backend|integrity"):
        load_controller_safety_case_evidence(path)


def test_controller_safety_case_readiness_blocks_without_external_evidence() -> None:
    """Check that controller safety case readiness remains blocked without external evidence."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )

    readiness = evaluate_controller_safety_case_readiness(evidence)

    assert isinstance(readiness, SafetyCaseReadinessEvidence)
    assert readiness.status == "blocked"
    assert readiness.safety_case_sha256
    assert "external_physics_validation_sha256" in readiness.blocking_reasons
    assert "target_hardware_timing_sha256" in readiness.blocking_reasons
    assert "hil_replay_evidence_sha256" in readiness.blocking_reasons
    assert "hdl_export_evidence_sha256" in readiness.blocking_reasons
    assert "codac_runtime_evidence_sha256" in readiness.blocking_reasons
    assert "websocket_runtime_evidence_sha256" in readiness.blocking_reasons
    assert "independent_safety_review_sha256" in readiness.blocking_reasons
    with pytest.raises(ValueError, match="blocked"):
        assert_controller_safety_case_readiness_admissible(readiness, evidence)


def test_controller_safety_case_readiness_digest_only_is_ready_but_not_admissible() -> None:
    """Distinguish digest completeness from qualified promotion evidence."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )

    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )

    assert readiness.status == "promotion_ready"
    assert readiness.blocking_reasons == ()
    assert readiness.external_physics_validation_sha256 == "1" * 64
    assert readiness.promotion_admissible is False
    with pytest.raises(ValueError, match="not admissible"):
        assert_controller_safety_case_readiness_admissible(readiness, evidence)


def test_controller_safety_case_readiness_rejects_fabricated_zero_digests() -> None:
    # Seven fabricated "0"*64 digests reach promotion_ready
    # (valid hex) but are refused at the promotion gate because they are unverified.
    """Exercise controller safety case readiness validation against fabricated zero digests."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )

    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="0" * 64,
        target_hardware_timing_sha256="0" * 64,
        hil_replay_evidence_sha256="0" * 64,
        hdl_export_evidence_sha256="0" * 64,
        codac_runtime_evidence_sha256="0" * 64,
        websocket_runtime_evidence_sha256="0" * 64,
        independent_safety_review_sha256="0" * 64,
    )

    assert readiness.status == "promotion_ready"
    assert readiness.promotion_admissible is False
    with pytest.raises(ValueError, match="not admissible"):
        assert_controller_safety_case_readiness_admissible(readiness, evidence)


def test_controller_safety_case_readiness_rejects_drift_and_bad_digest() -> None:
    """Exercise controller safety case readiness validation against drift and bad digest."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )
    drifted = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    object.__setattr__(drifted, "formal_max_depth", drifted.formal_max_depth + 1)

    with pytest.raises(ValueError, match="safety_case_sha256"):
        assert_controller_safety_case_readiness_admissible(readiness, drifted)
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_controller_safety_case_readiness(
            evidence,
            external_physics_validation_sha256="not-a-digest",
            target_hardware_timing_sha256="2" * 64,
            hil_replay_evidence_sha256="4" * 64,
            hdl_export_evidence_sha256="6" * 64,
            codac_runtime_evidence_sha256="5" * 64,
            websocket_runtime_evidence_sha256="7" * 64,
            independent_safety_review_sha256="3" * 64,
        )


def test_controller_safety_case_readiness_admission_rejects_type_and_state_drift() -> None:
    """Exercise controller safety case readiness admission validation against type and state drift."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )

    with pytest.raises(ValueError, match="readiness must"):
        assert_controller_safety_case_readiness_admissible(cast(SafetyCaseReadinessEvidence, object()), evidence)
    with pytest.raises(ValueError, match="safety_case must"):
        assert_controller_safety_case_readiness_admissible(readiness, cast(ControllerSafetyCaseEvidence, object()))

    stale_schema = SafetyCaseReadinessEvidence(**{**readiness.__dict__, "schema_version": 1})
    with pytest.raises(ValueError, match="schema_version"):
        assert_controller_safety_case_readiness_admissible(stale_schema, evidence)

    bad_status = SafetyCaseReadinessEvidence(**{**readiness.__dict__, "status": "unchecked"})
    with pytest.raises(ValueError, match="status"):
        assert_controller_safety_case_readiness_admissible(bad_status, evidence)

    # promotion_admissible=True clears the digest-only gate so the recompute-mismatch check
    # (the tamper detector) is reached; the drifted claim_status must fail it.
    drifted = SafetyCaseReadinessEvidence(
        **{**readiness.__dict__, "claim_status": "bounded stale readiness", "promotion_admissible": True}
    )
    with pytest.raises(ValueError, match="evidence mismatch"):
        assert_controller_safety_case_readiness_admissible(drifted, evidence)

    # Control for the tamper detector: the same record without the drift passes
    # the recompute check, so the refusal above is caused by the drift. No
    # producer sets this flag; evaluation, artifacts and loading all return False.
    consistent = SafetyCaseReadinessEvidence(**{**readiness.__dict__, "promotion_admissible": True})
    assert assert_controller_safety_case_readiness_admissible(consistent, evidence) is consistent

    digest_only = SafetyCaseReadinessEvidence(**{**readiness.__dict__, "promotion_admissible": False})
    with pytest.raises(ValueError, match="not admissible"):
        assert_controller_safety_case_readiness_admissible(digest_only, evidence)


def test_controller_safety_case_rejects_mismatched_evidence_chain() -> None:
    """Exercise controller safety case validation against mismatched evidence chain."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence("b" * 64)

    with pytest.raises(ValueError, match="digital twin"):
        controller_safety_case_evidence(artifact, transport, digital_twin)

    with pytest.raises(ValueError, match="transport evidence"):
        controller_safety_case_evidence(
            artifact, _transport_evidence("c" * 64), _digital_twin_evidence(controller_sha256)
        )


def test_controller_safety_case_rejects_invalid_public_input_types(tmp_path: Path) -> None:
    """Exercise controller safety case validation against invalid public input types."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)
    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)

    with pytest.raises(ValueError, match="safety_case must"):
        evaluate_controller_safety_case_readiness(cast(ControllerSafetyCaseEvidence, object()))
    with pytest.raises(ValueError, match="controller_artifact must"):
        controller_safety_case_evidence(cast(Artifact, object()), transport, digital_twin)
    with pytest.raises(ValueError, match="transport_evidence must"):
        controller_safety_case_evidence(artifact, cast(TransportDifferentiabilityEvidence, object()), digital_twin)
    with pytest.raises(ValueError, match="digital_twin_evidence must"):
        controller_safety_case_evidence(artifact, transport, cast(DigitalTwinUpdateEvidence, object()))
    with pytest.raises(ValueError, match="readiness must"):
        save_controller_safety_case_readiness(cast(SafetyCaseReadinessEvidence, object()), tmp_path / "readiness.json")
    with pytest.raises(ValueError, match="evidence must"):
        save_controller_safety_case_evidence(cast(ControllerSafetyCaseEvidence, object()), tmp_path / "evidence.json")
    with pytest.raises(ValueError, match="evidence must"):
        assert_controller_safety_case_admissible(
            cast(ControllerSafetyCaseEvidence, object()), artifact, transport, digital_twin
        )

    stale_schema = ControllerSafetyCaseEvidence(**{**evidence.__dict__, "schema_version": 2})
    with pytest.raises(ValueError, match="schema_version"):
        assert_controller_safety_case_admissible(stale_schema, artifact, transport, digital_twin)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("controller_artifact_sha256", "9" * 64, "controller_artifact_sha256"),
        ("formal_report_sha256", "9" * 64, "formal_report_sha256"),
        ("formal_backend", "explicit-state", "formal_backend"),
        ("formal_max_depth", 99, "formal_max_depth"),
        ("transport_evidence_sha256", "9" * 64, "transport_evidence_sha256"),
        ("digital_twin_evidence_sha256", "9" * 64, "digital_twin_evidence_sha256"),
        ("claim_status", "bounded but stale", "claim_status"),
    ],
)
def test_controller_safety_case_admission_rejects_each_evidence_field_drift(
    field_name: str, replacement: Any, message: str
) -> None:
    """Exercise controller safety case admission validation against each evidence field drift."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)
    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)
    drifted = ControllerSafetyCaseEvidence(**{**evidence.__dict__, field_name: replacement})

    with pytest.raises(ValueError, match=message):
        assert_controller_safety_case_admissible(drifted, artifact, transport, digital_twin)


def test_controller_safety_case_rejects_bad_transport_and_twin_claims() -> None:
    """Exercise controller safety case validation against bad transport and twin claims."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)

    failed_transport = _transport_evidence(controller_sha256)
    object.__setattr__(failed_transport, "audit_passed", False)
    with pytest.raises(ValueError, match="passed gradient audit"):
        controller_safety_case_evidence(artifact, failed_transport, digital_twin)

    non_jax_transport = _transport_evidence(controller_sha256)
    object.__setattr__(non_jax_transport, "backend", "numpy")
    with pytest.raises(ValueError, match="JAX backend"):
        controller_safety_case_evidence(artifact, non_jax_transport, digital_twin)

    non_improving_twin = _digital_twin_evidence(controller_sha256)
    object.__setattr__(non_improving_twin, "improved_over_baseline", False)
    with pytest.raises(ValueError, match="improve over baseline"):
        controller_safety_case_evidence(artifact, transport, non_improving_twin)

    missing_tsc_twin = _digital_twin_evidence(controller_sha256)
    object.__setattr__(missing_tsc_twin, "simulator_codes", ("TRANSP",))
    with pytest.raises(ValueError, match="TRANSP and TSC"):
        controller_safety_case_evidence(artifact, transport, missing_tsc_twin)


def test_controller_safety_case_rejects_non_passing_formal_proof() -> None:
    """Reject a controller safety case whose formal proof does not pass."""
    artifact = _controller_artifact()
    assert artifact.formal_verification is not None
    artifact.formal_verification.status = "blocked"
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)

    with pytest.raises(ValueError, match="safety-critical"):
        controller_safety_case_evidence(artifact, transport, digital_twin)
