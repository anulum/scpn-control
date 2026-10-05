# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Safety Case Tests
"""Safety-case artifact-envelope validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from safety_case_support import (
    _controller_artifact,
    _digital_twin_evidence,
    _readiness_artifacts,
    _transport_evidence,
)

from scpn_control.control.safety_case import (
    ReadinessArtifactEvidence,
    assert_controller_safety_case_readiness_admissible,
    controller_safety_case_evidence,
    evaluate_controller_safety_case_readiness,
    evaluate_controller_safety_case_readiness_from_artifacts,
    load_controller_safety_case_readiness,
    save_controller_safety_case_readiness,
)
from scpn_control.scpn.artifact import (
    compute_artifact_payload_sha256,
)


def test_controller_safety_case_readiness_artifacts_reject_wrong_kind_and_unsafe_uri(tmp_path: Path) -> None:
    """Reject readiness artifacts with the wrong kind or an unsafe URI."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    valid_artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    with pytest.raises(ValueError, match="kind"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                ReadinessArtifactEvidence(
                    kind="external_physics_validation",
                    artifact_sha256="1" * 64,
                    artifact_uri="validation/reports/external/physics_validation.json",
                    producer="independent-validation-campaign",
                    generated_utc="2026-05-31T00:00:00Z",
                ),
            ),
            artifact_root=tmp_path,
        )
    with pytest.raises(ValueError, match="artifact_uri"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                ReadinessArtifactEvidence(
                    kind="external_physics_validation",
                    artifact_sha256="1" * 64,
                    artifact_uri="../outside.json",
                    producer="independent-validation-campaign",
                    generated_utc="2026-05-31T00:00:00Z",
                ),
                ReadinessArtifactEvidence(
                    kind="target_hardware_timing",
                    artifact_sha256="2" * 64,
                    artifact_uri="validation/reports/hardware/target_timing.json",
                    producer="target-hardware-latency-bench",
                    generated_utc="2026-05-31T00:00:00Z",
                ),
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
            ),
            artifact_root=tmp_path,
        )

    for unsafe_uri in ("", "file:validation/report.json", "validation//report.json", "validation\\report.json"):
        with pytest.raises(ValueError, match="artifact_uri"):
            evaluate_controller_safety_case_readiness_from_artifacts(
                evidence,
                (
                    ReadinessArtifactEvidence(
                        kind="external_physics_validation",
                        artifact_sha256="1" * 64,
                        artifact_uri=unsafe_uri,
                        producer="independent-validation-campaign",
                        generated_utc="2026-05-31T00:00:00Z",
                    ),
                    ReadinessArtifactEvidence(
                        kind="target_hardware_timing",
                        artifact_sha256="2" * 64,
                        artifact_uri="validation/reports/hardware/target_timing.json",
                        producer="target-hardware-latency-bench",
                        generated_utc="2026-05-31T00:00:00Z",
                    ),
                    valid_artifacts[2],
                    valid_artifacts[3],
                    valid_artifacts[4],
                ),
                artifact_root=tmp_path,
            )


def test_controller_safety_case_readiness_artifacts_reject_invalid_envelope_contracts(tmp_path: Path) -> None:
    """Reject readiness artifacts whose evidence envelopes violate their contracts."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    valid_artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    with pytest.raises(ValueError, match="non-empty tuple"):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, (), artifact_root=tmp_path)
    with pytest.raises(ValueError, match="non-empty tuple"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence, cast(tuple[ReadinessArtifactEvidence, ...], list(valid_artifacts)), artifact_root=tmp_path
        )
    with pytest.raises(ValueError, match="ReadinessArtifactEvidence"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence, cast(tuple[ReadinessArtifactEvidence, ...], (object(),)), artifact_root=tmp_path
        )

    invalid_digest = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="g" * 64,
        artifact_uri="validation/reports/external/physics_validation.json",
        producer="independent-validation-campaign",
        generated_utc="2026-05-31T00:00:00Z",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                invalid_digest,
                valid_artifacts[1],
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
                valid_artifacts[5],
            ),
            artifact_root=tmp_path,
        )

    empty_producer = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="1" * 64,
        artifact_uri="validation/reports/external/physics_validation.json",
        producer="",
        generated_utc="2026-05-31T00:00:00Z",
    )
    with pytest.raises(ValueError, match="producer"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                empty_producer,
                valid_artifacts[1],
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
                valid_artifacts[5],
            ),
            artifact_root=tmp_path,
        )

    empty_generated = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="1" * 64,
        artifact_uri="validation/reports/external/physics_validation.json",
        producer="independent-validation-campaign",
        generated_utc="",
    )
    with pytest.raises(ValueError, match="generated_utc"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                empty_generated,
                valid_artifacts[1],
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
                valid_artifacts[5],
            ),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_artifacts_reject_missing_files(tmp_path: Path) -> None:
    """Reject readiness artifacts whose referenced files are absent."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    artifacts[0] = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="1" * 64,
        artifact_uri="validation/reports/external/missing_physics_validation.json",
        producer="independent-validation-campaign",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="does not resolve"):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, tuple(artifacts), artifact_root=tmp_path)


def test_controller_safety_case_readiness_artifacts_reject_duplicate_kind(tmp_path: Path) -> None:
    """Reject duplicate readiness-artifact kinds."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (artifacts[0], artifacts[0], artifacts[1], artifacts[3]),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_manifest_round_trips(tmp_path: Path) -> None:
    """Preserve readiness fields through manifest serialisation."""
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
    path = tmp_path / "controller_safety_case_readiness.json"

    save_controller_safety_case_readiness(readiness, path)
    loaded = load_controller_safety_case_readiness(path)

    # The round trip preserves promotion_admissible=False (digest-only), and the gate
    # still refuses it post-load.
    assert loaded == readiness
    assert loaded.promotion_admissible is False
    with pytest.raises(ValueError, match="not admissible"):
        assert_controller_safety_case_readiness_admissible(loaded, evidence)


def test_controller_safety_case_readiness_artifact_bytes_remain_unqualified_after_load(tmp_path: Path) -> None:
    """Neither in-process file hashes nor serialisation grant promotion."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = _readiness_artifacts(tmp_path, controller_sha256)
    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, artifacts, artifact_root=tmp_path)
    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256=artifacts[0].artifact_sha256,
        target_hardware_timing_sha256=artifacts[1].artifact_sha256,
        hil_replay_evidence_sha256=artifacts[2].artifact_sha256,
        hdl_export_evidence_sha256=artifacts[3].artifact_sha256,
        codac_runtime_evidence_sha256=artifacts[4].artifact_sha256,
        websocket_runtime_evidence_sha256=artifacts[5].artifact_sha256,
        independent_safety_review_sha256=artifacts[6].artifact_sha256,
    )
    assert readiness.promotion_admissible is False
    with pytest.raises(ValueError, match="not admissible"):
        assert_controller_safety_case_readiness_admissible(readiness, evidence)

    path = tmp_path / "controller_safety_case_readiness_artifact.json"
    save_controller_safety_case_readiness(readiness, path)
    loaded = load_controller_safety_case_readiness(path)

    # Loading a manifest cannot elevate the unqualified result.
    assert loaded.promotion_admissible is False
    with pytest.raises(ValueError, match="not admissible"):
        assert_controller_safety_case_readiness_admissible(loaded, evidence)


def test_controller_safety_case_readiness_forged_admissible_flag_is_refused(tmp_path: Path) -> None:
    # A crafted manifest with promotion_admissible=True + valid-hex
    # ("0"*64) fabricated digests must NOT pass the admissibility gate. Because a deserialised
    # readiness is digest-only by construction, the forged flag is forced False on load, so the
    # gate refuses it even though the (self-referential) integrity digest still matches — the flip
    # is invisible to the hash, which is exactly why the flag must never be trusted from the wire.
    """Refuse a forged deserialised admissibility flag."""
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
    path = tmp_path / "forged_readiness.json"
    save_controller_safety_case_readiness(readiness, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["readiness"]["promotion_admissible"] = True  # forge the admissible flag
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    loaded = load_controller_safety_case_readiness(path)
    assert loaded.promotion_admissible is False
    with pytest.raises(ValueError, match="not admissible"):
        assert_controller_safety_case_readiness_admissible(loaded, evidence)


def test_controller_safety_case_readiness_manifest_rejects_tampering(tmp_path: Path) -> None:
    """Exercise controller safety case readiness manifest validation against tampering."""
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
    path = tmp_path / "controller_safety_case_readiness.json"
    save_controller_safety_case_readiness(readiness, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["readiness"]["status"] = "blocked"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        load_controller_safety_case_readiness(path)


def test_controller_safety_case_readiness_manifest_rejects_malformed_schema(tmp_path: Path) -> None:
    """Exercise controller safety case readiness manifest validation against malformed schema."""
    path = tmp_path / "bad_controller_safety_case_readiness.json"
    path.write_text(json.dumps({"schema_version": 99, "readiness": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_controller_safety_case_readiness(path)


def test_controller_safety_case_readiness_manifest_rejects_unreadable_and_malformed_payloads(tmp_path: Path) -> None:
    """Exercise controller safety case readiness manifest validation against unreadable and malformed payloads."""
    unreadable_json = tmp_path / "not_json_readiness.json"
    unreadable_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="readable JSON"):
        load_controller_safety_case_readiness(unreadable_json)

    non_object = tmp_path / "non_object_readiness.json"
    non_object.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_controller_safety_case_readiness(non_object)

    missing_payload = tmp_path / "missing_readiness.json"
    missing_payload.write_text(json.dumps({"schema_version": 1, "readiness": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="payload"):
        load_controller_safety_case_readiness(missing_payload)
