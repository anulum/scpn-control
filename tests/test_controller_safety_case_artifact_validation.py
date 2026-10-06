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
    _codac_runtime_payload,
    _controller_artifact,
    _digital_twin_evidence,
    _hdl_export_payload,
    _qualified_hil_replay_payload,
    _readiness_artifacts,
    _transport_evidence,
    _websocket_runtime_payload,
    _write_readiness_file,
)

from scpn_control.control import codac_evidence
from scpn_control.control.codac_evidence import CODAC_RUNTIME_EVIDENCE_QUALIFIED
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


def _checked_before_hil(
    artifacts: tuple[ReadinessArtifactEvidence, ...], first: str | None = None
) -> tuple[ReadinessArtifactEvidence, ...]:
    """Order the artifacts so that the in-process HIL replay is checked last.

    Artifacts are checked in the caller's order and the first refusal ends the
    evaluation. A HIL replay produced in this process is never admissible, so
    in the usual order nothing behind it is reached. ``first`` names the kind
    that a test wants checked before all others.
    """

    def rank(artifact: ReadinessArtifactEvidence) -> int:
        """Place the selected kind first and the HIL replay last."""
        return 0 if artifact.kind == first else 2 if artifact.kind == "hil_replay_evidence" else 1

    return tuple(sorted(artifacts, key=rank))


def test_readiness_resolves_custody_only_artifacts_before_a_later_refusal(tmp_path: Path) -> None:
    """The two custody-only artifacts are resolved and re-hashed, then the HIL replay is refused.

    External physics validation and the independent review have no verifier
    of their own; their files are located and hashed. Checked first, they
    pass, and the refusal that ends the evaluation is the HIL one behind them.
    """
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    custody_only = ("external_physics_validation", "independent_safety_review")
    supplied = _readiness_artifacts(tmp_path, controller_sha256)
    # The HDL export of this controller is checked too and passes: it is bound
    # to the controller artifact of the safety case.
    first = (*custody_only, "hdl_export_evidence")
    ordered = (
        tuple(item for kind in first for item in supplied if item.kind == kind)
        + tuple(item for item in supplied if item.kind == "hil_replay_evidence")
        + tuple(item for item in supplied if item.kind not in (*first, "hil_replay_evidence"))
    )
    assert [item.kind for item in ordered[:4]] == [*first, "hil_replay_evidence"]
    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, ordered, artifact_root=tmp_path)
    # A custody-only artifact whose file changed is refused where it is resolved.
    target = tmp_path / ordered[0].artifact_uri
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError) as refused:
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, ordered, artifact_root=tmp_path)
    assert "HIL replay" not in str(refused.value)


@pytest.mark.parametrize(
    ("kind", "fault", "message"),
    [
        ("hdl_export_evidence", "local", "HDL export artifact is not admissible"),
        ("hdl_export_evidence", "other-controller", "not bound to the safety-case controller artifact"),
        ("codac_runtime_evidence", "local", "CODAC runtime artifact is not admissible"),
        ("websocket_runtime_evidence", "local", "WebSocket runtime artifact is not admissible"),
    ],
)
def test_readiness_refuses_unqualified_runtime_and_export_artifacts(
    tmp_path: Path, kind: str, fault: str, message: str
) -> None:
    """A local-only or wrongly bound HDL, CODAC or WebSocket artifact ends the evaluation."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_checked_before_hil(_readiness_artifacts(tmp_path, controller_sha256), first=kind))
    assert artifacts[0].kind == kind
    if kind == "hdl_export_evidence":
        payload = (
            _hdl_export_payload(tmp_path, controller_sha256, facility_claim_allowed=False)
            if fault == "local"
            else _hdl_export_payload(tmp_path, "b" * 64)
        )
    elif kind == "codac_runtime_evidence":
        payload = _codac_runtime_payload(facility_claim_allowed=False)
    else:
        payload = _websocket_runtime_payload(facility_claim_allowed=False)
    digest = _write_readiness_file(tmp_path, artifacts[0].artifact_uri, payload)
    artifacts[0] = ReadinessArtifactEvidence(
        kind=kind,
        artifact_sha256=digest,
        artifact_uri=artifacts[0].artifact_uri,
        producer=artifacts[0].producer,
        generated_utc=artifacts[0].generated_utc,
    )
    with pytest.raises(ValueError, match=message):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, tuple(artifacts), artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("hil_replay_evidence", "HIL replay artifact is not admissible: .*independent target hardware provenance"),
        ("codac_runtime_evidence", "CODAC runtime artifact is not admissible: .*independent runtime"),
    ],
)
def test_readiness_refuses_a_claimed_qualification_it_cannot_verify(tmp_path: Path, kind: str, message: str) -> None:
    """A HIL or CODAC artifact that claims qualification is refused for lack of independent evidence.

    The payloads are formed as an external qualifier would issue them, with
    consistent digests. This package has no verifier for either claim, so no
    set of artifacts completes the evaluation.
    """
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_checked_before_hil(_readiness_artifacts(tmp_path, controller_sha256), first=kind))
    assert artifacts[0].kind == kind
    if kind == "hil_replay_evidence":
        payload = _qualified_hil_replay_payload()
    else:
        payload = _codac_runtime_payload(facility_claim_allowed=False)
        payload["facility_claim_allowed"] = True
        payload["claim_status"] = CODAC_RUNTIME_EVIDENCE_QUALIFIED
        payload["payload_sha256"] = codac_evidence._payload_sha256(payload)
    digest = _write_readiness_file(tmp_path, artifacts[0].artifact_uri, payload)
    artifacts[0] = ReadinessArtifactEvidence(
        kind=kind,
        artifact_sha256=digest,
        artifact_uri=artifacts[0].artifact_uri,
        producer=artifacts[0].producer,
        generated_utc=artifacts[0].generated_utc,
    )
    with pytest.raises(ValueError, match=message):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, tuple(artifacts), artifact_root=tmp_path)
