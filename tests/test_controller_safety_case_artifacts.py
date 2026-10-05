# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Safety Case Tests
"""Safety-case readiness artifact admission tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from safety_case_support import (
    _codac_runtime_payload,
    _controller_artifact,
    _digital_twin_evidence,
    _hdl_export_payload,
    _local_hil_replay_payload,
    _readiness_artifacts,
    _target_hardware_latency_payload,
    _transport_evidence,
    _websocket_runtime_payload,
    _write_readiness_file,
)

from scpn_control.control.codac_interface import load_codac_runtime_evidence
from scpn_control.control.safety_case import (
    ReadinessArtifactEvidence,
    controller_safety_case_evidence,
    evaluate_controller_safety_case_readiness_from_artifacts,
)
from scpn_control.phase.ws_phase_stream import (
    load_websocket_runtime_evidence,
)
from scpn_control.scpn.artifact import (
    compute_artifact_payload_sha256,
)
from scpn_control.scpn.fpga_export import load_hdl_export_evidence
from validation.validate_e2e_latency_evidence import build_e2e_latency_evidence_payload


def test_controller_safety_case_readiness_refuses_unqualified_external_artifacts(tmp_path: Path) -> None:
    """File hashes cannot qualify self-authored physics and review reports."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            artifacts,
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_unqualified_timing_artifact(tmp_path: Path) -> None:
    """Exercise controller safety case readiness validation against unqualified timing artifact."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    timing_uri = artifacts[1].artifact_uri
    local_payload = _target_hardware_latency_payload()
    hardware = cast(dict[str, Any], local_payload["target_hardware"])
    hardware["id"] = "local-host-unqualified"
    hardware["class"] = "unspecified-local"
    hardware["rt_kernel"] = "unknown"
    timing_digest = _write_readiness_file(tmp_path, timing_uri, build_e2e_latency_evidence_payload(local_payload))
    artifacts[1] = ReadinessArtifactEvidence(
        kind="target_hardware_timing",
        artifact_sha256=timing_digest,
        artifact_uri=timing_uri,
        producer="target-hardware-latency-bench",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="target hardware timing artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_timing_artifact_digest_mismatch(tmp_path: Path) -> None:
    """Exercise controller safety case readiness validation against timing artifact digest mismatch."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    artifacts[1] = ReadinessArtifactEvidence(
        kind="target_hardware_timing",
        artifact_sha256="f" * 64,
        artifact_uri=artifacts[1].artifact_uri,
        producer="target-hardware-latency-bench",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="artifact_sha256"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_hil_replay_artifact(tmp_path: Path) -> None:
    """Exercise controller safety case readiness validation against local HIL replay artifact."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    hil_uri = artifacts[2].artifact_uri
    hil_digest = _write_readiness_file(tmp_path, hil_uri, _local_hil_replay_payload())
    artifacts[2] = ReadinessArtifactEvidence(
        kind="hil_replay_evidence",
        artifact_sha256=hil_digest,
        artifact_uri=hil_uri,
        producer="target-hardware-hil-replay",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_hdl_export_artifact(tmp_path: Path) -> None:
    """Exercise controller safety case readiness validation against local HDL export artifact."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    hdl_uri = artifacts[3].artifact_uri
    hdl_digest = _write_readiness_file(
        tmp_path,
        hdl_uri,
        _hdl_export_payload(tmp_path, controller_sha256, facility_claim_allowed=False),
    )
    artifacts[3] = ReadinessArtifactEvidence(
        kind="hdl_export_evidence",
        artifact_sha256=hdl_digest,
        artifact_uri=hdl_uri,
        producer="target-hardware-hdl-export",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="local-only"):
        load_hdl_export_evidence(tmp_path / hdl_uri, require_facility_claim=True, artifact_root=tmp_path)
    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_hdl_export_controller_mismatch(tmp_path: Path) -> None:
    """Exercise controller safety case readiness validation against HDL export controller mismatch."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    hdl_uri = artifacts[3].artifact_uri
    hdl_digest = _write_readiness_file(
        tmp_path,
        hdl_uri,
        _hdl_export_payload(tmp_path, "b" * 64),
    )
    artifacts[3] = ReadinessArtifactEvidence(
        kind="hdl_export_evidence",
        artifact_sha256=hdl_digest,
        artifact_uri=hdl_uri,
        producer="target-hardware-hdl-export",
        generated_utc="2026-05-31T00:00:00Z",
    )

    loaded_hdl = load_hdl_export_evidence(tmp_path / hdl_uri, require_facility_claim=True, artifact_root=tmp_path)
    assert loaded_hdl.controller_artifact_sha256 != controller_sha256
    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_codac_runtime_artifact(tmp_path: Path) -> None:
    """Exercise controller safety case readiness validation against local CODAC runtime artifact."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    codac_uri = artifacts[4].artifact_uri
    codac_digest = _write_readiness_file(
        tmp_path,
        codac_uri,
        _codac_runtime_payload(facility_claim_allowed=False),
    )
    artifacts[4] = ReadinessArtifactEvidence(
        kind="codac_runtime_evidence",
        artifact_sha256=codac_digest,
        artifact_uri=codac_uri,
        producer="target-hardware-codac-runtime",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="local-only"):
        load_codac_runtime_evidence(tmp_path / codac_uri, require_facility_claim=True)
    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_websocket_runtime_artifact(tmp_path: Path) -> None:
    """Exercise controller safety case readiness validation against local WebSocket runtime artifact."""
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    websocket_uri = artifacts[5].artifact_uri
    websocket_digest = _write_readiness_file(
        tmp_path,
        websocket_uri,
        _websocket_runtime_payload(facility_claim_allowed=False),
    )
    artifacts[5] = ReadinessArtifactEvidence(
        kind="websocket_runtime_evidence",
        artifact_sha256=websocket_digest,
        artifact_uri=websocket_uri,
        producer="target-hardware-websocket-runtime",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="local-only"):
        load_websocket_runtime_evidence(tmp_path / websocket_uri, require_facility_claim=True)
    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )
