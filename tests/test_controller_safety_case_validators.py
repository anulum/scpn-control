# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Safety-Case Mapping Validator Tests
"""Negative-path coverage for the controller safety-case mapping validators and
readiness-artifact resolution helpers, driven through crafted payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scpn_control.control.safety_case import (
    ReadinessArtifactEvidence,
    _controller_safety_case_evidence_from_mapping,
    _resolve_readiness_artifact_path,
    _safety_case_readiness_from_mapping,
    _validate_readiness_artifact,
)


def _evidence_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "controller_artifact_sha256": "a" * 64,
        "formal_report_sha256": "b" * 64,
        "formal_backend": "z3",
        "formal_max_depth": 8,
        "transport_evidence_sha256": "c" * 64,
        "digital_twin_evidence_sha256": "d" * 64,
        "claim_status": "bounded proof boundary",
    }


def _readiness_payload() -> dict[str, Any]:
    # A valid "blocked" readiness: every required digest is absent, so the
    # blocking reasons must list exactly those missing artifact names.
    return {
        "schema_version": 5,
        "safety_case_sha256": "a" * 64,
        "status": "blocked",
        "external_physics_validation_sha256": None,
        "target_hardware_timing_sha256": None,
        "hil_replay_evidence_sha256": None,
        "hdl_export_evidence_sha256": None,
        "codac_runtime_evidence_sha256": None,
        "websocket_runtime_evidence_sha256": None,
        "independent_safety_review_sha256": None,
        "blocking_reasons": (
            "external_physics_validation_sha256",
            "target_hardware_timing_sha256",
            "hil_replay_evidence_sha256",
            "hdl_export_evidence_sha256",
            "codac_runtime_evidence_sha256",
            "websocket_runtime_evidence_sha256",
            "independent_safety_review_sha256",
        ),
        "claim_status": "bounded promotion boundary",
        "promotion_admissible": False,
    }


class TestEvidenceMappingValidator:
    def test_accepts_valid_payload(self) -> None:
        evidence = _controller_safety_case_evidence_from_mapping(_evidence_payload())
        assert evidence.schema_version == 1

    def test_rejects_missing_key(self) -> None:
        payload = _evidence_payload()
        del payload["schema_version"]
        with pytest.raises(ValueError, match="payload is malformed"):
            _controller_safety_case_evidence_from_mapping(payload)

    def test_rejects_unsupported_schema_version(self) -> None:
        payload = _evidence_payload()
        payload["schema_version"] = 2
        with pytest.raises(ValueError, match="schema_version is unsupported"):
            _controller_safety_case_evidence_from_mapping(payload)

    def test_rejects_wrong_length_digest(self) -> None:
        payload = _evidence_payload()
        payload["controller_artifact_sha256"] = "a" * 63
        with pytest.raises(ValueError, match="must be a SHA-256 digest"):
            _controller_safety_case_evidence_from_mapping(payload)

    def test_rejects_non_hexadecimal_digest(self) -> None:
        payload = _evidence_payload()
        payload["transport_evidence_sha256"] = "z" * 64
        with pytest.raises(ValueError, match="must be a SHA-256 digest"):
            _controller_safety_case_evidence_from_mapping(payload)

    def test_rejects_negative_formal_max_depth(self) -> None:
        payload = _evidence_payload()
        payload["formal_max_depth"] = -1
        with pytest.raises(ValueError, match="formal_max_depth must be >= 0"):
            _controller_safety_case_evidence_from_mapping(payload)

    def test_rejects_unbounded_claim_status(self) -> None:
        payload = _evidence_payload()
        payload["claim_status"] = "unrestricted"
        with pytest.raises(ValueError, match="must state a bounded boundary"):
            _controller_safety_case_evidence_from_mapping(payload)


class TestReadinessMappingValidator:
    def test_accepts_valid_blocked_payload(self) -> None:
        readiness = _safety_case_readiness_from_mapping(_readiness_payload())
        assert readiness.status == "blocked"

    def test_rejects_missing_key(self) -> None:
        payload = _readiness_payload()
        del payload["safety_case_sha256"]
        with pytest.raises(ValueError, match="payload is malformed"):
            _safety_case_readiness_from_mapping(payload)

    def test_rejects_unsupported_schema_version(self) -> None:
        payload = _readiness_payload()
        payload["schema_version"] = 3
        with pytest.raises(ValueError, match="readiness schema_version is unsupported"):
            _safety_case_readiness_from_mapping(payload)

    def test_rejects_malformed_safety_case_digest(self) -> None:
        payload = _readiness_payload()
        payload["safety_case_sha256"] = "not-a-digest"
        with pytest.raises(ValueError, match="safety_case_sha256 must be a SHA-256 digest"):
            _safety_case_readiness_from_mapping(payload)

    def test_rejects_unsupported_status(self) -> None:
        payload = _readiness_payload()
        payload["status"] = "unknown"
        # blocking_reasons must clear so the status branch is the first failure.
        payload["blocking_reasons"] = ()
        with pytest.raises(ValueError, match="readiness status is unsupported"):
            _safety_case_readiness_from_mapping(payload)

    def test_rejects_promotion_ready_with_blocking_reasons(self) -> None:
        payload = _readiness_payload()
        digest = "e" * 64
        for name in (
            "external_physics_validation_sha256",
            "target_hardware_timing_sha256",
            "hil_replay_evidence_sha256",
            "hdl_export_evidence_sha256",
            "codac_runtime_evidence_sha256",
            "websocket_runtime_evidence_sha256",
            "independent_safety_review_sha256",
        ):
            payload[name] = digest
        payload["status"] = "promotion_ready"
        payload["blocking_reasons"] = ("external_physics_validation_sha256",)
        with pytest.raises(ValueError, match="promotion_ready cannot have blocking reasons"):
            _safety_case_readiness_from_mapping(payload)

    def test_rejects_promotion_ready_missing_evidence(self) -> None:
        payload = _readiness_payload()
        payload["status"] = "promotion_ready"
        payload["blocking_reasons"] = ()
        # all digests remain None → promotion_ready is missing required evidence.
        with pytest.raises(ValueError, match="promotion_ready is missing required evidence"):
            _safety_case_readiness_from_mapping(payload)

    def test_rejects_blocked_with_mismatched_blocking_reasons(self) -> None:
        payload = _readiness_payload()
        payload["blocking_reasons"] = ("external_physics_validation_sha256",)
        with pytest.raises(ValueError, match="blocking_reasons must match missing evidence"):
            _safety_case_readiness_from_mapping(payload)

    def test_rejects_unbounded_claim_status(self) -> None:
        payload = _readiness_payload()
        payload["claim_status"] = "unrestricted"
        with pytest.raises(ValueError, match="must state a bounded boundary"):
            _safety_case_readiness_from_mapping(payload)


class TestReadinessArtifactHelpers:
    def test_rejects_unsupported_artifact_kind(self) -> None:
        artifact = ReadinessArtifactEvidence(
            kind="not_a_real_kind",
            artifact_sha256="a" * 64,
            artifact_uri="reports/timing.json",
            producer="ci",
            generated_utc="2026-05-31T00:00:00Z",
        )
        with pytest.raises(ValueError, match="kind is unsupported"):
            _validate_readiness_artifact(artifact)

    def test_resolve_rejects_path_escaping_root(self, tmp_path: Path) -> None:
        artifact = ReadinessArtifactEvidence(
            kind="external_physics_validation",
            artifact_sha256="a" * 64,
            artifact_uri="../escape.json",
            producer="ci",
            generated_utc="2026-05-31T00:00:00Z",
        )
        with pytest.raises(ValueError, match="escapes artifact_root"):
            _resolve_readiness_artifact_path(artifact, tmp_path)
