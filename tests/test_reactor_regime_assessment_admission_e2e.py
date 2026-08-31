# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MIF to SPO assessment to CONTROL admission

"""Portable E2E over immutable public MIF and SPO byte contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scpn_phase_orchestrator.reactor_semantics import (
    ReactorRegimeAssessment,
    build_abstaining_regime_assessment,
    mif_merge_compression_handoff_from_mif_bytes,
    mif_merge_compression_handoff_to_bytes,
    regime_assessment_to_bytes,
)

from scpn_control.reactor_semantic_admission import (
    ReactorRegimeAssessmentAdmissionPolicy,
    admit_reactor_regime_assessment,
    regime_assessment_admission_decision_from_bytes,
    regime_assessment_admission_decision_to_bytes,
    regime_assessment_axis_custody_digest,
    regime_assessment_clock_custody_digest,
    regime_assessment_registry_custody_digest,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/mif_merge_compression_observation_v1.json"
CONTROL_ROOT = Path(__file__).resolve().parents[1]
MIF_SOURCE_SHA256 = "c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca"
SPO_HANDOFF_SHA256 = "c0f03b7c49346c39342598275556e8ac28c93138ba14f6e21d6739400e0edeb2"
SPO_ASSESSMENT_SHA256 = "3a5077b95d8b94b23a647d57a8b25f80cb798f712f00d0a34e71b95c600b154b"
SPO_REVISION = "c2a7581d58819060806c6f173da941c822103695"
SPO_WHEEL_SHA256 = "c2d7c0a5c0ad47f420fee02e54ccc28122bf8d128eb3b80ca51ba5f034320274"


def _assessment_chain() -> tuple[ReactorRegimeAssessment, bytes]:
    source = FIXTURE.read_bytes()
    assert len(source) == 2_475
    assert hashlib.sha256(source).hexdigest() == MIF_SOURCE_SHA256
    handoff = mif_merge_compression_handoff_from_mif_bytes(
        source,
        expected_sha256=MIF_SOURCE_SHA256,
    )
    handoff_bytes = mif_merge_compression_handoff_to_bytes(handoff)
    assert len(handoff_bytes) == 101_652
    assert hashlib.sha256(handoff_bytes).hexdigest() == SPO_HANDOFF_SHA256
    assessment = build_abstaining_regime_assessment(
        handoff,
        producer_revision=SPO_REVISION,
        producer_artifact_sha256=SPO_WHEEL_SHA256,
    )
    assessment_bytes = regime_assessment_to_bytes(assessment)
    assert len(assessment_bytes) == 11_943
    assert hashlib.sha256(assessment_bytes).hexdigest() == SPO_ASSESSMENT_SHA256
    return assessment, assessment_bytes


def _policy(
    assessment: ReactorRegimeAssessment,
    assessment_bytes: bytes,
) -> ReactorRegimeAssessmentAdmissionPolicy:
    return ReactorRegimeAssessmentAdmissionPolicy(
        expected_assessment_sha256=hashlib.sha256(assessment_bytes).hexdigest(),
        expected_assessment_id=assessment.assessment_id,
        expected_reactor_context_id=assessment.reactor_context_id,
        expected_configuration=assessment.configuration,
        expected_event_id=assessment.event_id,
        expected_producer_project=assessment.producer_project,
        expected_producer_revision=assessment.producer_revision,
        expected_producer_artifact_sha256=assessment.producer_artifact_sha256,
        expected_source_project=assessment.source_project,
        expected_source_revision=assessment.source_revision,
        expected_source_handoff_schema=assessment.source_handoff_schema,
        expected_source_handoff_sha256=assessment.source_handoff_sha256,
        expected_source_semantic_ids=assessment.source_semantic_ids,
        expected_assessment_schema_version=assessment.schema_version,
        expected_registry_custody_sha256=regime_assessment_registry_custody_digest(assessment),
        expected_clock_custody_sha256=regime_assessment_clock_custody_digest(assessment),
        expected_axis_custody_sha256=regime_assessment_axis_custody_digest(assessment),
        expected_axis_ids=tuple(axis.axis_id for axis in assessment.axes),
        expected_axis_provenance=tuple((axis.axis_id, axis.provenance_id) for axis in assessment.axes),
        checked_at_ns=assessment.evidence_timestamp_ns,
        max_evidence_age_ns=0,
    )


def _reseal_assessment(record: dict[str, object]) -> bytes:
    payload = record["payload"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    record["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def test_exact_mif_spo_control_assessment_byte_exchange() -> None:
    """Cross both public SPO boundaries and CONTROL's sealed review gate."""
    assessment, assessment_bytes = _assessment_chain()
    decision = admit_reactor_regime_assessment(
        assessment_bytes,
        policy=_policy(assessment, assessment_bytes),
    )
    decision_bytes = regime_assessment_admission_decision_to_bytes(decision)

    assert regime_assessment_admission_decision_from_bytes(decision_bytes) == decision
    assert decision.admitted is True
    assert decision.review_only is True
    assert decision.actionable is False
    assert decision.refusal_codes == ()
    assert len(assessment.axes) == 8
    assert sum(axis.disposition.value == "unknown" for axis in assessment.axes) == 7
    assert sum(axis.disposition.value == "not_applicable" for axis in assessment.axes) == 1


def test_e2e_uses_installed_public_spo_1_3_1_distribution() -> None:
    """Bind the integration test to the installed public package release."""
    distribution = importlib.metadata.distribution("scpn-phase-orchestrator")
    package_root = Path(str(distribution.locate_file("scpn_phase_orchestrator"))).resolve()

    assert distribution.version == "1.3.1"
    assert "site-packages" in package_root.parts


def test_assessment_admission_import_isolated_from_action_surfaces() -> None:
    """Keep assessment review independent of controller and actuator modules."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CONTROL_ROOT / "src")
    command = (
        "import sys; import scpn_control; "
        "import scpn_control.reactor_semantic_admission as admission; "
        "assert hasattr(admission, 'admit_reactor_regime_assessment'); "
        "assert not hasattr(scpn_control, 'admit_reactor_regime_assessment'); "
        "forbidden=('scpn_control.control','scpn_control.scpn','scpn_control.codac',"
        "'scpn_control.hardware'); "
        "assert not any(any(name.startswith(prefix) for prefix in forbidden) "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=CONTROL_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.stdout == ""


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "whitespace",
        "bom",
        "version",
        "action",
        "authority",
        "classification",
    ],
)
def test_control_ingress_refuses_resealed_or_noncanonical_assessment_drift(mutation: str) -> None:
    """Refuse duplicate, noncanonical, version, authority, and classifier drift."""
    assessment, valid = _assessment_chain()
    if mutation == "duplicate":
        changed = valid.replace(b'{"payload":', b'{"payload":{},"payload":', 1)
    elif mutation == "whitespace":
        changed = valid + b"\n"
    elif mutation == "bom":
        changed = b"\xef\xbb\xbf" + valid
    else:
        record = json.loads(valid)
        if mutation == "version":
            record["schema_version"] = "2.0.0"
        elif mutation == "action":
            record["payload"]["actionable"] = True
        elif mutation == "authority":
            record["payload"]["authority"] = "control"
        else:
            record["payload"]["classification_performed"] = True
        changed = _reseal_assessment(record)

    decision = admit_reactor_regime_assessment(changed, policy=_policy(assessment, valid))

    assert decision.admitted is False
    assert decision.actionable is False
    assert decision.refusal_codes == ("assessment_decode_failed",)
    assert decision.assessment_id is None
