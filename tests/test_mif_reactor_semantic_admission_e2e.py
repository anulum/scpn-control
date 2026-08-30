# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MIF to SPO to CONTROL semantic exchange

"""Portable MIF producer to SPO semantics to CONTROL decision exchange."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scpn_phase_orchestrator.reactor_semantics import (
    MIFMergeCompressionHandoff,
    SemanticCarrier,
    mif_merge_compression_handoff_from_mif_bytes,
    mif_merge_compression_handoff_to_bytes,
)

from scpn_control.reactor_semantic_admission import (
    admission_decision_from_bytes,
    admission_decision_to_bytes,
)
from scpn_control.reactor_semantic_admission.mif_admission import (
    MIFReactorSemanticAdmissionPolicy,
    admit_mif_reactor_semantic_handoff,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/mif_merge_compression_observation_v1.json"
CONTROL_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca"
HANDOFF_SHA256 = "c0f03b7c49346c39342598275556e8ac28c93138ba14f6e21d6739400e0edeb2"


def _policy(handoff: MIFMergeCompressionHandoff) -> MIFReactorSemanticAdmissionPolicy:
    observables = handoff.observables
    semantics = handoff.semantics
    return MIFReactorSemanticAdmissionPolicy(
        expected_handoff_sha256=HANDOFF_SHA256,
        expected_source_schema="scpn-mif-core.merge-compression-observation.v1",
        expected_source_revision="f60dbae4b2ea3344ac0cb086a3b7d248d65cf92f",
        expected_source_envelope_sha256=SOURCE_SHA256,
        expected_event_id="shot_2026_08_30.event_0001",
        expected_context_id="spo.mif.frc_compression.c780706abd5a0b185a95e857",
        expected_registry_version="1.0.0",
        expected_registry_digest=("786d9542ce76c56dd7748fa948b17efed6c073525e527ce90e6d5e29a2d00090"),
        expected_observation_clock=observables[0].clock,
        reference_clock=observables[0].clock,
        max_evidence_age_ns=0,
        max_calibration_age_ns=0,
        allowed_calibration_ids=frozenset({"mif.merge_compression.model_declared_units.v1"}),
        allowed_transfer_function_ids=frozenset({"mif.merge_compression.identity_projection.v1"}),
        expected_observable_ids=frozenset(item.observable_id for item in observables),
        expected_semantic_carriers=tuple(sorted((item.phase_id, item.carrier_type) for item in semantics)),
        required_numerical_phase_ids=frozenset(
            item.phase_id for item in semantics if item.carrier_type is SemanticCarrier.NUMERICAL_PHASE
        ),
        required_provenance_attributes=(
            ("backend", "python"),
            ("backend_version", "0.1.1"),
            ("uncertainty_basis", "serialized_model_state_not_physical_uncertainty"),
        ),
        min_numerical_observability=1.0,
        min_numerical_confidence=1.0,
        max_numerical_circular_std_rad=0.0,
    )


def _reseal_handoff(record: dict[str, object]) -> bytes:
    body = record["payload"]
    canonical_body = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    record["payload_sha256"] = hashlib.sha256(canonical_body).hexdigest()
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def test_exact_mif_spo_control_public_bytes_exchange() -> None:
    """Cross the MIF producer, installed SPO, and CONTROL byte boundaries."""
    source = FIXTURE.read_bytes()
    assert len(source) == 2_475
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256

    handoff = mif_merge_compression_handoff_from_mif_bytes(
        source,
        expected_sha256=SOURCE_SHA256,
    )
    handoff_bytes = mif_merge_compression_handoff_to_bytes(handoff)
    assert len(handoff_bytes) == 101_652
    assert hashlib.sha256(handoff_bytes).hexdigest() == HANDOFF_SHA256

    decision = admit_mif_reactor_semantic_handoff(
        handoff_bytes,
        policy=_policy(handoff),
    )
    decision_bytes = admission_decision_to_bytes(decision)

    assert admission_decision_from_bytes(decision_bytes) == decision
    assert decision.admitted is True
    assert decision.review_only is True
    assert decision.actionable is False
    assert decision.refusal_codes == ()


def test_mif_admission_import_does_not_load_control_action_surface() -> None:
    """Keep the MIF review boundary independent of CONTROL action modules."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CONTROL_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scpn_control.reactor_semantic_admission.mif_admission; "
            "assert not any(name.startswith('scpn_control.control') for name in sys.modules)",
        ],
        cwd=CONTROL_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.stdout == ""


def test_public_control_ingress_refuses_authority_mutation() -> None:
    """Reject a resealed handoff that attempts to become actionable."""
    source = FIXTURE.read_bytes()
    handoff = mif_merge_compression_handoff_from_mif_bytes(
        source,
        expected_sha256=SOURCE_SHA256,
    )
    valid = mif_merge_compression_handoff_to_bytes(handoff)
    record = json.loads(valid)
    record["payload"]["actionable"] = True
    changed = _reseal_handoff(record)

    decision = admit_mif_reactor_semantic_handoff(
        changed,
        policy=_policy(handoff),
    )

    assert decision.admitted is False
    assert decision.actionable is False
    assert decision.refusal_codes == ("handoff_decode_failed",)
    assert decision.event_id is None
