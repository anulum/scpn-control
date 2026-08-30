# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FUSION to SPO to CONTROL semantic exchange

"""Portable E2E over immutable FUSION and SPO receipts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scpn_phase_orchestrator.reactor_semantics import (
    ClockKind,
    ClockReference,
    coupled_transport_handoff_from_fusion_bytes,
    handoff_to_bytes,
)

from scpn_control.reactor_semantic_admission import (
    ReactorSemanticAdmissionPolicy,
    admission_decision_from_bytes,
    admission_decision_to_bytes,
    admit_reactor_semantic_handoff,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/torax_runtime_review_envelope_v1.json"
CONTROL_ROOT = Path(__file__).resolve().parents[1]
FUSION_SOURCE_REVISION = "314463489c95692d851cf6b9102ca733d878ca8a"
FUSION_FIXTURE_SHA256 = "b594e2f8b72056426d628b638f6a849ef39e75daddc827305002b109365596c4"
SPO_HANDOFF_SHA256 = "38885e7c8f72a349703f36620714fb416de5e1c003d4e53cf1cb9930e64df043"


def _reseal_handoff(record: dict[str, object]) -> bytes:
    body = record["payload"]
    canonical_body = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    record["payload_sha256"] = hashlib.sha256(canonical_body).hexdigest()
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def _policy(handoff_bytes: bytes) -> ReactorSemanticAdmissionPolicy:
    return ReactorSemanticAdmissionPolicy(
        expected_handoff_sha256=hashlib.sha256(handoff_bytes).hexdigest(),
        expected_source_schema="scpn-fusion-core.torax-runtime-review-envelope.v1",
        expected_source_revision=FUSION_SOURCE_REVISION,
        expected_source_envelope_sha256=FUSION_FIXTURE_SHA256,
        reference_clock=ClockReference(
            domain="simulation_monotonic",
            kind=ClockKind.SIMULATION_MONOTONIC,
            epoch="scenario_start",
            timestamp_ns=20_000_000,
            sample_rate_hz=100.0,
            latency_s=0.0,
            picosecond_offset=0,
            synchronized_to=None,
        ),
        max_evidence_age_ns=0,
        max_calibration_age_ns=20_000_000,
        allowed_calibration_ids=frozenset({"fusion.torax.simulation_declared_units.v1"}),
        allowed_transfer_function_ids=frozenset({"fusion.torax.identity_projection.v1"}),
    )


def test_exact_fusion_spo_control_public_bytes_exchange() -> None:
    """Cross the portable FUSION, SPO, and CONTROL byte boundaries."""
    fusion_bytes = FIXTURE.read_bytes()
    assert hashlib.sha256(fusion_bytes).hexdigest() == FUSION_FIXTURE_SHA256

    handoff = coupled_transport_handoff_from_fusion_bytes(
        fusion_bytes,
        expected_sha256=FUSION_FIXTURE_SHA256,
    )
    handoff_bytes = handoff_to_bytes(handoff)
    assert len(handoff_bytes) == 71_090
    assert hashlib.sha256(handoff_bytes).hexdigest() == SPO_HANDOFF_SHA256

    decision = admit_reactor_semantic_handoff(handoff_bytes, policy=_policy(handoff_bytes))
    decision_bytes = admission_decision_to_bytes(decision)
    assert admission_decision_from_bytes(decision_bytes) == decision
    assert decision.admitted is True
    assert decision.review_only is True
    assert decision.actionable is False
    assert decision.refusal_codes == ()


def test_admission_import_does_not_load_control_action_surface() -> None:
    """Keep the review boundary independent of CONTROL actuation modules."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CONTROL_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scpn_control.reactor_semantic_admission; "
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


@pytest.mark.parametrize("mutation", ["duplicate", "whitespace", "bom", "version", "action", "phase"])
def test_public_control_ingress_refuses_handoff_drift(mutation: str) -> None:
    """Refuse duplicate, noncanonical, version, action, and phase drift."""
    source = FIXTURE.read_bytes()
    valid = handoff_to_bytes(
        coupled_transport_handoff_from_fusion_bytes(
            source,
            expected_sha256=FUSION_FIXTURE_SHA256,
        )
    )
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
        else:
            record["payload"]["semantics"][0]["payload"]["phase_rad"] = 0.0
        changed = _reseal_handoff(record)

    decision = admit_reactor_semantic_handoff(changed, policy=_policy(valid))

    assert decision.admitted is False
    assert decision.actionable is False
    assert decision.refusal_codes == ("handoff_decode_failed",)
    assert decision.event_id is None
