# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — HIL claim source tests.
"""Exercise HIL facility-claim admission through the public evidence API."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_control.control.hil_evidence_contracts import _sha256_json
from scpn_control.control.hil_harness import (
    ControlLoopMetrics,
    assert_hil_replay_evidence_admissible,
    hil_replay_evidence,
    load_hil_replay_evidence,
    save_hil_replay_evidence,
)


def _metrics() -> ControlLoopMetrics:
    """Provide internally consistent caller-owned timing data."""
    return ControlLoopMetrics(
        iterations=10,
        target_dt_us=1000.0,
        measured_dt_us=[20.0] * 10,
        p50_latency_us=20.0,
        p95_latency_us=20.0,
        p99_latency_us=20.0,
        max_latency_us=20.0,
        min_latency_us=20.0,
        mean_latency_us=20.0,
        jitter_std_us=0.0,
        overrun_count=0,
        overrun_fraction=0.0,
        sub_ms_achieved=True,
    )


def _local_evidence() -> dict[str, Any]:
    """Build local evidence through the public builder."""
    return hil_replay_evidence(_metrics(), controller_id="test-controller")


def test_caller_metadata_cannot_grant_target_hardware_claim() -> None:
    """A named rig and good timings do not prove independent target origin."""
    with pytest.raises(ValueError, match="independent"):
        hil_replay_evidence(
            _metrics(),
            controller_id="test-controller",
            target_hardware_id="jetson-orin-lab-01",
            target_hardware_class="jetson-orin-preempt-rt",
            rt_kernel="linux-rt-6.8.0-lab",
            deployment_claim_allowed=True,
        )


def test_resealed_qualified_payload_cannot_be_loaded_or_saved(tmp_path: Path) -> None:
    """Recomputed hashes cannot turn caller bytes into hardware evidence."""
    payload = copy.deepcopy(_local_evidence())
    hardware = payload["target_hardware"]
    hardware.update(
        target_hardware_id="jetson-orin-lab-01",
        target_hardware_class="jetson-orin-preempt-rt",
        rt_kernel="linux-rt-6.8.0-lab",
    )
    payload["admission"].update(
        deployment_claim_allowed=True,
        claim_status="qualified_target_hardware_deployment_evidence",
    )
    payload["replay_digest"] = _sha256_json(
        {key: payload[key] for key in ("controller_id", "target_hardware", "timing", "safety_events", "admission")}
    )
    payload["payload_sha256"] = _sha256_json({key: value for key, value in payload.items() if key != "payload_sha256"})
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="independent"):
        load_hil_replay_evidence(report)
    with pytest.raises(ValueError, match="independent"):
        assert_hil_replay_evidence_admissible(payload)
    destination = tmp_path / "new" / "report.json"
    with pytest.raises(ValueError, match="independent"):
        save_hil_replay_evidence(payload, destination)
    assert not destination.parent.exists()


def test_legacy_v1_contract_is_rejected(tmp_path: Path) -> None:
    """Old self-qualified schema cannot be reinterpreted under new rules."""
    payload = _local_evidence()
    payload["schema_version"] = "scpn-control.hil-replay-evidence.v1"
    payload["payload_sha256"] = _sha256_json({key: value for key, value in payload.items() if key != "payload_sha256"})
    report = tmp_path / "legacy.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_hil_replay_evidence(report)


def test_resealed_nonfinite_extension_is_rejected(tmp_path: Path) -> None:
    """An extension field cannot smuggle nonstandard JSON into HIL custody."""
    payload = _local_evidence()
    payload["extra"] = float("nan")
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "payload_sha256"},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report = tmp_path / "nonfinite.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_hil_replay_evidence(report)
