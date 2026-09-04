# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exact-current LIF receipt verification.

"""Reproduce every numerical digest in the exact-current LIF receipt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any, cast

from scpn_control.scpn import (
    ExactCurrentLIFProfileBinding,
    ExactCurrentLIFRuntime,
    ExactCurrentLIFTransitionTick,
)
from scpn_control.scpn.exact_current_lif_runtime import (
    PACKET_SCHEMA,
    PROFILE_NAME,
    PROFILE_SCHEMA,
    SC_CONTRACT_COMMIT,
    SC_IMPLEMENTATION_COMMIT,
    SC_MODEL_SOURCE_SHA256,
    SC_PROFILE_DIGEST,
    SC_PROFILE_SHA256,
    SC_REFERENCE_PACKET_SHA256,
    STATE_SCHEMA,
)

_RECEIPT = Path("validation/reports/exact_current_lif_runtime_receipt.json")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _reference_bytes() -> bytes:
    return (
        resources.files("sc_neurocore.neurons")
        .joinpath("reference_trace_data/exact_current_lif_multitick_v1.json")
        .read_bytes()
    )


def test_receipt_reproduces_exact_public_runtime_packet() -> None:
    """Re-execute the immutable SC input and match all receipt digests."""
    receipt = cast(dict[str, Any], json.loads(_RECEIPT.read_bytes()))
    reference_bytes = _reference_bytes()
    reference = cast(dict[str, Any], json.loads(reference_bytes))
    binding = ExactCurrentLIFProfileBinding.from_installed_reference()
    runtime = ExactCurrentLIFRuntime(("plasma-transition",), binding, shot_id="mif-reference-v1")
    ticks = tuple(
        ExactCurrentLIFTransitionTick(
            tick["duration_ms"],
            (tuple(tick["currents"]),),
        )
        for tick in reference["ticks"]
    )
    execution = runtime.execute(ticks)
    packet = cast(dict[str, Any], json.loads(execution.packets[0].packet_json))
    observed = {
        "aggregate_execution_sha256": execution.sha256,
        "event_count": len(packet["events"]),
        "events_sha256": _digest(packet["events"]),
        "final_state": packet["final_state"],
        "final_state_sha256": _digest(packet["final_state"]),
        "initial_state_sha256": _digest(packet["initial_state"]),
        "packet_sha256": execution.packets[0].sha256,
        "reference_artifact_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "state_sample_count": len(packet["state_trace"]),
        "state_trace_sha256": _digest(packet["state_trace"]),
        "tick_count": len(packet["ticks"]),
        "ticks_sha256": _digest(packet["ticks"]),
    }

    assert packet == reference
    assert observed == receipt["reference_execution"]


def test_receipt_binds_exact_commits_profile_and_semantics() -> None:
    """Receipt identities match code constants and the implementation parent."""
    receipt = cast(dict[str, Any], json.loads(_RECEIPT.read_bytes()))
    implementation_commit = receipt["control"]["implementation_commit"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{implementation_commit}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert receipt["schema"] == "scpn-control.exact-current-lif-runtime-receipt.v1"
    assert implementation_commit == "783f90d93590365a2725b77c22f82ca34a3afda3"
    assert receipt["control"]["public_entry_point"] == ("scpn_control.scpn.ExactCurrentLIFRuntime")
    assert receipt["sc_neurocore"] == {
        "contract_commit": SC_CONTRACT_COMMIT,
        "distribution_version": "3.16.0",
        "implementation_commit": SC_IMPLEMENTATION_COMMIT,
        "package": "sc-neurocore",
    }
    assert receipt["profile"] == {
        "artifact_sha256": SC_PROFILE_SHA256,
        "canonical_sha256": SC_PROFILE_DIGEST,
        "model_source_sha256": SC_MODEL_SOURCE_SHA256,
        "name": PROFILE_NAME,
        "packet_schema": PACKET_SCHEMA,
        "profile_schema": PROFILE_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "units": {
            "current": "normalized_current",
            "resistance": "normalized_resistance",
            "time": "ms",
            "voltage": "normalized_voltage",
        },
    }
    assert receipt["reference_execution"]["reference_artifact_sha256"] == (SC_REFERENCE_PACKET_SHA256)
    assert receipt["semantics"] == {
        "input_delivery": "simultaneous current contributions summed at tick start",
        "numeric": "IEEE-754 binary64/round_to_nearest_ties_to_even/fail_closed_non_finite",
        "reset_boundary": "explicit_shot_reset_only",
        "rng": "none",
        "solver": "closed_form_piecewise_constant_event_driven",
        "threshold_comparison": "greater_than_or_equal",
    }
    assert receipt["verification"] == {
        "branch_coverage_percent": 100,
        "covered_branches": 80,
        "covered_statements": 285,
        "reference_comparison": "complete packet equality",
        "result": "pass",
        "statement_coverage_percent": 100,
    }
