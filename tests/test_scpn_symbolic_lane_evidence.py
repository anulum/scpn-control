# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Symbolic-lane evidence tests
"""Verify the symbolic-transition lane is at parity with the neuromorphic readout,
load-bearing, and NOT overstated as superior."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_MOD = ROOT / "validation" / "scpn_symbolic_lane_evidence.py"
_SPEC = importlib.util.spec_from_file_location("scpn_symbolic_lane_evidence", _MOD)
assert _SPEC and _SPEC.loader
evidence = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evidence
_SPEC.loader.exec_module(evidence)


def test_symbolic_lane_is_at_parity_with_neuromorphic_and_load_bearing() -> None:
    # Reduced but real seeds/steps (the committed report uses 4 seeds x 2 step counts).
    report = evidence.evaluate_evidence(seeds=(42, 7), step_counts=(240,))
    r = report["ratio_symbolic_over_neuromorphic"]
    # Parity: the symbolic controller tracks within a tight band of the neuromorphic
    # baseline on unseen seeds — neither a bypassed lane nor a superiority claim.
    assert report["parity_holds"] is True
    assert 0.85 <= r["min"] <= r["max"] <= 1.15
    assert 0.9 <= r["mean"] <= 1.1
    # Load-bearing: removing the drain saturates a_* and the symbolic controller diverges.
    assert report["load_bearing"]["diverges_without_drain"] is True
    assert report["load_bearing"]["undrained_over_reference"] > 3.0
    assert "NOT a superiority claim" in report["claim"]


def test_report_payload_digest_is_self_consistent() -> None:
    report = evidence.generate_report(seeds=(42,), step_counts=(240,))
    stored = report["payload_sha256"]
    assert isinstance(stored, str) and len(stored) == 64
    payload = dict(report)
    payload["payload_sha256"] = None
    recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == stored


def test_symbolic_reads_transition_outputs_but_neuromorphic_does_not() -> None:
    # The structural distinction the claim rests on: the symbolic controller's readout
    # reads transition-output places (load-bearing); the neuromorphic baseline does not.
    assert evidence._build_variant("symbolic").readout_reads_transition_outputs is True
    assert evidence._build_variant("neuromorphic").readout_reads_transition_outputs is False


def test_committed_report_exists_and_is_self_consistent() -> None:
    path = ROOT / "validation" / "reports" / "scpn_symbolic_lane_evidence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "scpn-control.symbolic-lane-evidence.v1"
    assert data["parity_holds"] is True
    assert data["load_bearing"]["diverges_without_drain"] is True
    stored = data["payload_sha256"]
    payload = dict(data)
    payload["payload_sha256"] = None
    recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == stored
