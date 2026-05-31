# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SCPN Formal Verification Tests

"""Behavioural tests for SCPN formal reachability and safety proofs."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    EventuallyFires,
    FireLeadsToMarking,
    FormalViolation,
    FormalPetriNetVerifier,
    NeverCoMarked,
    PlaceInvariant,
    verify_formal_contracts,
)
from scpn_control.scpn.structure import StochasticPetriNet

from scpn_control.scpn.z3_model_checking import (  # noqa: E402
    Z3BoundedModelChecker,
    Z3FormalVerificationReport,
    Z3ModelCheckingReport,
    build_blocked_z3_formal_report_payload,
    build_z3_formal_report_payload,
    validate_z3_formal_report_payload,
    verify_z3_formal_contracts,
    write_z3_formal_report,
)

requires_z3 = pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="z3-solver optional dependency absent")


def _transfer_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def test_reachability_finds_named_marking_without_random_trials() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).analyze_reachability(max_depth=2)

    assert report.holds is True
    assert report.reachable_count == 2
    assert {state.marking["sink"] for state in report.reachable_states} == {0.0, 1.0}
    assert report.fired_transitions == {"move"}


def test_marking_bounds_report_counterexample_path() -> None:
    bounds = {"sink": (0.0, 0.5)}
    report = FormalPetriNetVerifier(_transfer_net()).prove_marking_bounds(bounds, max_depth=2)

    assert report.holds is False
    assert report.violations[0].property_name == "marking_bounds"
    assert report.violations[0].place == "sink"
    assert report.violations[0].path == ["move"]
    assert report.violations[0].marking["sink"] == 1.0


def test_transition_liveness_distinguishes_dead_transition() -> None:
    net = StochasticPetriNet()
    net.add_place("p0", initial_tokens=0.0)
    net.add_place("p1", initial_tokens=0.0)
    net.add_transition("needs_token", threshold=1.0)
    net.add_arc("p0", "needs_token", weight=1.0)
    net.add_arc("needs_token", "p1", weight=1.0)
    net.compile()

    report = FormalPetriNetVerifier(net).prove_transition_liveness(max_depth=3)

    assert report.holds is False
    assert report.dead_transitions == {"needs_token"}
    assert report.violations[0].property_name == "transition_liveness"


def test_place_invariant_proves_token_conservation_structurally() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).prove_place_invariants(
        [PlaceInvariant("total_token_conserved", {"source": 1.0, "sink": 1.0})],
        max_depth=3,
    )

    assert report.holds is True
    assert report.checked_specs == ["total_token_conserved"]


def test_place_invariant_reports_transition_that_breaks_conservation() -> None:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("duplicate", threshold=1.0)
    net.add_arc("source", "duplicate", weight=1.0)
    net.add_arc("duplicate", "sink", weight=2.0)
    net.compile()

    report = FormalPetriNetVerifier(net).prove_place_invariants(
        [PlaceInvariant("total_token_conserved", {"source": 1.0, "sink": 1.0})],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "total_token_conserved"
    assert report.violations[0].transition == "duplicate"
    assert "changes invariant" in report.violations[0].message


def test_temporal_specs_cover_always_eventually_and_never() -> None:
    specs = [
        AlwaysBounded("all_markings_safe", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
        EventuallyFires("move_eventually_fires", "move"),
        NeverCoMarked("exclusive_source_sink", "source", "sink", threshold=0.5),
        AlwaysEventuallyMarked("sink_recoverable", "sink", threshold=0.5),
        FireLeadsToMarking("move_marks_sink", "move", "sink", threshold=0.5, within=0),
    ]

    report = FormalPetriNetVerifier(_transfer_net()).verify_temporal_specs(specs, max_depth=2)

    assert report.holds is True
    assert report.checked_specs == [
        "all_markings_safe",
        "move_eventually_fires",
        "exclusive_source_sink",
        "sink_recoverable",
        "move_marks_sink",
    ]


def test_temporal_response_checks_all_bounded_firing_paths() -> None:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("safe", initial_tokens=0.0)
    net.add_place("unsafe", initial_tokens=0.0)
    net.add_transition("actuate", threshold=1.0)
    net.add_transition("drop", threshold=1.0)
    net.add_arc("armed", "actuate", weight=1.0)
    net.add_arc("actuate", "safe", weight=1.0)
    net.add_arc("armed", "drop", weight=1.0)
    net.add_arc("drop", "unsafe", weight=1.0)
    net.compile()

    report = FormalPetriNetVerifier(net).verify_temporal_specs(
        [FireLeadsToMarking("drop_must_mark_safe", "drop", "safe", threshold=0.5, within=0)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].transition == "drop"
    assert report.violations[0].place == "safe"
    assert report.violations[0].path == ["drop"]


def test_temporal_recurrence_reports_nonrecoverable_marking() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).verify_temporal_specs(
        [AlwaysEventuallyMarked("source_recovers", "source", threshold=0.5)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "source_recovers"
    assert report.violations[0].path == ["move"]


def test_temporal_specs_return_actionable_counterexample() -> None:
    specs = [NeverCoMarked("sink_never_marked", "sink", "sink", threshold=0.5)]

    report = FormalPetriNetVerifier(_transfer_net()).verify_temporal_specs(specs, max_depth=2)

    assert report.holds is False
    assert report.violations[0].property_name == "sink_never_marked"
    assert report.violations[0].path == ["move"]


def test_verify_formal_contracts_combines_safety_liveness_and_temporal_specs() -> None:
    report = verify_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
    )

    assert report.holds is True
    assert report.reachability.reachable_count == 2
    assert report.safety.holds is True
    assert report.liveness.holds is True
    assert report.temporal.holds is True


def test_formal_verifier_rejects_non_integer_depth_and_nonfinite_bounds() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net())

    try:
        verifier.analyze_reachability(max_depth=True)
    except ValueError as exc:
        assert "max_depth" in str(exc)
    else:
        raise AssertionError("boolean max_depth must be rejected")

    try:
        verifier.prove_marking_bounds({"sink": (0.0, float("inf"))}, max_depth=2)
    except ValueError as exc:
        assert "numeric values" in str(exc)
    else:
        raise AssertionError("non-finite marking bounds must be rejected")


def test_z3_checker_rejects_uncompiled_net_before_solver_use() -> None:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)

    with pytest.raises(RuntimeError, match="compiled"):
        Z3BoundedModelChecker(net)


def test_z3_checker_rejects_invalid_safety_domains_before_solver_use() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())

    with pytest.raises(ValueError, match="max_depth"):
        checker.prove_marking_bounds({"source": (0.0, 1.0)}, max_depth=True)
    with pytest.raises(ValueError, match="must not be empty"):
        checker.prove_marking_bounds({}, max_depth=1)
    with pytest.raises(ValueError, match="unknown place"):
        checker.prove_marking_bounds({"missing": (0.0, 1.0)}, max_depth=1)
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        checker.prove_marking_bounds({"source": (1.0, 0.0)}, max_depth=1)


def test_z3_checker_rejects_invalid_temporal_domains_before_solver_use() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())

    with pytest.raises(ValueError, match="unknown transition"):
        checker.verify_temporal_specs([EventuallyFires("missing_fires", "missing")], max_depth=1)
    with pytest.raises(ValueError, match="unknown place"):
        checker.verify_temporal_specs([AlwaysEventuallyMarked("missing_marked", "missing")], max_depth=1)
    with pytest.raises(ValueError, match="within"):
        checker.verify_temporal_specs([FireLeadsToMarking("bad_window", "move", "sink", within=-1)], max_depth=1)


@requires_z3
def test_z3_model_checker_proves_safe_marking_bounds() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).prove_marking_bounds(
        {"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        max_depth=2,
    )

    assert report.holds is True
    assert report.backend == "z3"
    assert report.solver_status == "unsat"
    assert report.violations == []


@requires_z3
def test_z3_model_checker_returns_marking_bound_counterexample() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).prove_marking_bounds({"sink": (0.0, 0.5)}, max_depth=2)

    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].property_name == "marking_bounds"
    assert report.violations[0].place == "sink"
    assert report.violations[0].path == ["move"]
    assert report.violations[0].marking["sink"] == pytest.approx(1.0)


@requires_z3
def test_z3_model_checker_distinguishes_dead_transition_liveness() -> None:
    net = StochasticPetriNet()
    net.add_place("empty", initial_tokens=0.0)
    net.add_place("marked", initial_tokens=0.0)
    net.add_transition("needs_token", threshold=1.0)
    net.add_arc("empty", "needs_token", weight=1.0)
    net.add_arc("needs_token", "marked", weight=1.0)
    net.compile()

    report = Z3BoundedModelChecker(net).verify_temporal_specs(
        [EventuallyFires("dead_transition_detected", "needs_token")],
        max_depth=2,
    )

    assert report.holds is False
    assert report.solver_status == "unsat"
    assert report.violations[0].transition == "needs_token"
    assert "cannot fire" in report.violations[0].message


@requires_z3
def test_z3_temporal_specs_find_exclusivity_counterexample() -> None:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("a", initial_tokens=0.0)
    net.add_place("b", initial_tokens=0.0)
    net.add_transition("split", threshold=1.0)
    net.add_arc("armed", "split", weight=1.0)
    net.add_arc("split", "a", weight=1.0)
    net.add_arc("split", "b", weight=1.0)
    net.compile()

    report = Z3BoundedModelChecker(net).verify_temporal_specs(
        [NeverCoMarked("a_b_exclusive", "a", "b", threshold=0.5)],
        max_depth=1,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "a_b_exclusive"
    assert report.violations[0].path == ["split"]


@requires_z3
def test_z3_temporal_specs_prove_exclusivity_for_token_transfer() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [NeverCoMarked("source_sink_exclusive", "source", "sink", threshold=0.5)],
        max_depth=2,
    )

    assert report.holds is True
    assert report.solver_status == "unsat"
    assert report.checked_specs == ["source_sink_exclusive"]


@requires_z3
def test_z3_temporal_specs_prove_response_contract() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [FireLeadsToMarking("move_marks_sink", "move", "sink", threshold=0.5, within=0)],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == ["move_marks_sink"]


@requires_z3
def test_z3_temporal_specs_report_response_deadline_counterexample() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [FireLeadsToMarking("move_must_restore_source", "move", "source", threshold=0.5, within=0)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].transition == "move"
    assert report.violations[0].place == "source"
    assert report.violations[0].path == ["move"]


@requires_z3
def test_z3_temporal_specs_report_nonrecoverable_marking_path() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [AlwaysEventuallyMarked("sink_is_universally_marked", "sink", threshold=0.5)],
        max_depth=2,
    )

    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].place == "sink"
    assert report.violations[0].path == []


@requires_z3
def test_z3_temporal_specs_prove_all_supported_contracts_together() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [
            AlwaysBounded("bounded_transfer", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            EventuallyFires("move_fires", "move"),
            NeverCoMarked("exclusive_transfer", "source", "sink", threshold=0.5),
            AlwaysEventuallyMarked("source_initially_marked", "source", threshold=0.5),
            FireLeadsToMarking("move_marks_sink", "move", "sink", threshold=0.5, within=0),
        ],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == [
        "bounded_transfer",
        "move_fires",
        "exclusive_transfer",
        "source_initially_marked",
        "move_marks_sink",
    ]


@requires_z3
def test_z3_formal_report_writer_publishes_json_and_markdown(tmp_path: Path) -> None:
    report = verify_z3_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
    )
    json_path = tmp_path / "formal_z3.json"
    markdown_path = tmp_path / "formal_z3.md"

    write_z3_formal_report(report, json_path=json_path, markdown_path=markdown_path)

    assert report.holds is True
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scpn-control.z3-formal-report.v1"
    assert payload["status"] == "pass"
    assert payload["checked_specs"] == ["marking_bounds", "move_eventually_fires"]
    assert validate_z3_formal_report_payload(payload) == payload
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# SCPN Z3 Formal Verification Report" in markdown
    assert "bounded SMT evidence" in markdown
    assert payload["payload_sha256"] in markdown


def test_z3_formal_payload_records_fail_closed_counterexample_evidence() -> None:
    violation = FormalViolation(
        property_name="unsafe_bound",
        message="sink exceeds admitted control envelope",
        marking={"sink": 1.0},
        path=["move"],
        place="sink",
    )
    report = Z3FormalVerificationReport(
        holds=False,
        backend="z3",
        max_depth=2,
        safety=Z3ModelCheckingReport(False, "z3", 2, "sat", [violation], ["unsafe_bound"]),
        temporal=Z3ModelCheckingReport(True, "z3", 2, "unsat", [], ["response_ok"]),
    )

    payload = build_z3_formal_report_payload(report)

    assert payload["status"] == "fail"
    assert payload["holds"] is False
    assert payload["checked_specs"] == ["marking_bounds", "unsafe_bound", "response_ok"]
    assert payload["safety"]["violations"][0]["path"] == ["move"]
    assert validate_z3_formal_report_payload(payload) == payload


@requires_z3
def test_z3_formal_report_validator_rejects_tampered_checked_specs(tmp_path: Path) -> None:
    report = verify_z3_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        temporal_specs=[EventuallyFires("move_eventually_fires", "move")],
    )
    json_path = tmp_path / "formal_z3.json"
    markdown_path = tmp_path / "formal_z3.md"
    write_z3_formal_report(report, json_path=json_path, markdown_path=markdown_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["checked_specs"] = ["marking_bounds"]

    with pytest.raises(ValueError, match="payload_sha256"):
        validate_z3_formal_report_payload(payload)


def test_z3_formal_report_validator_rejects_inconsistent_safety_case_payloads() -> None:
    valid = build_blocked_z3_formal_report_payload("z3-solver unavailable")

    tampered = dict(valid)
    tampered["status"] = "pass"
    tampered["holds"] = False
    tampered["safety"] = {}
    tampered["temporal"] = {}
    tampered["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_z3_formal_report_payload(tampered)

    blocked_with_sections = dict(valid)
    blocked_with_sections["safety"] = {"holds": False}
    blocked_with_sections["payload_sha256"] = build_blocked_z3_formal_report_payload("z3-solver unavailable")[
        "payload_sha256"
    ]
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_z3_formal_report_payload(blocked_with_sections)


def _refresh_z3_payload_digest(payload: dict[str, object]) -> dict[str, object]:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def test_z3_formal_report_validator_rejects_semantically_invalid_blocked_reports() -> None:
    payload = build_blocked_z3_formal_report_payload("z3-solver unavailable")

    holds_payload = _refresh_z3_payload_digest({**payload, "holds": True})
    with pytest.raises(ValueError, match="must not hold"):
        validate_z3_formal_report_payload(holds_payload)

    missing_reason = dict(payload)
    missing_reason.pop("reason")
    _refresh_z3_payload_digest(missing_reason)
    with pytest.raises(ValueError, match="include a reason"):
        validate_z3_formal_report_payload(missing_reason)

    section_payload = _refresh_z3_payload_digest({**payload, "safety": {"holds": False}})
    with pytest.raises(ValueError, match="must not carry proof sections"):
        validate_z3_formal_report_payload(section_payload)


def test_z3_formal_report_validator_rejects_semantically_invalid_pass_fail_reports() -> None:
    valid = build_z3_formal_report_payload(
        Z3FormalVerificationReport(
            holds=True,
            backend="z3",
            max_depth=1,
            safety=Z3ModelCheckingReport(True, "z3", 1, "unsat", [], ["safety_ok"]),
            temporal=Z3ModelCheckingReport(True, "z3", 1, "unsat", [], ["temporal_ok"]),
        )
    )

    pass_without_hold = _refresh_z3_payload_digest({**valid, "holds": False})
    with pytest.raises(ValueError, match="passing"):
        validate_z3_formal_report_payload(pass_without_hold)

    fail_with_hold = _refresh_z3_payload_digest({**valid, "status": "fail", "holds": True})
    with pytest.raises(ValueError, match="failed"):
        validate_z3_formal_report_payload(fail_with_hold)

    depth_mismatch = dict(valid)
    depth_mismatch["safety"] = {**valid["safety"], "max_depth": 2}
    _refresh_z3_payload_digest(depth_mismatch)
    with pytest.raises(ValueError, match="depth"):
        validate_z3_formal_report_payload(depth_mismatch)

    bad_solver_status = dict(valid)
    bad_solver_status["temporal"] = {**valid["temporal"], "solver_status": "unchecked"}
    _refresh_z3_payload_digest(bad_solver_status)
    with pytest.raises(ValueError, match="solver_status"):
        validate_z3_formal_report_payload(bad_solver_status)

    checked_spec_mismatch = _refresh_z3_payload_digest({**valid, "checked_specs": ["marking_bounds"]})
    with pytest.raises(ValueError, match="checked_specs"):
        validate_z3_formal_report_payload(checked_spec_mismatch)


def test_blocked_z3_formal_report_is_schema_versioned_and_fail_closed() -> None:
    payload = build_blocked_z3_formal_report_payload("z3-solver unavailable")

    assert payload["schema_version"] == "scpn-control.z3-formal-report.v1"
    assert payload["status"] == "blocked"
    assert payload["holds"] is False
    assert payload["safety"] is None
    assert payload["temporal"] is None
    assert validate_z3_formal_report_payload(payload) == payload
