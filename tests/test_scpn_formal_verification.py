# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SCPN Formal Verification Tests

"""Behavioural tests for SCPN formal reachability and safety proofs."""

from __future__ import annotations

import pytest

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    CTLFormula,
    EventuallyFires,
    FireLeadsToMarking,
    FormalPetriNetVerifier,
    LTLFormula,
    NeverCoMarked,
    PlaceInvariant,
    verify_formal_contracts,
)
from scpn_control.scpn.structure import StochasticPetriNet


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


def test_ctl_specs_compile_to_bounded_petri_net_obligations() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net())
    report = verifier.verify_ctl_specs(
        [
            CTLFormula.ag_bounded("AG_safe_markings", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            CTLFormula.ef_fires("EF_move_fires", "move"),
            CTLFormula.ag_not_comarked("AG_exclusive_transfer", "source", "sink", threshold=0.5),
            CTLFormula.ag_ef_marked("AG_EF_sink_marked", "sink", threshold=0.5),
        ],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == [
        "CTL:AG_safe_markings:AG",
        "CTL:EF_move_fires:EF",
        "CTL:AG_exclusive_transfer:AG",
        "CTL:AG_EF_sink_marked:AG_EF",
    ]


def test_ctl_specs_return_actionable_counterexample() -> None:
    report = FormalPetriNetVerifier(_transfer_net()).verify_ctl_specs(
        [CTLFormula.ag_bounded("AG_sink_upper_bound", {"sink": (0.0, 0.5)})],
        max_depth=2,
    )

    assert report.holds is False
    assert report.violations[0].property_name == "CTL:AG_sink_upper_bound:AG"
    assert report.violations[0].path == ["move"]
    assert report.violations[0].place == "sink"


def test_ltl_specs_compile_to_bounded_path_obligations() -> None:
    verifier = FormalPetriNetVerifier(_transfer_net())
    report = verifier.verify_ltl_specs(
        [
            LTLFormula.globally_bounded("G_safe_markings", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
            LTLFormula.eventually_fires("F_move_fires", "move"),
            LTLFormula.globally_fire_leads_to_marking("G_move_implies_sink", "move", "sink", threshold=0.5, within=0),
        ],
        max_depth=2,
    )

    assert report.holds is True
    assert report.checked_specs == [
        "LTL:G_safe_markings:G",
        "LTL:F_move_fires:F",
        "LTL:G_move_implies_sink:G_implies_F",
    ]


def test_ltl_specs_reject_unsupported_operator_domains() -> None:
    with pytest.raises(ValueError, match="unsupported LTL operator"):
        LTLFormula("bad_until", "U", "transition", {"transition": "move"})


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
