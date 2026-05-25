# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SCPN Formal Verification Tests

"""Behavioural tests for SCPN formal reachability and safety proofs."""

from __future__ import annotations

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    EventuallyFires,
    FormalPetriNetVerifier,
    NeverCoMarked,
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


def test_temporal_specs_cover_always_eventually_and_never() -> None:
    specs = [
        AlwaysBounded("all_markings_safe", {"source": (0.0, 1.0), "sink": (0.0, 1.0)}),
        EventuallyFires("move_eventually_fires", "move"),
        NeverCoMarked("exclusive_source_sink", "source", "sink", threshold=0.5),
    ]

    report = FormalPetriNetVerifier(_transfer_net()).verify_temporal_specs(specs, max_depth=2)

    assert report.holds is True
    assert report.checked_specs == ["all_markings_safe", "move_eventually_fires", "exclusive_source_sink"]


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
