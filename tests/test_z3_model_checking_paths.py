# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Z3 model-checking solver, compile and report-schema branches

"""Solver-status, formula-compilation and report-schema branches of the Z3
bounded model checker.

Covers the optional-dependency guard, the ``unknown`` solver-status fall-through
on every proof obligation, the bounded trigger->response latency contract guards
(including the denormal underflow that collapses the latency window), the CTL/LTL
facade compilation errors, the inhibitor-arc transition relation, and the
single-field tampering rejections of the schema-versioned Z3 formal report.
"""

from __future__ import annotations

import builtins
import importlib.util
from typing import Any

import pytest

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    CTLFormula,
    FireLeadsToMarking,
    LTLFormula,
    NeverCoMarked,
)
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.z3_model_checking import Z3BoundedModelChecker, _require_z3

requires_z3 = pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="z3-solver optional dependency absent")


# ── net builders ──────────────────────────────────────────────────────


def _transfer_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def _latency_chain_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("armed", initial_tokens=1.0)
    net.add_place("staging", initial_tokens=0.0)
    net.add_place("ready", initial_tokens=0.0)
    net.add_place("safe", initial_tokens=0.0)
    net.add_transition("trigger", threshold=1.0)
    net.add_transition("stage", threshold=1.0)
    net.add_transition("response", threshold=1.0)
    net.add_arc("armed", "trigger", weight=1.0)
    net.add_arc("trigger", "staging", weight=1.0)
    net.add_arc("staging", "stage", weight=1.0)
    net.add_arc("stage", "ready", weight=1.0)
    net.add_arc("ready", "response", weight=1.0)
    net.add_arc("response", "safe", weight=1.0)
    net.compile()
    return net


def _inhibitor_net() -> StochasticPetriNet:
    """A guard place inhibits the firing transition."""
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("guard", initial_tokens=0.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("guard", "move", weight=1.0, inhibitor=True)
    net.add_arc("move", "sink", weight=1.0)
    net.compile(allow_inhibitor=True)
    return net


# ── optional-dependency guard ──────────────────────────────────


def _block_z3_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "z3":
            raise ModuleNotFoundError("No module named 'z3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_require_z3_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_z3_import(monkeypatch)
    with pytest.raises(RuntimeError, match="requires the optional z3-solver package"):
        _require_z3()


def test_checker_rejects_compiled_net_without_weight_matrices() -> None:
    net = _transfer_net()
    net.W_in = None  # a corrupted post-compile state must fail closed
    with pytest.raises(RuntimeError, match="compiled"):
        Z3BoundedModelChecker(net)


# ── unknown solver-status fall-through on every obligation ────────────


class _UnknownSolver:
    """Wraps the live solver but always reports an indeterminate result."""

    def __init__(self, z3: Any) -> None:
        self._z3 = z3

    def add(self, *args: Any) -> None:
        return None

    def check(self) -> Any:
        return self._z3.unknown

    def model(self) -> Any:  # pragma: no cover - never reached on an unknown verdict
        raise AssertionError("model() must not be consulted on an unknown verdict")


def _force_unknown(checker: Z3BoundedModelChecker, monkeypatch: pytest.MonkeyPatch) -> None:
    original = checker._transition_system

    def patched(max_depth: int, *, weight_bounds: Any = None) -> Any:
        z3, _solver, markings, firings, idle = original(max_depth, weight_bounds=weight_bounds)
        return z3, _UnknownSolver(z3), markings, firings, idle

    monkeypatch.setattr(checker, "_transition_system", patched)


@requires_z3
def test_marking_bounds_reports_indeterminate_solver_status(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = Z3BoundedModelChecker(_transfer_net())
    _force_unknown(checker, monkeypatch)
    report = checker.prove_marking_bounds({"sink": (0.0, 1.0)}, max_depth=2)
    assert report.holds is False
    assert report.solver_status == "unknown"
    assert report.violations == []


@requires_z3
def test_trigger_latency_reports_indeterminate_solver_status(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    _force_unknown(checker, monkeypatch)
    report = checker.verify_trigger_response_latency(
        trigger_transition="trigger",
        response_transition="response",
        max_latency_ns=20.0,
        tick_period_ns=10.0,
        max_depth=3,
    )
    assert report.holds is False
    assert report.solver_status == "unknown"
    assert report.checked_specs == ["trigger_response_latency"]


@requires_z3
@pytest.mark.parametrize(
    "spec",
    [
        NeverCoMarked("co_marked", "source", "sink", threshold=0.5),
        FireLeadsToMarking("response", "move", "sink", threshold=0.5, within=0),
        AlwaysEventuallyMarked("recurs", "sink", threshold=0.5),
    ],
)
def test_temporal_obligations_report_indeterminate_solver_status(spec: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    checker = Z3BoundedModelChecker(_transfer_net())
    _force_unknown(checker, monkeypatch)
    report = checker.verify_temporal_specs([spec], max_depth=2)
    assert report.holds is False
    assert report.solver_status == "unknown"


# ── verify_temporal_specs dispatch branches ───────────────────────────


@requires_z3
def test_temporal_specs_short_circuit_on_failed_marking_bound() -> None:
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [AlwaysBounded("too_tight", {"sink": (0.0, 0.5)})],
        max_depth=2,
    )
    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.checked_specs == ["too_tight"]


def test_temporal_specs_reject_unsupported_specification_type() -> None:
    class _ForeignSpec:
        name = "foreign"

    with pytest.raises(TypeError, match="unsupported temporal specification: _ForeignSpec"):
        Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs([_ForeignSpec()], max_depth=1)  # type: ignore[list-item]


# ── trigger->response latency contract guards ─────────────────────────


def test_trigger_latency_rejects_non_positive_max_latency() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    with pytest.raises(ValueError, match="max_latency_ns must be finite and > 0"):
        checker.verify_trigger_response_latency(
            trigger_transition="trigger",
            response_transition="response",
            max_latency_ns=0.0,
            tick_period_ns=10.0,
            max_depth=3,
        )


def test_trigger_latency_rejects_non_positive_tick_period() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    with pytest.raises(ValueError, match="tick_period_ns must be finite and > 0"):
        checker.verify_trigger_response_latency(
            trigger_transition="trigger",
            response_transition="response",
            max_latency_ns=20.0,
            tick_period_ns=0.0,
            max_depth=3,
        )


def test_trigger_latency_rejects_window_that_underflows_to_zero_steps() -> None:
    # A denormal latency divided by a huge tick period underflows the ratio to
    # 0.0, so the ceil-derived window collapses below one step.
    checker = Z3BoundedModelChecker(_latency_chain_net())
    with pytest.raises(ValueError, match="computed latency window must be >= 1"):
        checker.verify_trigger_response_latency(
            trigger_transition="trigger",
            response_transition="response",
            max_latency_ns=5e-324,
            tick_period_ns=1e308,
            max_depth=3,
        )


def test_trigger_latency_rejects_depth_below_latency_window() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    with pytest.raises(ValueError, match="max_depth must be >= ceil"):
        checker.verify_trigger_response_latency(
            trigger_transition="trigger",
            response_transition="response",
            max_latency_ns=50.0,
            tick_period_ns=10.0,
            max_depth=2,
        )


@requires_z3
def test_trigger_latency_fails_closed_when_horizon_exceeds_depth() -> None:
    # max_depth == latency window: every step's deadline lands at or beyond the
    # horizon, so no obligation can be discharged within depth.
    checker = Z3BoundedModelChecker(_latency_chain_net())
    report = checker.verify_trigger_response_latency(
        trigger_transition="trigger",
        response_transition="response",
        max_latency_ns=20.0,
        tick_period_ns=10.0,
        max_depth=2,
    )
    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].property_name == "trigger_response_latency"
    assert "latency horizon exceeds configured depth" in report.violations[0].message


# ── CTL/LTL facade compilation errors ─────────────────────────────────


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        (
            CTLFormula(name="ag", operator="AG", target="marking_bounds", params={"bounds": "not-a-dict"}),
            "requires marking bounds",
        ),
        (
            CTLFormula(name="ef", operator="EF", target="transition_fires", params={"transition": ""}),
            "requires a transition",
        ),
        (CTLFormula(name="excl", operator="AG", target="not_comarked", params={"place_a": "a"}), "requires two places"),
        (CTLFormula(name="mark", operator="AG_EF", target="marked", params={"place": ""}), "requires a place"),
        (
            CTLFormula(name="combo", operator="EF", target="marked", params={"place": "sink"}),
            "unsupported CTL formula combination",
        ),
    ],
)
def test_ctl_compilation_rejects_malformed_formulas(formula: CTLFormula, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Z3BoundedModelChecker(_transfer_net()).verify_ctl_specs([formula], max_depth=1)


@requires_z3
def test_ctl_and_ltl_facades_compile_co_marking_recurrence_and_response() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())
    ctl = checker.verify_ctl_specs(
        [
            CTLFormula.ag_not_comarked("exclusive", "source", "sink", threshold=0.5),
            CTLFormula.ag_ef_marked("source_recurs", "source", threshold=0.5),
        ],
        max_depth=2,
    )
    ltl = checker.verify_ltl_specs(
        [LTLFormula.globally_fire_leads_to_marking("move_marks_sink", "move", "sink", threshold=0.5, within=0)],
        max_depth=2,
    )
    assert ctl.checked_specs == ["CTL:exclusive:AG", "CTL:source_recurs:AG_EF"]
    assert ltl.checked_specs == ["LTL:move_marks_sink:G_implies_F"]


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        (
            LTLFormula(name="g", operator="G", target="marking_bounds", params={"bounds": "not-a-dict"}),
            "requires marking bounds",
        ),
        (
            LTLFormula(name="f", operator="F", target="transition_fires", params={"transition": ""}),
            "requires a transition",
        ),
        (
            LTLFormula(
                name="lead",
                operator="G_implies_F",
                target="fire_leads_to_marking",
                params={"trigger_transition": "move"},
            ),
            "requires trigger transition and target place",
        ),
        (
            LTLFormula(name="combo", operator="G", target="transition_fires", params={"transition": "move"}),
            "unsupported LTL formula combination",
        ),
    ],
)
def test_ltl_compilation_rejects_malformed_formulas(formula: LTLFormula, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Z3BoundedModelChecker(_transfer_net()).verify_ltl_specs([formula], max_depth=1)


# ── fire-leads-to-marking deadline beyond depth ───────────────────────


@requires_z3
def test_fire_leads_to_marking_flags_trigger_whose_deadline_exceeds_depth() -> None:
    # within == max_depth pushes the response deadline past the horizon for the
    # first step, exercising the over-the-horizon counterexample branch.
    report = Z3BoundedModelChecker(_transfer_net()).verify_temporal_specs(
        [FireLeadsToMarking("late_response", "move", "sink", threshold=0.5, within=2)],
        max_depth=2,
    )
    assert report.holds is False
    assert report.solver_status == "sat"
    assert report.violations[0].transition == "move"


# ── inhibitor-arc transition relation and weight bounds ───────────────


def test_inhibitor_arc_is_extracted_into_transition_relation() -> None:
    checker = Z3BoundedModelChecker(_inhibitor_net())
    inputs, _outputs, inhibitors = checker._extract_transition_relation()
    move = checker._transition_index["move"]
    guard = checker._place_index["guard"]
    source = checker._place_index["source"]
    assert guard in inhibitors[move]
    assert source in inputs[move]


@requires_z3
def test_inhibitor_constraint_keeps_marking_within_bounds() -> None:
    report = Z3BoundedModelChecker(_inhibitor_net()).prove_marking_bounds(
        {"sink": (0.0, 1.0)},
        max_depth=2,
    )
    assert report.holds is True
    assert report.solver_status == "unsat"


def test_symbiyosys_contract_encodes_inhibitor_threshold() -> None:
    contract = Z3BoundedModelChecker(_inhibitor_net()).build_symbiyosys_contract(max_depth=5)
    assert "fire_0_move" in contract["smt2"]
    # the inhibitor place appears in a strict-less-than guard
    assert "(< m_0_" in contract["smt2"]


def test_symbiyosys_contract_emits_empty_response_window_obligation() -> None:
    # A one-step latency window leaves the final trigger step with no in-horizon
    # response slot, so the contract asserts that the trigger cannot fire there.
    contract = Z3BoundedModelChecker(_latency_chain_net()).build_symbiyosys_contract(
        max_depth=4,
        trigger_transition="trigger",
        response_transition="response",
        max_latency_ns=10.0,
        tick_period_ns=10.0,
        no_stall_window_ns=10.0,
    )
    assert "(assert (not fire_3_trigger))" in contract["smt2"]


def test_weight_bounds_reject_negative_lower_bound() -> None:
    checker = Z3BoundedModelChecker(_transfer_net())
    with pytest.raises(ValueError, match="must be non-negative"):
        checker.prove_marking_bounds({"sink": (0.0, 1.0)}, max_depth=1, weight_bounds={"move": (-1.0, 1.0)})


def test_build_symbiyosys_contract_rejects_unpaired_trigger_and_budget_violations() -> None:
    checker = Z3BoundedModelChecker(_latency_chain_net())
    with pytest.raises(ValueError, match="must be provided together"):
        checker.build_symbiyosys_contract(max_depth=5, trigger_transition="trigger")
    with pytest.raises(ValueError, match="max_latency_ns must be finite and > 0"):
        checker.build_symbiyosys_contract(max_depth=5, max_latency_ns=0.0)
    with pytest.raises(ValueError, match="no_stall_window_ns must be finite and > 0"):
        checker.build_symbiyosys_contract(max_depth=5, no_stall_window_ns=0.0)
    with pytest.raises(ValueError, match="tick_period_ns must be finite and > 0"):
        checker.build_symbiyosys_contract(max_depth=5, tick_period_ns=0.0)
