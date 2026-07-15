# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal verification validator and fail-closed paths
"""Validator and fail-closed branches of the formal Petri-net verifier.

Exercises the formula dataclass guards, the backend resolver, the verifier
construction and spec/bound guards, the CTL/LTL compilation guards, and the
inhibitor and defensive firing branches of the reachability engine. The
certificate, bundle, and bundle-artifact schema validators live in
``test_formal_safety_certificate``.
"""

from __future__ import annotations

import builtins
import types
from fractions import Fraction

import pytest

import scpn_control.scpn.formal_verification as fv
from scpn_control.scpn.formal_verification import (
    CTLFormula,
    EventuallyFires,
    FireLeadsToMarking,
    FormalPetriNetVerifier,
    LTLFormula,
    PlaceInvariant,
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


def _inhibitor_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("guard", initial_tokens=1.0)
    net.add_place("token", initial_tokens=1.0)
    net.add_place("out", initial_tokens=0.0)
    net.add_transition("fire", threshold=1.0)
    net.add_arc("token", "fire", weight=1.0)
    net.add_arc("guard", "fire", weight=1.0, inhibitor=True)
    net.add_arc("fire", "out", weight=1.0)
    net.compile(allow_inhibitor=True)
    return net


def _permissive_inhibitor_net() -> StochasticPetriNet:
    """Inhibitor net whose guard place is below threshold, so the arc does not block."""
    net = StochasticPetriNet()
    net.add_place("guard", initial_tokens=0.0)  # below the inhibitor threshold -> permits firing
    net.add_place("token", initial_tokens=1.0)
    net.add_place("out", initial_tokens=0.0)
    net.add_transition("fire", threshold=1.0)
    net.add_arc("token", "fire", weight=1.0)
    net.add_arc("guard", "fire", weight=1.0, inhibitor=True)
    net.add_arc("fire", "out", weight=1.0)
    net.compile(allow_inhibitor=True)
    return net


class TestFormulaDataclassGuards:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": "", "operator": "AG", "target": "marking_bounds", "params": {"a": 1}}, "name must be"),
            (
                {"name": "f", "operator": "XX", "target": "marking_bounds", "params": {"a": 1}},
                "unsupported CTL operator",
            ),
            ({"name": "f", "operator": "AG", "target": "xx", "params": {"a": 1}}, "unsupported CTL target"),
            ({"name": "f", "operator": "AG", "target": "marking_bounds", "params": {}}, "non-empty mapping"),
        ],
    )
    def test_rejects_invalid_ctl(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CTLFormula(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": "", "operator": "G", "target": "marking_bounds", "params": {"a": 1}}, "name must be"),
            (
                {"name": "f", "operator": "XX", "target": "marking_bounds", "params": {"a": 1}},
                "unsupported LTL operator",
            ),
            ({"name": "f", "operator": "G", "target": "marking_bounds", "params": {}}, "non-empty mapping"),
        ],
    )
    def test_rejects_invalid_ltl(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            LTLFormula(**kwargs)

    def test_rejects_unsupported_ltl_target(self):
        with pytest.raises(ValueError, match="unsupported LTL target"):
            LTLFormula(name="f", operator="G", target="xx", params={"a": 1})


# ── backend resolution ────────────────────────────────────────────────


class TestBackendResolution:
    def test_explicit_z3_resolves_when_present(self):
        assert fv._resolve_backend("z3") == "z3"

    def test_rejects_unsupported_backend(self):
        with pytest.raises(ValueError, match="unsupported formal verification backend"):
            fv._resolve_backend("smt-magic")

    def test_explicit_z3_requires_solver(self, monkeypatch):
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "z3":
                raise ModuleNotFoundError("no z3")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(RuntimeError, match="z3-solver"):
            fv._resolve_backend("z3")

    def test_auto_falls_back_to_explicit_state(self, monkeypatch):
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "z3":
                raise ModuleNotFoundError("no z3")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        assert fv._resolve_backend("auto") == "explicit-state"


# ── verifier construction and spec guards ─────────────────────────────


class TestVerifierConstruction:
    def test_rejects_uncompiled_net(self):
        net = StochasticPetriNet()
        net.add_place("p", initial_tokens=1.0)
        net.add_transition("t", threshold=1.0)
        net.add_arc("p", "t", weight=1.0)
        with pytest.raises(RuntimeError, match="compiled before formal verification"):
            FormalPetriNetVerifier(net)

    def test_rejects_compiled_net_without_weight_matrices(self):
        net = _transfer_net()
        net.W_in = None
        with pytest.raises(RuntimeError, match="compiled before formal verification"):
            FormalPetriNetVerifier(net)


class TestSpecAndBoundGuards:
    def test_rejects_empty_marking_bounds(self):
        with pytest.raises(ValueError, match="marking bounds must not be empty"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_marking_bounds({}, max_depth=1)

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_marking_bounds(
                {"sink": (1.0, 0.0)}, max_depth=1
            )

    def test_rejects_unknown_place_in_bounds(self):
        with pytest.raises(ValueError, match="unknown place"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_marking_bounds(
                {"ghost": (0.0, 1.0)}, max_depth=1
            )

    def test_rejects_empty_invariant_list(self):
        with pytest.raises(ValueError, match="place invariants must not be empty"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_place_invariants([], max_depth=1)

    @pytest.mark.parametrize(
        ("invariant", "match"),
        [
            (PlaceInvariant("", {"source": 1.0}), "place invariant name must not be empty"),
            (PlaceInvariant("inv", {}), "must include at least one place weight"),
            (PlaceInvariant("inv", {"source": 0.0, "sink": 0.0}), "must not be identically zero"),
        ],
    )
    def test_rejects_degenerate_invariants(self, invariant, match):
        with pytest.raises(ValueError, match=match):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").prove_place_invariants(
                [invariant], max_depth=1
            )

    def test_rejects_unknown_transition_in_eventually_fires(self):
        with pytest.raises(ValueError, match="unknown transition"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").verify_temporal_specs(
                [EventuallyFires("live", "ghost")], max_depth=2
            )

    def test_eventually_fires_reports_unfired_transition(self):
        net = StochasticPetriNet()
        net.add_place("locked", initial_tokens=0.0)
        net.add_place("done", initial_tokens=0.0)
        net.add_transition("needs_token", threshold=1.0)
        net.add_arc("locked", "needs_token", weight=1.0)
        net.add_arc("needs_token", "done", weight=1.0)
        net.compile()
        report = FormalPetriNetVerifier(net, backend="explicit-state").verify_temporal_specs(
            [EventuallyFires("live", "needs_token")], max_depth=2
        )
        assert report.holds is False
        assert report.violations[0].transition == "needs_token"

    def test_rejects_negative_response_window(self):
        with pytest.raises(ValueError, match="within must be an integer >= 0"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").verify_temporal_specs(
                [FireLeadsToMarking("resp", "move", "sink", within=-1)], max_depth=2
            )

    def test_rejects_unsupported_temporal_spec_type(self):
        bogus = types.SimpleNamespace(name="bogus")
        with pytest.raises(TypeError, match="unsupported temporal specification"):
            FormalPetriNetVerifier(_transfer_net(), backend="explicit-state").verify_temporal_specs(
                [bogus],  # type: ignore[list-item]
                max_depth=1,
            )


class TestCTLLTLCompilation:
    def _verifier(self) -> FormalPetriNetVerifier:
        return FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")

    @pytest.mark.parametrize(
        ("formula", "match"),
        [
            (CTLFormula("f", "AG", "marking_bounds", {"bounds": "x"}), "requires marking bounds"),
            (CTLFormula("f", "EF", "transition_fires", {"transition": ""}), "requires a transition"),
            (CTLFormula("f", "AG", "not_comarked", {"place_a": "a", "place_b": 1}), "requires two places"),
            (CTLFormula("f", "AG_EF", "marked", {"place": 1}), "requires a place"),
            (CTLFormula("f", "EF", "marking_bounds", {"bounds": {}}), "unsupported CTL formula combination"),
        ],
    )
    def test_rejects_invalid_ctl_compilation(self, formula, match):
        with pytest.raises(ValueError, match=match):
            self._verifier().verify_ctl_specs([formula], max_depth=1)

    @pytest.mark.parametrize(
        ("formula", "match"),
        [
            (LTLFormula("f", "G", "marking_bounds", {"bounds": "x"}), "requires marking bounds"),
            (LTLFormula("f", "F", "transition_fires", {"transition": ""}), "requires a transition"),
            (
                LTLFormula(
                    "f", "G_implies_F", "fire_leads_to_marking", {"trigger_transition": "t", "target_place": ""}
                ),
                "requires trigger transition and target place",
            ),
            (LTLFormula("f", "F", "marking_bounds", {"bounds": {}}), "unsupported LTL formula combination"),
        ],
    )
    def test_rejects_invalid_ltl_compilation(self, formula, match):
        with pytest.raises(ValueError, match=match):
            self._verifier().verify_ltl_specs([formula], max_depth=1)


# ── firing relation: inhibitor and defensive negative guard ───────────


class TestFiringRelation:
    def test_inhibitor_arc_blocks_transition(self):
        report = FormalPetriNetVerifier(_inhibitor_net(), backend="explicit-state").analyze_reachability(max_depth=2)
        assert "fire" not in report.fired_transitions

    def test_inhibitor_arc_permits_transition_below_threshold(self):
        """A guard marking below the inhibitor threshold must not block firing (loop-continue branch)."""
        report = FormalPetriNetVerifier(_permissive_inhibitor_net(), backend="explicit-state").analyze_reachability(
            max_depth=2
        )
        assert "fire" in report.fired_transitions
        fired_markings = [state.marking for state in report.reachable_states if state.path == ["fire"]]
        assert fired_markings == [{"guard": 0.0, "token": 0.0, "out": 1.0}]

    def test_defensive_negative_marking_is_rejected(self):
        verifier = FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")
        t_idx = verifier._transition_index["move"]
        verifier._outputs[t_idx] = {verifier._place_index["sink"]: Fraction(-5)}
        assert verifier._fire_if_enabled(verifier._initial, "move") is None

    def test_defensive_reachable_invariant_drift_is_reported(self, monkeypatch):
        verifier = FormalPetriNetVerifier(_transfer_net(), backend="explicit-state")
        monkeypatch.setattr(verifier, "_transition_invariant_delta", lambda transition, weights: Fraction(0))
        report = verifier.prove_place_invariants([PlaceInvariant("sink_only", {"sink": 1.0})], max_depth=2)
        assert report.holds is False
        assert "changes invariant" in report.violations[0].message
