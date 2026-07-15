# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Structure Edge Paths
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Structure module edge path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for structure.py: unknown node arc (130), inhibitor weight (150),
strict validation (208), inhibitor compile guard (244), and verify methods
when called on nets with specific patterns (492, 517, 546)."""

from __future__ import annotations

import pytest

from scpn_control.scpn.structure import StochasticPetriNet


def _compiled_self_loop_net() -> StochasticPetriNet:
    """Return a compiled one-place self-loop net for verification tests."""
    net = StochasticPetriNet()
    net.add_place("P1", initial_tokens=0.5)
    net.add_transition("T1")
    net.add_arc("P1", "T1", weight=0.5)
    net.add_arc("T1", "P1", weight=0.5)
    net.compile()
    return net


class TestArcValidation:
    def test_unknown_source_raises(self) -> None:
        """Unknown source node raises ValueError (line 130)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        with pytest.raises(ValueError, match="Unknown node"):
            net.add_arc("NOSRC", "P1")

    def test_unknown_target_raises(self) -> None:
        """Unknown target node raises ValueError (line 132)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        with pytest.raises(ValueError, match="Unknown node"):
            net.add_arc("P1", "NOTGT")

    def test_inhibitor_zero_weight_raises(self) -> None:
        """Inhibitor arc with weight <= 0 raises (line 150)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        net.add_transition("T1")
        with pytest.raises(ValueError, match="inhibitor arc weight must be > 0"):
            net.add_arc("P1", "T1", weight=-1.0, inhibitor=True)

    def test_inhibitor_on_transition_to_place_raises(self) -> None:
        """Inhibitor arc from Transition→Place raises (line 146-148)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        net.add_transition("T1")
        with pytest.raises(ValueError, match="inhibitor arcs are only supported"):
            net.add_arc("T1", "P1", inhibitor=True)

    def test_negative_transition_threshold_raises(self) -> None:
        """Negative transition thresholds are rejected at definition time."""
        net = StochasticPetriNet()

        with pytest.raises(ValueError, match="threshold must be >= 0"):
            net.add_transition("T1", threshold=-0.1)

    def test_duplicate_transition_name_raises(self) -> None:
        """Transition names cannot collide with existing nodes."""
        net = StochasticPetriNet()
        net.add_transition("T1")

        with pytest.raises(ValueError, match="already exists"):
            net.add_transition("T1")


class TestStrictValidation:
    def test_dead_place_strict_raises(self) -> None:
        """strict_validation with dead place raises ValueError (line 208)."""
        net = StochasticPetriNet()
        net.add_place("P_dead", initial_tokens=0.5)
        net.add_place("P_active", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P_active", "T1")
        net.add_arc("T1", "P_active")
        with pytest.raises(ValueError, match="Topology validation failed"):
            net.compile(strict_validation=True)


class TestInhibitorCompileGuard:
    def test_inhibitor_without_allow_raises(self) -> None:
        """Compiling with inhibitor arcs and allow_inhibitor=False raises (line 231-235)."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1", weight=1.0, inhibitor=True)
        net.add_arc("T1", "P1")
        with pytest.raises(ValueError, match="allow_inhibitor"):
            net.compile(allow_inhibitor=False)

    def test_inhibitor_with_allow_succeeds(self) -> None:
        """Compiling with inhibitor arcs and allow_inhibitor=True works."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1", weight=1.0, inhibitor=True)
        net.add_arc("T1", "P1")
        net.compile(allow_inhibitor=True)
        assert net.is_compiled

    def test_validate_topology_skips_inhibitor_from_positive_weight_sum(self) -> None:
        """A negative-weight inhibitor input arc is excluded from the positive-input-weight sum."""
        net = StochasticPetriNet()
        net.add_place("guard", initial_tokens=1.0)
        net.add_place("tok", initial_tokens=1.0)
        net.add_transition("fire")
        net.add_arc("tok", "fire", weight=1.0)
        net.add_arc("guard", "fire", weight=1.0, inhibitor=True)

        report = net.validate_topology()
        assert report["input_weight_overflow_transitions"] == []

    def test_transition_to_place_inhibitor_invariant_fails_closed(self) -> None:
        """A corrupted Transition->Place inhibitor arc fails closed at compile time."""
        net = StochasticPetriNet()
        net.add_place("P1")
        net.add_transition("T1")
        net.add_arc("T1", "P1")
        net._arcs[0] = ("T1", "P1", -1.0, True)

        with pytest.raises(ValueError, match="Place->Transition"):
            net.compile()


class TestSparseMatrixBuild:
    def test_duplicate_triplets_are_coalesced(self) -> None:
        """Parallel arcs to the same matrix cell are summed during compilation."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1", weight=0.25)
        net.add_arc("P1", "T1", weight=0.25)
        net.add_arc("T1", "P1", weight=0.4)
        net.add_arc("T1", "P1", weight=0.1)

        net.compile()

        assert net.W_in is not None
        assert net.W_out is not None
        assert net.W_in.toarray().item(0) == pytest.approx(0.5)
        assert net.W_out.toarray().item(0) == pytest.approx(0.5)


class TestVerifyBoundednessLiveness:
    def test_verify_uncompiled_raises(self) -> None:
        """verify_boundedness on uncompiled net raises (line 481-482)."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1")
        net.add_arc("T1", "P1")
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_boundedness()

    def test_verify_liveness_uncompiled_raises(self) -> None:
        """verify_liveness on uncompiled net raises (line 536-537)."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1")
        net.add_arc("T1", "P1")
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_liveness()

    def test_verify_boundedness_missing_matrix_fails_closed(self) -> None:
        """verify_boundedness rejects a compiled net with missing matrices."""
        net = _compiled_self_loop_net()
        net.W_in = None

        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_boundedness(n_steps=1, n_trials=1)

    def test_verify_liveness_missing_matrix_fails_closed(self) -> None:
        """verify_liveness rejects a compiled net with missing matrices."""
        net = _compiled_self_loop_net()
        net.W_out = None

        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_liveness(n_steps=1, n_trials=1)


class TestUnseededPlaceCycle:
    def test_strict_validation_reports_unseeded_place_cycle(self) -> None:
        """A token-free place cycle is flagged under strict validation.

        Places P1 and P2 form a cycle through transitions T1 and T2 with zero
        initial tokens, so the cycle can never fire; strict validation lists it as
        an unseeded place cycle and compilation fails closed.
        """
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.0)
        net.add_place("P2", initial_tokens=0.0)
        net.add_transition("T1")
        net.add_transition("T2")
        net.add_arc("P1", "T1")
        net.add_arc("T1", "P2")
        net.add_arc("P2", "T2")
        net.add_arc("T2", "P1")
        with pytest.raises(ValueError, match="unseeded_place_cycles"):
            net.compile(strict_validation=True)


def test_summary_of_uncompiled_net_omits_compiled_matrices() -> None:
    """summary() on a net that has not been compiled reports arcs without the matrix section."""
    net = StochasticPetriNet()
    net.add_place("p", initial_tokens=1.0)
    net.add_transition("t")
    net.add_arc("p", "t", weight=1.0)

    text = net.summary()
    assert "p --(1.000)--> t" in text
    assert "W_in" not in text
