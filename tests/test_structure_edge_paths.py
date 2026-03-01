# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Structure module edge path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for structure.py: unknown node arc (130), inhibitor weight (150),
strict validation (208), inhibitor compile guard (244), and verify methods
when called on nets with specific patterns (492, 517, 546)."""

from __future__ import annotations

import pytest

from scpn_control.scpn.structure import StochasticPetriNet


class TestArcValidation:
    def test_unknown_source_raises(self):
        """Unknown source node raises ValueError (line 130)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        with pytest.raises(ValueError, match="Unknown node"):
            net.add_arc("NOSRC", "P1")

    def test_unknown_target_raises(self):
        """Unknown target node raises ValueError (line 132)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        with pytest.raises(ValueError, match="Unknown node"):
            net.add_arc("P1", "NOTGT")

    def test_inhibitor_zero_weight_raises(self):
        """Inhibitor arc with weight <= 0 raises (line 150)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        net.add_transition("T1")
        with pytest.raises(ValueError, match="inhibitor arc weight must be > 0"):
            net.add_arc("P1", "T1", weight=-1.0, inhibitor=True)

    def test_inhibitor_on_transition_to_place_raises(self):
        """Inhibitor arc from Transition→Place raises (line 146-148)."""
        net = StochasticPetriNet()
        net.add_place("P1")
        net.add_transition("T1")
        with pytest.raises(ValueError, match="inhibitor arcs are only supported"):
            net.add_arc("T1", "P1", inhibitor=True)


class TestStrictValidation:
    def test_dead_place_strict_raises(self):
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
    def test_inhibitor_without_allow_raises(self):
        """Compiling with inhibitor arcs and allow_inhibitor=False raises (line 231-235)."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1", weight=1.0, inhibitor=True)
        net.add_arc("T1", "P1")
        with pytest.raises(ValueError, match="allow_inhibitor"):
            net.compile(allow_inhibitor=False)

    def test_inhibitor_with_allow_succeeds(self):
        """Compiling with inhibitor arcs and allow_inhibitor=True works."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1", weight=1.0, inhibitor=True)
        net.add_arc("T1", "P1")
        net.compile(allow_inhibitor=True)
        assert net.is_compiled


class TestVerifyBoundednessLiveness:
    def test_verify_uncompiled_raises(self):
        """verify_boundedness on uncompiled net raises (line 481-482)."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1")
        net.add_arc("T1", "P1")
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_boundedness()

    def test_verify_liveness_uncompiled_raises(self):
        """verify_liveness on uncompiled net raises (line 536-537)."""
        net = StochasticPetriNet()
        net.add_place("P1", initial_tokens=0.5)
        net.add_transition("T1")
        net.add_arc("P1", "T1")
        net.add_arc("T1", "P1")
        with pytest.raises(RuntimeError, match="compiled"):
            net.verify_liveness()
