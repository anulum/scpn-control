# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal verification auto-backend contract tests.
"""Formal-verification backend selection contract tests."""

from __future__ import annotations

import pytest

import scpn_control.scpn.formal_verification as formal_verification
from scpn_control.scpn.formal_verification import verify_formal_contracts
from scpn_control.scpn.structure import StochasticPetriNet


def _transfer_net() -> StochasticPetriNet:
    """Return a compiled one-transition net for formal backend tests."""
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def test_auto_backend_records_explicit_state_even_when_z3_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto mode does not label explicit-state proofs as z3 evidence."""

    def z3_available() -> bool:
        return True

    monkeypatch.setattr(formal_verification, "_z3_solver_available", z3_available)

    report = verify_formal_contracts(
        _transfer_net(),
        max_depth=2,
        marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
        backend="auto",
    )

    assert formal_verification._resolve_backend("auto") == "explicit-state"
    assert report.reachability.backend == "explicit-state"
    assert report.safety.backend == "explicit-state"
    assert report.liveness.backend == "explicit-state"
    assert report.temporal.backend == "explicit-state"
