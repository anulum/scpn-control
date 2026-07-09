# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Structure liveness validation tests.
"""Regression tests for Petri-net liveness random-walk admission."""

from __future__ import annotations

from typing import cast

import pytest

from scpn_control.scpn.structure import StochasticPetriNet


def _traffic_light_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("P_red", initial_tokens=1.0)
    net.add_place("P_green", initial_tokens=0.0)
    net.add_place("P_yellow", initial_tokens=0.0)
    net.add_transition("T_r2g", threshold=0.5)
    net.add_transition("T_g2y", threshold=0.5)
    net.add_transition("T_y2r", threshold=0.5)
    net.add_arc("P_red", "T_r2g", weight=1.0)
    net.add_arc("T_r2g", "P_green", weight=1.0)
    net.add_arc("P_green", "T_g2y", weight=1.0)
    net.add_arc("T_g2y", "P_yellow", weight=1.0)
    net.add_arc("P_yellow", "T_y2r", weight=1.0)
    net.add_arc("T_y2r", "P_red", weight=1.0)
    net.compile()
    return net


def _overflowing_live_transition_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("overflow", threshold=1.0)
    net.add_arc("source", "overflow", weight=1.0)
    net.add_arc("overflow", "sink", weight=1.0)
    net.add_arc("overflow", "sink", weight=0.25)
    net.compile()
    return net


def _nonfinite_transition_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("nonfinite", threshold=1.0)
    net.add_arc("source", "nonfinite", weight=1.0)
    net.add_arc("nonfinite", "sink", weight=float("inf"))
    net.compile()
    return net


def test_liveness_reports_valid_marking_walk() -> None:
    result = _traffic_light_net().verify_liveness(n_steps=200, n_trials=500)
    transition_fire_pct = cast(dict[str, float], result["transition_fire_pct"])

    assert result["live"] is True
    assert result["marking_bounds_valid"] is True
    assert result["marking_violation"] is None
    assert result["min_fire_pct"] == pytest.approx(1.0)
    assert set(transition_fire_pct) == {"T_r2g", "T_g2y", "T_y2r"}


def test_liveness_fails_closed_on_out_of_bounds_random_walk() -> None:
    result = _overflowing_live_transition_net().verify_liveness(n_steps=1, n_trials=1)
    transition_fire_pct = cast(dict[str, float], result["transition_fire_pct"])
    marking_violation = cast(dict[str, object], result["marking_violation"])

    assert result["live"] is False
    assert transition_fire_pct == {"overflow": 1.0}
    assert result["marking_bounds_valid"] is False
    assert result["max_marking"] == pytest.approx(1.25)
    assert marking_violation == {
        "reason": "marking_out_of_bounds",
        "trial": 0,
        "step": 0,
        "transition": "overflow",
        "min_marking": 0.0,
        "max_marking": 1.25,
    }


def test_liveness_fails_closed_on_nonfinite_random_walk() -> None:
    result = _nonfinite_transition_net().verify_liveness(n_steps=1, n_trials=1)
    marking_violation = cast(dict[str, object], result["marking_violation"])

    assert result["live"] is False
    assert result["marking_bounds_valid"] is False
    assert marking_violation["reason"] == "non_finite_marking"
    assert marking_violation["transition"] == "nonfinite"


def test_liveness_rejects_empty_campaign() -> None:
    net = _traffic_light_net()

    with pytest.raises(ValueError, match="n_trials must be positive"):
        net.verify_liveness(n_steps=1, n_trials=0)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        net.verify_liveness(n_steps=0, n_trials=1)


def test_liveness_rejects_non_integer_campaign_size() -> None:
    with pytest.raises(TypeError, match="n_steps must be an integer"):
        _traffic_light_net().verify_liveness(n_steps=True, n_trials=1)
