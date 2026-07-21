# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent ballooning reference tests
"""Offline tests for :mod:`validation.ballooning_independent_reference`."""

from __future__ import annotations

import numpy as np
import pytest

from validation.ballooning_independent_reference import (
    BallooningReferencePoint,
    ballooning_alpha_crit,
    ballooning_mode_has_node,
    ballooning_reference_curve,
)


def test_low_alpha_is_stable_high_alpha_unstable() -> None:
    # At unit shear a small pressure gradient is ballooning-stable (no node),
    # a large one is unstable (a node appears).
    assert ballooning_mode_has_node(1.0, 0.05) is False
    assert ballooning_mode_has_node(1.0, 1.5) is True


def test_alpha_crit_matches_prototype_at_unit_shear() -> None:
    # Numerically resolved marginal boundary at s=1 is ~0.61 (independent of the
    # production s(1-s/2)/0.6s fit).
    assert ballooning_alpha_crit(1.0) == pytest.approx(0.612, abs=0.02)


def test_alpha_crit_increases_with_shear() -> None:
    low = ballooning_alpha_crit(0.7)
    high = ballooning_alpha_crit(1.5)
    assert high > low


def test_reference_curve_flags_low_shear_second_stability() -> None:
    # s=1.0 is a clean first-stability point; very low shear runs into the
    # alpha_max ceiling (second-stability access), flagged out of regime.
    points = ballooning_reference_curve(np.array([0.2, 1.0]))
    assert all(isinstance(p, BallooningReferencePoint) for p in points)
    by_s = {round(p.s, 2): p for p in points}
    assert by_s[1.0].first_stability_regime is True
    assert by_s[0.2].first_stability_regime is False


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"s": float("nan"), "alpha": 0.1}, "s must be finite"),
        ({"s": 1.0, "alpha": -0.1}, "alpha must be non-negative"),
        ({"s": 1.0, "alpha": 0.1, "n_theta": 10}, "n_theta must be an integer"),
        ({"s": 1.0, "alpha": 0.1, "theta_span": 0.0}, "theta_span must be positive"),
    ],
)
def test_has_node_rejects_invalid(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ballooning_mode_has_node(**kwargs)  # type: ignore[arg-type]


def test_alpha_crit_rejects_nonpositive_shear() -> None:
    with pytest.raises(ValueError, match="s must be positive"):
        ballooning_alpha_crit(0.0)


def test_reference_curve_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="shear_values must be a non-empty 1-D array"):
        ballooning_reference_curve(np.zeros((2, 2)))
