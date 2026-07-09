# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto Lyapunov exponent validation tests.
"""Focused validation tests for Kuramoto Lyapunov exponent heuristics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from scpn_control.phase.kuramoto import LYAPUNOV_VALUE_FLOOR, lyapunov_exponent


def test_lyapunov_exponent_uses_sample_elapsed_time() -> None:
    """The denominator is one interval for two sampled states."""
    value = lyapunov_exponent([1.0, 0.5], dt=0.25)

    assert value == pytest.approx(math.log(0.5) / 0.25)


def test_lyapunov_exponent_returns_zero_for_short_finite_history() -> None:
    """A single valid sample has no elapsed interval."""
    assert lyapunov_exponent([1.0], dt=0.25) == 0.0


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, math.inf, -math.inf, math.nan, True])
def test_lyapunov_exponent_rejects_invalid_dt(bad_dt: float | bool) -> None:
    """The timestep must be finite, positive, and not boolean."""
    with pytest.raises(ValueError, match="dt must be"):
        lyapunov_exponent([1.0, 0.5], dt=bad_dt)


@pytest.mark.parametrize(
    "bad_history",
    [
        [1.0, math.inf],
        [1.0, -math.inf],
        [1.0, math.nan],
        [1.0, -0.5],
        [1.0, "invalid"],
        [[1.0], [0.5]],
    ],
)
def test_lyapunov_exponent_rejects_invalid_history(bad_history: Sequence[object]) -> None:
    """History samples must be finite, one-dimensional, and non-negative."""
    with pytest.raises(ValueError):
        lyapunov_exponent(bad_history, dt=0.25)  # type: ignore[arg-type]


def test_lyapunov_exponent_uses_named_floor_for_zero_endpoint() -> None:
    """Zero endpoints are floored so the log ratio stays finite."""
    value = lyapunov_exponent([0.0, 1.0e-12], dt=0.5)

    assert value == pytest.approx(math.log(1.0e-12 / LYAPUNOV_VALUE_FLOOR) / 0.5)
