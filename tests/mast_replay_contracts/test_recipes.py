# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public MAST numerical channel recipe contracts.
"""Exercise real recipe inputs; no mocks or physical-admission claims."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.disruption_channel_recipes import (
    amperes_to_megamperes,
    dbdt_gauss_per_s,
    locked_mode_envelope,
    n_mode_amplitude,
    per_1e19,
    q_at_psi_norm,
    toroidal_harmonic,
    vacuum_toroidal_field,
)

ANGLES = np.arange(12, dtype=np.float64) * np.pi / 6.0


@pytest.mark.parametrize("function", [amperes_to_megamperes, per_1e19])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_unit_conversion_refuses_nonfinite(function: Callable[..., object], bad: float) -> None:
    """Nonfinite measured values cannot be silently scaled into a candidate."""
    with pytest.raises(ValueError, match="must be finite"):
        function(np.array([1.0, bad], dtype=np.float64))


@pytest.mark.parametrize("count", [True, 1.5, "1", None, 0, -1])
def test_harmonic_count_domain(count: object) -> None:
    """The actual harmonic API rejects lossy/noninteger mode identities."""
    function: Callable[..., object] = toroidal_harmonic
    with pytest.raises(ValueError, match="positive integer"):
        function(np.ones((8, 12), dtype=np.float64), ANGLES, count)


@pytest.mark.parametrize("field", ["saddle", "angles"])
def test_harmonic_finite_geometry(field: str) -> None:
    """Nonfinite measurements and geometry both have authored refusals."""
    saddle = np.ones((8, 12), dtype=np.float64)
    angles = ANGLES.copy()
    if field == "saddle":
        saddle[0, 0] = np.nan
    else:
        angles[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        toroidal_harmonic(saddle, angles, 1)


@pytest.mark.parametrize("window", [True, 2.5, "3", None, 0, -1])
def test_envelope_count_domain(window: object) -> None:
    """Refuse fractional, boolean and nonpositive convolution windows."""
    function: Callable[..., object] = locked_mode_envelope
    with pytest.raises(ValueError, match="positive number of samples"):
        function(np.ones((8, 12), dtype=np.float64), ANGLES, window=window)


@pytest.mark.parametrize("time", [np.array([0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, np.nan])])
def test_derivative_clock_domain(time: NDArray[np.float64]) -> None:
    """A derivative requires two finite chronological samples."""
    with pytest.raises(ValueError):
        dbdt_gauss_per_s(np.ones_like(time), time)


def test_derivative_nonfinite_field() -> None:
    """Refuse a nonfinite magnetic measurement before differentiating it."""
    with pytest.raises(ValueError, match="b_tesla must be finite"):
        dbdt_gauss_per_s(np.array([0.0, np.inf]), np.array([0.0, 1.0]))


@pytest.mark.parametrize("grid", [np.array([]), np.array([0.0, 0.0]), np.array([0.0, np.nan])])
def test_flux_knots_domain(grid: NDArray[np.float64]) -> None:
    """Reject empty, duplicate and nonfinite interpolation knots."""
    with pytest.raises(ValueError):
        q_at_psi_norm(np.ones(grid.size, dtype=np.float64), grid)


def test_safety_factor_nonfinite_inputs() -> None:
    """Check nonfinite profile and target domains through the public API."""
    with pytest.raises(ValueError, match="q_profile must be finite"):
        q_at_psi_norm(np.array([1.0, np.inf]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="target must be finite"):
        q_at_psi_norm(np.array([1.0, 2.0]), np.array([0.0, 1.0]), target=np.nan)


def test_interpolation_retains_sorting_single_knots_and_clamping() -> None:
    """Preserve the established interpolation behaviour on valid inputs."""
    profile = np.array([[3.0, 1.0, 2.0]])
    grid = np.array([1.0, 0.0, 0.5])
    assert q_at_psi_norm(profile, grid, target=-1.0)[0] == 1.0
    assert q_at_psi_norm(profile, grid, target=2.0)[0] == 3.0
    assert q_at_psi_norm(np.array([4.0]), np.array([0.5]))[0] == 4.0
    assert q_at_psi_norm(np.empty((0, 2), dtype=np.float64), np.array([0.0, 1.0])).shape == (0,)


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_vacuum_field_radius_domain(radius: float) -> None:
    """Finite positive radius is required before calculating field magnitude."""
    with pytest.raises(ValueError, match="positive and finite"):
        vacuum_toroidal_field(np.array([1.0e6]), radius, n_turns=100)


@pytest.mark.parametrize("turns", [True, 2.5, "100", None, 0, -1])
def test_vacuum_field_turn_count(turns: object) -> None:
    """The field recipe refuses lossy/noninteger machine turn counts."""
    function: Callable[..., object] = vacuum_toroidal_field
    with pytest.raises(ValueError, match="positive integer"):
        function(np.array([1.0e6]), 0.7, n_turns=turns)


def test_valid_readonly_views_retain_input_bytes() -> None:
    """Real public recipes consume read-only strided inputs without mutation."""
    base = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    time = base[::2]
    time.flags.writeable = False
    field = (3.0 * time).copy()
    field.flags.writeable = False
    before = (base.tobytes(), field.tobytes())
    np.testing.assert_allclose(dbdt_gauss_per_s(field, time), 3.0e4)
    assert np.all(np.isfinite(amperes_to_megamperes(field)))
    assert np.all(np.isfinite(per_1e19(field)))
    np.testing.assert_allclose(
        vacuum_toroidal_field(field, 0.7, n_turns=100), 4.0e-7 * np.pi * 100 * field / (2 * np.pi * 0.7)
    )
    assert (base.tobytes(), field.tobytes()) == before
    saddle = np.cos(ANGLES)[None, :].repeat(8, axis=0)
    saddle.flags.writeable = False
    np.testing.assert_allclose(n_mode_amplitude(saddle, ANGLES, 1), 1.0)
    assert locked_mode_envelope(saddle, ANGLES, window=2).shape == (8,)
