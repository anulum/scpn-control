# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the disruption channel derivation recipes
"""Offline tests for :mod:`validation.disruption_channel_recipes`."""

from __future__ import annotations

import numpy as np
import pytest

from validation.disruption_channel_recipes import (
    MU0,
    TESLA_TO_GAUSS,
    amperes_to_megamperes,
    dbdt_gauss_per_s,
    locked_mode_envelope,
    n_mode_amplitude,
    per_1e19,
    q_at_psi_norm,
    toroidal_harmonic,
    vacuum_toroidal_field,
)

_ANGLES = 2.0 * np.pi * np.arange(12, dtype=np.float64) / 12.0


def _pure_mode(amp: np.ndarray, n: int) -> np.ndarray:
    """Build a (n_samples, 12) saddle array carrying a pure toroidal mode n."""
    return np.outer(np.asarray(amp, dtype=np.float64), np.cos(n * _ANGLES))


# --------------------------------------------------------------------------- #
# unit transforms
# --------------------------------------------------------------------------- #
def test_amperes_to_megamperes() -> None:
    assert np.allclose(amperes_to_megamperes(np.array([1.0e6, 2.5e6])), [1.0, 2.5])


def test_per_1e19() -> None:
    assert np.allclose(per_1e19(np.array([3.0e19, 1.0e19])), [3.0, 1.0])


# --------------------------------------------------------------------------- #
# toroidal_harmonic / n_mode_amplitude
# --------------------------------------------------------------------------- #
def test_toroidal_harmonic_recovers_n1_amplitude() -> None:
    amp = np.array([1.0, 2.0, 0.5])
    n1 = n_mode_amplitude(_pure_mode(amp, 1), _ANGLES, 1)
    assert np.allclose(n1, np.abs(amp))
    # a pure n=1 mode carries no n=2 content
    n2 = n_mode_amplitude(_pure_mode(amp, 1), _ANGLES, 2)
    assert np.allclose(n2, 0.0, atol=1e-12)


def test_toroidal_harmonic_recovers_n2_amplitude() -> None:
    amp = np.array([1.5, 0.25])
    assert np.allclose(n_mode_amplitude(_pure_mode(amp, 2), _ANGLES, 2), np.abs(amp))
    assert np.allclose(n_mode_amplitude(_pure_mode(amp, 2), _ANGLES, 1), 0.0, atol=1e-12)


def test_toroidal_harmonic_returns_complex() -> None:
    result = toroidal_harmonic(_pure_mode(np.array([1.0]), 1), _ANGLES, 1)
    assert result.dtype == np.complex128


def test_toroidal_harmonic_rejects_non_2d_saddle() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        toroidal_harmonic(np.ones(12), _ANGLES, 1)


def test_toroidal_harmonic_rejects_angle_mismatch() -> None:
    with pytest.raises(ValueError, match="one angle per coil"):
        toroidal_harmonic(_pure_mode(np.array([1.0]), 1), _ANGLES[:6], 1)


def test_toroidal_harmonic_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        toroidal_harmonic(_pure_mode(np.array([1.0]), 1), _ANGLES, 0)


def test_toroidal_harmonic_rejects_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        toroidal_harmonic(_pure_mode(np.array([1.0]), 1), _ANGLES, 6)


# --------------------------------------------------------------------------- #
# locked_mode_envelope
# --------------------------------------------------------------------------- #
def test_locked_mode_static_component_survives() -> None:
    # A non-rotating n=1 field keeps unit locked amplitude in the interior.
    saddle = _pure_mode(np.ones(60), 1)
    locked = locked_mode_envelope(saddle, _ANGLES, window=12)
    assert np.allclose(locked[20:40], 1.0, atol=1e-9)


def test_locked_mode_rotating_component_averages_out() -> None:
    # A rigidly rotating n=1 field averages to ~0 over a full rotation period.
    t = np.arange(60, dtype=np.float64)
    saddle = np.cos(_ANGLES[np.newaxis, :] - 2.0 * np.pi * t[:, np.newaxis] / 12.0)
    locked = locked_mode_envelope(saddle, _ANGLES, window=12)
    assert np.all(locked[20:40] < 1.0e-6)


def test_locked_mode_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="positive number of samples"):
        locked_mode_envelope(_pure_mode(np.ones(10), 1), _ANGLES, window=0)


def test_locked_mode_rejects_window_over_samples() -> None:
    with pytest.raises(ValueError, match="not exceed"):
        locked_mode_envelope(_pure_mode(np.ones(10), 1), _ANGLES, window=11)


# --------------------------------------------------------------------------- #
# dbdt_gauss_per_s
# --------------------------------------------------------------------------- #
def test_dbdt_linear_ramp() -> None:
    time = np.linspace(0.0, 1.0, 50)
    field = 3.0 * time  # 3 T/s
    assert np.allclose(dbdt_gauss_per_s(field, time), 3.0 * TESLA_TO_GAUSS)


def test_dbdt_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching 1-D arrays"):
        dbdt_gauss_per_s(np.ones(10), np.ones(9))


# --------------------------------------------------------------------------- #
# q_at_psi_norm
# --------------------------------------------------------------------------- #
def test_q_at_psi_norm_two_dimensional() -> None:
    psi = np.array([0.0, 0.5, 1.0])
    profile = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    assert np.allclose(q_at_psi_norm(profile, psi), [2.9, 5.8])


def test_q_at_psi_norm_promotes_single_profile() -> None:
    psi = np.array([0.0, 0.5, 1.0])
    assert np.allclose(q_at_psi_norm(np.array([1.0, 2.0, 3.0]), psi), [2.9])


def test_q_at_psi_norm_sorts_unordered_grid() -> None:
    psi = np.array([1.0, 0.0, 0.5])
    profile = np.array([[3.0, 1.0, 2.0]])
    assert np.allclose(q_at_psi_norm(profile, psi), [2.9])


def test_q_at_psi_norm_rejects_grid_mismatch() -> None:
    with pytest.raises(ValueError, match="one value per profile column"):
        q_at_psi_norm(np.array([[1.0, 2.0, 3.0]]), np.array([0.0, 1.0]))


def test_q_at_psi_norm_rejects_high_dimensional_profile() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        q_at_psi_norm(np.ones((2, 3, 4)), np.array([0.0, 0.5, 1.0]))


# --------------------------------------------------------------------------- #
# vacuum_toroidal_field
# --------------------------------------------------------------------------- #
def test_vacuum_toroidal_field_matches_formula() -> None:
    current = np.array([1.0e6, 2.0e6])
    field = vacuum_toroidal_field(current, 0.7, n_turns=100)
    expected = MU0 * 100.0 * current / (2.0 * np.pi * 0.7)
    assert np.allclose(field, expected)


def test_vacuum_toroidal_field_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError, match="r_geo_m must be positive"):
        vacuum_toroidal_field(np.array([1.0e6]), 0.0, n_turns=100)


def test_vacuum_toroidal_field_rejects_non_positive_turns() -> None:
    with pytest.raises(ValueError, match="n_turns must be positive"):
        vacuum_toroidal_field(np.array([1.0e6]), 0.7, n_turns=0)
