# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime State Sanitization Tests
"""Tests for the runtime numerical-hardening helper.

Covers non-finite replacement with a fallback, the optional lower/upper clamps,
the recovered-count report, and input immutability extracted from the integrated
transport solver.
"""

from __future__ import annotations

import numpy as np

from scpn_control.core.runtime_sanitization import sanitize_with_fallback


class TestSanitizeWithFallback:
    def test_all_finite_no_recovery(self) -> None:
        """A finite profile within bounds is returned unchanged with zero recoveries."""
        arr = np.array([1.0, 2.0, 3.0])
        fb = np.zeros_like(arr)
        out, recovered = sanitize_with_fallback(arr, fb, floor=0.0, ceil=10.0)
        assert recovered == 0
        np.testing.assert_array_equal(out, arr)

    def test_replaces_nonfinite_with_fallback(self) -> None:
        """NaN and inf entries are replaced element-wise from the fallback."""
        arr = np.array([1.0, np.nan, np.inf, -np.inf])
        fb = np.array([9.0, 8.0, 7.0, 6.0])
        out, recovered = sanitize_with_fallback(arr, fb)
        assert recovered == 3
        np.testing.assert_array_equal(out, np.array([1.0, 8.0, 7.0, 6.0]))

    def test_floor_clamps_low_values(self) -> None:
        """A lower bound clamps sub-floor entries up."""
        arr = np.array([-5.0, 0.005, 2.0])
        out, recovered = sanitize_with_fallback(arr, np.zeros_like(arr), floor=0.01)
        assert recovered == 0
        assert np.all(out >= 0.01)
        assert out[2] == 2.0

    def test_ceil_clamps_high_values(self) -> None:
        """An upper bound clamps super-ceiling entries down."""
        arr = np.array([1.0, 500.0, 1e6])
        out, _ = sanitize_with_fallback(arr, np.zeros_like(arr), ceil=1e3)
        assert np.all(out <= 1e3)
        assert out[0] == 1.0

    def test_no_bounds_leaves_finite_values(self) -> None:
        """With no floor/ceil, finite values pass through untouched."""
        arr = np.array([-100.0, 0.0, 1e9])
        out, recovered = sanitize_with_fallback(arr, np.zeros_like(arr))
        assert recovered == 0
        np.testing.assert_array_equal(out, arr)

    def test_replacement_then_clamp(self) -> None:
        """Replacement happens before clamping, so a fallback is bounded too."""
        arr = np.array([np.nan, 2.0])
        fb = np.array([1e6, 2.0])  # fallback itself exceeds the ceiling
        out, recovered = sanitize_with_fallback(arr, fb, floor=0.01, ceil=1e3)
        assert recovered == 1
        assert out[0] == 1e3  # replaced by 1e6 then clamped to the ceiling

    def test_does_not_mutate_input(self) -> None:
        """The input array is copied, not mutated in place."""
        arr = np.array([np.nan, 5.0, 2000.0])
        original = arr.copy()
        sanitize_with_fallback(arr, np.zeros_like(arr), floor=0.0, ceil=1e3)
        np.testing.assert_array_equal(arr, original)
        assert np.isnan(arr[0])
