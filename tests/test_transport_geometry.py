# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Radial-Grid Geometry Tests
"""Tests for the toroidal-geometry helpers of the transport radial grid.

Covers the per-cell toroidal volume element, the Martin L-H plasma surface-area
estimate (including the degenerate-geometry floors), and validation/construction
of the canonical normalised radial grid extracted from the integrated transport
solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.transport_geometry import (
    canonical_radial_grid,
    estimate_plasma_surface_area_m2,
    is_canonical_radial_grid,
    rho_volume_element,
)


class TestRhoVolumeElement:
    def test_matches_analytic_formula(self) -> None:
        """The volume element equals the closed-form toroidal shell expression."""
        rho = np.linspace(0.0, 1.0, 8)
        drho = float(rho[1] - rho[0])
        r_min, r_max = 1.5, 2.5
        r0 = (r_min + r_max) / 2.0
        a_minor = (r_max - r_min) / 2.0
        expected = 2.0 * np.pi * r0 * 2.0 * np.pi * rho * a_minor**2 * drho
        out = rho_volume_element(rho, drho, r_min, r_max)
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)

    def test_returns_float64(self) -> None:
        """The result is a float64 array regardless of the input dtype."""
        rho = np.linspace(0.0, 1.0, 4, dtype=np.float32)
        out = rho_volume_element(rho, 1.0 / 3.0, 1.0, 3.0)
        assert out.dtype == np.float64

    def test_scales_linearly_with_rho(self) -> None:
        """The element is proportional to rho, so the edge cell is the largest."""
        rho = np.linspace(0.0, 1.0, 5)
        out = rho_volume_element(rho, float(rho[1] - rho[0]), 1.0, 3.0)
        assert out[0] == 0.0
        assert np.all(np.diff(out) > 0.0)


class TestEstimatePlasmaSurfaceArea:
    def test_known_circular_geometry(self) -> None:
        """A circular (kappa=1) plasma reduces to 2 pi R0 times 2 pi a."""
        r0, a = 2.0, 0.5
        expected = 2.0 * np.pi * r0 * (2.0 * np.pi * np.sqrt((a * a + a * a) / 2.0))
        assert estimate_plasma_surface_area_m2(r0, a, 1.0) == pytest.approx(expected)

    def test_elongation_increases_area(self) -> None:
        """Increasing the elongation strictly increases the estimated area."""
        base = estimate_plasma_surface_area_m2(2.0, 0.5, 1.0)
        elongated = estimate_plasma_surface_area_m2(2.0, 0.5, 1.8)
        assert elongated > base

    def test_degenerate_inputs_are_floored(self) -> None:
        """Non-positive geometry is floored to a tiny positive value, not NaN."""
        area = estimate_plasma_surface_area_m2(0.0, 0.0, 0.0)
        assert np.isfinite(area)
        assert area > 0.0


class TestIsCanonicalRadialGrid:
    def test_valid_grid(self) -> None:
        """A finite, monotone [0, 1] grid with matching spacing is canonical."""
        rho, drho = canonical_radial_grid(6)
        assert is_canonical_radial_grid(rho, 6, drho) is True

    def test_wrong_length(self) -> None:
        """A grid whose length differs from nr is rejected."""
        rho, drho = canonical_radial_grid(6)
        assert is_canonical_radial_grid(rho, 5, drho) is False

    def test_too_few_points(self) -> None:
        """Fewer than two points cannot form a valid grid."""
        assert is_canonical_radial_grid(np.array([0.0]), 1, 1.0) is False

    def test_non_finite_entry(self) -> None:
        """A non-finite grid entry invalidates the grid."""
        rho = np.array([0.0, np.nan, 1.0])
        assert is_canonical_radial_grid(rho, 3, 0.5) is False

    def test_wrong_start(self) -> None:
        """A grid not starting at zero is rejected."""
        rho = np.array([0.1, 0.55, 1.0])
        assert is_canonical_radial_grid(rho, 3, 0.45) is False

    def test_wrong_end(self) -> None:
        """A grid not ending at one is rejected."""
        rho = np.array([0.0, 0.45, 0.9])
        assert is_canonical_radial_grid(rho, 3, 0.45) is False

    def test_non_monotone(self) -> None:
        """A non-strictly-increasing grid is rejected."""
        rho = np.array([0.0, 0.6, 0.5, 1.0])
        assert is_canonical_radial_grid(rho, 4, 0.25) is False

    def test_non_finite_spacing(self) -> None:
        """A non-finite spacing invalidates the grid."""
        rho, _ = canonical_radial_grid(4)
        assert is_canonical_radial_grid(rho, 4, np.nan) is False

    def test_non_positive_spacing(self) -> None:
        """A non-positive spacing invalidates the grid."""
        rho, _ = canonical_radial_grid(4)
        assert is_canonical_radial_grid(rho, 4, -0.1) is False


class TestCanonicalRadialGrid:
    def test_spans_unit_interval(self) -> None:
        """The canonical grid is a uniform linspace over [0, 1]."""
        rho, drho = canonical_radial_grid(5)
        np.testing.assert_allclose(rho, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        assert drho == pytest.approx(0.25)

    def test_is_self_consistent(self) -> None:
        """A freshly built grid validates against its own spacing."""
        rho, drho = canonical_radial_grid(11)
        assert is_canonical_radial_grid(rho, 11, drho) is True

    def test_rejects_too_few_points(self) -> None:
        """Fewer than two points raises ValueError."""
        with pytest.raises(ValueError, match="nr must be at least 2"):
            canonical_radial_grid(1)
