# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Plasma Power Source/Sink Terms Tests
"""Tests for the fusion-reactivity and radiation power kernels.

Covers the D-T Bosch-Hale reactivity heating source and the tungsten line
and bremsstrahlung radiation sinks extracted from the integrated transport
solver, including every piecewise branch and the low-temperature clamps.
"""

from __future__ import annotations

import numpy as np

from scpn_control.core.plasma_power_terms import (
    bosch_hale_dt_reactivity,
    bremsstrahlung_power_density,
    tungsten_radiation_rate,
)


class TestBoschHaleReactivity:
    def test_positive_for_fusion_temperatures(self) -> None:
        """<sigma*v> is positive and finite for T > 0.2 keV."""
        T = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        sv = bosch_hale_dt_reactivity(T)
        assert np.all(sv > 0)
        assert np.all(np.isfinite(sv))

    def test_increases_with_temperature(self) -> None:
        """The NRL fit rises monotonically over 1-50 keV (peak beyond 100 keV)."""
        T = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        sv = bosch_hale_dt_reactivity(T)
        for i in range(1, len(sv)):
            assert sv[i] > sv[i - 1], f"sigma_v not increasing at T={T[i]}"

    def test_clamps_below_0p2_keV(self) -> None:
        """T below 0.2 keV is floored, so no divergence or NaN and a shared value."""
        T_low = np.array([0.0, 0.05, 0.1, 0.19, 0.2])
        sv = bosch_hale_dt_reactivity(T_low)
        assert np.all(np.isfinite(sv))
        assert np.all(sv > 0)
        np.testing.assert_allclose(sv[:4], sv[4], rtol=1e-12)


class TestTungstenRadiationRate:
    def test_all_branches_positive_finite(self) -> None:
        """Every piecewise branch returns a positive, finite coefficient."""
        Te = np.array([0.5, 3.0, 10.0, 30.0])  # <1, [1,5), [5,20), >=20
        Lz = tungsten_radiation_rate(Te)
        assert np.all(Lz > 0)
        assert np.all(np.isfinite(Lz))

    def test_branch_values_match_fit(self) -> None:
        """Each branch reproduces its Pütterich/ADAS power-law value."""
        np.testing.assert_allclose(tungsten_radiation_rate(np.array([0.25])), 5.0e-31 * np.sqrt(0.25))
        np.testing.assert_allclose(tungsten_radiation_rate(np.array([3.0])), 5.0e-31)
        np.testing.assert_allclose(tungsten_radiation_rate(np.array([10.0])), 2.0e-31 * 10.0**0.3)
        np.testing.assert_allclose(tungsten_radiation_rate(np.array([50.0])), 8.0e-31)

    def test_clamps_low_temperature(self) -> None:
        """Te below 0.01 keV is floored before the fit is evaluated."""
        Lz = tungsten_radiation_rate(np.array([0.0, 0.005, 0.01]))
        assert np.all(np.isfinite(Lz))
        np.testing.assert_allclose(Lz[:2], 5.0e-31 * np.sqrt(0.01))


class TestBremsstrahlungPowerDensity:
    def test_scales_with_density_squared(self) -> None:
        """Doubling the density quadruples the bremsstrahlung power (P ∝ ne^2)."""
        ne = np.array([1.0, 2.0])
        p = bremsstrahlung_power_density(ne, np.array([10.0, 10.0]), Z_eff=1.0)
        np.testing.assert_allclose(p[1] / p[0], 4.0, rtol=1e-12)

    def test_formula_matches_reference(self) -> None:
        """The kernel reproduces the NRL Formulary closed form."""
        p = bremsstrahlung_power_density(np.array([5.0]), np.array([4.0]), Z_eff=1.5)
        expected = 5.35e-37 * 1.5 * (5.0e19) ** 2 * np.sqrt(4.0)
        np.testing.assert_allclose(p, expected, rtol=1e-12)

    def test_clamps_low_temperature(self) -> None:
        """Te below 0.01 keV is floored inside the square-root factor."""
        p = bremsstrahlung_power_density(np.array([1.0]), np.array([0.0]), Z_eff=1.0)
        expected = 5.35e-37 * 1.0 * (1.0e19) ** 2 * np.sqrt(0.01)
        np.testing.assert_allclose(p, expected, rtol=1e-12)
