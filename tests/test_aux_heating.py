# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Auxiliary Heating Source Deposition Tests
"""Tests for the auxiliary-heating source deposition helper.

Covers the fail-soft branches (non-positive/non-finite power, degenerate volume
normalisation), the power-conservation guarantee (reconstructed == target by
construction), and the ion/electron split with fraction clipping extracted from
the integrated transport solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.aux_heating import aux_heating_source_profiles


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (rho, ne, dV) tuple with a physically positive volume element."""
    rho = np.linspace(0.0, 1.0, 21)
    ne = 5.0 * (1.0 - rho**2) + 0.5
    # Toroidal-like volume element: grows with radius, zero on axis.
    dV = 2.0 * np.pi**2 * 3.0 * 1.0**2 * rho * (rho[1] - rho[0])
    return rho, ne, dV


class TestAuxHeatingSourceProfiles:
    def test_zero_power_returns_zero_sources(self) -> None:
        """Zero requested power yields zero sources and a zeroed balance."""
        rho, ne, dV = _grid()
        s_i, s_e, balance = aux_heating_source_profiles(0.0, rho, ne, dV, profile_width=0.1, electron_fraction=0.5)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)
        assert balance["target_total_MW"] == 0.0
        assert balance["reconstructed_total_MW"] == 0.0

    def test_negative_power_clamps_target_to_zero(self) -> None:
        """Negative power is treated as zero (finite branch of the guard)."""
        rho, ne, dV = _grid()
        s_i, s_e, balance = aux_heating_source_profiles(-5.0, rho, ne, dV, profile_width=0.1, electron_fraction=0.5)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)
        assert balance["target_total_MW"] == 0.0

    def test_nonfinite_power_returns_zero_target(self) -> None:
        """A non-finite power records a zero target (non-finite branch of the guard)."""
        rho, ne, dV = _grid()
        s_i, s_e, balance = aux_heating_source_profiles(np.nan, rho, ne, dV, profile_width=0.1, electron_fraction=0.5)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)
        assert balance["target_total_MW"] == 0.0

    def test_degenerate_volume_falls_soft_to_zero(self) -> None:
        """A vanishing volume element exercises both norm fallbacks and returns zeros."""
        rho, ne, _dV = _grid()
        dV = np.zeros_like(rho)  # -> sum(shape*dV)=0 -> ones fallback -> still 0 -> zeros
        s_i, s_e, balance = aux_heating_source_profiles(50.0, rho, ne, dV, profile_width=0.1, electron_fraction=0.5)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)
        # The target is still recorded even though deposition failed soft.
        assert balance["target_total_MW"] == pytest.approx(50.0)
        assert balance["reconstructed_total_MW"] == 0.0

    def test_power_is_conserved_by_construction(self) -> None:
        """Reconstructed power matches the request for an arbitrary positive volume."""
        rho, ne, dV = _grid()
        s_i, s_e, balance = aux_heating_source_profiles(40.0, rho, ne, dV, profile_width=0.1, electron_fraction=0.6)
        assert np.all(np.isfinite(s_i))
        assert np.all(np.isfinite(s_e))
        assert s_i.shape == rho.shape
        assert balance["target_ion_MW"] == pytest.approx(0.4 * 40.0)
        assert balance["target_electron_MW"] == pytest.approx(0.6 * 40.0)
        assert balance["reconstructed_ion_MW"] == pytest.approx(balance["target_ion_MW"], rel=1e-9)
        assert balance["reconstructed_electron_MW"] == pytest.approx(balance["target_electron_MW"], rel=1e-9)
        assert balance["reconstructed_total_MW"] == pytest.approx(40.0, rel=1e-9)

    def test_all_electron_heating_zeroes_ion_source(self) -> None:
        """electron_fraction == 1 puts all power on electrons."""
        rho, ne, dV = _grid()
        s_i, s_e, balance = aux_heating_source_profiles(30.0, rho, ne, dV, profile_width=0.1, electron_fraction=1.0)
        assert np.all(s_i == 0.0)
        assert np.any(s_e > 0.0)
        assert balance["reconstructed_electron_MW"] == pytest.approx(30.0, rel=1e-9)

    def test_all_ion_heating_zeroes_electron_source(self) -> None:
        """electron_fraction == 0 puts all power on ions."""
        rho, ne, dV = _grid()
        s_i, s_e, balance = aux_heating_source_profiles(30.0, rho, ne, dV, profile_width=0.1, electron_fraction=0.0)
        assert np.any(s_i > 0.0)
        assert np.all(s_e == 0.0)
        assert balance["reconstructed_ion_MW"] == pytest.approx(30.0, rel=1e-9)

    def test_electron_fraction_is_clipped(self) -> None:
        """Out-of-range electron fractions are clipped to [0, 1]."""
        rho, ne, dV = _grid()
        s_i_hi, s_e_hi, _ = aux_heating_source_profiles(30.0, rho, ne, dV, profile_width=0.1, electron_fraction=1.5)
        assert np.all(s_i_hi == 0.0)  # clipped to 1.0 -> all electron
        s_i_lo, s_e_lo, _ = aux_heating_source_profiles(30.0, rho, ne, dV, profile_width=0.1, electron_fraction=-0.5)
        assert np.all(s_e_lo == 0.0)  # clipped to 0.0 -> all ion
