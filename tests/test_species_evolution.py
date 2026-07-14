# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-Ion Species Evolution Tests
"""Tests for the multi-ion (D/T/He-ash) species evolution kernel.

Covers the fusion helium source, helium pumping, the quasineutral electron
density and effective charge, tungsten line radiation, the CFL sub-stepping
stability, and the edge boundary conditions extracted from the integrated
transport solver.
"""

from __future__ import annotations

import numpy as np

from scpn_control.core.plasma_power_terms import bosch_hale_dt_reactivity
from scpn_control.core.species_evolution import (
    SpeciesEvolutionResult,
    evolve_multi_ion_species,
)


def _state() -> dict[str, object]:
    """Build a well-conditioned multi-ion evolution input state."""
    rho = np.linspace(0.0, 1.0, 21)
    drho = float(rho[1] - rho[0])
    return {
        "n_D": 0.5 * np.ones_like(rho),
        "n_T": 0.5 * np.ones_like(rho),
        "n_He": 0.05 * np.ones_like(rho),
        "Ti": 15.0 * (1.0 - rho**2) + 0.5,
        "Te": 15.0 * (1.0 - rho**2) + 0.5,
        "n_impurity": 0.001 * np.ones_like(rho),
        "dV": 2.0 * np.pi**2 * 3.0 * 1.0**2 * rho * drho,
        "drho": drho,
        "D_species": 0.3,
        "tau_He": 2.5,
        "dt": 0.001,
    }


class TestEvolveMultiIonSpecies:
    def test_returns_finite_result_of_expected_shape(self) -> None:
        """A well-conditioned step returns finite densities and diagnostics."""
        state = _state()
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        assert isinstance(result, SpeciesEvolutionResult)
        shape = np.asarray(state["n_D"]).shape
        for arr in (result.n_D, result.n_T, result.n_He, result.ne, result.S_He, result.P_rad_line):
            assert arr.shape == shape
            assert np.all(np.isfinite(arr))
        assert np.isfinite(result.Z_eff)
        assert np.isfinite(result.particle_balance_error)

    def test_electron_density_follows_quasineutrality(self) -> None:
        """ne = D + T + 2·He + Z_W·impurity, floored at 0.1 (Z_W = 10)."""
        state = _state()
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        expected = result.n_D + result.n_T + 2.0 * result.n_He + 10.0 * np.maximum(np.asarray(state["n_impurity"]), 0.0)
        expected = np.maximum(expected, 0.1)
        np.testing.assert_allclose(result.ne, expected, rtol=1e-12)

    def test_helium_source_is_fusion_rate(self) -> None:
        """S_He equals the D-T fusion reaction rate from the incoming densities."""
        state = _state()
        n_D = np.asarray(state["n_D"])
        n_T = np.asarray(state["n_T"])
        sigmav = bosch_hale_dt_reactivity(np.asarray(state["Ti"]))
        expected_S_He = (n_D * 1e19) * (n_T * 1e19) * sigmav / 1e19
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        np.testing.assert_allclose(result.S_He, expected_S_He, rtol=1e-12)
        assert np.all(result.S_He > 0.0)  # burning plasma produces helium

    def test_z_eff_is_clipped_to_physical_range(self) -> None:
        """Z_eff stays within [1, 10] even for a heavy-impurity state."""
        state = _state()
        state["n_impurity"] = 5.0 * np.ones_like(np.asarray(state["n_D"]))  # heavy tungsten load
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        assert 1.0 <= result.Z_eff <= 10.0

    def test_tungsten_radiation_scales_with_impurity(self) -> None:
        """More tungsten impurity radiates more line power."""
        low = _state()
        high = _state()
        high["n_impurity"] = 10.0 * np.asarray(low["n_impurity"])
        r_low = evolve_multi_ion_species(**low)  # type: ignore[arg-type]
        r_high = evolve_multi_ion_species(**high)  # type: ignore[arg-type]
        assert float(np.sum(r_high.P_rad_line)) > float(np.sum(r_low.P_rad_line))

    def test_helium_pumping_removes_ash_without_fusion(self) -> None:
        """With no fusion (T absent) helium ash is depleted by pumping."""
        state = _state()
        state["n_T"] = np.zeros_like(np.asarray(state["n_D"]))  # no D-T reactions
        state["tau_He"] = 0.5  # strong pumping
        n_He_before = float(np.sum(np.asarray(state["n_He"])))
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        assert np.all(result.S_He == 0.0)  # no fusion -> no helium source
        assert float(np.sum(result.n_He)) < n_He_before

    def test_edge_boundary_conditions(self) -> None:
        """Deuterium/tritium hold an edge recycling floor; helium edge is pumped to zero."""
        state = _state()
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        assert result.n_D[-1] == 0.01
        assert result.n_T[-1] == 0.01
        assert result.n_He[-1] == 0.0

    def test_cfl_substepping_stays_stable_for_large_dt(self) -> None:
        """A large dt triggers many CFL sub-steps yet stays finite and bounded."""
        state = _state()
        state["dt"] = 1.0  # >> dt_CFL, forces hundreds of sub-steps
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        assert np.all(np.isfinite(result.n_D))
        assert np.all(np.isfinite(result.n_He))
        assert np.all(result.n_D >= 0.001)
        assert np.all(result.n_He >= 0.0)

    def test_zero_diffusivity_is_handled(self) -> None:
        """A zero particle diffusivity is floored so the CFL step stays finite."""
        state = _state()
        state["D_species"] = 0.0  # exercises the max(D_species, 1e-10) floor
        result = evolve_multi_ion_species(**state)  # type: ignore[arg-type]
        assert np.all(np.isfinite(result.n_D))
        assert np.all(np.isfinite(result.n_He))
