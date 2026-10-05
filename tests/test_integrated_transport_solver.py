# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated Transport Solver Tests
"""Exercise initialization, profile evolution, conservation and stopping criteria."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
)

# ── Minimal config for fast tests ────────────────────────────────────


# ── 1. Initialization ────────────────────────────────────────────────


class TestInitialization:
    """Check constructor guards, profile allocation and optional species state."""

    def test_init_default(self, config_file: Path) -> None:
        """TransportSolver initializes with correct default profile shapes."""
        ts = TransportSolver(str(config_file))
        assert ts.Ti.shape == (50,)
        assert ts.Te.shape == (50,)
        assert ts.ne.shape == (50,)
        assert ts.nr == 50
        assert ts.rho[0] == 0.0
        assert ts.rho[-1] == 1.0

    def test_init_rejects_invalid_grid_count(self, config_file: Path) -> None:
        """Radial grid must have at least two points for finite differencing."""
        with pytest.raises(ValueError, match="nr"):
            TransportSolver(str(config_file), nr=1)

        with pytest.raises(ValueError, match="nr"):
            TransportSolver(str(config_file), nr=cast(int, 2.5))

    def test_init_multi_ion(self, config_file: Path) -> None:
        """multi_ion=True creates D, T, He-ash arrays on the rho grid."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.multi_ion is True
        assert isinstance(ts.n_D, np.ndarray)
        assert isinstance(ts.n_T, np.ndarray)
        assert isinstance(ts.n_He, np.ndarray)
        assert ts.n_D.shape == (50,)
        assert ts.n_T.shape == (50,)
        assert ts.n_He.shape == (50,)

    def test_init_single_ion_no_species(self, config_file: Path) -> None:
        """multi_ion=False leaves species arrays as None."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        assert ts.multi_ion is False
        assert ts.n_D is None
        assert ts.n_T is None
        assert ts.n_He is None

    def test_profiles_shape(self, solver: TransportSolver) -> None:
        """Ti, Te, ne should all be length 50 on the rho grid."""
        assert len(solver.Ti) == 50
        assert len(solver.Te) == 50
        assert len(solver.ne) == 50
        assert len(solver.rho) == 50

    def test_transport_coefficients_initialized(self, solver: TransportSolver) -> None:
        """chi_e, chi_i, D_n should exist with correct length."""
        assert solver.chi_e.shape == (50,)
        assert solver.chi_i.shape == (50,)
        assert solver.D_n.shape == (50,)

    def test_chang_hinton_chi_profile_uses_explicit_t_i_and_n_e(
        self, solver: TransportSolver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """chang_hinton_chi_profile honours explicit t_i / n_e over the Ti / ne fallbacks (branches 541->543, 546->548)."""
        monkeypatch.setattr(solver, "t_i", np.asarray(solver.Ti, dtype=np.float64), raising=False)
        monkeypatch.setattr(solver, "n_e", np.asarray(solver.ne, dtype=np.float64), raising=False)
        chi = solver.chang_hinton_chi_profile()
        assert chi.shape == solver.rho.shape
        assert np.all(np.isfinite(chi))

    def test_tau_he_factor_default(self, config_file: Path) -> None:
        """Default He-ash pumping time factor is 5.0."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.tau_He_factor == 5.0

    def test_d_species_default(self, config_file: Path) -> None:
        """Default particle diffusivity for species transport is 0.3."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.D_species == 0.3


# ── 2. Profile Evolution ─────────────────────────────────────────────


# ── 3. Multi-Ion Species ──────────────────────────────────────────────


# ── 4. Steady State Run ──────────────────────────────────────────────


# ── 5. Neoclassical Transport ─────────────────────────────────────────


# ── 8. Impurity Injection ────────────────────────────────────────────


# ── 9. Gyro-Bohm & Neoclassical Method Adapter ───────────────────────


# ── 10. Zero Aux Heating Overshoot Guard ──────────────────────────────


# ── 11. Transport model branch regressions ───────────────────────────


# ── 11. Module-level scalar/profile validators ───────────────────────


class TestRadialGridRestoration:
    """Check grid repair and refusal of an unusable radial node count."""

    def test_restores_corrupted_same_shape_grid_in_place(self, solver: TransportSolver) -> None:
        """A decreasing grid is restored to uniform axis-to-edge coordinates."""
        solver.rho = np.linspace(1.0, 0.0, solver.nr)  # decreasing, correct shape
        restored = solver._ensure_valid_radial_grid()
        assert restored == 1
        np.testing.assert_allclose(solver.rho, np.linspace(0.0, 1.0, solver.nr))

    def test_rebuilds_wrong_shape_grid(self, solver: TransportSolver) -> None:
        """A wrong-length grid is rebuilt to the configured node count."""
        solver.rho = np.zeros(4)
        restored = solver._ensure_valid_radial_grid()
        assert restored == 1
        assert solver.rho.shape == (solver.nr,)

    def test_raises_when_nr_below_two(self, solver: TransportSolver) -> None:
        """Grid recovery refuses a node count unable to represent axis and edge."""
        solver.nr = 1
        solver.rho = np.array([0.0])
        with pytest.raises(ValueError, match="nr must be at least 2"):
            solver._ensure_valid_radial_grid()

    def test_restores_grid_without_momentum_solver(self, config_file: Path) -> None:
        """Grid restoration skips the momentum-solver sync when none is configured (arc 492->496)."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        assert ts._momentum_solver is None  # no set_neoclassical -> momentum solver never built
        ts.rho[:] = 0.0  # break the canonical grid without changing its shape
        restored = ts._ensure_valid_radial_grid()
        assert restored == 1
        assert ts._momentum_solver is None  # the skipped branch left it untouched
        np.testing.assert_allclose(ts.rho, np.linspace(0.0, 1.0, ts.nr))
