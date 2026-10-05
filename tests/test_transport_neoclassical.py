# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Neoclassical tests
"""Exercise the named transport neoclassical surface through the integrated solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
    _load_gyro_bohm_coefficient,
    calculate_sauter_bootstrap_current_full,
    chang_hinton_chi_profile,
)


class TestNeoclassical:
    """Check neoclassical geometry and profile contracts, including bootstrap-current edge cases."""

    def test_set_neoclassical_stores_params(self, solver: TransportSolver) -> None:
        """set_neoclassical stores parameters for Chang-Hinton model."""
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        assert solver.neoclassical_params is not None
        assert solver.neoclassical_params["R0"] == 6.2
        assert solver.neoclassical_params["a"] == 2.0
        assert solver.neoclassical_params["B0"] == 5.3

    def test_set_neoclassical_rejects_nonphysical_geometry(self, solver: TransportSolver) -> None:
        """Neoclassical transport geometry must stay in the tokamak domain."""
        with pytest.raises(ValueError, match="a must be smaller"):
            solver.set_neoclassical(R0=2.0, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="A_ion"):
            solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3, A_ion=0.0)

        with pytest.raises(ValueError, match="q_edge"):
            solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3, q0=3.0, q_edge=2.0)

    def test_chang_hinton_profile_shape(self) -> None:
        """Chang-Hinton neoclassical chi should match input rho shape."""
        rho = np.linspace(0, 1, 50)
        Ti = 5.0 * (1 - rho**2)
        ne = 8.0 * (1 - rho**2) ** 0.5
        q = 1.0 + 3.0 * rho**2
        chi = chang_hinton_chi_profile(rho, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert chi.shape == (50,)
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0.01)  # floor applied

    def test_chang_hinton_rejects_invalid_profiles(self) -> None:
        """Non-physical Chang-Hinton profiles fail closed instead of being floored."""
        rho = np.linspace(0, 1, 50)
        Ti = np.full_like(rho, 5.0)
        ne = np.full_like(rho, 8.0)
        q = np.full_like(rho, 2.0)

        with pytest.raises(ValueError, match="rho"):
            chang_hinton_chi_profile(rho[::-1], Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="T_i"):
            chang_hinton_chi_profile(rho, np.zeros_like(Ti), ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="q"):
            chang_hinton_chi_profile(rho, Ti, ne, np.zeros_like(q), R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="a must be smaller"):
            chang_hinton_chi_profile(rho, Ti, ne, q, R0=2.0, a=2.0, B0=5.3)

    def test_neoclassical_profiles_require_full_radial_domain(self) -> None:
        """Neoclassical closures require an axis-to-edge normalized radial grid."""
        rho_missing_axis = np.linspace(0.1, 1.0, 50)
        rho_missing_edge = np.linspace(0.0, 0.9, 50)
        Ti = np.full(50, 5.0)
        Te = np.full(50, 5.0)
        ne = np.full(50, 8.0)
        q = np.full(50, 2.0)

        with pytest.raises(ValueError, match="rho must start at 0"):
            chang_hinton_chi_profile(rho_missing_axis, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="rho must end at 1"):
            calculate_sauter_bootstrap_current_full(rho_missing_edge, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

    def test_bootstrap_current_shape(self) -> None:
        """Sauter bootstrap current profile should match rho shape."""
        rho = np.linspace(0, 1, 50)
        Te = 5.0 * (1 - rho**2)
        Ti = 5.0 * (1 - rho**2)
        ne = 8.0 * (1 - rho**2) ** 0.5
        q = 1.0 + 3.0 * rho**2
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert j_bs.shape == (50,)
        assert np.all(np.isfinite(j_bs))
        # Should be zero at the boundary (j_bs[0] and j_bs[-1])
        assert j_bs[0] == 0.0
        assert j_bs[-1] == 0.0

    def test_sauter_bootstrap_rejects_invalid_profiles(self) -> None:
        """Sauter bootstrap current requires positive finite kinetic profiles."""
        rho = np.linspace(0, 1, 50)
        Te = np.full_like(rho, 5.0)
        Ti = np.full_like(rho, 5.0)
        ne = np.full_like(rho, 8.0)
        q = np.full_like(rho, 2.0)

        with pytest.raises(ValueError, match="Te"):
            calculate_sauter_bootstrap_current_full(rho, np.zeros_like(Te), Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="ne"):
            calculate_sauter_bootstrap_current_full(rho, Te, Ti, -ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="Z_eff"):
            calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, Z_eff=0.0)

    def test_chang_hinton_near_axis_floor_for_tiny_inverse_aspect_ratio(self) -> None:
        """Near-axis Chang-Hinton points should use the finite diffusivity floor."""
        rho = np.linspace(0.0, 1.0, 50)
        Ti = np.full_like(rho, 5.0)
        ne = np.full_like(rho, 8.0)
        q = np.full_like(rho, 2.0)

        chi = chang_hinton_chi_profile(rho, Ti, ne, q, R0=1.0e12, a=1.0, B0=5.0)

        assert chi[0] == pytest.approx(0.01)
        assert chi[1] == pytest.approx(0.01)
        assert np.all(chi >= 0.01)

    def test_bootstrap_current_skips_cells_with_near_zero_poloidal_field(self) -> None:
        """Sauter bootstrap current should stay finite and zero when B_pol is negligible."""
        rho = np.linspace(0.0, 1.0, 50)
        Te = 5.0 * (1.0 - rho**2)
        Ti = 5.0 * (1.0 - rho**2)
        ne = 8.0 * (1.0 - rho**2) ** 0.5
        q = 1.0 + 3.0 * rho**2

        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=1.0e-15)

        assert j_bs.shape == rho.shape
        assert np.all(np.isfinite(j_bs))
        assert np.all(j_bs[1:-1] == 0.0)


class TestModuleValidators:
    """Retain direct unit probes for scalar, radial-grid and profile validation."""

    def test_finite_scalar_rejects_nonfinite_and_negative(self) -> None:
        """The legacy scalar probe rejects infinity and a negative nonnegative-domain value."""
        from scpn_control.core.integrated_transport_solver import _finite_scalar

        with pytest.raises(ValueError, match="must be finite"):
            _finite_scalar("x", float("inf"))
        with pytest.raises(ValueError, match="must be non-negative"):
            _finite_scalar("x", -1.0, nonnegative=True)

    def test_normalised_radius_rejects_malformed_grids(self) -> None:
        """The legacy grid probe rejects wrong dimensions, NaNs, invalid bounds and ordering."""
        from scpn_control.core.integrated_transport_solver import _normalised_radius

        with pytest.raises(ValueError, match="one-dimensional"):
            _normalised_radius(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="one-dimensional"):
            _normalised_radius(np.array([0.0]))
        with pytest.raises(ValueError, match="finite"):
            _normalised_radius(np.array([0.0, np.nan, 1.0]))
        with pytest.raises(ValueError, match="normalised interval"):
            _normalised_radius(np.array([0.0, 1.5]))
        with pytest.raises(ValueError, match="strictly increasing"):
            _normalised_radius(np.array([0.0, 0.5, 0.5, 1.0]))

    def test_profile_array_rejects_shape_and_nonfinite(self) -> None:
        """The legacy profile probe rejects mismatched shape and nonfinite entries."""
        from scpn_control.core.integrated_transport_solver import _profile_array

        with pytest.raises(ValueError, match="match the rho grid shape"):
            _profile_array("Te", np.zeros(3), (4,))
        with pytest.raises(ValueError, match="finite values"):
            _profile_array("Te", np.array([np.nan, 1.0]), (2,))


class TestGyroBohmCoefficientLoader:
    """Check nested nominal coefficients and the missing-key default."""

    def test_load_from_scaling_parameters_nominal(self, tmp_path: Path) -> None:
        """The loader recognizes c_gB_nominal inside scaling_parameters."""
        p = tmp_path / "c_gB.json"
        p.write_text(json.dumps({"scaling_parameters": {"c_gB_nominal": 0.27}}), encoding="utf-8")
        assert _load_gyro_bohm_coefficient(p) == pytest.approx(0.27)

    def test_missing_coefficient_key_falls_back_to_default(self, tmp_path: Path) -> None:
        """JSON without a recognized coefficient key uses the 0.1 default."""
        p = tmp_path / "no_key.json"
        p.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
        assert _load_gyro_bohm_coefficient(p) == pytest.approx(0.1)


class TestSauterBootstrapInteriorSkips:
    """Check current suppression in singular geometry, empty cells and unresolved gradients."""

    def test_skips_negligible_inverse_aspect_ratio(self) -> None:
        """The tested vanishing-aspect-ratio regime contributes zero bootstrap current."""
        rho = np.linspace(0.0, 1.0, 20)
        Te = np.linspace(5.0, 0.1, 20)
        Ti = Te.copy()
        ne = np.linspace(8.0, 0.5, 20)
        q = np.linspace(1.0, 4.0, 20)
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=10.0, a=1.0e-6, B0=5.0)
        assert np.allclose(j_bs, 0.0)

    def test_skips_interior_cells_with_zero_density(self) -> None:
        """An empty interior cell contributes no bootstrap current."""
        rho = np.linspace(0.0, 1.0, 20)
        Te = np.linspace(5.0, 0.1, 20)
        Ti = Te.copy()
        ne = np.linspace(8.0, 0.5, 20)
        ne[10] = 0.0
        q = np.linspace(1.0, 4.0, 20)
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert j_bs[10] == 0.0

    def test_skips_cells_with_degenerate_radial_spacing(self) -> None:
        """Spacing below the gradient floor suppresses current in the affected cell."""
        rho = np.array([0.0, 0.5 - 1e-13, 0.5, 0.5 + 1e-13, 1.0])
        Te = np.array([5.0, 4.0, 3.0, 2.0, 0.0])
        Ti = Te.copy()
        ne = np.array([8.0, 6.0, 4.0, 2.0, 0.5])
        q = np.array([1.0, 2.0, 3.0, 3.5, 4.0])
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        # Central cell has |rho[i+1]-rho[i-1]| below the 1e-12 gradient floor.
        assert j_bs[2] == 0.0


class TestChangHintonAdapter:
    """Check adapter defaults and repair of a wrong-length safety-factor profile."""

    def test_defaults_when_neoclassical_absent(self, config_file: Path) -> None:
        """The direct adapter returns the required shape with default parameters."""
        ts = TransportSolver(str(config_file))
        ts.neoclassical_params = None
        chi = ts.chang_hinton_chi_profile()
        assert chi.shape == (ts.nr,)

    def test_rebuilds_mismatched_q_profile(self, config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrong-length safety-factor profile is rebuilt before adapter evaluation."""
        ts = TransportSolver(str(config_file))
        monkeypatch.setattr(ts, "q_profile", np.array([1.0, 2.0]), raising=False)
        chi = ts.chang_hinton_chi_profile()
        assert chi.shape == (ts.nr,)


class TestBootstrapCurrentDispatch:
    """Check configured bootstrap dispatch and the explicit simplified fallback."""

    def test_sauter_path_with_neoclassical(self, solver: TransportSolver) -> None:
        """Configured bootstrap dispatch returns a finite radial current profile."""
        j_bs = solver.calculate_bootstrap_current(6.2, np.full(solver.nr, 0.5))
        assert j_bs.shape == (solver.nr,)
        assert np.all(np.isfinite(j_bs))

    def test_fails_closed_without_neoclassical(self, config_file: Path) -> None:
        """Bootstrap dispatch refuses missing geometry without an explicit fallback."""
        ts = TransportSolver(str(config_file))
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required for bootstrap"):
            ts.calculate_bootstrap_current(6.2, np.full(ts.nr, 0.5))

    def test_legacy_fallback_recomputes_geometry(self, config_file: Path) -> None:
        """The opted-in simplified path returns zero endpoint current on a radial profile."""
        ts = TransportSolver(
            str(config_file),
            allow_simplified_bootstrap_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        j_bs = ts.calculate_bootstrap_current(0.0, np.full(ts.nr, 0.05))
        assert j_bs[0] == 0.0
        assert j_bs[-1] == 0.0
        assert j_bs.shape == (ts.nr,)


@pytest.mark.parametrize("helium", [0.0, 0.5, 2.0])
def test_legacy_bootstrap_uses_species_pressure(config_file: Path, helium: float) -> None:
    """Public legacy current uses actual ion counts for an analytic pressure gradient."""
    ts = TransportSolver(
        config_file, nr=25, multi_ion=True, allow_simplified_bootstrap_fallback=True, allow_legacy_approximations=True
    )
    shape = 1.0 - 0.5 * ts.rho**2
    ts.n_D = 2.0 * shape
    ts.n_T = np.zeros(25)
    ts.n_He = helium * shape
    ts.n_impurity = np.zeros(25)
    ts.ne = (2.0 + 2.0 * helium) * shape
    ts.Ti = np.full(25, 3.0)
    ts.Te = np.ones(25)
    pressure_amplitude = ((2.0 + helium) * 3.0 + 2.0 + 2.0 * helium) * 1e19 * 1.602176634e-16
    pressure_gradient = -pressure_amplitude * ts.rho
    trapped = 1.46 * np.sqrt(ts.rho * 2.0 / 12.0)
    expected = 1.2 * trapped / 0.5 * pressure_gradient / 2.0
    expected[0] = expected[-1] = 0.0
    actual = ts.calculate_bootstrap_current(6.0, np.full(25, 0.5))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
