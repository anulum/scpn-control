# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Model Selection tests
"""Exercise the named transport model selection surface through the integrated solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
    _load_gyro_bohm_coefficient,
    chang_hinton_chi_profile,
)


def _configured_neoclassical_params(solver: TransportSolver) -> dict[str, object]:
    """Return the configured coefficient input after verifying its runtime shape."""
    params = solver.neoclassical_params
    assert isinstance(params, dict)
    return params


class TestImpurityInjection:
    """Check edge impurity deposition and nonnegative impurity densities."""

    def test_inject_impurities_increases_edge(self, solver: TransportSolver) -> None:
        """Injecting impurities should increase the total impurity count."""
        imp_before = np.sum(solver.n_impurity)
        solver.inject_impurities(flux_from_wall_per_sec=1e20, dt=0.01)
        imp_after = np.sum(solver.n_impurity)
        assert imp_after > imp_before

    def test_impurities_non_negative(self, solver: TransportSolver) -> None:
        """Impurity profiles should remain non-negative after injection."""
        solver.inject_impurities(flux_from_wall_per_sec=1e20, dt=0.01)
        assert np.all(solver.n_impurity >= 0)


class TestGyroBohm:
    """Check coefficient loading, geometry requirements and explicit legacy fallback opt-ins."""

    def test_load_gyro_bohm_fallback_missing_file(self) -> None:
        """A missing coefficient file selects the documented 0.1 default."""
        val = _load_gyro_bohm_coefficient("/nonexistent/file.json")
        assert val == pytest.approx(0.1)

    def test_load_gyro_bohm_from_json(self, tmp_path: Path) -> None:
        """A top-level c_gB value is retained rather than replaced by the default."""
        p = tmp_path / "c_gB.json"
        p.write_text(json.dumps({"c_gB": 0.042}))
        val = _load_gyro_bohm_coefficient(str(p))
        assert val == pytest.approx(0.042)

    def test_load_gyro_bohm_bad_json(self, tmp_path: Path) -> None:
        """Malformed JSON selects the documented coefficient fallback."""
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        val = _load_gyro_bohm_coefficient(str(p))
        assert val == pytest.approx(0.1)

    def test_gyro_bohm_chi_with_neoclassical(self, solver: TransportSolver) -> None:
        """Configured geometry yields a finite radial profile above the 0.01 floor."""
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        chi = solver._gyro_bohm_chi()
        assert chi.shape == solver.rho.shape
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0.01)

    def test_gyro_bohm_chi_no_neoclassical(self, solver: TransportSolver) -> None:
        """Missing geometry is refused instead of silently selecting a fallback."""
        solver.neoclassical_params = None
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required for gyro-Bohm"):
            solver._gyro_bohm_chi()

    def test_gyro_bohm_chi_legacy_constant_opt_in(self, solver: TransportSolver) -> None:
        """Both legacy opt-ins admit the documented constant 0.5 diffusivity."""
        solver.neoclassical_params = None
        solver.allow_legacy_approximations = True
        solver.allow_constant_transport_fallback = True
        chi = solver._gyro_bohm_chi()
        assert np.all(chi == pytest.approx(0.5))

    def test_chang_hinton_method_adapter(self, solver: TransportSolver) -> None:
        """The configured adapter returns finite values on the solver grid."""
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        chi = solver.chang_hinton_chi_profile()
        assert chi.shape == solver.rho.shape
        assert np.all(np.isfinite(chi))

    def test_update_transport_neoclassical_path(self, solver: TransportSolver) -> None:
        """A configured public update returns finite positive electron diffusivity."""
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        solver.update_transport_model(50.0)
        assert np.all(np.isfinite(solver.chi_e))
        assert np.all(solver.chi_e > 0)


class TestTransportModelBranches:
    """Check model selection and the global gate for legacy approximation flags."""

    def test_legacy_flags_require_global_gate(self, config_file: Path) -> None:
        """An individual legacy flag cannot bypass the constructor-wide opt-in."""
        with pytest.raises(ValueError, match="allow_legacy_approximations=True"):
            TransportSolver(
                str(config_file),
                allow_constant_transport_fallback=True,
            )

    def test_update_transport_without_neoclassical_fails_closed(self, config_file: Path) -> None:
        """The default transport update refuses missing neoclassical configuration."""
        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required"):
            ts.update_transport_model(50.0)

    def test_gyrokinetic_transport_model(self, config_file: Path) -> None:
        """transport_model='gyrokinetic' exercises GyrokineticTransportModel path."""
        ts = TransportSolver(str(config_file), transport_model="gyrokinetic")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_i))
        assert np.all(ts.chi_i > 0)

    def test_tglf_native_transport_model(self, config_file: Path) -> None:
        """transport_model='tglf_native' exercises TGLFNativeSolver path."""
        ts = TransportSolver(str(config_file), transport_model="tglf_native")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_i))
        assert np.all(ts.chi_i > 0)


class TestExternalGKSolverFallback:
    """Retain isolated coefficient-routing tests for unconverged or invalid output."""

    def test_gk_solver_unconverged_fails_closed_by_default(self, config_file: Path) -> None:
        """External GK unconverged results must fail closed by default."""
        ts = TransportSolver(str(config_file), transport_model="external_gk")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        ts._gk_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._external_gk_transport(_configured_neoclassical_params(ts))

    def test_tglf_native_unconverged_fails_closed_by_default(self, config_file: Path) -> None:
        """Native TGLF unconverged results must fail closed by default."""
        ts = TransportSolver(str(config_file), transport_model="tglf_native")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        ts._tglf_native_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._tglf_native_transport(_configured_neoclassical_params(ts))

    def test_tglf_native_legacy_fallback_opt_in(self, config_file: Path) -> None:
        """Legacy fallback for native TGLF is available only with explicit opt-in."""
        ts = TransportSolver(
            str(config_file),
            transport_model="tglf_native",
            tglf_native_allow_gyrobohm_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        ts._tglf_native_solver = mock_solver

        chi_i = ts._tglf_native_transport(_configured_neoclassical_params(ts))
        assert np.all(np.isfinite(chi_i))
        assert np.all(chi_i >= 0.01)

    def test_gk_solver_invalid_converged_flux_fails_closed(self, config_file: Path) -> None:
        """External GK converged flag with invalid flux values must fail closed."""
        ts = TransportSolver(str(config_file), transport_model="external_gk")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=np.nan, chi_e=1.0, D_e=0.1, converged=True)
        ts._gk_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._external_gk_transport(_configured_neoclassical_params(ts))

    def test_tglf_native_invalid_converged_flux_fails_closed(self, config_file: Path) -> None:
        """Native TGLF converged flag with invalid flux values must fail closed."""
        ts = TransportSolver(str(config_file), transport_model="tglf_native")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=np.inf, chi_e=-1.0, D_e=np.nan, converged=True)
        ts._tglf_native_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._tglf_native_transport(_configured_neoclassical_params(ts))

    def test_tglf_native_invalid_converged_flux_legacy_fallback_opt_in(self, config_file: Path) -> None:
        """Native TGLF legacy fallback can absorb invalid converged flux only by explicit opt-in."""
        ts = TransportSolver(
            str(config_file),
            transport_model="tglf_native",
            tglf_native_allow_gyrobohm_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=np.nan, chi_e=np.nan, D_e=np.nan, converged=True)
        ts._tglf_native_solver = mock_solver

        chi_i = ts._tglf_native_transport(_configured_neoclassical_params(ts))
        assert np.all(np.isfinite(chi_i))
        assert np.all(chi_i >= 0.01)

    def test_gk_solver_lazy_init(self, config_file: Path) -> None:
        """Exercise ITS lines 540-543: lazy init of _gk_solver."""
        ts = TransportSolver(
            str(config_file),
            transport_model="gyrokinetic",
            external_gk_allow_gyrobohm_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        if hasattr(ts, "_gk_solver"):
            delattr(ts, "_gk_solver")

        chi_i = ts._external_gk_transport(_configured_neoclassical_params(ts))
        assert hasattr(ts, "_gk_solver")
        assert np.all(np.isfinite(chi_i))


class TestPedestalBoundary:
    """Check pedestal boundaries and the EPED import-failure path."""

    def test_eped_import_error_fallback(self, config_file: Path) -> None:
        """Exercise ITS lines 811-813: ImportError in EPED -> fallback chi suppression."""
        from unittest.mock import patch

        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        # High P_aux should exceed the Martin-threshold H-mode trigger here.
        # Patch EPED import to fail -> lines 811-813 fallback
        with patch.dict("sys.modules", {"scpn_control.core.eped_pedestal": None}):
            ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_e))
        # Edge chi should be suppressed by 0.1 factor
        edge_mask = ts.rho > 0.9
        assert np.all(ts.chi_e[edge_mask] > 0)

    def test_pedestal_boundary_conditions(self, config_file: Path) -> None:
        """Exercise ITS lines 1379-1386: pedestal boundary conditions in evolve."""
        from scpn_control.core.pedestal import PedestalParams, PedestalProfile

        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)

        ped_params = PedestalParams(f_ped=3.0, f_sep=0.1, x_ped=0.95, delta=0.04)
        ped = PedestalProfile(ped_params)

        ts.evolve_profiles(dt=0.01, P_aux=50.0, ped_ti=ped, ped_te=ped)
        # Pedestal applied from rho >= x_ped - 2*delta = 0.87
        mask = ts.rho >= 0.87
        assert np.all(np.isfinite(ts.Ti[mask]))
        assert np.all(np.isfinite(ts.Te[mask]))


class TestGyroBohmExplicitCoefficient:
    """Check explicit coefficient selection and the diffusivity floor."""

    def test_uses_explicit_c_gb_from_params(self, solver: TransportSolver) -> None:
        """An explicit coefficient produces a radial profile respecting its diffusivity floor."""
        assert solver.neoclassical_params is not None
        solver.neoclassical_params["c_gB"] = 0.2
        chi = solver._gyro_bohm_chi()
        assert chi.shape == (solver.nr,)
        assert np.all(chi >= 0.01)


class TestExternalGKDispatchAndFailure:
    """Retain mocked coefficient routing, separate from real-provider qualification."""

    def _prepared_solver(self, config_file: Path) -> TransportSolver:
        """Prepare the legacy coefficient-adapter test with resolved profiles and explicit geometry."""
        ts = TransportSolver(str(config_file), transport_model="external_gk")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        return ts

    def test_solver_execution_failure_fails_closed(self, config_file: Path) -> None:
        """An injected provider exception becomes the expected coefficient-adapter error."""
        from unittest.mock import MagicMock

        ts = self._prepared_solver(config_file)
        mock_solver = MagicMock()
        mock_solver.run_from_params.side_effect = RuntimeError("boom")
        ts._gk_solver = mock_solver
        with pytest.raises(RuntimeError, match="solver execution failed"):
            ts._external_gk_transport(_configured_neoclassical_params(ts))

    def test_update_transport_model_uses_valid_external_gk_fluxes(self, config_file: Path) -> None:
        """Mocked converged coefficients exercise dispatch only, not real-provider fidelity."""
        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        ts = self._prepared_solver(config_file)
        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=1.0, chi_e=0.8, D_e=0.1, converged=True)
        ts._gk_solver = mock_solver
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_i))
        assert np.all(np.isfinite(ts.chi_e))


class TestConstantTransportFallback:
    """Check finite coefficients from the explicitly enabled constant fallback."""

    def test_update_transport_model_constant_fallback(self, config_file: Path) -> None:
        """The opted-in constant model yields finite electron and ion diffusivity."""
        ts = TransportSolver(
            str(config_file),
            allow_constant_transport_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_e))
        assert np.all(np.isfinite(ts.chi_i))


class TestHModePedestalSuccess:
    """Check high-power routing and positive edge transport after an injected import failure."""

    def test_high_power_triggers_eped_pedestal_branch(self, config_file: Path) -> None:
        """The high-power path retains finite electron temperature and diffusivity."""
        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(500.0)
        assert np.all(np.isfinite(ts.chi_e))
        assert np.all(np.isfinite(ts.Te))

    def test_high_power_eped_import_failure_uses_fallback_suppression(self, config_file: Path) -> None:
        """An injected EPED import failure leaves finite positive edge diffusivity."""
        from unittest.mock import patch

        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        chi_turb_edge_before = ts.rho > 0.9
        with patch.dict("sys.modules", {"scpn_control.core.eped_pedestal": None}):
            ts.update_transport_model(500.0)
        assert np.all(np.isfinite(ts.chi_e))
        assert np.all(ts.chi_e[chi_turb_edge_before] > 0.0)


@pytest.mark.parametrize("temperature_ratio", [0.5, 2.0])
def test_builtin_gyrokinetic_channels_survive_transport_update(config_file: Path, temperature_ratio: float) -> None:
    """Real quasilinear outputs survive public dispatch without legacy channel overwrites."""
    from scpn_control.core.gyrokinetic_transport import GyrokineticTransportModel

    ts = TransportSolver(config_file, nr=25, transport_model="gyrokinetic")
    ts.Ti = 0.5 + 4 * (1 - ts.rho**2)
    ts.Te = temperature_ratio * ts.Ti
    ts.ne = 2 + 3 * (1 - ts.rho**2)
    ts.set_neoclassical(R0=6.0, a=2.0, B0=5.3)
    params = ts.neoclassical_params
    assert params is not None
    profiles = dict(
        R0=6.0,
        a=2.0,
        B0=5.3,
        q=params["q_profile"],
        Te=ts.Te,
        Ti=ts.Ti,
        ne=ts.ne,
        dTe_dr=np.gradient(ts.Te, ts.rho * 2.0),
        dTi_dr=np.gradient(ts.Ti, ts.rho * 2.0),
        dne_dr=np.gradient(ts.ne, ts.rho * 2.0),
        Z_eff=params["Z_eff"],
    )
    expected_i, expected_e, expected_d = GyrokineticTransportModel().evaluate_profile(ts.rho, profiles)
    neo = chang_hinton_chi_profile(
        ts.rho, ts.Ti, ts.ne, params["q_profile"], 6.0, 2.0, 5.3, params["A_ion"], params["Z_eff"]
    )
    assert np.any(np.abs(expected_i - expected_e) > 1e-8)
    ts.update_transport_model(0.0)
    np.testing.assert_allclose(ts.chi_i, expected_i + neo, rtol=1e-12)
    np.testing.assert_allclose(ts.chi_e, expected_e, rtol=1e-12)
    np.testing.assert_allclose(ts.D_n, expected_d, rtol=1e-12)


@pytest.mark.parametrize("temperature_ratio", [0.5, 2.0])
def test_real_native_channels_survive_public_update(config_file: Path, temperature_ratio: float) -> None:
    """A real native solve at rho=.5 agrees with independently specified local gradients."""
    from scpn_control.core.gk_interface import GKLocalParams
    from scpn_control.core.gk_tglf_native import TGLFNativeConfig, TGLFNativeSolver

    ts = TransportSolver(config_file, nr=5, transport_model="tglf_native")
    ts.Ti = 4.5 - 4 * ts.rho**2
    ts.Te = temperature_ratio * ts.Ti
    ts.ne = 5 - 3 * ts.rho**2
    ts.set_neoclassical(R0=6.0, a=4.0, B0=5.3)
    local = GKLocalParams(
        R_L_Ti=6 / 3.5,
        R_L_Te=6 / 3.5,
        R_L_ne=4.5 / 4.25,
        q=1.5,
        s_hat=2 / 3,
        Te_Ti=temperature_ratio,
        epsilon=1 / 3,
        R0=6.0,
        a=4.0,
        B0=5.3,
        n_e=4.25,
        T_i_keV=3.5,
        T_e_keV=3.5 * temperature_ratio,
    )
    expected = TGLFNativeSolver(TGLFNativeConfig(n_ky_ion=12, n_theta=32)).run_from_params(local)
    assert expected.converged
    assert expected.chi_e > 0.01 and expected.D_e > 0.001
    neo = chang_hinton_chi_profile(
        ts.rho,
        ts.Ti,
        ts.ne,
        1 + 2 * ts.rho**2,
        6.0,
        4.0,
        5.3,
        2.0,
        1.5,
    )
    ts.update_transport_model(0.0)
    assert ts.chi_i[2] == pytest.approx(max(expected.chi_i, 0.01) + neo[2], rel=1e-12)
    assert ts.chi_e[2] == pytest.approx(expected.chi_e, rel=1e-12)
    assert ts.D_n[2] == pytest.approx(expected.D_e, rel=1e-12)
    assert not np.allclose(ts.chi_i, ts.chi_e)
    assert not np.allclose(ts.D_n, 0.1 * ts.chi_e)


@pytest.mark.parametrize("coefficient", [0.0, 100.0])
@pytest.mark.parametrize("mode", ["tglf_native", "external_gk"])
@pytest.mark.parametrize("allowed", [False, True])
def test_real_provider_failure_public_routing(
    config_file: Path, tmp_path: Path, mode: str, allowed: bool, coefficient: float
) -> None:
    """Real invalid native output or missing external binary requires explicit cellwise fallback."""
    from scpn_control.core.anomalous_transport import gyro_bohm_chi_profile
    from scpn_control.core.gk_tglf import TGLFSolver
    from scpn_control.core.gk_tglf_native import TGLFNativeConfig, TGLFNativeSolver

    ts = TransportSolver(
        config_file,
        nr=5,
        transport_model=mode,
        allow_legacy_approximations=True,
        tglf_native_allow_gyrobohm_fallback=allowed,
        external_gk_allow_gyrobohm_fallback=allowed,
    )
    ts.Ti = 4.5 - 4 * ts.rho**2
    ts.Te = 0.5 * ts.Ti
    ts.ne = 5 - 3 * ts.rho**2
    ts.set_neoclassical(R0=6.0, a=4.0, B0=5.3)
    params = ts.neoclassical_params
    assert params is not None
    params["c_gB"] = coefficient
    if mode == "tglf_native":
        ts._tglf_native_solver = TGLFNativeSolver(TGLFNativeConfig(alpha_exb=np.nan, n_ky_ion=12, n_theta=32))
        error = "tglf_native returned unconverged"
    else:
        ts._gk_solver = TGLFSolver(binary=str(tmp_path / "absent-tglf"), work_dir=tmp_path / "run")
        error = "external_gk solver execution failed"
    if not allowed:
        before = [channel.copy() for channel in (ts.chi_i, ts.chi_e, ts.D_n)]
        with pytest.raises(RuntimeError, match=error):
            ts.update_transport_model(0.0)
        for channel, original in zip((ts.chi_i, ts.chi_e, ts.D_n), before):
            np.testing.assert_array_equal(channel, original)
        return
    expected = gyro_bohm_chi_profile(ts.rho, ts.Ti, ts.Te, params["q_profile"], 6.0, 4.0, 5.3, 2.0, coefficient)
    expected[0] = 0.01
    if coefficient == 0.0:
        np.testing.assert_array_equal(expected, np.full(5, 0.01))
    else:
        assert np.any(expected[1:] > 0.01)
    neo = chang_hinton_chi_profile(
        ts.rho,
        ts.Ti,
        ts.ne,
        1 + 2 * ts.rho**2,
        6.0,
        4.0,
        5.3,
        2.0,
        1.5,
    )
    ts.update_transport_model(0.0)
    np.testing.assert_allclose(ts.chi_i, expected + neo, rtol=1e-12)
    np.testing.assert_allclose(ts.chi_e, expected, rtol=1e-12)
    expected_d = 0.1 * expected
    expected_d[0] = 0.01
    np.testing.assert_allclose(ts.D_n, expected_d, rtol=1e-12)
