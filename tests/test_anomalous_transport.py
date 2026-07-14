# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Anomalous Transport Coefficient Models Tests
"""Tests for the anomalous (turbulent) transport coefficient models.

Covers the gyro-Bohm diffusivity scaling (including its temperature/safety-factor
floors and the non-finite guard) and the shared gyrokinetic per-flux-surface
driver (vacuum/core-edge floors, converged-flux acceptance, fail-closed error
paths, and the explicit gyro-Bohm legacy fallback) extracted from the integrated
transport solver.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_control.core.anomalous_transport import (
    gk_flux_surface_transport,
    gyro_bohm_chi_profile,
)
from scpn_control.core.gk_interface import GKOutput


def _flat_gk_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build a well-conditioned (rho, Te, Ti, ne, params) tuple for the GK driver."""
    rho = np.linspace(0.0, 1.0, 11)
    Te = 5.0 * (1.0 - rho**2) + 0.5
    Ti = 5.0 * (1.0 - rho**2) + 0.5
    ne = 8.0 * (1.0 - rho**2) ** 0.5 + 0.5
    params: dict[str, object] = {
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "q_profile": np.linspace(1.0, 3.0, 11),
    }
    return rho, Te, Ti, ne, params


class TestGyroBohmChiProfile:
    def test_shape_and_floor(self) -> None:
        """A flat finite profile yields a full-length, everywhere-floored profile."""
        rho = np.linspace(0.0, 1.0, 11)
        Ti = np.full_like(rho, 5.0)
        Te = np.full_like(rho, 5.0)
        q = np.full_like(rho, 2.0)
        chi = gyro_bohm_chi_profile(rho, Ti, Te, q, R0=6.2, a=2.0, B0=5.3, A_ion=2.0, c_gB=0.1)
        assert chi.shape == rho.shape
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0.01)

    def test_reference_value(self) -> None:
        """A known unfloored configuration reproduces the analytic gyro-Bohm value."""
        rho = np.linspace(0.0, 1.0, 11)
        Ti = np.full_like(rho, 10.0)
        Te = np.full_like(rho, 10.0)
        q = np.full_like(rho, 1.5)
        chi = gyro_bohm_chi_profile(rho, Ti, Te, q, R0=3.0, a=0.5, B0=1.0, A_ion=2.0, c_gB=0.1)
        assert chi[5] == pytest.approx(6.422086862637514, rel=1e-9)

    def test_field_inverse_square_scaling(self) -> None:
        """chi_gB scales as 1/B0^2 in the unfloored regime (rho_s^2 ∝ 1/B0^2)."""
        rho = np.linspace(0.0, 1.0, 11)
        Ti = np.full_like(rho, 10.0)
        Te = np.full_like(rho, 10.0)
        q = np.full_like(rho, 1.5)
        chi_lo = gyro_bohm_chi_profile(rho, Ti, Te, q, R0=3.0, a=0.5, B0=1.0, A_ion=2.0, c_gB=0.1)
        chi_hi = gyro_bohm_chi_profile(rho, Ti, Te, q, R0=3.0, a=0.5, B0=1.5, A_ion=2.0, c_gB=0.1)
        np.testing.assert_allclose(chi_lo / chi_hi, 2.25, rtol=1e-9)

    def test_temperature_and_q_floors(self) -> None:
        """Sub-floor temperatures and safety factor still produce a finite floored profile."""
        rho = np.linspace(0.0, 1.0, 11)
        Ti = np.full_like(rho, 1e-6)  # below the 0.01 keV floor
        Te = np.full_like(rho, 1e-6)
        q = np.full_like(rho, 0.1)  # below the 0.5 floor
        chi = gyro_bohm_chi_profile(rho, Ti, Te, q, R0=6.2, a=2.0, B0=5.3, A_ion=2.0, c_gB=0.1)
        assert np.all(np.isfinite(chi))
        assert np.all(chi == pytest.approx(0.01))

    def test_nonfinite_chi_is_floored(self) -> None:
        """A non-finite intermediate (infinite Ti) collapses the cell to the 0.01 floor."""
        rho = np.linspace(0.0, 1.0, 11)
        Ti = np.full_like(rho, 10.0)
        Ti[4] = np.inf
        Te = np.full_like(rho, 10.0)
        q = np.full_like(rho, 1.5)
        chi = gyro_bohm_chi_profile(rho, Ti, Te, q, R0=3.0, a=0.5, B0=1.0, A_ion=2.0, c_gB=0.1)
        assert chi[4] == pytest.approx(0.01)
        assert np.all(np.isfinite(chi))


class TestGkFluxSurfaceTransport:
    def _fallback(self, n: int) -> object:
        return lambda: np.full(n, 0.3)

    def test_converged_valid_fluxes_are_used(self) -> None:
        """A converged solver with valid fluxes sets the interior diffusivities."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        solver = MagicMock()
        solver.run_from_params.return_value = GKOutput(chi_i=1.5, chi_e=1.2, D_e=0.3, converged=True)
        chi_i, chi_e, d_n = gk_flux_surface_transport(
            solver=solver,
            rho=rho,
            Te=Te,
            Ti=Ti,
            ne=ne,
            params=params,
            solver_label="tglf_native",
            catch_execution_errors=False,
            allow_gyrobohm_fallback=False,
            gyro_bohm_fallback=self._fallback(len(rho)),
        )
        # rho[0] == 0.0 <= 0.05 -> core-edge floor; interior cells take solver output.
        assert chi_i[0] == pytest.approx(0.01)
        assert chi_e[0] == pytest.approx(0.01)
        assert d_n[0] == pytest.approx(0.01)
        assert chi_i[5] == pytest.approx(1.5)
        assert chi_e[5] == pytest.approx(1.2)
        assert d_n[5] == pytest.approx(0.3)

    def test_solver_output_is_floored(self) -> None:
        """Tiny converged fluxes are floored to the model minima."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        solver = MagicMock()
        solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=True)
        chi_i, chi_e, d_n = gk_flux_surface_transport(
            solver=solver,
            rho=rho,
            Te=Te,
            Ti=Ti,
            ne=ne,
            params=params,
            solver_label="tglf_native",
            catch_execution_errors=False,
            allow_gyrobohm_fallback=False,
            gyro_bohm_fallback=self._fallback(len(rho)),
        )
        assert chi_i[5] == pytest.approx(0.01)
        assert chi_e[5] == pytest.approx(0.01)
        assert d_n[5] == pytest.approx(0.001)

    def test_degenerate_density_cells_are_floored(self) -> None:
        """Non-finite and vanishing density cells are floored without invoking the solver."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        ne = ne.copy()
        ne[5] = np.nan  # non-finite density
        ne[6] = 0.0  # vanishing density
        solver = MagicMock()
        solver.run_from_params.return_value = GKOutput(chi_i=1.5, chi_e=1.2, D_e=0.3, converged=True)
        chi_i, chi_e, d_n = gk_flux_surface_transport(
            solver=solver,
            rho=rho,
            Te=Te,
            Ti=Ti,
            ne=ne,
            params=params,
            solver_label="tglf_native",
            catch_execution_errors=False,
            allow_gyrobohm_fallback=False,
            gyro_bohm_fallback=self._fallback(len(rho)),
        )
        for idx in (5, 6):
            assert chi_i[idx] == pytest.approx(0.01)
            assert chi_e[idx] == pytest.approx(0.01)
            assert d_n[idx] == pytest.approx(0.01)

    def test_subfloor_temperature_zeroes_gradient_drive(self) -> None:
        """Sub-floor Te/Ti cells set R/L_T to zero yet still yield finite output."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        Te = Te.copy()
        Ti = Ti.copy()
        Te[3] = 0.005  # -> Te_keV == 0.01, not > 0.01 -> R_L_Te = 0.0
        Ti[4] = 0.005  # -> Ti_keV == 0.01, not > 0.01 -> R_L_Ti = 0.0
        captured: list[object] = []

        def _run(local_params: object, **_: object) -> GKOutput:
            captured.append(local_params)
            return GKOutput(chi_i=1.0, chi_e=1.0, D_e=0.2, converged=True)

        solver = MagicMock()
        solver.run_from_params.side_effect = _run
        chi_i, _chi_e, _d_n = gk_flux_surface_transport(
            solver=solver,
            rho=rho,
            Te=Te,
            Ti=Ti,
            ne=ne,
            params=params,
            solver_label="tglf_native",
            catch_execution_errors=False,
            allow_gyrobohm_fallback=False,
            gyro_bohm_fallback=self._fallback(len(rho)),
        )
        assert np.all(np.isfinite(chi_i))
        # The cell with sub-floor Te has zero electron gradient drive.
        te_params = [p for p in captured if getattr(p, "T_e_keV", None) == pytest.approx(0.01)]
        assert te_params and all(p.R_L_Te == pytest.approx(0.0) for p in te_params)
        ti_params = [p for p in captured if getattr(p, "T_i_keV", None) == pytest.approx(0.01)]
        assert ti_params and all(p.R_L_Ti == pytest.approx(0.0) for p in ti_params)

    def test_unconverged_fails_closed_without_fallback(self) -> None:
        """An unconverged result raises when the gyro-Bohm fallback is not enabled."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        solver = MagicMock()
        solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        with pytest.raises(RuntimeError, match="tglf_native returned unconverged transport"):
            gk_flux_surface_transport(
                solver=solver,
                rho=rho,
                Te=Te,
                Ti=Ti,
                ne=ne,
                params=params,
                solver_label="tglf_native",
                catch_execution_errors=False,
                allow_gyrobohm_fallback=False,
                gyro_bohm_fallback=self._fallback(len(rho)),
            )

    def test_invalid_flux_fails_closed_without_fallback(self) -> None:
        """A converged result carrying non-finite fluxes still fails closed."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        solver = MagicMock()
        solver.run_from_params.return_value = GKOutput(chi_i=np.nan, chi_e=1.0, D_e=0.1, converged=True)
        with pytest.raises(RuntimeError, match="tglf_native returned unconverged transport"):
            gk_flux_surface_transport(
                solver=solver,
                rho=rho,
                Te=Te,
                Ti=Ti,
                ne=ne,
                params=params,
                solver_label="tglf_native",
                catch_execution_errors=False,
                allow_gyrobohm_fallback=False,
                gyro_bohm_fallback=self._fallback(len(rho)),
            )

    def test_unconverged_uses_gyrobohm_fallback_when_enabled(self) -> None:
        """With the fallback enabled, unconverged cells take the gyro-Bohm estimate."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        solver = MagicMock()
        solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        chi_i, chi_e, d_n = gk_flux_surface_transport(
            solver=solver,
            rho=rho,
            Te=Te,
            Ti=Ti,
            ne=ne,
            params=params,
            solver_label="tglf_native",
            catch_execution_errors=False,
            allow_gyrobohm_fallback=True,
            gyro_bohm_fallback=self._fallback(len(rho)),
        )
        # Fallback profile is a constant 0.3 -> chi_i = 0.3, chi_e = 0.3, D = 0.03.
        assert chi_i[5] == pytest.approx(0.3)
        assert chi_e[5] == pytest.approx(0.3)
        assert d_n[5] == pytest.approx(0.03)

    def test_execution_error_fails_closed_without_fallback(self) -> None:
        """A raised solver exception fails closed with the external execution message."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        solver = MagicMock()
        solver.run_from_params.side_effect = RuntimeError("solver crashed")
        with pytest.raises(RuntimeError, match="external_gk solver execution failed"):
            gk_flux_surface_transport(
                solver=solver,
                rho=rho,
                Te=Te,
                Ti=Ti,
                ne=ne,
                params=params,
                solver_label="external_gk",
                catch_execution_errors=True,
                allow_gyrobohm_fallback=False,
                gyro_bohm_fallback=self._fallback(len(rho)),
            )

    def test_execution_error_uses_fallback_when_enabled(self) -> None:
        """A caught solver exception falls back to gyro-Bohm when the opt-in is set."""
        rho, Te, Ti, ne, params = _flat_gk_inputs()
        solver = MagicMock()
        solver.run_from_params.side_effect = RuntimeError("solver crashed")
        chi_i, chi_e, d_n = gk_flux_surface_transport(
            solver=solver,
            rho=rho,
            Te=Te,
            Ti=Ti,
            ne=ne,
            params=params,
            solver_label="external_gk",
            catch_execution_errors=True,
            allow_gyrobohm_fallback=True,
            gyro_bohm_fallback=self._fallback(len(rho)),
        )
        assert chi_i[5] == pytest.approx(0.3)
        assert chi_e[5] == pytest.approx(0.3)
        assert d_n[5] == pytest.approx(0.03)
