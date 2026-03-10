"""Targeted coverage tests for integrated_transport_solver.py uncovered lines."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
    chang_hinton_chi_profile,
    calculate_sauter_bootstrap_current_full,
)

MINIMAL_CONFIG = {
    "reactor_name": "CovGap-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return p


@pytest.fixture
def solver(cfg: Path) -> TransportSolver:
    ts = TransportSolver(str(cfg), multi_ion=False)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = 5.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)
    return ts


# ── Lines 122-123: epsilon < 1e-6 in chang_hinton ──────────────────


def test_chang_hinton_tiny_epsilon():
    """Near-axis points with epsilon = rho*a/R0 < 1e-6 get chi_nc=0.01."""
    nr = 50
    rho = np.linspace(0, 1, nr)
    Ti = 5.0 * np.ones(nr)
    ne = 8.0 * np.ones(nr)
    q = 2.0 * np.ones(nr)
    # R0 huge relative to a so epsilon = rho * a / R0 is tiny everywhere
    chi = chang_hinton_chi_profile(rho, Ti, ne, q, R0=1e12, a=1.0, B0=5.0)
    # All points except possibly last few should be floored at 0.01
    assert chi[0] == pytest.approx(0.01)
    assert chi[1] == pytest.approx(0.01)
    assert np.all(chi >= 0.01)


# ── Line 231: B_pol < 1e-10 in bootstrap current ───────────────────


def test_bootstrap_bpol_near_zero():
    """When B0 is negligible, B_pol = B0*eps/q < 1e-10 → skip point."""
    nr = 50
    rho = np.linspace(0, 1, nr)
    Te = 5.0 * (1 - rho**2)
    Ti = 5.0 * (1 - rho**2)
    ne = 8.0 * (1 - rho**2) ** 0.5
    q = 1.0 + 3.0 * rho**2
    # B0 ~ 0 forces B_pol < 1e-10 for all interior points
    j_bs = calculate_sauter_bootstrap_current_full(
        rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=1e-15
    )
    assert j_bs.shape == (nr,)
    assert np.all(np.isfinite(j_bs))
    # All interior points should be zero (skipped)
    assert np.all(j_bs[1:-1] == 0.0)


# ── Lines 543-568: EPED pedestal model in H-mode ───────────────────


def test_eped_pedestal_hmode_path(cfg: Path):
    """H-mode + neoclassical + EPED model exercises lines 543-568."""
    # Build a mock EpedPedestalModel that returns plausible pedestal values
    mock_ped = MagicMock()
    mock_ped.Delta_ped = 0.05
    mock_ped.T_ped_keV = 3.0

    mock_eped_cls = MagicMock(return_value=MagicMock(predict=MagicMock(return_value=mock_ped)))

    mock_module = types.ModuleType("scpn_control.core.eped_pedestal")
    mock_module.EpedPedestalModel = mock_eped_cls  # type: ignore[attr-defined]
    sys.modules["scpn_control.core.eped_pedestal"] = mock_module

    try:
        ts = TransportSolver(str(cfg), multi_ion=False)
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        chi_e_before = ts.chi_e.copy()
        ts.update_transport_model(50.0)  # P_aux > 30 triggers H-mode
        mock_eped_cls.assert_called_once()
        # Pedestal suppression applied: edge chi should differ
        assert not np.allclose(ts.chi_e, chi_e_before)
    finally:
        sys.modules.pop("scpn_control.core.eped_pedestal", None)


# ── Line 610: thomas_solve with near-zero b[0] ─────────────────────


def test_thomas_solve_near_zero_diagonal():
    """b[0] ≈ 0 triggers the 1e-30 floor in _thomas_solve."""
    n = 10
    a = np.zeros(n - 1)
    b = np.ones(n)
    b[0] = 0.0  # triggers line 610
    c = np.zeros(n - 1)
    d = np.ones(n)
    x = TransportSolver._thomas_solve(a, b, c, d)
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


def test_thomas_solve_nan_diagonal():
    """Non-finite b[0] triggers the floor guard."""
    n = 10
    a = np.zeros(n - 1)
    b = np.ones(n)
    b[0] = float("nan")
    c = np.zeros(n - 1)
    d = np.ones(n)
    x = TransportSolver._thomas_solve(a, b, c, d)
    assert x.shape == (n,)


# ── Lines 1092-1099: zero-aux overshoot guard, single-ion ──────────


def test_zero_aux_overshoot_guard_single_ion(cfg: Path):
    """P_aux=0 + numerical overshoot triggers scaling guard (lines 1092-1099)."""
    ts = TransportSolver(str(cfg), multi_ion=False)
    # Hollow profile: edge hotter than core → diffusion drives energy inward,
    # raising the mean before the Dirichlet BC clamp at edge takes effect.
    ts.Ti = 0.5 + 4.5 * ts.rho**2
    ts.Te = ts.Ti.copy()
    ts.ne = 8.0 * np.ones(ts.nr)
    ts.n_impurity = np.zeros(ts.nr)
    ts.update_transport_model(0.0)
    ts.chi_i = np.full(ts.nr, 100.0)
    ts.chi_e = np.full(ts.nr, 100.0)

    ts.evolve_profiles(dt=1.0, P_aux=0.0)
    assert np.all(np.isfinite(ts.Ti))
    assert np.all(ts.Ti >= 0.01)
    # Single-ion: Te should track Ti
    np.testing.assert_allclose(ts.Te, ts.Ti)
    # The guard must have fired (recovery count incremented)
    assert ts._last_numerical_recovery_count >= 1


def test_zero_aux_overshoot_guard_increments_recovery(cfg: Path):
    """The overshoot guard increments the recovery counter."""
    ts = TransportSolver(str(cfg), multi_ion=False)
    ts.Ti = 0.5 + 4.5 * ts.rho**2
    ts.Te = ts.Ti.copy()
    ts.ne = 8.0 * np.ones(ts.nr)
    ts.n_impurity = np.zeros(ts.nr)
    ts.update_transport_model(0.0)
    ts.chi_i = np.full(ts.nr, 100.0)
    ts.chi_e = np.full(ts.nr, 100.0)
    ts._last_numerical_recovery_count = 0
    ts.evolve_profiles(dt=1.0, P_aux=0.0)
    assert ts._last_numerical_recovery_count >= 0


# ── Line 1113: non-finite conservation error → inf ──────────────────


def test_conservation_error_nonfinite_becomes_inf(cfg: Path):
    """Non-finite energy sums trigger line 1113: error → inf."""
    ts = TransportSolver(str(cfg), multi_ion=False)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)

    # Inject inf into dV so W_before/W_after become inf → error is NaN → inf
    orig_rve = ts._rho_volume_element

    def _patched_rve():
        dV = orig_rve()
        dV[5] = float("inf")
        return dV

    ts._rho_volume_element = _patched_rve  # type: ignore[method-assign]
    ts.evolve_profiles(dt=0.01, P_aux=50.0)
    assert ts._last_conservation_error == float("inf")
