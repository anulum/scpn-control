# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Transport Hmode Edge
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Transport Solver H-mode & Edge Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for H-mode pedestal paths, neoclassical q-mismatch fallback,
bootstrap current edge cases, and P_aux=0 numerical recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
    calculate_sauter_bootstrap_current_full,
)

MOCK_CONFIG = {
    "reactor_name": "HModeEdge-Test",
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


@pytest.fixture()
def cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    return p


def _make_solver(cfg: Path, *, multi_ion: bool = False) -> TransportSolver:
    ts = TransportSolver(str(cfg), multi_ion=multi_ion)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = 5.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
    return ts


class TestHModeFallback:
    def test_hmode_with_neoclassical_params(self, cfg):
        """H-mode + neoclassical triggers EPED pedestal model."""
        ts = _make_solver(cfg)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(P_aux=50.0)
        # EPED pedestal model applies chi_turb *= 0.05 in transport barrier
        assert np.all(np.isfinite(ts.chi_i))
        assert ts.chi_i.shape == (ts.nr,)

    def test_hmode_without_neoclassical_params(self, cfg):
        """H-mode update without neoclassical parameters fails closed."""
        ts = _make_solver(cfg)
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required"):
            ts.update_transport_model(P_aux=50.0)

    def test_lmode_no_pedestal(self, cfg):
        """Low-power update without neoclassical parameters fails closed."""
        ts = _make_solver(cfg)
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required"):
            ts.update_transport_model(P_aux=0.5)

    def test_hmode_trigger_uses_martin_threshold(self, cfg):
        """High-vs-low auxiliary power splits on Martin L-H threshold logic."""
        ts_low = _make_solver(cfg)
        ts_low.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts_low.update_transport_model(P_aux=0.5)
        edge_low = float(np.mean(ts_low.chi_i[-5:]))

        ts_high = _make_solver(cfg)
        ts_high.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts_high.update_transport_model(P_aux=200.0)
        edge_high = float(np.mean(ts_high.chi_i[-5:]))

        assert edge_high < edge_low


class TestNeoclassicalQMismatch:
    def test_q_profile_wrong_shape(self, cfg):
        """q_profile shape mismatch triggers linspace fallback (line 350)."""
        ts = _make_solver(cfg)
        ts.q_profile = np.array([1.0, 2.0, 3.0])
        chi = ts.chang_hinton_chi_profile()
        assert chi.shape == (ts.nr,)
        assert np.all(np.isfinite(chi))


class TestBootstrapEdgeCases:
    def test_zero_bpol_skip(self):
        """B_pol < 1e-10 causes bootstrap skip (line 210)."""
        nr = 20
        rho = np.linspace(0, 1, nr)
        Te = 5.0 * (1 - rho**2)
        Ti = Te.copy()
        ne = 8.0 * (1 - rho**2) ** 0.5
        # q = 1e6 → eps/q → 0 → B_pol < 1e-10
        q = np.full(nr, 1e6)
        j_bs = calculate_sauter_bootstrap_current_full(
            rho,
            Te,
            Ti,
            ne,
            q,
            R0=6.2,
            a=2.0,
            B0=5.3,
        )
        assert np.all(np.isfinite(j_bs))

    def test_zero_density_skip(self):
        """ne=0 at grid point causes skip (line 166)."""
        nr = 20
        rho = np.linspace(0, 1, nr)
        Te = 5.0 * (1 - rho**2)
        Ti = Te.copy()
        ne = np.zeros(nr)
        q = 1.0 + 2.0 * rho**2
        j_bs = calculate_sauter_bootstrap_current_full(
            rho,
            Te,
            Ti,
            ne,
            q,
            R0=6.2,
            a=2.0,
            B0=5.3,
        )
        assert np.all(j_bs == 0.0)


class TestAuxHeatingZeroPower:
    def test_zero_aux_numerical_recovery(self, cfg):
        """P_aux=0 triggers numerical recovery path (lines 1071-1082)."""
        ts = _make_solver(cfg)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(P_aux=0.0)
        for _ in range(10):
            ts.evolve_profiles(dt=0.01, P_aux=0.0)
        assert np.all(np.isfinite(ts.Ti))
        assert np.all(ts.Ti >= 0.01)

    def test_zero_norm_fallback(self, cfg):
        """Zero norm in aux heating shape triggers fallback (line 771-784)."""
        ts = _make_solver(cfg)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.rho[:] = 0.0
        ts.ne[:] = 0.0
        ts.update_transport_model(P_aux=10.0)
        ts.evolve_profiles(dt=0.001, P_aux=10.0)
        assert np.all(np.isfinite(ts.Ti))


class TestMultiIonSolver:
    def test_multi_ion_evolve(self, cfg):
        ts = _make_solver(cfg, multi_ion=True)
        ts.n_D = 0.5 * ts.ne.copy()
        ts.n_T = 0.5 * ts.ne.copy()
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(P_aux=20.0)
        for _ in range(5):
            ts.evolve_profiles(dt=0.001, P_aux=20.0)
        assert np.all(np.isfinite(ts.Ti))
        assert np.all(np.isfinite(ts.Te))
        assert isinstance(ts.n_He, np.ndarray)
