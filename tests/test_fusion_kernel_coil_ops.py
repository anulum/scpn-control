# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Fusion Kernel Coil Operations Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for CoilSet ops: Green's function, mutual inductance,
coil current optimization, free-boundary solve, interp_psi,
and Rust multigrid fallback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.fusion_kernel import CoilSet, FusionKernel


def _write_config(
    path: Path,
    *,
    grid: tuple[int, int] = (16, 16),
    method: str = "sor",
    max_iter: int = 30,
    tol: float = 1e-4,
) -> Path:
    cfg = {
        "reactor_name": "Coil-Test",
        "grid_resolution": list(grid),
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {
            "max_iterations": max_iter,
            "convergence_threshold": tol,
            "relaxation_factor": 0.15,
            "solver_method": method,
        },
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def kernel(tmp_path):
    cfg = _write_config(tmp_path / "cfg.json", grid=(12, 12), max_iter=10)
    return FusionKernel(cfg)


def _make_coilset(n_coils: int = 3, with_limits: bool = False, with_targets: bool = False):
    positions = [(3.0 + i * 0.5, 2.0 * (-1) ** i) for i in range(n_coils)]
    currents = np.ones(n_coils) * 1e4
    turns = [10] * n_coils
    cs = CoilSet(
        positions=positions,
        currents=currents,
        turns=turns,
    )
    if with_limits:
        cs.current_limits = np.ones(n_coils) * 5e4
    if with_targets:
        cs.target_flux_points = np.array([
            [3.5, 0.0],
            [4.0, 0.5],
            [4.5, -0.5],
        ])
    return cs


class TestGreenFunction:
    def test_near_origin_returns_zero(self):
        # denom < 1e-30 only when R_src + R_obs ≈ 0 and Z_src ≈ Z_obs
        val = FusionKernel._green_function(0.0, 0.0, 0.0, 0.0)
        assert val == 0.0

    def test_coincident_finite(self):
        val = FusionKernel._green_function(3.0, 0.0, 3.0, 0.0)
        assert np.isfinite(val)

    def test_positive_for_separated_coils(self):
        val = FusionKernel._green_function(3.0, 0.0, 4.0, 1.0)
        assert np.isfinite(val)
        assert val != 0.0

    def test_symmetry_in_z(self):
        val1 = FusionKernel._green_function(3.0, 0.0, 4.0, 1.0)
        val2 = FusionKernel._green_function(3.0, 0.0, 4.0, -1.0)
        assert abs(val1 - val2) < 1e-12


class TestComputeExternalFlux:
    def test_shape(self, kernel):
        cs = _make_coilset(2)
        psi_ext = kernel._compute_external_flux(cs)
        assert psi_ext.shape == (len(kernel.Z), len(kernel.R))

    def test_nonzero(self, kernel):
        cs = _make_coilset(2)
        psi_ext = kernel._compute_external_flux(cs)
        assert np.any(psi_ext != 0.0)


class TestMutualInductanceMatrix:
    def test_shape(self, kernel):
        cs = _make_coilset(3, with_targets=True)
        obs = cs.target_flux_points
        M = kernel._build_mutual_inductance_matrix(cs, obs)
        assert M.shape == (3, 3)

    def test_finite(self, kernel):
        cs = _make_coilset(3, with_targets=True)
        obs = cs.target_flux_points
        M = kernel._build_mutual_inductance_matrix(cs, obs)
        assert np.all(np.isfinite(M))


class TestOptimizeCoilCurrents:
    def test_returns_currents(self, kernel):
        cs = _make_coilset(3, with_limits=True, with_targets=True)
        target_flux = np.array([0.1, 0.2, 0.15])
        I_opt = kernel.optimize_coil_currents(cs, target_flux)
        assert I_opt.shape == (3,)
        assert np.all(np.isfinite(I_opt))

    def test_no_limits(self, kernel):
        cs = _make_coilset(3, with_targets=True)
        target_flux = np.array([0.1, 0.2, 0.15])
        I_opt = kernel.optimize_coil_currents(cs, target_flux)
        assert I_opt.shape == (3,)

    def test_no_target_points_raises(self, kernel):
        cs = _make_coilset(3)
        with pytest.raises(ValueError, match="target_flux_points"):
            kernel.optimize_coil_currents(cs, np.array([0.1, 0.2, 0.15]))


class TestFreeBoundarySolve:
    def test_basic_free_boundary(self, kernel):
        cs = _make_coilset(2)
        result = kernel.solve_free_boundary(cs, max_outer_iter=3, tol=1e-2)
        assert "outer_iterations" in result
        assert "final_diff" in result
        assert "coil_currents" in result
        assert result["outer_iterations"] <= 3

    def test_with_shape_optimization(self, kernel):
        cs = _make_coilset(3, with_limits=True, with_targets=True)
        result = kernel.solve_free_boundary(
            cs, max_outer_iter=2, tol=1e-2,
            optimize_shape=True, tikhonov_alpha=1e-3,
        )
        assert "outer_iterations" in result
        assert result["coil_currents"].shape == (3,)

    def test_convergence(self, kernel):
        cs = _make_coilset(2)
        result = kernel.solve_free_boundary(cs, max_outer_iter=50, tol=1e10)
        assert result["outer_iterations"] == 1


class TestInterpPsi:
    def test_interior_point(self, kernel):
        kernel.Psi = np.ones_like(kernel.Psi)
        val = kernel._interp_psi(4.0, 0.0)
        assert abs(val - 1.0) < 1e-10

    def test_edge_clamping(self, kernel):
        kernel.Psi = np.random.default_rng(0).normal(size=kernel.Psi.shape)
        val = kernel._interp_psi(kernel.R[0], kernel.Z[0])
        assert np.isfinite(val)


class TestRustMultigridFallback:
    def test_boundary_constraint_fallback(self, tmp_path):
        cfg = _write_config(
            tmp_path / "rust.json", grid=(10, 10), max_iter=5,
            method="rust_multigrid",
        )
        fk = FusionKernel(cfg)
        result = fk._solve_via_rust_multigrid(
            preserve_initial_state=True,
            boundary_flux=np.zeros((len(fk.Z), len(fk.R))),
        )
        assert result["solver_method"] in ("sor", "anderson")

    def test_rust_unavailable_fallback(self, tmp_path):
        cfg = _write_config(
            tmp_path / "rust2.json", grid=(10, 10), max_iter=5,
            method="rust_multigrid",
        )
        fk = FusionKernel(cfg)
        result = fk._solve_via_rust_multigrid()
        assert result["solver_method"] in ("sor", "anderson")


class TestSolveEquilibriumMethods:
    def test_anderson_method(self, tmp_path):
        cfg = _write_config(
            tmp_path / "anderson.json", grid=(10, 10),
            max_iter=15, method="anderson",
        )
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert result["solver_method"] == "anderson"

    def test_preserve_initial_state(self, tmp_path):
        cfg = _write_config(tmp_path / "pres.json", grid=(10, 10), max_iter=5)
        fk = FusionKernel(cfg)
        fk.Psi[5, 5] = 99.0
        result = fk.solve_equilibrium(preserve_initial_state=True)
        assert "psi" in result

    def test_boundary_flux(self, tmp_path):
        cfg = _write_config(tmp_path / "bflux.json", grid=(10, 10), max_iter=5)
        fk = FusionKernel(cfg)
        boundary = np.ones((len(fk.Z), len(fk.R))) * 0.5
        result = fk.solve_equilibrium(boundary_flux=boundary)
        assert "psi" in result

    def test_save_results(self, tmp_path):
        cfg = _write_config(tmp_path / "save.json", grid=(8, 8), max_iter=3)
        fk = FusionKernel(cfg)
        fk.solve_equilibrium()
        out = tmp_path / "state.npz"
        fk.save_results(str(out))
        assert out.exists()
        data = np.load(str(out))
        assert "Psi" in data
