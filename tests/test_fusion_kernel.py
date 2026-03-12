# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Fusion Kernel Test Suite
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.fusion_kernel import CoilSet, FusionKernel, _select_x_point_index


# ── helpers ──────────────────────────────────────────────────────────


def _write_config(
    path: Path,
    *,
    grid: tuple[int, int] = (16, 16),
    method: str = "sor",
    max_iter: int = 30,
    tol: float = 1e-4,
    profile_mode: str | None = None,
    fail_on_diverge: bool = False,
    require_gs_residual: bool = False,
    gs_residual_threshold: float | None = None,
    extra_solver: dict | None = None,
    free_boundary: dict | None = None,
) -> Path:
    cfg: dict = {
        "reactor_name": "Test-Reactor",
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
            "fail_on_diverge": fail_on_diverge,
            "require_gs_residual": require_gs_residual,
        },
    }
    if gs_residual_threshold is not None:
        cfg["solver"]["gs_residual_threshold"] = gs_residual_threshold
    if profile_mode:
        cfg["physics"]["profiles"] = {"mode": profile_mode}
    if extra_solver:
        cfg["solver"].update(extra_solver)
    if free_boundary:
        cfg["free_boundary"] = free_boundary
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def kernel(tmp_path):
    cfg = _write_config(tmp_path / "cfg.json")
    return FusionKernel(cfg)


@pytest.fixture
def small_kernel(tmp_path):
    cfg = _write_config(tmp_path / "cfg_small.json", grid=(8, 8), max_iter=5)
    return FusionKernel(cfg)


# ── construction & grid ──────────────────────────────────────────────


class TestConstruction:
    def test_grid_shape(self, kernel):
        assert kernel.Psi.shape == (16, 16)
        assert kernel.J_phi.shape == (16, 16)
        assert kernel.R.shape == (16,)
        assert kernel.Z.shape == (16,)

    def test_grid_spacing_positive(self, kernel):
        assert kernel.dR > 0
        assert kernel.dZ > 0

    def test_rr_zz_meshgrid(self, kernel):
        assert kernel.RR.shape == (16, 16)
        assert kernel.ZZ.shape == (16, 16)
        assert float(kernel.RR[0, 0]) == pytest.approx(2.0)
        assert float(kernel.ZZ[0, 0]) == pytest.approx(-3.0)

    def test_initial_psi_zero(self, kernel):
        assert np.all(kernel.Psi == 0.0)

    def test_config_loaded(self, kernel):
        assert kernel.cfg["reactor_name"] == "Test-Reactor"

    def test_invalid_config_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(KeyError):
            FusionKernel(path)

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FusionKernel(tmp_path / "nonexistent.json")

    def test_default_boundary_variant_is_fixed(self, kernel):
        assert kernel.boundary_variant == "fixed_boundary"

    def test_invalid_boundary_variant_raises(self, tmp_path):
        cfg = _write_config(tmp_path / "bad_variant.json", extra_solver={"boundary_variant": "chaos"})
        with pytest.raises(ValueError, match="boundary_variant"):
            FusionKernel(cfg)


class TestBoundaryVariants:
    def test_build_coilset_from_config(self, tmp_path):
        cfg = _write_config(
            tmp_path / "free.json",
            extra_solver={"boundary_variant": "free_boundary"},
            free_boundary={
                "current_limits": [5.0e4, 5.0e4],
                "target_flux_points": [[3.5, 0.0], [4.0, 0.5]],
                "target_flux_values": [0.1, 0.2],
                "x_point_target": [4.2, -1.4],
                "x_point_flux_target": 0.15,
                "x_point_weight": 2.0,
                "x_point_null_weight": 3.0,
                "divertor_strike_points": [[3.1, -2.6], [4.9, -2.6]],
                "divertor_flux_value": 0.15,
                "divertor_weight": 0.75,
            },
        )
        kernel = FusionKernel(cfg)
        coils = kernel.build_coilset_from_config()

        assert coils.positions == [(3.0, 4.0), (5.0, -4.0)]
        np.testing.assert_allclose(coils.currents, [2.0, -1.0])
        np.testing.assert_allclose(coils.current_limits, [5.0e4, 5.0e4])
        assert coils.target_flux_points is not None
        assert coils.target_flux_points.shape == (2, 2)
        assert coils.target_flux_values is not None
        np.testing.assert_allclose(coils.target_flux_values, [0.1, 0.2])
        assert coils.x_point_target is not None
        np.testing.assert_allclose(coils.x_point_target, [4.2, -1.4])
        assert coils.x_point_flux_target == pytest.approx(0.15)
        assert coils.x_point_weight == pytest.approx(2.0)
        assert coils.x_point_null_weight == pytest.approx(3.0)
        assert coils.divertor_strike_points is not None
        assert coils.divertor_strike_points.shape == (2, 2)
        assert coils.divertor_flux_values is not None
        np.testing.assert_allclose(coils.divertor_flux_values, [0.15, 0.15])
        assert coils.divertor_weight == pytest.approx(0.75)

    def test_build_coilset_from_config_scalar_target_flux(self, tmp_path):
        cfg = _write_config(
            tmp_path / "free_scalar.json",
            extra_solver={"boundary_variant": "free_boundary"},
            free_boundary={
                "target_flux_points": [[3.5, 0.0], [4.0, 0.5]],
                "target_flux_value": 0.125,
            },
        )
        kernel = FusionKernel(cfg)
        coils = kernel.build_coilset_from_config()

        assert coils.target_flux_values is not None
        np.testing.assert_allclose(coils.target_flux_values, [0.125, 0.125])

    def test_build_coilset_from_config_scalar_divertor_flux(self, tmp_path):
        cfg = _write_config(
            tmp_path / "free_divertor_scalar.json",
            extra_solver={"boundary_variant": "free_boundary"},
            free_boundary={
                "divertor_strike_points": [[3.25, -2.5], [4.75, -2.5]],
                "divertor_flux_value": 0.2,
            },
        )
        kernel = FusionKernel(cfg)
        coils = kernel.build_coilset_from_config()

        assert coils.divertor_flux_values is not None
        np.testing.assert_allclose(coils.divertor_flux_values, [0.2, 0.2])

    def test_free_boundary_objective_tolerances_from_config(self, tmp_path):
        cfg = _write_config(
            tmp_path / "free_objective_tolerances.json",
            extra_solver={"boundary_variant": "free_boundary"},
            free_boundary={
                "x_point_target": [4.2, -1.4],
                "objective_tolerances": {"x_point_position": 0.25},
            },
        )
        kernel = FusionKernel(cfg)
        coils = kernel.build_coilset_from_config()
        result = kernel.solve_free_boundary(
            coils,
            max_outer_iter=1,
            tol=1e10,
            optimize_shape=True,
        )

        assert result["objective_tolerances"]["x_point_position"] == pytest.approx(0.25)
        assert result["objective_convergence_active"] is True

    def test_solve_dispatch_fixed(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path / "fixed_dispatch.json")
        kernel = FusionKernel(cfg)
        calls: dict[str, int] = {"fixed": 0}

        def fake_fixed(*, preserve_initial_state=False, boundary_flux=None):
            calls["fixed"] += 1
            return {"boundary_variant": "fixed_boundary", "converged": True}

        monkeypatch.setattr(kernel, "solve_fixed_boundary", fake_fixed)
        result = kernel.solve()

        assert calls["fixed"] == 1
        assert result["boundary_variant"] == "fixed_boundary"

    def test_solve_dispatch_free_from_config(self, tmp_path, monkeypatch):
        cfg = _write_config(
            tmp_path / "free_dispatch.json",
            extra_solver={"boundary_variant": "free_boundary"},
            free_boundary={
                "current_limits": [5.0e4, 5.0e4],
                "target_flux_points": [[3.5, 0.0], [4.0, 0.5]],
            },
        )
        kernel = FusionKernel(cfg)
        seen: dict[str, CoilSet] = {}

        def fake_free(coils, max_outer_iter=20, tol=1e-4, optimize_shape=False, tikhonov_alpha=1e-4):
            seen["coils"] = coils
            return {"boundary_variant": "free_boundary", "outer_iterations": 1, "final_diff": 0.0}

        monkeypatch.setattr(kernel, "solve_free_boundary", fake_free)
        result = kernel.solve()

        assert result["boundary_variant"] == "free_boundary"
        assert "coils" in seen
        assert seen["coils"].target_flux_points is not None
        assert seen["coils"].target_flux_points.shape == (2, 2)

    def test_solve_equilibrium_reports_fixed_variant(self, tmp_path):
        cfg = _write_config(tmp_path / "fixed_variant_result.json", grid=(10, 10), max_iter=3)
        kernel = FusionKernel(cfg)
        result = kernel.solve_fixed_boundary()
        assert result["boundary_variant"] == "fixed_boundary"


# ── vacuum field ─────────────────────────────────────────────────────


class TestVacuumField:
    def test_vacuum_field_shape(self, kernel):
        psi_vac = kernel.calculate_vacuum_field()
        assert psi_vac.shape == kernel.Psi.shape

    def test_vacuum_field_finite(self, kernel):
        psi_vac = kernel.calculate_vacuum_field()
        assert np.all(np.isfinite(psi_vac))

    def test_vacuum_field_nonzero(self, kernel):
        psi_vac = kernel.calculate_vacuum_field()
        assert np.max(np.abs(psi_vac)) > 0
        # Physics: Flux should be higher at the edge closer to coils
        # PF1 at (3, 4), PF2 at (5, -4). 
        # Grid R in [2, 6], Z in [-3, 3].
        # Point (3, 3) is closer to PF1 than (3, 0)
        assert abs(psi_vac[-1, 4]) > abs(psi_vac[len(kernel.Z)//2, 4])

    def test_no_coils_gives_zero_vac(self, tmp_path):
        cfg = {
            "reactor_name": "No-Coils",
            "grid_resolution": [8, 8],
            "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
            "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
            "coils": [],
            "solver": {"max_iterations": 3, "convergence_threshold": 1e-4, "relaxation_factor": 0.1},
        }
        p = tmp_path / "no_coils.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        fk = FusionKernel(p)
        psi_vac = fk.calculate_vacuum_field()
        assert np.allclose(psi_vac, 0.0)


# ── topology ─────────────────────────────────────────────────────────


class TestTopology:
    def test_find_x_point_returns_tuple(self, kernel):
        psi_vac = kernel.calculate_vacuum_field()
        pos, psi_x = kernel.find_x_point(psi_vac)
        assert len(pos) == 2
        assert isinstance(psi_x, float)

    def test_select_x_point_index_prefers_saddle_candidates(self):
        gradient_norm = np.array(
            [
                [0.40, 0.01, 0.30],
                [0.50, 0.20, 0.25],
                [0.60, 0.70, 0.80],
            ],
            dtype=float,
        )
        hessian_det = np.array(
            [
                [1.0, 1.0, 1.0],
                [-0.5, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        search_mask = np.ones_like(gradient_norm, dtype=bool)

        iz, ir, used_saddle = _select_x_point_index(gradient_norm, search_mask, hessian_det)

        assert (iz, ir) == (1, 0)
        assert used_saddle is True

    def test_find_x_point_identifies_saddle_field(self, kernel):
        kernel.Psi = (kernel.RR - 4.0) ** 2 - (kernel.ZZ + 2.0) ** 2
        pos, psi_x = kernel.find_x_point(kernel.Psi)

        assert abs(pos[0] - 4.0) <= kernel.dR
        assert abs(pos[1] + 2.0) <= kernel.dZ
        assert abs(psi_x) <= 4.0 * max(kernel.dR, kernel.dZ)

    def test_magnetic_axis_initial(self, kernel):
        kernel.Psi = kernel.calculate_vacuum_field()
        iz, ir, psi_axis = kernel._find_magnetic_axis()
        assert 0 <= iz < kernel.NZ
        assert 0 <= ir < kernel.NR
        assert isinstance(psi_axis, float)


# ── profile functions ────────────────────────────────────────────────


class TestProfiles:
    def test_mtanh_zero_outside(self):
        psi_norm = np.array([-0.1, 1.5, 2.0])
        params = {"ped_top": 0.92, "ped_width": 0.05, "ped_height": 1.0, "core_alpha": 0.3}
        result = FusionKernel.mtanh_profile(psi_norm, params)
        assert np.allclose(result, 0.0)

    def test_mtanh_positive_inside(self):
        psi_norm = np.linspace(0.0, 0.9, 50)
        params = {"ped_top": 0.92, "ped_width": 0.05, "ped_height": 1.0, "core_alpha": 0.3}
        result = FusionKernel.mtanh_profile(psi_norm, params)
        assert np.all(result >= 0)
        assert np.max(result) > 0

    def test_mtanh_pedestal_peak(self):
        psi_norm = np.linspace(0.0, 0.99, 200)
        params = {"ped_top": 0.92, "ped_width": 0.05, "ped_height": 1.0, "core_alpha": 0.3}
        result = FusionKernel.mtanh_profile(psi_norm, params)
        assert result[0] > result[-1]


# ── source term ──────────────────────────────────────────────────────


class TestSourceTerm:
    def test_source_shape(self, kernel):
        J = kernel.update_plasma_source_nonlinear(Psi_axis=1.0, Psi_boundary=0.0)
        assert J.shape == kernel.Psi.shape

    def test_source_finite(self, kernel):
        kernel.Psi = kernel.calculate_vacuum_field()
        # Find axis to set a physical source
        iz, ir, psi_ax = kernel._find_magnetic_axis()
        J = kernel.update_plasma_source_nonlinear(Psi_axis=psi_ax, Psi_boundary=0.0)
        assert np.all(np.isfinite(J))
        # Physics: Current should peak near magnetic axis (within 3 grid points for coarse mesh)
        iz_j, ir_j = np.unravel_index(np.argmax(np.abs(J)), J.shape)
        assert abs(iz_j - iz) <= 3
        assert abs(ir_j - ir) <= 3

    def test_source_current_normalisation(self, kernel):
        kernel.Psi = kernel.calculate_vacuum_field()
        iz, ir, psi_ax = kernel._find_magnetic_axis()
        J = kernel.update_plasma_source_nonlinear(Psi_axis=psi_ax, Psi_boundary=0.0)
        I_total = float(np.sum(J)) * kernel.dR * kernel.dZ
        I_target = kernel.cfg["physics"]["plasma_current_target"]
        assert abs(I_total - I_target) < 0.01 * abs(I_target) # Tighten to 1%

    def test_hmode_profile(self, tmp_path):
        cfg = _write_config(tmp_path / "hmode.json", profile_mode="h-mode")
        fk = FusionKernel(cfg)
        fk.Psi = fk.calculate_vacuum_field()
        J = fk.update_plasma_source_nonlinear(Psi_axis=1.0, Psi_boundary=0.0)
        assert np.all(np.isfinite(J))


# ── elliptic sub-solvers ─────────────────────────────────────────────


class TestSubSolvers:
    def test_jacobi_step_preserves_shape(self, small_kernel):
        src = np.ones_like(small_kernel.Psi)
        result = small_kernel._jacobi_step(small_kernel.Psi, src)
        assert result.shape == small_kernel.Psi.shape

    def test_sor_step_finite(self, small_kernel):
        small_kernel.Psi = np.random.default_rng(42).standard_normal(small_kernel.Psi.shape)
        src = np.ones_like(small_kernel.Psi) * 0.01
        result = small_kernel._sor_step(small_kernel.Psi, src, omega=1.4)
        assert np.all(np.isfinite(result))

    def test_multigrid_vcycle_reduces_residual(self, small_kernel):
        rng = np.random.default_rng(99)
        small_kernel.Psi = rng.standard_normal(small_kernel.Psi.shape) * 0.01
        src = np.ones_like(small_kernel.Psi) * 0.001

        r_before = small_kernel._mg_residual(small_kernel.Psi, src, small_kernel.RR, small_kernel.dR, small_kernel.dZ)
        norm_before = float(np.sqrt(np.mean(r_before**2)))

        psi_after = small_kernel._multigrid_vcycle(
            small_kernel.Psi.copy(),
            src,
            small_kernel.RR,
            small_kernel.dR,
            small_kernel.dZ,
        )
        r_after = small_kernel._mg_residual(psi_after, src, small_kernel.RR, small_kernel.dR, small_kernel.dZ)
        norm_after = float(np.sqrt(np.mean(r_after**2)))
        assert norm_after < norm_before

    def test_restrict_prolongate_roundtrip(self):
        fine = np.random.default_rng(7).standard_normal((9, 9))
        coarse = FusionKernel._restrict_full_weight(fine)
        reconstructed = FusionKernel._prolongate_bilinear(coarse, 9, 9)
        assert reconstructed.shape == fine.shape
        assert np.all(np.isfinite(reconstructed))


# ── equilibrium solver ───────────────────────────────────────────────


class TestSolveEquilibrium:
    def test_sor_converges(self, tmp_path):
        cfg = _write_config(tmp_path / "sor.json", grid=(16, 16), method="sor", max_iter=50)
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert "psi" in result
        assert "converged" in result
        assert "iterations" in result
        assert result["iterations"] > 0
        assert np.all(np.isfinite(fk.Psi))

    def test_multigrid_converges(self, tmp_path):
        cfg = _write_config(tmp_path / "mg.json", grid=(16, 16), method="multigrid", max_iter=50)
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert np.all(np.isfinite(fk.Psi))
        assert result["solver_method"] == "multigrid"

    def test_anderson_converges(self, tmp_path):
        cfg = _write_config(
            tmp_path / "and.json",
            grid=(16, 16),
            method="anderson",
            max_iter=50,
            extra_solver={"anderson_depth": 3},
        )
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert np.all(np.isfinite(fk.Psi))
        assert result["solver_method"] == "anderson"

    def test_jacobi_converges(self, tmp_path):
        cfg = _write_config(tmp_path / "jac.json", grid=(16, 16), method="jacobi", max_iter=50)
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert np.all(np.isfinite(fk.Psi))

    def test_newton_solver(self, tmp_path):
        cfg = _write_config(tmp_path / "newton.json", grid=(16, 16), method="newton", max_iter=30)
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert result["solver_method"] == "newton"
        assert np.all(np.isfinite(fk.Psi))

    def test_residual_history_populated(self, tmp_path):
        cfg = _write_config(tmp_path / "hist.json", grid=(16, 16), method="sor", max_iter=10)
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert len(result["residual_history"]) > 0

    def test_gs_residual_reported(self, tmp_path):
        cfg = _write_config(tmp_path / "gs.json", grid=(16, 16), method="sor", max_iter=10)
        fk = FusionKernel(cfg)
        result = fk.solve_equilibrium()
        assert "gs_residual" in result
        assert np.isfinite(result["gs_residual"])

    def test_preserve_initial_state(self, tmp_path):
        cfg = _write_config(tmp_path / "preserve.json", grid=(16, 16), method="sor", max_iter=10)
        fk = FusionKernel(cfg)
        fk.Psi = fk.calculate_vacuum_field()
        boundary = fk.Psi.copy()
        fk.solve_equilibrium(preserve_initial_state=True)
        # Boundaries should match the vacuum field
        assert np.allclose(fk.Psi[0, :], boundary[0, :])
        assert np.allclose(fk.Psi[-1, :], boundary[-1, :])

    def test_boundary_flux_shape_mismatch_raises(self, tmp_path):
        cfg = _write_config(tmp_path / "bad_bc.json", grid=(16, 16), method="sor", max_iter=3)
        fk = FusionKernel(cfg)
        with pytest.raises(ValueError, match="shape"):
            fk.solve_equilibrium(boundary_flux=np.zeros((4, 4)))


# ── B-field ──────────────────────────────────────────────────────────


class TestBField:
    def test_b_field_computed_after_solve(self, tmp_path):
        cfg = _write_config(tmp_path / "bf.json", grid=(16, 16), method="sor", max_iter=10)
        fk = FusionKernel(cfg)
        fk.solve_equilibrium()
        assert hasattr(fk, "B_R")
        assert hasattr(fk, "B_Z")
        assert fk.B_R.shape == fk.Psi.shape
        assert fk.B_Z.shape == fk.Psi.shape
        assert np.all(np.isfinite(fk.B_R))
        assert np.all(np.isfinite(fk.B_Z))

    def test_compute_b_field_matches_analytic_quadratic(self, kernel):
        kernel.Psi = kernel.RR**2 + 2.0 * kernel.ZZ**2
        kernel.compute_b_field()

        r_safe = np.maximum(kernel.RR, 1e-6)
        expected_b_r = -(4.0 * kernel.ZZ) / r_safe
        expected_b_z = np.full_like(kernel.RR, 2.0)

        np.testing.assert_allclose(kernel.B_R[1:-1, 1:-1], expected_b_r[1:-1, 1:-1], atol=1e-10)
        np.testing.assert_allclose(kernel.B_Z[1:-1, 1:-1], expected_b_z[1:-1, 1:-1], atol=1e-10)


# ── GS residual enforcement ─────────────────────────────────────────


class TestGSResidual:
    def test_require_gs_residual_negative_threshold_raises(self, tmp_path):
        cfg = _write_config(
            tmp_path / "gs_bad.json",
            grid=(8, 8),
            method="sor",
            max_iter=5,
            require_gs_residual=True,
            gs_residual_threshold=-1.0,
        )
        fk = FusionKernel(cfg)
        with pytest.raises(ValueError, match="gs_residual_threshold"):
            fk.solve_equilibrium()

    def test_gs_residual_rms_computed(self, kernel):
        kernel.Psi = kernel.calculate_vacuum_field()
        mu0 = kernel.cfg["physics"]["vacuum_permeability"]
        kernel.update_plasma_source_nonlinear(Psi_axis=1.0, Psi_boundary=0.0)
        source = -mu0 * kernel.RR * kernel.J_phi
        rms = kernel._compute_gs_residual_rms(source)
        assert isinstance(rms, float)
        assert np.isfinite(rms)


# ── CoilSet and free-boundary ────────────────────────────────────────


class TestCoilSet:
    def test_coilset_defaults(self):
        cs = CoilSet()
        assert cs.positions == []
        assert cs.currents.shape == (0,)
        assert cs.current_limits is None

    def test_free_boundary_basic(self, tmp_path):
        cfg = _write_config(tmp_path / "fb.json", grid=(16, 16), method="sor", max_iter=10)
        fk = FusionKernel(cfg)

        coils = CoilSet(
            positions=[(3.0, 4.0), (5.0, -4.0)],
            currents=np.array([2.0, -1.0]),
            turns=[10, 10],
        )
        result = fk.solve_free_boundary(coils, max_outer_iter=3, tol=1e-2)
        assert "outer_iterations" in result
        assert "final_diff" in result
        assert "coil_currents" in result
        assert np.all(np.isfinite(result["coil_currents"]))


# ── interpolation ────────────────────────────────────────────────────


class TestInterpolation:
    def test_interp_psi_in_domain(self, tmp_path):
        cfg = _write_config(tmp_path / "interp.json", grid=(16, 16), method="sor", max_iter=10)
        fk = FusionKernel(cfg)
        fk.solve_equilibrium()
        val = fk._interp_psi(4.0, 0.0)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_interp_psi_boundary(self, tmp_path):
        cfg = _write_config(tmp_path / "interp_bc.json", grid=(16, 16), method="sor", max_iter=5)
        fk = FusionKernel(cfg)
        fk.Psi = np.ones_like(fk.Psi) * 3.14
        val = fk._interp_psi(fk.R[0], fk.Z[0])
        assert abs(val - 3.14) < 0.01


# ── save results ─────────────────────────────────────────────────────


class TestSaveResults:
    def test_save_creates_file(self, tmp_path, small_kernel):
        small_kernel.solve_equilibrium()
        out = tmp_path / "eq.npz"
        small_kernel.save_results(str(out))
        assert out.exists()
        data = np.load(out)
        assert "R" in data
        assert "Z" in data
        assert "Psi" in data
        assert "J_phi" in data
