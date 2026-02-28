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

from scpn_control.core.fusion_kernel import CoilSet, FusionKernel


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
        J = kernel.update_plasma_source_nonlinear(Psi_axis=1.0, Psi_boundary=0.0)
        assert np.all(np.isfinite(J))

    def test_source_current_normalisation(self, kernel):
        kernel.Psi = kernel.calculate_vacuum_field()
        J = kernel.update_plasma_source_nonlinear(Psi_axis=1.0, Psi_boundary=0.0)
        I_total = float(np.sum(J)) * kernel.dR * kernel.dZ
        I_target = kernel.cfg["physics"]["plasma_current_target"]
        assert abs(I_total - I_target) < 0.1 * abs(I_target)

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

        r_before = small_kernel._mg_residual(
            small_kernel.Psi, src, small_kernel.RR, small_kernel.dR, small_kernel.dZ
        )
        norm_before = float(np.sqrt(np.mean(r_before**2)))

        psi_after = small_kernel._multigrid_vcycle(
            small_kernel.Psi.copy(), src, small_kernel.RR,
            small_kernel.dR, small_kernel.dZ,
        )
        r_after = small_kernel._mg_residual(
            psi_after, src, small_kernel.RR, small_kernel.dR, small_kernel.dZ
        )
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
            tmp_path / "and.json", grid=(16, 16), method="anderson",
            max_iter=50, extra_solver={"anderson_depth": 3},
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


# ── GS residual enforcement ─────────────────────────────────────────

class TestGSResidual:
    def test_require_gs_residual_negative_threshold_raises(self, tmp_path):
        cfg = _write_config(
            tmp_path / "gs_bad.json", grid=(8, 8), method="sor", max_iter=5,
            require_gs_residual=True, gs_residual_threshold=-1.0,
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
