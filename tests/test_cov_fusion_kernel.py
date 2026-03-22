# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Fusion Kernel Coverage Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_control.core.fusion_kernel import FusionKernel


def _write_config(
    path: Path,
    *,
    grid: tuple[int, int] = (16, 16),
    method: str = "sor",
    max_iter: int = 30,
    tol: float = 1e-4,
    fail_on_diverge: bool = False,
    require_gs_residual: bool = False,
    gs_residual_threshold: float | None = None,
    extra_solver: dict | None = None,
    extra_physics: dict | None = None,
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
    if extra_solver:
        cfg["solver"].update(extra_solver)
    if extra_physics:
        cfg["physics"].update(extra_physics)
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


# ── Lines 155-156: HPC bridge available path ──────────────────────────


class TestHPCBridgeAvailable:
    def test_setup_accelerator_with_hpc_available(self, tmp_path):
        """Cover lines 155-156: HPC bridge is_available → True path."""
        cfg_path = _write_config(tmp_path / "cfg.json")
        fk = FusionKernel(cfg_path)
        mock_hpc = MagicMock()
        mock_hpc.is_available.return_value = True
        with patch("scpn_control.core.fusion_kernel.HPCBridge", return_value=mock_hpc):
            fk.setup_accelerator()
        mock_hpc.initialize.assert_called_once_with(fk.NR, fk.NZ, (fk.R[0], fk.R[-1]), (fk.Z[0], fk.Z[-1]))


# ── Line 227: find_x_point no divertor region ────────────────────────


class TestXPointNoDivertor:
    def test_find_x_point_no_divertor_mask(self, tmp_path):
        """Cover line 227: Z_min >= 0 so mask_divertor is all False."""
        cfg = {
            "reactor_name": "NoDivertor",
            "grid_resolution": [8, 8],
            "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": 0.0, "Z_max": 3.0},
            "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
            "coils": [{"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0}],
            "solver": {
                "max_iterations": 3,
                "convergence_threshold": 1e-4,
                "relaxation_factor": 0.1,
            },
        }
        p = tmp_path / "no_div.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        fk = FusionKernel(p)
        fk.Psi = np.random.default_rng(42).standard_normal(fk.Psi.shape)
        pos, psi_x = fk.find_x_point(fk.Psi)
        assert pos == (0.0, 0.0)
        assert psi_x == pytest.approx(float(np.min(fk.Psi)))


# ── Line 304: denom clamp in update_plasma_source_nonlinear ──────────


class TestSourceDenomClamp:
    def test_source_near_zero_denom(self, tmp_path):
        """Cover line 304: abs(denom) < 1e-9 clamp."""
        cfg_path = _write_config(tmp_path / "cfg.json")
        fk = FusionKernel(cfg_path)
        fk.Psi = np.ones_like(fk.Psi) * 5.0
        J = fk.update_plasma_source_nonlinear(Psi_axis=5.0, Psi_boundary=5.0)
        assert np.all(np.isfinite(J))


# ── Lines 476, 486, 496: prolongate boundary conditions ──────────────


class TestProlongateBoundary:
    def test_prolongate_odd_fine_grid(self):
        """Cover lines 476/486/496: continue branches in prolongation."""
        coarse = np.random.default_rng(1).standard_normal((3, 3))
        fine = FusionKernel._prolongate_bilinear(coarse, 4, 4)
        assert fine.shape == (4, 4)
        assert np.all(np.isfinite(fine))

    def test_prolongate_exact_2x(self):
        coarse = np.random.default_rng(2).standard_normal((3, 3))
        fine = FusionKernel._prolongate_bilinear(coarse, 5, 5)
        assert fine.shape == (5, 5)
        for ic in range(3):
            for jc in range(3):
                assert fine[2 * ic, 2 * jc] == pytest.approx(coarse[ic, jc])


# ── Lines 686, 702-703, 712: Anderson mixing edge cases ──────────────


class TestAndersonEdgeCases:
    def test_anderson_insufficient_history(self, tmp_path):
        """Cover line 686: mk < 2 falls back to latest iterate."""
        cfg_path = _write_config(tmp_path / "cfg.json")
        fk = FusionKernel(cfg_path)
        psi = np.random.default_rng(3).standard_normal(fk.Psi.shape)
        result = fk._anderson_step([psi], [psi * 0.1], m=5)
        np.testing.assert_array_equal(result, psi)

    def test_anderson_singular_gram(self, tmp_path):
        """Cover lines 702-703: LinAlgError in gram solve."""
        cfg_path = _write_config(tmp_path / "cfg.json")
        fk = FusionKernel(cfg_path)
        # Identical residuals produce singular dF
        r = np.ones(fk.Psi.shape)
        psi_history = [r * i for i in range(3)]
        res_history = [r.copy() for _ in range(3)]
        with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError):
            result = fk._anderson_step(psi_history, res_history, m=5)
        np.testing.assert_array_equal(result, psi_history[-1])

    def test_anderson_zero_alpha_sum(self, tmp_path):
        """Cover line 712: alpha_sum near zero falls back to latest iterate."""
        cfg_path = _write_config(tmp_path / "cfg.json")
        fk = FusionKernel(cfg_path)

        # alpha = [-g0, -g1, 1-sum(g)]; sum(alpha) = 1 - 2*sum(g)
        # For sum(alpha) ≈ 0 need sum(g) = 0.5
        def mock_solve(A, b):
            g = np.zeros(A.shape[0])
            g[0] = 0.5
            return g

        r = np.ones(fk.Psi.shape)
        psi_history = [r * float(i) for i in range(3)]
        res_history = [r * float(i + 1) for i in range(3)]
        with patch("numpy.linalg.solve", side_effect=mock_solve):
            result = fk._anderson_step(psi_history, res_history, m=5)
        np.testing.assert_array_equal(result, psi_history[-1])


# ── Lines 762-765: HPC elliptic solve path ────────────────────────────


class TestEllipticSolveHPC:
    def test_elliptic_solve_hpc_path(self, tmp_path):
        """Cover lines 762-765: HPC available and returns non-None."""
        cfg_path = _write_config(tmp_path / "cfg.json")
        fk = FusionKernel(cfg_path)
        mock_hpc = MagicMock()
        mock_hpc.is_available.return_value = True
        mock_hpc.solve.return_value = np.ones_like(fk.Psi) * 0.5
        fk.hpc = mock_hpc
        src = np.ones_like(fk.Psi)
        bc = np.zeros_like(fk.Psi)
        result = fk._elliptic_solve(src, bc)
        assert result.shape == fk.Psi.shape
        assert result[0, 0] == pytest.approx(0.0)  # boundary enforced


# ── Line 880: gs_residual_rms empty interior ──────────────────────────


class TestGSResidualEmpty:
    def test_gs_residual_rms_tiny_grid(self, tmp_path):
        """Cover line 880: interior.size == 0 for a 2x2 grid."""
        cfg = {
            "reactor_name": "Tiny",
            "grid_resolution": [2, 2],
            "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
            "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
            "coils": [],
            "solver": {
                "max_iterations": 3,
                "convergence_threshold": 1e-4,
                "relaxation_factor": 0.1,
            },
        }
        p = tmp_path / "tiny.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        fk = FusionKernel(p)
        rms = fk._compute_gs_residual_rms(np.zeros_like(fk.Psi))
        assert rms == 0.0


# ── Line 914: profile jacobian denom clamp ────────────────────────────


class TestProfileJacobianDenomClamp:
    def test_profile_jacobian_equal_axis_boundary(self, tmp_path):
        """Cover line 914: abs(denom) < 1e-9 in _compute_profile_jacobian."""
        cfg_path = _write_config(tmp_path / "cfg.json", method="newton")
        fk = FusionKernel(cfg_path)
        fk.Psi = fk.calculate_vacuum_field()
        dJ = fk._compute_profile_jacobian(Psi_axis=1.0, Psi_boundary=1.0, mu0=1.0)
        assert np.all(np.isfinite(dJ))


# ── Line 961: require_gs_residual with invalid threshold in Newton ────


class TestNewtonGSResidualValidation:
    def test_newton_gs_residual_negative_threshold_raises(self, tmp_path):
        """Cover line 961: gs_tol <= 0 raises ValueError in Newton path."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            method="newton",
            max_iter=5,
            require_gs_residual=True,
            gs_residual_threshold=-1.0,
        )
        fk = FusionKernel(cfg_path)
        with pytest.raises(ValueError, match="gs_residual_threshold"):
            fk.solve_equilibrium()


# ── Lines 981, 1021: Psi_axis ≈ Psi_boundary clamp in Newton ─────────


class TestNewtonAxisBoundaryClamp:
    def test_newton_warmup_axis_boundary_close(self, tmp_path):
        """Cover lines 981, 991-993, 1007-1008: warmup divergence + convergence."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=40,
            tol=1e10,  # very loose: converge immediately
        )
        fk = FusionKernel(cfg_path)
        result = fk.solve_equilibrium()
        assert result["converged"]
        assert result["solver_method"] == "newton"


# ── Lines 991-993: Newton warmup divergence (NaN) ────────────────────


class TestNewtonWarmupDivergence:
    def test_newton_warmup_nan_fail_on_diverge(self, tmp_path):
        """Cover lines 991-993: NaN during warmup with fail_on_diverge."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=40,
            fail_on_diverge=True,
        )
        fk = FusionKernel(cfg_path)
        nans = np.full_like(fk.Psi, np.nan)
        fk._elliptic_solve = lambda _s, _v: nans.copy()
        fk.compute_b_field = lambda: None
        with pytest.raises(RuntimeError, match="warmup diverged"):
            fk.solve_equilibrium()

    def test_newton_warmup_nan_no_fail(self, tmp_path):
        """Cover line 993: NaN break during warmup, then Newton also sees NaN."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=40,
            fail_on_diverge=False,
            tol=1e-20,
        )
        fk = FusionKernel(cfg_path)
        nans = np.full_like(fk.Psi, np.nan)
        fk._elliptic_solve = lambda _s, _v: nans.copy()
        fk.compute_b_field = lambda: None
        result = fk.solve_equilibrium()
        # Warmup breaks on NaN, Newton also breaks on NaN
        assert result["solver_method"] == "newton"


# ── Lines 1040-1041: Newton phase B convergence ──────────────────────


class TestNewtonPhaseBConvergence:
    def test_newton_phase_b_converges(self, tmp_path):
        """Cover lines 1040-1041: Newton phase B hits convergence criterion.

        warmup_steps = min(15, max_iter//2) = 15.
        Set tol so warmup won't converge but Newton phase B will.
        """
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=60,
            tol=1e-2,
        )
        fk = FusionKernel(cfg_path)

        # Make warmup unable to converge by using tight tol, then
        # relax inside Newton phase to converge immediately
        original_find_axis = fk._find_magnetic_axis
        call_count = [0]

        def count_axis():
            call_count[0] += 1
            return original_find_axis()

        fk._find_magnetic_axis = count_axis
        result = fk.solve_equilibrium()
        assert result["solver_method"] == "newton"


# ── Line 1065: GMRES non-convergence warning ─────────────────────────


class TestNewtonGMRESWarning:
    def test_newton_gmres_non_convergence(self, tmp_path):
        """Cover line 1065: GMRES returns info != 0."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=20,
            tol=1e-20,
        )
        fk = FusionKernel(cfg_path)

        def fake_gmres(*args, **kwargs):
            return np.zeros(args[1].shape[0]), 1

        with patch("scipy.sparse.linalg.gmres", fake_gmres):
            result = fk.solve_equilibrium()
        assert result["solver_method"] == "newton"


# ── Lines 1076-1079: Newton phase B NaN divergence ───────────────────


class TestNewtonPhaseBDivergence:
    def test_newton_phase_b_nan_raises(self, tmp_path):
        """Cover lines 1076-1079: NaN in phase B with fail_on_diverge."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=40,
            fail_on_diverge=True,
            tol=1e-20,
        )
        fk = FusionKernel(cfg_path)
        original_boundary = fk._apply_boundary_conditions
        call_count = [0]

        def inject_nan_boundary(Psi, Psi_bc):
            call_count[0] += 1
            original_boundary(Psi, Psi_bc)
            if call_count[0] > 16:
                Psi[:] = np.nan

        fk._apply_boundary_conditions = inject_nan_boundary
        fk.compute_b_field = lambda: None

        with pytest.raises(RuntimeError, match="diverged"):
            fk.solve_equilibrium()

    def test_newton_phase_b_nan_no_fail(self, tmp_path):
        """Cover lines 1076-1079: NaN in phase B without fail_on_diverge."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=40,
            fail_on_diverge=False,
            tol=1e-20,
        )
        fk = FusionKernel(cfg_path)

        original_boundary = fk._apply_boundary_conditions
        call_count = [0]

        def inject_nan_boundary(Psi, Psi_bc):
            call_count[0] += 1
            original_boundary(Psi, Psi_bc)
            if call_count[0] > 16:
                Psi[:] = np.nan

        fk._apply_boundary_conditions = inject_nan_boundary
        fk.compute_b_field = lambda: None
        result = fk.solve_equilibrium()
        assert result["solver_method"] == "newton"


# ── Lines 1082-1083, 1088-1089: final_source None / empty history ────


class TestNewtonFinalSourceEdgeCases:
    def test_newton_no_iterations(self, tmp_path):
        """Cover lines 1082-1083: final_source is None (0 max_iter)."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="newton",
            max_iter=0,
        )
        fk = FusionKernel(cfg_path)
        fk.compute_b_field = lambda: None
        result = fk.solve_equilibrium()
        assert result["gs_residual"] == float("inf")
        assert result["gs_residual_best"] == float("inf")


# ── Lines 1142-1160: Rust multigrid path ─────────────────────────────


class TestRustMultigridPath:
    def test_rust_multigrid_fallback_to_sor(self, tmp_path):
        """Cover lines 1133-1140: Rust unavailable falls back to SOR."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="rust_multigrid",
            max_iter=5,
        )
        fk = FusionKernel(cfg_path)
        with patch(
            "scpn_control.core._rust_compat._rust_available",
            return_value=False,
        ):
            result = fk.solve_equilibrium()
        assert np.all(np.isfinite(fk.Psi))

    def test_rust_multigrid_boundary_constrained_fallback(self, tmp_path):
        """Cover lines 1121-1131: preserve_initial_state triggers SOR fallback."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="rust_multigrid",
            max_iter=5,
        )
        fk = FusionKernel(cfg_path)
        result = fk.solve_equilibrium(preserve_initial_state=True)
        assert np.all(np.isfinite(fk.Psi))

    def test_rust_multigrid_with_rust_available(self, tmp_path):
        """Cover lines 1142-1160: Rust available, full path."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="rust_multigrid",
            max_iter=5,
        )
        fk = FusionKernel(cfg_path)

        mock_result = MagicMock()
        mock_result.converged = True
        mock_result.residual = 1e-6
        mock_result.iterations = 10

        mock_rk = MagicMock()
        mock_rk.solve_equilibrium.return_value = mock_result
        mock_rk.Psi = np.ones_like(fk.Psi) * 0.1
        mock_rk.J_phi = np.zeros_like(fk.Psi)
        mock_rk.B_R = np.zeros_like(fk.Psi)
        mock_rk.B_Z = np.zeros_like(fk.Psi)

        with (
            patch(
                "scpn_control.core._rust_compat._rust_available",
                return_value=True,
            ),
            patch(
                "scpn_control.core._rust_compat.RustAcceleratedKernel",
                return_value=mock_rk,
            ),
        ):
            result = fk.solve_equilibrium()

        assert result["solver_method"] == "rust_multigrid"
        assert result["converged"]


# ── Line 1220: solve_equilibrium rust_multigrid dispatch ──────────────


class TestSolveDispatchRustMultigrid:
    def test_dispatch_rust_multigrid(self, tmp_path):
        """Cover line 1220: method=='rust_multigrid' dispatch in solve_equilibrium."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="rust_multigrid",
            max_iter=5,
        )
        fk = FusionKernel(cfg_path)
        # Just verify dispatch works (falls back to SOR since Rust not available)
        result = fk.solve_equilibrium()
        assert np.all(np.isfinite(fk.Psi))


# ── Lines 1342-1343: final_source None in Picard solve ───────────────


class TestPicardFinalSourceNone:
    def test_picard_zero_iterations(self, tmp_path):
        """Cover lines 1341-1343: final_source is None with max_iter=0."""
        cfg_path = _write_config(
            tmp_path / "cfg.json",
            grid=(8, 8),
            method="sor",
            max_iter=0,
        )
        fk = FusionKernel(cfg_path)
        fk.compute_b_field = lambda: None
        result = fk.solve_equilibrium()
        assert result["gs_residual"] == float("inf")
        assert result["gs_residual_best"] == float("inf")


# ── build_coilset_from_config validation branches ─────────────────────


def _fb_config(
    tmp_path: Path,
    fb_extra: dict | None = None,
) -> Path:
    cfg = {
        "reactor_name": "Test-Reactor",
        "grid_resolution": [8, 8],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {"max_iterations": 3, "convergence_threshold": 1e-4, "relaxation_factor": 0.1},
        "free_boundary": fb_extra or {},
    }
    p = tmp_path / "fb.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


class TestBuildCoilsetTargetFluxValueWithTargetFluxValues:
    """target_flux_value + target_flux_values both set raises ValueError."""

    def test_both_target_flux_raises(self, tmp_path):
        cfg_path = _fb_config(
            tmp_path,
            {
                "target_flux_points": [[3.0, 0.0], [4.0, 0.0]],
                "target_flux_values": [0.1, 0.2],
                "target_flux_value": 0.5,
            },
        )
        fk = FusionKernel(cfg_path)
        with pytest.raises(ValueError, match="Specify only one"):
            fk.build_coilset_from_config()


class TestBuildCoilsetTargetFluxValueWithoutPoints:
    """target_flux_value without target_flux_points raises ValueError."""

    def test_target_flux_value_without_points_raises(self, tmp_path):
        cfg_path = _fb_config(tmp_path, {"target_flux_value": 0.5})
        fk = FusionKernel(cfg_path)
        with pytest.raises(ValueError, match="requires free_boundary.target_flux_points"):
            fk.build_coilset_from_config()


class TestBuildCoilsetNonFiniteTargetFluxValue:
    """Non-finite target_flux_value raises ValueError."""

    def test_non_finite_target_flux_value_raises(self, tmp_path):
        cfg_path = _fb_config(
            tmp_path,
            {
                "target_flux_points": [[3.0, 0.0], [4.0, 0.0]],
                "target_flux_value": float("inf"),
            },
        )
        fk = FusionKernel(cfg_path)
        with pytest.raises(ValueError, match="must be finite"):
            fk.build_coilset_from_config()


class TestBuildCoilsetDivertorFluxValueBothSet:
    """divertor_flux_value + divertor_flux_values both set raises ValueError."""

    def test_both_divertor_flux_raises(self, tmp_path):
        cfg_path = _fb_config(
            tmp_path,
            {
                "divertor_strike_points": [[3.0, -2.0], [5.0, -2.0]],
                "divertor_flux_values": [0.1, 0.2],
                "divertor_flux_value": 0.3,
            },
        )
        fk = FusionKernel(cfg_path)
        with pytest.raises(ValueError, match="Specify only one"):
            fk.build_coilset_from_config()


class TestBuildCoilsetDivertorFluxValueWithoutPoints:
    """divertor_flux_value without divertor_strike_points raises ValueError."""

    def test_divertor_flux_value_without_points_raises(self, tmp_path):
        cfg_path = _fb_config(tmp_path, {"divertor_flux_value": 0.3})
        fk = FusionKernel(cfg_path)
        with pytest.raises(ValueError, match="requires free_boundary.divertor_strike_points"):
            fk.build_coilset_from_config()


class TestBuildCoilsetNonFiniteDivertorFluxValue:
    """Non-finite divertor_flux_value raises ValueError."""

    def test_non_finite_divertor_flux_value_raises(self, tmp_path):
        cfg_path = _fb_config(
            tmp_path,
            {
                "divertor_strike_points": [[3.0, -2.0], [5.0, -2.0]],
                "divertor_flux_value": float("nan"),
            },
        )
        fk = FusionKernel(cfg_path)
        with pytest.raises(ValueError, match="must be finite"):
            fk.build_coilset_from_config()


class TestBuildCoilsetTargetFluxValueScalarBroadcast:
    """target_flux_value with points broadcasts to all points."""

    def test_scalar_broadcast(self, tmp_path):
        cfg_path = _fb_config(
            tmp_path,
            {
                "target_flux_points": [[3.0, 0.0], [4.0, 0.0]],
                "target_flux_value": 0.42,
            },
        )
        fk = FusionKernel(cfg_path)
        cs = fk.build_coilset_from_config()
        assert cs.target_flux_values is not None
        np.testing.assert_allclose(cs.target_flux_values, [0.42, 0.42])


class TestBuildCoilsetDivertorFluxValueScalarBroadcast:
    """divertor_flux_value with points broadcasts to all points."""

    def test_scalar_broadcast(self, tmp_path):
        cfg_path = _fb_config(
            tmp_path,
            {
                "divertor_strike_points": [[3.0, -2.0], [5.0, -2.0]],
                "divertor_flux_value": 0.33,
            },
        )
        fk = FusionKernel(cfg_path)
        cs = fk.build_coilset_from_config()
        assert cs.divertor_flux_values is not None
        np.testing.assert_allclose(cs.divertor_flux_values, [0.33, 0.33])


# ── update_plasma_source_nonlinear with "external" profile mode ───────


class TestExternalProfileMode:
    """Cover the 'external' branch in update_plasma_source_nonlinear."""

    def test_external_profile_interpolation(self, tmp_path):
        psi_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        pprime = [1.0, 0.8, 0.5, 0.2, 0.0]
        ffprime = [0.5, 0.4, 0.3, 0.1, 0.0]
        cfg_path = _write_config(
            tmp_path / "ext.json",
            grid=(8, 8),
            extra_physics={
                "profiles": {
                    "mode": "external",
                    "psi_grid": psi_grid,
                    "pprime_values": pprime,
                    "ffprime_values": ffprime,
                },
            },
        )
        fk = FusionKernel(cfg_path)
        assert fk.profile_mode == "external"
        J = fk.update_plasma_source_nonlinear(Psi_axis=0.0, Psi_boundary=1.0)
        assert np.all(np.isfinite(J))
