"""Tests for scpn_control.control.analytic_solver."""

import numpy as np
import pytest

from scpn_control.control.analytic_solver import AnalyticEquilibriumSolver


class FakeKernel:
    """Minimal FusionKernel stub for unit testing."""

    def __init__(self, config_path: str):
        self.R = np.linspace(4.0, 8.5, 65)
        self.Z = np.linspace(-4.0, 4.0, 65)
        self.dR = float(self.R[1] - self.R[0])
        self.cfg = {
            "coils": [
                {"name": "PF1", "current": 0.0, "R": 4.5, "Z": 3.0},
                {"name": "PF2", "current": 0.0, "R": 8.0, "Z": 0.0},
                {"name": "PF3", "current": 0.0, "R": 4.5, "Z": -3.0},
            ],
        }

    def calculate_vacuum_field(self):
        nz, nr = len(self.Z), len(self.R)
        psi = np.zeros((nz, nr))
        for coil in self.cfg["coils"]:
            I = coil["current"]
            if abs(I) < 1e-12:
                continue
            Rc, Zc = coil["R"], coil["Z"]
            for j, r in enumerate(self.R):
                for i, z in enumerate(self.Z):
                    dist = max(np.sqrt((r - Rc) ** 2 + (z - Zc) ** 2), 0.01)
                    psi[i, j] += I * r / dist
        return psi


@pytest.fixture
def solver(tmp_path):
    cfg = tmp_path / "test_config.json"
    cfg.write_text("{}")
    return AnalyticEquilibriumSolver(
        str(cfg), kernel_factory=FakeKernel, verbose=False
    )


class TestAnalyticEquilibriumSolver:
    def test_calculate_bv_sign(self, solver):
        Bv = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)
        assert Bv < 0  # Shafranov: vertical field is negative for outward force balance

    def test_calculate_bv_scales_with_current(self, solver):
        Bv_low = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=5.0)
        Bv_high = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)
        assert abs(Bv_high) > abs(Bv_low)

    def test_rejects_nonpositive_inputs(self, solver):
        with pytest.raises(ValueError):
            solver.calculate_required_Bv(R_geo=0.0, a_min=2.0, Ip_MA=15.0)
        with pytest.raises(ValueError):
            solver.calculate_required_Bv(R_geo=6.2, a_min=-1.0, Ip_MA=15.0)

    def test_coil_efficiencies_shape(self, solver):
        eff = solver.compute_coil_efficiencies(target_R=6.2)
        assert eff.shape == (3,)

    def test_solve_coil_currents_shape(self, solver):
        currents = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2)
        assert currents.shape == (3,)

    def test_apply_currents_length_mismatch(self, solver):
        with pytest.raises(ValueError, match="mismatch"):
            solver.apply_currents(np.array([1.0, 2.0]))  # 2 instead of 3

    def test_apply_currents_updates_config(self, solver):
        currents = np.array([1.0, 2.0, 3.0])
        solver.apply_currents(currents)
        for i, coil in enumerate(solver.kernel.cfg["coils"]):
            assert coil["current"] == pytest.approx(currents[i])

    def test_bv_shafranov_formula_magnitude(self, solver):
        """Bv for ITER-like params: |Bv| should be O(0.01-1 T)."""
        Bv = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)
        assert 0.001 < abs(Bv) < 10.0

    def test_coil_efficiencies_positive(self, solver):
        eff = solver.compute_coil_efficiencies(target_R=6.2)
        assert np.all(np.isfinite(eff))

    def test_solve_and_apply_round_trip(self, solver):
        """Solve for coil currents then verify applied currents match."""
        currents = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2)
        solver.apply_currents(currents)
        applied = np.array([c["current"] for c in solver.kernel.cfg["coils"]])
        assert np.allclose(applied, currents)


# ── Zero-coil kernel edge case ────────────────────────────────────

class FakeKernelNoCoils:
    def __init__(self, config_path: str):
        self.R = np.linspace(4.0, 8.5, 10)
        self.Z = np.linspace(-4.0, 4.0, 10)
        self.dR = float(self.R[1] - self.R[0])
        self.cfg = {"coils": []}

    def calculate_vacuum_field(self):
        return np.zeros((10, 10))


class TestComputeCoilEfficienciesEdgeCases:
    def test_rejects_zero_coils(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(
            str(cfg), kernel_factory=FakeKernelNoCoils, verbose=False
        )
        with pytest.raises(ValueError, match="no coils"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonpositive_target_r(self, solver):
        with pytest.raises(ValueError, match="target_R must be > 0"):
            solver.compute_coil_efficiencies(target_R=0.0)

    def test_restores_currents_after_efficiency_calc(self, solver):
        solver.kernel.cfg["coils"][0]["current"] = 7.5
        solver.compute_coil_efficiencies(target_R=6.2)
        assert solver.kernel.cfg["coils"][0]["current"] == 7.5


# ── Ridge-regularized solve ──────────────────────────────────────

class TestSolveCoilCurrentsRidge:
    def test_ridge_shrinks_norm(self, solver):
        c_noreg = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=0.0)
        c_ridge = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=50.0)
        assert np.linalg.norm(c_ridge) <= np.linalg.norm(c_noreg) + 1e-12

    def test_negative_ridge_clamped_to_zero(self, solver):
        c1 = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=0.0)
        c2 = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=-5.0)
        np.testing.assert_allclose(c1, c2)


# ── apply_and_save ───────────────────────────────────────────────

class TestApplyAndSave:
    def test_writes_valid_json(self, solver, tmp_path):
        import json
        out = solver.apply_and_save(
            np.array([0.5, -0.3, 1.2]),
            output_path=str(tmp_path / "out.json"),
        )
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["coils"][0]["current"] == 0.5
        assert data["coils"][2]["current"] == 1.2

    def test_default_path_used_when_none(self, solver):
        # Just verify it doesn't raise when output_path=None
        # (writes to validation/ dir which may or may not exist)
        import os
        out = solver.apply_and_save(np.array([0.1, 0.2, 0.3]))
        assert os.path.isfile(out)
        os.remove(out)


# ── run_analytic_solver ──────────────────────────────────────────

from scpn_control.control.analytic_solver import run_analytic_solver


class TestRunAnalyticSolver:
    def test_smoke(self, tmp_path):
        summary = run_analytic_solver(
            config_path="dummy.json",
            save_config=True,
            output_config_path=str(tmp_path / "analytic.json"),
            verbose=False,
            kernel_factory=FakeKernel,
        )
        for key in (
            "required_bv_t",
            "coil_currents_ma",
            "coil_current_l2_norm",
            "max_abs_coil_current_ma",
            "target_r_m",
            "ip_target_ma",
        ):
            assert key in summary
        assert np.isfinite(summary["required_bv_t"])
        assert summary["target_r_m"] == 6.2

    def test_no_save(self):
        summary = run_analytic_solver(
            config_path="dummy.json",
            save_config=False,
            verbose=False,
            kernel_factory=FakeKernel,
        )
        assert summary["output_config_path"] is None

    def test_deterministic(self):
        kwargs = dict(
            config_path="dummy.json",
            save_config=False,
            verbose=False,
            kernel_factory=FakeKernel,
        )
        a = run_analytic_solver(**kwargs)
        b = run_analytic_solver(**kwargs)
        assert a["required_bv_t"] == b["required_bv_t"]
        assert a["coil_current_l2_norm"] == b["coil_current_l2_norm"]

    def test_custom_physics_params(self, tmp_path):
        summary = run_analytic_solver(
            config_path="dummy.json",
            target_r=5.0,
            a_minor=1.5,
            ip_target_ma=10.0,
            ridge_lambda=1.0,
            save_config=True,
            output_config_path=str(tmp_path / "custom.json"),
            verbose=False,
            kernel_factory=FakeKernel,
        )
        assert summary["target_r_m"] == 5.0
        assert summary["ip_target_ma"] == 10.0
