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
