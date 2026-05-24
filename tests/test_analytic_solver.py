# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Analytic Solver

"""Tests for scpn_control.control.analytic_solver."""

import numpy as np
import pytest
from pathlib import Path

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
    return AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernel, verbose=False)


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

    def test_rejects_nonfinite_or_invalid_shafranov_domain(self, solver):
        with pytest.raises(ValueError, match="R_geo must exceed a_min"):
            solver.calculate_required_Bv(R_geo=1.0, a_min=1.0, Ip_MA=15.0)
        with pytest.raises(ValueError, match="beta_p must be finite and >= 0"):
            solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0, beta_p=float("nan"))
        with pytest.raises(ValueError, match="li must be finite and >= 0"):
            solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0, li=-0.1)

    def test_coil_efficiencies_shape(self, solver):
        eff = solver.compute_coil_efficiencies(target_R=6.2)
        assert eff.shape == (3,)

    def test_solve_coil_currents_shape(self, solver):
        currents = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2)
        assert currents.shape == (3,)

    def test_apply_currents_length_mismatch(self, solver):
        with pytest.raises(ValueError, match="mismatch"):
            solver.apply_currents(np.array([1.0, 2.0]))  # 2 instead of 3

    def test_apply_currents_rejects_nonfinite_values(self, solver):
        with pytest.raises(ValueError, match="currents must contain only finite values"):
            solver.apply_currents(np.array([1.0, np.nan, 3.0]))

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


class FakeKernelZeroInfluence(FakeKernel):
    def calculate_vacuum_field(self):
        return np.zeros((len(self.Z), len(self.R)))


class FakeKernelBadFieldShape(FakeKernel):
    def calculate_vacuum_field(self):
        return np.zeros((len(self.Z), len(self.R) - 1))


class FakeKernelNonfiniteField(FakeKernel):
    def calculate_vacuum_field(self):
        psi = np.zeros((len(self.Z), len(self.R)))
        psi[0, 0] = np.nan
        return psi


class TestComputeCoilEfficienciesEdgeCases:
    def test_rejects_zero_coils(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelNoCoils, verbose=False)
        with pytest.raises(ValueError, match="no coils"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonpositive_target_r(self, solver):
        with pytest.raises(ValueError, match="target_R must be finite and > 0"):
            solver.compute_coil_efficiencies(target_R=0.0)

    def test_rejects_nonfinite_target_z(self, solver):
        with pytest.raises(ValueError, match="target_Z must be finite"):
            solver.compute_coil_efficiencies(target_R=6.2, target_Z=float("nan"))

    def test_rejects_target_r_outside_finite_difference_domain(self, solver):
        with pytest.raises(ValueError, match="target_R must lie inside the kernel R grid interior"):
            solver.compute_coil_efficiencies(target_R=100.0)

    def test_rejects_target_z_outside_kernel_grid(self, solver):
        with pytest.raises(ValueError, match="target_Z must lie inside the kernel Z grid"):
            solver.compute_coil_efficiencies(target_R=6.2, target_Z=100.0)

    def test_restores_currents_after_efficiency_calc(self, solver):
        solver.kernel.cfg["coils"][0]["current"] = 7.5
        solver.compute_coil_efficiencies(target_R=6.2)
        assert solver.kernel.cfg["coils"][0]["current"] == 7.5

    def test_rejects_vacuum_field_shape_mismatch(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelBadFieldShape, verbose=False)

        with pytest.raises(ValueError, match="vacuum field shape must match kernel grid"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonfinite_vacuum_field(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelNonfiniteField, verbose=False)

        with pytest.raises(ValueError, match="vacuum field must contain only finite values"):
            solver.compute_coil_efficiencies(target_R=6.2)


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

    def test_rejects_nonfinite_target_and_ridge(self, solver):
        with pytest.raises(ValueError, match="target_Bv must be finite"):
            solver.solve_coil_currents(target_Bv=float("inf"), target_R=6.2)
        with pytest.raises(ValueError, match="ridge_lambda must be finite"):
            solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=float("nan"))


class TestSolveCoilCurrentsInfeasibleInfluence:
    def test_rejects_nonzero_target_with_zero_influence(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelZeroInfluence, verbose=False)

        with pytest.raises(ValueError, match="nonzero target_Bv requires nonzero coil influence"):
            solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2)

    def test_zero_target_with_zero_influence_returns_zero_currents(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelZeroInfluence, verbose=False)

        currents = solver.solve_coil_currents(target_Bv=0.0, target_R=6.2)

        np.testing.assert_allclose(currents, np.zeros(3))


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

    def test_preserves_existing_spdx_metadata_and_final_newline(self, solver, tmp_path):
        import json

        out_path = tmp_path / "out.json"
        out_path.write_text(
            json.dumps(
                {
                    "license": "SPDX-License-Identifier: AGPL-3.0-or-later",
                    "reactor_name": "existing validated artefact",
                    "coils": [],
                },
                indent=4,
            )
            + "\n",
            encoding="utf-8",
        )

        out = solver.apply_and_save(np.array([0.5, -0.3, 1.2]), output_path=str(out_path))
        text = Path(out).read_text(encoding="utf-8")
        data = json.loads(text)

        assert data["license"] == "SPDX-License-Identifier: AGPL-3.0-or-later"
        assert text.endswith("\n")

    def test_default_path_used_when_none(self, solver):
        import os

        repo_default = Path(__file__).resolve().parents[1] / "validation" / "iter_analytic_config.json"
        original = repo_default.read_bytes() if repo_default.exists() else None

        out = str(repo_default)
        try:
            out = solver.apply_and_save(np.array([0.1, 0.2, 0.3]))
            assert os.path.isfile(out)
        finally:
            if original is None:
                Path(out).unlink(missing_ok=True)
            else:
                repo_default.write_bytes(original)


# ── run_analytic_solver ──────────────────────────────────────────

from scpn_control.control.analytic_solver import run_analytic_solver


class TestRunAnalyticSolver:
    def test_legacy_config_fallback_requires_explicit_opt_in(self):
        with pytest.raises(ValueError, match="allow_legacy_config_fallback=True"):
            run_analytic_solver(
                config_path="dummy.json",
                allow_config_fallback=True,
                allow_legacy_config_fallback=False,
                save_config=False,
                verbose=False,
                kernel_factory=FakeKernel,
            )

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
