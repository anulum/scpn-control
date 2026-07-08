# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Analytic Solver

"""Tests for scpn_control.control.analytic_solver."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
import pytest

from scpn_control.control.analytic_solver import AnalyticEquilibriumSolver


class FakeKernel:
    """Minimal FusionKernel stub for unit testing."""

    def __init__(self, config_path: str) -> None:
        self.R = cast("npt.NDArray[np.float64]", np.linspace(4.0, 8.5, 65, dtype=np.float64))
        self.Z = cast("npt.NDArray[np.float64]", np.linspace(-4.0, 4.0, 65, dtype=np.float64))
        self.dR = float(self.R[1] - self.R[0])
        self.cfg: dict[str, list[dict[str, Any]]] = {
            "coils": [
                {"name": "PF1", "current": 0.0, "R": 4.5, "Z": 3.0},
                {"name": "PF2", "current": 0.0, "R": 8.0, "Z": 0.0},
                {"name": "PF3", "current": 0.0, "R": 4.5, "Z": -3.0},
            ],
        }

    def calculate_vacuum_field(self) -> npt.NDArray[np.float64]:
        nz, nr = len(self.Z), len(self.R)
        psi = np.zeros((nz, nr), dtype=np.float64)
        for coil in self.cfg["coils"]:
            current = float(coil["current"])
            if abs(current) < 1e-12:
                continue
            Rc, Zc = float(coil["R"]), float(coil["Z"])
            for j, r in enumerate(self.R):
                for i, z in enumerate(self.Z):
                    dist = max(np.sqrt((r - Rc) ** 2 + (z - Zc) ** 2), 0.01)
                    psi[i, j] += current * r / dist
        return psi


@pytest.fixture
def solver(tmp_path: Path) -> AnalyticEquilibriumSolver:
    cfg = tmp_path / "test_config.json"
    cfg.write_text("{}")
    return AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernel, verbose=False)


class TestAnalyticEquilibriumSolver:
    def test_calculate_bv_sign(self, solver: AnalyticEquilibriumSolver) -> None:
        Bv = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)
        assert Bv < 0  # Shafranov: vertical field is negative for outward force balance

    def test_calculate_bv_scales_with_current(self, solver: AnalyticEquilibriumSolver) -> None:
        Bv_low = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=5.0)
        Bv_high = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)
        assert abs(Bv_high) > abs(Bv_low)

    def test_rejects_nonpositive_inputs(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError):
            solver.calculate_required_Bv(R_geo=0.0, a_min=2.0, Ip_MA=15.0)
        with pytest.raises(ValueError):
            solver.calculate_required_Bv(R_geo=6.2, a_min=-1.0, Ip_MA=15.0)

    def test_rejects_nonfinite_or_invalid_shafranov_domain(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="R_geo must exceed a_min"):
            solver.calculate_required_Bv(R_geo=1.0, a_min=1.0, Ip_MA=15.0)
        with pytest.raises(ValueError, match="beta_p must be finite and >= 0"):
            solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0, beta_p=float("nan"))
        with pytest.raises(ValueError, match="li must be finite and >= 0"):
            solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0, li=-0.1)

    def test_coil_efficiencies_shape(self, solver: AnalyticEquilibriumSolver) -> None:
        eff = solver.compute_coil_efficiencies(target_R=6.2)
        assert eff.shape == (3,)

    def test_solve_coil_currents_shape(self, solver: AnalyticEquilibriumSolver) -> None:
        currents = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2)
        assert currents.shape == (3,)

    def test_apply_currents_length_mismatch(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            solver.apply_currents(np.array([1.0, 2.0]))  # 2 instead of 3

    def test_apply_currents_rejects_nonfinite_values(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="currents must contain only finite values"):
            solver.apply_currents(np.array([1.0, np.nan, 3.0]))

    def test_apply_currents_updates_config(self, solver: AnalyticEquilibriumSolver) -> None:
        currents = np.array([1.0, 2.0, 3.0])
        solver.apply_currents(currents)
        for i, coil in enumerate(solver.kernel.cfg["coils"]):
            assert coil["current"] == pytest.approx(currents[i])

    def test_bv_shafranov_formula_magnitude(self, solver: AnalyticEquilibriumSolver) -> None:
        """Bv for ITER-like params: |Bv| should be O(0.01-1 T)."""
        Bv = solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)
        assert 0.001 < abs(Bv) < 10.0

    def test_coil_efficiencies_positive(self, solver: AnalyticEquilibriumSolver) -> None:
        eff = solver.compute_coil_efficiencies(target_R=6.2)
        assert np.all(np.isfinite(eff))

    def test_solve_and_apply_round_trip(self, solver: AnalyticEquilibriumSolver) -> None:
        """Solve for coil currents then verify applied currents match."""
        currents = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2)
        solver.apply_currents(currents)
        applied = np.array([c["current"] for c in solver.kernel.cfg["coils"]])
        assert np.allclose(applied, currents)


# ── Zero-coil kernel edge case ────────────────────────────────────


class FakeKernelNoCoils:
    def __init__(self, config_path: str) -> None:
        self.R = cast("npt.NDArray[np.float64]", np.linspace(4.0, 8.5, 10, dtype=np.float64))
        self.Z = cast("npt.NDArray[np.float64]", np.linspace(-4.0, 4.0, 10, dtype=np.float64))
        self.dR = float(self.R[1] - self.R[0])
        self.cfg: dict[str, list[dict[str, Any]]] = {"coils": []}

    def calculate_vacuum_field(self) -> npt.NDArray[np.float64]:
        return np.zeros((10, 10), dtype=np.float64)


class FakeKernelZeroInfluence(FakeKernel):
    def calculate_vacuum_field(self) -> npt.NDArray[np.float64]:
        return np.zeros((len(self.Z), len(self.R)), dtype=np.float64)


class FakeKernelBadFieldShape(FakeKernel):
    def calculate_vacuum_field(self) -> npt.NDArray[np.float64]:
        return np.zeros((len(self.Z), len(self.R) - 1), dtype=np.float64)


class FakeKernelNonfiniteField(FakeKernel):
    def calculate_vacuum_field(self) -> npt.NDArray[np.float64]:
        psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        psi[0, 0] = np.nan
        return psi


class FakeKernelNonmonotonicR(FakeKernel):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.R = self.R.copy()
        self.R[10] = self.R[9]


class FakeKernelNonuniformR(FakeKernel):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.R = self.R.copy()
        self.R[10] += 0.25 * self.dR


class FakeKernelBadSpacing(FakeKernel):
    """Kernel with an invalid finite-difference spacing attribute."""

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.dR = 0.0


class TestComputeCoilEfficienciesEdgeCases:
    def test_rejects_zero_coils(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelNoCoils, verbose=False)
        with pytest.raises(ValueError, match="no coils"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonpositive_target_r(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="target_R must be finite and > 0"):
            solver.compute_coil_efficiencies(target_R=0.0)

    def test_rejects_nonfinite_target_z(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="target_Z must be finite"):
            solver.compute_coil_efficiencies(target_R=6.2, target_Z=float("nan"))

    def test_rejects_target_r_outside_finite_difference_domain(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="target_R must lie inside the kernel R grid interior"):
            solver.compute_coil_efficiencies(target_R=100.0)

    def test_rejects_target_z_outside_kernel_grid(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="target_Z must lie inside the kernel Z grid"):
            solver.compute_coil_efficiencies(target_R=6.2, target_Z=100.0)

    def test_restores_currents_after_efficiency_calc(self, solver: AnalyticEquilibriumSolver) -> None:
        solver.kernel.cfg["coils"][0]["current"] = 7.5
        solver.compute_coil_efficiencies(target_R=6.2)
        assert solver.kernel.cfg["coils"][0]["current"] == 7.5

    def test_rejects_vacuum_field_shape_mismatch(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelBadFieldShape, verbose=False)

        with pytest.raises(ValueError, match="vacuum field shape must match kernel grid"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonfinite_vacuum_field(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelNonfiniteField, verbose=False)

        with pytest.raises(ValueError, match="vacuum field must contain only finite values"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonmonotonic_r_grid(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelNonmonotonicR, verbose=False)

        with pytest.raises(ValueError, match="kernel R grid must be strictly monotonic"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonuniform_r_grid(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelNonuniformR, verbose=False)

        with pytest.raises(ValueError, match="kernel R grid must be uniformly spaced"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_rejects_nonpositive_kernel_spacing(self, tmp_path: Path) -> None:
        """Reject kernels whose explicit radial spacing cannot support derivatives."""
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelBadSpacing, verbose=False)

        with pytest.raises(ValueError, match="Kernel grid spacing dR must be > 0"):
            solver.compute_coil_efficiencies(target_R=6.2)

    def test_verbose_solver_emits_log_records(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Exercise the verbose logging path used by command-line examples."""
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernel, verbose=True)

        with caplog.at_level("INFO", logger="scpn_control.control.analytic_solver"):
            solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)

        assert "SHAFRANOV EQUILIBRIUM CHECK" in caplog.text


# ── Ridge-regularized solve ──────────────────────────────────────


class TestSolveCoilCurrentsRidge:
    def test_ridge_shrinks_norm(self, solver: AnalyticEquilibriumSolver) -> None:
        c_noreg = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=0.0)
        c_ridge = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=50.0)
        assert np.linalg.norm(c_ridge) <= np.linalg.norm(c_noreg) + 1e-12

    def test_negative_ridge_clamped_to_zero(self, solver: AnalyticEquilibriumSolver) -> None:
        c1 = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=0.0)
        c2 = solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=-5.0)
        np.testing.assert_allclose(c1, c2)

    def test_rejects_nonfinite_target_and_ridge(self, solver: AnalyticEquilibriumSolver) -> None:
        with pytest.raises(ValueError, match="target_Bv must be finite"):
            solver.solve_coil_currents(target_Bv=float("inf"), target_R=6.2)
        with pytest.raises(ValueError, match="ridge_lambda must be finite"):
            solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2, ridge_lambda=float("nan"))


class TestSolveCoilCurrentsInfeasibleInfluence:
    def test_rejects_nonzero_target_with_zero_influence(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelZeroInfluence, verbose=False)

        with pytest.raises(ValueError, match="nonzero target_Bv requires nonzero coil influence"):
            solver.solve_coil_currents(target_Bv=-0.1, target_R=6.2)

    def test_zero_target_with_zero_influence_returns_zero_currents(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=FakeKernelZeroInfluence, verbose=False)

        currents = solver.solve_coil_currents(target_Bv=0.0, target_R=6.2)

        np.testing.assert_allclose(currents, np.zeros(3))


# ── apply_and_save ───────────────────────────────────────────────


class TestApplyAndSave:
    def test_writes_valid_json(self, solver: AnalyticEquilibriumSolver, tmp_path: Path) -> None:
        import json

        out = solver.apply_and_save(
            np.array([0.5, -0.3, 1.2]),
            output_path=str(tmp_path / "out.json"),
        )
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["coils"][0]["current"] == 0.5
        assert data["coils"][2]["current"] == 1.2

    def test_preserves_existing_spdx_metadata_and_final_newline(
        self,
        solver: AnalyticEquilibriumSolver,
        tmp_path: Path,
    ) -> None:
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

    def test_default_path_used_when_none(self, solver: AnalyticEquilibriumSolver) -> None:
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
    def test_legacy_config_fallback_requires_explicit_opt_in(self) -> None:
        with pytest.raises(ValueError, match="allow_legacy_config_fallback=True"):
            run_analytic_solver(
                config_path="dummy.json",
                allow_config_fallback=True,
                allow_legacy_config_fallback=False,
                save_config=False,
                verbose=False,
                kernel_factory=FakeKernel,
            )

    def test_smoke(self, tmp_path: Path) -> None:
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

    def test_no_save(self) -> None:
        summary = run_analytic_solver(
            config_path="dummy.json",
            save_config=False,
            verbose=False,
            kernel_factory=FakeKernel,
        )
        assert summary["output_config_path"] is None

    def test_deterministic(self) -> None:
        kwargs: dict[str, Any] = dict(
            config_path="dummy.json",
            save_config=False,
            verbose=False,
            kernel_factory=FakeKernel,
        )
        a = run_analytic_solver(**kwargs)
        b = run_analytic_solver(**kwargs)
        assert a["required_bv_t"] == b["required_bv_t"]
        assert a["coil_current_l2_norm"] == b["coil_current_l2_norm"]

    def test_custom_physics_params(self, tmp_path: Path) -> None:
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


class _BadRGridKernel(FakeKernel):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.R = cast("npt.NDArray[np.float64]", np.linspace(4.0, 8.5, 2, dtype=np.float64))  # fewer than 3 points


class _BadZGridKernel(FakeKernel):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.Z = cast("npt.NDArray[np.float64]", np.array([], dtype=np.float64))  # empty Z grid


def _solver_with(kernel_factory: Callable[[str], Any], tmp_path: Path) -> AnalyticEquilibriumSolver:
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    return AnalyticEquilibriumSolver(str(cfg), kernel_factory=kernel_factory, verbose=False)


def test_compute_coil_efficiencies_rejects_short_r_grid(tmp_path: Path) -> None:
    solver = _solver_with(_BadRGridKernel, tmp_path)
    with pytest.raises(ValueError, match="kernel R grid must be finite with at least 3 points"):
        solver.compute_coil_efficiencies(target_R=6.2, target_Z=0.0)


def test_compute_coil_efficiencies_rejects_empty_z_grid(tmp_path: Path) -> None:
    solver = _solver_with(_BadZGridKernel, tmp_path)
    with pytest.raises(ValueError, match="kernel Z grid must be finite with at least 1 point"):
        solver.compute_coil_efficiencies(target_R=6.2, target_Z=0.0)


def test_solve_coil_currents_rejects_nonfinite_influence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    solver = _solver_with(FakeKernel, tmp_path)

    def nonfinite_efficiencies(*args: Any, **kwargs: Any) -> npt.NDArray[np.float64]:
        return np.array([np.nan, 0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(
        solver,
        "compute_coil_efficiencies",
        nonfinite_efficiencies,
    )
    with pytest.raises(ValueError, match="coil influence matrix must contain only finite values"):
        solver.solve_coil_currents(target_Bv=-1.0, target_R=6.2)


def _patch_config_existence(monkeypatch: pytest.MonkeyPatch, *, preferred: bool, fallback: bool) -> None:
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        name = str(self)
        if name.endswith("iter_genetic_temp.json"):
            return preferred
        if name.endswith("iter_validated_config.json"):
            return fallback
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)


def test_run_resolves_preferred_default_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_config_existence(monkeypatch, preferred=True, fallback=False)
    summary = run_analytic_solver(
        config_path=None,
        save_config=False,
        verbose=False,
        kernel_factory=FakeKernel,
    )
    assert np.isfinite(summary["required_bv_t"])


def test_run_uses_legacy_fallback_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_config_existence(monkeypatch, preferred=False, fallback=True)
    summary = run_analytic_solver(
        config_path=None,
        allow_config_fallback=True,
        allow_legacy_config_fallback=True,
        save_config=False,
        verbose=False,
        kernel_factory=FakeKernel,
    )
    assert np.isfinite(summary["required_bv_t"])


def test_run_raises_when_enabled_fallback_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_config_existence(monkeypatch, preferred=False, fallback=False)
    with pytest.raises(FileNotFoundError, match="no validated config exists"):
        run_analytic_solver(
            config_path=None,
            allow_config_fallback=True,
            allow_legacy_config_fallback=True,
            save_config=False,
            verbose=False,
            kernel_factory=FakeKernel,
        )


def test_run_raises_when_default_config_missing_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_config_existence(monkeypatch, preferred=False, fallback=False)
    with pytest.raises(FileNotFoundError, match="Default analytic configuration is missing"):
        run_analytic_solver(
            config_path=None,
            allow_config_fallback=False,
            save_config=False,
            verbose=False,
            kernel_factory=FakeKernel,
        )


def test_module_import_guard_reports_missing_fusion_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute the module import guard when both FusionKernel providers are absent."""
    module_path = Path(__file__).parents[1] / "src" / "scpn_control" / "control" / "analytic_solver.py"
    spec = importlib.util.spec_from_file_location("analytic_solver_import_guard_probe", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = ModuleType("analytic_solver_import_guard_probe")
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name in {"scpn_control.core._rust_compat", "scpn_control.core.fusion_kernel"}:
            raise ImportError(f"blocked test import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setitem(sys.modules, "analytic_solver_import_guard_probe", module)

    with pytest.raises(ImportError, match="Unable to import FusionKernel"):
        spec.loader.exec_module(module)
