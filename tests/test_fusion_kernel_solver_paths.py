# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fusion kernel config validators, accelerator and solver branches

"""Config-validator, accelerator-bridge and solver-path branches of the fusion kernel.

Drives the Pydantic domain/physics/grid validators, the HPC and Rust multigrid
acceleration bridges (via stubbed backends), the Anderson mixing fallbacks, the
bilinear prolongation boundary clipping, the free-boundary objective-tolerance
resolution and status evaluation, and the Picard/Newton convergence breaks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import scpn_control.core._rust_compat as rust_compat
import scpn_control.core.fusion_kernel as fk
from scpn_control.core.fusion_kernel import (
    DimensionsConfig,
    FusionKernel,
    FusionKernelConfig,
    PhysicsConfig,
)


def _config_dict(*, grid: tuple[int, int] = (12, 12), method: str = "sor", **solver: Any) -> dict[str, Any]:
    base_solver = {
        "max_iterations": 6,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.15,
        "solver_method": method,
    }
    base_solver.update(solver)
    return {
        "reactor_name": "Test-Reactor",
        "grid_resolution": list(grid),
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": base_solver,
    }


def _write_config(path: Path, **kwargs: Any) -> Path:
    path.write_text(json.dumps(_config_dict(**kwargs)), encoding="utf-8")
    return path


def _kernel(tmp_path: Path, **kwargs: Any) -> FusionKernel:
    return FusionKernel(_write_config(tmp_path / "cfg.json", **kwargs))


# ── Pydantic config validators ────────────────────────────────────────


def test_dimensions_reject_non_finite_bound() -> None:
    with pytest.raises(ValidationError, match="dimension bounds must be finite"):
        DimensionsConfig(R_min=float("inf"), R_max=6.0, Z_min=-3.0, Z_max=3.0)


def test_physics_accepts_explicit_none_optionals() -> None:
    # the None short-circuit branches of the sign / permeability validators
    physics = PhysicsConfig(plasma_current_target=1.0, plasma_current_sign=None, vacuum_permeability=None)
    assert physics.plasma_current_sign is None
    assert physics.vacuum_permeability is None


def test_physics_rejects_invalid_current_sign() -> None:
    with pytest.raises(ValidationError, match="plasma_current_sign must be -1.0 or 1.0"):
        PhysicsConfig(plasma_current_target=1.0, plasma_current_sign=2.0)


def test_physics_rejects_non_positive_permeability() -> None:
    with pytest.raises(ValidationError, match="vacuum_permeability must be positive finite"):
        PhysicsConfig(plasma_current_target=1.0, vacuum_permeability=0.0)


def test_config_rejects_blank_reactor_name() -> None:
    with pytest.raises(ValidationError, match="non-empty 'reactor_name'"):
        FusionKernelConfig(
            reactor_name="   ",
            dimensions={"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
            grid_resolution=(8, 8),
            physics={"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        )


def test_config_rejects_degenerate_grid_resolution() -> None:
    with pytest.raises(ValidationError, match="grid_resolution entries must be integers >= 3"):
        FusionKernelConfig(
            reactor_name="r",
            dimensions={"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
            grid_resolution=(2, 8),
            physics={"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        )


# ── free-boundary objective tolerance resolution ──────────────────────


def test_resolve_objective_tolerances_merges_and_validates() -> None:
    merged = FusionKernel._resolve_free_boundary_objective_tolerances({"shape_rms": 0.01}, {"x_point_position": 0.02})
    assert merged == {"shape_rms": 0.01, "x_point_position": 0.02}


@pytest.mark.parametrize(
    ("cfg_tol", "match"),
    [
        ("not-a-mapping", "must be a mapping of tolerance names"),
        ({"bogus": 1.0}, "Unknown free_boundary.objective_tolerances key"),
        ({"shape_rms": -1.0}, "must be finite and >= 0"),
    ],
)
def test_resolve_objective_tolerances_rejects_malformed(cfg_tol: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FusionKernel._resolve_free_boundary_objective_tolerances(cfg_tol)


# ── free-boundary objective status evaluation ─────────────────────────


def test_evaluate_objective_status_checks_every_configured_tolerance() -> None:
    tolerances = {
        "shape_rms": 0.1,
        "shape_max_abs": 0.1,
        "x_point_position": 0.1,
        "x_point_gradient": 0.1,
        "x_point_flux": 0.1,
        "divertor_rms": 0.1,
        "divertor_max_abs": 0.1,
    }
    status = FusionKernel._evaluate_free_boundary_objective_status(
        tolerances,
        shape_error_rms=0.01,
        shape_error_max_abs=0.01,
        x_point_detected_error=0.01,
        x_point_gradient_norm=float("inf"),  # exercises the non-finite metric guard
        x_point_flux_error=0.01,
        divertor_error_rms=0.01,
        divertor_error_max_abs=0.01,
    )
    assert status["objective_convergence_active"] is True
    assert status["objective_checks"]["shape_rms"] is True
    assert status["objective_checks"]["x_point_gradient"] is False  # inf metric fails the finite guard
    assert status["objective_converged"] is False


def test_evaluate_objective_status_is_vacuously_converged_without_metrics() -> None:
    status = FusionKernel._evaluate_free_boundary_objective_status(
        {"shape_rms": 0.1},
        shape_error_rms=None,
        shape_error_max_abs=None,
        x_point_detected_error=None,
        x_point_gradient_norm=None,
        x_point_flux_error=None,
        divertor_error_rms=None,
        divertor_error_max_abs=None,
    )
    assert status["objective_convergence_active"] is False
    assert status["objective_converged"] is True


# ── Anderson mixing fallbacks ─────────────────────────────────────────


def test_anderson_step_falls_back_with_insufficient_history(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    psi = np.ones((4, 4))
    result = kernel._anderson_step([psi], [psi * 0.1], m=5)
    assert np.array_equal(result, psi)


def test_anderson_step_mixes_well_conditioned_history(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    rng = np.random.default_rng(0)
    psi_history = [rng.standard_normal((4, 4)) for _ in range(4)]
    res_history = [rng.standard_normal((4, 4)) * 0.1 for _ in range(4)]
    mixed = kernel._anderson_step(psi_history, res_history, m=5)
    assert mixed.shape == (4, 4)
    assert np.all(np.isfinite(mixed))


# ── divertor configuration labelling ──────────────────────────────────


@pytest.mark.parametrize(
    ("points", "label"),
    [
        (None, "none"),
        (np.empty((0, 2)), "none"),
        (np.array([[6.0, -3.0]]), "single_strike"),
        (np.array([[6.0, -3.0], [6.0, 3.0]]), "double_strike"),
        (np.array([[6.0, -3.0], [6.0, 3.0], [5.5, 0.0]]), "multi_strike"),
    ],
)
def test_divertor_configuration_label(points: Any, label: str) -> None:
    assert FusionKernel._divertor_configuration_label(points) == label


# ── bilinear prolongation boundary clipping ───────────────────────────


def test_prolongate_bilinear_clips_out_of_range_fine_indices() -> None:
    coarse = np.arange(9, dtype=np.float64).reshape(3, 3)
    fine = FusionKernel._prolongate_bilinear(coarse, 3, 3)
    assert fine.shape == (3, 3)
    assert fine[0, 0] == coarse[0, 0]


# ── shape error metrics ───────────────────────────────────────────────


def test_shape_error_metrics_handles_empty_targets() -> None:
    metrics = FusionKernel._shape_error_metrics(np.array([]), np.array([]))
    assert metrics == {"shape_error_rms": 0.0, "shape_error_max_abs": 0.0}


# ── HPC acceleration bridge ───────────────────────────────────────────


class _FakeHPC:
    def is_available(self) -> bool:
        return True

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        return None

    def solve(self, j_phi: np.ndarray, iterations: int = 50) -> np.ndarray:
        return np.zeros_like(j_phi)


def test_hpc_bridge_is_initialised_and_drives_elliptic_solve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fk, "HPCBridge", _FakeHPC)
    kernel = _kernel(tmp_path)
    assert kernel.hpc.is_available() is True
    source = np.zeros_like(kernel.J_phi)
    psi_bc = np.zeros_like(kernel.Psi)
    result = kernel._elliptic_solve(source, psi_bc)
    assert result.shape == kernel.Psi.shape


# ── Rust multigrid backend ────────────────────────────────────────────


class _FakeRustResult:
    converged = True
    residual = 1e-6
    iterations = 7


class _FakeRustKernel:
    def __init__(self, config_path: Any) -> None:
        self._path = config_path
        self.Psi = np.zeros((12, 12))
        self.J_phi = np.zeros((12, 12))
        self.B_R = np.zeros((12, 12))
        self.B_Z = np.zeros((12, 12))

    def set_solver_method(self, method: str) -> None:
        self._method = method

    def solve_equilibrium(self) -> _FakeRustResult:
        return _FakeRustResult()


def test_rust_multigrid_backend_drives_full_solve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_compat, "_rust_available", lambda: True)
    monkeypatch.setattr(rust_compat, "RustAcceleratedKernel", _FakeRustKernel)
    kernel = _kernel(tmp_path, method="rust_multigrid")
    result = kernel.solve_equilibrium()
    assert result["converged"] is True
    assert result["iterations"] == 7


# ── Newton / Picard solver guards and convergence ─────────────────────


def test_solver_rejects_non_positive_gs_residual_threshold(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, method="newton", require_gs_residual=True, gs_residual_threshold=-1.0)
    with pytest.raises(ValueError, match="gs_residual_threshold must be > 0"):
        kernel.solve_equilibrium()


def test_picard_warmup_converges_on_loose_tolerance(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path, method="newton", convergence_threshold=1e9)
    result = kernel.solve_equilibrium()
    assert result["converged"] is True


def test_newton_phase_converges_when_warmup_is_skipped(tmp_path: Path) -> None:
    # max_iterations=1 makes warmup_steps == 0, so convergence is reached in the
    # Newton phase on the first iterate under a loose tolerance.
    kernel = _kernel(tmp_path, method="newton", max_iterations=1, convergence_threshold=1e9)
    result = kernel.solve_equilibrium()
    assert result["converged"] is True
