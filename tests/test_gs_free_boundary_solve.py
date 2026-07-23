# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for free-boundary solve orchestration

"""Drive production free-boundary solve orchestration on real surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scpn_control.core.gs_free_boundary_solve as fb_solve
from scpn_control.core.fusion_kernel import CoilSet, FusionKernel


def _write_config(path: Path) -> Path:
    raw = {
        "reactor_name": "FB-Solve-Test",
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "grid_resolution": [12, 12],
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "solver": {
            "boundary_variant": "fixed",
            "solver_method": "sor",
            "max_iterations": 10,
            "convergence_threshold": 1e-4,
            "relaxation_factor": 0.15,
        },
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0, "turns": 10},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0, "turns": 10},
        ],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _coilset(*, with_targets: bool = False) -> CoilSet:
    cs = CoilSet(
        positions=[(3.0, 2.0), (5.0, -2.0)],
        currents=np.array([1.0e4, -1.0e4], dtype=float),
        turns=[10, 10],
    )
    if with_targets:
        cs.current_limits = np.ones(2) * 5e4
        cs.target_flux_points = np.array([[3.5, 0.0], [4.0, 0.5], [4.5, -0.5]], dtype=float)
        cs.target_flux_values = np.array([0.10, 0.20, 0.15], dtype=float)
    return cs


def test_owner_solve_free_boundary_matches_leaf(tmp_path: Path) -> None:
    """FusionKernel free-boundary solve wrapper is the production leaf driver."""
    kernel_owner = FusionKernel(_write_config(tmp_path / "owner.json"))
    kernel_leaf = FusionKernel(_write_config(tmp_path / "leaf.json"))
    coils_owner = _coilset()
    coils_leaf = _coilset()
    owner = kernel_owner.solve_free_boundary(coils_owner, max_outer_iter=3, tol=1e-2)
    leaf = fb_solve.solve_free_boundary(kernel_leaf, coils_leaf, max_outer_iter=3, tol=1e-2)
    assert owner["outer_iterations"] == leaf["outer_iterations"]
    assert owner["boundary_variant"] == "free_boundary"
    assert leaf["boundary_variant"] == "free_boundary"
    np.testing.assert_allclose(owner["final_diff"], leaf["final_diff"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(owner["coil_currents"], leaf["coil_currents"], rtol=1e-12, atol=1e-14)
    assert np.all(np.isfinite(owner["coil_currents"]))
    assert owner["outer_iterations"] <= 3


def test_solve_free_boundary_with_shape_optim_returns_metrics(tmp_path: Path) -> None:
    """Shape-optim path reports structured finite objective metrics."""
    kernel = FusionKernel(_write_config(tmp_path / "shape.json"))
    coils = _coilset(with_targets=True)
    result = kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1e-2,
        optimize_shape=True,
        tikhonov_alpha=1e-3,
    )
    assert result["shape_objective_mode"] == "explicit_target"
    assert result["shape_error_final_rms"] is not None
    assert np.isfinite(result["shape_error_final_rms"])
    assert result["coil_currents"].shape == (2,)
    assert "objective_checks" in result


def test_solve_free_boundary_fail_closed_malformed_objective_tolerances(tmp_path: Path) -> None:
    """Production free-boundary solve fails closed on malformed objective tolerances."""
    kernel = FusionKernel(_write_config(tmp_path / "bad_tol.json"))
    coils = _coilset()
    with pytest.raises(ValueError, match="must be a mapping"):
        kernel.solve_free_boundary(
            coils,
            max_outer_iter=1,
            objective_tolerances="not-a-mapping",  # type: ignore[arg-type]
        )


def test_kernel_satisfies_free_boundary_solve_protocol(tmp_path: Path) -> None:
    """Live FusionKernel implements the leaf Protocol surface."""
    kernel = FusionKernel(_write_config(tmp_path / "proto.json"))
    assert isinstance(kernel, fb_solve.FreeBoundarySolveKernel)
