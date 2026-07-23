# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for free-boundary control helpers

"""Drive production free-boundary optim/objective helpers on real surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scpn_control.core.gs_free_boundary_control as fb
from scpn_control.core.fusion_kernel import CoilSet, FusionKernel


def _write_config(path: Path) -> Path:
    raw = {
        "reactor_name": "FB-Control-Test",
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "grid_resolution": [12, 12],
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "solver": {
            "boundary_variant": "fixed",
            "solver_method": "sor",
            "max_iterations": 10,
            "convergence_threshold": 1e-4,
        },
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0, "turns": 10},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0, "turns": 10},
            {"name": "PF3", "r": 4.0, "z": 0.0, "current": 1.0, "turns": 10},
        ],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _coilset_with_targets() -> CoilSet:
    return CoilSet(
        positions=[(3.0, 2.0), (3.5, -2.0), (4.0, 2.0)],
        currents=np.ones(3) * 1e4,
        turns=[10, 10, 10],
        current_limits=np.ones(3) * 5e4,
        target_flux_points=np.array([[3.5, 0.0], [4.0, 0.5], [4.5, -0.5]], dtype=float),
    )


def test_coilset_reexported_from_fusion_kernel() -> None:
    """Public CoilSet identity is the free-boundary control leaf type."""
    assert CoilSet is fb.CoilSet


def test_shape_error_metrics_empty_and_nonempty() -> None:
    """Leaf shape metrics handle empty residuals and finite nonzero residuals."""
    empty = fb.shape_error_metrics(np.array([]), np.array([]))
    assert empty == {"shape_error_rms": 0.0, "shape_error_max_abs": 0.0}
    metrics = fb.shape_error_metrics(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert metrics["shape_error_max_abs"] == pytest.approx(2.0)
    assert metrics["shape_error_rms"] == pytest.approx(np.sqrt(2.0))
    owner = FusionKernel._shape_error_metrics(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert owner == metrics


def test_objective_tolerances_fail_closed() -> None:
    """Malformed objective tolerances raise ValueError on the production path."""
    with pytest.raises(ValueError, match="must be a mapping"):
        fb.resolve_free_boundary_objective_tolerances("not-a-mapping")
    with pytest.raises(ValueError, match="Unknown free_boundary.objective_tolerances key"):
        fb.resolve_free_boundary_objective_tolerances({"bogus": 1.0})
    with pytest.raises(ValueError, match="must be finite and >= 0"):
        fb.resolve_free_boundary_objective_tolerances({"shape_rms": -1.0})
    merged = fb.resolve_free_boundary_objective_tolerances(
        {"shape_rms": 0.2},
        {"shape_max_abs": 0.1},
    )
    assert merged == {"shape_rms": 0.2, "shape_max_abs": 0.1}
    assert FusionKernel._resolve_free_boundary_objective_tolerances({"shape_rms": 0.2}) == {"shape_rms": 0.2}


def test_objective_status_and_divertor_label_match_owner() -> None:
    """Owner wrappers are the production leaf objective evaluators."""
    tolerances = {"shape_rms": 0.1, "x_point_gradient": 0.1}
    leaf = fb.evaluate_free_boundary_objective_status(
        tolerances,
        shape_error_rms=0.01,
        shape_error_max_abs=None,
        x_point_detected_error=None,
        x_point_gradient_norm=float("inf"),
        x_point_flux_error=None,
        divertor_error_rms=None,
        divertor_error_max_abs=None,
    )
    owner = FusionKernel._evaluate_free_boundary_objective_status(
        tolerances,
        shape_error_rms=0.01,
        shape_error_max_abs=None,
        x_point_detected_error=None,
        x_point_gradient_norm=float("inf"),
        x_point_flux_error=None,
        divertor_error_rms=None,
        divertor_error_max_abs=None,
    )
    assert owner == leaf
    assert leaf["objective_checks"]["shape_rms"] is True
    assert leaf["objective_checks"]["x_point_gradient"] is False
    assert leaf["objective_converged"] is False
    assert fb.divertor_configuration_label(None) == "none"
    assert FusionKernel._divertor_configuration_label(np.array([[1.0, -1.0], [2.0, -1.0]])) == "double_strike"


def test_owner_optimize_coil_currents_matches_leaf(tmp_path: Path) -> None:
    """FusionKernel coil optim wrapper matches the pure leaf on a real kernel."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg.json"))
    coils = _coilset_with_targets()
    target = np.array([0.1, 0.2, 0.15], dtype=float)
    owner = kernel.optimize_coil_currents(coils, target, tikhonov_alpha=1e-3)
    leaf = fb.optimize_coil_currents(
        coils,
        target,
        build_mutual_inductance_matrix=kernel._build_mutual_inductance_matrix,
        coil_flux_response_at_point=kernel._coil_flux_response_at_point,
        coil_flux_gradient_response=kernel._coil_flux_gradient_response,
        tikhonov_alpha=1e-3,
    )
    np.testing.assert_allclose(owner, leaf, rtol=1.0e-12, atol=1.0e-14)
    assert owner.shape == (3,)
    assert np.all(np.isfinite(owner))
    assert np.all(np.abs(owner) <= coils.current_limits + 1.0e-9)


def test_optimize_coil_currents_fail_closed_without_targets(tmp_path: Path) -> None:
    """Production optim fails closed when no free-boundary targets are set."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg2.json"))
    coils = CoilSet(positions=[(3.0, 0.0)], currents=np.array([1.0]), turns=[1])
    with pytest.raises(ValueError, match="target_flux requires CoilSet.target_flux_points"):
        kernel.optimize_coil_currents(coils, np.array([0.1]))
    with pytest.raises(ValueError, match="At least one free-boundary optimisation target"):
        fb.optimize_coil_currents(
            coils,
            np.array([], dtype=float),
            build_mutual_inductance_matrix=kernel._build_mutual_inductance_matrix,
            coil_flux_response_at_point=kernel._coil_flux_response_at_point,
            coil_flux_gradient_response=kernel._coil_flux_gradient_response,
        )


def test_estimate_point_gradient_central_difference() -> None:
    """Finite-difference gradient recovers known linear field slopes."""

    def sample(r: float, z: float) -> float:
        return 2.0 * r - 3.0 * z

    d_dr, d_dz = fb.estimate_point_gradient(
        sample,
        1.0,
        0.0,
        r_min=0.0,
        r_max=2.0,
        z_min=-1.0,
        z_max=1.0,
        dR=0.1,
        dZ=0.1,
    )
    assert float(d_dr) == pytest.approx(2.0, rel=1e-9)
    assert float(d_dz) == pytest.approx(-3.0, rel=1e-9)


def test_resolve_shape_and_separatrix_targets() -> None:
    """Shape and separatrix target resolvers preserve explicit vs tracking modes."""
    coils = _coilset_with_targets()
    coils.target_flux_values = np.array([0.1, 0.2, 0.15], dtype=float)
    current = np.array([0.0, 0.0, 0.0], dtype=float)
    target, mode = fb.resolve_shape_target_flux(coils, current)
    assert mode == "explicit_target"
    np.testing.assert_allclose(target, coils.target_flux_values)
    sep = fb.resolve_separatrix_flux_target(coils, target)
    assert sep == pytest.approx(float(np.mean(coils.target_flux_values)))
