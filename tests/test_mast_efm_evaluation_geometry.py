# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public MAST flux geometry tests.

"""Exercise diagnostic axis/contour geometry through the owning public flux API."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.mast_efm_evaluation_geometry import evaluate_flux_geometry


def geometry_reference() -> tuple[NDArray[np.float64], dict[str, NDArray[Any]]]:
    """Provide an analytic axis and circular boundary sampled on a real two-dimensional metre grid."""
    r = np.linspace(0.4, 1.0, 21)
    z = np.linspace(-0.3, 0.3, 21)
    rr, zz = np.meshgrid(r, z)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    return ((rr - 0.7) ** 2 + zz**2)[None], {
        "r_grid_m": r,
        "z_grid_m": z,
        "psi_axis_Wb_per_rad": np.array([0.0]),
        "psi_boundary_Wb_per_rad": np.array([0.18**2]),
        "magnetic_axis_r_m": np.array([0.7]),
        "magnetic_axis_z_m": np.array([0.0]),
        "lcfs_r_m": 0.7 + 0.18 * np.cos(angles),
        "lcfs_z_m": 0.18 * np.sin(angles),
    }


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("inverted", [False, True])
def test_source_grids_and_flux_orientation_preserve_geometry(descending: bool, inverted: bool) -> None:
    """Ascending/descending coordinates and both axis-to-boundary flux signs recover the same physical axis."""
    psi, data = geometry_reference()
    if descending:
        data["r_grid_m"] = data["r_grid_m"][::-1]
        data["z_grid_m"] = data["z_grid_m"][::-1]
        psi = psi[:, ::-1, ::-1]
    if inverted:
        psi = -psi
        data["psi_boundary_Wb_per_rad"] = -data["psi_boundary_Wb_per_rad"]
    metrics, arrays = evaluate_flux_geometry(psi, data)
    assert metrics["magnetic_axis_rmse_m"] < 1e-12
    assert metrics["boundary_mean_distance_m"] < 0.02
    assert metrics["derived_lcfs_success_count"] == 1
    np.testing.assert_allclose(arrays["derived_magnetic_axis_r_m"], [0.7])


def test_inferred_coordinate_envelope_is_explicitly_diagnostic() -> None:
    """An absent source grid uses the padded reference envelope and records its inference provenance."""
    psi, data = geometry_reference()
    data.pop("r_grid_m")
    data.pop("z_grid_m")
    metrics, _ = evaluate_flux_geometry(psi, data)
    assert metrics["coordinate_grid_provenance"] == "inferred: reference LCFS envelope"


@pytest.mark.parametrize("key", ["r_grid_m", "z_grid_m"])
@pytest.mark.parametrize("bad", [np.ones(1), np.ones((2, 2)), np.array([0.0, np.nan]), np.array([0.0, 0.0])])
def test_invalid_source_coordinate_grid_refuses(key: str, bad: NDArray[Any]) -> None:
    """Malformed source grids refuse instead of silently switching to inferred coordinates."""
    psi, data = geometry_reference()
    data[key] = bad
    with pytest.raises(ValueError):
        evaluate_flux_geometry(psi, data)


@pytest.mark.parametrize("missing", ["r_grid_m", "z_grid_m"])
def test_partial_source_grid_refuses(missing: str) -> None:
    """A provided coordinate direction requires its paired source grid."""
    psi, data = geometry_reference()
    data.pop(missing)
    with pytest.raises(ValueError, match="supplied together"):
        evaluate_flux_geometry(psi, data)


def test_source_grid_dimensions_match_prediction() -> None:
    """Prediction shape and coordinate lengths cannot silently describe different flux samples."""
    psi, data = geometry_reference()
    data["r_grid_m"] = np.linspace(0.4, 1.0, 20)
    with pytest.raises(ValueError, match="lengths must match"):
        evaluate_flux_geometry(psi, data)


@pytest.mark.parametrize("bad", [np.zeros((3, 4)), np.full((1, 21, 21), np.nan)])
def test_invalid_prediction_refuses(bad: NDArray[Any]) -> None:
    """Malformed or nonfinite predictions cannot produce an axis/boundary diagnostic."""
    _, data = geometry_reference()
    with pytest.raises(ValueError):
        evaluate_flux_geometry(bad, data)


def test_missing_boundary_and_axis_observations_produce_null_metrics() -> None:
    """No contour or finite observed axis yields explicit null residual metrics rather than perfect scores."""
    psi, data = geometry_reference()
    data.pop("magnetic_axis_r_m")
    data.pop("magnetic_axis_z_m")
    data["psi_boundary_Wb_per_rad"] = np.array(10.0)
    data["lcfs_valid_mask"] = np.zeros_like(data["lcfs_r_m"], dtype=bool)
    metrics, arrays = evaluate_flux_geometry(psi, data)
    assert metrics["magnetic_axis_rmse_m"] is None
    assert metrics["boundary_mean_distance_m"] is None
    assert metrics["boundary_p95_distance_m"] is None
    assert metrics["derived_lcfs_success_count"] == 0
    np.testing.assert_array_equal(arrays["derived_lcfs_point_count"], [0])


def test_flat_flux_cannot_prove_a_boundary_contour() -> None:
    """A constant field at the selected level has no unique interpolated edge crossings."""
    psi, data = geometry_reference()
    data["psi_boundary_Wb_per_rad"] = np.array([0.0])
    metrics, _ = evaluate_flux_geometry(np.zeros_like(psi), data)
    assert metrics["derived_lcfs_success_count"] == 0


def test_inferred_grid_requires_finite_geometry() -> None:
    """A coordinate-free bundle with insufficient LCFS observations refuses inferred physical coordinates."""
    psi, data = geometry_reference()
    data.pop("r_grid_m")
    data.pop("z_grid_m")
    data["lcfs_r_m"] = np.full(64, np.nan)
    with pytest.raises(ValueError, match="coordinate inference requires"):
        evaluate_flux_geometry(psi, data)


def test_axis_observation_count_must_match_prediction_rows() -> None:
    """Misaligned axis observations cannot silently enter per-equilibrium metric comparisons."""
    psi, data = geometry_reference()
    data["magnetic_axis_r_m"] = np.array([0.7, 0.8])
    with pytest.raises(ValueError, match="per-equilibrium values"):
        evaluate_flux_geometry(psi, data)


def test_representable_extreme_linear_flux_has_correct_boundary_coordinates() -> None:
    """Finite endpoint fluxes with an overflowing difference still place the zero contour at the physical midpoint."""
    psi = np.array([[[-1e308, 1e308], [-1e308, 1e308]]])
    data = {
        "r_grid_m": np.array([0.4, 1.0]),
        "z_grid_m": np.array([-0.3, 0.3]),
        "psi_axis_Wb_per_rad": np.array([-1e308]),
        "psi_boundary_Wb_per_rad": np.array([0.0]),
        "magnetic_axis_r_m": np.array([0.4]),
        "magnetic_axis_z_m": np.array([-0.3]),
        "lcfs_r_m": np.full(3, 0.7),
        "lcfs_z_m": np.array([-0.3, 0.0, 0.3]),
    }
    metrics, arrays = evaluate_flux_geometry(psi, data)
    assert metrics["boundary_mean_distance_m"] < 1e-12
    assert metrics["boundary_p95_distance_m"] < 1e-12
    np.testing.assert_array_equal(arrays["derived_lcfs_point_count"], [2])
