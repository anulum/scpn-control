# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for GS green / vacuum helpers

"""Drive production Green's function and vacuum helpers on real surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scpn_control.core.gs_green_vacuum as green
from scpn_control.core.fusion_kernel import FusionKernel


def _write_config(path: Path) -> Path:
    raw = {
        "reactor_name": "Green-Test",
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "grid_resolution": [12, 12],
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "solver": {"boundary_variant": "fixed"},
        "coils": [
            {"r": 3.2, "z": 0.5, "current": 1.0e5, "turns": 1},
            {"r": 3.2, "z": -0.5, "current": 1.0e5, "turns": 2},
        ],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_green_function_singular_denom_guard_returns_zero() -> None:
    """Near-singular source/observer geometry hits the production denom guard."""
    assert green.green_function(0.0, 0.0, 0.0, 0.0) == 0.0
    assert FusionKernel._green_function(0.0, 0.0, 0.0, 0.0) == 0.0
    # Non-singular off-axis evaluation remains finite and non-zero.
    psi = green.green_function(2.0, 0.0, 1.5, 0.2)
    assert np.isfinite(psi)
    assert psi != 0.0


def test_green_function_array_matches_scalar() -> None:
    """Vectorised Green matches scalar evaluations at the same observers."""
    r_src, z_src = 2.0, 0.1
    r_obs = np.array([1.2, 1.8, 2.4])
    z_obs = np.array([0.0, 0.2, -0.3])
    arr = green.green_function_array(r_src, z_src, r_obs, z_obs)
    assert arr.shape == r_obs.shape
    for r, z, expected in zip(r_obs, z_obs, arr):
        assert green.green_function(r_src, z_src, float(r), float(z)) == pytest.approx(
            float(expected), rel=1.0e-12, abs=1.0e-14
        )
    owner = FusionKernel._green_function_array(r_src, z_src, r_obs, z_obs)
    np.testing.assert_allclose(owner, arr, rtol=1.0e-12, atol=1.0e-14)


def test_vacuum_field_via_owner_matches_leaf(tmp_path: Path) -> None:
    """FusionKernel vacuum path is the leaf production function."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg.json"))
    leaf_psi = green.vacuum_poloidal_flux(
        kernel.RR,
        kernel.ZZ,
        kernel.cfg["coils"],
        float(kernel.cfg["physics"].get("vacuum_permeability", 1.0)),
    )
    owner_psi = kernel.calculate_vacuum_field()
    np.testing.assert_allclose(owner_psi, leaf_psi, rtol=1.0e-12, atol=1.0e-14)
    assert np.all(np.isfinite(owner_psi))
    assert owner_psi.shape == (kernel.NZ, kernel.NR)


def test_mutual_matrix_shape_and_fail_closed() -> None:
    """Mutual matrix has (n_coils, n_pts); bad obs shape fails closed."""
    coils = SimpleNamespace(
        positions=[(2.0, 0.5), (2.1, -0.5)],
        turns=[1, 3],
        currents=[1.0, 1.0],
    )
    obs = np.array([[1.5, 0.0], [1.7, 0.1], [1.9, -0.2]])
    matrix = green.build_mutual_inductance_matrix(coils, obs)
    assert matrix.shape == (2, 3)
    assert np.all(np.isfinite(matrix))
    # turns scale the second coil response by 3 relative to unit-turn same geometry.
    unit_turns = SimpleNamespace(positions=coils.positions, turns=[1, 1], currents=coils.currents)
    unit = green.build_mutual_inductance_matrix(unit_turns, obs)
    np.testing.assert_allclose(matrix[1], 3.0 * unit[1], rtol=1.0e-12, atol=1.0e-14)
    with pytest.raises(ValueError, match="obs_points"):
        green.build_mutual_inductance_matrix(coils, np.array([1.0, 2.0, 3.0]))


def test_external_flux_from_coilset_finite() -> None:
    """External flux assembly returns a finite mesh-shaped field."""
    r = np.linspace(1.0, 3.0, 8)
    z = np.linspace(-1.0, 1.0, 10)
    coils = SimpleNamespace(
        positions=[(3.2, 0.4)],
        currents=[5.0e4],
        turns=[2],
    )
    psi = green.external_flux_from_coilset(r, z, coils)
    assert psi.shape == (10, 8)
    assert np.all(np.isfinite(psi))
    # Non-zero current should produce non-trivial flux somewhere on the mesh.
    assert float(np.max(np.abs(psi))) > 0.0
