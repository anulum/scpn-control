# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for GS multigrid primitives

"""Drive production multigrid restrict/prolongate/smooth/V-cycle helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scpn_control.core.gs_multigrid as mg
from scpn_control.core.fusion_kernel import FusionKernel


def _write_config(path: Path) -> Path:
    raw = {
        "reactor_name": "MG-Test",
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "grid_resolution": [17, 17],
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "solver": {"boundary_variant": "fixed", "solver_method": "multigrid"},
        "coils": [{"r": 3.2, "z": 0.0, "current": 1.0e5, "turns": 1}],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_owner_restrict_prolongate_bind_to_leaf() -> None:
    """FusionKernel static multigrid transfers are the production leaf functions."""
    fine = np.arange(81, dtype=float).reshape(9, 9)
    leaf_coarse = mg.restrict_full_weight(fine)
    owner_coarse = FusionKernel._restrict_full_weight(fine)
    np.testing.assert_allclose(owner_coarse, leaf_coarse, rtol=1.0e-14, atol=1.0e-14)
    leaf_fine = mg.prolongate_bilinear(leaf_coarse, 9, 9)
    owner_fine = FusionKernel._prolongate_bilinear(owner_coarse, 9, 9)
    np.testing.assert_allclose(owner_fine, leaf_fine, rtol=1.0e-14, atol=1.0e-14)


def test_restrict_full_weight_rectangular_odd_grid_preserves_edges() -> None:
    """Rectangular odd grids keep edge injection extents separate (production path)."""
    fine = np.arange(7 * 9, dtype=float).reshape(7, 9)
    coarse = mg.restrict_full_weight(fine)
    assert coarse.shape == (4, 5)
    np.testing.assert_allclose(coarse[0, :], fine[0, ::2][:5])
    np.testing.assert_allclose(coarse[-1, :], fine[-1, ::2][:5])
    np.testing.assert_allclose(coarse[:, 0], fine[::2, 0][:4])
    np.testing.assert_allclose(coarse[:, -1], fine[::2, -1][:4])


def test_vcycle_reduces_residual_via_owner(tmp_path: Path) -> None:
    """One owner V-cycle reduces GS* residual on a real FusionKernel grid."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg.json"))
    # Smooth interior source with Dirichlet-zero boundary state.
    psi = np.zeros_like(kernel.Psi)
    source = np.zeros_like(kernel.Psi)
    source[1:-1, 1:-1] = 1.0
    r_before = kernel._mg_residual(psi, source, kernel.RR, kernel.dR, kernel.dZ)
    rms_before = float(np.sqrt(np.mean(r_before[1:-1, 1:-1] ** 2)))
    psi_after = kernel._multigrid_vcycle(
        psi,
        source,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        omega=1.5,
        pre_smooth=2,
        post_smooth=2,
        min_grid=5,
    )
    r_after = kernel._mg_residual(psi_after, source, kernel.RR, kernel.dR, kernel.dZ)
    rms_after = float(np.sqrt(np.mean(r_after[1:-1, 1:-1] ** 2)))
    assert np.all(np.isfinite(psi_after))
    assert rms_after < rms_before


def test_leaf_vcycle_matches_owner(tmp_path: Path) -> None:
    """Leaf V-cycle matches the FusionKernel wrapper on the same inputs."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg.json"))
    psi = np.zeros_like(kernel.Psi)
    source = np.zeros_like(kernel.Psi)
    source[1:-1, 1:-1] = 0.5
    leaf = mg.multigrid_vcycle(
        psi.copy(),
        source,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        omega=1.4,
        pre_smooth=2,
        post_smooth=2,
        min_grid=5,
    )
    owner = kernel._multigrid_vcycle(
        psi.copy(),
        source,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        omega=1.4,
        pre_smooth=2,
        post_smooth=2,
        min_grid=5,
    )
    np.testing.assert_allclose(owner, leaf, rtol=1.0e-12, atol=1.0e-14)


def test_mg_smooth_zero_source_preserves_zeros() -> None:
    """Zero source and zero state stay zero after smoothing (production algebra)."""
    n = 11
    r = np.linspace(1.0, 2.0, n)
    z = np.linspace(-1.0, 1.0, n)
    rr, _zz = np.meshgrid(r, z)
    psi = np.zeros((n, n))
    source = np.zeros((n, n))
    out = mg.mg_smooth(psi, source, rr, float(r[1] - r[0]), float(z[1] - z[0]), 1.5, 3)
    np.testing.assert_allclose(out, 0.0, atol=1.0e-15)
