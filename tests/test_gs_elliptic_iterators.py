# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for GS elliptic iterators

"""Drive production Jacobi/SOR/Anderson elliptic helpers on real surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scpn_control.core.gs_elliptic_iterators as ell
from scpn_control.core.fusion_kernel import FusionKernel


def _write_config(path: Path, method: str = "sor") -> Path:
    raw = {
        "reactor_name": "Elliptic-Test",
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "grid_resolution": [13, 13],
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "solver": {"boundary_variant": "fixed", "solver_method": method, "sor_omega": 1.5},
        "coils": [{"r": 3.2, "z": 0.0, "current": 1.0e5, "turns": 1}],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_owner_jacobi_and_sor_match_leaf(tmp_path: Path) -> None:
    """FusionKernel Jacobi/SOR wrappers are the production leaf functions."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg.json"))
    src = np.zeros_like(kernel.Psi)
    src[1:-1, 1:-1] = 0.25
    leaf_j = ell.jacobi_step(kernel.Psi, src, kernel.RR, kernel.dR, kernel.dZ)
    owner_j = kernel._jacobi_step(kernel.Psi, src)
    np.testing.assert_allclose(owner_j, leaf_j, rtol=1.0e-14, atol=1.0e-14)
    leaf_s = ell.sor_step(kernel.Psi, src, kernel.RR, kernel.dR, kernel.dZ, omega=1.4)
    owner_s = kernel._sor_step(kernel.Psi, src, omega=1.4)
    np.testing.assert_allclose(owner_s, leaf_s, rtol=1.0e-14, atol=1.0e-14)
    assert np.all(np.isfinite(owner_j))
    assert np.all(np.isfinite(owner_s))


def test_anderson_step_falls_back_with_insufficient_history() -> None:
    """Anderson mixing with fewer than two residuals returns the latest iterate."""
    psi = np.ones((5, 5))
    result = ell.anderson_step([psi], [psi * 0.1], m=5)
    np.testing.assert_allclose(result, psi)
    assert result is not psi


def test_anderson_step_mixes_well_conditioned_history() -> None:
    """Anderson mixing returns a finite field from a short residual history."""
    psi0 = np.zeros((6, 6))
    psi1 = np.ones((6, 6))
    psi2 = np.full((6, 6), 0.5)
    res0 = psi1 - psi0
    res1 = psi2 - psi1
    mixed = ell.anderson_step([psi0, psi1, psi2], [res0, res1], m=5)
    assert mixed.shape == psi0.shape
    assert np.all(np.isfinite(mixed))


def test_elliptic_solve_python_enforces_boundary(tmp_path: Path) -> None:
    """Python elliptic path enforces Dirichlet boundary from Psi_bc."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg_mg.json", method="jacobi"))
    psi_bc = np.full_like(kernel.Psi, 3.0)
    source = np.zeros_like(kernel.Psi)
    source[1:-1, 1:-1] = 0.1
    out = ell.elliptic_solve_python(
        "jacobi",
        kernel.Psi,
        source,
        psi_bc,
        kernel.RR,
        kernel.dR,
        kernel.dZ,
        omega=1.5,
    )
    np.testing.assert_allclose(out[0, :], 3.0)
    np.testing.assert_allclose(out[-1, :], 3.0)
    np.testing.assert_allclose(out[:, 0], 3.0)
    np.testing.assert_allclose(out[:, -1], 3.0)
    owner = kernel._elliptic_solve(source, psi_bc)
    np.testing.assert_allclose(owner, out, rtol=1.0e-12, atol=1.0e-14)


def test_apply_boundary_conditions_in_place() -> None:
    """Boundary helper writes Psi_bc edges into Psi (production in-place path)."""
    psi = np.zeros((4, 5))
    bc = np.arange(20, dtype=float).reshape(4, 5)
    ell.apply_boundary_conditions(psi, bc)
    np.testing.assert_allclose(psi[0, :], bc[0, :])
    np.testing.assert_allclose(psi[-1, :], bc[-1, :])
    np.testing.assert_allclose(psi[:, 0], bc[:, 0])
    np.testing.assert_allclose(psi[:, -1], bc[:, -1])
    # Interior remains zero.
    np.testing.assert_allclose(psi[1:-1, 1:-1], 0.0)
