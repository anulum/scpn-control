# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fusion-kernel phase-boundary coverage tests.
"""Exercise public FusionKernel branches relevant to complete owner coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.fusion_kernel import FusionKernel


@pytest.fixture
def kernel(tmp_path: Path) -> FusionKernel:
    """Return a compact public FusionKernel instance."""
    config = {
        "reactor_name": "Phase-boundary coverage",
        "grid_resolution": [16, 16],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [],
        "solver": {
            "max_iterations": 3,
            "convergence_threshold": 1.0e-4,
            "relaxation_factor": 0.1,
            "solver_method": "sor",
        },
    }
    path = tmp_path / "fusion_kernel.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return FusionKernel(path)


def test_find_x_point_uses_gradient_candidate_when_no_saddle(kernel: FusionKernel) -> None:
    """Use the public topology API on a convex field with no saddle cell."""
    convex_flux = (kernel.RR - 4.0) ** 2 + (kernel.ZZ + 2.0) ** 2

    position, psi_x = kernel.find_x_point(convex_flux)

    assert position != (0.0, 0.0)
    assert np.isfinite(psi_x)


def test_find_x_point_falls_back_when_search_region_is_empty(kernel: FusionKernel) -> None:
    """Return a finite public fallback when no grid cell is in the search region."""
    kernel.ZZ[:] = 1.0
    flux = np.arange(kernel.NR * kernel.NZ, dtype=float).reshape(kernel.NZ, kernel.NR)

    position, psi_x = kernel.find_x_point(flux)

    assert position == (0.0, 0.0)
    assert psi_x == 0.0


def test_zero_iteration_public_solve_returns_bounded_nonconvergence(kernel: FusionKernel) -> None:
    """Report a valid zero-iteration result without pretending convergence."""
    kernel.cfg["solver"]["max_iterations"] = 0

    result = kernel.solve_equilibrium()

    assert result["converged"] is False
    assert result["iterations"] == 0
    assert result["residual"] == 1.0
    assert result["boundary_variant"] == "fixed_boundary"
    assert np.isfinite(kernel.B_R).all()
    assert np.isfinite(kernel.B_Z).all()
