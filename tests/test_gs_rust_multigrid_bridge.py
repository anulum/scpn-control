# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for Rust multigrid bridge

"""Drive production rust_multigrid bridge fallbacks and success path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scpn_control.core.gs_rust_multigrid_bridge as bridge
from scpn_control.core import _rust_compat as rust_compat
from scpn_control.core.fusion_kernel import FusionKernel


def _write_config(path: Path, *, method: str = "rust_multigrid") -> Path:
    raw = {
        "reactor_name": "Rust-MG-Bridge-Test",
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "grid_resolution": [10, 10],
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "solver": {
            "solver_method": method,
            "max_iterations": 5,
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


class _FakeRustResult:
    converged = True
    iterations = 7
    residual = 1e-5


class _FakeRustKernel:
    def __init__(self, config_path: object) -> None:
        self._config_path = config_path
        self.Psi = np.full((10, 10), 0.25)
        self.J_phi = np.full((10, 10), 0.1)
        self.B_R = np.zeros((10, 10))
        self.B_Z = np.zeros((10, 10))
        self._method = "multigrid"

    def set_solver_method(self, method: str) -> None:
        self._method = method

    def solve_equilibrium(self) -> _FakeRustResult:
        return _FakeRustResult()


def test_kernel_satisfies_rust_multigrid_owner_protocol(tmp_path: Path) -> None:
    """Live FusionKernel implements the bridge Protocol surface."""
    kernel = FusionKernel(_write_config(tmp_path / "proto.json"))
    assert isinstance(kernel, bridge.RustMultigridOwner)


def test_owner_wrapper_matches_leaf_on_rust_unavailable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner and leaf share the Python SOR fallback when Rust is unavailable."""
    monkeypatch.setattr(rust_compat, "_rust_available", lambda: False)
    owner_kernel = FusionKernel(_write_config(tmp_path / "owner.json"))
    leaf_kernel = FusionKernel(_write_config(tmp_path / "leaf.json"))
    owner = owner_kernel._solve_via_rust_multigrid()
    leaf = bridge.solve_via_rust_multigrid(leaf_kernel)
    assert owner["solver_method"] in ("sor", "anderson")
    assert leaf["solver_method"] in ("sor", "anderson")
    assert "psi" in owner and "psi" in leaf
    assert np.all(np.isfinite(owner["psi"]))
    assert np.all(np.isfinite(leaf["psi"]))


def test_boundary_constrained_path_falls_back_to_python(tmp_path: Path) -> None:
    """Boundary-constrained rust_multigrid requests fall back to Python SOR."""
    kernel = FusionKernel(_write_config(tmp_path / "bc.json"))
    boundary = np.zeros_like(kernel.Psi)
    result = bridge.solve_via_rust_multigrid(
        kernel,
        preserve_initial_state=True,
        boundary_flux=boundary,
    )
    assert result["solver_method"] in ("sor", "anderson")
    assert np.all(np.isfinite(result["psi"]))


def test_fake_rust_backend_success_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When Rust is available, the bridge syncs state and returns rust_multigrid."""
    monkeypatch.setattr(rust_compat, "_rust_available", lambda: True)
    monkeypatch.setattr(rust_compat, "RustAcceleratedKernel", _FakeRustKernel)
    kernel = FusionKernel(_write_config(tmp_path / "ok.json"))
    result = kernel.solve_equilibrium()
    assert result["solver_method"] == "rust_multigrid"
    assert result["converged"] is True
    assert result["iterations"] == 7
    np.testing.assert_allclose(kernel.Psi, 0.25)
    assert np.all(np.isfinite(result["gs_residual"]))
