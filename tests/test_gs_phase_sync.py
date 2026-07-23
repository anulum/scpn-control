# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for GS phase-sync helpers

"""Drive production FusionKernel phase-sync steps on real surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scpn_control.core.gs_phase_sync as phase
from scpn_control.core.fusion_kernel import FusionKernel


def _write_config(path: Path) -> Path:
    raw = {
        "reactor_name": "Phase-Sync-Test",
        "dimensions": {"R_min": 0.5, "R_max": 2.5, "Z_min": -1.5, "Z_max": 1.5},
        "grid_resolution": [9, 9],
        "coils": [],
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "solver": {"solver_method": "sor", "max_iterations": 10, "tol": 1e-4},
        "phase_sync": {"K": 2.0, "zeta": 0.5, "psi_mode": "external", "alpha": 0.0},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_owner_phase_sync_step_matches_leaf(tmp_path: Path) -> None:
    """FusionKernel phase_sync_step wrapper is the production leaf function."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg.json"))
    rng = np.random.default_rng(42)
    theta = rng.uniform(-np.pi, np.pi, 32)
    omega = rng.normal(0, 0.5, 32)
    owner = kernel.phase_sync_step(theta, omega, dt=0.01, psi_driver=0.0)
    leaf = phase.phase_sync_step(
        theta,
        omega,
        dt=0.01,
        psi_driver=0.0,
        phase_sync_cfg=kernel.cfg.get("phase_sync", {}),
    )
    np.testing.assert_allclose(owner["theta1"], leaf["theta1"], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(owner["R"], leaf["R"], rtol=1e-14, atol=1e-14)
    assert owner["theta1"].shape == (32,)
    assert np.all(np.isfinite(owner["theta1"]))


def test_phase_sync_step_applies_config_zeta(tmp_path: Path) -> None:
    """Config zeta drives a positive phase update under a positive Ψ driver."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg_z.json"))
    theta = np.zeros(20)
    omega = np.zeros(20)
    out = kernel.phase_sync_step(theta, omega, dt=0.01, psi_driver=1.0)
    assert np.all(out["dtheta"] > 0.0)


def test_owner_phase_sync_lyapunov_matches_leaf(tmp_path: Path) -> None:
    """Lyapunov multi-step path matches leaf and reports stable λ < 0."""
    kernel = FusionKernel(_write_config(tmp_path / "cfg_l.json"))
    rng = np.random.default_rng(42)
    theta = rng.uniform(-np.pi, np.pi, 50)
    omega = np.zeros(50)
    owner = kernel.phase_sync_step_lyapunov(
        theta,
        omega,
        n_steps=100,
        dt=0.01,
        zeta=3.0,
        psi_driver=0.5,
    )
    leaf = phase.phase_sync_step_lyapunov(
        theta,
        omega,
        n_steps=100,
        dt=0.01,
        zeta=3.0,
        psi_driver=0.5,
        phase_sync_cfg=kernel.cfg.get("phase_sync", {}),
    )
    np.testing.assert_allclose(owner["theta_final"], leaf["theta_final"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(owner["R_hist"], leaf["R_hist"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(owner["V_hist"], leaf["V_hist"], rtol=1e-12, atol=1e-14)
    assert owner["lambda"] == pytest.approx(leaf["lambda"])
    assert owner["lambda"] < 0.0
    assert owner["stable"] is True
    assert owner["R_hist"].shape == (100,)


def test_phase_sync_lyapunov_fail_closed_invalid_step_params() -> None:
    """Production Lyapunov path fails closed on invalid n_steps or dt."""
    theta = np.zeros(8)
    omega = np.zeros(8)
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        phase.phase_sync_step_lyapunov(theta, omega, n_steps=0, dt=0.01)
    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        phase.phase_sync_step_lyapunov(theta, omega, n_steps=10, dt=0.0)
