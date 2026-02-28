# ──────────────────────────────────────────────────────────────────────
# SCPN Control — FusionKernel Profile Config & Anderson Mixing Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for fusion_kernel.py profile_mode setup, Anderson mixing
LinAlgError branch, and alpha normalisation guard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.fusion_kernel import FusionKernel


def _cfg(tmp_path: Path, extra_physics=None) -> str:
    config = {
        "reactor_name": "Profiles-Test",
        "grid_resolution": [20, 20],
        "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
        "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
        "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
        "solver": {
            "max_iterations": 10,
            "convergence_threshold": 1e-4,
            "relaxation_factor": 0.1,
        },
    }
    if extra_physics:
        config["physics"].update(extra_physics)
    p = tmp_path / "fk.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return str(p)


class TestProfileConfig:
    def test_profiles_mode_and_ped_params(self, tmp_path):
        """physics.profiles config sets profile_mode and merges ped_params."""
        path = _cfg(
            tmp_path,
            extra_physics={
                "profiles": {
                    "mode": "h-mode",
                    "p_prime": {"alpha": 2.5},
                    "ff_prime": {"beta": 0.8},
                },
            },
        )
        fk = FusionKernel(path)
        assert fk.profile_mode == "h-mode"
        assert fk.ped_params_p["alpha"] == 2.5
        assert fk.ped_params_ff["beta"] == 0.8

    def test_no_profiles_config(self, tmp_path):
        """Default FusionKernel without profiles config."""
        path = _cfg(tmp_path)
        fk = FusionKernel(path)
        assert fk.profile_mode == "l-mode"


class TestAndersonMixing:
    def test_singular_gram_fallback(self, tmp_path):
        """LinAlgError in Anderson mixing returns latest iterate."""
        path = _cfg(tmp_path)
        fk = FusionKernel(path)
        # Provide identical residuals → singular dF
        r = np.ones((fk.NZ, fk.NR)) * 0.1
        psi0 = fk.Psi.copy()
        psi1 = fk.Psi.copy() + 0.01
        psi_history = [psi0, psi1]
        res_history = [r, r]
        result = fk._anderson_step(psi_history, res_history, m=2)
        assert result.shape == fk.Psi.shape
        assert np.all(np.isfinite(result))

    def test_normal_anderson_mixing(self, tmp_path):
        """Non-singular Anderson mixing produces valid output."""
        path = _cfg(tmp_path)
        fk = FusionKernel(path)
        psi0 = fk.Psi.copy()
        psi1 = fk.Psi.copy() + 0.01
        r0 = np.random.default_rng(0).standard_normal((fk.NZ, fk.NR)) * 0.01
        r1 = np.random.default_rng(1).standard_normal((fk.NZ, fk.NR)) * 0.01
        result = fk._anderson_step([psi0, psi1], [r0, r1], m=2)
        assert result.shape == fk.Psi.shape
