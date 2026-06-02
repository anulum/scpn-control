# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Fusion Kernel Profiles
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — FusionKernel Profile Config & Anderson Mixing Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for fusion_kernel.py profile_mode setup, Anderson mixing
LinAlgError branch, and alpha normalisation guard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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

    def test_hmode_profile_jacobian_uses_mtanh_slope(self, tmp_path):
        """H-mode Newton Jacobian must follow the mtanh pedestal derivative."""
        path = _cfg(
            tmp_path,
            extra_physics={
                "profiles": {
                    "mode": "h-mode",
                    "p_prime": {"ped_top": 0.65, "ped_width": 0.08, "ped_height": 2.0, "core_alpha": 0.7},
                    "ff_prime": {"ped_top": 0.65, "ped_width": 0.08, "ped_height": 1.0, "core_alpha": 0.4},
                },
            },
        )
        fk = FusionKernel(path)
        fk.Psi = np.linspace(0.0, 1.0, fk.NZ * fk.NR).reshape(fk.NZ, fk.NR)

        jac = fk._compute_profile_jacobian(Psi_axis=0.0, Psi_boundary=1.0, mu0=1.0)
        plasma = (fk.Psi >= 0.0) & (fk.Psi < 1.0)

        assert np.all(np.isfinite(jac))
        assert np.std(jac[plasma]) > 1.0e-6
        iz, ir = fk.NZ // 2, fk.NR // 2
        eps = 1.0e-6
        p_plus = fk.mtanh_profile(fk.Psi + eps, fk.ped_params_p)
        p_minus = fk.mtanh_profile(fk.Psi - eps, fk.ped_params_p)
        ff_plus = fk.mtanh_profile(fk.Psi + eps, fk.ped_params_ff)
        ff_minus = fk.mtanh_profile(fk.Psi - eps, fk.ped_params_ff)
        finite_diff = fk.RR * (p_plus - p_minus) / (2.0 * eps) + (ff_plus - ff_minus) / (2.0 * eps) / fk.RR
        np.testing.assert_allclose(jac[iz, ir], finite_diff[iz, ir], rtol=2.0e-4, atol=1.0e-7)


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
