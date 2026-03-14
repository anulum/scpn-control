# ──────────────────────────────────────────────────────────────────────
# SCPN Control — External Profile Mode Tests (GEQDSK → GS Solver)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.eqdsk import read_geqdsk
from scpn_control.core.fusion_kernel import FusionKernel

_SPARC_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference_data" / "sparc"
_LMODE_FILES = sorted(_SPARC_DIR.glob("lmode_*.geqdsk"))


def _write_cfg(tmp_path, cfg, name="cfg.json"):
    p = tmp_path / name
    p.write_text(json.dumps(cfg))
    return p


class TestExternalProfileConfig:
    """GEqdsk.to_config(external_profiles=True) passes pprime/ffprime."""

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1], ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_config_includes_profiles(self, geqdsk_path):
        eq = read_geqdsk(geqdsk_path)
        cfg = eq.to_config(external_profiles=True)
        profiles = cfg["physics"]["profiles"]
        assert profiles["mode"] == "external"
        assert len(profiles["pprime_values"]) == eq.nw
        assert len(profiles["ffprime_values"]) == eq.nw

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1], ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_config_without_flag_has_no_profiles(self, geqdsk_path):
        eq = read_geqdsk(geqdsk_path)
        cfg = eq.to_config(external_profiles=False)
        assert "profiles" not in cfg.get("physics", {})


class TestExternalProfileSolver:
    """GS solver converges with external pprime/ffprime from GEQDSK."""

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1], ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_converges_with_external_profiles(self, geqdsk_path, tmp_path):
        eq = read_geqdsk(geqdsk_path)
        cfg = eq.to_config(external_profiles=True)
        cfg["solver"]["solver_method"] = "sor"
        cfg["solver"]["max_iterations"] = 500
        cfg["solver"]["convergence_threshold"] = 1e-4
        cfg["solver"]["relaxation_factor"] = 0.15
        cfg["solver"]["sor_omega"] = 1.6

        fk = FusionKernel(_write_cfg(tmp_path, cfg))
        result = fk.solve_equilibrium()

        assert not np.any(np.isnan(result["psi"]))
        assert not np.any(np.isinf(result["psi"]))
        assert result["residual"] < 0.01

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1], ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_external_source_differs_from_lmode(self, geqdsk_path, tmp_path):
        """External pprime/ffprime produce different J_phi than L-mode linear model."""
        eq = read_geqdsk(geqdsk_path)

        cfg_ext = eq.to_config(external_profiles=True)
        cfg_ext["solver"]["solver_method"] = "sor"
        cfg_ext["solver"]["max_iterations"] = 5
        cfg_ext["solver"]["convergence_threshold"] = 1e-12
        cfg_ext["solver"]["relaxation_factor"] = 0.15
        cfg_ext["solver"]["sor_omega"] = 1.6

        cfg_lm = eq.to_config(external_profiles=False)
        cfg_lm["solver"] = dict(cfg_ext["solver"])

        fk_ext = FusionKernel(_write_cfg(tmp_path, cfg_ext, "ext.json"))
        fk_lm = FusionKernel(_write_cfg(tmp_path, cfg_lm, "lm.json"))

        # Give both the same non-trivial Psi so topology detection works
        mu0 = fk_ext.cfg["physics"]["vacuum_permeability"]
        fk_ext._seed_plasma(mu0)
        fk_lm.Psi = fk_ext.Psi.copy()
        fk_lm.J_phi = fk_ext.J_phi.copy()

        _, _, Psi_axis = fk_ext._find_magnetic_axis()
        _, Psi_boundary = fk_ext.find_x_point(fk_ext.Psi)

        J_ext = fk_ext.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)
        J_lm = fk_lm.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)

        diff = float(np.max(np.abs(J_ext - J_lm)))
        assert diff > 1e-6, f"External and L-mode J_phi are identical (max diff = {diff:.2e})"

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1], ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_external_profile_deterministic(self, geqdsk_path, tmp_path):
        eq = read_geqdsk(geqdsk_path)
        cfg = eq.to_config(external_profiles=True)
        cfg["solver"]["solver_method"] = "sor"
        cfg["solver"]["max_iterations"] = 200
        cfg["solver"]["convergence_threshold"] = 1e-4
        cfg["solver"]["relaxation_factor"] = 0.15
        cfg["solver"]["sor_omega"] = 1.6

        fk1 = FusionKernel(_write_cfg(tmp_path, cfg, "a.json"))
        r1 = fk1.solve_equilibrium()

        fk2 = FusionKernel(_write_cfg(tmp_path, cfg, "b.json"))
        r2 = fk2.solve_equilibrium()

        diff = float(np.max(np.abs(r1["psi"] - r2["psi"])))
        assert diff < 1e-12, f"Non-deterministic external profile solve: max |Δψ| = {diff:.2e}"
