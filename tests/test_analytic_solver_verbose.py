# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Analytic solver verbose + dR guard tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for _log verbose path (46) and dR <= 0 guard (107)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from scpn_control.control.analytic_solver import AnalyticEquilibriumSolver


class _FakeKernel:
    def __init__(self, config_path: str):
        self.R = np.linspace(4.0, 8.5, 65)
        self.Z = np.linspace(-4.0, 4.0, 65)
        self.dR = float(self.R[1] - self.R[0])
        self.cfg = {
            "coils": [
                {"name": "PF1", "current": 0.0, "R": 4.5, "Z": 3.0},
            ],
        }

    def calculate_vacuum_field(self):
        return np.zeros((len(self.Z), len(self.R)))


class _BadDrKernel(_FakeKernel):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.dR = 0.0


class TestVerboseLog:
    def test_verbose_logs_output(self, tmp_path, caplog):
        """verbose=True triggers _log logging (line 46)."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=_FakeKernel, verbose=True)
        with caplog.at_level(logging.INFO, logger="scpn_control.control.analytic_solver"):
            solver.calculate_required_Bv(R_geo=6.2, a_min=2.0, Ip_MA=15.0)
        assert "SHAFRANOV" in caplog.text


class TestDrGuard:
    def test_zero_dr_raises(self, tmp_path):
        """dR <= 0 raises ValueError (line 107)."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text("{}")
        solver = AnalyticEquilibriumSolver(str(cfg), kernel_factory=_BadDrKernel, verbose=False)
        with pytest.raises(ValueError, match="dR must be > 0"):
            solver.compute_coil_efficiencies(target_R=6.2)
