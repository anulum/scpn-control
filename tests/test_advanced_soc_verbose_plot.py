# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Advanced Soc Verbose Plot.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Advanced SOC verbose+plot path test
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for verbose + plot_saved log (line 357)."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("matplotlib")

from scpn_control.control.advanced_soc_fusion_learning import run_advanced_learning_sim


class TestVerbosePlotSaved:
    def test_verbose_with_save_plot(self, tmp_path, caplog):
        """verbose=True + save_plot=True triggers line 357."""
        out = str(tmp_path / "soc.png")
        with caplog.at_level(logging.INFO, logger="scpn_control.control.advanced_soc_fusion_learning"):
            result = run_advanced_learning_sim(
                size=8,
                time_steps=20,
                seed=0,
                save_plot=True,
                output_path=out,
                verbose=True,
            )
        assert result["plot_saved"] is True
        assert "Simulation complete" in caplog.text
