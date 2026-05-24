# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Tokamak Digital Twin Ids
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Tokamak digital twin IDS validation tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for history/pulse validation (428, 430) and
run_digital_twin with time_steps > 50 (337-338, 365)."""

from __future__ import annotations

import pytest

from scpn_control.control.tokamak_digital_twin import (
    run_digital_twin,
    run_digital_twin_ids_history,
    run_digital_twin_ids_pulse,
)


class TestIdsHistoryValidation:
    def test_string_history_steps_raises(self):
        """String history_steps raises ValueError (line 430)."""
        with pytest.raises(ValueError, match="history_steps must be a sequence"):
            run_digital_twin_ids_history("invalid", seed=42)

    def test_empty_history_steps_raises(self):
        """Empty history_steps raises ValueError (line 432)."""
        with pytest.raises(ValueError, match="at least one"):
            run_digital_twin_ids_history([], seed=42)

    def test_history_mode_rejects_direct_time_steps_override(self):
        """History-mode IDS export derives time_steps from history_steps."""
        with pytest.raises(ValueError, match="time_steps is controlled"):
            run_digital_twin_ids_history([10], time_steps=5)


class TestIdsPulseValidation:
    def test_pulse_mode_rejects_direct_time_steps_override(self):
        """Pulse-mode IDS export derives time_steps from history_steps."""
        with pytest.raises(ValueError, match="time_steps is controlled"):
            run_digital_twin_ids_pulse([10], time_steps=5)


class TestRunDigitalTwinMovingAvg:
    def test_many_steps_covers_moving_avg(self):
        """time_steps > 50 activates moving average branch (lines 337-338)."""
        result = run_digital_twin(time_steps=60, seed=42, save_plot=False)
        assert result["steps"] == 60
        assert "final_avg_temp" in result
