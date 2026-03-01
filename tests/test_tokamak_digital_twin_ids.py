# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Tokamak digital twin IDS validation tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for history/pulse validation (428, 430) and
run_digital_twin with time_steps > 50 (337-338, 365)."""

from __future__ import annotations

import pytest

from scpn_control.control.tokamak_digital_twin import (
    run_digital_twin,
    run_digital_twin_ids_history,
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


class TestRunDigitalTwinMovingAvg:
    def test_many_steps_covers_moving_avg(self):
        """time_steps > 50 activates moving average branch (lines 337-338)."""
        result = run_digital_twin(time_steps=60, seed=42, save_plot=False)
        assert result["steps"] == 60
        assert "final_avg_temp" in result
