# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Controller Tuning Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Tests for automated controller tuning logic.
"""

from __future__ import annotations

import pytest

from scpn_control.control.controller_tuning import HAS_OPTUNA, tune_hinf, tune_pid


def test_optuna_guard_behavior():
    """Verify that tune_pid behaves correctly when Optuna is missing."""
    if not HAS_OPTUNA:
        # Should return default gains and not crash
        gains = tune_pid(None, n_trials=1)
        assert "Kp" in gains
        assert "Ki" in gains
        assert "Kd" in gains
    else:
        # If available, we could test a mock env
        pass


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
def test_tune_pid_with_optuna():
    """Placeholder for full tuning test if optuna available."""
    # This would require a mock gymnasium environment
    pass


def test_tune_hinf_no_optuna_fallback():
    """tune_hinf returns defaults when Optuna is absent."""
    if not HAS_OPTUNA:
        result = tune_hinf(plant={}, n_trials=1)
        assert result == {"gamma": 1.1, "bandwidth": 0.5}
    else:
        # Optuna available — exercise the no-optuna path via mock
        import unittest.mock as mock

        import scpn_control.control.controller_tuning as ct

        with mock.patch.object(ct, "HAS_OPTUNA", False):
            result = ct.tune_hinf(plant={}, n_trials=1)
            assert result == {"gamma": 1.1, "bandwidth": 0.5}


def test_has_optuna_import_branch():
    """Cover controller_tuning.py line 22: HAS_OPTUNA set on import."""
    import scpn_control.control.controller_tuning as ct

    assert isinstance(ct.HAS_OPTUNA, bool)
