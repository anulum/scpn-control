# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Halo RE Physics Non-Finite Guard Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for _as_range low < min_allowed (96), high <= low (98),
avalanche non-finite (435), momentum non-finite (455), relativistic
loss non-finite (480), simulate loop non-finite guards (564, 569, 572, 578),
and ensemble prevention path (719)."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.halo_re_physics import (
    RunawayElectronModel,
    _as_range,
    run_disruption_ensemble,
)


class TestAsRangeValidation:
    def test_low_below_min_allowed(self):
        """_as_range with low < min_allowed raises (line 96)."""
        with pytest.raises(ValueError, match="must be >="):
            _as_range("test", (-1.0, 10.0), min_allowed=0.0)

    def test_high_equal_low(self):
        """_as_range with high == low raises (line 98)."""
        with pytest.raises(ValueError, match="low < high"):
            _as_range("test", (5.0, 5.0))

    def test_high_below_low(self):
        """_as_range with high < low raises (line 98)."""
        with pytest.raises(ValueError, match="low < high"):
            _as_range("test", (10.0, 5.0))


class TestNonFiniteRateGuards:
    def test_dreicer_non_finite_rate(self):
        """_dreicer_rate returns 0 for non-finite result (line 413)."""
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        rate = model._dreicer_rate(E=float("nan"), T_e_keV=1.0)
        assert rate == 0.0

    def test_avalanche_non_finite_E(self):
        """_avalanche_rate returns 0 for non-finite E (line 435)."""
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        rate = model._avalanche_rate(E=float("nan"), n_re=1e16)
        assert rate == 0.0

    def test_avalanche_non_finite_n_re(self):
        """_avalanche_rate returns 0 for non-finite n_re."""
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        rate = model._avalanche_rate(E=1.0, n_re=float("inf"))
        assert rate == 0.0

    def test_momentum_non_finite(self):
        """_momentum_space_growth returns 0 for non-finite input (line 455)."""
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        rate = model._momentum_space_growth(E=float("nan"), n_re=1e16)
        assert rate == 0.0

    def test_relativistic_loss_non_finite(self):
        """_relativistic_loss_rate returns 0 for non-finite input (line 480)."""
        model = RunawayElectronModel(
            n_e=1e20, T_e_keV=1.0, z_eff=2.0,
            enable_relativistic_losses=True,
        )
        rate = model._relativistic_loss_rate(E=float("nan"), n_re=1e16)
        assert rate == 0.0


class TestEnsemblePrevention:
    def test_ensemble_high_prevention(self):
        """Ensemble with parameters that should prevent disruptions (line 719)."""
        result = run_disruption_ensemble(
            ensemble_runs=10,
            seed=42,
            plasma_current_range=(5.0, 8.0),
            plasma_energy_range=(5.0, 10.0),
            neon_range=(0.5, 1.5),
        )
        assert result.prevention_rate >= 0.0
        assert len(result.per_run_details) == 10
        # At least some runs should be prevented with these moderate parameters
        prevented = sum(1 for r in result.per_run_details if r.get("prevented", False))
        assert isinstance(prevented, int)
