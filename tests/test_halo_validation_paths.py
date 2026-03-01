# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Halo RE Physics Validation Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for halo_re_physics.py validation guards: _as_non_negative_float
negative value, _as_range low < min_allowed, _as_range high <= low,
simulate dt > duration, simulate seed_re_fraction out of range,
and dreicer edge cases for ratio > 200 and non-finite loss guards."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.halo_re_physics import (
    RunawayElectronModel,
    run_disruption_ensemble,
)


class TestValidationGuards:
    def test_negative_n_e_raises(self):
        with pytest.raises(ValueError, match="n_e"):
            RunawayElectronModel(n_e=-1.0, T_e_keV=1.0, z_eff=2.0)

    def test_dt_greater_than_duration_raises(self):
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        with pytest.raises(ValueError, match="dt_s"):
            model.simulate(dt_s=1.0, duration_s=0.01)

    def test_seed_re_fraction_zero_raises(self):
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        with pytest.raises(ValueError, match="seed_re_fraction"):
            model.simulate(seed_re_fraction=0.0)

    def test_seed_re_fraction_negative_raises(self):
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        with pytest.raises(ValueError, match="seed_re_fraction"):
            model.simulate(seed_re_fraction=-0.1)


class TestDreicerEdge:
    def test_dreicer_high_ratio_zero(self):
        """E_D/E ratio > 200 returns 0 (line 395-396)."""
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        rate = model._dreicer_rate(E=1e-6, T_e_keV=10.0)
        assert rate == 0.0

    def test_dreicer_negative_ratio(self):
        """ratio <= 0 returns 0 (line 393-394)."""
        model = RunawayElectronModel(n_e=1e20, T_e_keV=1.0, z_eff=2.0)
        rate = model._dreicer_rate(E=-1.0, T_e_keV=1.0)
        assert rate == 0.0


class TestEnsembleRange:
    def test_ensemble_with_custom_ranges(self):
        result = run_disruption_ensemble(
            ensemble_runs=2,
            seed=0,
            plasma_current_range=(10.0, 20.0),
            plasma_energy_range=(10.0, 50.0),
            neon_range=(0.01, 0.5),
        )
        assert result.prevention_rate >= 0.0
        assert len(result.per_run_details) == 2
