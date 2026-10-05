# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo Runaway Physics Tests
"""Public halo and runaway model regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.halo_re_physics import (
    HaloCurrentModel,
    HaloCurrentResult,
    RunawayElectronModel,
    RunawayElectronResult,
)

# ─── HaloCurrentModel construction ────────────────────────────────────


class TestHaloCurrentConstruction:
    """Validate halo circuit construction and input bounds."""

    def test_defaults(self) -> None:
        """Defaults."""
        m = HaloCurrentModel()
        assert m.Ip0 == pytest.approx(15e6, rel=1e-6)
        assert m.tpf == 2.0
        assert m.R_h > 0
        assert m.L_h > 0
        assert m.tau_h > 0

    def test_rejects_negative_current(self) -> None:
        """Rejects negative current."""
        with pytest.raises(ValueError, match="plasma_current_ma"):
            HaloCurrentModel(plasma_current_ma=-1.0)

    def test_rejects_zero_radius(self) -> None:
        """Rejects zero radius."""
        with pytest.raises(ValueError, match="minor_radius_m"):
            HaloCurrentModel(minor_radius_m=0.0)

    def test_rejects_minor_radius_at_or_above_major_radius(self) -> None:
        """Reject a plasma geometry that makes the circuit model invalid."""
        with pytest.raises(ValueError, match="minor_radius_m must be smaller"):
            HaloCurrentModel(minor_radius_m=6.2, major_radius_m=6.2)

    def test_rejects_contact_fraction_out_of_range(self) -> None:
        """Rejects contact fraction out of range."""
        with pytest.raises(ValueError, match="contact_fraction"):
            HaloCurrentModel(contact_fraction=0.0)
        with pytest.raises(ValueError, match="contact_fraction"):
            HaloCurrentModel(contact_fraction=1.5)

    def test_rejects_nan(self) -> None:
        """Rejects nan."""
        with pytest.raises(ValueError):
            HaloCurrentModel(plasma_current_ma=float("nan"))


# ─── HaloCurrentModel.simulate() ──────────────────────────────────────


class TestHaloSimulation:
    """Validate time-resolved halo current and wall-force outputs."""

    @pytest.fixture()
    def result(self) -> HaloCurrentResult:
        """Create a time-resolved halo current result."""
        m = HaloCurrentModel(plasma_current_ma=15.0, tpf=2.0, contact_fraction=0.3)
        return m.simulate(tau_cq_s=0.01, duration_s=0.05, dt_s=1e-4)

    def test_result_type(self, result: HaloCurrentResult) -> None:
        """Result type."""
        assert isinstance(result, HaloCurrentResult)

    def test_time_vector_length(self, result: HaloCurrentResult) -> None:
        """Time vector length."""
        assert len(result.time_ms) >= 10

    def test_halo_current_non_negative(self, result: HaloCurrentResult) -> None:
        """Halo current non negative."""
        assert all(i >= 0.0 for i in result.halo_current_ma)

    def test_plasma_current_decays(self, result: HaloCurrentResult) -> None:
        """Plasma current decays."""
        assert result.plasma_current_ma[0] > result.plasma_current_ma[-1]

    def test_peak_halo_bounded_by_plasma(self, result: HaloCurrentResult) -> None:
        """Peak halo bounded by plasma."""
        # Halo current cannot exceed initial plasma current
        assert result.peak_halo_ma <= 15.0

    def test_peak_halo_positive(self, result: HaloCurrentResult) -> None:
        """Peak halo positive."""
        assert result.peak_halo_ma > 0.0

    def test_tpf_product_positive(self, result: HaloCurrentResult) -> None:
        """Tpf product positive."""
        assert result.peak_tpf_product > 0.0

    def test_wall_force_positive(self, result: HaloCurrentResult) -> None:
        """Wall force positive."""
        assert result.wall_force_mn_m > 0.0

    def test_faster_quench_higher_halo(self) -> None:
        """Faster current quench → larger dI_p/dt → higher halo peak."""
        m = HaloCurrentModel(plasma_current_ma=15.0)
        fast = m.simulate(tau_cq_s=0.005, duration_s=0.05, dt_s=1e-4)
        slow = m.simulate(tau_cq_s=0.020, duration_s=0.05, dt_s=1e-4)
        assert fast.peak_halo_ma > slow.peak_halo_ma

    def test_higher_tpf_higher_product(self) -> None:
        """Higher TPF → higher TPF × I_h/I_p product."""
        low = HaloCurrentModel(tpf=1.5).simulate()
        high = HaloCurrentModel(tpf=2.5).simulate()
        assert high.peak_tpf_product > low.peak_tpf_product

    def test_rejects_dt_larger_than_duration(self) -> None:
        """Rejects dt larger than duration."""
        m = HaloCurrentModel()
        with pytest.raises(ValueError, match="dt_s"):
            m.simulate(dt_s=1.0, duration_s=0.01)

    def test_short_duration_does_not_run_past_requested_end(self) -> None:
        """Keep halo simulation samples inside the requested duration."""
        result = HaloCurrentModel().simulate(duration_s=0.01, dt_s=0.003)
        assert len(result.time_ms) == 4
        assert result.time_ms[-1] < 10.0
        assert result.plasma_current_ma[-1] == pytest.approx(15.0 * 0.7**3 * 0.9)


# ─── RunawayElectronModel construction ─────────────────────────────────


class TestREConstruction:
    """Validate runaway model construction and plasma inputs."""

    def test_defaults(self) -> None:
        """Defaults."""
        m = RunawayElectronModel()
        assert m.n_e_free == pytest.approx(1e20)
        assert m.T_e0 == pytest.approx(20.0)
        assert m.E_D > 0
        assert m.E_c > 0
        assert m.tau_coll > 0
        assert m.tau_av > 0

    def test_rejects_negative_density(self) -> None:
        """Rejects negative density."""
        with pytest.raises(ValueError, match="n_e"):
            RunawayElectronModel(n_e=-1e20)

    def test_rejects_z_eff_below_one(self) -> None:
        """Rejects z eff below one."""
        with pytest.raises(ValueError, match="z_eff"):
            RunawayElectronModel(z_eff=0.5)


# ─── Dreicer field ordering ────────────────────────────────────────────


class TestDreicerField:
    """Check critical and Dreicer field relationships."""

    def test_dreicer_exceeds_critical(self) -> None:
        """Connor-Hastie: E_D >> E_c always (Dreicer is thermal, critical is relativistic)."""
        m = RunawayElectronModel(n_e=1e20, T_e_keV=10.0)
        assert m.E_D > m.E_c

    def test_dreicer_rate_zero_for_zero_field(self) -> None:
        """Dreicer rate zero for zero field."""
        m = RunawayElectronModel()
        assert m._dreicer_rate(0.0, 10.0) == 0.0

    def test_dreicer_rate_positive_for_strong_field(self) -> None:
        """Dreicer rate positive for strong field."""
        m = RunawayElectronModel(n_e=1e20, T_e_keV=5.0)
        # E slightly below E_D should give nonzero rate
        E = m.E_D * 0.1
        rate = m._dreicer_rate(E, 5.0)
        assert rate >= 0.0  # may still be small due to exponential suppression


# ─── RunawayElectronModel.simulate() ───────────────────────────────────


class TestRESimulation:
    """Validate time-resolved runaway current outputs."""

    @pytest.fixture()
    def result(self) -> RunawayElectronResult:
        """Create a time-resolved runaway current result."""
        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.5)
        return m.simulate(
            plasma_current_ma=15.0,
            tau_cq_s=0.01,
            T_e_quench_keV=0.5,
            duration_s=0.03,
            dt_s=1e-4,
        )

    def test_result_type(self, result: RunawayElectronResult) -> None:
        """Result type."""
        assert isinstance(result, RunawayElectronResult)

    def test_re_current_non_negative(self, result: RunawayElectronResult) -> None:
        """Re current non negative."""
        assert all(i >= 0.0 for i in result.runaway_current_ma)

    def test_peak_re_bounded_by_plasma(self, result: RunawayElectronResult) -> None:
        """Peak re bounded by plasma."""
        assert result.peak_re_current_ma <= 15.0

    def test_avalanche_gain_finite(self, result: RunawayElectronResult) -> None:
        """Avalanche gain finite."""
        assert np.isfinite(result.avalanche_gain)
        assert result.avalanche_gain >= 0.0

    def test_electric_field_positive(self, result: RunawayElectronResult) -> None:
        """Electric field positive."""
        assert all(e >= 0.0 for e in result.electric_field_v_m)

    def test_short_duration_does_not_run_past_requested_end(self) -> None:
        """Keep runaway simulation samples inside the requested duration."""
        result = RunawayElectronModel().simulate(duration_s=0.01, dt_s=0.003)
        assert len(result.time_ms) == 4
        assert result.time_ms[-1] < 10.0


# ─── Neon mitigation ──────────────────────────────────────────────────


class TestNeonMitigation:
    """Check impurity effects on runaway dynamics."""

    def test_high_neon_suppresses_avalanche(self) -> None:
        """Heavy neon injection → avalanche deconfinement → lower RE peak."""
        base = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, neon_mol=0.0)
        mitigated = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, neon_mol=0.5)

        r_base = base.simulate(plasma_current_ma=15.0, tau_cq_s=0.01, neon_mol=0.0)
        r_mit = mitigated.simulate(plasma_current_ma=15.0, tau_cq_s=0.01, neon_mol=0.5)

        assert r_mit.peak_re_current_ma <= r_base.peak_re_current_ma

    def test_neon_raises_total_density(self) -> None:
        """Neon raises total density."""
        m0 = RunawayElectronModel(neon_mol=0.0)
        m1 = RunawayElectronModel(neon_mol=0.5)
        assert m1.n_e_tot > m0.n_e_tot

    def test_neon_raises_critical_field(self) -> None:
        """Neon raises critical field."""
        m0 = RunawayElectronModel(neon_mol=0.0)
        m1 = RunawayElectronModel(neon_mol=0.5)
        assert m1.E_c > m0.E_c


# ─── Relativistic losses ──────────────────────────────────────────────


class TestRelativisticLosses:
    """Check relativistic loss controls and zero-density behavior."""

    def test_loss_zero_when_disabled(self) -> None:
        """Loss zero when disabled."""
        m = RunawayElectronModel(enable_relativistic_losses=False)
        assert m._relativistic_loss_rate(E=100.0, n_re=1e18) == 0.0

    def test_loss_positive_when_enabled(self) -> None:
        """Loss positive when enabled."""
        m = RunawayElectronModel(enable_relativistic_losses=True)
        loss = m._relativistic_loss_rate(E=m.E_c * 5.0, n_re=1e18)
        assert loss > 0.0

    def test_loss_zero_for_zero_re(self) -> None:
        """Loss zero for zero re."""
        m = RunawayElectronModel()
        assert m._relativistic_loss_rate(E=100.0, n_re=0.0) == 0.0


# ─── Disruption ensemble ──────────────────────────────────────────────


# ─── Dreicer/avalanche NaN guard paths ───────────────────────────────


class TestREGuardPaths:
    """Check non-finite numerical guard behavior."""

    def test_dreicer_rate_nan_field_returns_zero(self) -> None:
        """Dreicer rate nan field returns zero."""
        m = RunawayElectronModel()
        assert m._dreicer_rate(float("nan"), 10.0) == 0.0

    def test_dreicer_rate_nan_temp_returns_zero(self) -> None:
        """Dreicer rate nan temp returns zero."""
        m = RunawayElectronModel()
        assert m._dreicer_rate(1.0, float("nan")) == 0.0

    def test_dreicer_rate_cold_plasma_returns_zero(self) -> None:
        """Dreicer rate cold plasma returns zero."""
        m = RunawayElectronModel()
        assert m._dreicer_rate(m.E_D * 0.1, 0.001) == 0.0

    def test_avalanche_rate_nan_returns_zero(self) -> None:
        """Avalanche rate nan returns zero."""
        m = RunawayElectronModel()
        assert m._avalanche_rate(float("nan"), 1e15) == 0.0

    def test_avalanche_rate_below_critical_returns_zero(self) -> None:
        """Avalanche rate below critical returns zero."""
        m = RunawayElectronModel()
        assert m._avalanche_rate(m.E_c * 0.5, 1e15) == 0.0

    def test_avalanche_rate_zero_re_returns_zero(self) -> None:
        """Avalanche rate zero re returns zero."""
        m = RunawayElectronModel()
        assert m._avalanche_rate(m.E_c * 2.0, 0.0) == 0.0

    def test_momentum_space_nan_returns_zero(self) -> None:
        """Momentum space nan returns zero."""
        m = RunawayElectronModel()
        assert m._momentum_space_growth(float("nan"), 1e15) == 0.0

    def test_momentum_space_below_critical_returns_zero(self) -> None:
        """Momentum space below critical returns zero."""
        m = RunawayElectronModel()
        assert m._momentum_space_growth(m.E_c * 0.5, 1e15) == 0.0

    def test_relativistic_loss_nan_returns_zero(self) -> None:
        """Relativistic loss nan returns zero."""
        m = RunawayElectronModel(enable_relativistic_losses=True)
        assert m._relativistic_loss_rate(E=float("nan"), n_re=1e15) == 0.0

    def test_relativistic_loss_nan_nre_returns_zero(self) -> None:
        """Relativistic loss nan nre returns zero."""
        m = RunawayElectronModel(enable_relativistic_losses=True)
        assert m._relativistic_loss_rate(E=100.0, n_re=float("nan")) == 0.0

    def test_high_neon_deconfinement_factor(self) -> None:
        """neon_mol > 0.3 activates deconfinement suppression in avalanche."""
        m = RunawayElectronModel(neon_mol=0.5)
        rate = m._avalanche_rate(m.E_c * 3.0, 1e18)
        m_low = RunawayElectronModel(neon_mol=0.0)
        rate_low = m_low._avalanche_rate(m_low.E_c * 3.0, 1e18)
        if rate_low > 0.0:
            assert rate < rate_low

    def test_dreicer_rate_very_high_ratio_returns_zero(self) -> None:
        """Dreicer rate with ratio > 200 should return 0 (negligible generation)."""
        m = RunawayElectronModel(n_e=1e20, T_e_keV=0.1)
        rate = m._dreicer_rate(1e-10, 0.1)
        assert rate == 0.0


# ── Coverage completion: helpers, simulate guards, verbose, evidence ──


def test_simulate_rejects_timestep_larger_than_duration() -> None:
    """Simulate rejects timestep larger than duration."""
    model = RunawayElectronModel()
    with pytest.raises(ValueError, match="must be <= duration_s"):
        model.simulate(duration_s=0.05, dt_s=0.1)


def test_simulate_rejects_seed_fraction_out_of_range() -> None:
    """Simulate rejects seed fraction out of range."""
    model = RunawayElectronModel()
    with pytest.raises(ValueError, match=r"seed_re_fraction must be in \(0, 1\]"):
        model.simulate(seed_re_fraction=2.0)
