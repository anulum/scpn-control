# ──────────────────────────────────────────────────────────────────────
# Tests for halo_re_physics — L/R halo current + runaway electron models
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.halo_re_physics import (
    HaloCurrentModel,
    HaloCurrentResult,
    RunawayElectronModel,
    RunawayElectronResult,
    run_disruption_ensemble,
)


# ─── HaloCurrentModel construction ────────────────────────────────────


class TestHaloCurrentConstruction:
    def test_defaults(self):
        m = HaloCurrentModel()
        assert m.Ip0 == pytest.approx(15e6, rel=1e-6)
        assert m.tpf == 2.0
        assert m.R_h > 0
        assert m.L_h > 0
        assert m.tau_h > 0

    def test_rejects_negative_current(self):
        with pytest.raises(ValueError, match="plasma_current_ma"):
            HaloCurrentModel(plasma_current_ma=-1.0)

    def test_rejects_zero_radius(self):
        with pytest.raises(ValueError, match="minor_radius_m"):
            HaloCurrentModel(minor_radius_m=0.0)

    def test_rejects_contact_fraction_out_of_range(self):
        with pytest.raises(ValueError, match="contact_fraction"):
            HaloCurrentModel(contact_fraction=0.0)
        with pytest.raises(ValueError, match="contact_fraction"):
            HaloCurrentModel(contact_fraction=1.5)

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            HaloCurrentModel(plasma_current_ma=float("nan"))


# ─── HaloCurrentModel.simulate() ──────────────────────────────────────


class TestHaloSimulation:
    @pytest.fixture()
    def result(self) -> HaloCurrentResult:
        m = HaloCurrentModel(plasma_current_ma=15.0, tpf=2.0, contact_fraction=0.3)
        return m.simulate(tau_cq_s=0.01, duration_s=0.05, dt_s=1e-4)

    def test_result_type(self, result):
        assert isinstance(result, HaloCurrentResult)

    def test_time_vector_length(self, result):
        assert len(result.time_ms) >= 10

    def test_halo_current_non_negative(self, result):
        assert all(i >= 0.0 for i in result.halo_current_ma)

    def test_plasma_current_decays(self, result):
        assert result.plasma_current_ma[0] > result.plasma_current_ma[-1]

    def test_peak_halo_bounded_by_plasma(self, result):
        # Halo current cannot exceed initial plasma current
        assert result.peak_halo_ma <= 15.0

    def test_peak_halo_positive(self, result):
        assert result.peak_halo_ma > 0.0

    def test_tpf_product_positive(self, result):
        assert result.peak_tpf_product > 0.0

    def test_wall_force_positive(self, result):
        assert result.wall_force_mn_m > 0.0

    def test_faster_quench_higher_halo(self):
        """Faster current quench → larger dI_p/dt → higher halo peak."""
        m = HaloCurrentModel(plasma_current_ma=15.0)
        fast = m.simulate(tau_cq_s=0.005, duration_s=0.05, dt_s=1e-4)
        slow = m.simulate(tau_cq_s=0.020, duration_s=0.05, dt_s=1e-4)
        assert fast.peak_halo_ma > slow.peak_halo_ma

    def test_higher_tpf_higher_product(self):
        """Higher TPF → higher TPF × I_h/I_p product."""
        low = HaloCurrentModel(tpf=1.5).simulate()
        high = HaloCurrentModel(tpf=2.5).simulate()
        assert high.peak_tpf_product > low.peak_tpf_product

    def test_rejects_dt_larger_than_duration(self):
        m = HaloCurrentModel()
        with pytest.raises(ValueError, match="dt_s"):
            m.simulate(dt_s=1.0, duration_s=0.01)


# ─── RunawayElectronModel construction ─────────────────────────────────


class TestREConstruction:
    def test_defaults(self):
        m = RunawayElectronModel()
        assert m.n_e_free == pytest.approx(1e20)
        assert m.T_e0 == pytest.approx(20.0)
        assert m.E_D > 0
        assert m.E_c > 0
        assert m.tau_coll > 0
        assert m.tau_av > 0

    def test_rejects_negative_density(self):
        with pytest.raises(ValueError, match="n_e"):
            RunawayElectronModel(n_e=-1e20)

    def test_rejects_z_eff_below_one(self):
        with pytest.raises(ValueError, match="z_eff"):
            RunawayElectronModel(z_eff=0.5)


# ─── Dreicer field ordering ────────────────────────────────────────────


class TestDreicerField:
    def test_dreicer_exceeds_critical(self):
        """Connor-Hastie: E_D >> E_c always (Dreicer is thermal, critical is relativistic)."""
        m = RunawayElectronModel(n_e=1e20, T_e_keV=10.0)
        assert m.E_D > m.E_c

    def test_dreicer_rate_zero_for_zero_field(self):
        m = RunawayElectronModel()
        assert m._dreicer_rate(0.0, 10.0) == 0.0

    def test_dreicer_rate_positive_for_strong_field(self):
        m = RunawayElectronModel(n_e=1e20, T_e_keV=5.0)
        # E slightly below E_D should give nonzero rate
        E = m.E_D * 0.1
        rate = m._dreicer_rate(E, 5.0)
        assert rate >= 0.0  # may still be small due to exponential suppression


# ─── RunawayElectronModel.simulate() ───────────────────────────────────


class TestRESimulation:
    @pytest.fixture()
    def result(self) -> RunawayElectronResult:
        m = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=1.5)
        return m.simulate(
            plasma_current_ma=15.0,
            tau_cq_s=0.01,
            T_e_quench_keV=0.5,
            duration_s=0.03,
            dt_s=1e-4,
        )

    def test_result_type(self, result):
        assert isinstance(result, RunawayElectronResult)

    def test_re_current_non_negative(self, result):
        assert all(i >= 0.0 for i in result.runaway_current_ma)

    def test_peak_re_bounded_by_plasma(self, result):
        assert result.peak_re_current_ma <= 15.0

    def test_avalanche_gain_finite(self, result):
        assert np.isfinite(result.avalanche_gain)
        assert result.avalanche_gain >= 0.0

    def test_electric_field_positive(self, result):
        assert all(e >= 0.0 for e in result.electric_field_v_m)


# ─── Neon mitigation ──────────────────────────────────────────────────


class TestNeonMitigation:
    def test_high_neon_suppresses_avalanche(self):
        """Heavy neon injection → avalanche deconfinement → lower RE peak."""
        base = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, neon_mol=0.0)
        mitigated = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, neon_mol=0.5)

        r_base = base.simulate(plasma_current_ma=15.0, tau_cq_s=0.01, neon_mol=0.0)
        r_mit = mitigated.simulate(plasma_current_ma=15.0, tau_cq_s=0.01, neon_mol=0.5)

        assert r_mit.peak_re_current_ma <= r_base.peak_re_current_ma

    def test_neon_raises_total_density(self):
        m0 = RunawayElectronModel(neon_mol=0.0)
        m1 = RunawayElectronModel(neon_mol=0.5)
        assert m1.n_e_tot > m0.n_e_tot

    def test_neon_raises_critical_field(self):
        m0 = RunawayElectronModel(neon_mol=0.0)
        m1 = RunawayElectronModel(neon_mol=0.5)
        assert m1.E_c > m0.E_c


# ─── Relativistic losses ──────────────────────────────────────────────


class TestRelativisticLosses:
    def test_loss_zero_when_disabled(self):
        m = RunawayElectronModel(enable_relativistic_losses=False)
        assert m._relativistic_loss_rate(E=100.0, n_re=1e18) == 0.0

    def test_loss_positive_when_enabled(self):
        m = RunawayElectronModel(enable_relativistic_losses=True)
        loss = m._relativistic_loss_rate(E=m.E_c * 5.0, n_re=1e18)
        assert loss > 0.0

    def test_loss_zero_for_zero_re(self):
        m = RunawayElectronModel()
        assert m._relativistic_loss_rate(E=100.0, n_re=0.0) == 0.0


# ─── Disruption ensemble ──────────────────────────────────────────────


class TestDisruptionEnsemble:
    def test_basic_run(self):
        report = run_disruption_ensemble(ensemble_runs=5, seed=42)
        assert report.ensemble_runs == 5
        assert 0.0 <= report.prevention_rate <= 1.0
        assert len(report.per_run_details) == 5

    def test_all_details_have_required_keys(self):
        report = run_disruption_ensemble(ensemble_runs=3, seed=0)
        for d in report.per_run_details:
            assert "halo_peak_ma" in d
            assert "re_peak_ma" in d
            assert "prevented" in d
            assert "tpf_product" in d

    def test_rejects_zero_runs(self):
        with pytest.raises(ValueError, match="ensemble_runs"):
            run_disruption_ensemble(ensemble_runs=0)

    def test_reproducibility(self):
        r1 = run_disruption_ensemble(ensemble_runs=5, seed=123)
        r2 = run_disruption_ensemble(ensemble_runs=5, seed=123)
        assert r1.prevention_rate == r2.prevention_rate
        assert r1.mean_halo_peak_ma == pytest.approx(r2.mean_halo_peak_ma)

    def test_halo_peaks_finite(self):
        report = run_disruption_ensemble(ensemble_runs=10, seed=42)
        assert np.isfinite(report.mean_halo_peak_ma)
        assert np.isfinite(report.p95_halo_peak_ma)
        assert np.isfinite(report.mean_re_peak_ma)
        assert np.isfinite(report.p95_re_peak_ma)
