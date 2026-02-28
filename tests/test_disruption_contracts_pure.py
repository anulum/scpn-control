# Tests for pure-function disruption contracts (no RL agent needed).

import numpy as np
import pytest

from scpn_control.control.disruption_contracts import (
    impurity_transport_response,
    mcnp_lite_tbr,
    post_disruption_halo_runaway,
    require_finite_float,
    require_fraction,
    require_int,
    require_positive_float,
    synthetic_disruption_signal,
)


class TestRequireHelpers:
    def test_require_finite_float(self):
        assert require_finite_float("x", 1.5) == 1.5

    def test_require_finite_float_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            require_finite_float("x", float("nan"))

    def test_require_positive_float(self):
        assert require_positive_float("x", 2.0) == 2.0

    def test_require_positive_float_rejects_zero(self):
        with pytest.raises(ValueError, match="> 0"):
            require_positive_float("x", 0.0)

    def test_require_int(self):
        assert require_int("n", 5, minimum=1) == 5

    def test_require_int_rejects_bool(self):
        with pytest.raises(ValueError, match="integer"):
            require_int("n", True, minimum=0)

    def test_require_int_rejects_below_minimum(self):
        with pytest.raises(ValueError, match="integer >= 3"):
            require_int("n", 2, minimum=3)

    def test_require_fraction(self):
        assert require_fraction("f", 0.5) == 0.5

    def test_require_fraction_rejects_negative(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            require_fraction("f", -0.1)

    def test_require_fraction_rejects_above_one(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            require_fraction("f", 1.5)


class TestSyntheticDisruptionSignal:
    def test_shape(self):
        rng = np.random.default_rng(42)
        signal, toroidal = synthetic_disruption_signal(rng=rng, disturbance=0.5)
        assert signal.shape == (220,)
        assert np.all(np.isfinite(signal))
        assert np.all(signal > 0)

    def test_custom_window(self):
        rng = np.random.default_rng(0)
        signal, _ = synthetic_disruption_signal(rng=rng, disturbance=0.0, window=100)
        assert signal.shape == (100,)

    def test_toroidal_keys(self):
        rng = np.random.default_rng(1)
        _, toroidal = synthetic_disruption_signal(rng=rng, disturbance=0.8)
        for key in [
            "toroidal_n1_amp",
            "toroidal_n2_amp",
            "toroidal_n3_amp",
            "toroidal_asymmetry_index",
            "toroidal_radial_spread",
        ]:
            assert key in toroidal
            assert np.isfinite(toroidal[key])

    def test_higher_disturbance_higher_n1(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        _, t_low = synthetic_disruption_signal(rng=rng1, disturbance=0.0)
        _, t_high = synthetic_disruption_signal(rng=rng2, disturbance=1.0)
        assert t_high["toroidal_n1_amp"] > t_low["toroidal_n1_amp"]


class TestMcnpLiteTbr:
    def test_nominal(self):
        tbr, factor = mcnp_lite_tbr(
            base_tbr=1.0, li6_enrichment=0.9, be_multiplier_fraction=0.6, reflector_albedo=0.5
        )
        assert np.isfinite(tbr)
        assert tbr > 0.0
        assert factor > 1.0

    def test_rejects_bad_base(self):
        with pytest.raises(ValueError, match="> 0"):
            mcnp_lite_tbr(
                base_tbr=0.0, li6_enrichment=0.9, be_multiplier_fraction=0.5, reflector_albedo=0.5
            )

    def test_higher_enrichment_higher_tbr(self):
        tbr_low, _ = mcnp_lite_tbr(
            base_tbr=1.0, li6_enrichment=0.0, be_multiplier_fraction=0.5, reflector_albedo=0.5
        )
        tbr_high, _ = mcnp_lite_tbr(
            base_tbr=1.0, li6_enrichment=1.0, be_multiplier_fraction=0.5, reflector_albedo=0.5
        )
        assert tbr_high > tbr_low


class TestImpurityTransportResponse:
    def test_nominal(self):
        result = impurity_transport_response(
            neon_quantity_mol=0.5,
            argon_quantity_mol=0.2,
            xenon_quantity_mol=0.05,
            disturbance=0.5,
            seed_shift=0,
        )
        assert "zeff_eff" in result
        assert "impurity_radiation_mw" in result
        assert "impurity_decay_tau_ms" in result
        assert result["zeff_eff"] >= 1.0
        assert result["impurity_radiation_mw"] > 0.0

    def test_zero_impurity(self):
        result = impurity_transport_response(
            neon_quantity_mol=0.0,
            argon_quantity_mol=0.0,
            xenon_quantity_mol=0.0,
            disturbance=0.0,
            seed_shift=0,
        )
        assert result["total_impurity_mol"] == 0.0

    def test_more_argon_higher_zeff(self):
        r1 = impurity_transport_response(
            neon_quantity_mol=0.5, argon_quantity_mol=0.0,
            xenon_quantity_mol=0.0, disturbance=0.5, seed_shift=0,
        )
        r2 = impurity_transport_response(
            neon_quantity_mol=0.0, argon_quantity_mol=0.5,
            xenon_quantity_mol=0.0, disturbance=0.5, seed_shift=0,
        )
        assert r2["zeff_eff"] > r1["zeff_eff"]


class TestPostDisruptionHaloRunaway:
    def test_nominal(self):
        result = post_disruption_halo_runaway(
            pre_current_ma=15.0,
            tau_cq_s=0.010,
            disturbance=0.5,
            mitigation_strength=0.5,
            zeff_eff=2.0,
        )
        assert "halo_current_ma" in result
        assert "runaway_beam_ma" in result
        assert result["halo_current_ma"] >= 0.0
        assert result["runaway_beam_ma"] >= 0.0

    def test_higher_mitigation_less_runaway(self):
        r_low = post_disruption_halo_runaway(
            pre_current_ma=15.0, tau_cq_s=0.010,
            disturbance=0.8, mitigation_strength=0.1, zeff_eff=1.5,
        )
        r_high = post_disruption_halo_runaway(
            pre_current_ma=15.0, tau_cq_s=0.010,
            disturbance=0.8, mitigation_strength=0.9, zeff_eff=1.5,
        )
        assert r_high["runaway_peak_ma"] <= r_low["runaway_peak_ma"]

    def test_zero_current(self):
        result = post_disruption_halo_runaway(
            pre_current_ma=0.0, tau_cq_s=0.010,
            disturbance=0.0, mitigation_strength=0.5, zeff_eff=1.0,
        )
        assert result["halo_peak_ma"] == pytest.approx(0.0, abs=1e-6)
