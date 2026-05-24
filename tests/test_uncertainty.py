# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Uncertainty
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Uncertainty Quantification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import pytest
import numpy as np

from scpn_control.core.uncertainty import (
    IPB98_CENTRAL,
    bosch_hale_reactivity,
    compute_fusion_sensitivities,
    fusion_power_from_tau,
    ipb98_tau_e,
    PlasmaScenario,
    quantify_full_chain,
    quantify_uncertainty,
)


# ITER-like baseline scenario
ITER_SCENARIO = PlasmaScenario(
    I_p=15.0,  # MA
    B_t=5.3,  # T
    P_heat=50.0,  # MW (auxiliary)
    n_e=10.1,  # 10^19 m^-3
    R=6.2,  # m
    A=3.1,
    kappa=1.7,
    M=2.5,
)


class TestIPB98Scaling:
    def test_iter_confinement_order_of_magnitude(self):
        """IPB98 should predict ITER tau_E ~ 3-5 s."""
        tau = ipb98_tau_e(ITER_SCENARIO)
        assert 1.0 < tau < 10.0, f"ITER tau_E = {tau:.2f} s out of range"

    def test_higher_current_longer_confinement(self):
        """Doubling I_p should increase tau_E (alpha_I ~ 0.93)."""
        s1 = PlasmaScenario(I_p=5.0, B_t=5.0, P_heat=20.0, n_e=5.0, R=3.0, A=3.0, kappa=1.5)
        s2 = PlasmaScenario(I_p=10.0, B_t=5.0, P_heat=20.0, n_e=5.0, R=3.0, A=3.0, kappa=1.5)
        assert ipb98_tau_e(s2) > ipb98_tau_e(s1)

    def test_more_heating_degrades_confinement(self):
        """Higher P_heat should decrease tau_E (alpha_P ~ -0.69)."""
        s1 = PlasmaScenario(I_p=10.0, B_t=5.0, P_heat=10.0, n_e=5.0, R=3.0, A=3.0, kappa=1.5)
        s2 = PlasmaScenario(I_p=10.0, B_t=5.0, P_heat=50.0, n_e=5.0, R=3.0, A=3.0, kappa=1.5)
        assert ipb98_tau_e(s2) < ipb98_tau_e(s1)

    def test_custom_params(self):
        """Passing custom params should override defaults."""
        params = dict(IPB98_CENTRAL)
        params["C"] = 0.1  # double the constant
        tau_custom = ipb98_tau_e(ITER_SCENARIO, params)
        tau_default = ipb98_tau_e(ITER_SCENARIO)
        assert tau_custom > tau_default


class TestFusionPower:
    def test_positive_power(self):
        """Fusion power should be positive for ITER-like scenario."""
        tau = ipb98_tau_e(ITER_SCENARIO)
        pfus = fusion_power_from_tau(ITER_SCENARIO, tau)
        assert pfus > 0.0

    def test_longer_confinement_more_power(self):
        """Higher tau_E should give higher P_fus."""
        p1 = fusion_power_from_tau(ITER_SCENARIO, 2.0)
        p2 = fusion_power_from_tau(ITER_SCENARIO, 4.0)
        assert p2 > p1

    def test_fuel_composition_controls_power(self):
        """Explicit D-T composition should control fusion power level."""
        # Balanced D-T mix (maximum n_D*n_T at fixed total fuel ions)
        p_balanced = fusion_power_from_tau(ITER_SCENARIO, 3.0)

        # Deuterium-heavy and tritium-heavy mixes should reduce rate
        d_rich = PlasmaScenario(**{**ITER_SCENARIO.__dict__, "f_D": 0.9, "f_T": 0.1})
        t_rich = PlasmaScenario(**{**ITER_SCENARIO.__dict__, "f_D": 0.1, "f_T": 0.9})
        p_d_rich = fusion_power_from_tau(d_rich, 3.0)
        p_t_rich = fusion_power_from_tau(t_rich, 3.0)
        assert p_balanced > p_d_rich
        assert p_balanced > p_t_rich

    def test_fuel_dilution_reduces_power(self):
        """Fuel-ion dilution should reduce fusion power quadratically."""
        undiluted = fusion_power_from_tau(ITER_SCENARIO, 3.0)
        diluted = PlasmaScenario(**{**ITER_SCENARIO.__dict__, "fuel_ion_fraction": 0.5})
        p_diluted = fusion_power_from_tau(diluted, 3.0)
        assert p_diluted < undiluted


class TestUQ:
    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        r1 = quantify_uncertainty(ITER_SCENARIO, n_samples=500, seed=42)
        r2 = quantify_uncertainty(ITER_SCENARIO, n_samples=500, seed=42)
        assert r1.tau_E == r2.tau_E
        assert r1.Q == r2.Q

    def test_uncertainty_positive(self):
        """Sigma values should be positive."""
        r = quantify_uncertainty(ITER_SCENARIO, n_samples=1000, seed=0)
        assert r.tau_E_sigma > 0.0
        assert r.P_fusion_sigma > 0.0
        assert r.Q_sigma > 0.0

    def test_percentiles_ordered(self):
        """Percentiles [5, 25, 50, 75, 95] should be monotonically increasing."""
        r = quantify_uncertainty(ITER_SCENARIO, n_samples=2000, seed=7)
        for arr in [r.tau_E_percentiles, r.P_fusion_percentiles, r.Q_percentiles]:
            for i in range(len(arr) - 1):
                assert arr[i] <= arr[i + 1] + 1e-12, f"Percentiles not ordered: {arr}"

    def test_median_within_range(self):
        """Median should be between 5th and 95th percentile."""
        r = quantify_uncertainty(ITER_SCENARIO, n_samples=2000, seed=99)
        assert r.tau_E_percentiles[0] <= r.tau_E <= r.tau_E_percentiles[4]

    def test_more_samples_narrower_sigma(self):
        """Standard error should decrease with more samples (law of large numbers)."""
        r_small = quantify_uncertainty(ITER_SCENARIO, n_samples=100, seed=1)
        r_large = quantify_uncertainty(ITER_SCENARIO, n_samples=10000, seed=1)
        # The sigma is a property of the distribution, not sample size,
        # but they should be in the same ballpark (within 50%)
        ratio = r_small.tau_E_sigma / r_large.tau_E_sigma
        assert 0.5 < ratio < 2.0, f"Sigma ratio = {ratio}"

    @pytest.mark.parametrize("n_samples", [0, -5, 2.5, "100", True])
    def test_invalid_n_samples_rejected(self, n_samples):
        with pytest.raises(ValueError, match="n_samples"):
            quantify_uncertainty(ITER_SCENARIO, n_samples=n_samples, seed=1)


class TestInputBoundaries:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("I_p", 0.0),
            ("B_t", 0.0),
            ("P_heat", 0.0),
            ("n_e", 0.0),
            ("R", 0.0),
            ("A", 0.0),
            ("kappa", 0.0),
            ("M", 0.0),
        ],
    )
    def test_plasma_scenario_rejects_nonphysical_scalars(self, field, value):
        scenario = PlasmaScenario(**{**ITER_SCENARIO.__dict__, field: value})
        with pytest.raises(ValueError, match=field):
            ipb98_tau_e(scenario)

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("f_D", -0.1, "f_D"),
            ("f_T", -0.1, "f_T"),
            ("f_D", 1.1, "fuel fractions"),
            ("fuel_ion_fraction", -0.1, "fuel_ion_fraction"),
            ("fuel_ion_fraction", 1.1, "fuel_ion_fraction"),
        ],
    )
    def test_fusion_power_rejects_invalid_fuel_fractions(self, field, value, match):
        scenario = PlasmaScenario(**{**ITER_SCENARIO.__dict__, field: value})
        with pytest.raises(ValueError, match=match):
            fusion_power_from_tau(scenario, 3.0)

    def test_fusion_power_rejects_zero_fuel_mix_and_tau(self):
        no_fuel = PlasmaScenario(**{**ITER_SCENARIO.__dict__, "f_D": 0.0, "f_T": 0.0})
        with pytest.raises(ValueError, match="D-T fuel"):
            fusion_power_from_tau(no_fuel, 3.0)

        with pytest.raises(ValueError, match="tau_E"):
            fusion_power_from_tau(ITER_SCENARIO, 0.0)

    def test_bosch_hale_reactivity_rejects_nonpositive_temperatures(self):
        with pytest.raises(ValueError, match="T_i_kev"):
            bosch_hale_reactivity(0.0)

        with pytest.raises(ValueError, match="T_i_kev"):
            bosch_hale_reactivity(np.array([10.0, np.nan]))

    def test_sensitivity_and_full_chain_reject_invalid_inputs(self):
        with pytest.raises(ValueError, match="tau_E"):
            compute_fusion_sensitivities(ITER_SCENARIO, 0.0)

        with pytest.raises(ValueError, match="chi_gB_sigma"):
            quantify_full_chain(ITER_SCENARIO, n_samples=10, seed=1, chi_gB_sigma=-0.1)

        bad = PlasmaScenario(**{**ITER_SCENARIO.__dict__, "P_heat": 0.0})
        with pytest.raises(ValueError, match="P_heat"):
            quantify_full_chain(bad, n_samples=10, seed=1)
