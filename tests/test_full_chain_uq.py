# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Full-Chain Uncertainty Quantification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
import pytest
import numpy as np

from scpn_control.core.uncertainty import (
    PlasmaScenario,
    FullChainUQResult,
    quantify_full_chain,
    summarize_uq,
)

ITER = PlasmaScenario(
    I_p=15.0, B_t=5.3, P_heat=50.0, n_e=10.1,
    R=6.2, A=3.1, kappa=1.7, M=2.5,
)


class TestQuantifyFullChain:

    def test_deterministic_with_seed(self):
        r1 = quantify_full_chain(ITER, n_samples=200, seed=42)
        r2 = quantify_full_chain(ITER, n_samples=200, seed=42)
        assert r1.tau_E == r2.tau_E
        assert r1.Q == r2.Q
        assert np.array_equal(r1.tau_E_bands, r2.tau_E_bands)

    def test_central_estimates_positive(self):
        r = quantify_full_chain(ITER, n_samples=300, seed=0)
        assert r.tau_E > 0
        assert r.P_fusion > 0
        assert r.Q > 0

    def test_sigma_positive(self):
        r = quantify_full_chain(ITER, n_samples=500, seed=1)
        assert r.tau_E_sigma > 0
        assert r.P_fusion_sigma > 0
        assert r.Q_sigma > 0

    def test_bands_shape(self):
        r = quantify_full_chain(ITER, n_samples=200, seed=2)
        for arr in [r.psi_nrmse_bands, r.tau_E_bands, r.P_fusion_bands,
                     r.Q_bands, r.beta_N_bands]:
            assert arr.shape == (3,), f"Expected (3,), got {arr.shape}"

    def test_percentiles_shape(self):
        r = quantify_full_chain(ITER, n_samples=200, seed=3)
        for arr in [r.tau_E_percentiles, r.P_fusion_percentiles, r.Q_percentiles]:
            assert arr.shape == (5,), f"Expected (5,), got {arr.shape}"

    def test_bands_ordered(self):
        """5th <= 50th <= 95th percentile."""
        r = quantify_full_chain(ITER, n_samples=1000, seed=4)
        for arr in [r.tau_E_bands, r.P_fusion_bands, r.Q_bands, r.beta_N_bands]:
            assert arr[0] <= arr[1] <= arr[2], f"Bands not ordered: {arr}"

    def test_percentiles_ordered(self):
        """5th <= 25th <= 50th <= 75th <= 95th."""
        r = quantify_full_chain(ITER, n_samples=1000, seed=5)
        for arr in [r.tau_E_percentiles, r.P_fusion_percentiles, r.Q_percentiles]:
            for i in range(len(arr) - 1):
                assert arr[i] <= arr[i + 1] + 1e-12

    def test_psi_nrmse_nonnegative(self):
        r = quantify_full_chain(ITER, n_samples=300, seed=6)
        assert np.all(r.psi_nrmse_bands >= 0)

    def test_beta_n_positive(self):
        r = quantify_full_chain(ITER, n_samples=300, seed=7)
        assert np.all(r.beta_N_bands > 0)

    def test_n_samples_stored(self):
        r = quantify_full_chain(ITER, n_samples=123, seed=8)
        assert r.n_samples == 123

    def test_larger_chi_sigma_widens_tau_spread(self):
        r_tight = quantify_full_chain(ITER, n_samples=2000, seed=9, chi_gB_sigma=0.1)
        r_wide = quantify_full_chain(ITER, n_samples=2000, seed=9, chi_gB_sigma=0.8)
        assert r_wide.tau_E_sigma > r_tight.tau_E_sigma

    def test_zero_sigmas_narrow_spread(self):
        r = quantify_full_chain(
            ITER, n_samples=500, seed=10,
            chi_gB_sigma=0.0, pedestal_sigma=0.0, boundary_sigma=0.0,
        )
        # With zero transport/pedestal/boundary perturbation, spread comes only from
        # IPB98 coefficient sampling
        assert r.tau_E_sigma > 0  # still nonzero from scaling-law sampling

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_negative_sigma_rejected(self, bad):
        with pytest.raises(ValueError):
            quantify_full_chain(ITER, n_samples=10, seed=0, chi_gB_sigma=bad)


class TestSummarizeUQ:

    def test_returns_dict(self):
        r = quantify_full_chain(ITER, n_samples=100, seed=20)
        d = summarize_uq(r)
        assert isinstance(d, dict)
        assert "central" in d
        assert "sigma" in d
        assert "bands_5_50_95" in d

    def test_json_serialisable(self):
        import json
        r = quantify_full_chain(ITER, n_samples=100, seed=21)
        d = summarize_uq(r)
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_n_samples_present(self):
        r = quantify_full_chain(ITER, n_samples=77, seed=22)
        d = summarize_uq(r)
        assert d["n_samples"] == 77

    def test_bands_have_three_elements(self):
        r = quantify_full_chain(ITER, n_samples=100, seed=23)
        d = summarize_uq(r)
        for key in ["psi_nrmse", "tau_E_s", "P_fusion_MW", "Q", "beta_N"]:
            assert len(d["bands_5_50_95"][key]) == 3
