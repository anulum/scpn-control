# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MHD Stability Tests

"""Tests for scpn_control.core.stability_mhd."""

import numpy as np
import pytest

from scpn_control.core.stability_mhd import (
    QProfile,
    compute_q_profile,
    mercier_stability,
    ballooning_stability,
    kruskal_shafranov_stability,
    troyon_beta_limit,
    ntm_stability,
    run_full_stability_check,
)


@pytest.fixture
def iter_like_qprofile():
    """ITER-like q-profile from cylindrical approximation."""
    n = 101
    rho = np.linspace(0, 1, n)
    ne = 10.0 * (1 - 0.8 * rho**2)
    Ti = 15.0 * (1 - rho**2) ** 1.5
    Te = 15.0 * (1 - rho**2) ** 1.5
    return compute_q_profile(rho, ne, Ti, Te, R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, kappa=1.7, delta=0.3)


class TestComputeQProfile:
    def test_q_axis_positive(self, iter_like_qprofile):
        assert iter_like_qprofile.q[0] > 0

    def test_q_edge_greater_than_one(self, iter_like_qprofile):
        assert iter_like_qprofile.q_edge > 1.0

    def test_shear_zero_at_axis(self, iter_like_qprofile):
        assert iter_like_qprofile.shear[0] == 0.0

    def test_alpha_mhd_nonnegative(self, iter_like_qprofile):
        assert np.all(iter_like_qprofile.alpha_mhd >= 0.0)

    def test_rho_length(self, iter_like_qprofile):
        assert len(iter_like_qprofile.rho) == 101


class TestMercier:
    def test_returns_correct_shape(self, iter_like_qprofile):
        result = mercier_stability(iter_like_qprofile)
        assert result.D_M.shape == iter_like_qprofile.rho.shape

    def test_stable_array_is_boolean(self, iter_like_qprofile):
        result = mercier_stability(iter_like_qprofile)
        assert result.stable.dtype == np.bool_

    def test_reports_first_unstable_radius_for_unstable_shear_band(self):
        rho = np.linspace(0.0, 1.0, 50)
        q = 1.0 + 2.0 * rho
        shear = 2.0 * rho / (1.0 + 2.0 * rho)
        alpha = np.zeros_like(rho)
        qp = QProfile(
            rho=rho,
            q=q,
            shear=shear,
            alpha_mhd=alpha,
            q_min=float(q.min()),
            q_min_rho=float(rho[np.argmin(q)]),
            q_edge=float(q[-1]),
        )

        result = mercier_stability(qp)

        assert result.first_unstable_rho is not None
        assert 0.0 < result.first_unstable_rho <= 1.0


class TestBallooning:
    def test_returns_margin(self, iter_like_qprofile):
        result = ballooning_stability(iter_like_qprofile)
        assert result.margin.shape == iter_like_qprofile.rho.shape

    def test_alpha_crit_nonneg(self, iter_like_qprofile):
        result = ballooning_stability(iter_like_qprofile)
        assert np.all(result.alpha_crit >= 0.0)


class TestKruskalShafranov:
    def test_stable_for_high_q_edge(self, iter_like_qprofile):
        result = kruskal_shafranov_stability(iter_like_qprofile)
        assert result.stable
        assert result.margin > 0.0

    def test_unstable_for_low_q_edge(self):
        qp = QProfile(
            rho=np.array([0.0, 1.0]),
            q=np.array([0.5, 0.8]),
            shear=np.array([0.0, 0.5]),
            alpha_mhd=np.array([0.0, 0.0]),
            q_min=0.5,
            q_min_rho=0.0,
            q_edge=0.8,
        )
        result = kruskal_shafranov_stability(qp)
        assert not result.stable
        assert result.margin < 0.0


class TestTroyon:
    def test_low_beta_stable(self):
        result = troyon_beta_limit(beta_t=0.01, Ip_MA=15.0, a=2.0, B0=5.3)
        assert result.stable_nowall

    def test_high_beta_unstable(self):
        result = troyon_beta_limit(beta_t=0.15, Ip_MA=1.0, a=0.5, B0=2.0)
        assert not result.stable_nowall

    def test_beta_n_positive(self):
        result = troyon_beta_limit(beta_t=0.025, Ip_MA=15.0, a=2.0, B0=5.3)
        assert result.beta_N > 0.0


class TestNTM:
    def test_returns_correct_shape(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.linspace(0.0, 0.3, n)
        j_total = np.ones(n) * 1.0
        result = ntm_stability(iter_like_qprofile, j_bs, j_total, a=2.0)
        assert result.delta_prime.shape == (n,)
        assert result.w_marginal.shape == (n,)

    def test_zero_bootstrap_no_ntm(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.zeros(n)
        j_total = np.ones(n)
        result = ntm_stability(iter_like_qprofile, j_bs, j_total, a=2.0)
        assert not np.any(result.ntm_unstable)
        assert result.most_unstable_rho is None

    def test_positive_bootstrap_triggers_ntm(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.ones(n) * 0.5
        j_total = np.ones(n) * 1.0
        result = ntm_stability(iter_like_qprofile, j_bs, j_total, a=2.0)
        assert np.any(result.ntm_unstable)
        assert result.most_unstable_rho is not None

    def test_w_marginal_nonneg(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.linspace(0.0, 1.0, n)
        j_total = np.ones(n) * 2.0
        result = ntm_stability(iter_like_qprofile, j_bs, j_total, a=2.0)
        assert np.all(result.w_marginal >= 0.0)

    def test_positive_delta_prime_no_ntm(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.ones(n) * 0.5
        j_total = np.ones(n) * 1.0
        result = ntm_stability(
            iter_like_qprofile,
            j_bs,
            j_total,
            a=2.0,
            r_s_delta_prime=2.0,
        )
        assert not np.any(result.ntm_unstable)

    def test_higher_shear_reduces_marginal_width(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.ones(n) * 0.5
        j_total = np.ones(n)

        qp_low_shear = QProfile(
            rho=iter_like_qprofile.rho,
            q=iter_like_qprofile.q,
            shear=np.full(n, 0.2),
            alpha_mhd=iter_like_qprofile.alpha_mhd,
            q_min=iter_like_qprofile.q_min,
            q_min_rho=iter_like_qprofile.q_min_rho,
            q_edge=iter_like_qprofile.q_edge,
        )
        qp_high_shear = QProfile(
            rho=iter_like_qprofile.rho,
            q=iter_like_qprofile.q,
            shear=np.full(n, 2.0),
            alpha_mhd=iter_like_qprofile.alpha_mhd,
            q_min=iter_like_qprofile.q_min,
            q_min_rho=iter_like_qprofile.q_min_rho,
            q_edge=iter_like_qprofile.q_edge,
        )

        w_low = ntm_stability(qp_low_shear, j_bs, j_total, a=2.0).w_marginal
        w_high = ntm_stability(qp_high_shear, j_bs, j_total, a=2.0).w_marginal
        assert float(np.mean(w_low)) > float(np.mean(w_high))

    def test_higher_alpha_increases_drive(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.ones(n) * 0.5
        j_total = np.ones(n)

        qp_low_alpha = QProfile(
            rho=iter_like_qprofile.rho,
            q=iter_like_qprofile.q,
            shear=iter_like_qprofile.shear,
            alpha_mhd=np.full(n, 0.2),
            q_min=iter_like_qprofile.q_min,
            q_min_rho=iter_like_qprofile.q_min_rho,
            q_edge=iter_like_qprofile.q_edge,
        )
        qp_high_alpha = QProfile(
            rho=iter_like_qprofile.rho,
            q=iter_like_qprofile.q,
            shear=iter_like_qprofile.shear,
            alpha_mhd=np.full(n, 5.0),
            q_min=iter_like_qprofile.q_min,
            q_min_rho=iter_like_qprofile.q_min_rho,
            q_edge=iter_like_qprofile.q_edge,
        )

        r_low = ntm_stability(qp_low_alpha, j_bs, j_total, a=2.0)
        r_high = ntm_stability(qp_high_alpha, j_bs, j_total, a=2.0)
        assert float(np.mean(r_high.j_bs_drive)) > float(np.mean(r_low.j_bs_drive))


class TestRunFullStabilityCheck:
    def test_three_criteria_only(self, iter_like_qprofile):
        summary = run_full_stability_check(iter_like_qprofile)
        assert summary.n_criteria_checked == 3
        assert summary.troyon is None
        assert summary.ntm is None

    def test_with_troyon(self, iter_like_qprofile):
        summary = run_full_stability_check(
            iter_like_qprofile,
            beta_t=0.01,
            Ip_MA=15.0,
            a=2.0,
            B0=5.3,
        )
        assert summary.n_criteria_checked == 4
        assert summary.troyon is not None
        assert summary.troyon.beta_N > 0.0

    def test_with_ntm(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.linspace(0.0, 0.3, n)
        j_total = np.ones(n) * 1.0
        summary = run_full_stability_check(
            iter_like_qprofile,
            a=2.0,
            j_bs=j_bs,
            j_total=j_total,
        )
        assert summary.n_criteria_checked == 4
        assert summary.ntm is not None

    def test_all_five_criteria(self, iter_like_qprofile):
        n = len(iter_like_qprofile.rho)
        j_bs = np.zeros(n)
        j_total = np.ones(n) * 1.0
        summary = run_full_stability_check(
            iter_like_qprofile,
            beta_t=0.01,
            Ip_MA=15.0,
            a=2.0,
            B0=5.3,
            j_bs=j_bs,
            j_total=j_total,
        )
        assert summary.n_criteria_checked == 5
        assert summary.n_criteria_stable >= 3  # KS, Troyon, NTM stable

    def test_overall_unstable_when_kink(self):
        qp = QProfile(
            rho=np.array([0.0, 1.0]),
            q=np.array([0.5, 0.8]),
            shear=np.array([0.0, 0.5]),
            alpha_mhd=np.array([0.0, 0.0]),
            q_min=0.5,
            q_min_rho=0.0,
            q_edge=0.8,
        )
        summary = run_full_stability_check(qp)
        assert not summary.overall_stable
        assert not summary.kruskal_shafranov.stable

    def test_n_stable_leq_n_checked(self, iter_like_qprofile):
        summary = run_full_stability_check(iter_like_qprofile)
        assert summary.n_criteria_stable <= summary.n_criteria_checked


class TestStabilityInputBoundaries:
    """Non-physical MHD stability inputs fail closed instead of being regularised."""

    def test_compute_q_profile_rejects_unsorted_radius(self):
        rho = np.array([0.0, 0.4, 0.3, 1.0])
        ne = np.ones_like(rho) * 1e20
        Ti = np.ones_like(rho) * 2e3
        Te = np.ones_like(rho) * 1e3

        with pytest.raises(ValueError, match="strictly increasing"):
            compute_q_profile(rho, ne, Ti, Te, R0=3.0, a=1.0, B0=5.0, Ip_MA=10.0)

    def test_compute_q_profile_requires_full_normalised_radius_domain(self):
        rho_missing_axis = np.array([0.1, 0.4, 0.7, 1.0])
        rho_missing_edge = np.array([0.0, 0.3, 0.6, 0.9])
        ne = np.ones(4) * 1e20
        Ti = np.ones(4) * 2e3
        Te = np.ones(4) * 1e3

        with pytest.raises(ValueError, match="rho must start at 0"):
            compute_q_profile(rho_missing_axis, ne, Ti, Te, R0=3.0, a=1.0, B0=5.0, Ip_MA=10.0)
        with pytest.raises(ValueError, match="rho must end at 1"):
            compute_q_profile(rho_missing_edge, ne, Ti, Te, R0=3.0, a=1.0, B0=5.0, Ip_MA=10.0)

    def test_compute_q_profile_rejects_nonphysical_profiles(self):
        rho = np.linspace(0.0, 1.0, 5)
        ne = np.ones_like(rho) * 1e20
        Ti = np.ones_like(rho) * 2e3
        Te = np.ones_like(rho) * 1e3
        ne[2] = 0.0

        with pytest.raises(ValueError, match="ne"):
            compute_q_profile(rho, ne, Ti, Te, R0=3.0, a=1.0, B0=5.0, Ip_MA=10.0)

        with pytest.raises(ValueError, match="Ti"):
            compute_q_profile(rho, np.ones_like(rho) * 1e20, -Ti, Te, R0=3.0, a=1.0, B0=5.0, Ip_MA=10.0)

    def test_compute_q_profile_rejects_invalid_geometry_and_shape(self):
        rho = np.linspace(0.0, 1.0, 5)
        ne = np.ones_like(rho) * 1e20
        Ti = np.ones_like(rho) * 2e3
        Te = np.ones_like(rho) * 1e3

        with pytest.raises(ValueError, match="same|match"):
            compute_q_profile(rho, ne[:-1], Ti, Te, R0=3.0, a=1.0, B0=5.0, Ip_MA=10.0)

        with pytest.raises(ValueError, match="a must be smaller"):
            compute_q_profile(rho, ne, Ti, Te, R0=1.0, a=1.0, B0=5.0, Ip_MA=10.0)

        with pytest.raises(ValueError, match="delta"):
            compute_q_profile(rho, ne, Ti, Te, R0=3.0, a=1.0, B0=5.0, Ip_MA=10.0, delta=1.0)

    def test_stability_criteria_reject_inconsistent_q_profile(self, iter_like_qprofile):
        qp = iter_like_qprofile
        broken = QProfile(
            rho=np.array([0.0, 0.6, 0.4, 1.0]),
            q=qp.q[:4],
            shear=qp.shear[:4],
            alpha_mhd=qp.alpha_mhd[:4],
            q_min=float(np.min(qp.q[:4])),
            q_min_rho=float(qp.rho[:4][np.argmin(qp.q[:4])]),
            q_edge=float(qp.q[3]),
        )

        with pytest.raises(ValueError, match="strictly increasing"):
            mercier_stability(broken)
        with pytest.raises(ValueError, match="strictly increasing"):
            ballooning_stability(broken)
        with pytest.raises(ValueError, match="strictly increasing"):
            kruskal_shafranov_stability(broken)

    def test_stability_criteria_reject_partial_radius_domain(self):
        rho = np.array([0.1, 0.4, 0.7, 1.0])
        q = np.array([1.2, 1.4, 1.8, 2.2])
        qp = QProfile(
            rho=rho,
            q=q,
            shear=np.array([0.0, 0.3, 0.5, 0.7]),
            alpha_mhd=np.zeros_like(rho),
            q_min=float(q.min()),
            q_min_rho=float(rho[np.argmin(q)]),
            q_edge=float(q[-1]),
        )

        with pytest.raises(ValueError, match="rho must start at 0"):
            run_full_stability_check(qp)

    def test_stability_criteria_reject_wrong_q_min_radius_metadata(self, iter_like_qprofile):
        qp = iter_like_qprofile
        true_min_idx = int(np.argmin(qp.q))
        wrong_min_idx = 0 if true_min_idx != 0 else len(qp.rho) - 1
        broken = QProfile(
            rho=qp.rho,
            q=qp.q,
            shear=qp.shear,
            alpha_mhd=qp.alpha_mhd,
            q_min=float(np.min(qp.q)),
            q_min_rho=float(qp.rho[wrong_min_idx]),
            q_edge=float(qp.q[-1]),
        )

        with pytest.raises(ValueError, match="q_min_rho"):
            run_full_stability_check(broken)

    def test_troyon_beta_limit_rejects_nonphysical_inputs(self):
        with pytest.raises(ValueError, match="beta_t"):
            troyon_beta_limit(-0.01, Ip_MA=10.0, a=1.0, B0=5.0)
        with pytest.raises(ValueError, match="beta_t must not exceed 1"):
            troyon_beta_limit(1.1, Ip_MA=10.0, a=1.0, B0=5.0)
        with pytest.raises(ValueError, match="Ip_MA"):
            troyon_beta_limit(0.01, Ip_MA=0.0, a=1.0, B0=5.0)
        with pytest.raises(ValueError, match="g_wall"):
            troyon_beta_limit(0.01, Ip_MA=10.0, a=1.0, B0=5.0, g_nowall=3.5, g_wall=2.8)

    def test_ntm_stability_rejects_current_and_shape_errors(self, iter_like_qprofile):
        qp = iter_like_qprofile
        j_bs = np.full_like(qp.rho, 1.0e5)
        j_total = np.full_like(qp.rho, 1.0e6)

        with pytest.raises(ValueError, match="j_bs"):
            ntm_stability(qp, -j_bs, j_total, a=1.0)
        with pytest.raises(ValueError, match="j_total"):
            ntm_stability(qp, j_bs, np.zeros_like(j_total), a=1.0)
        with pytest.raises(ValueError, match="shape|match"):
            ntm_stability(qp, j_bs[:-1], j_total, a=1.0)
        with pytest.raises(ValueError, match="a"):
            ntm_stability(qp, j_bs, j_total, a=0.0)

    def test_ntm_stability_rejects_singular_tearing_boundary(self, iter_like_qprofile):
        qp = iter_like_qprofile
        j_bs = np.full_like(qp.rho, 1.0e5)
        j_total = np.full_like(qp.rho, 1.0e6)

        with pytest.raises(ValueError, match="r_s_delta_prime"):
            ntm_stability(qp, j_bs, j_total, a=1.0, r_s_delta_prime=0.0)

    def test_full_stability_check_rejects_partial_optional_contracts(self, iter_like_qprofile):
        with pytest.raises(ValueError, match="Troyon"):
            run_full_stability_check(iter_like_qprofile, beta_t=0.02, Ip_MA=10.0, a=1.0)

        with pytest.raises(ValueError, match="NTM"):
            run_full_stability_check(iter_like_qprofile, j_bs=np.ones_like(iter_like_qprofile.rho), a=1.0)
