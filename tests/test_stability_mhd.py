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
)


@pytest.fixture
def iter_like_qprofile():
    """ITER-like q-profile from cylindrical approximation."""
    n = 101
    rho = np.linspace(0, 1, n)
    ne = 10.0 * (1 - 0.8 * rho**2)
    Ti = 15.0 * (1 - rho**2) ** 1.5
    Te = 15.0 * (1 - rho**2) ** 1.5
    return compute_q_profile(rho, ne, Ti, Te, R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)


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
            q_min=0.5, q_min_rho=0.0, q_edge=0.8,
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
