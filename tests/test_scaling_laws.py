"""Tests for scpn_control.core.scaling_laws (IPB98(y,2))."""

import numpy as np
import pytest

from scpn_control.core.scaling_laws import (
    ipb98y2_tau_e,
    load_ipb98y2_coefficients,
)


@pytest.fixture(scope="module")
def coefficients():
    return load_ipb98y2_coefficients()


class TestLoadCoefficients:
    def test_loads_successfully(self, coefficients):
        assert "C" in coefficients
        assert "exponents" in coefficients

    def test_c_positive(self, coefficients):
        assert coefficients["C"] > 0

    def test_all_exponent_keys_present(self, coefficients):
        required = {"Ip_MA", "BT_T", "ne19_1e19m3", "Ploss_MW", "R_m", "kappa", "epsilon", "M_AMU"}
        assert required.issubset(coefficients["exponents"].keys())


class TestIPB98y2:
    def test_iter_baseline(self, coefficients):
        # ITER: Ip=15 MA, BT=5.3 T, ne19=10.1, Ploss=87 MW, R=6.2 m, kappa=1.7, eps=0.32
        tau = ipb98y2_tau_e(
            Ip=15.0, BT=5.3, ne19=10.1, Ploss=87.0,
            R=6.2, kappa=1.7, epsilon=0.32, M=2.5,
            coefficients=coefficients,
        )
        assert 1.0 < tau < 10.0  # ITER expects ~3.7 s

    def test_positive_output(self, coefficients):
        tau = ipb98y2_tau_e(
            Ip=1.0, BT=2.0, ne19=3.0, Ploss=5.0,
            R=1.6, kappa=1.5, epsilon=0.3,
            coefficients=coefficients,
        )
        assert tau > 0.0

    def test_monotonic_in_current(self, coefficients):
        base = dict(BT=5.3, ne19=10.0, Ploss=50.0, R=6.2, kappa=1.7, epsilon=0.32)
        tau_low = ipb98y2_tau_e(Ip=10.0, coefficients=coefficients, **base)
        tau_high = ipb98y2_tau_e(Ip=15.0, coefficients=coefficients, **base)
        assert tau_high > tau_low

    def test_monotonic_in_power_loss(self, coefficients):
        base = dict(Ip=15.0, BT=5.3, ne19=10.0, R=6.2, kappa=1.7, epsilon=0.32)
        tau_low_p = ipb98y2_tau_e(Ploss=50.0, coefficients=coefficients, **base)
        tau_high_p = ipb98y2_tau_e(Ploss=100.0, coefficients=coefficients, **base)
        assert tau_low_p > tau_high_p  # Higher loss power => lower confinement

    def test_rejects_zero_current(self, coefficients):
        with pytest.raises(ValueError):
            ipb98y2_tau_e(Ip=0.0, BT=5.0, ne19=5.0, Ploss=50.0,
                          R=6.0, kappa=1.5, epsilon=0.3, coefficients=coefficients)

    def test_rejects_negative_field(self, coefficients):
        with pytest.raises(ValueError):
            ipb98y2_tau_e(Ip=15.0, BT=-5.0, ne19=5.0, Ploss=50.0,
                          R=6.0, kappa=1.5, epsilon=0.3, coefficients=coefficients)

    def test_rejects_nan(self, coefficients):
        with pytest.raises(ValueError):
            ipb98y2_tau_e(Ip=float("nan"), BT=5.0, ne19=5.0, Ploss=50.0,
                          R=6.0, kappa=1.5, epsilon=0.3, coefficients=coefficients)
