"""Tests for scpn_control.core.scaling_laws (IPB98(y,2))."""

import numpy as np
import pytest

from scpn_control.core.scaling_laws import (
    _require_finite_number,
    _require_positive_finite,
    _validate_ipb98y2_coefficients,
    ipb98y2_tau_e,
    ipb98y2_with_uncertainty,
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

    def test_rejects_inf(self, coefficients):
        with pytest.raises(ValueError):
            ipb98y2_tau_e(Ip=float("inf"), BT=5.0, ne19=5.0, Ploss=50.0,
                          R=6.0, kappa=1.5, epsilon=0.3, coefficients=coefficients)


# ── _require_positive_finite / _require_finite_number ─────────────────

class TestRequirePositiveFinite:
    def test_accepts_positive(self):
        assert _require_positive_finite("x", 1.5) == 1.5

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="x"):
            _require_positive_finite("x", 0.0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="x"):
            _require_positive_finite("x", -1.0)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="x"):
            _require_positive_finite("x", float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="x"):
            _require_positive_finite("x", float("inf"))


class TestRequireFiniteNumber:
    def test_accepts_zero(self):
        assert _require_finite_number("x", 0.0) == 0.0

    def test_accepts_negative(self):
        assert _require_finite_number("x", -3.14) == pytest.approx(-3.14)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            _require_finite_number("x", float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            _require_finite_number("x", float("inf"))

    def test_rejects_string(self):
        with pytest.raises(ValueError, match="numeric"):
            _require_finite_number("x", "abc")

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="numeric"):
            _require_finite_number("x", None)


# ── _validate_ipb98y2_coefficients ────────────────────────────────────

class TestValidateIPB98y2Coefficients:
    def _minimal_valid(self) -> dict:
        return {
            "C": 0.0562,
            "exponents": {
                "Ip_MA": 0.93, "BT_T": 0.15, "ne19_1e19m3": 0.41,
                "Ploss_MW": -0.69, "R_m": 1.97, "kappa": 0.78,
                "epsilon": 0.58, "M_AMU": 0.19,
            },
        }

    def test_valid_minimal(self):
        result = _validate_ipb98y2_coefficients(self._minimal_valid())
        assert result["C"] == pytest.approx(0.0562)

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError, match="JSON object"):
            _validate_ipb98y2_coefficients([1, 2, 3])

    def test_rejects_missing_C(self):
        with pytest.raises(ValueError, match="missing required key 'C'"):
            _validate_ipb98y2_coefficients({"exponents": {}})

    def test_rejects_negative_C(self):
        raw = self._minimal_valid()
        raw["C"] = -0.01
        with pytest.raises(ValueError, match="C must be > 0"):
            _validate_ipb98y2_coefficients(raw)

    def test_rejects_missing_exponents(self):
        with pytest.raises(ValueError, match="exponents"):
            _validate_ipb98y2_coefficients({"C": 0.05})

    def test_rejects_missing_exponent_key(self):
        raw = self._minimal_valid()
        del raw["exponents"]["Ip_MA"]
        with pytest.raises(ValueError, match="Ip_MA"):
            _validate_ipb98y2_coefficients(raw)

    def test_validates_sigma_lnC(self):
        raw = self._minimal_valid()
        raw["sigma_lnC"] = 0.14
        result = _validate_ipb98y2_coefficients(raw)
        assert result["sigma_lnC"] == pytest.approx(0.14)

    def test_rejects_negative_sigma_lnC(self):
        raw = self._minimal_valid()
        raw["sigma_lnC"] = -0.01
        with pytest.raises(ValueError, match="sigma_lnC"):
            _validate_ipb98y2_coefficients(raw)

    def test_validates_exponent_uncertainties(self):
        raw = self._minimal_valid()
        raw["exponent_uncertainties"] = {"Ip_MA": 0.02, "BT_T": 0.04}
        result = _validate_ipb98y2_coefficients(raw)
        assert result["exponent_uncertainties"]["Ip_MA"] == pytest.approx(0.02)

    def test_rejects_negative_exponent_uncertainty(self):
        raw = self._minimal_valid()
        raw["exponent_uncertainties"] = {"Ip_MA": -0.01}
        with pytest.raises(ValueError, match="exponent_uncertainties.Ip_MA"):
            _validate_ipb98y2_coefficients(raw)

    def test_rejects_non_dict_exponent_uncertainties(self):
        raw = self._minimal_valid()
        raw["exponent_uncertainties"] = "bad"
        with pytest.raises(ValueError, match="exponent_uncertainties must be an object"):
            _validate_ipb98y2_coefficients(raw)


# ── ipb98y2_with_uncertainty ──────────────────────────────────────────

class TestIPB98y2WithUncertainty:
    def test_returns_tau_and_sigma(self):
        tau, sigma = ipb98y2_with_uncertainty(
            Ip=15.0, BT=5.3, ne19=10.1, Ploss=87.0,
            R=6.2, kappa=1.7, epsilon=0.32, M=2.5,
        )
        assert tau > 0.0
        assert sigma > 0.0

    def test_sigma_positive_finite(self):
        tau, sigma = ipb98y2_with_uncertainty(
            Ip=10.0, BT=3.0, ne19=5.0, Ploss=30.0,
            R=3.0, kappa=1.5, epsilon=0.3,
        )
        assert np.isfinite(sigma)
        assert sigma > 0.0

    def test_tau_matches_point_estimate(self):
        coeff = load_ipb98y2_coefficients()
        tau_point = ipb98y2_tau_e(
            Ip=15.0, BT=5.3, ne19=10.1, Ploss=87.0,
            R=6.2, kappa=1.7, epsilon=0.32, coefficients=coeff,
        )
        tau_unc, _ = ipb98y2_with_uncertainty(
            Ip=15.0, BT=5.3, ne19=10.1, Ploss=87.0,
            R=6.2, kappa=1.7, epsilon=0.32, coefficients=coeff,
        )
        assert tau_unc == pytest.approx(tau_point)
