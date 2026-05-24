# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Scaling Laws Edge
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Scaling Laws Edge Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for ipb98y2_tau_e invalid tau (line 240) and
ipb98y2_with_uncertainty non-finite variance (lines 323-324)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from scpn_control.core.scaling_laws import ipb98y2_tau_e, ipb98y2_with_uncertainty


class TestIPB98y2Invalid:
    def test_zero_power_raises(self):
        """Zero Ploss raises at validation (line 213)."""
        with pytest.raises(ValueError, match="Ploss"):
            ipb98y2_tau_e(
                Ip=15.0,
                BT=5.3,
                ne19=10.0,
                Ploss=0.0,
                R=6.2,
                kappa=1.7,
                epsilon=0.32,
            )

    def test_nan_power_raises(self):
        """NaN Ploss raises at validation."""
        with pytest.raises(ValueError, match="Ploss"):
            ipb98y2_tau_e(
                Ip=15.0,
                BT=5.3,
                ne19=10.0,
                Ploss=float("nan"),
                R=6.2,
                kappa=1.7,
                epsilon=0.32,
            )


class TestIPB98y2Uncertainty:
    def test_normal_uncertainty(self):
        """Normal case returns finite tau and sigma."""
        tau, sigma = ipb98y2_with_uncertainty(
            Ip=15.0,
            BT=5.3,
            ne19=10.0,
            Ploss=50.0,
            R=6.2,
            kappa=1.7,
            epsilon=0.32,
        )
        assert np.isfinite(tau) and tau > 0
        assert np.isfinite(sigma) and sigma >= 0

    def test_rejects_nonfinite_exponent_uncertainty_metadata(self):
        """Non-finite exponent uncertainty metadata must fail closed."""
        from scpn_control.core import scaling_laws as scaling_mod

        coeff = scaling_mod.load_ipb98y2_coefficients()
        coeff["uncertainties_1sigma"]["Ip_MA"] = float("nan")

        with (
            patch.object(scaling_mod, "_validate_ipb98y2_coefficients", return_value=coeff),
            pytest.raises(ValueError, match="uncertainty is invalid"),
        ):
            scaling_mod.ipb98y2_with_uncertainty(
                Ip=15.0,
                BT=5.3,
                ne19=10.1,
                Ploss=87.0,
                R=6.2,
                kappa=1.7,
                epsilon=0.32,
                M=2.5,
                coefficients=coeff,
            )

    def test_rejects_extreme_uncertainty_metadata_overflow(self):
        """Extreme finite exponent uncertainty metadata must fail closed."""
        huge_uncertainties = {
            key: 1.0e200
            for key in [
                "Ip_MA",
                "BT_T",
                "ne19_1e19m3",
                "Ploss_MW",
                "R_m",
                "kappa",
                "epsilon",
                "M_AMU",
            ]
        }
        coefficients = {
            "C": 0.0562,
            "exponents": {
                "Ip_MA": 0.93,
                "BT_T": 0.15,
                "ne19_1e19m3": 0.41,
                "Ploss_MW": -0.69,
                "R_m": 1.97,
                "kappa": 0.78,
                "epsilon": 0.58,
                "M_AMU": 0.19,
            },
            "uncertainties_1sigma": huge_uncertainties,
        }

        with pytest.raises(ValueError, match="uncertainty is invalid"):
            ipb98y2_with_uncertainty(
                Ip=15.0,
                BT=5.3,
                ne19=10.1,
                Ploss=87.0,
                R=6.2,
                kappa=1.7,
                epsilon=0.32,
                M=2.5,
                coefficients=coefficients,
            )
