# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Uncertainty sigma validation edge tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Coverage for full_chain_uq _validate_sigma non-parseable (lines 248-249)."""

from __future__ import annotations

import pytest

from scpn_control.core.uncertainty import PlasmaScenario, quantify_full_chain

_SCENARIO = PlasmaScenario(
    I_p=15.0,
    B_t=5.3,
    P_heat=50.0,
    n_e=10.0,
    R=6.2,
    A=3.1,
    kappa=1.7,
)


class TestValidateSigma:
    def test_non_parseable_sigma_raises(self):
        """Non-parseable chi_gB_sigma triggers _validate_sigma TypeError (248-249)."""
        with pytest.raises(ValueError, match="must be finite"):
            quantify_full_chain(_SCENARIO, chi_gB_sigma="not_a_number")  # type: ignore[arg-type]

    def test_nan_sigma_raises(self):
        """NaN chi_gB_sigma triggers _validate_sigma (250-251)."""
        with pytest.raises(ValueError, match="must be finite"):
            quantify_full_chain(_SCENARIO, chi_gB_sigma=float("nan"))

    def test_negative_sigma_raises(self):
        """Negative sigma triggers _validate_sigma."""
        with pytest.raises(ValueError, match="must be finite"):
            quantify_full_chain(_SCENARIO, chi_gB_sigma=-0.1)
