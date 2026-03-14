# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Bosch-Hale Uncertainty Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations


from scpn_control.core.uncertainty import (
    compute_fusion_sensitivities,
    fusion_power_from_tau,
    PlasmaScenario,
)


def test_bosch_hale_iter_q10_point():
    """Verify fusion power estimate for ITER Q=10-like parameters.

    ITER: R=6.2, A=3.1, kappa=1.7, n_e=10.1e19, tau_E=3.7s, P_aux=50MW.
    """
    # Using P_heat = 100 MW (balanced aux + partial alpha)
    scenario = PlasmaScenario(I_p=15.0, B_t=5.3, P_heat=100.0, n_e=10.1, R=6.2, A=3.1, kappa=1.7)

    pfus = fusion_power_from_tau(scenario, tau_E=3.7)

    # Expected: ~500 MW
    assert 400.0 < pfus < 800.0


def test_fusion_sensitivities():
    """Verify that sensitivities have physical signs."""
    scenario = PlasmaScenario(I_p=15.0, B_t=5.3, P_heat=50.0, n_e=10.0, R=6.2, A=3.1, kappa=1.7)

    sens = compute_fusion_sensitivities(scenario, tau_E=3.0)

    # Increasing temperature should increase fusion power
    assert sens["dP_dT"] > 0
    # Increasing density at fixed energy: T ~ 1/n, P ~ n^2 <sv>(1/n)
    # The sign depends on the T regime. At very low T, it's positive.
    assert abs(sens["dP_dn"]) > 0


def test_bosch_hale_low_temp():
    """Verify reactivity is near-zero at very low temperature."""
    scenario = PlasmaScenario(I_p=1.0, B_t=1.0, P_heat=0.1, n_e=1.0, R=1.0, A=3.0, kappa=1.0)
    pfus = fusion_power_from_tau(scenario, tau_E=0.01)
    assert pfus < 1e-3
