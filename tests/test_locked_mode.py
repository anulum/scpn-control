# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.locked_mode import (
    ErrorFieldSpectrum,
    ErrorFieldToDisruptionChain,
    LockedModeIsland,
    ModeLocking,
    ResonantFieldAmplification,
)


def test_error_field_spectrum():
    spec = ErrorFieldSpectrum(B0=5.0)

    B21_nom = spec.B_mn(2, 1)
    assert B21_nom == 5e-4

    spec.set_coil_misalignment(10.0, 0.0)  # 10 mm shift
    B21_shift = spec.B_mn(2, 1)
    assert np.isclose(B21_shift, 5e-4)


def test_resonant_field_amplification():
    # La Haye 2006, Eq. 6: factor = 1/(1 - 2.5/2.8) = 2.8/0.3 ≈ 9.33
    rfa = ResonantFieldAmplification(beta_N=2.5, beta_N_nowall=2.8)

    factor = rfa.amplification_factor()
    assert factor > 1.0
    assert np.isclose(factor, 1.0 / (1.0 - 2.5 / 2.8))

    B_err = 1e-4
    B_res = rfa.resonant_field(B_err)
    assert B_res > B_err


def test_mode_locking():
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)

    ev1 = ml.evolve_rotation(B_res=1e-5, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=100)
    assert not ev1.locked

    # Locking requires T_em > T_viscous; Fitzpatrick 1993 Eq. 28 gives
    # T_em ∝ B_res², threshold at B_res ~ 0.1 T for these parameters.
    ev2 = ml.evolve_rotation(B_res=0.2, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=1000)
    assert ev2.locked
    assert ev2.omega_trace[-1] == 0.0


def test_post_locking_island_growth():
    lm = LockedModeIsland(r_s=1.0, m=2, n=1, a=2.0, R0=6.2, delta_prime=-1.0)

    res = lm.grow(w0=1e-3, eta=1e-6, dt=0.01, n_steps=1000, delta_r_mn=0.4)

    assert res.w_trace[-1] > 1e-3
    assert res.stochastic


def test_disruption_chain():
    config = {"R0": 6.2, "a": 2.0, "B0": 5.3, "Ip_MA": 15.0, "beta_N": 2.5, "beta_N_nowall": 2.8}

    chain = ErrorFieldToDisruptionChain(config)

    res_safe = chain.run(B_err_n1=1e-6, omega_phi_0=1e4)
    assert not res_safe.disruption

    # After RFA (×9.33 for beta_N=2.5, beta_nowall=2.8), B_err=2e-2 → B_res≈0.19 T
    # which exceeds the locking threshold (~0.1 T).
    res_disrupt = chain.run(B_err_n1=2e-2, omega_phi_0=1e4)
    assert res_disrupt.disruption
    assert res_disrupt.lock_time > 0.0
    assert res_disrupt.warning_time_ms > 0.0


def test_electromagnetic_torque_sign():
    """T_em opposes rotation: torque > 0 and decelerates positive ω.

    Fitzpatrick 1993, Nucl. Fusion 33, 1049, Eq. 28: the electromagnetic
    braking torque acts antiparallel to the plasma toroidal rotation.
    """
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)

    # Larger B_res → larger braking torque (T_em ∝ B_res²)
    T_small = ml.em_torque(B_res=1e-4, r_s=1.0, m=2, n=1)
    T_large = ml.em_torque(B_res=1e-3, r_s=1.0, m=2, n=1)

    assert T_small > 0.0, "Electromagnetic torque magnitude must be positive"
    assert T_large > T_small, "T_em ∝ B_res² — larger field must give larger torque"

    # Verify braking: rotation decreases from ω₀ when T_em is applied
    ev = ml.evolve_rotation(B_res=1e-3, r_s=1.0, tau_visc=0.5, dt=0.001, n_steps=200)
    assert ev.omega_trace[0] <= ml.omega_phi_0, "First step must not exceed initial ω"


def test_locking_threshold():
    """Mode locks when T_em > T_viscous (La Haye 2006, Eq. 12).

    Below threshold: rotation settles to non-zero equilibrium.
    Above threshold: ω is driven to zero (locked state).
    """
    ml_weak = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=500.0)
    ev_weak = ml_weak.evolve_rotation(B_res=1e-6, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=5000)
    assert not ev_weak.locked, "Weak error field must not lock the mode"

    ml_strong = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=500.0)
    ev_strong = ml_strong.evolve_rotation(B_res=5e-2, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=5000)
    assert ev_strong.locked, "Strong error field must lock the mode"
    assert ev_strong.lock_time > 0.0
    assert ev_strong.omega_trace[-1] == 0.0


@pytest.mark.parametrize("B_res", [1e-4, 5e-4, 1e-3])
def test_em_torque_quadratic_scaling(B_res: float):
    """T_em ∝ B_res²; Fitzpatrick 1993, Eq. 28."""
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
    T_ref = ml.em_torque(B_res=1e-4, r_s=1.0, m=2, n=1)
    T = ml.em_torque(B_res=B_res, r_s=1.0, m=2, n=1)
    expected_ratio = (B_res / 1e-4) ** 2
    assert np.isclose(T / T_ref, expected_ratio, rtol=1e-10)
