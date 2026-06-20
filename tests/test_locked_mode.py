# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Locked-Mode Chain Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.locked_mode import (
    ErrorFieldSpectrum,
    ErrorFieldToDisruptionChain,
    IslandGrowth,
    LockedModeIsland,
    ModeLocking,
    ResonantFieldAmplification,
    RotationEvolution,
)


def test_error_field_spectrum():
    spec = ErrorFieldSpectrum(B0=5.0)

    B21_nom = spec.B_mn(2, 1)
    assert B21_nom == 5e-4

    spec.set_coil_misalignment(10.0, 0.0)  # 10 mm shift
    B21_shift = spec.B_mn(2, 1)
    assert np.isclose(B21_shift, 5e-4)


def test_error_field_spectrum_rejects_nonphysical_inputs():
    with pytest.raises(ValueError, match="B0"):
        ErrorFieldSpectrum(B0=0.0)

    spec = ErrorFieldSpectrum(B0=5.0)

    with pytest.raises(ValueError, match="misalignment"):
        spec.set_coil_misalignment(-1.0, 0.0)

    with pytest.raises(ValueError, match="mode numbers"):
        spec.B_mn(0, 1)

    with pytest.raises(ValueError, match="I_correction"):
        spec.corrected_B_mn(2, 1, I_correction=-1.0)


def test_resonant_field_amplification():
    # La Haye 2006, Eq. 6: factor = 1/(1 - 2.5/2.8) = 2.8/0.3 ≈ 9.33
    rfa = ResonantFieldAmplification(beta_N=2.5, beta_N_nowall=2.8)

    factor = rfa.amplification_factor()
    assert factor > 1.0
    assert np.isclose(factor, 1.0 / (1.0 - 2.5 / 2.8))

    B_err = 1e-4
    B_res = rfa.resonant_field(B_err)
    assert B_res > B_err


def test_rfa_rejects_nonphysical_inputs():
    with pytest.raises(ValueError, match="beta_N"):
        ResonantFieldAmplification(beta_N=-0.1, beta_N_nowall=2.8)

    with pytest.raises(ValueError, match="beta_N_nowall"):
        ResonantFieldAmplification(beta_N=1.0, beta_N_nowall=0.0)

    rfa = ResonantFieldAmplification(beta_N=1.0, beta_N_nowall=2.8)
    with pytest.raises(ValueError, match="B_err"):
        rfa.resonant_field(-1.0e-4)


def test_mode_locking():
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)

    ev1 = ml.evolve_rotation(B_res=1e-5, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=100)
    assert not ev1.locked

    # Locking requires T_em > T_viscous; Fitzpatrick 1993 Eq. 28 gives
    # T_em ∝ B_res², threshold at B_res ~ 0.1 T for these parameters.
    ev2 = ml.evolve_rotation(B_res=0.2, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=1000)
    assert ev2.locked
    assert ev2.omega_trace[-1] == 0.0


def test_mode_locking_rejects_nonphysical_domains():
    with pytest.raises(ValueError, match="R0"):
        ModeLocking(R0=0.0, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)

    with pytest.raises(ValueError, match="omega_phi_0"):
        ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=-1.0)

    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)

    with pytest.raises(ValueError, match="B_res"):
        ml.em_torque(B_res=-1.0, r_s=1.0, m=2, n=1)

    with pytest.raises(ValueError, match="mode numbers"):
        ml.em_torque(B_res=1e-4, r_s=1.0, m=0, n=1)

    with pytest.raises(ValueError, match="tau_visc"):
        ml.evolve_rotation(B_res=1e-4, r_s=1.0, tau_visc=0.0, dt=0.001, n_steps=10)


def test_post_locking_island_growth():
    lm = LockedModeIsland(r_s=1.0, m=2, n=1, a=2.0, R0=6.2, delta_prime=-1.0)

    res = lm.grow(w0=1e-3, eta=1e-6, dt=0.01, n_steps=1000, delta_r_mn=0.4)

    assert res.w_trace[-1] > 1e-3
    assert res.stochastic


def test_locked_mode_island_rejects_nonphysical_domains():
    with pytest.raises(ValueError, match="r_s"):
        LockedModeIsland(r_s=0.0, m=2, n=1, a=2.0, R0=6.2, delta_prime=-1.0)

    with pytest.raises(ValueError, match="mode numbers"):
        LockedModeIsland(r_s=1.0, m=0, n=1, a=2.0, R0=6.2, delta_prime=-1.0)

    lm = LockedModeIsland(r_s=1.0, m=2, n=1, a=2.0, R0=6.2, delta_prime=-1.0)

    with pytest.raises(ValueError, match="eta"):
        lm.grow(w0=1e-3, eta=-1e-6, dt=0.01, n_steps=100)

    with pytest.raises(ValueError, match="delta_r_mn"):
        lm.grow(w0=1e-3, eta=1e-6, dt=0.01, n_steps=100, delta_r_mn=0.0)


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


def test_rfa_at_nowall_limit():
    """Amplification factor tends to infinity at the no-wall beta limit."""
    rfa_at = ResonantFieldAmplification(beta_N=2.8, beta_N_nowall=2.8)
    assert rfa_at.amplification_factor() == float("inf")
    rfa_above = ResonantFieldAmplification(beta_N=3.0, beta_N_nowall=2.8)
    assert rfa_above.amplification_factor() == float("inf")


def test_error_field_correction():
    """Correction coils reduce the raw resonant field in the linear actuator model."""
    spec = ErrorFieldSpectrum(B0=5.0)
    B_raw = spec.B_mn(2, 1)
    B_corr = spec.corrected_B_mn(2, 1, I_correction=10.0)
    assert B_corr < B_raw
    B_overcorr = spec.corrected_B_mn(2, 1, I_correction=1e6)
    assert B_overcorr == 0.0


def test_mode_locking_with_ne19():
    """Finite density sets the effective toroidal inertia used by the rotation model."""
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4, ne_19=5.0)
    ev = ml.evolve_rotation(B_res=0.2, r_s=1.0, tau_visc=0.1, dt=0.001, n_steps=1000)
    assert ev.locked or ev.omega_trace[-1] < 1e4


def test_viscous_torque():
    """Viscous restoring torque follows the thin-shell island-volume model."""
    ml = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=1e4)
    T_v = ml.viscous_torque(omega=1000.0, r_s=1.0, rho_chi=1.0)
    assert T_v > 0.0
    assert ml.viscous_torque(omega=0.0, r_s=1.0) == 0.0


def test_chain_no_locking():
    """The disruption chain remains safe when locking does not occur."""
    config = {"R0": 6.2, "a": 2.0, "B0": 5.3, "Ip_MA": 15.0, "beta_N": 1.0, "beta_N_nowall": 2.8}
    chain = ErrorFieldToDisruptionChain(config)
    res = chain.run(B_err_n1=1e-7, omega_phi_0=1e4)
    assert not res.disruption
    assert res.lock_time == -1.0


def test_error_field_spectrum_rejects_negative_corrections() -> None:
    with pytest.raises(ValueError, match="n_corrections must be non-negative"):
        ErrorFieldSpectrum(B0=5.3, n_corrections=-1)


def test_mode_locking_rejects_aspect_ratio_violation() -> None:
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        ModeLocking(R0=1.0, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=100.0)


def test_evolve_rotation_rejects_nonpositive_steps() -> None:
    locker = ModeLocking(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, omega_phi_0=100.0)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        locker.evolve_rotation(B_res=0.0, r_s=1.0, tau_visc=0.1, dt=1.0e-3, n_steps=0)


def test_locked_mode_island_rejects_aspect_ratio_violation() -> None:
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        LockedModeIsland(r_s=1.0, m=2, n=1, a=2.0, R0=1.0, delta_prime=-1.0)


def test_locked_mode_island_rejects_nonfinite_delta_prime() -> None:
    with pytest.raises(ValueError, match="delta_prime must be finite"):
        LockedModeIsland(r_s=1.0, m=2, n=1, a=2.0, R0=6.2, delta_prime=np.inf)


def test_island_grow_rejects_nonpositive_steps() -> None:
    island = LockedModeIsland(r_s=1.0, m=2, n=1, a=2.0, R0=6.2, delta_prime=-1.0)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        island.grow(w0=1.0e-3, eta=1.0e-7, dt=1.0e-3, n_steps=0)


def test_chain_returns_nondisruptive_when_locking_consumes_all_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    # lock_time 2.5 s with the chain's internal dt=1e-3 and n_steps=2000 leaves no
    # steps for island growth (steps_left <= 0), so the chain short-circuits.
    monkeypatch.setattr(
        ModeLocking,
        "evolve_rotation",
        lambda self, *args, **kwargs: RotationEvolution(np.zeros(1), True, 2.5),
    )
    result = ErrorFieldToDisruptionChain({}).run(B_err_n1=1.0e-3, omega_phi_0=100.0)
    assert result.disruption is False
    assert result.lock_time == 2.5
    assert result.tq_trigger_time == -1.0


def test_chain_returns_nondisruptive_when_island_does_not_reach_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ModeLocking,
        "evolve_rotation",
        lambda self, *args, **kwargs: RotationEvolution(np.zeros(1), True, 0.0),
    )
    monkeypatch.setattr(
        LockedModeIsland,
        "grow",
        lambda self, *args, **kwargs: IslandGrowth(np.array([1.0e-3]), 0.0, False),
    )
    result = ErrorFieldToDisruptionChain({}).run(B_err_n1=1.0e-3, omega_phi_0=100.0)
    assert result.disruption is False
    assert result.island_width_at_tq == pytest.approx(1.0e-3)
