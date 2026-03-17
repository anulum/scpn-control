# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851 — Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np

from scpn_control.control.rwm_feedback import (
    RWMFeedbackController,
    RWMPhysics,
    RWMStabilityAnalysis,
)


# ── existing tests (unchanged) ───────────────────────────────────────────────


def test_rwm_stable_below_nowall():
    rwm = RWMPhysics(beta_n=2.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)
    assert not rwm.is_unstable()
    assert rwm.growth_rate() == 0.0


def test_rwm_unstable_between_limits():
    rwm = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)
    assert rwm.is_unstable()
    gamma = rwm.growth_rate()
    assert gamma > 0.0
    # γ = 1/0.01 × (3.0−2.8)/(3.5−3.0) = 100 × 0.2/0.5 = 40 s⁻¹
    assert np.isclose(gamma, 40.0)


def test_rwm_ideal_kink():
    rwm = RWMPhysics(beta_n=3.6, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)
    assert rwm.growth_rate() >= 1e6


def test_rwm_feedback_stabilization():
    rwm = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)

    ctrl_weak = RWMFeedbackController(n_sensors=1, n_coils=1, G_p=0.5, G_d=0.0, M_coil=1.0)
    assert ctrl_weak.effective_growth_rate(rwm) > 0.0
    assert not ctrl_weak.is_stabilized(rwm)

    ctrl_strong = RWMFeedbackController(n_sensors=1, n_coils=1, G_p=2.0, G_d=0.0, M_coil=1.0)
    assert ctrl_strong.effective_growth_rate(rwm) < 0.0
    assert ctrl_strong.is_stabilized(rwm)


def test_rwm_step():
    ctrl = RWMFeedbackController(n_sensors=2, n_coils=2, G_p=2.0, G_d=0.1)
    B_r = np.array([1.0, -1.0])
    I_coil = ctrl.step(B_r, dt=0.01)
    # dB_dt = B_r / 0.01 = [100, −100]; I = 2·B_r + 0.1·dB_dt = [12, −12]
    assert np.allclose(I_coil, [12.0, -12.0])


def test_wall_time_limits():
    rwm_ideal_wall = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=float("inf"))
    assert rwm_ideal_wall.growth_rate() == 0.0

    rwm_no_wall = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.0)
    assert rwm_no_wall.growth_rate() >= 1e6


def test_required_gain():
    gain = RWMStabilityAnalysis.required_feedback_gain(
        beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01, tau_controller=1e-4
    )
    # γ = 40; 1 + 40×1e-4 = 1.004
    assert np.isclose(gain, 1.004)


# ── new tests ────────────────────────────────────────────────────────────────


def test_rotation_stabilization():
    """
    High Ω_φ drives γ_total < 0 without active coil feedback.

    Fitzpatrick 2001, Phys. Plasmas 8, 4489, Eq. (14):
      γ_rot = −Ω_φ² τ_eff / (1 + (Ω_φ τ_eff)²)
    When Ω_φ τ_eff ≫ 1, γ_rot ≈ −1/τ_eff, which overwhelms γ_wall.
    """
    tau_wall = 0.01  # s
    # γ_wall = 40 s⁻¹ (same as test_rwm_unstable_between_limits)
    # Need γ_rot < −40, i.e. Ω_φ τ_wall ≫ 1.  Use Ω_φ = 10 000 rad/s → x = 100.
    omega_high = 10_000.0  # rad s⁻¹; Ω_φ τ_wall = 100 ≫ 1
    rwm = RWMPhysics(
        beta_n=3.0,
        beta_n_nowall=2.8,
        beta_n_wall=3.5,
        tau_wall=tau_wall,
        omega_phi=omega_high,
    )
    assert rwm.growth_rate() < 0.0, "rotation should suppress RWM growth rate below zero"


def test_critical_rotation():
    """
    At Ω_φ = Ω_crit the total growth rate is marginally stable (γ_total ≈ 0).

    Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709, Eq. (7):
      Ω_crit = (1/τ_eff) √[(β_wall − β_N)/(β_N − β_nowall)]
    """
    rwm_ref = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)
    omega_c = rwm_ref.critical_rotation()
    assert math.isfinite(omega_c) and omega_c > 0.0

    rwm_at_crit = RWMPhysics(
        beta_n=3.0,
        beta_n_nowall=2.8,
        beta_n_wall=3.5,
        tau_wall=0.01,
        omega_phi=omega_c,
    )
    # γ_wall + γ_rot = 0 at Ω_crit by construction
    assert abs(rwm_at_crit.growth_rate()) < 1e-9, f"expected γ ≈ 0 at Ω_crit, got {rwm_at_crit.growth_rate()}"


def test_wall_geometry():
    """
    τ_eff > τ_wall when the conducting wall is further from the plasma (b > d).

    Strait et al. 2003, Nucl. Fusion 43, 430, Eq. (3):
      τ_eff = τ_wall × (b/d)²
    """
    tau_wall = 0.01  # s
    b = 0.6  # wall minor radius, m
    d = 0.5  # plasma-edge minor radius, m  →  b/d = 1.2 > 1

    rwm = RWMPhysics(
        beta_n=3.0,
        beta_n_nowall=2.8,
        beta_n_wall=3.5,
        tau_wall=tau_wall,
        wall_radius=b,
        plasma_radius=d,
    )
    tau_e = rwm.tau_eff()
    expected = tau_wall * (b / d) ** 2
    assert np.isclose(tau_e, expected)
    assert tau_e > tau_wall, "τ_eff must exceed τ_wall when wall radius > plasma radius"


def test_rotation_plus_feedback():
    """
    Combined rotation + active feedback is strictly more stabilizing than either alone.

    Setup: Ω_φ = 50 rad s⁻¹, τ_wall = 0.01 s → x = 0.5
      γ_wall = 40 s⁻¹, γ_rot ≈ −20 s⁻¹ → γ_total ≈ 20 s⁻¹  (rotation alone: still unstable)
      G_p = 1.005 with γ_total = 40 → γ_eff ≈ −0.040 s⁻¹    (feedback alone: barely stable)
      Combined: γ_total = 20 → γ_eff ≈ −0.060 s⁻¹            (more negative than either)

    Garofalo et al. 2002, Phys. Plasmas 9, 1997: rotation and active feedback
    are additive stabilization channels.
    """
    tau_wall = 0.01
    omega_moderate = 50.0  # rad s⁻¹; Ω_φ τ_wall = 0.5 (partial: γ_rot halves γ_wall)

    rwm_no_rot = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=tau_wall)
    rwm_rot = RWMPhysics(
        beta_n=3.0,
        beta_n_nowall=2.8,
        beta_n_wall=3.5,
        tau_wall=tau_wall,
        omega_phi=omega_moderate,
    )

    # G_p just above the no-rotation required gain (1.004) — marginally stabilizes
    # without rotation, and more deeply stabilizes with rotation.
    ctrl = RWMFeedbackController(n_sensors=1, n_coils=1, G_p=1.005, G_d=0.0, M_coil=1.0)

    gamma_fb_only = ctrl.effective_growth_rate(rwm_no_rot)  # ≈ −0.040 s⁻¹
    gamma_rot_only = rwm_rot.growth_rate()  # ≈ +20 s⁻¹
    gamma_combined = ctrl.effective_growth_rate(rwm_rot)  # ≈ −0.060 s⁻¹

    assert gamma_combined < gamma_fb_only, "combined must beat feedback alone"
    assert gamma_combined < gamma_rot_only, "combined must beat rotation alone"
