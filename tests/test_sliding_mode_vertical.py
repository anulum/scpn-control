# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np

from scpn_control.control.sliding_mode_vertical import (
    SuperTwistingSMC,
    VerticalStabilizer,
    estimate_convergence_time,
    lyapunov_certificate,
)


def test_gain_verification():
    L_max = 10.0
    alpha = math.sqrt(2.0 * L_max) + 1.0
    beta = L_max + 1.0

    assert lyapunov_certificate(alpha, beta, L_max)

    assert not lyapunov_certificate(1.0, beta, L_max)  # alpha too small
    assert not lyapunov_certificate(alpha, 1.0, L_max)  # beta too small


def test_convergence_time_estimate():
    s0 = 4.0
    L_max = 2.0
    alpha = 5.0
    beta = 5.0

    t_conv = estimate_convergence_time(alpha, beta, L_max, s0)
    assert t_conv > 0.0
    assert t_conv < 10.0


def test_no_disturbance_convergence():
    smc = SuperTwistingSMC(alpha=50.0, beta=100.0, c=5.0, u_max=500.0)

    # Simulate simple double integrator: x_ddot = u
    dt = 0.001
    x = 1.0
    x_dot = 0.0

    for _ in range(5000):
        u = smc.step(x, x_dot, dt)
        x_dot += u * dt
        x += x_dot * dt

    # Should converge to origin
    assert abs(x) < 0.5
    assert abs(x_dot) < 0.5


def test_constant_disturbance_rejection():
    smc = SuperTwistingSMC(alpha=50.0, beta=100.0, c=5.0, u_max=500.0)

    dt = 0.001
    x = 0.0
    x_dot = 0.0
    dist = 50.0  # Constant disturbance force

    for _ in range(5000):
        u = smc.step(x, x_dot, dt)
        x_dot += (u + dist) * dt
        x += x_dot * dt

    # Integral action (v) should reject the disturbance
    assert abs(x) < 0.5
    assert abs(x_dot) < 0.5
    # The integral term should equal -dist to cancel it
    assert np.isclose(smc.v, -dist, atol=2.0)


def test_actuator_saturation():
    smc = SuperTwistingSMC(alpha=100.0, beta=100.0, c=1.0, u_max=10.0)
    u = smc.step(10.0, 10.0, 0.01)

    # Very large error should saturate the output
    assert abs(u) <= 10.0


def test_vertical_stabilizer_wrapper():
    smc = SuperTwistingSMC(alpha=10.0, beta=20.0, c=1.0, u_max=100.0)
    vs = VerticalStabilizer(n_index=-1.0, Ip_MA=15.0, R0=6.2, m_eff=1.0, tau_wall=0.01, smc=smc)

    u = vs.step(Z_meas=0.1, Z_ref=0.0, dZ_dt_meas=0.0, dt=0.01)
    assert u != 0.0


def test_smc_chattering_bounded():
    """Boundary-layer output magnitude < k (Slotine & Li 1991, Ch. 7, §7.2).

    |u| ≤ α |s|^{1/2} |sat(s)| + |v|.  With |sat(s)| ≤ 1, the
    continuous term is bounded by k = u_max.
    """
    k = 10.0
    smc = SuperTwistingSMC(alpha=5.0, beta=2.0, c=1.0, u_max=k)

    outputs = []
    s_val = 0.5
    for _ in range(50):
        u = smc.step(s_val, 0.0, 0.01)
        outputs.append(abs(u))
        s_val *= 0.9

    assert all(o <= k + 1e-9 for o in outputs), "output exceeds u_max (chattering not bounded)"


def test_smc_reaching_condition():
    """Sliding condition: s · ds/dt < 0 outside the boundary layer.

    Utkin 1992, Ch. 2: reaching condition s ṡ < 0 guarantees
    the state trajectory moves toward s = 0.
    """
    smc = SuperTwistingSMC(alpha=20.0, beta=10.0, c=1.0, u_max=200.0)
    dt = 0.001

    # Initial condition far from sliding surface
    e, de_dt = 2.0, 0.0

    s_prev = smc.sliding_surface(e, de_dt)
    u = smc.step(e, de_dt, dt)

    # Approximate ṡ = de/dt + c * u (double integrator, dë = u)
    de_dt_next = de_dt + u * dt
    e_next = e + de_dt * dt
    s_next = smc.sliding_surface(e_next, de_dt_next)

    ds_dt = (s_next - s_prev) / dt
    assert s_prev * ds_dt < 0.0, f"reaching condition violated: s={s_prev:.4f}, ds/dt={ds_dt:.4f}"
