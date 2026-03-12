# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Scaling Laws Audit Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations


from scpn_control.core.scaling_laws import (
    compute_betan,
    greenwald_limit,
    ipb98y2_tau_e,
)


def test_ipb98y2_iter_point():
    """Reproduce ITER confinement prediction.

    Ref: ITER Physics Basis, NF 39 (1999) 2175, Table 5.
    Scenario: Ip=15MA, BT=5.3T, n=10.1e19, P=150MW, R=6.2m, a=2.0m, k=1.7, M=2.5.
    """
    Ip = 15.0
    BT = 5.3
    ne19 = 10.1
    Ploss = 150.0 # MW (includes alpha heating)
    R = 6.2
    kappa = 1.7
    epsilon = 2.0 / 6.2 # a/R
    M = 2.5

    tau = ipb98y2_tau_e(Ip, BT, ne19, Ploss, R, kappa, epsilon, M)

    # For P=150MW, the result is ~2.48s.
    # The often-cited 3.7s is for P ~ 80MW or with different H-factor.
    assert 2.4 < tau < 2.6


def test_greenwald_iter_point():
    """Verify Greenwald limit for ITER.

    Ref: Greenwald (2002).
    Scenario: Ip=15MA, a=2.0m.
    n_GW = 15 / (pi * 2^2) = 15 / 12.56 ~ 1.19 e20 m^-3.
    """
    Ip = 15.0
    a = 2.0
    ngw = greenwald_limit(Ip, a)

    assert 1.18 < ngw < 1.20


def test_troyon_beta_iter():
    """Verify Troyon beta normalization for ITER baseline.

    Ref: Troyon (1984).
    Scenario: beta_t = 2.5%, a=2.0m, B=5.3T, Ip=15MA.
    beta_N = beta_t / (Ip / aB) = 2.5 / (15 / (2*5.3)) = 2.5 / (15/10.6) = 2.5 / 1.415 ~ 1.76.
    """
    beta_t = 2.5
    a = 2.0
    B = 5.3
    Ip = 15.0

    bn = compute_betan(beta_t, a, B, Ip)

    assert 1.7 < bn < 1.8
