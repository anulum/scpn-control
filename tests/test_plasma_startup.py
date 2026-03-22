# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Plasma Startup Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.core.plasma_startup import (
    BurnThrough,
    PaschenBreakdown,
    StartupController,
    StartupPhase,
    StartupSequence,
    TownsendAvalanche,
    _E_IZ_EV,
)


def test_paschen_breakdown():
    pb = PaschenBreakdown("D2")

    # V_min ≈ 305 V for D₂ at SI Townsend coefficients (Lieberman 2005)
    p_opt = pb.optimal_prefill_pressure(V_loop_max=400.0)
    assert p_opt > 0.0

    # Curve is U-shaped: voltages away from minimum are higher
    V_opt = pb.breakdown_voltage(p_opt, 100.0)
    V_high = pb.breakdown_voltage(p_opt * 10.0, 100.0)
    V_low = pb.breakdown_voltage(p_opt * 0.1, 100.0)

    assert V_high > V_opt
    assert V_low > V_opt

    # 400 V exceeds V_min; well below minimum does not break down
    assert pb.is_breakdown(V_loop=400.0, p_Pa=p_opt)
    assert not pb.is_breakdown(V_loop=0.01, p_Pa=p_opt)


def test_townsend_avalanche():
    pb = PaschenBreakdown()
    p_opt = pb.optimal_prefill_pressure(400.0)

    ava = TownsendAvalanche(V_loop=400.0, p_Pa=p_opt, R0=6.2, a=2.0)
    res = ava.evolve(1e-4, 100)

    assert res.time_to_full_ionization_ms > 0.0
    assert res.ne_trace[-1] > 1e16


def test_burn_through():
    bt = BurnThrough(R0=6.2, a=2.0, B0=5.3, V_loop=15.0)

    # Clean plasma: low density, low impurity fraction
    res_clean = bt.evolve(ne_19=0.1, f_imp=1e-4, dt=1e-3, n_steps=100, impurity="C")
    assert res_clean.success
    assert res_clean.time_to_burn_through_ms > 0.0

    # Dense dirty plasma: ne_19=1.0 raises radiation above ohmic power
    res_dirty = bt.evolve(ne_19=1.0, f_imp=0.9, dt=1e-3, n_steps=100, impurity="C")
    assert not res_dirty.success


def test_critical_impurity_fraction():
    bt = BurnThrough(R0=6.2, a=2.0, B0=5.3, V_loop=15.0)

    f_crit_C = bt.critical_impurity_fraction(Te_eV=10.0, ne_19=0.1, Ip_kA=100.0, impurity="C")
    f_crit_W = bt.critical_impurity_fraction(Te_eV=50.0, ne_19=0.1, Ip_kA=100.0, impurity="W")

    # W should be much lower than C
    assert f_crit_W < f_crit_C


def test_startup_sequence():
    pb = PaschenBreakdown("D2")
    p_opt = pb.optimal_prefill_pressure(V_loop_max=400.0)
    seq = StartupSequence(R0=6.2, a=2.0, B0=5.3, V_loop=400.0, p_prefill_Pa=p_opt, f_imp=1e-4)
    res = seq.run()

    assert res.success
    assert res.breakdown_time_ms > 0.0
    assert res.burn_through_time_ms > 0.0
    assert res.Ip_at_100ms_kA > 100.0


def test_startup_controller():
    ctrl = StartupController(V_loop_max=15.0, gas_puff_max=10.0)

    c1 = ctrl.step(ne=0.0, Te=0.0, Ip=0.0, t=0.05, dt=0.01)
    assert c1.phase == StartupPhase.GAS_PUFF
    assert c1.V_loop == 0.0

    c2 = ctrl.step(ne=0.0, Te=0.0, Ip=0.0, t=0.15, dt=0.01)
    assert c2.phase == StartupPhase.BREAKDOWN
    assert c2.V_loop == 15.0

    c3 = ctrl.step(ne=1e19, Te=5.0, Ip=100.0, t=0.2, dt=0.01)
    assert c3.phase == StartupPhase.BURN_THROUGH

    c4 = ctrl.step(ne=1e19, Te=100.0, Ip=200.0, t=0.3, dt=0.01)
    assert c4.phase == StartupPhase.RAMP
    assert c4.V_loop == 7.5


def test_paschen_minimum_exists():
    """V_min > 0 at finite pressure — Lieberman 2005 Eq. 14.3.2."""
    pb = PaschenBreakdown("D2")
    # V_min = e·B/A·ln(1/γ+1) > 0 by construction
    assert pb.v_paschen_min > 0.0
    # pd at minimum > 0
    assert pb.pd_at_minimum > 0.0
    # Verify the Paschen curve is U-shaped: voltages far from minimum are higher
    pd_min = pb.pd_at_minimum
    conn = 1.0
    p_min = pd_min / conn
    V_at_min = pb.breakdown_voltage(p_min, conn)
    assert V_at_min > 0.0
    assert pb.breakdown_voltage(p_min * 10.0, conn) > V_at_min
    assert pb.breakdown_voltage(p_min * 0.1, conn) > V_at_min


def test_ionization_rate_positive():
    """k_iz·n₀ > 0 for T_e > E_iz — Janev 1987 Ch. 2."""
    pb = PaschenBreakdown()
    p_opt = pb.optimal_prefill_pressure(20.0)
    ava = TownsendAvalanche(V_loop=20.0, p_Pa=p_opt, R0=6.2, a=2.0)
    # Rate must be positive for T_e well above threshold
    rate = ava.ionization_rate(_E_IZ_EV * 2.0)
    assert rate > 0.0
    # Rate must be zero at very cold Te
    assert ava.ionization_rate(0.05) == 0.0


def test_paschen_breakdown_edge_cases():
    """Lines 85, 88: pd <= 0 and log_arg <= 1 return inf."""
    pb = PaschenBreakdown("D2")
    assert pb.breakdown_voltage(0.0, 100.0) == float("inf")
    assert pb.breakdown_voltage(1e-10, 100.0) == float("inf")


def test_paschen_curve_array():
    """Line 98: paschen_curve returns an array of voltages."""
    pb = PaschenBreakdown("D2")
    p_range = np.array([0.01, 0.1, 1.0])
    curve = pb.paschen_curve(p_range, connection_length_m=100.0)
    assert len(curve) == 3
    assert all(v > 0 for v in curve)


def test_startup_sequence_no_breakdown():
    """Lines 225, 231, 292: sequence with impossible breakdown returns failure."""
    seq = StartupSequence(R0=6.2, a=2.0, B0=5.3, V_loop=0.001, p_prefill_Pa=1e-10, f_imp=0.5)
    res = seq.run()
    assert not res.success
    assert res.breakdown_time_ms == -1.0
