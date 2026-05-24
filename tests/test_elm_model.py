# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.elm_model import (
    ELM_ENERGY_FRACTION_MAX,
    ELM_ENERGY_FRACTION_MIN,
    ELMCrashModel,
    ELMCycler,
    PeelingBallooningBoundary,
    RMPSuppression,
    Type3ELMCrashModel,
    elm_power_balance_frequency,
)


def test_pb_boundary_subcritical():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    # Very low alpha and j
    assert not pb.is_unstable(alpha_edge=0.01, j_edge=1e4, s_edge=2.0)
    assert pb.stability_margin(alpha_edge=0.01, j_edge=1e4, s_edge=2.0) > 0.0


def test_pb_boundary_supercritical():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    # High alpha
    assert pb.is_unstable(alpha_edge=25.0, j_edge=1e6, s_edge=2.0)
    assert pb.stability_margin(alpha_edge=25.0, j_edge=1e6, s_edge=2.0) < 0.0


def test_elm_crash_model():
    crash = ELMCrashModel(f_elm_fraction=0.1)

    W_ped = 100.0  # MJ
    T_ped = 5.0
    n_ped = 5.0

    res = crash.crash(T_ped, n_ped, W_ped)

    # 10% of W_ped
    assert np.isclose(res.delta_W_MJ, 10.0)
    assert res.T_ped_post < T_ped
    assert res.n_ped_post < n_ped


def test_crash_profile_flattening():
    crash = ELMCrashModel(f_elm_fraction=0.1)

    rho = np.linspace(0, 1, 100)
    Te = np.ones(100) * 5.0
    ne = np.ones(100) * 5.0

    Te_new, ne_new = crash.apply_to_profiles(rho, Te, ne, rho_ped=0.9)

    # Core unchanged
    assert Te_new[0] == 5.0
    # Edge drops
    assert Te_new[95] < 5.0


def test_elm_frequency_power_balance():
    f = elm_power_balance_frequency(P_SOL_MW=100.0, W_ped_MJ=100.0, f_elm_fraction=0.1)
    # 100 / (0.1 * 100) = 100 / 10 = 10 Hz
    assert np.isclose(f, 10.0)


def test_rmp_suppression():
    rmp = RMPSuppression()
    q = np.linspace(1, 3, 50)
    rho = np.linspace(0, 1, 50)

    # Very weak B_r
    chir = rmp.chirikov_parameter(q, rho, delta_B_r=1e-6, B0=5.3, R0=6.2)
    assert not rmp.suppressed(chir)
    assert rmp.density_pump_out(chir) == 0.0

    # Strong B_r (overlap > 1)
    chir2 = rmp.chirikov_parameter(q, rho, delta_B_r=1e-1, B0=5.3, R0=6.2)
    assert rmp.suppressed(chir2)
    assert rmp.density_pump_out(chir2) > 0.0


def test_elm_cycler():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    crash = ELMCrashModel(f_elm_fraction=0.1)

    cycler = ELMCycler(pb, crash)

    # Step 1: Stable
    ev = cycler.step(0.1, alpha_edge=0.1, j_edge=1e4, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0)
    assert ev is None

    # Step 2: Unstable
    ev2 = cycler.step(0.1, alpha_edge=10.0, j_edge=1e6, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0)
    assert ev2 is not None
    assert ev2.delta_W_MJ == 10.0


def test_elm_energy_loss_fraction():
    """ΔW/W_ped ∈ [0.04, 0.15] for Type I. Loarte et al. 2003, PPCF 45, 1549, Fig. 12."""
    W_ped = 50.0
    for frac in [ELM_ENERGY_FRACTION_MIN, 0.08, ELM_ENERGY_FRACTION_MAX]:
        crash = ELMCrashModel(f_elm_fraction=frac)
        res = crash.crash(T_ped=5.0, n_ped=4.0, W_ped=W_ped)
        ratio = res.delta_W_MJ / W_ped
        assert ELM_ENERGY_FRACTION_MIN <= ratio <= ELM_ENERGY_FRACTION_MAX


def test_elm_fraction_out_of_range():
    """ELMCrashModel rejects fractions outside Type I bounds."""
    with pytest.raises(ValueError):
        ELMCrashModel(f_elm_fraction=0.001)
    with pytest.raises(ValueError):
        ELMCrashModel(f_elm_fraction=0.5)


def test_elm_type3_smaller():
    """Type III ΔW < Type I ΔW for identical W_ped. Zohm 1996, PPCF 38, 105."""
    W_ped = 50.0
    type1 = ELMCrashModel(f_elm_fraction=0.08)
    type3 = Type3ELMCrashModel()

    res1 = type1.crash(T_ped=5.0, n_ped=4.0, W_ped=W_ped)
    res3 = type3.crash(T_ped=5.0, n_ped=4.0, W_ped=W_ped)

    assert res3.delta_W_MJ < res1.delta_W_MJ


def test_rmp_single_element_profile():
    """Single-point q profiles use the documented fallback edge shear."""
    rmp = RMPSuppression()
    q = np.array([3.0])
    rho = np.array([1.0])
    chir = rmp.chirikov_parameter(q, rho, delta_B_r=0.01, B0=5.3, R0=6.2)
    assert chir > 0.0


def test_rmp_chirikov_zero_delta_B():
    """Non-positive radial perturbation gives zero Chirikov overlap."""
    rmp = RMPSuppression()
    q = np.linspace(1, 3, 50)
    rho = np.linspace(0, 1, 50)
    assert rmp.chirikov_parameter(q, rho, delta_B_r=0.0, B0=5.3, R0=6.2) == 0.0
    assert rmp.chirikov_parameter(q, rho, delta_B_r=-1.0, B0=5.3, R0=6.2) == 0.0


def test_elm_power_balance_zero_inputs():
    """Zero pedestal energy or crash fraction gives zero power-balance frequency."""
    assert elm_power_balance_frequency(P_SOL_MW=100.0, W_ped_MJ=0.0, f_elm_fraction=0.1) == 0.0
    assert elm_power_balance_frequency(P_SOL_MW=100.0, W_ped_MJ=100.0, f_elm_fraction=0.0) == 0.0


def test_peeling_limit_q95_scaling():
    """Higher q95 should reduce peeling current threshold."""
    pb_low_q = PeelingBallooningBoundary(q95=2.5, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    pb_high_q = PeelingBallooningBoundary(q95=5.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    j_low_q = pb_low_q.peeling_limit(j_edge=5e5, n_mode=10)
    j_high_q = pb_high_q.peeling_limit(j_edge=5e5, n_mode=10)
    assert j_low_q > j_high_q


def test_peeling_limit_mode_scaling():
    """Higher toroidal mode number lowers peeling stability threshold."""
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    j_n5 = pb.peeling_limit(j_edge=5e5, n_mode=5)
    j_n20 = pb.peeling_limit(j_edge=5e5, n_mode=20)
    assert j_n5 > j_n20


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"q95": 0.0, "kappa": 1.7, "delta": 0.3, "a": 2.0, "R0": 6.2}, "q95"),
        ({"q95": 3.0, "kappa": 0.0, "delta": 0.3, "a": 2.0, "R0": 6.2}, "kappa"),
        ({"q95": 3.0, "kappa": 1.7, "delta": 0.3, "a": 0.0, "R0": 6.2}, "a and R0"),
    ),
)
def test_peeling_ballooning_boundary_rejects_nonphysical_geometry(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PeelingBallooningBoundary(**kwargs)


def test_peeling_ballooning_boundary_rejects_nonphysical_limits() -> None:
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)

    with pytest.raises(ValueError, match="n_mode"):
        pb.peeling_limit(j_edge=5e5, n_mode=0)
    with pytest.raises(ValueError, match="s_edge"):
        pb.ballooning_limit(s_edge=-1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"T_ped": 0.0, "n_ped": 5.0, "W_ped": 100.0}, "T_ped"),
        ({"T_ped": 5.0, "n_ped": 0.0, "W_ped": 100.0}, "n_ped"),
        ({"T_ped": 5.0, "n_ped": 5.0, "W_ped": 0.0}, "W_ped"),
        ({"T_ped": 5.0, "n_ped": 5.0, "W_ped": 100.0, "A_wet": 0.0}, "A_wet"),
    ),
)
def test_elm_crash_rejects_nonphysical_pedestal_state(kwargs, message) -> None:
    crash = ELMCrashModel(f_elm_fraction=0.1)

    with pytest.raises(ValueError, match=message):
        crash.crash(**kwargs)


def test_elm_profile_application_rejects_nonphysical_profiles() -> None:
    crash = ELMCrashModel(f_elm_fraction=0.1)
    rho = np.linspace(0.0, 1.0, 5)
    Te = np.ones(5)
    ne = np.ones(5)

    with pytest.raises(ValueError, match="equal length"):
        crash.apply_to_profiles(rho, Te[:-1], ne, rho_ped=0.5)
    with pytest.raises(ValueError, match="sorted"):
        crash.apply_to_profiles(rho[::-1], Te, ne, rho_ped=0.5)
    with pytest.raises(ValueError, match="positive"):
        crash.apply_to_profiles(rho, np.zeros(5), ne, rho_ped=0.5)
    with pytest.raises(ValueError, match="within the rho grid"):
        crash.apply_to_profiles(rho, Te, ne, rho_ped=1.5)


@pytest.mark.parametrize("fraction", (0.0, ELM_ENERGY_FRACTION_MIN))
def test_type3_elm_rejects_nonphysical_or_type_i_fraction(fraction) -> None:
    with pytest.raises(ValueError, match="Type III"):
        Type3ELMCrashModel(f_elm_fraction=fraction)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"n_coils": 0, "I_rmp_kA": 90.0, "n_toroidal": 3}, "n_coils"),
        ({"n_coils": 3, "I_rmp_kA": -1.0, "n_toroidal": 3}, "I_rmp_kA"),
        ({"n_coils": 3, "I_rmp_kA": 90.0, "n_toroidal": 0}, "n_toroidal"),
    ),
)
def test_rmp_suppression_rejects_nonphysical_hardware_inputs(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RMPSuppression(**kwargs)


def test_rmp_chirikov_rejects_nonphysical_profiles() -> None:
    rmp = RMPSuppression()
    q = np.linspace(1.0, 3.0, 5)
    rho = np.linspace(0.0, 1.0, 5)

    with pytest.raises(ValueError, match="equal non-zero length"):
        rmp.chirikov_parameter(q[:-1], rho, delta_B_r=0.01, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="q_profile"):
        rmp.chirikov_parameter(np.zeros(5), rho, delta_B_r=0.01, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="rho"):
        rmp.chirikov_parameter(q, rho[::-1], delta_B_r=0.01, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="B0 and R0"):
        rmp.chirikov_parameter(q, rho, delta_B_r=0.01, B0=0.0, R0=6.2)


def test_elm_power_balance_rejects_negative_sol_power() -> None:
    with pytest.raises(ValueError, match="P_SOL_MW"):
        elm_power_balance_frequency(P_SOL_MW=-1.0, W_ped_MJ=100.0, f_elm_fraction=0.1)


def test_elm_cycler_rejects_nonphysical_step_inputs() -> None:
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    crash = ELMCrashModel(f_elm_fraction=0.1)
    cycler = ELMCycler(pb, crash)

    with pytest.raises(ValueError, match="dt"):
        cycler.step(0.0, alpha_edge=0.1, j_edge=1e4, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0)
    with pytest.raises(ValueError, match="non-negative"):
        cycler.step(0.1, alpha_edge=-0.1, j_edge=1e4, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0)
