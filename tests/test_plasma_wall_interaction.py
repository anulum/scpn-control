# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Plasma-Wall Interaction Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.plasma_wall_interaction import (
    E_TH_D_W_EV,
    DivertorLifetimeAssessment,
    ErosionModel,
    SputteringYield,
    TransientThermalLoad,
    WallThermalModel,
)
from scpn_control.core.sol_model import TwoPointSOL


def test_sputtering_threshold():
    """
    No sputtering below E_th ≈ 160 eV (D→W).
    Behrisch & Eckstein 2007, "Sputtering by Particle Bombardment", Table 3.1.
    """
    sputt = SputteringYield()

    assert sputt.yield_at_energy(100.0) == 0.0

    Y_1keV = sputt.yield_at_energy(1000.0)
    assert 0.001 < Y_1keV < 0.05


def test_sputtering_angular_dependence():
    sputt = SputteringYield()

    Y_0 = sputt.yield_at_energy(1000.0, 0.0)
    Y_60 = sputt.yield_at_energy(1000.0, 60.0)

    assert Y_60 > Y_0


def test_sputtering_rejects_non_d_to_w_configuration():
    with pytest.raises(ValueError, match="D-to-W"):
        SputteringYield(target="C")

    with pytest.raises(ValueError, match="D-to-W"):
        SputteringYield(projectile="He")


def test_sputtering_rejects_nonphysical_incidence_angles():
    sputt = SputteringYield()

    with pytest.raises(ValueError, match="theta_deg"):
        sputt.yield_at_energy(1000.0, theta_deg=-1.0)

    with pytest.raises(ValueError, match="theta_deg"):
        sputt.yield_at_energy(1000.0, theta_deg=90.0)


def test_erosion_model():
    erosion = ErosionModel()

    flux = 1e24  # High flux
    gross = erosion.gross_erosion_rate(flux, 1000.0)

    assert gross > 0.0

    net = erosion.net_erosion_rate(gross, f_redeposition=0.99)
    assert np.isclose(net, gross * 0.01)


def test_erosion_model_rejects_nonphysical_domains():
    with pytest.raises(ValueError, match="n_atom"):
        ErosionModel(n_atom=0.0)

    erosion = ErosionModel()

    with pytest.raises(ValueError, match="ion_flux"):
        erosion.gross_erosion_rate(-1.0, 1000.0)

    with pytest.raises(ValueError, match="f_redeposition"):
        erosion.net_erosion_rate(1.0e20, f_redeposition=1.01)

    with pytest.raises(ValueError, match="wall_thickness_mm"):
        erosion.lifetime_estimate(wall_thickness_mm=0.0, net_rate_m_s=1.0e-9)


def test_wall_thermal_steady_state():
    wall = WallThermalModel()

    T_steady = wall.step(10.0, 10.0)  # 10 MW/m2

    assert T_steady > 400.0
    assert T_steady < wall.T_melt
    assert not wall.is_melted()


def test_wall_thermal_model_rejects_invalid_geometry_and_step_domains():
    with pytest.raises(ValueError, match="thickness_mm"):
        WallThermalModel(thickness_mm=0.0)

    with pytest.raises(ValueError, match="n_nodes"):
        WallThermalModel(n_nodes=1)

    wall = WallThermalModel()

    with pytest.raises(ValueError, match="dt"):
        wall.step(dt=0.0, q_surface_MW_m2=10.0)

    with pytest.raises(ValueError, match="q_surface_MW_m2"):
        wall.step(dt=1.0, q_surface_MW_m2=-1.0)


def test_wall_thermal_melting():
    wall = WallThermalModel()

    T_steady = wall.step(10.0, 100.0)  # 100 MW/m2 -> guaranteed to melt W

    assert T_steady > wall.T_melt
    assert wall.is_melted()


def test_transient_thermal_load():
    wall = WallThermalModel()
    trans = TransientThermalLoad(wall)

    delta_T = trans.elm_load(delta_W_MJ=20.0, A_wet_m2=2.0)

    # 20 MJ over 2 m2 in 0.25 ms is a massive load
    assert delta_T > 1000.0


def test_transient_thermal_load_rejects_negative_energy():
    wall = WallThermalModel()
    trans = TransientThermalLoad(wall)

    with pytest.raises(ValueError, match="delta_W_MJ"):
        trans.elm_load(delta_W_MJ=-1.0, A_wet_m2=2.0)


def test_divertor_lifetime_assessment():
    sol = TwoPointSOL(R0=6.2, a=2.0, q95=3.0, B_pol=0.56)
    sputt = SputteringYield()
    eros = ErosionModel()
    wall = WallThermalModel()

    assessment = DivertorLifetimeAssessment(sol, sputt, eros, wall)

    rep = assessment.assess(P_SOL_MW=100.0, n_u_19=4.0, f_ELM_Hz=5.0, delta_W_ELM_MJ=1.0)

    assert rep.limiting_factor in ["Erosion", "Fatigue", "Melting"]
    if rep.limiting_factor == "Melting":
        assert rep.lifetime_years == 0.0
    else:
        assert rep.lifetime_years > 0.0


def test_divertor_lifetime_no_melting_below_melt_temperature():
    """A low-power scenario stays below the melt temperature (branch 389->393).

    When the peak ELM surface temperature does not exceed the wall melt point,
    the melting override is skipped and the reported lifetime is set by erosion
    or fatigue rather than being zeroed out.
    """
    sol = TwoPointSOL(R0=6.2, a=2.0, q95=3.0, B_pol=0.56)
    assessment = DivertorLifetimeAssessment(sol, SputteringYield(), ErosionModel(), WallThermalModel())

    rep = assessment.assess(P_SOL_MW=0.5, n_u_19=10.0, f_ELM_Hz=20.0, delta_W_ELM_MJ=0.005)

    assert rep.limiting_factor != "Melting"
    assert rep.peak_T_elm_K < WallThermalModel().T_melt
    assert rep.lifetime_years > 0.0


def test_sputtering_below_threshold_zero():
    """
    Y = 0 for E_ion ≤ E_th.
    Behrisch & Eckstein 2007, "Sputtering by Particle Bombardment", Table 3.1.
    """
    sputt = SputteringYield()
    assert sputt.yield_at_energy(E_TH_D_W_EV) == 0.0
    assert sputt.yield_at_energy(E_TH_D_W_EV - 1.0) == 0.0
    assert sputt.yield_at_energy(1.0) == 0.0


def test_sputtering_yield_positive_above_threshold():
    """
    Y > 0 for E_ion > E_th.
    Eckstein & Preuss 2003, J. Nucl. Mater. 320, 209.
    """
    sputt = SputteringYield()
    Y = sputt.yield_at_energy(E_TH_D_W_EV + 100.0)
    assert Y > 0.0


def test_sputtering_yield_monotone_above_threshold():
    """Y increases from threshold through the near-threshold region."""
    sputt = SputteringYield()
    Y_low = sputt.yield_at_energy(E_TH_D_W_EV + 50.0)
    Y_high = sputt.yield_at_energy(E_TH_D_W_EV + 500.0)
    assert Y_high > Y_low


def test_erosion_lifetime_zero_rate():
    """Zero net erosion gives an infinite erosion-limited lifetime."""
    erosion = ErosionModel()
    assert erosion.lifetime_estimate(wall_thickness_mm=10.0, net_rate_m_s=0.0) == float("inf")


def test_tritium_retention():
    """Tritium retention scales linearly with fluence in the fitted regime."""
    erosion = ErosionModel()
    ret_low = erosion.tritium_retention(ion_fluence_D_m2=1e23)
    ret_high = erosion.tritium_retention(ion_fluence_D_m2=1e24)
    assert ret_high > ret_low
    assert ret_high > 0.0


def test_transient_elm_zero_area():
    """Unavailable wetted area or pulse duration gives no resolvable transient load."""
    wall = WallThermalModel()
    trans = TransientThermalLoad(wall)
    assert trans.elm_load(delta_W_MJ=10.0, A_wet_m2=0.0) == 0.0
    assert trans.elm_load(delta_W_MJ=10.0, A_wet_m2=1.0, tau_IR_ms=0.0) == 0.0


def test_fatigue_small_delta_t():
    """Small temperature excursions remain in the long-cycle fatigue regime."""
    wall = WallThermalModel()
    trans = TransientThermalLoad(wall)
    cycles = trans.n_elm_cycles_to_fatigue(delta_T_K=50.0)
    assert cycles == 10000000


def test_erosion_lifetime_finite_for_positive_net_rate() -> None:
    """A positive net erosion rate gives a finite component lifetime in years."""
    erosion = ErosionModel()
    seconds_per_year = 365.25 * 24 * 3600
    lifetime = erosion.lifetime_estimate(wall_thickness_mm=5.0, net_rate_m_s=1.0e-9)
    assert lifetime == pytest.approx((5.0e-3) / (1.0e-9 * seconds_per_year))
    assert lifetime > 0.0


def test_transient_disruption_load_delegates_to_elm_load() -> None:
    """The disruption-load wrapper returns the same peak temperature rise as elm_load."""
    trans = TransientThermalLoad(WallThermalModel())
    direct = trans.elm_load(delta_W_MJ=20.0, A_wet_m2=2.0, tau_IR_ms=1.0)
    via_wrapper = trans.disruption_load(W_th_MJ=20.0, A_wet_m2=2.0, tau_TQ_ms=1.0)
    assert via_wrapper == pytest.approx(direct)
    assert via_wrapper > 0.0
