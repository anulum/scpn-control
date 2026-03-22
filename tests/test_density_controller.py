# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np

from scpn_control.control.density_controller import (
    DensityController,
    FuelingOptimizer,
    KalmanDensityEstimator,
    ParticleTransportModel,
)


def test_particle_transport_model_sources():
    model = ParticleTransportModel(n_rho=20, R0=6.2, a=2.0)

    gas = model.gas_puff_source(rate=1e21, penetration_depth=0.1)
    assert np.all(gas >= 0)
    assert gas[-1] > gas[0]  # Edge peaked

    pellet = model.pellet_source(speed_ms=500.0, radius_mm=2.0)
    assert np.all(pellet >= 0)
    assert pellet[10] > pellet[0]  # Deeply deposited

    nbi = model.nbi_source(beam_energy_keV=100.0, power_MW=10.0)
    assert np.all(nbi >= 0)
    assert nbi[5] > nbi[-1]  # Core peaked

    pump = model.cryopump_sink(pump_speed=10.0, ne_edge=1e19)
    assert pump[-1] > 0.0
    assert pump[0] == 0.0


def test_particle_transport_model_step():
    model = ParticleTransportModel(n_rho=10)
    ne = np.ones(10) * 1e19
    sources = np.zeros(10)
    sources[-1] = 1e20  # Strong edge puff

    # Use extremely small dt to avoid explicit diffusion instability (CFL limit)
    ne_new = model.step(ne, sources, dt=1e-6)

    # Check shape
    assert ne_new.shape == (10,)
    # Just verify no NaN or Inf (stability check)
    assert np.all(np.isfinite(ne_new))


def test_density_controller():
    model = ParticleTransportModel(n_rho=10)
    ctrl = DensityController(model, dt_control=0.01)

    # Target high density
    target = np.ones(10) * 5e19
    ctrl.set_target(target)

    # Measure low density
    meas_low = np.ones(10) * 1e19
    cmd = ctrl.step(meas_low)

    assert cmd.gas_puff_rate > 0.0
    assert cmd.cryo_pump_speed == 0.0

    # Measure high density -> pump
    meas_high = np.ones(10) * 6e19
    ctrl.integral_error = 0.0  # reset for clean test
    cmd_high = ctrl.step(meas_high)

    assert cmd_high.gas_puff_rate == 0.0
    assert cmd_high.cryo_pump_speed > 0.0


def test_greenwald_limit_override():
    model = ParticleTransportModel(n_rho=10)
    ctrl = DensityController(model, dt_control=0.01)

    target = np.ones(10) * 5e19
    ctrl.set_target(target)

    # Set n_GW such that meas is over Greenwald
    ctrl.set_constraints(n_GW=1e19, gas_max=1e22, pellet_freq_max=10.0, pump_max=10.0)

    meas = np.ones(10) * 2e19  # Higher than n_GW = 1e19 -> f_GW > 1.0
    cmd = ctrl.step(meas)

    # Should aggressively pump, regardless of target
    assert cmd.gas_puff_rate == 0.0
    assert cmd.cryo_pump_speed == ctrl.pump_max


def test_kalman_estimator():
    est = KalmanDensityEstimator(n_rho=20, n_chords=5)

    ne_pred = np.ones(20) * 1e19
    meas = np.ones(5) * 1e19 * 2.0  # chord integrated mock
    angles = np.zeros(5)

    ne_upd = est.update(ne_pred, meas, angles)
    assert ne_upd.shape == (20,)


def test_fueling_optimizer():
    opt = FuelingOptimizer()
    sched = opt.optimize_pellet_sequence(np.zeros(10), np.ones(10), n_pellets=3, time_horizon=1.0)

    assert len(sched.times) == 3
    assert len(sched.speeds) == 3
    assert len(sched.sizes) == 3
    assert sched.times[0] == 0.25  # 1.0 / 4


def test_greenwald_limit_iter():
    """n_GW ≈ 1.19×10^20 m^-3 for ITER (15 MA, a=2.0 m).

    Greenwald 2002, PPCF 44, R27, Eq. 1: n_GW = I_p / (π a²) [10^20 m^-3].
    ITER parameters: I_p = 15 MA, a = 2.0 m.
    """
    I_p_MA = 15.0
    a_m = 2.0
    n_GW = DensityController.compute_greenwald_limit(I_p_MA, a_m)

    # Expected: 15 / (π × 4) × 10^20 ≈ 1.194×10^20 m^-3
    expected = 15.0 / (math.pi * 4.0) * 1e20
    assert abs(n_GW - expected) < 1e16, f"n_GW={n_GW:.4e} expected≈{expected:.4e}"
    # Must be between 1.0 and 1.3 × 10^20 m^-3 for ITER parameters.
    assert 1.0e20 < n_GW < 1.3e20


def test_density_below_greenwald():
    """Controller keeps n < n_GW after triggering the pump-out threshold.

    When n/n_GW exceeds 0.95 the controller activates maximum pumping with
    zero fueling, pushing density back below the limit.
    Greenwald 2002, PPCF 44, R27, Eq. 1.
    """
    model = ParticleTransportModel(n_rho=10, R0=6.2, a=2.0)
    ctrl = DensityController(model, dt_control=0.01)

    # Set n_GW to 1e19 so that a flat profile of 2×10^19 is well above limit.
    ctrl.set_constraints(n_GW=1e19, gas_max=1e22, pellet_freq_max=10.0, pump_max=10.0)
    ctrl.set_target(np.ones(10) * 5e19)

    ne_over = np.ones(10) * 2e19
    cmd = ctrl.step(ne_over)

    assert cmd.gas_puff_rate == 0.0, "No fueling when above Greenwald limit"
    assert cmd.cryo_pump_speed == ctrl.pump_max, "Max pumping when above Greenwald limit"


def test_set_transport():
    """Lines 42-43: ParticleTransportModel.set_transport sets D and V_pinch."""
    model = ParticleTransportModel(n_rho=10)
    D_new = np.ones(10) * 2.0
    V_new = -np.ones(10) * 0.5
    model.set_transport(D_new, V_new)
    np.testing.assert_array_equal(model.D, D_new)
    np.testing.assert_array_equal(model.V_pinch, V_new)


def test_pellet_source_zero_radius():
    """Line 64: pellet_source returns zeros when radius_mm <= 0."""
    model = ParticleTransportModel(n_rho=10)
    result = model.pellet_source(speed_ms=500.0, radius_mm=0.0)
    np.testing.assert_array_equal(result, np.zeros(10))
    result_neg = model.pellet_source(speed_ms=500.0, radius_mm=-1.0)
    np.testing.assert_array_equal(result_neg, np.zeros(10))


def test_nbi_source_zero_power():
    """Line 78: nbi_source returns zeros when power_MW <= 0."""
    model = ParticleTransportModel(n_rho=10)
    result = model.nbi_source(beam_energy_keV=100.0, power_MW=0.0)
    np.testing.assert_array_equal(result, np.zeros(10))


def test_recycling_source():
    """Line 98: recycling_source delegates to gas_puff_source with 0.97 coeff."""
    model = ParticleTransportModel(n_rho=10)
    outflux = 1e20
    recycled = model.recycling_source(outflux, recycling_coeff=0.97)
    assert recycled.shape == (10,)
    assert np.all(recycled >= 0)
    assert np.sum(recycled) > 0


def test_step_cfl_dt_clamp():
    """Lines 105-106: step clamps dt to CFL limit when dt exceeds it."""
    model = ParticleTransportModel(n_rho=10)
    ne = np.ones(10) * 1e19
    sources = np.zeros(10)
    # D_max=1.0, drho=0.111, a=2.0: dt_cfl = (0.111*2)^2 / (2*1) ~ 0.025
    # Passing dt=1.0 >> dt_cfl forces clamping
    ne_new = model.step(ne, sources, dt=1.0)
    assert np.all(np.isfinite(ne_new))


def test_greenwald_fraction():
    """Lines 190-195: greenwald_fraction computes volume-averaged n / n_GW."""
    model = ParticleTransportModel(n_rho=10, R0=6.2, a=2.0)
    ctrl = DensityController(model)
    ne = np.ones(10) * 1e19
    frac = ctrl.greenwald_fraction(ne, I_p_MA=15.0, a=2.0)
    assert 0.0 < frac < 1.0
    assert np.isfinite(frac)


def test_below_greenwald_safety_margin():
    """Lines 202-204: below_greenwald_safety_margin returns True/False."""
    model = ParticleTransportModel(n_rho=10)
    ctrl = DensityController(model)
    ctrl.n_GW = 1e20

    ne_low = np.ones(10) * 1e18
    assert ctrl.below_greenwald_safety_margin(ne_low) is True

    ne_high = np.ones(10) * 1e20
    assert ctrl.below_greenwald_safety_margin(ne_high) is False


def test_kalman_predict():
    """Lines 255-257: KalmanDensityEstimator.predict."""
    est = KalmanDensityEstimator(n_rho=10, n_chords=4)
    ne = np.ones(10) * 1e19
    ne_pred = est.predict(ne, dt=0.01)
    np.testing.assert_array_equal(ne_pred, ne)
    assert est.P[0, 0] > 1e38  # P grew by Q*dt


def test_fueling_optimizer_zero_pellets():
    """Line 282: optimize_pellet_sequence with n_pellets=0 returns empty."""
    opt = FuelingOptimizer()
    sched = opt.optimize_pellet_sequence(np.zeros(10), np.ones(10), n_pellets=0, time_horizon=1.0)
    assert sched.times == []
    assert sched.speeds == []
    assert sched.sizes == []
