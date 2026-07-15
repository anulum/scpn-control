# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density-controller tests
from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_control.control.density_controller import (
    ActuatorCommand,
    DensityControlClaimEvidence,
    DensityController,
    FuelingOptimizer,
    KalmanDensityEstimator,
    ParticleTransportModel,
    _non_empty_text,
    assert_density_control_facility_claim_admissible,
    density_control_claim_evidence,
    save_density_control_claim_evidence,
)
from scpn_control.core.pellet_injection import PelletParams, PelletTrajectory


def test_particle_transport_model_sources():
    model = ParticleTransportModel(n_rho=20, R0=6.2, a=2.0)

    gas = model.gas_puff_source(rate=1e21, penetration_depth=0.1)
    assert np.all(gas >= 0)
    assert gas[-1] > gas[0]  # Edge peaked

    pellet = model.pellet_source(speed_ms=500.0, radius_mm=2.0)
    assert np.all(pellet >= 0)
    assert np.count_nonzero(pellet) > 0

    nbi = model.nbi_source(beam_energy_keV=100.0, power_MW=10.0)
    assert np.all(nbi >= 0)
    assert nbi[5] > nbi[-1]  # Core peaked

    pump = model.cryopump_sink(pump_speed=10.0, ne_edge=1e19)
    assert pump[-1] > 0.0
    assert pump[0] == 0.0


def test_pellet_source_uses_ngs_trajectory_deposition() -> None:
    model = ParticleTransportModel(n_rho=32, R0=6.2, a=2.0)
    ne_profile = np.linspace(0.8e20, 1.2e20, model.n_rho)
    te_profile = np.linspace(8000.0, 1200.0, model.n_rho)

    pellet = model.pellet_source(
        speed_ms=700.0,
        radius_mm=2.5,
        ne_profile=ne_profile,
        Te_eV_profile=te_profile,
        B0_T=5.3,
        injection_side="HFS",
    )
    expected = PelletTrajectory(
        PelletParams(r_p_mm=2.5, v_p_m_s=700.0, injection_side="HFS"),
        R0=6.2,
        a=2.0,
        B0=5.3,
    ).simulate(model.rho, ne_profile / 1e19, te_profile)

    np.testing.assert_allclose(pellet, expected.deposition_profile, rtol=1e-12, atol=0.0)


def test_particle_transport_model_rejects_nonphysical_geometry():
    for kwargs in (
        {"n_rho": 1},
        {"R0": 0.0},
        {"a": -1.0},
    ):
        with pytest.raises(ValueError, match="physical"):
            ParticleTransportModel(**kwargs)


def test_particle_transport_model_rejects_invalid_transport_profiles():
    model = ParticleTransportModel(n_rho=10)

    with pytest.raises(ValueError, match="shape"):
        model.set_transport(np.ones(9), np.ones(10))

    with pytest.raises(ValueError, match="finite"):
        model.set_transport(np.full(10, np.nan), np.ones(10))

    with pytest.raises(ValueError, match="non-negative"):
        model.set_transport(-np.ones(10), np.ones(10))


def test_particle_transport_model_rejects_nonphysical_source_inputs():
    model = ParticleTransportModel(n_rho=10)

    with pytest.raises(ValueError, match="non-negative"):
        model.gas_puff_source(rate=-1.0)

    with pytest.raises(ValueError, match="positive"):
        model.gas_puff_source(rate=1.0, penetration_depth=0.0)

    with pytest.raises(ValueError, match="positive"):
        model.pellet_source(speed_ms=-1.0, radius_mm=1.0)

    with pytest.raises(ValueError, match="positive"):
        model.nbi_source(beam_energy_keV=0.0, power_MW=1.0)

    with pytest.raises(ValueError, match="non-negative"):
        model.cryopump_sink(pump_speed=-1.0, ne_edge=1e19)

    with pytest.raises(ValueError, match="between 0 and 1"):
        model.recycling_source(outflux=1.0, recycling_coeff=1.1)


def test_particle_transport_step_rejects_invalid_state_and_timestep():
    model = ParticleTransportModel(n_rho=10)
    ne = np.ones(10) * 1e19
    sources = np.zeros(10)

    with pytest.raises(ValueError, match="shape"):
        model.step(np.ones(9), sources, dt=1e-6)

    with pytest.raises(ValueError, match="finite"):
        model.step(np.full(10, np.nan), sources, dt=1e-6)

    with pytest.raises(ValueError, match="non-negative"):
        model.step(ne, -np.ones(10), dt=1e-6)

    with pytest.raises(ValueError, match="positive"):
        model.step(ne, sources, dt=0.0)


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


def test_particle_transport_step_without_diffusion_skips_cfl_clamp():
    """A zero diffusion profile skips the CFL clamp and evolves from sources alone."""
    model = ParticleTransportModel(n_rho=12, R0=6.2, a=2.0)
    model.set_transport(np.zeros(model.n_rho), np.zeros(model.n_rho))
    assert float(np.max(model.D)) == 0.0

    controller = DensityController(model, dt_control=0.01)
    controller.set_constraints(n_GW=1.0e20, gas_max=1.0e22, pellet_freq_max=10.0, pump_max=10.0)
    controller.set_target(np.ones(model.n_rho) * 5.0e19)
    ne_before = np.ones(model.n_rho) * 2.0e19
    sources = model.gas_puff_source(rate=1.0e20, penetration_depth=0.08)
    ne_after = model.step(ne_before, sources, dt=1.0)
    assert np.all(np.isfinite(ne_after))

    # The claim-evidence builder shares the same diffusion-free CFL guard.
    evidence = density_control_claim_evidence(
        model,
        controller,
        source="synthetic_regression_reference",
        source_id="density-control-zero-diffusion",
        geometry_source="repository circular ITER-like geometry fixture",
        transport_source="repository diffusion-free fixture",
        actuator_source="repository gas-puff and cryopump actuator limits",
        diagnostic_source="repository density profile fixture",
        ne_before=ne_before,
        ne_after=ne_after,
        sources=sources,
        command=controller.step(ne_before),
        dt_requested_s=1.0,
    )
    assert evidence.cfl_limited is False


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


def test_density_controller_small_deficit_uses_gas_without_pellets():
    """A small density deficit fuels with gas alone; the pellet branch stays inactive."""
    model = ParticleTransportModel(n_rho=10)
    ctrl = DensityController(model, dt_control=0.01)
    ctrl.set_constraints(n_GW=1.0e20, gas_max=1.0e22, pellet_freq_max=10.0, pump_max=10.0)
    ctrl.set_target(np.ones(10) * 5.0e19)

    cmd = ctrl.step(np.ones(10) * 4.9999e19)

    assert cmd.gas_puff_rate > 0.0
    assert cmd.pellet_freq == 0.0
    assert cmd.cryo_pump_speed == 0.0


def test_density_controller_rejects_nonphysical_domains():
    model = ParticleTransportModel(n_rho=10)

    with pytest.raises(ValueError, match="dt_control"):
        DensityController(model, dt_control=0.0)

    ctrl = DensityController(model, dt_control=0.01)

    with pytest.raises(ValueError, match="ne_target"):
        ctrl.set_target(np.full(10, np.nan))

    with pytest.raises(ValueError, match="ne_target"):
        ctrl.set_target(-np.ones(10))

    with pytest.raises(ValueError, match="n_GW"):
        ctrl.set_constraints(n_GW=0.0, gas_max=1e22, pellet_freq_max=10.0, pump_max=10.0)

    with pytest.raises(ValueError, match="gas_max"):
        ctrl.set_constraints(n_GW=1e20, gas_max=-1.0, pellet_freq_max=10.0, pump_max=10.0)

    with pytest.raises(ValueError, match="ne_measured"):
        ctrl.step(np.full(10, np.inf))


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
    meas = np.ones(5) * 1e19 * 2.0  # chord-integrated reference fixture
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

    for current, minor_radius in ((0.0, a_m), (I_p_MA, 0.0)):
        with pytest.raises(ValueError, match="positive"):
            DensityController.compute_greenwald_limit(current, minor_radius)


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
    """Transport setter preserves validated diffusivity and pinch profiles."""
    model = ParticleTransportModel(n_rho=10)
    D_new = np.ones(10) * 2.0
    V_new = -np.ones(10) * 0.5
    model.set_transport(D_new, V_new)
    np.testing.assert_array_equal(model.D, D_new)
    np.testing.assert_array_equal(model.V_pinch, V_new)


def test_pellet_source_zero_radius():
    """Zero-radius pellets deposit no particles and avoid trajectory integration."""
    model = ParticleTransportModel(n_rho=10)
    result = model.pellet_source(speed_ms=500.0, radius_mm=0.0)
    np.testing.assert_array_equal(result, np.zeros(10))
    result_neg = model.pellet_source(speed_ms=500.0, radius_mm=-1.0)
    np.testing.assert_array_equal(result_neg, np.zeros(10))


def test_nbi_source_zero_power():
    """Zero beam power contributes no neutral-beam particle source."""
    model = ParticleTransportModel(n_rho=10)
    result = model.nbi_source(beam_energy_keV=100.0, power_MW=0.0)
    np.testing.assert_array_equal(result, np.zeros(10))


def test_recycling_source():
    """Recycling returns a finite non-negative edge-localised source."""
    model = ParticleTransportModel(n_rho=10)
    outflux = 1e20
    recycled = model.recycling_source(outflux, recycling_coeff=0.97)
    assert recycled.shape == (10,)
    assert np.all(recycled >= 0)
    assert np.sum(recycled) > 0


def test_step_cfl_dt_clamp():
    """Transport integration remains finite when requested dt exceeds the CFL limit."""
    model = ParticleTransportModel(n_rho=10)
    ne = np.ones(10) * 1e19
    sources = np.zeros(10)
    # D_max=1.0, drho=0.111, a=2.0: dt_cfl = (0.111*2)^2 / (2*1) ~ 0.025
    # Passing dt=1.0 >> dt_cfl forces clamping
    ne_new = model.step(ne, sources, dt=1.0)
    assert np.all(np.isfinite(ne_new))


def test_greenwald_fraction():
    """Greenwald fraction is finite for a physical density profile."""
    model = ParticleTransportModel(n_rho=10, R0=6.2, a=2.0)
    ctrl = DensityController(model)
    ne = np.ones(10) * 1e19
    frac = ctrl.greenwald_fraction(ne, I_p_MA=15.0, a=2.0)
    assert 0.0 < frac < 1.0
    assert np.isfinite(frac)

    with pytest.raises(ValueError, match="ne"):
        ctrl.greenwald_fraction(-ne, I_p_MA=15.0, a=2.0)


def test_below_greenwald_safety_margin():
    """Safety-margin predicate switches at the ITER Greenwald fraction boundary."""
    model = ParticleTransportModel(n_rho=10)
    ctrl = DensityController(model)
    ctrl.n_GW = 1e20

    ne_low = np.ones(10) * 1e18
    assert ctrl.below_greenwald_safety_margin(ne_low) is True

    ne_high = np.ones(10) * 1e20
    assert ctrl.below_greenwald_safety_margin(ne_high) is False

    with pytest.raises(ValueError, match="ne"):
        ctrl.below_greenwald_safety_margin(-ne_low)


def test_kalman_predict():
    """Kalman prediction carries density state forward and inflates covariance."""
    est = KalmanDensityEstimator(n_rho=10, n_chords=4)
    ne = np.ones(10) * 1e19
    ne_pred = est.predict(ne, dt=0.01)
    np.testing.assert_array_equal(ne_pred, ne)
    assert est.P[0, 0] > 1e38  # P grew by Q*dt


def test_fueling_optimizer_zero_pellets():
    """Zero-pellet optimisation produces an empty schedule."""
    opt = FuelingOptimizer()
    sched = opt.optimize_pellet_sequence(np.zeros(10), np.ones(10), n_pellets=0, time_horizon=1.0)
    assert sched.times == []
    assert sched.speeds == []
    assert sched.sizes == []


def test_density_control_claim_evidence_records_bounded_provenance(tmp_path):
    model = ParticleTransportModel(n_rho=12, R0=6.2, a=2.0)
    controller = DensityController(model, dt_control=0.01)
    controller.set_constraints(n_GW=1.0e20, gas_max=1.0e22, pellet_freq_max=10.0, pump_max=10.0)
    controller.set_target(np.ones(model.n_rho) * 5.0e19)
    ne_before = np.ones(model.n_rho) * 2.0e19
    sources = model.gas_puff_source(rate=1.0e20, penetration_depth=0.08)
    ne_after = model.step(ne_before, sources, dt=1.0)
    command = controller.step(ne_before)

    evidence = density_control_claim_evidence(
        model,
        controller,
        source="synthetic_regression_reference",
        source_id="density-control-bounded-regression-v1",
        geometry_source="repository circular ITER-like geometry fixture",
        transport_source="repository finite-volume diffusion-pinch fixture",
        actuator_source="repository gas-puff and cryopump actuator limits",
        diagnostic_source="repository density profile fixture",
        ne_before=ne_before,
        ne_after=ne_after,
        sources=sources,
        command=command,
        dt_requested_s=1.0,
    )
    report_path = tmp_path / "density_control_claim.json"
    save_density_control_claim_evidence(evidence, report_path)

    assert isinstance(evidence, DensityControlClaimEvidence)
    assert evidence.facility_density_claim_allowed is False
    assert evidence.claim_status == "bounded_density_control_evidence"
    assert evidence.n_rho == model.n_rho
    assert evidence.cfl_limited is True
    assert evidence.greenwald_fraction >= 0.0
    assert evidence.below_iter_greenwald_margin is True
    assert evidence.total_source_particles_per_s > 0.0
    assert np.isfinite(evidence.particle_inventory_delta)
    assert '"facility_density_claim_allowed": false' in report_path.read_text(encoding="utf-8")


def test_density_control_facility_admission_requires_matched_greenwald_and_inventory_references():
    model = ParticleTransportModel(n_rho=12, R0=6.2, a=2.0)
    controller = DensityController(model, dt_control=0.01)
    controller.set_constraints(n_GW=1.0e20, gas_max=1.0e22, pellet_freq_max=10.0, pump_max=10.0)
    controller.set_target(np.ones(model.n_rho) * 5.0e19)
    ne_before = np.ones(model.n_rho) * 2.0e19
    sources = model.gas_puff_source(rate=1.0e20, penetration_depth=0.08)
    ne_after = model.step(ne_before, sources, dt=1.0e-4)
    command = controller.step(ne_before)
    base = density_control_claim_evidence(
        model,
        controller,
        source="synthetic_regression_reference",
        source_id="density-reference-base",
        geometry_source="repository circular ITER-like geometry fixture",
        transport_source="repository finite-volume diffusion-pinch fixture",
        actuator_source="repository gas-puff and cryopump actuator limits",
        diagnostic_source="repository density profile fixture",
        ne_before=ne_before,
        ne_after=ne_after,
        sources=sources,
        command=command,
        dt_requested_s=1.0e-4,
    )

    matched = density_control_claim_evidence(
        model,
        controller,
        source="facility_replay",
        source_id="matched-density-reference",
        geometry_source="documented geometry",
        transport_source="documented density transport replay",
        actuator_source="documented actuator calibration",
        diagnostic_source="documented interferometer profile",
        ne_before=ne_before,
        ne_after=ne_after,
        sources=sources,
        command=command,
        dt_requested_s=1.0e-4,
        reference_greenwald_fraction=base.greenwald_fraction,
        reference_inventory_delta=base.particle_inventory_delta,
    )
    assert_density_control_facility_claim_admissible(matched)
    assert matched.facility_density_claim_allowed is True

    mismatched = density_control_claim_evidence(
        model,
        controller,
        source="facility_replay",
        source_id="mismatched-density-reference",
        geometry_source="documented geometry",
        transport_source="documented density transport replay",
        actuator_source="documented actuator calibration",
        diagnostic_source="documented interferometer profile",
        ne_before=ne_before,
        ne_after=ne_after,
        sources=sources,
        command=command,
        dt_requested_s=1.0e-4,
        reference_greenwald_fraction=base.greenwald_fraction + 0.1,
        reference_inventory_delta=base.particle_inventory_delta,
        greenwald_fraction_abs_tolerance=0.01,
    )
    with pytest.raises(ValueError, match="facility density-control claim requires matched"):
        assert_density_control_facility_claim_admissible(mismatched)
    assert mismatched.facility_density_claim_allowed is False


def test_density_control_claim_evidence_rejects_invalid_inputs():
    model = ParticleTransportModel(n_rho=8, R0=6.2, a=2.0)
    controller = DensityController(model, dt_control=0.01)
    controller.set_target(np.ones(model.n_rho) * 5.0e19)
    ne_before = np.ones(model.n_rho) * 2.0e19
    sources = np.zeros(model.n_rho)
    ne_after = model.step(ne_before, sources, dt=1.0e-4)
    command = ActuatorCommand(0.0, 0.0, 500.0, 0.0)

    with pytest.raises(ValueError, match="source must be one of"):
        density_control_claim_evidence(
            model,
            controller,
            source="untracked_reference",
            source_id="bad-source",
            geometry_source="documented geometry",
            transport_source="documented transport",
            actuator_source="documented actuators",
            diagnostic_source="documented diagnostics",
            ne_before=ne_before,
            ne_after=ne_after,
            sources=sources,
            command=command,
            dt_requested_s=1.0e-4,
        )

    with pytest.raises(ValueError, match="sources"):
        density_control_claim_evidence(
            model,
            controller,
            source="facility_replay",
            source_id="bad-sources",
            geometry_source="documented geometry",
            transport_source="documented transport",
            actuator_source="documented actuators",
            diagnostic_source="documented diagnostics",
            ne_before=ne_before,
            ne_after=ne_after,
            sources=-np.ones(model.n_rho),
            command=command,
            dt_requested_s=1.0e-4,
        )

    with pytest.raises(ValueError, match="dt_requested_s"):
        density_control_claim_evidence(
            model,
            controller,
            source="facility_replay",
            source_id="bad-dt",
            geometry_source="documented geometry",
            transport_source="documented transport",
            actuator_source="documented actuators",
            diagnostic_source="documented diagnostics",
            ne_before=ne_before,
            ne_after=ne_after,
            sources=sources,
            command=command,
            dt_requested_s=0.0,
        )


def test_pellet_source_rejects_nonfinite_radius() -> None:
    model = ParticleTransportModel(n_rho=16)
    with pytest.raises(ValueError, match="radius_mm must be finite"):
        model.pellet_source(speed_ms=200.0, radius_mm=math.nan)


def test_pellet_source_rejects_nonfinite_launch_angle() -> None:
    model = ParticleTransportModel(n_rho=16)
    with pytest.raises(ValueError, match="launch_angle_deg must be finite"):
        model.pellet_source(speed_ms=200.0, radius_mm=3.0, launch_angle_deg=math.inf)


def test_pellet_source_rejects_nonpositive_b0() -> None:
    model = ParticleTransportModel(n_rho=16)
    with pytest.raises(ValueError, match="B0_T must be finite and positive"):
        model.pellet_source(speed_ms=200.0, radius_mm=3.0, B0_T=0.0)


def test_pellet_source_rejects_unknown_injection_side() -> None:
    model = ParticleTransportModel(n_rho=16)
    with pytest.raises(ValueError, match="injection_side must be 'HFS' or 'LFS'"):
        model.pellet_source(speed_ms=200.0, radius_mm=3.0, injection_side="TOP")


def test_pellet_source_rejects_nonpositive_density_profile() -> None:
    model = ParticleTransportModel(n_rho=16)
    ne_profile = np.full(model.n_rho, 1.0e20)
    ne_profile[3] = 0.0
    with pytest.raises(ValueError, match="ne_profile must be positive"):
        model.pellet_source(speed_ms=200.0, radius_mm=3.0, ne_profile=ne_profile)


def test_pellet_source_rejects_negative_temperature_profile() -> None:
    model = ParticleTransportModel(n_rho=16)
    te_profile = np.full(model.n_rho, 2000.0)
    te_profile[5] = -1.0
    with pytest.raises(ValueError, match="Te_eV_profile must be non-negative"):
        model.pellet_source(speed_ms=200.0, radius_mm=3.0, Te_eV_profile=te_profile)


def test_nbi_source_rejects_negative_power() -> None:
    model = ParticleTransportModel(n_rho=16)
    with pytest.raises(ValueError, match="NBI power_MW must be finite and non-negative"):
        model.nbi_source(beam_energy_keV=100.0, power_MW=-1.0)


def test_cryopump_sink_rejects_negative_edge_density() -> None:
    model = ParticleTransportModel(n_rho=16)
    with pytest.raises(ValueError, match="Cryopump ne_edge must be finite and non-negative"):
        model.cryopump_sink(pump_speed=1.0, ne_edge=-1.0)


def test_recycling_source_rejects_negative_outflux() -> None:
    model = ParticleTransportModel(n_rho=16)
    with pytest.raises(ValueError, match="Recycling outflux must be finite and non-negative"):
        model.recycling_source(outflux=-1.0)


def test_step_rejects_negative_density() -> None:
    model = ParticleTransportModel(n_rho=16)
    ne = np.full(model.n_rho, 1.0e20)
    ne[2] = -1.0
    sources = np.zeros(model.n_rho)
    with pytest.raises(ValueError, match="ne must be non-negative"):
        model.step(ne, sources, dt=0.01)


def test_non_empty_text_rejects_blank_string() -> None:
    assert _non_empty_text("source_id", "  iter-12345  ") == "iter-12345"
    with pytest.raises(ValueError, match="source_id must be a non-empty string"):
        _non_empty_text("source_id", "   ")
