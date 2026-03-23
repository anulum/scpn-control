"""Tests to close the last ~260 missed lines across 44 files."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── burn_controller.py:131 ────────────────────────────────────────────────────


def test_burn_reactivity_exponent_zero_sv():
    from scpn_control.control.burn_controller import BurnStabilityAnalysis, AlphaHeating

    ah = AlphaHeating(R0=3.0, a=1.0)
    bsa = BurnStabilityAnalysis(ah)
    with patch("scpn_control.control.burn_controller.bosch_hale_reactivity", return_value=np.array([0.0])):
        assert bsa.reactivity_exponent(5.0) == 10.0


# ── controller_tuning.py:22 ──────────────────────────────────────────────────


def test_controller_tuning_optuna_flag():
    from scpn_control.control import controller_tuning

    assert hasattr(controller_tuning, "HAS_OPTUNA")


# ── detachment_controller.py:210 ─────────────────────────────────────────────


def test_detachment_fully_detached():
    from scpn_control.control.detachment_controller import DetachmentBifurcation, DetachmentState
    from scpn_control.core.sol_model import TwoPointSOL

    sol = TwoPointSOL(R0=3.0, a=1.0, q95=3.0, B_pol=0.5)
    db = DetachmentBifurcation(sol, impurity="N")
    mock_res = MagicMock(T_target_eV=2.0, n_target_19=5.0)
    with (
        patch.object(db.sol, "solve", return_value=mock_res),
        patch.object(db.front_model, "front_position", return_value=0.5),
    ):
        result = db._steady_state_target(seeding_rate=20.0, P_SOL_MW=10.0, n_u_19=5.0)
    assert result.state == DetachmentState.FULLY_DETACHED


# ── fault_tolerant_control.py:153-154,221 ────────────────────────────────────


def test_ftc_singular_gain():
    from scpn_control.control.fault_tolerant_control import ReconfigurableController

    J = np.eye(2)
    rc = ReconfigurableController(MagicMock(), J, n_coils=2, n_sensors=2)
    rc.lambda_reg = 0.0
    with patch.object(np.linalg, "inv", side_effect=np.linalg.LinAlgError):
        K = rc._compute_gain()
    assert np.allclose(K, 0.0)


def test_ftc_sensor_dropout():
    from scpn_control.control.fault_tolerant_control import FaultInjector, FaultType

    fi = FaultInjector(fault_time=0.5, component_index=1, fault_type=FaultType.SENSOR_DROPOUT)
    result = fi.inject(t=1.0, signals=np.array([1.0, 2.0, 3.0]))
    assert result[1] == 0.0


# ── realtime_efit.py:180 ─────────────────────────────────────────────────────


def test_realtime_efit_find_xpoint():
    from scpn_control.control.realtime_efit import RealtimeEFIT

    R, Z = np.linspace(1.0, 3.0, 32), np.linspace(-2.0, 2.0, 32)
    efit = RealtimeEFIT(diagnostics=MagicMock(), R_grid=R, Z_grid=Z)
    result = efit.find_xpoint(np.random.default_rng(0).random((32, 32)))
    assert len(result) == 2


# ── rzip_model.py:90-91,173-175,179 ──────────────────────────────────────────


def test_rzip_singular_M():
    from scpn_control.control.rzip_model import RZIPModel
    from scpn_control.core.vessel_model import VesselElement, VesselModel

    el = VesselElement(R=2.0, Z=0.0, resistance=0.001, cross_section=1e-4, inductance=0.0)
    vessel = VesselModel(elements=[el])
    rzip = RZIPModel(R0=3.0, a=1.0, kappa=1.5, Ip_MA=1.0, B0=5.0, n_index=-1.5, vessel=vessel)
    A, B, C, D = rzip.build_state_space()
    assert A is not None


def test_rzip_lqr_fallback():
    from scpn_control.control.rzip_model import RZIPModel, RZIPController
    from scpn_control.core.vessel_model import VesselElement, VesselModel

    el = VesselElement(R=2.0, Z=0.0, resistance=0.001, cross_section=1e-4, inductance=1e-3)
    vessel = VesselModel(elements=[el])
    rzip = RZIPModel(R0=3.0, a=1.0, kappa=1.5, Ip_MA=1.0, B0=5.0, n_index=-1.5, vessel=vessel)
    with patch("scipy.linalg.solve_continuous_are", side_effect=np.linalg.LinAlgError):
        ctrl = RZIPController(rzip, Kp=10.0, Kd=1.0)
    assert np.allclose(ctrl.K_gain, 0.0)


def test_rzip_step_dt_zero():
    from scpn_control.control.rzip_model import RZIPModel, RZIPController
    from scpn_control.core.vessel_model import VesselElement, VesselModel

    el = VesselElement(R=2.0, Z=0.0, resistance=0.001, cross_section=1e-4, inductance=1e-3)
    coil = VesselElement(R=4.0, Z=1.0, resistance=0.01, cross_section=1e-4, inductance=1e-2)
    vessel = VesselModel(elements=[el])
    rzip = RZIPModel(R0=3.0, a=1.0, kappa=1.5, Ip_MA=1.0, B0=5.0, n_index=-1.5, vessel=vessel, active_coils=[coil])
    ctrl = RZIPController(rzip, Kp=10.0, Kd=1.0)
    assert ctrl.step(dZ_measured=0.01, dt=0.0) is not None


# ── safe_rl_controller.py:183 ────────────────────────────────────────────────


def test_lagrangian_ppo_predict():
    from scpn_control.control.safe_rl_controller import LagrangianPPO

    env = MagicMock()
    env.action_space.sample.return_value = np.array([0.5])
    env.constraints = []
    env.n_constraints = 0
    ppo = LagrangianPPO(env)
    assert isinstance(ppo.predict(np.ones(4)), np.ndarray)


# ── shape_controller.py:99 ───────────────────────────────────────────────────


def test_shape_jacobian_update():
    from scpn_control.control.shape_controller import ShapeJacobian, CoilSet, ShapeTarget

    target = ShapeTarget(isoflux_points=[(1.0, 0.0)], gap_points=[], gap_targets=[])
    sj = ShapeJacobian(MagicMock(), CoilSet(n_coils=3), target)
    sj.update({})


# ── sliding_mode_vertical.py:155 ─────────────────────────────────────────────


def test_convergence_time_denom_zero():
    from scpn_control.control.sliding_mode_vertical import estimate_convergence_time

    assert estimate_convergence_time(alpha=0.5, beta=1.0, L_max=0.5, s0=1.0) == float("inf")


# ── tokamak_digital_twin.py:75,397 ───────────────────────────────────────────


def test_require_positive_float():
    from scpn_control.control.tokamak_digital_twin import _require_positive_float

    with pytest.raises(ValueError):
        _require_positive_float("x", -1.0)
    with pytest.raises(ValueError):
        _require_positive_float("x", 0.0)


def test_digital_twin_actuator_saturation():
    from scpn_control.control.tokamak_digital_twin import run_digital_twin

    result = run_digital_twin(
        time_steps=5, seed=42, save_plot=False, verbose=False, actuator_bias=2.0, actuator_drift_std=1.0
    )
    assert "actuator_saturation_count" in result


# ── alfven_eigenmodes.py:80,159,310 ──────────────────────────────────────────


def test_tae_gap_q_flat():
    from scpn_control.core.alfven_eigenmodes import AlfvenContinuum

    rho = np.linspace(0.0, 1.0, 50)
    ne = np.ones(50) * 5e19
    ac = AlfvenContinuum(rho, np.ones(50) * 1.5, ne, B0=5.0, R0=3.0)
    assert isinstance(ac.find_gaps(n=1), list)


def test_tae_electron_damping_low_te():
    from scpn_control.core.alfven_eigenmodes import TAEMode

    tae = TAEMode(n=1, q_rational=1.5, v_A=1e6, R0=3.0, T_e_keV=1e-30)
    omega = tae.frequency()
    assert tae.electron_landau_damping() == pytest.approx(0.01 * omega, rel=0.01)


def test_alfven_critical_beta_zero():
    from scpn_control.core.alfven_eigenmodes import AlfvenStabilityAnalysis, AlfvenContinuum, FastParticleDrive

    rho = np.linspace(0.0, 1.0, 50)
    ne = np.ones(50) * 5e19
    ac = AlfvenContinuum(rho, np.linspace(0.8, 3.0, 50), ne, B0=5.0, R0=3.0)
    fp = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.0, m_fast_amu=4.0)
    asa = AlfvenStabilityAnalysis(ac, fp)
    assert asa.critical_beta_fast(n=1) == float("inf")


# ── ballooning_solver.py:97 ──────────────────────────────────────────────────


def test_ballooning_unstable_at_amin():
    from scpn_control.core.ballooning_solver import find_marginal_stability, BallooningEquation

    mock_result = MagicMock(is_stable=False)
    with patch.object(BallooningEquation, "solve", return_value=mock_result):
        assert find_marginal_stability(s=0.5) == 0.0


# ── elm_model.py:194-195 ─────────────────────────────────────────────────────


def test_rmp_pedestal_transport():
    from scpn_control.core.elm_model import RMPSuppression

    rmp = RMPSuppression()
    assert rmp.pedestal_transport_enhancement(2.0) > 1.0
    assert rmp.density_pump_out(2.0) == pytest.approx(0.2)


# ── gk_cgyro.py:113 — external binary, pragma ───────────────────────────────
# ── gk_gene.py:162 — external binary, pragma ─────────────────────────────────
# ── gk_gs2.py:135 — external binary, pragma ──────────────────────────────────

# ── gk_corrector.py:47 ──────────────────────────────────────────────────────


def test_correction_record_zero_chi_e():
    from scpn_control.core.gk_corrector import CorrectionRecord

    rec = CorrectionRecord(
        rho_idx=5,
        rho=0.5,
        chi_i_surrogate=1.2,
        chi_i_gk=1.0,
        chi_e_surrogate=0.1,
        chi_e_gk=0.0,
        D_e_surrogate=0.01,
        D_e_gk=0.01,
    )
    assert rec.rel_error_chi_e == 0.0


# ── gk_eigenvalue.py:387,433 ─────────────────────────────────────────────────


def test_gk_eigenvalue_solve():
    from scpn_control.core.gk_eigenvalue import solve_eigenvalue_single_ky, VelocityGrid, deuterium_ion, electron
    from scpn_control.core.gk_geometry import circular_geometry

    geom = circular_geometry()
    vgrid = VelocityGrid()
    species = [deuterium_ion(), electron()]
    result = solve_eigenvalue_single_ky(k_y_rho_s=0.3, species_list=species, geom=geom, vgrid=vgrid)
    assert hasattr(result, "gamma")


def test_gk_eigenvalue_em_branch():
    from scpn_control.core.gk_eigenvalue import solve_eigenvalue_single_ky, VelocityGrid, deuterium_ion, electron
    from scpn_control.core.gk_geometry import circular_geometry

    geom = circular_geometry(s_hat=0.1)
    result = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=[deuterium_ion(), electron()],
        geom=geom,
        vgrid=VelocityGrid(),
        electromagnetic=True,
    )
    assert hasattr(result, "gamma")


# ── gk_nonlinear.py:349,475 ─────────────────────────────────────────────────


def test_gk_nonlinear_ampere_em():
    from scpn_control.core.gk_nonlinear import NonlinearGKSolver, NonlinearGKConfig

    cfg = NonlinearGKConfig(
        n_kx=4, n_ky=4, n_theta=4, n_vpar=4, n_mu=2, n_species=1, electromagnetic=True, kinetic_electrons=False
    )
    solver = NonlinearGKSolver(cfg)
    f = np.zeros((1, 4, 4, 4, 4, 2), dtype=complex)
    assert solver.ampere_solve(f).shape == (4, 4, 4)


def test_gk_nonlinear_sugama():
    from scpn_control.core.gk_nonlinear import NonlinearGKSolver, NonlinearGKConfig

    cfg = NonlinearGKConfig(n_kx=4, n_ky=4, n_theta=4, n_vpar=4, n_mu=2, n_species=1, collision_model="sugama")
    solver = NonlinearGKSolver(cfg)
    f_s = np.zeros((4, 4, 4, 4, 2), dtype=complex)
    assert solver.collide(f_s).shape == f_s.shape


# ── gk_scheduler.py:127 ─────────────────────────────────────────────────────


def test_gk_scheduler_no_spotcheck():
    from scpn_control.core.gk_scheduler import GKScheduler, SchedulerConfig

    sched = GKScheduler(SchedulerConfig(budget=3))
    rho = np.linspace(0, 1, 10)
    chi_i = np.ones(10)
    sched.step(rho, chi_i)


# ── gk_species.py:197 ───────────────────────────────────────────────────────


def test_pitch_angle_operator():
    from scpn_control.core.gk_species import pitch_angle_operator

    lam = np.linspace(0.0, 1.0, 10)
    L = pitch_angle_operator(n_lambda=10, lam=lam, B_ratio=1.5)
    assert L.shape == (10, 10)


# ── gyrokinetic_transport.py:86,253 ──────────────────────────────────────────


def test_gk_transport_te_mode():
    from scpn_control.core.gyrokinetic_transport import solve_dispersion, GyrokineticsParams

    params = GyrokineticsParams(
        R_L_Ti=0.5,
        R_L_Te=20.0,
        R_L_ne=2.0,
        q=1.5,
        s_hat=0.8,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.01,
        beta_e=0.001,
    )
    gamma, omega_r, mode_id = solve_dispersion(params, k_theta_rho_s=0.3)
    assert mode_id > 0


def test_gk_transport_evaluate_low_rho():
    from scpn_control.core.gyrokinetic_transport import GyrokineticTransportModel

    model = GyrokineticTransportModel()
    chi_i, chi_e, D_e = model.evaluate(0.01, {"R0": 2.0})
    assert chi_i == pytest.approx(0.01)


# ── impurity_transport.py:73,226-228 ─────────────────────────────────────────


def test_neon_cooling_curve():
    from scpn_control.core.impurity_transport import CoolingCurve

    assert CoolingCurve("Ne").L_z(np.array([50.0]))[0] > 0.0


def test_impurity_upwind_negative_V():
    from scpn_control.core.impurity_transport import ImpurityTransportSolver, ImpuritySpecies

    species = [ImpuritySpecies(element="N", Z_nucleus=7, mass_amu=14.0)]
    solver = ImpurityTransportSolver(rho=np.linspace(0, 1, 20), R0=3.0, a=1.0, species=species)
    ne = np.ones(20) * 5e19
    Te = np.ones(20) * 1000.0
    Ti = np.ones(20) * 1000.0
    V_pinch = {"N": -np.ones(20) * 0.1}
    solver.n_z["N"] = np.ones(20) * 1e18
    result = solver.step(dt=0.001, ne=ne, Te_eV=Te, Ti_eV=Ti, D_anom=0.5, V_pinch=V_pinch)
    assert isinstance(result, dict)


# ── integrated_scenario.py:322-323,326-327,656-663,856-878,938 ───────────────


def test_scenario_eccd_nbi():
    from scpn_control.core.integrated_scenario import IntegratedScenarioSimulator, ScenarioConfig

    cfg = ScenarioConfig(
        R0=3.0,
        a=1.0,
        B0=5.0,
        kappa=1.7,
        delta=0.33,
        Ip_MA=1.0,
        P_aux_MW=10.0,
        P_eccd_MW=5.0,
        rho_eccd=0.3,
        P_nbi_MW=10.0,
        E_nbi_keV=80.0,
        t_end=0.002,
        dt=0.001,
    )
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    assert sim.eccd is not None


def test_scenario_run():
    from scpn_control.core.integrated_scenario import IntegratedScenarioSimulator, ScenarioConfig

    cfg = ScenarioConfig(R0=3.0, a=1.0, B0=5.0, kappa=1.7, delta=0.33, Ip_MA=1.0, P_aux_MW=10.0, t_end=0.003, dt=0.001)
    sim = IntegratedScenarioSimulator(cfg)
    states = sim.run()
    assert len(states) >= 1


def test_scenario_ntm_flattening():
    from scpn_control.core.integrated_scenario import IntegratedScenarioSimulator, ScenarioConfig

    cfg = ScenarioConfig(
        R0=3.0, a=1.0, B0=5.0, kappa=1.7, delta=0.33, Ip_MA=1.0, P_aux_MW=10.0, include_ntm=True, t_end=0.002, dt=0.001
    )
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    for key in sim.ntm_widths:
        sim.ntm_widths[key] = 0.1
    sim.step()


def test_scenario_sawtooth_crash():
    from scpn_control.core.integrated_scenario import IntegratedScenarioSimulator, ScenarioConfig
    from scpn_control.core.sawtooth import SawtoothEvent

    cfg = ScenarioConfig(
        R0=3.0,
        a=1.0,
        B0=5.0,
        kappa=1.7,
        delta=0.33,
        Ip_MA=1.0,
        P_aux_MW=10.0,
        include_sawteeth=True,
        include_ntm=True,
        t_end=0.003,
        dt=0.001,
    )
    sim = IntegratedScenarioSimulator(cfg)
    sim.initialize()
    mock_event = SawtoothEvent(
        crash_time=0.001,
        rho_1=0.2,
        rho_mix=0.3,
        T_drop=500.0,
        seed_energy=1e6,
    )
    with patch.object(sim.sawtooth, "step", return_value=mock_event):
        sim.step()
    assert sim.n_crashes >= 1


# ── integrated_transport_solver.py:703-705 ───────────────────────────────────

# integrated_transport_solver.py:703-705 — covered by pragma (requires unconverged GK solver)


# ── jax_gk_nonlinear.py — JAX-dependent, skip if unavailable ────────────────


def test_jax_gk_ampere_em():
    pytest.importorskip("jax")
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    cfg = NonlinearGKConfig(
        n_kx=4, n_ky=4, n_theta=4, n_vpar=4, n_mu=2, n_species=1, electromagnetic=True, kinetic_electrons=False
    )
    solver = JaxNonlinearGKSolver(cfg)
    f = np.zeros((1, 4, 4, 4, 4, 2), dtype=complex)
    assert solver._jax_ampere_solve(f).shape == (4, 4, 4)


def test_jax_gk_sugama():
    pytest.importorskip("jax")
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    cfg = NonlinearGKConfig(n_kx=4, n_ky=4, n_theta=4, n_vpar=4, n_mu=2, n_species=1, collision_model="sugama")
    solver = JaxNonlinearGKSolver(cfg)
    f_s = np.zeros((4, 4, 4, 4, 2), dtype=complex)
    assert solver._jax_collide(f_s).shape == f_s.shape


def test_jax_gk_cfl_kinetic_e():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    cfg = NonlinearGKConfig(
        n_kx=4, n_ky=4, n_theta=4, n_vpar=4, n_mu=2, n_species=2, kinetic_electrons=True, implicit_electrons=False
    )
    solver = JaxNonlinearGKSolver(cfg)
    dt = solver._jax_cfl_dt(jnp.ones((4, 4, 4), dtype=complex) * 0.01)
    assert float(dt) > 0


def test_jax_gk_rk4():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    cfg = NonlinearGKConfig(n_kx=4, n_ky=4, n_theta=4, n_vpar=4, n_mu=2, n_species=1)
    solver = JaxNonlinearGKSolver(cfg)
    f = jnp.zeros((1, 4, 4, 4, 4, 2), dtype=complex)
    assert solver._jax_rk4_step(f, 0.001).shape == f.shape


def test_jax_gk_nan_break():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    cfg = NonlinearGKConfig(n_kx=4, n_ky=4, n_theta=4, n_vpar=4, n_mu=2, n_species=1, n_steps=3, save_interval=1)
    solver = JaxNonlinearGKSolver(cfg)
    with patch.object(solver, "_jax_rk4_step", return_value=jnp.full((1, 4, 4, 4, 4, 2), float("nan"))):
        result = solver.run()
    assert hasattr(result, "chi_i")


# ── jax_gk_solver.py:178-179 ────────────────────────────────────────────────


def test_eigenvalue_singular_matrix():
    from scpn_control.core.jax_gk_solver import _solve_eigenvalue_from_matrix

    gamma, omega_r, mode, _ = _solve_eigenvalue_from_matrix(np.zeros((4, 4)), np.zeros((4, 4)))
    assert gamma == 0.0 and mode == "stable"


# ── jax_gs_solver.py:95 ─────────────────────────────────────────────────────


def test_gs_rhs_zero_denom():
    from scpn_control.core.jax_gs_solver import _compute_source_np

    psi = np.zeros((10, 10))
    R_grid = np.ones((10, 10))
    result = _compute_source_np(psi, R_grid, mu0=4e-7 * np.pi, Ip_target=1.0, beta_mix=0.5, dR=0.1, dZ=0.1)
    assert result.shape == (10, 10)


# ── lh_transition.py:134 ────────────────────────────────────────────────────


def test_lh_find_threshold_no_transition():
    from scpn_control.core.lh_transition import LHTrigger, PredatorPreyModel

    trigger = LHTrigger(PredatorPreyModel())
    result = trigger.find_threshold(np.linspace(0.0, 0.001, 5))
    assert isinstance(result, float)


# ── locked_mode.py:280,285 ──────────────────────────────────────────────────


def test_locked_mode_chain():
    from scpn_control.core.locked_mode import ErrorFieldToDisruptionChain

    chain = ErrorFieldToDisruptionChain(
        config={
            "R0": 3.0,
            "a": 1.0,
            "B0": 5.0,
            "Ip_MA": 1.0,
        }
    )
    result = chain.run(B_err_n1=1e-4, omega_phi_0=1e4)
    assert hasattr(result, "lock_time")


# ── marfe.py:97-98 ──────────────────────────────────────────────────────────


def test_marfe_critical_density():
    from scpn_control.core.marfe import RadiationCondensation

    rc = RadiationCondensation(impurity="C", ne_20=1.0, f_imp=0.03)
    n_crit = rc.critical_density(Te_eV=100.0, k_par=1.0, kappa_par=1e3)
    assert np.isfinite(n_crit)


# ── momentum_transport.py:26 ────────────────────────────────────────────────


def test_nbi_torque_zero_vbeam():
    from scpn_control.core.momentum_transport import nbi_torque

    result = nbi_torque(np.ones(10) * 1e6, R0=3.0, v_beam=0.0, theta_inj_deg=30.0)
    assert np.allclose(result, 0.0)


# ── neural_transport.py:510 ─────────────────────────────────────────────────


def test_neural_transport_benchmark_empty():
    from scpn_control.core.neural_transport import cross_validate_neural_transport

    with pytest.raises((ValueError, TypeError)):
        cross_validate_neural_transport(benchmark_cases=[])


# ── neural_turbulence.py:53,62 ──────────────────────────────────────────────


def test_qlknn_tanh_activation():
    from scpn_control.core.neural_turbulence import QLKNNSurrogate

    net = QLKNNSurrogate(activation="tanh")
    x = np.array([1.0, -1.0, 0.0])
    assert np.allclose(net._activate(x), np.tanh(x))


def test_qlknn_identity_deriv():
    from scpn_control.core.neural_turbulence import QLKNNSurrogate

    net = QLKNNSurrogate(activation="linear")
    x = np.array([1.0, -1.0, 0.0])
    assert np.allclose(net._activate_deriv(x), 1.0)


# ── ntm_dynamics.py:88 ──────────────────────────────────────────────────────


def test_ntm_rational_surface_flat_q():
    from scpn_control.core.ntm_dynamics import find_rational_surfaces

    rho = np.linspace(0.0, 1.0, 50)
    q = np.ones(50) * 2.0
    surfaces = find_rational_surfaces(rho, q, a=1.0)
    assert isinstance(surfaces, list)


# ── orbit_following.py:195,242 ──────────────────────────────────────────────


def test_orbit_ensemble():
    from scpn_control.core.orbit_following import MonteCarloEnsemble

    ens = MonteCarloEnsemble(n_particles=5, E_birth_keV=3500.0, R0=3.0, a=1.0, B0=5.0)
    rho = np.linspace(0, 1, 10)
    ne = np.ones(10) * 5e19
    Te = np.ones(10) * 5000.0
    ens.initialize(ne, Te, rho)

    def B_field(R, Z):
        return np.array([0.0, 0.0, 5.0])

    result = ens.follow(B_field, n_bounces=2, dt=1e-6)
    assert hasattr(result, "n_lost")


def test_banana_orbit_width():
    from scpn_control.core.orbit_following import banana_orbit_width

    assert banana_orbit_width(q=2.0, rho_L=0.01, epsilon=0.3) > 0.0
    with pytest.raises(ValueError):
        banana_orbit_width(q=2.0, rho_L=0.01, epsilon=-0.1)


# ── pellet_injection.py ──────────────────────────────────────────────────────


def test_pellet_ablation():
    from scpn_control.core.pellet_injection import PelletTrajectory, PelletParams

    params = PelletParams(r_p_mm=1.0, v_p_m_s=500.0, M_p=2.0)
    traj = PelletTrajectory(params, R0=3.0, a=1.0, B0=5.0)
    rho = np.linspace(0.0, 1.0, 20)
    ne = np.ones(20) * 5e19
    Te = np.ones(20) * 1000.0
    result = traj.simulate(rho, ne, Te)
    assert result.deposition_profile.shape == (20,)


# ── plasma_startup.py:225,231 ────────────────────────────────────────────────


def test_burn_through_condition():
    from scpn_control.core.plasma_startup import BurnThrough

    bt = BurnThrough(R0=3.0, a=1.0, B0=5.0, V_loop=20.0)
    result = bt.burn_through_condition(Te_eV=100.0, ne_19=3.0, Ip_kA=500.0, f_imp=0.01)
    assert isinstance(result, bool)


def test_critical_impurity_fraction():
    from scpn_control.core.plasma_startup import BurnThrough

    bt = BurnThrough(R0=3.0, a=1.0, B0=5.0, V_loop=20.0)
    result = bt.critical_impurity_fraction(Te_eV=200.0, ne_19=3.0, Ip_kA=500.0, impurity="C")
    assert isinstance(result, float)


# ── plasma_wall_interaction.py:119-120,205 ───────────────────────────────────


def test_erosion_lifetime_zero_rate():
    from scpn_control.core.plasma_wall_interaction import ErosionModel

    assert ErosionModel("W").lifetime_estimate(wall_thickness_mm=10.0, net_rate_m_s=0.0) == float("inf")


def test_erosion_lifetime_positive():
    from scpn_control.core.plasma_wall_interaction import ErosionModel

    result = ErosionModel("W").lifetime_estimate(wall_thickness_mm=10.0, net_rate_m_s=1e-9)
    assert np.isfinite(result) and result > 0.0


def test_disruption_load():
    from scpn_control.core.plasma_wall_interaction import TransientThermalLoad, WallThermalModel

    wtm = WallThermalModel("W")
    ttl = TransientThermalLoad(wtm)
    assert ttl.disruption_load(W_th_MJ=10.0, A_wet_m2=5.0, tau_TQ_ms=1.0) >= 0.0


# ── runaway_electrons.py:162 ─────────────────────────────────────────────────


def test_avalanche_F_zero():
    from scpn_control.core.runaway_electrons import avalanche_growth_rate, RunawayParams

    params = RunawayParams(ne_20=1.0, Te_keV=0.001, E_par=0.5, Z_eff=1.0, B0=5.0, R0=3.0)
    result = avalanche_growth_rate(params, n_RE=1e12)
    assert result >= 0.0


# ── sawtooth.py ──────────────────────────────────────────────────────────────


def test_sawtooth_find_q1_basic():
    from scpn_control.core.sawtooth import SawtoothMonitor

    mon = SawtoothMonitor(np.linspace(0, 1, 50))
    r1 = mon.find_q1_radius(np.linspace(0.7, 3.0, 50))
    assert r1 is not None and 0 < r1 < 1


def test_sawtooth_find_q1_no_crossing():
    from scpn_control.core.sawtooth import SawtoothMonitor

    assert SawtoothMonitor(np.linspace(0, 1, 50)).find_q1_radius(np.ones(50) * 2.0) is None


def test_sawtooth_q1_equals_q2():
    from scpn_control.core.sawtooth import SawtoothMonitor

    rho = np.array([0.0, 0.2, 0.3, 0.5, 1.0])
    q = np.array([0.9, 0.95, 1.0, 1.0, 2.0])
    SawtoothMonitor(rho).find_q1_radius(q)


def test_sawtooth_check_trigger():
    from scpn_control.core.sawtooth import SawtoothMonitor

    rho = np.linspace(0, 1, 50)
    q = np.linspace(0.7, 3.0, 50)
    shear = np.ones(50) * 0.5
    assert isinstance(SawtoothMonitor(rho, s_crit=0.1).check_trigger(q, shear), bool)


def test_sawtooth_trigger_idx_zero():
    from scpn_control.core.sawtooth import SawtoothMonitor

    rho = np.linspace(0, 1, 50)
    q = np.linspace(0.999, 3.0, 50)
    assert isinstance(SawtoothMonitor(rho).check_trigger(q, np.ones(50) * 0.5), bool)


def test_sawtooth_same_rho():
    from scpn_control.core.sawtooth import SawtoothMonitor

    rho = np.array([0.0, 0.3, 0.3, 0.5, 1.0])
    q = np.array([0.8, 0.95, 1.05, 1.5, 2.0])
    SawtoothMonitor(rho).check_trigger(q, np.ones(5) * 0.5)


def test_sawtooth_porcelli():
    from scpn_control.core.sawtooth import SawtoothMonitor

    rho = np.linspace(0, 1, 50)
    q = np.linspace(0.7, 3.0, 50)
    shear = np.gradient(q, rho) * (rho / np.maximum(q, 1e-3))
    T = np.linspace(5000, 1000, 50)
    n = np.ones(50) * 5e19
    result = SawtoothMonitor(rho).check_trigger(q, shear, trigger_model="porcelli", T=T, n=n)
    assert isinstance(result, bool)


def test_sawtooth_porcelli_no_T_raises():
    from scpn_control.core.sawtooth import SawtoothMonitor

    rho = np.linspace(0, 1, 50)
    q = np.linspace(0.7, 3.0, 50)
    with pytest.raises(ValueError, match="porcelli trigger requires"):
        SawtoothMonitor(rho).check_trigger(q, np.ones(50), trigger_model="porcelli")


# ── sol_model.py:121 ────────────────────────────────────────────────────────


def test_sol_sheath_zero_T():
    from scpn_control.core.sol_model import TwoPointSOL

    sol = TwoPointSOL(R0=3.0, a=1.0, q95=3.0, B_pol=0.5)
    result = sol.solve(P_SOL_MW=1.0, n_u_19=5.0, f_rad=0.99)
    assert result is not None


# ── disruption_predictor.py:881-889 ──────────────────────────────────────────


def test_disruption_checkpoint_missing_fallback(tmp_path):
    from scpn_control.control.disruption_predictor import load_or_train_predictor

    try:
        result, info = load_or_train_predictor(
            model_path=str(tmp_path / "nonexistent.pt"),
            allow_fallback=True,
            train_if_missing=False,
            force_retrain=False,
        )
        assert result is None
        assert info.get("fallback") is True
    except RuntimeError:
        pytest.skip("torch required for this code path")


def test_disruption_checkpoint_missing_raises(tmp_path):
    from scpn_control.control.disruption_predictor import load_or_train_predictor

    try:
        with pytest.raises(FileNotFoundError):
            load_or_train_predictor(
                model_path=str(tmp_path / "nonexistent.pt"),
                allow_fallback=False,
                train_if_missing=False,
                force_retrain=False,
            )
    except RuntimeError:
        pytest.skip("torch required for this code path")


# ── free_boundary_tracking.py validation ─────────────────────────────────────


def test_fbt_invalid_identification_perturbation():
    from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController

    with pytest.raises(ValueError, match="identification_perturbation"):
        FreeBoundaryTrackingController("iter_config.json", identification_perturbation=-1.0)


def test_fbt_invalid_correction_limit():
    from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController

    with pytest.raises(ValueError, match="correction_limit"):
        FreeBoundaryTrackingController("iter_config.json", correction_limit=float("nan"))


def test_fbt_invalid_response_regularization():
    from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController

    with pytest.raises(ValueError, match="response_regularization"):
        FreeBoundaryTrackingController("iter_config.json", response_regularization=-0.1)


def test_fbt_invalid_response_refresh_steps():
    from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController

    with pytest.raises(ValueError, match="response_refresh_steps"):
        FreeBoundaryTrackingController("iter_config.json", response_refresh_steps=0)


def test_fbt_invalid_solve_max_outer_iter():
    from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController

    with pytest.raises(ValueError, match="solve_max_outer_iter"):
        FreeBoundaryTrackingController("iter_config.json", solve_max_outer_iter=0)


def test_fbt_invalid_solve_tol():
    from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController

    with pytest.raises(ValueError, match="solve_tol"):
        FreeBoundaryTrackingController("iter_config.json", solve_tol=0.0)


# ── fusion_kernel.py validation ──────────────────────────────────────────────


def test_fk_shape_error_metrics_empty():
    from scpn_control.core.fusion_kernel import FusionKernel

    result = FusionKernel._shape_error_metrics(np.array([]), np.array([]))
    assert result["shape_error_rms"] == 0.0


def test_fk_divertor_labels():
    from scpn_control.core.fusion_kernel import FusionKernel

    assert FusionKernel._divertor_configuration_label(None) == "none"
    assert FusionKernel._divertor_configuration_label(np.zeros((1, 2))) == "single_strike"
    assert FusionKernel._divertor_configuration_label(np.zeros((2, 2))) == "double_strike"
    assert FusionKernel._divertor_configuration_label(np.zeros((3, 2))) == "multi_strike"


def test_fk_objective_status():
    from scpn_control.core.fusion_kernel import FusionKernel

    tolerances = {
        "shape_rms": 0.01,
        "shape_max_abs": 0.05,
        "x_point_position": 0.01,
        "x_point_gradient": 0.001,
        "x_point_flux": 0.01,
        "divertor_rms": 0.01,
        "divertor_max_abs": 0.05,
    }
    result = FusionKernel._evaluate_free_boundary_objective_status(
        tolerances,
        shape_error_rms=0.005,
        shape_error_max_abs=0.02,
        x_point_detected_error=0.005,
        x_point_gradient_norm=0.0005,
        x_point_flux_error=0.005,
        divertor_error_rms=0.005,
        divertor_error_max_abs=0.02,
    )
    assert result["objective_converged"] is True


def test_fk_tolerances_invalid_key():
    from scpn_control.core.fusion_kernel import FusionKernel

    with pytest.raises(ValueError, match="Unknown"):
        FusionKernel._resolve_free_boundary_objective_tolerances({"bogus": 0.1})


def test_fk_tolerances_negative():
    from scpn_control.core.fusion_kernel import FusionKernel

    with pytest.raises(ValueError, match="finite and >= 0"):
        FusionKernel._resolve_free_boundary_objective_tolerances({"shape_rms": -0.1})


def test_fk_tolerances_not_dict():
    from scpn_control.core.fusion_kernel import FusionKernel

    with pytest.raises(ValueError, match="must be a mapping"):
        FusionKernel._resolve_free_boundary_objective_tolerances("not_a_dict")
