#!/usr/bin/env python3
"""Tutorial 07: Phase 4 Advanced Control Modules.

Demonstrates the full suite of advanced tokamak control algorithms:
  1. Sliding-mode vertical stabilizer (super-twisting SMC)
  2. Gain-scheduled multi-regime controller with regime detection
  3. Resistive wall mode (RWM) feedback stabilization
  4. Mu-synthesis (structured uncertainty, D-K iteration)
  5. Fault-tolerant control with fault detection and isolation
  6. Plasma shape controller (isoflux boundary, Tikhonov pseudoinverse)
  7. Scenario scheduler (offline trajectory optimization)
  8. Controller comparison: H-infinity vs MPC vs PID

Usage:
    pip install scpn-control
    python examples/tutorial_07_advanced_controllers.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 1: Sliding-Mode Vertical Stabilizer                     ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.sliding_mode_vertical import (
    SuperTwistingSMC,
    VerticalStabilizer,
    estimate_convergence_time,
    lyapunov_certificate,
)

print("=" * 60)
print("SECTION 1: Super-Twisting Sliding-Mode Vertical Stabilizer")
print("=" * 60)

# Vertical instability in a tokamak with elongated cross-section:
#   m_eff * d²Z/dt² = (n-1) * μ₀ Ip² / (4π R₀) * Z + F_coil
# where n is the decay index. For n > 1 the equilibrium is unstable.
# The super-twisting algorithm provides finite-time convergence
# to s = 0 with continuous (chattering-free) control output.

ALPHA_SMC = 500.0
BETA_SMC = 1000.0
L_MAX = 50.0  # upper bound on perturbation derivative, N/s
M_EFF = 10.0  # kg (effective mass for vertical displacement mode)

smc = SuperTwistingSMC(alpha=ALPHA_SMC, beta=BETA_SMC, c=0.1, u_max=1e5)

vstab = VerticalStabilizer(
    n_index=1.5,
    Ip_MA=15.0,
    R0=6.2,
    m_eff=M_EFF,
    tau_wall=0.5,
    smc=smc,
)

cert = lyapunov_certificate(ALPHA_SMC, BETA_SMC, L_MAX)
t_conv = estimate_convergence_time(ALPHA_SMC, BETA_SMC, L_MAX, s0=0.05)

print(f"  SMC gains:     alpha={ALPHA_SMC}, beta={BETA_SMC}")
print(f"  Plant:         n_index=1.5, Ip=15 MA, R0=6.2 m, m_eff={M_EFF} kg")
print(f"  Lyapunov cert: {cert} (alpha > sqrt(2*L_max), beta > L_max)")
print(f"  T_conv upper:  {t_conv * 1e3:.2f} ms (to reach s=0 from s0=0.05)")
print()

# Simulate 5 cm initial vertical displacement
dt = 1e-4  # 10 kHz control rate
Z = 0.05  # 5 cm displacement
dZ_dt = 0.0
Z_ref = 0.0

Z_history = []
u_history = []
for step_i in range(5000):
    u = vstab.step(Z, Z_ref, dZ_dt, dt)
    u_history.append(u)

    # Plant: m_eff * a = F_coil
    accel = u / M_EFF
    dZ_dt += accel * dt
    Z += dZ_dt * dt
    Z_history.append(Z)

Z_arr = np.array(Z_history)
t_1mm = None
for i, z in enumerate(Z_arr):
    if abs(z) < 1e-3:
        t_1mm = i * dt
        break

print(f"  Initial Z:     {0.05 * 100:.1f} cm")
print(f"  Final Z:       {Z_arr[-1] * 1e3:.3f} mm (after {len(Z_arr) * dt * 1e3:.0f} ms)")
print(f"  Time to 1 mm:  {t_1mm * 1e3:.1f} ms" if t_1mm else "  Time to 1 mm:  not reached")
print(f"  Peak |u|:      {max(abs(u) for u in u_history):.0f} N")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 2: Gain-Scheduled Controller with Regime Detection      ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.gain_scheduled_controller import (
    GainScheduledController,
    OperatingRegime,
    RegimeController,
    RegimeDetector,
)

print("\n" + "=" * 60)
print("SECTION 2: Gain-Scheduled Multi-Regime Controller")
print("=" * 60)

# Each regime has tuned PID gains and reference setpoints.
# The controller interpolates gains during transitions (bumpless transfer).
# RegimeDetector classifies the plasma state using dIp/dt, tau_E, p_disrupt.

n_state = 3  # [Ip, beta_N, li]

controllers = {}
for regime, kp, ki, kd, xr in [
    (OperatingRegime.RAMP_UP, 2.0, 0.1, 0.5, [5.0, 1.5, 0.9]),
    (OperatingRegime.L_MODE_FLAT, 1.0, 0.2, 0.3, [15.0, 2.0, 0.85]),
    (OperatingRegime.H_MODE_FLAT, 1.5, 0.3, 0.2, [15.0, 2.8, 0.80]),
    (OperatingRegime.RAMP_DOWN, 1.0, 0.05, 0.5, [5.0, 1.0, 0.90]),
    (OperatingRegime.DISRUPTION_MITIGATION, 5.0, 0.0, 1.0, [0.0, 0.0, 1.0]),
]:
    controllers[regime] = RegimeController(
        regime=regime,
        Kp=np.full(n_state, kp),
        Ki=np.full(n_state, ki),
        Kd=np.full(n_state, kd),
        x_ref=np.array(xr),
        constraints={},
    )

detector = RegimeDetector()
gs_ctrl = GainScheduledController(controllers)

# Simulate discharge phases: ramp-up → L-mode → H-mode → ramp-down
phases = [
    ("Ramp-up", 0.5, 0.8, 0.0),  # (dIp/dt, tau_E, p_disrupt)
    ("L-mode", 0.0, 1.0, 0.0),
    ("LH trans", 0.0, 1.6, 0.0),
    ("H-mode", 0.0, 2.5, 0.0),
    ("Ramp-down", -0.3, 1.5, 0.0),
]

print("  State vector: [Ip, beta_N, li]")
print(f"  Bumpless transfer tau: {gs_ctrl.tau_switch:.1f} s")
print()
print(f"  {'Phase':>12s}  {'dIp/dt':>7s}  {'tau_E':>5s}  {'Detected regime':<28s}")
print(f"  {'-' * 12}  {'-' * 7}  {'-' * 5}  {'-' * 28}")

x = np.array([2.0, 1.0, 0.9])
t = 0.0
dt_gs = 0.1
for name, dip, tau_e, p_dis in phases:
    # Feed detector several times to overcome hysteresis
    for _ in range(6):
        state_vec = np.array([x[0], x[1], 0.0])
        dstate = np.array([dip, 0.0, 0.0])
        regime = detector.detect(state_vec, dstate, tau_e, p_dis)

    u = gs_ctrl.step(x, t, dt_gs, regime)
    t += dt_gs
    print(f"  {name:>12s}  {dip:7.2f}  {tau_e:5.1f}  {regime.name:<28s}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 3: RWM Feedback Stabilization                           ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.rwm_feedback import RWMFeedbackController, RWMPhysics

print("\n" + "=" * 60)
print("SECTION 3: Resistive Wall Mode Feedback")
print("=" * 60)

# RWM growth rate:
#   gamma_wall = (1/tau_wall) * (beta_N - beta_N^nowall) / (beta_N^wall - beta_N)
# Feedback stabilization requires:
#   G_p * M_coil / (1 + gamma * tau_ctrl) > 1

BETA_N_NOWALL = 2.5
BETA_N_WALL = 5.0
TAU_WALL = 5e-3  # 5 ms wall time

rwm_ctrl = RWMFeedbackController(
    n_sensors=8,
    n_coils=6,
    G_p=5.0,
    G_d=0.01,
    tau_controller=1e-4,
    M_coil=1.0,
)

print("  Sensors: 8, Coils: 6")
print(f"  beta_N limits: no-wall={BETA_N_NOWALL}, with-wall={BETA_N_WALL}")
print(f"  tau_wall: {TAU_WALL * 1e3:.1f} ms")
print(f"  Feedback gains: G_p={rwm_ctrl.G_p}, G_d={rwm_ctrl.G_d}")
print()

beta_scan = np.linspace(2.0, 4.5, 11)
print(f"  {'beta_N':>6s}  {'gamma_OL':>10s}  {'gamma_CL':>10s}  {'Stabilized':>11s}")
print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 11}")
for bn in beta_scan:
    rwm = RWMPhysics(bn, BETA_N_NOWALL, BETA_N_WALL, TAU_WALL)
    gamma_ol = rwm.growth_rate()
    gamma_cl = rwm_ctrl.effective_growth_rate(rwm)
    stab = rwm_ctrl.is_stabilized(rwm)
    g_ol_str = f"{gamma_ol:10.1f}" if gamma_ol < 1e5 else "    stable"
    g_cl_str = f"{gamma_cl:10.1f}" if abs(gamma_cl) < 1e5 else "    stable"
    print(f"  {bn:6.2f}  {g_ol_str}  {g_cl_str}  {'YES' if stab else 'NO':>11s}")

# Stabilization margin at beta_N = 3.5
rwm_35 = RWMPhysics(3.5, BETA_N_NOWALL, BETA_N_WALL, TAU_WALL)
margin = -rwm_ctrl.effective_growth_rate(rwm_35) / rwm_35.growth_rate()
print(f"\n  Stabilization margin at beta_N=3.5: {margin:.2%}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 4: Mu-Synthesis (Structured Uncertainty)                ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.mu_synthesis import (
    StructuredUncertainty,
    UncertaintyBlock,
    compute_mu_upper_bound,
)

print("\n" + "=" * 60)
print("SECTION 4: Mu-Synthesis (Structured Singular Value)")
print("=" * 60)

# Structured uncertainty model for tokamak control:
#   Delta = diag(delta_Ip * I_2, delta_wall * I_2)
# where delta_Ip represents plasma current uncertainty (±20%)
# and delta_wall represents wall position uncertainty (±10%).
#
# Robust stability iff mu_Delta(M) < 1 for all frequencies.
# mu upper bound via D-scaling: min_D sigma_max(D M D^{-1}).

blocks = [
    UncertaintyBlock("plasma_current", size=2, bound=0.20, block_type="real_scalar"),
    UncertaintyBlock("wall_position", size=2, bound=0.10, block_type="real_scalar"),
]
unc = StructuredUncertainty(blocks)
delta_struct = unc.build_Delta_structure()

print("  Uncertainty blocks:")
for b in blocks:
    print(f"    {b.name}: size={b.size}, bound=+/-{b.bound:.0%}, type={b.block_type}")
print(f"  Total Delta size: {unc.total_size()}x{unc.total_size()}")
print()

# Frequency sweep: compute mu at each frequency
rng = np.random.default_rng(42)
n_freq = 8
freqs = np.logspace(-1, 3, n_freq)

print(f"  {'freq [rad/s]':>13s}  {'mu_upper':>9s}  {'Robust':>7s}")
print(f"  {'-' * 13}  {'-' * 9}  {'-' * 7}")

mu_peak = 0.0
for omega in freqs:
    # Construct frequency-dependent M matrix (transfer function at s = j*omega)
    # M = C (jωI - A)^{-1} B for a minimal example
    A_plant = np.array([[-10, omega], [-omega, -10]], dtype=complex)
    M = np.linalg.inv(
        1j * omega * np.eye(4)
        - np.block(
            [
                [A_plant, 0.1 * np.eye(2)],
                [0.2 * np.eye(2), A_plant * 0.5],
            ]
        )
    )
    mu_val = compute_mu_upper_bound(M, delta_struct)
    mu_peak = max(mu_peak, mu_val)
    robust = mu_val < 1.0
    print(f"  {omega:13.2f}  {mu_val:9.4f}  {'YES' if robust else 'NO':>7s}")

print(f"\n  Peak mu: {mu_peak:.4f}")
print(f"  Robust stability: {'GUARANTEED' if mu_peak < 1.0 else 'NOT GUARANTEED'}")
print(f"  Robustness margin: {1.0 / mu_peak:.2%}" if mu_peak > 0 else "")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 5: Fault-Tolerant Control (FDI)                         ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.fault_tolerant_control import (
    FaultInjector,
    FaultType,
    FDIMonitor,
)

print("\n" + "=" * 60)
print("SECTION 5: Fault Detection and Isolation")
print("=" * 60)

# Innovation-based FDI: monitors prediction residuals (innovations)
# and triggers when |nu_i| > threshold_sigma * sigma_i for n_alert
# consecutive samples.

N_SENSORS = 4
N_ACTUATORS = 3
FAULT_STEP = 10

fdi = FDIMonitor(N_SENSORS, N_ACTUATORS, threshold_sigma=2.0, n_alert=5)
injector = FaultInjector(
    fault_time=FAULT_STEP * 1e-3,
    component_index=2,
    fault_type=FaultType.SENSOR_DROPOUT,
)

print(f"  Sensors: {N_SENSORS}, Actuators: {N_ACTUATORS}")
print("  Detection threshold: 2-sigma, alert window: 5 samples")
print(f"  Fault injection: sensor 2 dropout at step {FAULT_STEP}")
print()

rng = np.random.default_rng(99)
detected_at = None
detected_type = None

# Use a large-amplitude signal so dropout (→0) creates a clear innovation spike
for step_i in range(60):
    t = step_i * 1e-3

    # Nominal sensor reading: large constant offset + small noise
    y_true = 5.0 * np.ones(N_SENSORS)
    y_pred = y_true.copy()

    y_meas = injector.inject(t, y_true + rng.normal(0, 0.01, N_SENSORS))

    faults = fdi.update(y_meas, y_pred, t)

    if faults and detected_at is None:
        detected_at = step_i
        detected_type = faults[0].fault_type

    if step_i in (0, FAULT_STEP - 1, FAULT_STEP, FAULT_STEP + 5, 59):
        n_faults = len(fdi.detected_faults)
        print(f"  Step {step_i:3d}: y_meas[2]={y_meas[2]:+7.4f}, faults_total={n_faults}")

if detected_at is not None:
    latency = detected_at - FAULT_STEP
    print(f"\n  Detection latency: {latency} steps ({latency * 1.0:.0f} ms)")
    print(f"  Fault classified:  {detected_type.name}")
else:
    print("\n  Fault not detected within 60 steps")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 6: Shape Controller (Isoflux Boundary)                  ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.shape_controller import (
    CoilSet,
    PlasmaShapeController,
    ShapeTarget,
)

print("\n" + "=" * 60)
print("SECTION 6: Plasma Shape Controller")
print("=" * 60)

# The shape controller minimizes the isoflux error and gap targets
# using a Tikhonov-regularized pseudoinverse of the shape Jacobian:
#   delta_I = -(J^T W J + lambda I)^{-1} J^T W * e_shape

isoflux_pts = [(6.2 + 2.0 * np.cos(th), 1.7 * np.sin(th)) for th in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
gap_pts = [
    (8.2, 0.0, -1.0, 0.0),  # outer midplane
    (4.2, 0.0, 1.0, 0.0),  # inner midplane
]
gap_targets = [0.08, 0.08]  # 8 cm gaps

target = ShapeTarget(
    isoflux_points=isoflux_pts,
    gap_points=gap_pts,
    gap_targets=gap_targets,
)

coils = CoilSet(n_coils=6)
shape_ctrl = PlasmaShapeController(target=target, coil_set=coils, kernel=None)

print(f"  Isoflux points: {len(isoflux_pts)}")
print(
    f"  Gap targets:    {len(gap_targets)} (outer={gap_targets[0] * 100:.0f} cm, inner={gap_targets[1] * 100:.0f} cm)"
)
print(f"  PF coils:       {coils.n_coils}")
print(f"  Jacobian shape: {shape_ctrl.jacobian.J.shape}")
print(f"  Regularization: lambda={shape_ctrl.lambda_reg}")
print()

psi_mock = np.ones((33, 33)) * 0.01
I_coils = np.zeros(coils.n_coils)

print(f"  {'Iter':>4s}  {'||delta_I|| [A]':>15s}  {'Isoflux err':>12s}  {'Min gap [cm]':>12s}")
print(f"  {'-' * 4}  {'-' * 15}  {'-' * 12}  {'-' * 12}")

for it in range(10):
    delta_I = shape_ctrl.step(psi_mock, I_coils)
    I_coils += delta_I
    perf = shape_ctrl.evaluate_performance(psi_mock)
    norm_dI = float(np.linalg.norm(delta_I))
    print(f"  {it + 1:4d}  {norm_dI:15.4f}  {perf.isoflux_error:12.6f}  {perf.min_gap * 100:12.4f}")

print("\n  Final coil currents [A]:")
for i, ic in enumerate(I_coils):
    print(f"    PF{i + 1}: {ic:+10.2f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 7: Scenario Scheduler                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.scenario_scheduler import (
    ScenarioOptimizer,
    iter_15ma_baseline,
)

print("\n" + "=" * 60)
print("SECTION 7: Scenario Scheduler & Trajectory Optimization")
print("=" * 60)

# Display the ITER 15 MA baseline scenario waveforms
baseline = iter_15ma_baseline()
errors = baseline.validate()

print("  ITER 15 MA baseline scenario:")
print(f"  Duration: {baseline.duration():.0f} s")
print(f"  Waveforms: {', '.join(baseline.waveforms.keys())}")
print(f"  Validation: {errors if errors else 'PASS'}")
print()

sample_times = [0, 10, 30, 60, 200, 400, 480]
print(f"  {'t [s]':>6s}  {'Ip [MA]':>8s}  {'P_NBI [MW]':>10s}  {'P_ECCD [MW]':>11s}")
print(f"  {'-' * 6}  {'-' * 8}  {'-' * 10}  {'-' * 11}")
for t in sample_times:
    vals = baseline.evaluate(float(t))
    print(f"  {t:6d}  {vals['Ip']:8.1f}  {vals['P_NBI']:10.1f}  {vals['P_ECCD']:11.1f}")


# Offline trajectory optimization with a simple integrator plant
def simple_plant(x, u, dt_p):
    """x' = A x + B u, Euler step. x=[pos, vel], u=[P_aux, Ip_ref]."""
    A_p = np.array([[0, 1], [-1, -0.5]])
    B_p = np.array([[0, 0.1], [0.5, 0]])
    return x + (A_p @ x + B_p @ u) * dt_p


target_state = np.array([1.0, 0.0])

print("\n  Offline trajectory optimization:")
print("    Plant: 2nd-order damped oscillator")
print(f"    Target state: {target_state}")
print("    Horizon: 10 s, dt=0.5 s")

optimizer = ScenarioOptimizer(simple_plant, target_state, T_total=10.0, dt=0.5)
t0 = time.perf_counter()
opt_schedule = optimizer.optimize(n_iter=50)
elapsed = time.perf_counter() - t0

print(f"    Optimization time: {elapsed * 1e3:.0f} ms (50 Nelder-Mead iterations)")
opt_vals = opt_schedule.evaluate(5.0)
print(f"    Optimized P_aux at t=5s: {opt_vals['P_aux']:.3f}")
print(f"    Optimized Ip at t=5s:    {opt_vals['Ip']:.3f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 8: Controller Comparison                                ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.fusion_sota_mpc import (
    ModelPredictiveController,
    NeuralSurrogate,
)
from scpn_control.control.h_infinity_controller import get_radial_robust_controller

print("\n" + "=" * 60)
print("SECTION 8: Controller Comparison (H-inf vs MPC vs PID)")
print("=" * 60)

# Compare per-step computational latency.
# MPC must solve an optimization at each step; H-inf and PID are O(n) per step.

N_STEPS = 500
DT_CMP = 1e-3

# --- H-infinity controller (synthesis + step latency) ---
t0_synth = time.perf_counter()
hinf = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
synth_time = time.perf_counter() - t0_synth

# Time N_STEPS calls to step() with dummy measurement
hinf.reset()
t0_hinf = time.perf_counter()
for k in range(N_STEPS):
    measurement = 0.01 * np.exp(-k * DT_CMP)
    hinf.step(measurement, DT_CMP)
lat_hinf = (time.perf_counter() - t0_hinf) / N_STEPS

rx, ry = hinf.riccati_residual_norms()

print("  H-infinity synthesis:")
print(f"    Synthesis time:     {synth_time * 1e3:.1f} ms")
print(f"    State dim:          {hinf.n}")
print(f"    gamma:              {hinf.gamma:.4f}")
print(f"    Gain margin:        {hinf.gain_margin_db:.1f} dB")
print(f"    Robust feasible:    {hinf.robust_feasible}")
print(f"    Riccati residuals:  ||R_X||={rx:.2e}, ||R_Y||={ry:.2e}")
print(f"    Step latency:       {lat_hinf * 1e6:.1f} us/step")
print()

# --- MPC (trajectory planner latency) ---
surrogate = NeuralSurrogate(2, 4, verbose=False)
surrogate.B = np.random.default_rng(0).standard_normal((4, 2)) * 0.01
mpc = ModelPredictiveController(
    surrogate,
    target_state=np.zeros(4),
    prediction_horizon=5,
    learning_rate=0.3,
    iterations=10,
    action_limit=5.0,
)

x_mpc = np.array([0.1, 0.0, -0.05, 0.02])
t0_mpc = time.perf_counter()
for k in range(N_STEPS):
    u_mpc = mpc.plan_trajectory(x_mpc)
    x_mpc *= 0.999
lat_mpc = (time.perf_counter() - t0_mpc) / N_STEPS

print("  MPC (gradient trajectory optimizer):")
print(f"    State dim:          {surrogate.B.shape[0]}")
print(f"    Coils:              {surrogate.B.shape[1]}")
print(f"    Horizon:            {mpc.horizon} steps")
print(f"    Inner iterations:   {mpc.iterations}")
print(f"    Step latency:       {lat_mpc * 1e6:.1f} us/step")
print()

# --- PID (baseline) ---
t0_pid = time.perf_counter()
x_pid = 0.1
integral_pid = 0.0
prev_pid = x_pid
for k in range(N_STEPS):
    integral_pid += x_pid * DT_CMP
    deriv = (x_pid - prev_pid) / DT_CMP
    u_pid = 1.0 * x_pid + 0.1 * integral_pid + 0.05 * deriv
    prev_pid = x_pid
    x_pid *= 0.999
lat_pid = (time.perf_counter() - t0_pid) / N_STEPS

print("  PID (baseline):")
print(f"    Step latency:       {lat_pid * 1e6:.1f} us/step")
print()

# Summary table
print(f"  {'Controller':>12s}  {'Latency [us/step]':>18s}  {'Robustness':>16s}")
print(f"  {'-' * 12}  {'-' * 18}  {'-' * 16}")
print(f"  {'H-infinity':>12s}  {lat_hinf * 1e6:18.1f}  {hinf.gain_margin_db:.1f} dB margin")
print(f"  {'MPC':>12s}  {lat_mpc * 1e6:18.1f}  receding horizon")
print(f"  {'PID':>12s}  {lat_pid * 1e6:18.1f}  no guarantees")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Summary                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("ADVANCED CONTROLLERS SUMMARY")
print("=" * 60)
print("  SuperTwistingSMC       -- finite-time chattering-free vertical control")
print("  GainScheduledController -- bumpless multi-regime PID with hysteresis")
print("  RWMFeedbackController  -- sensor-coil feedback for resistive wall modes")
print("  compute_mu_upper_bound -- structured singular value via D-scaling")
print("  FDIMonitor             -- innovation-based fault detection & isolation")
print("  PlasmaShapeController  -- Tikhonov isoflux boundary control")
print("  ScenarioOptimizer      -- offline trajectory design (Nelder-Mead)")
print("  HInfinityController    -- Riccati DARE robust synthesis")
print("  ModelPredictiveController -- gradient-based trajectory optimization")
