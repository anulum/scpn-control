#!/usr/bin/env python3
"""Tutorial 05: Adaptive Phase Dynamics & Lyapunov Stability.

Demonstrates the Paper 27 Kuramoto-Sakaguchi engine with online adaptation:
  1. Single-layer phase synchronization (order parameter R)
  2. 16-layer UPDE with Paper 27 coupling matrix
  3. Lyapunov stability guard (safety veto)
  4. RealtimeMonitor with trajectory recording
  5. Adaptive Knm engine (plasma-driven coupling updates)
  6. Full closed-loop: UPDE + adaptive Knm + guard

Usage:
    pip install scpn-control
    python examples/tutorial_05_adaptive_phase_dynamics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scpn_control.phase.knm import OMEGA_N_16, build_knm_paper27
from scpn_control.phase.kuramoto import (
    kuramoto_sakaguchi_step,
)
from scpn_control.phase.upde import UPDESystem

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 1: Single-Layer Kuramoto Synchronization                ║
# ╚══════════════════════════════════════════════════════════════════╝

print("═" * 60)
print("SECTION 1: Kuramoto Phase Synchronization")
print("═" * 60)

# N oscillators with random natural frequencies and coupling K.
# The Kuramoto model:
#   dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
# Synchronization emerges when K > K_critical ≈ 2/(π g(0))
# where g(0) is the frequency distribution peak.

N = 200
rng = np.random.default_rng(42)
theta = rng.uniform(-np.pi, np.pi, N)
omega = rng.normal(0, 1.0, N)  # Gaussian frequencies

print(f"  Oscillators: {N}")
print("  Coupling K:  3.0 (supercritical, mean-field)")
print("  Frequency:   N(0, 1) rad/s")
print()

R_history = []
for step in range(500):
    result = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=3.0, psi_driver=0.0)
    theta = result["theta1"]
    R = result["R"]
    R_history.append(R)

    if step % 100 == 0:
        print(f"  Step {step:4d}: R = {R:.4f}")

print(f"  Step  500: R = {R_history[-1]:.4f}")
print(f"  Sync achieved: R > 0.9 → {'YES' if R_history[-1] > 0.9 else 'NO'}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 2: 16-Layer UPDE (Paper 27 Knm)                        ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 2: 16-Layer UPDE System")
print("═" * 60)

# The UPDE (Unified Phase Dynamics Equation) couples L layers:
#   dθ_{m,i}/dt = ω_{m,i}
#     + K_{mm} R_m sin(ψ_m - θ_{m,i})         (intra-layer)
#     + Σ_{n≠m} K_{nm} R_n sin(ψ_n - θ_{m,i}) (inter-layer)
#     + ζ_m sin(Ψ - θ_{m,i})                   (global driver)

spec = build_knm_paper27(L=16, K_base=0.45, K_alpha=0.3, zeta_uniform=0.5)

print(f"  Layers:     {spec.L}")
print("  K_base:     0.45 (exponential decay α=0.3)")
print("  ζ (driver): 0.5 (global field coupling)")
print(f"  K[1,2]:     {spec.K[1, 2]:.3f} (calibration anchor)")
print(f"  K[2,3]:     {spec.K[2, 3]:.3f}")

# Natural frequencies from Paper 27 Table 1
print("\n  Natural frequencies (OMEGA_N_16):")
for i in range(0, 16, 4):
    freqs = ", ".join(f"ω_{i + j + 1}={OMEGA_N_16[i + j]:.3f}" for j in range(4))
    print(f"    {freqs}")

# Initialize UPDE
N_per = 50  # oscillators per layer
upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")

theta_layers = [rng.uniform(-np.pi, np.pi, N_per) for _ in range(16)]
omega_layers = [OMEGA_N_16[m] + rng.normal(0, 0.1, N_per) for m in range(16)]

# Run 200 steps
print(f"\n  Running {200} UPDE steps ({N_per} osc/layer, 16 layers)...")
for step in range(200):
    result = upde.step(
        theta_layers,
        omega_layers,
        psi_driver=0.0,
        pac_gamma=0.0,
    )
    theta_layers = result["theta1"]

    if step % 50 == 49:
        R_mean = result["R_layer"].mean()
        V_global = result["V_global"]
        print(f"    Step {step + 1:4d}: R_mean={R_mean:.3f}, V_global={V_global:.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 3: Lyapunov Stability Guard                             ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 3: Lyapunov Stability Guard")
print("═" * 60)

from scpn_control.phase.lyapunov_guard import LyapunovGuard

# The guard monitors stability via a sliding-window Lyapunov exponent:
#   V(t) = (1/N) sum(1 - cos(theta_i - Psi))
#   lambda_exp = slope of log(V) over the window
#   guard refuses when lambda > threshold for max_violations consecutive ticks

guard = LyapunovGuard(window=50, dt=1e-3, lambda_threshold=0.0, max_violations=3)

print(f"  Window:         {50} samples")
print("  lambda thresh:  0.0 (any growth triggers violation)")
print("  Max violations: 3 consecutive before refusal")

# Feed the current UPDE state into the guard
all_theta = np.concatenate([t.ravel() for t in theta_layers])
psi_global = result["Psi_global"]
verdict = guard.check(all_theta, psi_global)

print("\n  Guard check:")
print(f"    V:            {verdict.v:.6f}")
print(f"    lambda_exp:   {verdict.lambda_exp:.6f}")
print(f"    Approved:     {verdict.approved}")
print(f"    Score:        {verdict.score:.4f}")

# Simulate a destabilizing perturbation
print("\n  Simulating guard response to desync event...")
guard_test = LyapunovGuard(window=10, dt=1e-3, lambda_threshold=0.0, max_violations=3)
for i in range(8):
    # Progressively desynchronizing phases
    bad_theta = rng.uniform(-np.pi, np.pi, 100) * (1.0 + 0.1 * i)
    v = guard_test.check(bad_theta, 0.0)
    print(f"    Tick {i + 1}: approved={v.approved}, lambda={v.lambda_exp:.4f}, violations={v.consecutive_violations}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 4: RealtimeMonitor + Trajectory Recording               ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 4: RealtimeMonitor Dashboard Interface")
print("═" * 60)

from scpn_control.phase.realtime_monitor import RealtimeMonitor

# RealtimeMonitor wraps UPDE + guard into a single tick() call
# suitable for dashboard integration and WebSocket streaming.

monitor = RealtimeMonitor.from_paper27(psi_driver=0.0)

print(f"  Layers: {monitor.upde.spec.L}")
print(f"  Oscillators/layer: {len(monitor.theta_layers[0])}")
print()

# Run 300 ticks with trajectory recording
R_trace = []
V_trace = []
lambda_trace = []
t0 = time.perf_counter()

for tick in range(300):
    snap = monitor.tick()
    R_trace.append(snap["R_global"])
    V_trace.append(snap["V_global"])
    lambda_trace.append(snap["lambda_exp"])

elapsed = time.perf_counter() - t0

print(f"  300 ticks in {elapsed * 1e3:.1f} ms ({elapsed / 300 * 1e6:.0f} µs/tick)")
print(f"  Final R_global: {R_trace[-1]:.4f}")
print(f"  Final V_global: {V_trace[-1]:.6f}")
print(f"  Final λ_exp:    {lambda_trace[-1]:.6f}")
print(f"  Guard status:   {'APPROVED' if snap['guard_approved'] else 'REJECTED'}")

# Convergence summary
print("\n  R convergence:")
print(f"    Tick   1: R = {R_trace[0]:.4f}")
print(f"    Tick  50: R = {R_trace[49]:.4f}")
print(f"    Tick 100: R = {R_trace[99]:.4f}")
print(f"    Tick 200: R = {R_trace[199]:.4f}")
print(f"    Tick 300: R = {R_trace[-1]:.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 5: Adaptive Knm Engine                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 5: Adaptive Coupling (Plasma-Driven)")
print("═" * 60)

from scpn_control.phase.adaptive_knm import (
    AdaptiveKnmConfig,
    AdaptiveKnmEngine,
    DiagnosticSnapshot,
)

# The adaptive engine modifies K_nm in real-time based on
# tokamak diagnostic measurements:
#   1. Beta channel:     K *= (1 + β_scale · β_N)
#   2. MHD risk channel: K[risk_pairs] += risk_gain · disruption_risk
#   3. Coherence PI:     K[m,m] += PI(R_target - R_m)
#   4. Rate limiter:     |ΔK_ij| ≤ max_delta_per_tick
#   5. Guard veto:       revert to last-good if rejected

config = AdaptiveKnmConfig(
    beta_scale=0.3,
    risk_gain=0.4,
    coherence_Kp=0.15,
    coherence_Ki=0.02,
    coherence_R_target=0.6,
    max_delta_per_tick=0.02,
)
engine = AdaptiveKnmEngine(baseline_spec=spec, config=config)

print(f"  Beta scale:       {config.beta_scale}")
print(f"  Risk gain:        {config.risk_gain}")
print(f"  Coherence target: R = {config.coherence_R_target}")
print(f"  Rate limit:       {config.max_delta_per_tick}/tick")

# Simulate different plasma conditions
scenarios = [
    ("Quiet H-mode", 1.8, 3.5, 0.05, 0.01),
    ("High beta", 2.5, 3.0, 0.1, 0.02),
    ("MHD active", 1.5, 2.8, 0.6, 0.15),
    ("Pre-disruption", 0.8, 2.1, 0.85, 0.4),
]

print(f"\n  {'Scenario':>16s}  {'β_N':>4s}  {'q95':>4s}  {'Risk':>5s}  {'K_mean':>7s}  {'ΔK_max':>7s}")
print(f"  {'─' * 16}  {'─' * 4}  {'─' * 4}  {'─' * 5}  {'─' * 7}  {'─' * 7}")

K_baseline_mean = float(spec.K.mean())
for name, beta_n, q95, risk, mirnov in scenarios:
    diag = DiagnosticSnapshot(
        R_layer=np.array([0.6] * 16),
        V_layer=np.array([0.1] * 16),
        lambda_exp=-0.1,
        beta_n=beta_n,
        q95=q95,
        disruption_risk=risk,
        mirnov_rms=mirnov,
        guard_approved=True,
    )
    K_adapted = engine.update(diag)
    K_mean = float(K_adapted.mean())
    K_delta = float(np.max(np.abs(K_adapted - spec.K)))
    print(f"  {name:>16s}  {beta_n:4.1f}  {q95:4.1f}  {risk:5.2f}  {K_mean:7.4f}  {K_delta:7.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 6: Full Adaptive Closed Loop                            ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 6: Adaptive Closed-Loop Evolution")
print("═" * 60)

# Full pipeline: UPDE → diagnostics → adaptive Knm → guard check
# This is the complete real-time monitoring loop.

adaptive_engine = AdaptiveKnmEngine(baseline_spec=spec, config=config)
monitor_adaptive = RealtimeMonitor.from_paper27(psi_driver=0.0)
monitor_adaptive.adaptive_engine = adaptive_engine

# Simulate evolving plasma: beta_N ramps 1.5 -> 2.5 over 200 ticks
print("  Evolving plasma: beta_N ramps 1.5 -> 2.5 over 200 ticks")
print()
print(f"  {'Tick':>5s}  {'R_global':>8s}  {'V_global':>8s}  {'lam_exp':>8s}  {'b_N':>4s}  {'Guard':>6s}")
print(f"  {'---':>5s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'----':>4s}  {'------':>6s}")

for tick in range(200):
    beta_n = 1.5 + 1.0 * tick / 200
    risk = 0.05 + 0.3 * (tick / 200) ** 2

    snap = monitor_adaptive.tick(
        beta_n=beta_n,
        q95=3.5 - 0.5 * tick / 200,
        disruption_risk=risk,
        mirnov_rms=0.01 + 0.05 * risk,
    )

    if tick % 40 == 0 or tick == 199:
        guard_str = "OK" if snap["guard_approved"] else "REJECT"
        print(
            f"  {tick:5d}  {snap['R_global']:8.4f}  {snap['V_global']:8.5f}  "
            f"{snap['lambda_exp']:8.5f}  {beta_n:4.1f}  {guard_str:>6s}"
        )

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Summary                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("PHASE DYNAMICS SUMMARY")
print("═" * 60)
print("  kuramoto_sakaguchi_step — single-layer Euler step with driver")
print("  order_parameter         — Kuramoto R, ψ_r = <exp(iθ)>")
print("  UPDESystem              — 16-layer coupled phase evolution")
print("  LyapunovGuard           — safety veto (V, λ, R thresholds)")
print("  RealtimeMonitor         — dashboard-ready tick() interface")
print("  AdaptiveKnmEngine       — plasma-driven coupling updates")
print("  DiagnosticSnapshot      — tokamak diagnostic measurements")
