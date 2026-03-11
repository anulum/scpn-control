#!/usr/bin/env python3
"""Tutorial 01: Full Closed-Loop Tokamak Control Pipeline.

Demonstrates the complete scpn-control stack:
  1. Machine configuration (ITER, SPARC, DIII-D presets)
  2. Grad-Shafranov equilibrium solve
  3. Transport computation (analytic critical-gradient model)
  4. SPN → SNN compilation with contract checking
  5. H-infinity robust controller synthesis
  6. Digital twin closed-loop benchmark

Usage:
    pip install scpn-control
    python examples/tutorial_01_closed_loop_control.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 1: Machine Configuration                                ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.core.tokamak_config import TokamakConfig

# Named presets encode published machine parameters.
# TokamakConfig is a frozen dataclass — all fields are read-only.

iter_cfg = TokamakConfig.iter()
sparc_cfg = TokamakConfig.sparc()
diiid_cfg = TokamakConfig.diiid()

print("═" * 60)
print("SECTION 1: Machine Configurations")
print("═" * 60)
for cfg in [iter_cfg, sparc_cfg, diiid_cfg]:
    print(
        f"  {cfg.name:8s}  R0={cfg.R0:.2f}m  a={cfg.a:.2f}m  "
        f"B0={cfg.B0:.1f}T  Ip={cfg.Ip:.1f}MA  "
        f"A={cfg.aspect_ratio:.2f}"
    )

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 2: Equilibrium Solve (Grad-Shafranov)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.core.fusion_kernel import FusionKernel

print("\n" + "═" * 60)
print("SECTION 2: Grad-Shafranov Equilibrium")
print("═" * 60)

# FusionKernel solves the Grad-Shafranov equation:
#   Δ*ψ = -μ₀ R J_φ
# where Δ* is the toroidal elliptic operator.
# Picard iteration with under-relaxation on a (NR x NZ) grid.

kernel = FusionKernel.__new__(FusionKernel)
kernel.NR = 33
kernel.NZ = 33
kernel.R_min, kernel.R_max = 0.1, 2.0
kernel.Z_min, kernel.Z_max = -1.5, 1.5

R_1d = np.linspace(kernel.R_min, kernel.R_max, kernel.NR)
Z_1d = np.linspace(kernel.Z_min, kernel.Z_max, kernel.NZ)
kernel.R_grid, kernel.Z_grid = np.meshgrid(R_1d, Z_1d)
kernel.dR = R_1d[1] - R_1d[0]
kernel.dZ = Z_1d[1] - Z_1d[0]

# Simple analytic current profile J_φ = J0 * (1 - ρ²)
R0, a = 1.05, 0.7
rho = np.sqrt(
    ((kernel.R_grid - R0) / a) ** 2 + (kernel.Z_grid / (1.5 * a)) ** 2
)
kernel.J_phi = np.where(rho < 1.0, 1e6 * (1.0 - rho**2), 0.0)
kernel.Psi = np.zeros_like(kernel.R_grid)

# Manual Picard iteration (GS operator)
mu0 = 4e-7 * np.pi
for i in range(30):
    rhs = -mu0 * kernel.R_grid * kernel.J_phi
    psi_new = np.copy(kernel.Psi)
    for iz in range(1, kernel.NZ - 1):
        for ir in range(1, kernel.NR - 1):
            R = kernel.R_grid[iz, ir]
            psi_new[iz, ir] = 0.25 * (
                kernel.Psi[iz, ir + 1]
                + kernel.Psi[iz, ir - 1]
                + kernel.Psi[iz + 1, ir]
                + kernel.Psi[iz - 1, ir]
                - kernel.dR * kernel.dZ * rhs[iz, ir]
            )
    kernel.Psi = 0.7 * psi_new + 0.3 * kernel.Psi

psi_max = float(np.max(np.abs(kernel.Psi)))
print(f"  Grid: {kernel.NR}x{kernel.NZ}")
print(f"  Picard iterations: 30 (under-relaxation ω=0.7)")
print(f"  max|ψ| = {psi_max:.4e} Wb")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 3: Transport Computation                                ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.core.neural_transport import (
    TransportInputs,
    critical_gradient_model,
)

print("\n" + "═" * 60)
print("SECTION 3: Transport (Critical-Gradient Model)")
print("═" * 60)

# The critical-gradient model:
#   χ_i = χ_GB · max(0, R/L_Ti - crit_ITG)^stiffness
# where crit_ITG ≈ 4.0 (Dimits et al. 2000)

rho_pts = np.linspace(0.1, 0.9, 9)
print(f"  {'ρ':>5s}  {'R/L_Ti':>7s}  {'χ_e':>8s}  {'χ_i':>8s}  {'channel':>8s}")
print(f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
for rho_val in rho_pts:
    grad_ti = 2.0 + 6.0 * rho_val  # gradient increases toward edge
    inp = TransportInputs(
        rho=rho_val, te_kev=10.0 * (1 - rho_val**2),
        ti_kev=10.0 * (1 - rho_val**2), grad_ti=grad_ti,
        grad_te=grad_ti * 0.9, q=1.5 + 2.0 * rho_val**2,
    )
    fluxes = critical_gradient_model(inp)
    print(
        f"  {rho_val:5.2f}  {grad_ti:7.2f}  {fluxes.chi_e:8.3f}  "
        f"{fluxes.chi_i:8.3f}  {fluxes.channel:>8s}"
    )

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 4: SPN → SNN Compilation                                ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.structure import StochasticPetriNet

print("\n" + "═" * 60)
print("SECTION 4: Stochastic Petri Net → SNN Compilation")
print("═" * 60)

# Build a 4-place, 3-transition control net:
#   idle → [ignite] → rampup → [stabilize] → steady → [shutdown] → idle
net = StochasticPetriNet()
net.add_place("idle", initial_tokens=1.0)
net.add_place("rampup", initial_tokens=0.0)
net.add_place("steady", initial_tokens=0.0)
net.add_place("shutdown_complete", initial_tokens=0.0)

net.add_transition("ignite", threshold=0.5)
net.add_transition("stabilize", threshold=0.5)
net.add_transition("shutdown", threshold=0.5)

net.add_arc("idle", "ignite", weight=1.0)
net.add_arc("ignite", "rampup", weight=1.0)
net.add_arc("rampup", "stabilize", weight=1.0)
net.add_arc("stabilize", "steady", weight=1.0)
net.add_arc("steady", "shutdown", weight=1.0)
net.add_arc("shutdown", "shutdown_complete", weight=1.0)
net.compile()

compiler = FusionCompiler()
artifact = compiler.compile(net)

print(f"  Places:      {net.n_places}")
print(f"  Transitions: {net.n_transitions}")
print(f"  LIF neurons: {artifact.n_transitions}")
print(f"  W_in shape:  {artifact.W_in.shape}")
print(f"  W_out shape: {artifact.W_out.shape}")
print(f"  Thresholds:  {artifact.thresholds}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 5: H-infinity Robust Controller                         ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.h_infinity_controller import (
    get_radial_robust_controller,
)

print("\n" + "═" * 60)
print("SECTION 5: H-infinity Controller Synthesis")
print("═" * 60)

# H-inf controller minimizes the worst-case transfer from
# disturbance w to performance output z:
#   ||T_zw||_∞ < γ
# Solved via two algebraic Riccati equations (AREs).

ctrl = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
print(f"  γ (attenuation): {ctrl.gamma:.2f}")
print(f"  Gain margin:     {ctrl.gain_margin_db:.1f} dB")

# Closed-loop step response
errors = []
u_history = []
error = 0.1  # initial position error [m]
dt = 1e-3
for step_i in range(200):
    u = ctrl.step(error, dt=dt)
    error *= 0.98  # simple plant dynamics (decaying error)
    error -= 0.01 * u
    errors.append(error)
    u_history.append(u)

print(f"  Step response: error 0.100 → {errors[-1]:.4f} in 200 steps")
print(f"  Peak control: {max(abs(u) for u in u_history):.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 6: Digital Twin Closed-Loop Benchmark                   ║
# ╚══════════════════════════════════════════════════════════════════╝

from scpn_control.control.tokamak_digital_twin import run_digital_twin

print("\n" + "═" * 60)
print("SECTION 6: Digital Twin Closed-Loop Run")
print("═" * 60)

t0 = time.perf_counter()
result = run_digital_twin(
    time_steps=100,
    save_plot=False,
    verbose=False,
    seed=42,
    sensor_dropout_prob=0.0,
    chaos_monkey=False,
)
elapsed = time.perf_counter() - t0

print(f"  Time steps:     100")
print(f"  Final avg temp: {result['final_avg_temp']:.2f}")
print(f"  MHD islands:    {result['final_islands_px']} px")
print(f"  Final reward:   {result['final_reward']:.2f}")
print(f"  Wall time:      {elapsed:.3f}s")

# Chaos monkey mode (fault injection)
result_chaos = run_digital_twin(
    time_steps=100,
    save_plot=False,
    verbose=False,
    seed=42,
    sensor_dropout_prob=0.1,
    sensor_noise_std=0.05,
    chaos_monkey=True,
)
print(f"\n  With chaos monkey (10% dropout, 5% noise):")
print(f"  Final avg temp: {result_chaos['final_avg_temp']:.2f}")
print(f"  Sensor dropouts: {result_chaos['sensor_dropouts_total']}")
print(f"  Final reward:   {result_chaos['final_reward']:.2f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Summary                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("PIPELINE SUMMARY")
print("═" * 60)
print("  1. TokamakConfig   — machine parameters (ITER/SPARC/DIII-D)")
print("  2. FusionKernel    — Grad-Shafranov equilibrium solve")
print("  3. TransportInputs — critical-gradient χ_i, χ_e, D_e")
print("  4. SPN → SNN       — compile control logic to LIF neurons")
print("  5. H-inf           — Riccati DARE robust controller")
print("  6. Digital Twin    — closed-loop benchmark with fault injection")
print("\nFor JAX autodiff:    python examples/tutorial_02_jax_autodiff.py")
print("For PPO RL agent:    python examples/tutorial_03_ppo_rl_agent.py")
