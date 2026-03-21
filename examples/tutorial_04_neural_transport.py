#!/usr/bin/env python3
"""Tutorial 04: QLKNN-10D Neural Transport Surrogate.

Demonstrates the neural transport model and its physics:
  1. Critical-gradient model (analytic baseline)
  2. QLKNN-10D input space and normalization
  3. Transport regime classification (ITG, TEM, ETG, stable)
  4. Profile scan — diffusivity vs normalized gradient
  5. Comparison: analytic vs neural transport
  6. Integration with the Crank-Nicolson solver

Usage:
    pip install scpn-control
    python examples/tutorial_04_neural_transport.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scpn_control.core.neural_transport import (
    NeuralTransportModel,
    TransportInputs,
    critical_gradient_model,
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 1: Critical-Gradient Model (Analytic)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

print("═" * 60)
print("SECTION 1: Critical-Gradient Transport Model")
print("═" * 60)

# The critical-gradient model captures the key physics of ITG/TEM
# turbulent transport with minimal parameters:
#
#   χ_i = χ_GB · max(0, R/L_Ti − crit_ITG)^stiffness
#   χ_e = χ_GB · max(0, R/L_Te − crit_TEM)^stiffness
#   D_e = χ_e / 3   (particle diffusivity)
#
# Constants:
#   crit_ITG = 4.0      (Dimits et al. 2000)
#   crit_TEM = 5.0      (simplified TGLF approximation)
#   χ_GB = 1.0 m²/s     (Gyro-Bohm normalization)
#   stiffness = 2.0      (stiff critical-gradient exponent)

print("  Below threshold (R/L_Ti < 4.0):")
inp = TransportInputs(grad_ti=3.0, grad_te=3.0)
fluxes = critical_gradient_model(inp)
print(f"    R/L_Ti=3.0: χ_i={fluxes.chi_i:.4f}, χ_e={fluxes.chi_e:.4f} → {fluxes.channel}")

print("\n  Above threshold (R/L_Ti > 4.0):")
for grad in [5.0, 6.0, 8.0, 10.0]:
    inp = TransportInputs(grad_ti=grad, grad_te=grad * 0.9)
    f = critical_gradient_model(inp)
    print(f"    R/L_Ti={grad:.1f}: χ_i={f.chi_i:.3f}, χ_e={f.chi_e:.3f} → {f.channel}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 2: QLKNN-10D Input Space                                ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 2: QLKNN-10D Input Features")
print("═" * 60)

# The neural transport model uses 10 normalized plasma parameters
# (matching the QuaLiKiz NN paradigm of van de Plassche et al. 2020):

print("  TransportInputs fields:")
print(f"    {'Field':>10s}  {'Description':>30s}  {'Default':>8s}  {'Units':<12s}")
print(f"    {'─' * 10}  {'─' * 30}  {'─' * 8}  {'─' * 12}")
fields = [
    ("rho", "normalized flux coordinate", "0.5", "[-]"),
    ("te_kev", "electron temperature", "5.0", "[keV]"),
    ("ti_kev", "ion temperature", "5.0", "[keV]"),
    ("ne_19", "electron density", "5.0", "[10¹⁹ m⁻³]"),
    ("grad_te", "R/L_Te (norm gradient)", "6.0", "[-]"),
    ("grad_ti", "R/L_Ti (norm gradient)", "6.0", "[-]"),
    ("grad_ne", "R/L_ne (norm gradient)", "2.0", "[-]"),
    ("q", "safety factor", "1.5", "[-]"),
    ("s_hat", "magnetic shear", "0.8", "[-]"),
    ("beta_e", "electron beta", "0.01", "[-]"),
]
for name, desc, default, units in fields:
    print(f"    {name:>10s}  {desc:>30s}  {default:>8s}  {units:<12s}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 3: Regime Classification                                ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 3: Turbulent Transport Regimes")
print("═" * 60)

# ITG (Ion Temperature Gradient): dominant when R/L_Ti is large
# TEM (Trapped Electron Mode): dominant when R/L_Te is large
# ETG (Electron Temperature Gradient): at very steep electron gradients
# Stable: below critical gradients

scenarios = [
    ("Core (low grad)", TransportInputs(rho=0.2, grad_ti=3.0, grad_te=3.0, q=1.1)),
    ("Mid-radius ITG", TransportInputs(rho=0.5, grad_ti=7.0, grad_te=4.0, q=1.5)),
    ("Mid-radius TEM", TransportInputs(rho=0.5, grad_ti=3.0, grad_te=8.0, q=1.5)),
    ("Edge (steep)", TransportInputs(rho=0.8, grad_ti=10.0, grad_te=9.0, q=3.0)),
    ("Pedestal top", TransportInputs(rho=0.95, grad_ti=15.0, grad_te=14.0, q=5.0)),
]

print(f"  {'Scenario':>18s}  {'ρ':>4s}  {'R/L_Ti':>6s}  {'R/L_Te':>6s}  {'χ_i':>6s}  {'χ_e':>6s}  {'Channel':>8s}")
print(f"  {'─' * 18}  {'─' * 4}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 8}")
for name, inp in scenarios:
    f = critical_gradient_model(inp)
    print(
        f"  {name:>18s}  {inp.rho:4.2f}  {inp.grad_ti:6.1f}  {inp.grad_te:6.1f}  "
        f"{f.chi_i:6.3f}  {f.chi_e:6.3f}  {f.channel:>8s}"
    )

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 4: Gradient Scan                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 4: χ_i vs R/L_Ti (Gradient Scan)")
print("═" * 60)

# Scan R/L_Ti from 0 to 15 at fixed mid-radius conditions.
# Shows the stiffness cliff above the critical gradient.

grads = np.linspace(0, 15, 31)
chi_values = []
for g in grads:
    inp = TransportInputs(rho=0.5, grad_ti=float(g), grad_te=float(g) * 0.8, q=1.5)
    f = critical_gradient_model(inp)
    chi_values.append(f.chi_i)

print("  R/L_Ti  │  χ_i [m²/s]")
print("  ────────┼────────────")
for g, chi in zip(grads[::3], chi_values[::3]):
    bar = "█" * int(chi * 2)
    print(f"  {g:6.1f}  │  {chi:8.3f}  {bar}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 5: Neural vs Analytic Comparison                        ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 5: Neural vs Analytic Transport")
print("═" * 60)

weights_path = Path(__file__).resolve().parents[1] / "weights" / "neural_transport_qlknn.npz"

try:
    model = NeuralTransportModel(weights_path=weights_path)
    print(f"  Neural model loaded from: {weights_path}")
    print("  Architecture: 10→128→64→3 MLP")
    print()

    print(f"  {'R/L_Ti':>6s}  {'Analytic χ_i':>12s}  {'Neural χ_i':>10s}  {'Δ%':>6s}")
    print(f"  {'─' * 6}  {'─' * 12}  {'─' * 10}  {'─' * 6}")
    for g in [3.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        inp = TransportInputs(rho=0.5, grad_ti=g, grad_te=g * 0.9, q=1.5)
        f_analytic = critical_gradient_model(inp)
        f_neural = model.predict(inp)
        if f_analytic.chi_i > 0.001:
            delta = abs(f_neural.chi_i - f_analytic.chi_i) / f_analytic.chi_i * 100
            print(f"  {g:6.1f}  {f_analytic.chi_i:12.4f}  {f_neural.chi_i:10.4f}  {delta:5.1f}%")
        else:
            print(f"  {g:6.1f}  {f_analytic.chi_i:12.4f}  {f_neural.chi_i:10.4f}  {'N/A':>6s}")

except Exception as e:
    print(f"  Neural model not available: {e}")
    print("  Train with: python tools/train_neural_transport_qlknn.py --synthetic")
    print("  The analytic critical-gradient model is always available as fallback.")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 6: Transport + Crank-Nicolson Integration               ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 6: Transport-Coupled Temperature Evolution")
print("═" * 60)

from scpn_control.core.jax_solvers import crank_nicolson_step

nr = 50
rho_grid = np.linspace(0.01, 1.0, nr)
drho = float(rho_grid[1] - rho_grid[0])
T = 15.0 * (1.0 - rho_grid**2)  # parabolic initial profile
source = np.zeros(nr)

# Compute diffusivity profile from critical-gradient model
chi_profile = np.zeros(nr)
for i, r in enumerate(rho_grid):
    grad = 2.0 * 15.0 * r  # |dT/dr| ~ 2 T0 ρ
    inp = TransportInputs(rho=float(r), grad_ti=float(max(grad, 0.1)))
    f = critical_gradient_model(inp)
    chi_profile[i] = max(f.chi_i, 0.1)  # floor at 0.1 m²/s

# Evolve for 100 steps
dt = 1e-4
T_history = [T.copy()]
for step in range(100):
    T = crank_nicolson_step(T, chi_profile, source, rho_grid, drho, dt=dt, T_edge=0.5)
    if step % 20 == 19:
        T_history.append(T.copy())

print(f"  Grid: {nr} points, dt={dt * 1e3:.1f} ms, 100 steps")
print(f"  Initial T_axis = {T_history[0][0]:.2f} keV")
print(f"  Final T_axis   = {T_history[-1][0]:.4f} keV")
print(f"  χ_core = {chi_profile[0]:.3f} m²/s, χ_edge = {chi_profile[-1]:.3f} m²/s")

# Profile evolution table
print(f"\n  {'Step':>5s}  {'T_axis':>7s}  {'T(0.3)':>7s}  {'T(0.5)':>7s}  {'T(0.7)':>7s}  {'T(0.9)':>7s}")
print(f"  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
for i, T_snap in enumerate(T_history):
    step_n = i * 20 if i > 0 else 0
    i03 = int(0.3 / drho)
    i05 = int(0.5 / drho)
    i07 = int(0.7 / drho)
    i09 = int(0.9 / drho)
    print(
        f"  {step_n:5d}  {T_snap[0]:7.3f}  {T_snap[i03]:7.3f}  "
        f"{T_snap[i05]:7.3f}  {T_snap[i07]:7.3f}  {T_snap[min(i09, nr - 1)]:7.3f}"
    )

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Summary                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("NEURAL TRANSPORT SUMMARY")
print("═" * 60)
print("  critical_gradient_model  — analytic baseline (always available)")
print("  NeuralTransportModel     — QLKNN-10D trained MLP (10→128→64→3)")
print("  TransportInputs          — 10 normalized plasma parameters")
print("  TransportFluxes          — χ_e, χ_i, D_e + regime label")
print("  Crank-Nicolson coupling  — self-consistent profile evolution")
