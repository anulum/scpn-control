#!/usr/bin/env python3
"""scpn-control quickstart -- equilibrium + transport + SNN compile + autodiff.

Usage:
    pip install -e ".[jax]"
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- 1. Equilibrium solve (JAX-differentiable) --------------------------

from scpn_control.core.jax_gs_solver import jax_gs_solve

psi = jax_gs_solve(
    R_min=0.1,
    R_max=2.0,
    Z_min=-1.5,
    Z_max=1.5,
    NR=33,
    NZ=33,
    Ip_target=1e6,
    n_picard=40,
    n_jacobi=100,
)
print(f"Equilibrium: {psi.shape} grid, max(|psi|) = {np.max(np.abs(psi)):.4e}")

# -- 2. Transport step (Crank-Nicolson) ---------------------------------

from scpn_control.core.jax_solvers import crank_nicolson_step

nr = 50
rho = np.linspace(0.01, 1.0, nr)
drho = rho[1] - rho[0]
T = 10.0 * (1.0 - rho**2)
chi = 1.0 + 2.0 * rho**2
source = np.zeros(nr)

T_new = crank_nicolson_step(T, chi, source, rho, drho, dt=1e-4, T_edge=0.1)
print(f"Transport: T_axis {T[0]:.1f} keV -> {T_new[0]:.4f} keV")

# -- 3. SPN -> SNN compilation ------------------------------------------

from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.structure import StochasticPetriNet

net = StochasticPetriNet()
net.add_place("idle", initial_tokens=1.0)
net.add_place("heating", initial_tokens=0.0)
net.add_place("cooled", initial_tokens=0.0)
net.add_transition("ignite", threshold=0.5)
net.add_transition("quench", threshold=0.5)
net.add_arc("idle", "ignite", weight=1.0)
net.add_arc("ignite", "heating", weight=1.0)
net.add_arc("heating", "quench", weight=1.0)
net.add_arc("quench", "cooled", weight=1.0)
net.compile()

compiler = FusionCompiler()
artifact = compiler.compile(net)
print(f"SNN compiler: {net.n_places} places -> {artifact.n_transitions} LIF neurons, W_in shape {artifact.W_in.shape}")

# -- 4. JAX autodiff through full GS solve -------------------------------

try:
    from scpn_control.core.jax_gs_solver import jax_gs_grad_Ip

    grad = jax_gs_grad_Ip(1e6, n_picard=10, n_jacobi=30, NR=17, NZ=17)
    print(f"JAX autodiff: d(sum_psi)/d(Ip) = {grad:.6e}")
except RuntimeError:
    print("JAX not available -- autodiff demo skipped")

print("\nDone. See examples/ for more demos.")
