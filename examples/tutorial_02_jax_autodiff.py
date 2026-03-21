#!/usr/bin/env python3
"""Tutorial 02: JAX Autodiff & GPU-Accelerated Solvers.

Demonstrates JAX integration across the scpn-control stack:
  1. Thomas tridiagonal solver (JIT-compiled, differentiable)
  2. Crank-Nicolson transport step with jax.grad sensitivity
  3. Batched transport via jax.vmap (ensemble runs)
  4. JAX-differentiable Grad-Shafranov solver (full Picard iteration)
  5. Gradient of ψ w.r.t. plasma current (d(ψ)/d(Ip))
  6. GPU detection and dispatch

Prerequisites:
    pip install "scpn-control[jax]"
    python examples/tutorial_02_jax_autodiff.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scpn_control.core.jax_solvers import (
    batched_crank_nicolson,
    crank_nicolson_step,
    has_jax,
    has_jax_gpu,
    thomas_solve,
)

if not has_jax():
    print("JAX not installed. Install with: pip install 'scpn-control[jax]'")
    sys.exit(0)

import jax
import jax.numpy as jnp

print("═" * 60)
print("JAX AUTODIFF & GPU ACCELERATION")
print("═" * 60)
print(f"  JAX version:  {jax.__version__}")
print(f"  GPU available: {has_jax_gpu()}")
print(f"  Backend:       {jax.default_backend()}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 1: Thomas Tridiagonal Solver                            ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 1: Thomas Solver (JIT + Autodiff)")
print("═" * 60)

# Thomas algorithm solves tridiagonal systems Ax = d in O(n).
# The JAX version uses lax.scan for forward/back sweeps,
# enabling automatic differentiation through the solve.

n = 50
a = -0.5 * np.ones(n - 1)  # sub-diagonal
b = 2.0 * np.ones(n)  # diagonal
c = -0.5 * np.ones(n - 1)  # super-diagonal
d = np.sin(np.linspace(0, 2 * np.pi, n))  # RHS

# NumPy fallback
x_np = thomas_solve(a, b, c, d, use_jax=False)

# JAX JIT-compiled
x_jax = thomas_solve(a, b, c, d, use_jax=True)

# Verify parity
max_err = float(np.max(np.abs(x_np - np.asarray(x_jax))))
print(f"  Grid points: {n}")
print(f"  NumPy vs JAX max error: {max_err:.2e}")

# Benchmark: JIT warmup then timed run
_ = thomas_solve(a, b, c, d, use_jax=True)  # warmup
t0 = time.perf_counter()
for _ in range(100):
    thomas_solve(a, b, c, d, use_jax=True)
t_jax = (time.perf_counter() - t0) / 100 * 1e6

t0 = time.perf_counter()
for _ in range(100):
    thomas_solve(a, b, c, d, use_jax=False)
t_np = (time.perf_counter() - t0) / 100 * 1e6
print(f"  NumPy: {t_np:.1f} µs  |  JAX (JIT): {t_jax:.1f} µs")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 2: Crank-Nicolson Transport + jax.grad                  ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 2: Crank-Nicolson + Sensitivity Analysis")
print("═" * 60)

# The Crank-Nicolson scheme solves the cylindrical diffusion equation:
#   ∂T/∂t = (1/r) ∂/∂r (r χ ∂T/∂r) + S
# Implicit: (I - ½ dt L_h) T^{n+1} = (I + ½ dt L_h) T^n + dt S

nr = 50
rho = np.linspace(0.01, 1.0, nr)
drho = float(rho[1] - rho[0])
T_init = 10.0 * (1.0 - rho**2)
chi = 1.0 + 2.0 * rho**2
source = np.zeros(nr)

T_new = crank_nicolson_step(T_init, chi, source, rho, drho, 1e-4, 0.1)
print(f"  T_axis: {T_init[0]:.2f} → {T_new[0]:.4f} keV (dt=0.1ms)")

# Sensitivity: ∂T_axis/∂(dt) using jax.grad
# This differentiates through the entire CN solve.
T_j = jnp.asarray(T_init, dtype=jnp.float64)
chi_j = jnp.asarray(chi, dtype=jnp.float64)
src_j = jnp.asarray(source, dtype=jnp.float64)
rho_j = jnp.asarray(rho, dtype=jnp.float64)

from scpn_control.core.jax_solvers import _cn_step_jax  # type: ignore[attr-defined]


def axis_temp_vs_dt(dt_val):
    T_out = _cn_step_jax(T_j, chi_j, src_j, rho_j, drho, dt_val, 0.1)
    return T_out[0]


grad_fn = jax.grad(axis_temp_vs_dt)
dT_ddt = float(grad_fn(jnp.float64(1e-4)))
print(f"  dT_axis/d(dt) = {dT_ddt:.4f} keV/s (JAX autodiff)")

eps = 1e-7
fd = float(axis_temp_vs_dt(1e-4 + eps) - axis_temp_vs_dt(1e-4 - eps)) / (2 * eps)
print(f"  Finite diff:   {fd:.4f} keV/s (agreement: {abs(dT_ddt - fd) / max(abs(fd), 1e-15) * 100:.2f}%)")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 3: Batched Transport (jax.vmap)                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 3: Ensemble Transport (jax.vmap)")
print("═" * 60)

# vmap vectorizes over an ensemble of initial conditions.
# One JIT'd call processes all ensemble members simultaneously.

n_ensemble = 100
T_batch = np.stack([(10.0 + 2.0 * np.random.randn()) * (1.0 - rho**2) for _ in range(n_ensemble)])

t0 = time.perf_counter()
T_out = batched_crank_nicolson(T_batch, chi, source, rho, drho, dt=1e-4, T_edge=0.1)
elapsed = time.perf_counter() - t0

print(f"  Ensemble size: {n_ensemble}")
print(f"  Input T_axis:  {T_batch[:, 0].mean():.2f} ± {T_batch[:, 0].std():.2f} keV")
print(f"  Output T_axis: {T_out[:, 0].mean():.4f} ± {T_out[:, 0].std():.4f} keV")
print(f"  Wall time:     {elapsed * 1e3:.1f} ms ({elapsed / n_ensemble * 1e6:.1f} µs/member)")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 4: JAX Grad-Shafranov Solver                            ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 4: JAX-Differentiable GS Solver")
print("═" * 60)

from scpn_control.core.jax_gs_solver import jax_gs_solve

# Full Picard iteration via lax.fori_loop:
#   ψ^{k+1} = relax · GS_solve(ψ^k, J_φ(ψ^k)) + (1-relax) · ψ^k
# Inner Jacobi sweeps via lax.fori_loop.
# Entire pipeline is jax.grad-compatible.

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
print("  Grid: 33x33, Picard=40, Jacobi=100")
print(f"  max|ψ| = {float(np.max(np.abs(psi))):.4e} Wb")
print(f"  ψ shape: {psi.shape}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Section 5: Gradient d(ψ)/d(Ip)                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("SECTION 5: d(ψ)/d(Ip) — Equilibrium Sensitivity")
print("═" * 60)

from scpn_control.core.jax_gs_solver import jax_gs_grad_Ip

# Differentiates sum(|ψ|) w.r.t. Ip through the full Picard iteration.
# This enables adjoint-based shape optimization.

try:
    grad = jax_gs_grad_Ip(
        Ip_target=1e6,
        n_picard=10,
        n_jacobi=30,
        NR=17,
        NZ=17,
    )
    print(f"  d(Σ|ψ|)/d(Ip) = {grad:.6e}")
    print(f"  Sign: {'positive (more current → more flux)' if grad > 0 else 'negative'}")

    # Verify with finite difference
    eps = 1e3
    psi_p = jax_gs_solve(
        R_min=0.1,
        R_max=2.0,
        Z_min=-1.5,
        Z_max=1.5,
        NR=17,
        NZ=17,
        Ip_target=1e6 + eps,
        n_picard=10,
        n_jacobi=30,
    )
    psi_m = jax_gs_solve(
        R_min=0.1,
        R_max=2.0,
        Z_min=-1.5,
        Z_max=1.5,
        NR=17,
        NZ=17,
        Ip_target=1e6 - eps,
        n_picard=10,
        n_jacobi=30,
    )
    fd_grad = (float(np.sum(np.abs(psi_p))) - float(np.sum(np.abs(psi_m)))) / (2 * eps)
    print(f"  Finite diff:    {fd_grad:.6e}")
    if abs(fd_grad) > 1e-15:
        print(f"  Agreement:      {abs(grad - fd_grad) / abs(fd_grad) * 100:.1f}%")
except RuntimeError as e:
    print(f"  Skipped: {e}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ Summary                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("JAX INTEGRATION SUMMARY")
print("═" * 60)
print("  thomas_solve(backend='jax')     — O(n) tridiag, JIT + grad")
print("  crank_nicolson_step             — implicit diffusion, autodiff")
print("  batched_transport_step          — jax.vmap ensemble runs")
print("  jax_gs_solve                    — full GS Picard, differentiable")
print("  jax_gs_grad_Ip                  — d(ψ)/d(Ip) adjoint gradient")
print(f"\nGPU dispatch: {'ACTIVE' if has_jax_gpu() else 'CPU only (install jaxlib[cuda] for GPU)'}")
