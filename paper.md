---
title: 'SCPN Control: Compiling Stochastic Petri Nets into Spiking Neural Network Controllers for Real-Time Tokamak Plasma Control'
tags:
  - Python
  - Rust
  - plasma physics
  - tokamak control
  - spiking neural networks
  - Petri nets
  - neuro-symbolic AI
authors:
  - name: Miroslav Šotek
    orcid: 0009-0009-3560-0851
    affiliation: 1
    corresponding: true
  - name: Michal Reiprich
    affiliation: 1
affiliations:
  - name: ANULUM CH & LI
    index: 1
date: 10 March 2026
bibliography: paper.bib
---

# Summary

`scpn-control` is an open-source neuro-symbolic control engine that compiles
Stochastic Petri Net (SPN) graphs into spiking neural network (SNN) controllers
with formal contract verification for tokamak plasma control. The package
provides a complete closed-loop control stack: a fixed-boundary Grad-Shafranov
equilibrium solver with an experimental external-coil boundary-shaping scaffold,
a 1D Crank-Nicolson transport solver with multi-ion physics,
five runtime-selectable controllers (PID, MPC, $H_\infty$, SNN, neuro-cybernetic),
disruption prediction with shattered pellet injection mitigation, a trained
QLKNN-10D neural transport surrogate, a Gymnasium-compatible reinforcement
learning environment with a trained PPO agent, and JAX-accelerated transport and
equilibrium primitives with GPU dispatch and automatic differentiation.
A companion Rust backend (5 crates, PyO3 bindings) achieves 2.1 µs median
kernel latency.

# Statement of Need

Real-time plasma control in magnetic confinement fusion requires sub-millisecond
decision latencies, formal safety guarantees, and validated physics models.
Existing open-source tools address subsets of this problem: TORAX
[@torax2024] provides differentiable transport, TCV-RL [@degrave2022] applies
deep RL to tokamak control, and FreeGS [@freegs] solves free-boundary equilibria.
No single package combines compilable control logic (SPNs), neuromorphic
execution (SNNs), multi-layer phase dynamics, and validated equilibrium solvers
in one coherent stack.

`scpn-control` fills this gap. The SPN-to-SNN compiler translates control graphs
into leaky integrate-and-fire neuron pools with stochastic bitstream encoding,
enforcing pre/post-condition contracts on every observation and action.
The multi-layer Kuramoto-Sakaguchi phase engine, driven by both a theoretical
$K_{nm}$ coupling matrix [@sotek2026knm] and a plasma-native 8-layer $K_{nm}$
encoding experimentally grounded interactions (drift-wave/zonal-flow,
NTM/bootstrap-current, ELM/pedestal), enables cross-scale synchronisation
monitoring via a Lyapunov stability guard.

# Implementation

The Python package (57 modules, ~22,900 lines) is organised into four layers:

- **Core** (`scpn_control.core`): Grad-Shafranov solver (Picard iteration with
  multigrid V-cycle or SOR elliptic solve), 1D Crank-Nicolson transport with
  gyro-Bohm diffusivity, GEQDSK/IMAS I/O, IPB98(y,2) scaling law benchmark,
  neural equilibrium accelerator, uncertainty quantification.
- **Phase** (`scpn_control.phase`): Kuramoto-Sakaguchi stepper, UPDE multi-layer
  solver, adaptive $K_{nm}$ engine, Lyapunov guard, WebSocket real-time monitor.
- **SCPN** (`scpn_control.scpn`): SPN structure, compiler, contract system,
  artifact serialisation.
- **Control** (`scpn_control.control`): $H_\infty$ (Riccati DARE), MPC with
  neural surrogate, optimal control, SNN controller, disruption predictor
  (transformer-based), SPI mitigation, digital twin, flight simulator,
  Gymnasium environment, JAX-traceable runtime.

The Rust backend (`scpn-control-rs`, 5 crates) provides PyO3 bindings for the
Grad-Shafranov solver, SNN pool, MPC controller, transport solver, and realtime
monitor. All solvers automatically dispatch to the Rust backend when available.

JAX-accelerated transport primitives (`scpn_control.core.jax_solvers`) provide
JIT-compiled, GPU-compatible Thomas tridiagonal solver and Crank-Nicolson
diffusion operator with automatic differentiation support, enabling gradient-based
sensitivity analysis and ensemble runs via `jax.vmap`. A JAX neural equilibrium
accelerator (`scpn_control.core.jax_neural_equilibrium`) provides GPU-dispatched
MLP + PCA inference for Grad-Shafranov equilibria with `jax.grad` support for
adjoint-based shape optimisation. A JAX-differentiable fixed-boundary
Grad-Shafranov solver (`scpn_control.core.jax_gs_solver`) implements the full
Picard iteration via `jax.lax.fori_loop`, enabling `jax.grad` through the
complete equilibrium solve — matching the autodiff depth of TORAX and FUSE.

A QLKNN-10D neural transport model (`scpn_control.core.neural_transport`)
trained on critical-gradient data provides millisecond-scale turbulent
transport predictions as a drop-in replacement for the analytic model.
A PPO agent trained on the Gymnasium-compatible `TokamakEnv` (500K timesteps,
3 seeds) achieves mean reward 143.7, outperforming both 1-step MPC (58.1) and
PID (-912.3) baselines with 0% disruption rate.

# Validation

The solver is validated against real DIII-D disruption shot data (17 shots covering
H-mode, VDE, beta-limit, locked-mode, density-limit, tearing, and snowflake
configurations), SPARC GEQDSK equilibria from CFS SPARCPublic, and the ITPA
20-tokamak H-mode confinement database with IPB98(y,2) scaling law benchmarks.
CI enforces <2% RMSE on pressure and safety-factor profiles via an automated
RMSE gate.

The test suite comprises 2,641 Python tests and 108 Rust tests across 26 CI jobs
(Python 3.9--3.13 on Linux/Windows/macOS, Rust stable, JAX parity, Nengo Loihi
emulator, real DIII-D validation, tutorial smoke). Coverage gate is 99% (current: 99.99%, 10,142
statements, 0 missed).

# Acknowledgements

We thank the CFS SPARCPublic team for SPARC equilibrium data and the ITPA
H-mode confinement database contributors.

# References
