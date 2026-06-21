---
title: 'SCPN Control: Neuro-Symbolic Stochastic Petri Net Controllers with First-Principles Gyrokinetic Transport for Real-Time Tokamak Plasma Control'
tags:
  - Python
  - Rust
  - plasma physics
  - tokamak control
  - gyrokinetics
  - spiking neural networks
  - Petri nets
  - neuro-symbolic AI
authors:
  - name: Miroslav Šotek
    orcid: 0009-0009-3560-0851
    affiliation: 1
    corresponding: true
affiliations:
  - name: ANULUM CH & LI
    index: 1
date: 17 March 2026
bibliography: paper.bib
---
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->


# Summary

`scpn-control` is an open-source neuro-symbolic control engine for tokamak
plasma control combining Stochastic Petri Net (SPN) compilation into spiking
neural network (SNN) controllers, a five-tier gyrokinetic transport system
including a nonlinear $\delta f$ solver, and an 8-layer Kuramoto-Sakaguchi
phase dynamics engine.

The package provides: a Grad-Shafranov equilibrium solver (fixed and free
boundary, JAX-differentiable), a 1.5D Crank-Nicolson transport solver with
multi-ion physics, a native linear gyrokinetic eigenvalue solver with
electromagnetic extension, a native TGLF-equivalent quasilinear model
(SAT0/SAT1/SAT2 saturation rules), a nonlinear $\delta f$ gyrokinetic
solver in flux-tube geometry with JAX GPU acceleration measured in a bounded
local GPU benchmark (62$\times$ in that recorded context), interfaces to five
external GK codes (TGLF, GENE, GS2, CGYRO, QuaLiKiz) whose quantitative
external-code claims remain blocked until real artefacts are admitted, a
hybrid surrogate+GK validation layer with out-of-distribution detection
and online retraining, ten controllers (PID, MPC, $H_\infty$, $\mu$-synthesis,
NMPC, SNN, safe RL, sliding-mode, gain-scheduled, fault-tolerant),
disruption prediction with SPI mitigation, and a
companion Rust backend (5 crates, PyO3 bindings) with a reproducible
benchmark reporting a ~5 µs P50 native integrated control cycle on the CI
runner (2.85 µs on the local workstation).

The nonlinear gyrokinetic solver implements the Cyclone Base Case
configuration [@dimits2000], Rosenbluth-Hinton zonal-flow damping, kinetic
electrons, and Sugama collisions. After the v0.19.0 physics audit, its
late-time nonlinear heat-flux normalisation is treated as an open validation
target: the latest 2000-step adiabatic run had not reached saturated
$\chi_i$, and the linear local-dispersion path overpredicts the published
GENE growth rate. The manuscript therefore does not claim quantitative
nonlinear CBC agreement until a longer, reverified convergence campaign is
complete.

The codebase comprises a Python package and 5 Rust crates with broad Python
module tests, a 93% local package-coverage gate, and a multi-workflow CI
matrix.

# Statement of Need

Real-time plasma control in magnetic confinement fusion requires low-latency
decision paths, bounded and traceable transport-physics evidence, and robust
safety guarantees. Existing open-source tools address subsets of this problem:
TORAX [@torax2024] provides JAX-differentiable transport with TGLF coupling,
TCV-RL [@degrave2022] applies deep RL to tokamak control, FreeGS [@freegs]
solves free-boundary equilibria, and FUSE [@fuse2024] offers integrated
design-to-operations modelling. No single package combines compilable control
logic (SPNs), neuromorphic execution (SNNs), first-principles gyrokinetic
transport from quasilinear through nonlinear, and multi-layer phase dynamics
in one coherent stack.

`scpn-control` fills this gap with four distinguishing capabilities:

1. **Five-tier gyrokinetic transport** — spanning critical-gradient models
   ($\sim\mu$s), QLKNN surrogates ($\sim$24 ns), a native linear GK
   eigenvalue solver ($\sim$0.3 s per flux surface) [@dimits2000; @miller1998],
   a native TGLF-equivalent model with SAT0/SAT1/SAT2 spectral saturation
   [@staebler2007; @staebler2017; @maeyama2015], and a nonlinear $\delta f$
   gyrokinetic solver with dealiased E$\times$B bracket, ballooning connection
   boundary conditions, Rosenbluth-Hinton zonal flow physics
   [@rosenbluth1998], Sugama collision operator [@sugama2006], and optional
   kinetic electrons via semi-implicit backward-Euler treatment.
   JAX GPU acceleration is reported as a bounded local benchmark result
   (62$\times$ in the recorded GPU context), not as a facility or production
   timing claim.

2. **Cyclone Base Case validation harness** — the repository includes CBC
   scenarios, grid scans, kinetic-electron lanes, and published-code interface
   adapters. Quantitative nonlinear agreement is not yet claimed: the current
   audited result is an unsaturated adiabatic heat-flux trace after 2000 steps,
   and the linear local-dispersion path overpredicts the GENE reference growth
   rate. The remaining publication gate is a longer reverified convergence
   campaign plus native GK comparison against real TGLF/GENE-class binaries on
   identical inputs.

3. **SPN-to-SNN compilation** — translates control graphs into leaky
   integrate-and-fire neuron pools with stochastic bitstream encoding
   [@murata1989; @maass1997], enforcing pre/post-condition contracts on
   every observation and action. The pure-NumPy LIF+NEF engine requires
   no external dependencies.

4. **8-layer plasma phase dynamics** — a Kuramoto-Sakaguchi multi-layer
   UPDE engine [@kuramoto1975; @sotek2026knm] encoding experimentally
   grounded interactions (drift-wave/zonal-flow, NTM/bootstrap-current,
   ELM/pedestal), with GK-driven adaptive $K_{nm}$ coupling and Lyapunov
   stability monitoring.

# Implementation

The Python package is organised into four layers:

- **Core** (`scpn_control.core`): Grad-Shafranov solver (Picard iteration
  with multigrid or SOR), 1.5D Crank-Nicolson transport, GEQDSK/IMAS I/O,
  IPB98(y,2) scaling [@ipb1999], neural equilibrium accelerator, uncertainty
  quantification, and the gyrokinetic transport stack:
    - *Native linear GK*: Miller flux-tube geometry [@miller1998], per-species
      Gauss-Legendre velocity grid, local dispersion relation with Newton
      root-finding, mixing-length quasilinear fluxes. Electromagnetic
      extension adds KBM [@tang1980] and microtearing [@drake1977] modes.
    - *Native TGLF-equivalent*: SAT0/SAT1/SAT2 spectral saturation
      [@staebler2007; @staebler2017], E$\times$B shear quench [@waltz1997],
      trapped-particle damping [@connor1974], multi-scale ITG-ETG coupling
      [@maeyama2015]. No external binary required.
    - *Nonlinear $\delta f$*: 5D Vlasov in flux-tube geometry, dealiased
      E$\times$B bracket (Orszag 2/3 rule), 4th-order parallel streaming with
      ballooning connection BC, Rosenbluth-Hinton zonal Krook damping
      [@rosenbluth1998], Sugama collision operator [@sugama2006] with
      particle/momentum/energy conservation, optional kinetic electrons
      (semi-implicit backward-Euler for electron parallel streaming),
      RK4 with CFL-adaptive dt, JAX-accelerated variant with the recorded
      bounded local GPU speedup described above.
    - *External GK*: TGLF [@staebler2007], GENE [@jenko2000], GS2
      [@kotschenreuther1995], CGYRO [@candy2003], QuaLiKiz [@bourdelle2007]
      via subprocess with automatic input deck generation and output parsing.
    - *Hybrid*: OOD detection (Mahalanobis + ensemble + range), spot-check
      scheduling, multiplicative/additive correction with EMA smoothing,
      online surrogate retraining with validation holdout and rollback.
- **Phase** (`scpn_control.phase`): Kuramoto-Sakaguchi stepper, UPDE
  multi-layer solver, adaptive $K_{nm}$ engine with GK→UPDE bridge,
  Lyapunov guard, WebSocket real-time monitor.
- **SCPN** (`scpn_control.scpn`): SPN structure, compiler, contract system,
  artifact serialisation, FPGA bitstream export.
- **Control** (`scpn_control.control`): $H_\infty$ (Riccati DARE),
  $\mu$-synthesis (D-K iteration), NMPC (SQP), gain-scheduled PID,
  shape controller, safe RL (PPO with MHD constraint veto), sliding-mode
  vertical stability, scenario scheduler, fault-tolerant control, digital
  twin, flight simulator, Gymnasium environment, JAX-traceable runtime,
  ITER CODAC/EPICS interface, federated disruption prediction.

The Rust backend (`scpn-control-rs`, 5 crates, ndarray 0.16, rand 0.9)
provides PyO3 bindings for performance-critical paths. Its reproducible
benchmark reports a ~5 µs P50 native integrated control cycle on the CI runner
(2.85 µs locally); production runtime claims remain subject to runtime-admission
evidence.

# Validation

The solver is validated against:

- **Cyclone Base Case** [@dimits2000]: CBC input construction, nonlinear
  $\delta f$ evolution, JAX acceleration, kinetic-electron support, Sugama
  collisions, and grid-scan machinery are implemented. The latest audited
  2000-step adiabatic run did not reach saturated $\chi_i$, so the former
  saturated-heat-flux agreement claims are withdrawn pending a longer
  convergence campaign. The current linear local-dispersion result
  overpredicts the GENE reference growth rate; it is retained as a model
  limitation, not a quantitative validation claim.
- **Dimits shift** [@dimits2000]: the repository contains subcritical and
  supercritical scan machinery and zonal-flow damping, but the full Dimits-gap
  claim remains a revalidation target after the kinetic-electron and
  Gauss-Laguerre quadrature changes.
- **Sugama collision operator**: pitch-angle scattering with energy-dependent
  collision rate ($\nu(v) \propto v^{-3}$) and conservation corrections.
  Verified: $\int C[f]\,dv < 3\times10^{-8}$ (particles),
  $\int v_\parallel C[f]\,dv < 10^{-23}$ (momentum),
  $\int E\,C[f]\,dv < 2\times10^{-8}$ (energy). At low collisionality
  ($\nu = 0.01$), Sugama and Krook agree on the low-collisionality collision
  response in the verified operator tests; saturated nonlinear $\chi_i$ remains
  part of the CBC revalidation target.
- **SPARC/ITER equilibria**: RMSE-gated against CFS SPARCPublic GEQDSK files
  and ITER design parameters.
- **DIII-D disruption shots**: 17 synthetic fixture shots covering H-mode, VDE,
  beta-limit, locked-mode, density-limit, tearing, and snowflake
  configurations.
- **IMAS round-trip**: real `omas` ODS for equilibrium and core_profiles IDS.
- **IPB98(y,2)**: ITPA 20-tokamak H-mode confinement database [@ipb1999].

The test suite comprises Python module tests and Rust workspace tests across
CI jobs (Python 3.10–3.14 on Linux/Windows/macOS, Rust stable, JAX parity,
CodeQL security analysis, OpenSSF Scorecard). The local and CI coverage
configuration currently enforces a 93% package-coverage gate while publishing
XML coverage artefacts. The project holds an OpenSSF CII Best Practices badge.
All physics equations cite their source papers; ~80 citations spanning
Porcelli (1996), Sauter (1999), Rosenbluth-Putvinski (1997), Stix (1972),
Bosch-Hale (1992), Doyle (1989), Rawlings (2017), Stangeby (2000),
Hirshman (1983), and others. Twelve cross-module integration tests verify
consistency across physics chains (bootstrap→NTM, EPED→Troyon, L-H→EPED,
runaway→SPI).

**Limitations**: external GK interfaces are mock-tested (no real Fortran
binaries in CI); DIII-D shots use synthetic data, not real MDSplus archives;
the neural equilibrium has not been cross-validated against P-EFIT on
identical equilibria.

# Acknowledgements

We thank the CFS SPARCPublic team for SPARC equilibrium data, the ITPA
H-mode confinement database contributors, and the GACODE, GENE, GS2, and
QuaLiKiz development teams for their publicly documented input/output formats.

# References
