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
solver in flux-tube geometry with JAX GPU acceleration (62$\times$ speedup),
interfaces to five external GK codes (TGLF, GENE, GS2, CGYRO, QuaLiKiz),
a hybrid surrogate+GK validation layer with out-of-distribution detection
and online retraining, ten controllers (PID, MPC, $H_\infty$, $\mu$-synthesis,
NMPC, SNN, safe RL, sliding-mode, gain-scheduled, fault-tolerant),
disruption prediction with SPI mitigation, and a
companion Rust backend (5 crates, PyO3 bindings) achieving 11.9 µs median
kernel latency.

The nonlinear gyrokinetic solver reproduces the Cyclone Base Case
benchmark [@dimits2000] with $\chi_i = 2.0\,\chi_{gB}$ (adiabatic electrons)
and $\chi_i = 1.3\,\chi_{gB}$ (kinetic electrons), both within the published
GENE/GS2 range of 1–5 $\chi_{gB}$. The Dimits shift — zero transport below
the critical gradient due to zonal flow suppression — is demonstrated at
$n_{kx}=256$.

The codebase comprises 125 Python source modules and 5 Rust crates with
3,164+ Python tests at 100% coverage across 20 CI jobs.

# Statement of Need

Real-time plasma control in magnetic confinement fusion requires
sub-millisecond decision latencies, validated transport physics, and robust
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
   JAX GPU acceleration delivers 62$\times$ speedup on commodity hardware.

2. **Cyclone Base Case validation** — the nonlinear solver achieves turbulent
   saturation at $n_{kx}=128$ with $\chi_i = 2.0\,\chi_{gB}$ (adiabatic) and
   $\chi_i = 1.3\,\chi_{gB}$ (kinetic electrons), within the published GENE/GS2
   range [@dimits2000]. The linear solver reproduces CBC growth rates within 21%
   of GENE. Transport stiffness (increasing $\chi_i$ with $R/L_{T_i}$) is
   confirmed across a 7-point gradient scan.

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

The Python package (125 modules) is organised into four layers:

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
      RK4 with CFL-adaptive dt, JAX-accelerated variant (62$\times$ on GPU).
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
provides PyO3 bindings for performance-critical paths, achieving a median
kernel latency of 11.9 µs (Criterion-verified).

# Validation

The solver is validated against:

- **Cyclone Base Case** [@dimits2000]: nonlinear turbulent saturation at
  $n_{kx}=128$ produces $\chi_i = 2.0\,\chi_{gB}$ (Krook collisions,
  adiabatic electrons) and $\chi_i = 1.3\,\chi_{gB}$ (Sugama collisions,
  kinetic electrons via semi-implicit advance). Both values lie within
  the published GENE/GS2 range of 1–5 $\chi_{gB}$. The linear eigenvalue
  solver reproduces CBC ITG growth rates ($\gamma_{max} = 0.14\,c_s/a$ at
  $k_y\rho_s = 0.37$) within 21% of GENE ($\gamma_{max} \approx 0.18$).
  A convergence study from $n_{kx}=8$ to $n_{kx}=256$ demonstrates
  systematic reduction of late-time growth rate from 0.93 to 0.005.
  Transport stiffness (monotonic $\chi_i(R/L_{T_i})$) is confirmed over
  a 7-point gradient scan ($R/L_{T_i} = 3$–$6.9$).
- **Dimits shift** [@dimits2000]: at $n_{kx}=256$, the subcritical case
  ($R/L_{T_i} = 3.0$) shows zero transport ($\chi_i < 10^{-6}\,\chi_{gB}$,
  $\phi$ at noise level) while the supercritical case ($R/L_{T_i} = 6.9$)
  shows growing ITG turbulence (late growth rate 0.48). Zonal flows
  self-consistently suppress all turbulent transport below the critical
  gradient — the definitive validation for nonlinear gyrokinetic solvers.
- **Sugama collision operator**: pitch-angle scattering with energy-dependent
  collision rate ($\nu(v) \propto v^{-3}$) and conservation corrections.
  Verified: $\int C[f]\,dv < 3\times10^{-8}$ (particles),
  $\int v_\parallel C[f]\,dv < 10^{-23}$ (momentum),
  $\int E\,C[f]\,dv < 2\times10^{-8}$ (energy). At low collisionality
  ($\nu = 0.01$), Sugama and Krook give identical saturated $\chi_i$.
- **SPARC/ITER equilibria**: RMSE-gated against CFS SPARCPublic GEQDSK files
  and ITER design parameters.
- **DIII-D disruption shots**: 17 synthetic shots covering H-mode, VDE,
  beta-limit, locked-mode, density-limit, tearing, and snowflake
  configurations.
- **IMAS round-trip**: real `omas` ODS for equilibrium and core_profiles IDS.
- **IPB98(y,2)**: ITPA 20-tokamak H-mode confinement database [@ipb1999].

The test suite comprises 3,164+ Python tests and 317 Rust tests across 20
CI jobs (Python 3.10–3.14 on Linux/Windows/macOS, Rust stable, JAX parity,
CodeQL security analysis, OpenSSF Scorecard). Coverage gate is 99%
(current: 100%). The project holds an OpenSSF CII Best Practices badge.
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
