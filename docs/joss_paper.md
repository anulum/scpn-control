<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Control -->
<!-- Description: JOSS paper draft. -->

# JOSS Paper: SCPN Control

**Neuro-Symbolic Stochastic Petri Net Controllers with First-Principles Gyrokinetic Transport for Real-Time Tokamak Plasma Control**

*Miroslav Šotek* ([ORCID 0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)) — ANULUM CH & LI

*17 March 2026*

!!! note "JOSS submission"
    This is the preprint of the paper prepared for the [Journal of Open Source Software](https://joss.theoj.org/).
    The canonical JOSS-formatted source is [`paper.md`](https://github.com/anulum/scpn-control/blob/main/paper.md)
    with bibliography in [`paper.bib`](https://github.com/anulum/scpn-control/blob/main/paper.bib).

---

## Summary

`scpn-control` is an open-source neuro-symbolic control engine for tokamak
plasma control combining Stochastic Petri Net (SPN) compilation into spiking
neural network (SNN) controllers, a five-tier gyrokinetic transport system,
and an 8-layer Kuramoto-Sakaguchi phase dynamics engine.

The package provides: a Grad-Shafranov equilibrium solver (fixed and free
boundary, JAX-differentiable), a 1.5D Crank-Nicolson transport solver with
multi-ion physics, a native linear gyrokinetic eigenvalue solver with
electromagnetic extension, a native TGLF-like quasilinear approximation
(SAT0/SAT1/SAT2 saturation rules), a nonlinear $\delta f$ gyrokinetic
solver in flux-tube geometry with JAX GPU acceleration measured in a bounded
local GPU benchmark, interfaces to five external GK codes (TGLF, GENE, GS2,
CGYRO, QuaLiKiz) whose quantitative external-code claims remain blocked until
real artefacts are admitted, a hybrid surrogate+GK validation layer with
out-of-distribution detection and online retraining, ten controllers (PID,
MPC, $H_\infty$, $\mu$-synthesis, NMPC, SNN, safe RL, sliding-mode,
gain-scheduled, fault-tolerant), disruption prediction with SPI mitigation,
and a companion Rust backend (5 crates, PyO3 bindings) with a reproducible
benchmark reporting a 5.05 µs P50 native integrated control cycle on the CI
runner (2.85 µs on the local workstation).

The nonlinear gyrokinetic solver implements the Cyclone Base Case
configuration, Rosenbluth-Hinton zonal-flow damping, kinetic electrons, and
Sugama collisions. After the v0.19.0 physics audit, its late-time nonlinear
heat-flux normalisation is treated as an open validation target: the latest
2000-step adiabatic run had not reached saturated $\chi_i$, and the linear
local-dispersion path overpredicts the published GENE growth rate.

The codebase comprises a Python package and 5 Rust crates (ndarray 0.16,
rand 0.9, PyO3 0.25) with broad Python and Rust test coverage, a 93% local
package-coverage gate, and a multi-workflow CI matrix.

## Statement of Need

Real-time plasma control in magnetic confinement fusion requires low-latency
decision paths, bounded and traceable transport-physics evidence, and robust
safety guarantees. Existing open-source tools address subsets of this problem:
TORAX [@torax2024] provides JAX-differentiable transport with TGLF coupling,
TCV-RL [@degrave2022] applies deep RL to tokamak control, FreeGS [@freegs]
solves free-boundary equilibria, and FUSE [@fuse2024] offers integrated
design-to-operations modelling. No single package combines compilable control
logic (SPNs), neuromorphic execution (SNNs), first-principles gyrokinetic
transport (native eigenvalue solver + external code coupling + hybrid
surrogate validation), and multi-layer phase dynamics in one coherent stack.

`scpn-control` fills this gap with three distinguishing capabilities:

1. **Five-tier gyrokinetic transport** — critical-gradient baseline,
   QLKNN surrogate [@plassche2020], native linear GK eigenvalue solver
   in ballooning space (Miller geometry, Sugama collision operator)
   [@dimits2000; @miller1998; @sugama2006], native TGLF-like approximation
   (SAT0/SAT1/SAT2, no Fortran binary), and nonlinear δf GK (5D Vlasov,
   JAX-accelerable). Interfaces to five external GK codes via subprocess are
   present, but quantitative external-code claims remain blocked until real
   artefacts are admitted; the hybrid layer validates the surrogate against
   GK spot-checks with OOD detection, correction, and online retraining.

2. **SPN-to-SNN compilation** — translates control graphs into leaky
   integrate-and-fire neuron pools with stochastic bitstream encoding
   [@murata1989; @maass1997], enforcing pre/post-condition contracts on
   every observation and action.

3. **8-layer plasma phase dynamics** — a Kuramoto-Sakaguchi multi-layer
   UPDE engine [@kuramoto1975; @sotek2026knm] encoding experimentally
   grounded interactions (drift-wave/zonal-flow, NTM/bootstrap-current,
   ELM/pedestal), with GK-driven adaptive $K_{nm}$ coupling and Lyapunov
   stability monitoring.

## Implementation

The Python package is organised into four layers:

- **Core** (`scpn_control.core`): Grad-Shafranov solver (Picard iteration
  with multigrid or SOR), 1.5D Crank-Nicolson transport, GEQDSK/IMAS I/O,
  IPB98(y,2) scaling [@ipb1999], neural equilibrium accelerator, uncertainty
  quantification, neoclassical transport (Chang-Hinton/banana/plateau/
  Pfirsch-Schlüter with full Sauter bootstrap [@sauter1999]), EPED pedestal
  prediction (Snyder 2009/2011), sawtooth physics (Porcelli 1996 trigger,
  Kadomtsev crash), NTM dynamics (modified Rutherford equation with GGJ
  tearing stability), RWM feedback with rotation stabilization, Alfvén
  eigenmode analysis, L-H transition (Martin 2008 scaling), impurity
  transport, runaway electron physics (Connor-Hastie/Rosenbluth-Putvinski),
  SPI mitigation (Parks-Turnbull ablation), burn control, integrated
  scenario simulator with Strang operator splitting, and the gyrokinetic
  transport stack:
    - *Nonlinear GK*: 5D $\delta f$ Vlasov in flux-tube geometry, dealiased
      E×B bracket, ballooning connection BC, Rosenbluth-Hinton zonal Krook,
      kinetic electrons (semi-implicit backward-Euler), Sugama collisions
      (pitch-angle + energy diffusion), electromagnetic $A_\parallel$.
      CBC, Dimits-shift, kinetic-electron, and Sugama-collision machinery are
      implemented, but the latest audited nonlinear heat-flux traces had not
      reached saturated $\chi_i$ after 2000 steps. Quantitative nonlinear CBC
      agreement is therefore a revalidation target, not a submitted claim.
    - *Native linear GK*: Miller flux-tube geometry [@miller1998], per-species
      Gauss-Legendre velocity grid, Sugama collision operator [@sugama2006],
      response-matrix eigenvalue solver, mixing-length quasilinear fluxes.
      Electromagnetic extension adds KBM [@tang1980] and microtearing
      [@drake1977] modes via $A_\parallel$ and $\delta B_\parallel$.
    - *External GK*: TGLF [@staebler2007], GENE [@jenko2000], GS2
      [@kotschenreuther1995], CGYRO [@candy2003], QuaLiKiz [@bourdelle2007]
      via subprocess with automatic input deck generation and output parsing.
    - *Hybrid*: OOD detection (Mahalanobis + ensemble + range), spot-check
      scheduling (periodic/adaptive/critical-region), multiplicative/additive
      correction with EMA smoothing, online surrogate retraining with
      validation holdout and rollback.
- **Phase** (`scpn_control.phase`): Kuramoto-Sakaguchi stepper, UPDE
  multi-layer solver, adaptive $K_{nm}$ engine with GK→UPDE bridge,
  Lyapunov guard, WebSocket real-time monitor.
- **SCPN** (`scpn_control.scpn`): SPN structure, compiler, contract system,
  artifact serialisation.
- **Control** (`scpn_control.control`): $H_\infty$ (Riccati DARE,
  Doyle 1989, Zhou 1996), $\mu$-synthesis (DK-iteration, Doyle 1982),
  NMPC (Rawlings 2017), gain-scheduled PID (Rugh-Shamma 2000),
  shape controller (Ariola-Pironti 2008, ISOFLUX), safe RL (CPO with
  control barrier functions, Ames 2017), sliding-mode vertical stability
  (Utkin 1992, Humphreys 2009), fault-tolerant FDIR (Blanke 2006),
  disruption prediction (Kates-Harbeck 2019 FRNN), SPI mitigation
  (Commaux 2010, Parks-Turnbull ablation), volt-second management
  (Ejima 1982), density control (Greenwald 2002 limit), detachment
  control (Stangeby 2000 two-point model), burn controller (Bosch-Hale
  reactivity, Lawson criterion), scenario scheduler, digital twin,
  flight simulator, Gymnasium environment, JAX-traceable runtime.

The Rust backend (`scpn-control-rs`, 5 crates, ndarray 0.16, rand 0.9)
provides PyO3 bindings for performance-critical paths. Its reproducible
benchmark reports a 5.05 µs P50 native integrated control cycle on the CI runner
(2.85 µs locally); production runtime claims remain subject to runtime-admission
evidence.
The workspace includes Rust test and clippy coverage.

A JAX-accelerated GK backend (`jax_gk_solver.py`) batches eigenvalue solves
across the $k_y$ grid via `jax.vmap` and computes transport stiffness
$d\chi_i / d(R/L_{T_i})$ analytically via `jax.grad`.

## Validation

The solver is validated against:

- **Cyclone Base Case** [@dimits2000]: CBC input construction, nonlinear
  $\delta f$ evolution, JAX acceleration, kinetic-electron support, Sugama
  collisions, and grid-scan machinery are implemented. The latest audited
  2000-step adiabatic run did not reach saturated $\chi_i$, so quantitative
  nonlinear heat-flux agreement remains a revalidation target.
- **SPARC/ITER equilibria**: RMSE-gated against CFS SPARCPublic GEQDSK files
  and ITER design parameters.
- **DIII-D disruption shots**: repository reference disruption-shot artefacts
  and GEQDSK files covered by manifest checksums; synthetic shots remain CI
  plumbing fixtures only.
- **IMAS round-trip**: real `omas` ODS for equilibrium and core_profiles IDS,
  with bitwise fidelity on psi, pressure, and profile arrays.
- **IPB98(y,2)**: ITPA 20-tokamak H-mode confinement database [@ipb1999].
- **Dimits shift**: subcritical/supercritical scan machinery is present, but
  the full Dimits-gap result must be reverified after the kinetic-electron and
  velocity-grid changes.
- **Cross-module chains**: 12 integration tests verify physics consistency
  across modules (bootstrap→NTM, IPB98→power balance, EPED→Troyon limit,
  sawtooth→NTM seed, L-H→H-mode→EPED, runaway→SPI trigger).

The test suite comprises Python module tests and Rust workspace tests across
CI workflows for Python, Rust, JAX parity, LIF+NEF SNN emulation, security
analysis, and OpenSSF Scorecard checks. The local coverage configuration
currently enforces a 93% package-coverage gate while specialised tests continue
to replace broad coverage-bucket debt. The project holds an OpenSSF CII Best
Practices badge.
All physics equations cite their source papers; ~80 citations spanning
Porcelli (1996), Sauter (1999), Rosenbluth-Putvinski (1997), Connor-Hastie
(1975), Stix (1972), Martin (2008), Fitzpatrick (2001), Bosch-Hale (1992),
Doyle (1989), Rawlings (2017), Ames (2017), Kates-Harbeck (2019),
Stangeby (2000), Hirshman (1983), Boozer (1981), and others.

**Limitations**: the nonlinear $\delta f$ GK solver uses a flux-tube
approximation (no global effects or profile shearing); the Sugama collision
operator includes pitch-angle scattering and energy diffusion but omits
field-particle terms; external GK interfaces are mock-tested (no real
Fortran binaries in CI); DIII-D replay evidence is repository-artefact
validation, not live MDSplus or facility-control validation. The neural
equilibrium has not been cross-validated against P-EFIT on identical
equilibria.

## Acknowledgements

We thank the CFS SPARCPublic team for SPARC equilibrium data, the ITPA
H-mode confinement database contributors, and the GACODE, GENE, GS2, and
QuaLiKiz development teams for their publicly documented input/output formats.

## References

See [`paper.bib`](https://github.com/anulum/scpn-control/blob/main/paper.bib) for the full BibTeX bibliography.

## Practical use and scope

Use this paper draft as the publication-grade summary of the project contribution.

- Keep the manuscript aligned with reproducible code and benchmark evidence.
- Ensure scientific claims in this draft remain bounded by current validation status.
- Do not introduce new benchmark claims in this document without upstream evidence updates.
