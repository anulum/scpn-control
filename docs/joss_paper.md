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
multi-ion physics, a nonlinear $\delta f$ gyrokinetic solver with turbulent
saturation, kinetic electrons, Sugama collisions, and electromagnetic
$A_\parallel$ extension (KBM + microtearing modes), a native linear GK
eigenvalue solver, interfaces to five external GK codes (TGLF, GENE, GS2,
CGYRO, QuaLiKiz), a hybrid surrogate+GK validation layer with
out-of-distribution detection and online retraining, ten controllers (PID,
MPC, $H_\infty$, $\mu$-synthesis, NMPC, SNN, safe RL, sliding-mode,
gain-scheduled, fault-tolerant), disruption prediction with SPI mitigation,
and a companion Rust backend (5 crates, PyO3 bindings) achieving 11.9 µs
median kernel latency.

Pre-trained neural equilibrium weights for both SPARC and ITER geometries
enable sub-10ms CPU-only equilibrium inference. An interactive Streamlit
dashboard provides multi-machine shot replay (DIII-D, SPARC, ITER, NSTX-U,
JET), real-time GK transport visualisation, and OOD monitoring.

The codebase comprises 125 Python source modules and 5 Rust crates
(ndarray 0.16, rand 0.9, PyO3 0.25) with 3,300+ Python tests and 317 Rust
tests at 100% coverage across 20 CI jobs.

## Statement of Need

Real-time plasma control in magnetic confinement fusion requires
sub-millisecond decision latencies, validated transport physics, and robust
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
   [@dimits2000; @miller1998; @sugama2006], native TGLF-equivalent
   (SAT0/SAT1/SAT2, no Fortran binary), and nonlinear δf GK (5D Vlasov,
   JAX-accelerable). Interfaces to five external GK codes via subprocess,
   plus a hybrid layer that validates the surrogate against GK spot-checks
   with OOD detection, correction, and online retraining.

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

The Python package (125 modules) is organised into four layers:

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
      Dimits shift validated at $n_{kx}=256$; $\chi_i = 2.0\;\chi_{gB}$
      at CBC (GENE range: 1–5).
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
provides PyO3 bindings for performance-critical paths, achieving a median
kernel latency of 11.9 µs (Criterion-verified). The workspace passes 317
Rust tests with zero clippy warnings.

A JAX-accelerated GK backend (`jax_gk_solver.py`) batches eigenvalue solves
across the $k_y$ grid via `jax.vmap` and computes transport stiffness
$d\chi_i / d(R/L_{T_i})$ analytically via `jax.grad`.

## Validation

The solver is validated against:

- **Cyclone Base Case** [@dimits2000]: circular geometry ($R/a = 2.78$,
  $q = 1.4$, $\hat{s} = 0.78$, $R/L_{T_i} = 6.9$), producing positive
  ITG growth rates consistent with published benchmarks.
- **SPARC/ITER equilibria**: RMSE-gated against CFS SPARCPublic GEQDSK files
  and ITER design parameters. Pre-trained neural equilibrium weights for both
  machines achieve sub-10ms inference with <15% normalised RMSE.
- **DIII-D disruption shots**: 17 synthetic shots covering H-mode, VDE,
  beta-limit, locked-mode, density-limit, tearing, and snowflake
  configurations.
- **IMAS round-trip**: real `omas` ODS for equilibrium and core_profiles IDS,
  with bitwise fidelity on psi, pressure, and profile arrays.
- **IPB98(y,2)**: ITPA 20-tokamak H-mode confinement database [@ipb1999].
- **Dimits shift**: zero transport below the critical gradient at $n_{kx}=256$,
  verified against Dimits et al. (2000).
- **Cross-module chains**: 12 integration tests verify physics consistency
  across modules (bootstrap→NTM, IPB98→power balance, EPED→Troyon limit,
  sawtooth→NTM seed, L-H→H-mode→EPED, runaway→SPI trigger).

The test suite comprises 3,300+ Python tests and 317 Rust tests across 20 CI
jobs (Python 3.10–3.14 on Linux/Windows/macOS, Rust stable, JAX parity,
LIF+NEF SNN emulator, CodeQL security analysis, OpenSSF Scorecard). Coverage gate
is 99% (current: 100%). The project holds an OpenSSF CII Best Practices badge.
All physics equations cite their source papers; ~80 citations spanning
Porcelli (1996), Sauter (1999), Rosenbluth-Putvinski (1997), Connor-Hastie
(1975), Stix (1972), Martin (2008), Fitzpatrick (2001), Bosch-Hale (1992),
Doyle (1989), Rawlings (2017), Ames (2017), Kates-Harbeck (2019),
Stangeby (2000), Hirshman (1983), Boozer (1981), and others.

**Limitations**: the nonlinear $\delta f$ GK solver uses a flux-tube
approximation (no global effects or profile shearing); the Sugama collision
operator includes pitch-angle scattering and energy diffusion but omits
field-particle terms; external GK interfaces are mock-tested (no real
Fortran binaries in CI); DIII-D shots use synthetic data, not real MDSplus
archives. The neural equilibrium has not been cross-validated against P-EFIT
on identical equilibria.

## Acknowledgements

We thank the CFS SPARCPublic team for SPARC equilibrium data, the ITPA
H-mode confinement database contributors, and the GACODE, GENE, GS2, and
QuaLiKiz development teams for their publicly documented input/output formats.

## References

See [`paper.bib`](https://github.com/anulum/scpn-control/blob/main/paper.bib) for the full BibTeX bibliography.
