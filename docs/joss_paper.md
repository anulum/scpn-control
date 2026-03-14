# JOSS Paper: SCPN Control

**Neuro-Symbolic Stochastic Petri Net Controllers with First-Principles Gyrokinetic Transport for Real-Time Tokamak Plasma Control**

*Miroslav Šotek* ([ORCID 0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)) — ANULUM CH & LI

*14 March 2026*

!!! note "JOSS submission"
    This is the preprint of the paper prepared for the [Journal of Open Source Software](https://joss.theoj.org/).
    The canonical JOSS-formatted source is [`paper.md`](https://github.com/anulum/scpn-control/blob/main/paper.md)
    with bibliography in [`paper.bib`](https://github.com/anulum/scpn-control/blob/main/paper.bib).

---

## Summary

`scpn-control` is an open-source neuro-symbolic control engine for tokamak
plasma control combining Stochastic Petri Net (SPN) compilation into spiking
neural network (SNN) controllers, a three-path gyrokinetic transport system,
and an 8-layer Kuramoto-Sakaguchi phase dynamics engine.

The package provides: a Grad-Shafranov equilibrium solver (fixed and free
boundary, JAX-differentiable), a 1.5D Crank-Nicolson transport solver with
multi-ion physics, a native linear gyrokinetic eigenvalue solver with
electromagnetic extension (electrostatic + KBM + microtearing modes),
interfaces to five external GK codes (TGLF, GENE, GS2, CGYRO, QuaLiKiz),
a hybrid surrogate+GK validation layer with out-of-distribution detection
and online retraining, seven controllers (PID, MPC, $H_\infty$, $\mu$-synthesis,
NMPC, SNN, safe RL), disruption prediction with SPI mitigation, and a
companion Rust backend (5 crates, PyO3 bindings) achieving 11.9 µs median
kernel latency.

The codebase comprises 98 Python source modules and 5 Rust crates with
3,061+ tests at 100% coverage across 20 CI jobs.

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

1. **Three-path gyrokinetic transport** — a native linear GK eigenvalue
   solver in ballooning space (Miller geometry, Sugama collision operator)
   [@dimits2000; @miller1998; @sugama2006], interfaces to five external
   GK codes via subprocess, and a hybrid layer that validates the QLKNN
   surrogate [@plassche2020] against GK spot-checks with OOD detection,
   correction, and online retraining. No competing code has all three paths.

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

The Python package (98 modules) is organised into four layers:

- **Core** (`scpn_control.core`): Grad-Shafranov solver (Picard iteration
  with multigrid or SOR), 1.5D Crank-Nicolson transport, GEQDSK/IMAS I/O,
  IPB98(y,2) scaling [@ipb1999], neural equilibrium accelerator, uncertainty
  quantification, and the gyrokinetic transport stack:
    - *Native GK*: Miller flux-tube geometry [@miller1998], per-species
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
- **Control** (`scpn_control.control`): $H_\infty$ (Riccati DARE),
  $\mu$-synthesis (D-K iteration), NMPC (SQP), gain-scheduled PID,
  shape controller, safe RL (PPO with MHD constraint veto), sliding-mode
  vertical stability, scenario scheduler, fault-tolerant control, digital
  twin, flight simulator, Gymnasium environment, JAX-traceable runtime.

The Rust backend (`scpn-control-rs`, 5 crates) provides PyO3 bindings for
performance-critical paths, achieving a median kernel latency of 11.9 µs
(Criterion-verified).

A JAX-accelerated GK backend (`jax_gk_solver.py`) batches eigenvalue solves
across the $k_y$ grid via `jax.vmap` and computes transport stiffness
$d\chi_i / d(R/L_{T_i})$ analytically via `jax.grad`.

## Validation

The solver is validated against:

- **Cyclone Base Case** [@dimits2000]: circular geometry ($R/a = 2.78$,
  $q = 1.4$, $\hat{s} = 0.78$, $R/L_{T_i} = 6.9$), producing positive
  ITG growth rates consistent with published benchmarks.
- **SPARC/ITER equilibria**: RMSE-gated against CFS SPARCPublic GEQDSK files
  and ITER design parameters.
- **DIII-D disruption shots**: 17 synthetic shots covering H-mode, VDE,
  beta-limit, locked-mode, density-limit, tearing, and snowflake
  configurations.
- **IMAS round-trip**: real `omas` ODS for equilibrium and core_profiles IDS,
  with bitwise fidelity on psi, pressure, and profile arrays.
- **IPB98(y,2)**: ITPA 20-tokamak H-mode confinement database [@ipb1999].

The test suite comprises 3,061+ Python tests and 140+ Rust tests across 20 CI
jobs (Python 3.10–3.13 on Linux/Windows/macOS, Rust stable, JAX parity, Nengo
Loihi emulator, CodeQL security analysis, OpenSSF Scorecard). Coverage gate
is 99% (current: 100%).

**Limitations**: the native GK solver is linearised (no nonlinear turbulence);
the collision operator is simplified Sugama (pitch-angle only); external GK
interfaces are mock-tested (no real Fortran binaries in CI); DIII-D shots use
synthetic data, not real MDSplus archives. The neural equilibrium has not been
cross-validated against P-EFIT on identical equilibria.

## Acknowledgements

We thank the CFS SPARCPublic team for SPARC equilibrium data, the ITPA
H-mode confinement database contributors, and the GACODE, GENE, GS2, and
QuaLiKiz development teams for their publicly documented input/output formats.

## References

See [`paper.bib`](https://github.com/anulum/scpn-control/blob/main/paper.bib) for the full BibTeX bibliography.
