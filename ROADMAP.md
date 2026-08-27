# Release history and evidence boundaries

This public record summarizes shipped SCPN Control capabilities and the claim
boundaries that apply to the current package. Operational sequencing, internal
priorities, and release-control notes are intentionally not published here.
Live capability counts come from
`docs/_generated/capability_manifest.json`; bounded physics status comes from
`docs/physics_traceability.md`, generated from
`validation/physics_traceability.json`.

## Published release history

### v0.1.0 to v0.9.0 — package and control foundations

- Extracted the controller-facing Python modules and five Rust crates from the
  broader fusion stack.
- Added the CLI, Streamlit dashboard, WebSocket phase stream, Gymnasium
  environment, IMAS/OMAS adapter, JAX and Loihi optional paths, packaging,
  strict typing, cross-platform CI, supply-chain checks, and API documentation.
- Added Paper 27 Kuramoto-Sakaguchi phase dynamics. The Lyapunov monitor remains
  a research-prototype advisory surface with warm-up and consecutive-window
  fail-open behaviour; it is not a standalone fail-closed safety interlock.

### v0.10.0 to v0.15.0 — differentiable physics and controller evidence

- Added differentiable transport and equilibrium paths, a JAX
  Grad-Shafranov solver, neural-equilibrium and neural-transport facades, PPO
  training support, and reproducible controller comparisons.
- The committed RL benchmark records PPO reward=121.1 vs MPC=59.4 vs PID=-911.2 over
  50 episodes. This is repository simulation evidence, not a facility-control
  result.
- Corrected the cylindrical Grad-Shafranov stencil and added analytic Solov'ev
  regression checks.

### v0.16.0 to v0.17.0 — physics and control breadth

- Added gyrokinetic, ballooning, current-diffusion, current-drive, NTM, RWM,
  sawtooth, SOL, integrated-scenario, NMPC, mu-synthesis, real-time EFIT,
  gain-scheduled, shape, safe-RL, sliding-mode, scheduler, and fault-tolerant
  modules.
- Added external interfaces for TGLF, GENE, GS2, CGYRO, and QuaLiKiz, plus
  native linear, TGLF-like, nonlinear delta-f, hybrid/OOD, and JAX gyrokinetic
  paths.
- Electromagnetic and nonlinear gyrokinetic surfaces remain
  research-prototype evidence. Their A-parallel, KBM, MTM, saturation, and
  cross-code results require external-code revalidation before quantitative
  promotion.

### v0.18.0 to v0.23.0 — admission contracts and runtime integration

- Added kinetic electrons, collision operators, deeper control-facing physics,
  native formal and runtime evidence contracts, geometry-neutral replay,
  CODAC/EPICS and WebSocket runtime evidence, FPGA HDL export, and strict
  external-artifact admission schemas.
- Added digest-bound claim reports for mu-synthesis, EFIT-lite, native runtime,
  HIL replay, controller safety cases, and generated capability/traceability
  inventories.
- The current Python package declares version `0.23.0`. No `v1.0.0` tag exists;
  this file does not imply a future release date or production-readiness state.

## Current evidence boundaries

- Repository synthetic fixtures validate parsers, numerical contracts, and
  deterministic replay plumbing; they do not constitute measured-shot evidence.
- SPARC public GEQDSK files are design references, not facility measurements.
- ITPA data on this surface support published-reference scaling comparisons,
  not new facility validation.
- Neural transport currently falls back to an analytic critical-gradient path
  unless admitted trained weights are supplied. Quantitative QLKNN/QuaLiKiz
  claims remain blocked by the public claim report.
- Neural-equilibrium pretraining has no admitted EFIT/P-EFIT holdout result;
  latency and accuracy promotion remain blocked.
- The phase Lyapunov monitor retains its research-prototype,
  warm-up and consecutive-window fail-open boundary and is not a standalone
  fail-closed safety interlock.
- Native gyrokinetic results require external-code revalidation before being
  treated as quantitative TGLF, GENE, GS2, CGYRO, or QuaLiKiz agreement.
- Loopback native-handoff timing is local-proxy evidence and is not a fielded
  PCS-cycle, HIL, or production-runtime admission.

## Public collaboration interfaces

The following GitHub issues describe external evidence that maintainers can
review. They are contribution interfaces, not an internal execution order.

- [#46](https://github.com/anulum/scpn-control/issues/46): umbrella external-validation evidence intake.
- [#47](https://github.com/anulum/scpn-control/issues/47): gyrokinetic and external-code evidence.
- [#48](https://github.com/anulum/scpn-control/issues/48): equilibrium and reconstruction references.
- [#49](https://github.com/anulum/scpn-control/issues/49): transport, edge, MHD, and scenario benchmarks.
- [#50](https://github.com/anulum/scpn-control/issues/50): neural-surrogate datasets and provenance.
- [#51](https://github.com/anulum/scpn-control/issues/51): plasma-control and facility-replay evidence.
- [#52](https://github.com/anulum/scpn-control/issues/52): disruption and mitigation benchmarks.
- [#53](https://github.com/anulum/scpn-control/issues/53): hardware, HDL, CODAC/EPICS, and runtime evidence.

Any contribution is evaluated against the public validation schemas and claim
boundaries. A failed comparison remains useful evidence; it does not become a
positive scientific, facility, or production claim.
