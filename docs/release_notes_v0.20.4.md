# SCPN Control v0.20.4 Release Notes

v0.20.4 is a native-runtime evidence, formal-certificate, documentation, and
repository-polish release candidate. It packages the post-v0.20.3 native
execution work without upgrading any facility or target-hardware claim.

## What changed

- Bumped package, citation, API, README capability, and release-note metadata
  to `0.20.4`.
- Published the native handoff, formal-mode, AOT certificate, and spin-pacing
  report families as local-regression evidence under `validation/reports/`.
- Documented the difference between Python orchestration, fused Rust/PyO3
  execution, Rust-owned Z3 stride verification, asynchronous proof sampling,
  and compiled AOT certificate monitoring.
- Added reader-path documentation for native runtime evidence, benchmark
  interpretation, notebook onboarding, and deployment boundaries.
- Updated benchmark and validation summaries so non-isolated workstation timing
  cannot be promoted to production benchmark evidence.
- Corrected touched public metadata/documentation headers to the repository
  header form and removed the deprecated two-field project/purpose header form
  from touched files.

## Evidence boundary

This release does not claim commissioned plant PCS operation, target-hardware
control-cycle timing, PREEMPT_RT acceptance, external-code gyrokinetic agreement,
predictive EFIT/P-EFIT validation, saturated nonlinear Cyclone Base Case
agreement, or independent security-audit completion.

The committed native timing reports are local-regression evidence unless a
matching report explicitly records production benchmark context and passes the
release evidence validator. The AOT certificate lane is an admitted hot-path
monitor for the declared Petri-net certificate assumptions; it is not a live SMT
solver in the control loop.

## Publication boundary

This page records source-level release history only. Tag, package-index, hosted-CI, security, and deployment status are external mutable state and must be verified at the corresponding service; this document does not expose or prescribe release operations.

## Practical use and scope

Use this historical note as a baseline for what changed in 0.20.4.

- Use it to confirm feature provenance and avoid regressing known behavior.
- Keep this in the historical context when reviewing long-lived automation outputs.
- Use alongside newer release notes for present-time claims.
