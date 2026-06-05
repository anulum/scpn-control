<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — v0.20.4 release notes. -->
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

## Release checklist

Do not treat `v0.20.4` as published until all items are true:

- Version metadata and documentation updates are committed.
- Documentation builds successfully.
- Generated capability manifest and traceability checks pass.
- Release-evidence and benchmark-context gates pass for the committed reports.
- The branch is pushed and GitHub Actions for `main` are green.
- Pull requests and security alerts are triaged.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.20.4` tag.
