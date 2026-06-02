<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Control -->
<!-- Description: v0.20.2 release notes. -->

# SCPN Control v0.20.2 Release Notes

v0.20.2 is a documentation, release-readiness, security-boundary, and
evidence-admission polish release. It makes the public surface easier to read
while keeping quantitative physics, hardware, and deployment claims bounded by
the validators.

## What changed

- Reworked the README into clearer product, workflow, evidence, quick-start,
  and limitation sections so new readers can understand what SCPN Control is
  and what it is not.
- Added a public benchmark-regression admission gate for persisted latency
  evidence. The gate checks report digests, metric paths, thresholds, sample
  counts, hardware context, and claim-boundary text without generating new
  timing data.
- Hardened optional native C++ solver compilation with compiler admission,
  minimal build environment, symlink rejection, temporary build output, and
  atomic publication of regular shared-library files.
- Expanded the documentation around native-build security, benchmark admission,
  release boundaries, and collaboration value.
- Tightened public wording for gyrokinetic, real-time, disruption, and
  equilibrium claims where external-code, measured-shot, target-hardware, or
  peer-reviewed evidence remains missing.

## Evidence boundary

This release does not claim:

- commissioned plant PCS deployment;
- predictive EFIT/P-EFIT admission;
- external-code gyrokinetic agreement on identical inputs;
- saturated nonlinear Cyclone Base Case heat-flux agreement;
- target-hardware or HIL real-time PCS-cycle readiness;
- independent security audit completion.

Those claims require strict artefact admission before public promotion.

## Recommended reading order

1. [README](https://github.com/anulum/scpn-control) for the concise product
   overview and quick-start path.
2. [Onboarding](onboarding.md) for first-hour and first-day workflows.
3. [Use Cases and Market Value](use_cases.md) for who benefits and why.
4. [Production Readiness](production_readiness.md) for claim levels.
5. [Validation and QA](validation.md), [Benchmarks](benchmarks.md), and
   [Validation Summary](validation_summary.md) for current evidence.
6. [Compute Validation Funding](compute_validation_financing.md) for the open
   support and financing needs.

## Release checklist

Do not treat `v0.20.2` as published until all items are true:

- Version metadata and documentation updates are committed.
- Documentation builds successfully.
- Release-evidence and version-sync gates pass.
- The branch is pushed and GitHub Actions for `main` are green.
- Pull requests and security alerts are triaged.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.20.2` tag.
