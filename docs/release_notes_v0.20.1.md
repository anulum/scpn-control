<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Control — v0.20.1 release notes
-->

# SCPN Control v0.20.1 Release Notes

v0.20.1 is a documentation, evidence-admission, and repository-polish release
candidate. It packages the post-v0.20.0 hardening work into a clearer public
surface for users, collaborators, reviewers, and funders.

## Highlights

- The public landing, README, onboarding, tutorial, notebook, use-case,
  production-readiness, and financing pages now explain what SCPN Control is:
  a controller-facing evidence layer for fusion software.
- The documentation now states the market and collaboration value in concrete
  terms: controller concept review, bounded formal safety evidence,
  differentiable controller tuning, public-data validation, target-hardware
  timing, and local or air-gapped physics debugging.
- MAST EFM neural-equilibrium training remains correctly fail-closed until the
  storage-host payload is available on an admitted compute host and executed full-output
  training, holdout, latency, GPU-cost, and strict reference-admission artefacts
  exist.
- JAX gyrokinetic CPU/GPU parity evidence is published with aggregate digests
  and separate local CPU benchmark timing reports while preserving the
  backend-parity-only claim boundary.
- MkDocs navigation now exposes additional public guides so readers can find
  deployment, FAQ, physics-methods, validation-summary, validation-deficiency,
  and neural-transport training pages from the site navigation.

## Evidence boundary

This release candidate does not claim commissioned plant deployment, predictive
EFIT/P-EFIT admission, full external-code gyrokinetic validation, or target
hardware real-time readiness. Those claims remain blocked until the matching
strict validators admit the required external artefacts.

## Recommended reading order

1. [README](https://github.com/anulum/scpn-control) for the product summary.
2. [Onboarding](onboarding.md) for the first-hour and first-day workflows.
3. [Use Cases and Market Value](use_cases.md) for application and collaboration
   context.
4. [Production Readiness](production_readiness.md) for the allowed claim levels.
5. [Validation and QA](validation.md) and [Benchmarks](benchmarks.md) for the
   current evidence reports.
6. [Compute Validation Funding](compute_validation_financing.md) for the active
   data, GPU, external-code, and hardware support request.

## Release checklist

Do not treat `v0.20.1` as published until all items are true:

- Version metadata and generated capability files are committed.
- Documentation builds with MkDocs strict mode.
- Local release-evidence and policy gates pass.
- The branch is pushed and GitHub Actions for `main` are green.
- Open pull requests and security alerts are triaged.
- Historical failed or cancelled Actions/deployment records are deleted only
  when safe and only after replacement evidence is green.
- The GitHub release is created from the `v0.20.1` tag.

## Practical use and scope

Use this record to trace behavior introduced in the 0.20.1 cycle.

- Use it for reproducibility checkpoints in long-running benchmarking history.
- Validate any references to this release against current runtime constraints.
- Use historical notes to inform safe back-porting decisions.
