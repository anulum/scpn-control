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
6. [Compute and Validation Collaboration](compute_validation_financing.md) for
   public evidence interfaces and claim boundaries.

## Publication boundary

This page records source-level release history only. Tag, package-index, hosted-CI, security, and deployment status are external mutable state and must be verified at the corresponding service; this document does not expose or prescribe release operations.

## Practical use and scope

Use this record to trace behavior introduced in the 0.20.1 cycle.

- Use it for reproducibility checkpoints in long-running benchmarking history.
- Validate any references to this release against current runtime constraints.
- Use historical notes to inform safe back-porting decisions.
