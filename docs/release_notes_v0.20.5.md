# SCPN Control v0.20.5 Release Notes

v0.20.5 is a release-hygiene patch on top of v0.20.4. It fixes the remote
pre-commit and Docs Pages failures found after the v0.20.4 tag by documenting
the native Rust engine wrapper, preserving the AER domain acronym in the typo
configuration, synchronising archive metadata, and keeping internal documentation
excluded from the GitHub Pages build.

## What changed

- Bumped package, citation, archive, API, README capability, and release-note
  metadata to `0.20.5`.
- Added API documentation coverage for `scpn_control.core.rust_engine`.
- Added `AER/aer` to the typo configuration so the neuromorphic
  Address-Event Representation acronym is not rewritten.
- Fixed the `scpn_control.scpn.observation` NumPy return annotation path under
  strict mypy pre-commit checks.
- Kept the internal documentation tree excluded from the MkDocs/GitHub Pages build.
- Applied tracked validation-report newline and trailing-whitespace fixes from
  the pre-commit hooks.

## Evidence boundary

This patch changes release hygiene only. It does not upgrade local-regression
benchmark reports to production benchmark evidence, and it does not change the
blocked facility, target-hardware, EFIT/P-EFIT, external-code, or independent
security-audit claim boundaries declared for v0.20.4.

## Publication boundary

This page records source-level release history only. Tag, package-index, hosted-CI, security, and deployment status are external mutable state and must be verified at the corresponding service; this document does not expose or prescribe release operations.

## Practical use and scope

Use this record to understand release-time changes and fixes in 0.20.5.

- Use it for artifact comparison during replay and benchmark review.
- Confirm deployment assumptions with current runtime documentation.
- Preserve this as historical evidence and do not treat it as an active compliance statement.
