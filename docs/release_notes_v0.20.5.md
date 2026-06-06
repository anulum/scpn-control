<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — v0.20.5 release notes. -->
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

## Release checklist

Do not treat `v0.20.5` as published until all items are true:

- Version metadata and documentation updates are committed.
- Pre-commit, Docs Pages, CI, CodeQL, Scorecard, release, and publish workflows
  are green for the `v0.20.5` commit/tag.
- Pull requests and security alerts are triaged.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.20.5` tag.

## Practical use and scope

Use this record to understand release-time changes and fixes in 0.20.5.

- Use it for artifact comparison during replay and benchmark review.
- Confirm deployment assumptions with current runtime documentation.
- Preserve this as historical evidence and do not treat it as an active compliance statement.
