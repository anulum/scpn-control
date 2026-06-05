<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — v0.20.6 release notes. -->
# SCPN Control v0.20.6 Release Notes

v0.20.6 is a CI, dependency, and repository-hygiene patch on top of v0.20.5.
It restores the remote coverage margin with module-specific tests, merges the
pending dependency-maintenance pull requests after fresh green checks, and
reconfirms that public security-alert surfaces are clear.

## What changed

- Added focused `rust_engine` coverage for native PyO3 handoff policy
  forwarding, fail-closed hardware-loop behaviour, ITPA gyro-Bohm coefficient
  loading, transport-network bounds, heartbeat timeout handling, and hybrid-loop
  telemetry accounting.
- Added focused quantum disruption bridge coverage for contract tamper
  detection, backend attestation, kernel/certificate evidence, feature mapping,
  and advisory decision validation.
- Merged the green Dependabot updates for `click`, `z3-solver`, and
  `hypothesis`.
- Refreshed and merged the stale Dependabot updates for pinned GitHub Actions
  and Rust `log` after their updated branches produced fresh green checks.
- Rechecked code scanning, Dependabot alerts, and secret scanning after the
  dependency lane; all three reported zero open alerts.
- Bumped package, citation, API, README capability, release-note, and public
  changelog metadata to `0.20.6`.

## Evidence boundary

This patch does not upgrade local-regression benchmark reports to production
benchmark evidence. It also does not change the blocked facility,
target-hardware, EFIT/P-EFIT, external-code, independent security-audit, or
PREEMPT_RT production-claim boundaries declared for the previous release line.

## Release checklist

Do not treat `v0.20.6` as published until all items are true:

- Version metadata and documentation updates are committed.
- CI, CodeQL, Pre-commit, OpenSSF Scorecard, Docs Pages, release, and publish
  workflows are green for the `v0.20.6` commit/tag.
- Pull requests and security alerts remain clear.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.20.6` tag.
