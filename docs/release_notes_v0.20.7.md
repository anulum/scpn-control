<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — v0.20.7 release notes. -->
# SCPN Control v0.20.7 Release Notes

v0.20.7 is a CON-C pulsed-control acceptance release on top of v0.20.6. It
packages the scheduler, capacitor-bank import, AER observation, replay v1.1, and
pulsed-MPC hardening needed by the sibling MIF workflow while keeping all
facility and production-timing claims fail-closed.

## What changed

- Added the CONTROL-owned pulsed scheduler liveness proof artefact in Lean and
  kept the SCPN-MIF-CORE import path stable through
  `scpn_control.control.pulsed_scenario_scheduler`.
- Added the `scpn_control.control.capacitor_bank` compatibility surface so MIF
  consumers can import the capacitor-bank state model through the contracted
  module path without a second implementation of the RLC equations.
- Hardened AER observation tests with generated-stream determinism and
  admission-latch invariants, with local PyO3 parity confirming the Rust decoder
  and spike-buffer bindings.
- Hardened geometry-neutral replay v1.1 acceptance with eight v1
  back-compatibility fixtures and byte-stable v1.1 serialisation checks.
- Hardened pulsed-shot MPC adapter acceptance with a ten-tick scheduler/action
  integration sequence that spans the full lifecycle and admits burn actions
  only during the burn state.
- Updated package, citation, API, README capability, release-note, changelog,
  and MkDocs navigation metadata to `0.20.7`.

## Evidence boundary

This release adds acceptance coverage and compatibility surfaces. It does not
upgrade workstation timing, public-data validation, target-hardware evidence,
PREEMPT_RT timing, external-code comparison, or facility PCS claims. Benchmark
reports remain governed by their recorded host-load and isolation metadata.

## Release checklist

Do not treat `v0.20.7` as published until all items are true:

- Version metadata and documentation updates are committed.
- CI, CodeQL, Pre-commit, OpenSSF Scorecard, Docs Pages, release, and publish
  workflows are green for the `v0.20.7` commit/tag.
- Pull requests and security alerts remain clear.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.20.7` tag.

## Practical use and scope

Use this latest historical note to verify what changed in the 0.20.7 line.

- Use it as a checkpoint before upgrading local scripts or benchmarks.
- Confirm any referenced claims with current validation and admission surfaces.
- Keep this page as historical context for reproducibility evidence.
