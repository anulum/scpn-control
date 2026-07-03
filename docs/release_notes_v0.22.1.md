<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — v0.22.1 release notes. -->
# SCPN Control v0.22.1 Release Notes

v0.22.1 is a patch release for the v0.22 control-evidence line. It hardens
disruption replay statistics, preserves the RZIP vertical-feedback contract
under local SciPy/NumPy Riccati validation failures, refreshes generated
capability metadata, and closes the dependency-maintenance loop with green CI,
benchmark, and security evidence.

## What changed

### Numerical robustness

- Replaced scalar NumPy percentile calls in the disruption mitigation replay
  path with deterministic linear interpolation. This prevents local NumPy reload
  failures in SPI diagnostics and halo/runaway post-disruption summaries while
  preserving bounded replay output semantics.
- Hardened `RZIPController` so SciPy/NumPy Riccati validation failures fall back
  to a bounded NumPy discrete-Riccati gain before the existing zero-gain
  fail-closed path.

### Dependency and CI maintenance

- Merged the clean Dependabot updates for Rust crates, GitHub Actions pins,
  ESLint, and Prettier.
- Consolidated the remaining studio-web updates for Vite,
  `@module-federation/vite`, and TypeScript ESLint on top of the green main
  branch after the individual PR branches either failed CI or conflicted.
- Confirmed the combined studio-web surface with install, typecheck, lint,
  format check, tests with coverage, and Module Federation build gates.

### Release evidence

- Refreshed generated capability inventory metadata for the current source,
  test, and documentation inventory.
- Confirmed open Dependabot, code-scanning, and secret-scanning alert counts
  were zero before release.
- Purged failed and cancelled Actions history only after replacement `main` CI
  and benchmark evidence completed successfully.

## Evidence boundary

This release does not certify facility deployment, PREEMPT_RT operation,
external-code equivalence, P-EFIT predictive validity, TORAX parity, or plant
PCS readiness. The release evidence remains a bounded software, CI, dependency,
and local validation checkpoint.

## Release checklist

Do not treat `v0.22.1` as published until all items are true:

- Version metadata and documentation updates are committed.
- CI, CodeQL, Pre-commit, OpenSSF Scorecard, benchmark, and release workflows
  are green for the `v0.22.1` commit/tag.
- Pull requests and security alerts remain clear.
- Failed or cancelled Actions records are absent from the last 1000 runs after
  replacement evidence is green.
- The GitHub release is created from the `v0.22.1` tag.

## Practical use and scope

Use this note to verify what changed in the 0.22.1 line.

- Upgrade from `0.22.0` when local disruption replay or RZIP validation paths
  run under newer NumPy/SciPy stacks.
- Keep broader facility claims blocked unless their strict admission gates
  accept matching external, measured-shot, target-hardware, and review evidence.
