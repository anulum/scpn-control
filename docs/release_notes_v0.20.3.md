<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Control -->
<!-- Description: v0.20.3 release notes. -->

# SCPN Control v0.20.3 Release Notes

v0.20.3 is a patch release for Lean formal-evidence admission security. It
keeps the v0.20.2 documentation and evidence-bound claim posture, then adds
the CodeQL ReDoS remediation to the release tag.

## What changed

- Replaced regex-based Lean theorem, Lean module, and safety-case identifier
  checks with linear-time validators.
- Reused the same validators in both Lean report admission and controller
  artifact admission so the security boundary stays single-source.
- Preserved duplicate rejection, non-empty list checks, proof-assumption
  digests, required Lean contract coverage, and manifest/report consistency
  checks.
- Bumped package, citation, archive, API, README capability, and release-note
  metadata to `0.20.3`.

## Evidence boundary

This release does not change the public physics or deployment claim boundary.
Predictive EFIT/P-EFIT, external-code gyrokinetic agreement, saturated nonlinear
Cyclone Base Case agreement, target-hardware/HIL real-time PCS-cycle operation,
commissioned plant deployment, and independent security-audit completion remain
blocked until their strict admission artefacts exist.

## Release checklist

Do not treat `v0.20.3` as published until all items are true:

- Version metadata and documentation updates are committed.
- Documentation builds successfully.
- Release-evidence and version-sync gates pass.
- The branch is pushed and GitHub Actions for `main` are green.
- CodeQL no longer reports the Lean identifier ReDoS alert on the release tag.
- Pull requests and security alerts are triaged.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.20.3` tag.

## Practical use and scope

Use this release note as a compatibility checkpoint for 0.20.3.

- Use it to validate command and runtime behavior against pre-existing configurations.
- Compare to current release notes before extending benchmarks or deployment claims.
- Keep this historical context attached to any script-level migration work.
