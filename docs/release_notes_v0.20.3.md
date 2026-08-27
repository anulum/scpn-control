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

## Publication boundary

This page records source-level release history only. Tag, package-index, hosted-CI, security, and deployment status are external mutable state and must be verified at the corresponding service; this document does not expose or prescribe release operations.

## Practical use and scope

Use this release note as a compatibility checkpoint for 0.20.3.

- Use it to validate command and runtime behavior against pre-existing configurations.
- Compare to current release notes before extending benchmarks or deployment claims.
- Keep this historical context attached to any script-level migration work.
