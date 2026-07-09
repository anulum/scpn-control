# SCPN Control v0.19.3 Release Notes

Release status: candidate pending the current `main` GitHub Actions run after
push. Create the release tag only after the remote checks for the intended
release commit are successful.

## Scope

v0.19.3 is a validation-boundary and repository-polish patch release. It folds
in the recent physics, mathematics, reproducibility, and security hardening
series while preserving the public claim boundary: local controller-readiness
and differentiability evidence remains bounded unless the corresponding
external artefacts, facility replay data, target-hardware timing, and formal
proof digests are present and admitted by the strict validators.

## Included hardening areas

- Differentiable-transport full-fidelity readiness evidence now binds campaign
  metadata, one-step and rollout latency reports, gradient-audit digests,
  controller formal-proof digests, equilibrium-coupled metadata, and admitted
  external reference artefacts before a full-fidelity claim can pass.
- The differentiable-transport benchmark now publishes a separate readiness
  artefact in addition to the one-step and rollout latency reports.
- The differentiable-transport latency validator now structurally admits the
  readiness artefact while keeping `full_fidelity_ready` false when controller
  proof or external-reference evidence is missing.
- Documentation, API reference text, benchmark guidance, validation guidance,
  notebook catalogue text, generated capability metadata, and README capability
  snapshot have been refreshed for the v0.19.3 candidate.

## Required release gate

Do not create or move the `v0.19.3` tag until all items are true:

- The intended release commit is on `main` and pushed to `origin`.
- Required GitHub Actions checks for that commit are successful.
- Documentation pages build with `mkdocs build --strict`.
- `python tools/capability_manifest.py --check` succeeds.
- `python validation/validate_differentiable_transport_latency.py --require-admitted --json-out` succeeds and reports `full_fidelity_ready=false` unless external-reference and controller-proof evidence are supplied.
- No public documentation promotes full-fidelity or facility-validation claims
  for entries still blocked by traceability.

## Residual validation debt

The release keeps the current evidence boundary explicit:

- Full-fidelity public claims remain blocked for entries without required
  external artefacts.
- Differentiable transport remains a controller-facing validated facade until
  measured-discharge, integrated-modelling, or external-code artefacts are
  admitted by the relevant strict gates.
- Target-hardware latency and deployment readiness claims require qualified HIL,
  CODAC/EPICS/WebSocket runtime, and safety-case artefacts rather than local
  workstation timing alone.

## Practical use and scope

Use this historical release note to review feature and interface changes in that release window.

- Keep historical behavior distinctions when reproducing legacy campaigns.
- Cross-check with migration notes before changing dependent scripts.
- Pair with current release notes before claiming present-day capabilities.
