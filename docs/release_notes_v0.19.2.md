<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control - v0.19.2 release notes -->

# SCPN Control v0.19.2 Release Notes

Release status: candidate with current `main` GitHub Actions green. Create the
release tag only after a final tag-intent review confirms the same gates remain
green.

## Scope

v0.19.2 is a boundary-hardening patch release. It prepares the repository for a
tag after the recent physics and mathematics hardening series by documenting the
release scope, version metadata, validation gates, and residual evidence gaps.

The release does not expand public full-fidelity claims. The physics
traceability registry remains the authority for claim status, and unresolved
external-evidence requirements remain blocked.

## Included hardening areas

- MHD and stability boundary contracts.
- Pedestal, ELM, SOL, MARFE, and detachment edge-model boundaries.
- L-H transition, momentum transport, neoclassical transport, and blob
  transport contracts.
- Orbit-following, integrated scenario, integrated transport, uncertainty, and
  pellet-injection error boundaries.
- Differentiable transport campaign provenance, schema-versioned metadata
  persistence, and replay-drift rejection before controller-tuning reruns.
- JAX gyrokinetic stiffness-closure monotonicity under the CI JAX backend,
  retained as a bounded controller-tuning surrogate rather than a quantitative
  transport claim.
- Formatter alignment required by the remote pre-commit workflow.
- Traceability summary checks derive open-gap counts from the live registry, so
  the roadmap, report generator, and documentation tests track current evidence
  state.

## Test and policy posture

- Tests remain module-specific by default.
- No coverage-bucket, batch, round, final, remaining, push, miscellaneous, or
  new-module test files are part of the release scope.
- Assertions are retained only where they check behavioural, numerical,
  adapter, serialisation, or integration contracts.
- Optional external runtime dependencies are mocked only at adapter and
  unavailable-dependency boundaries.

## Required release gate

Do not create or move the `v0.19.2` tag until all items are true:

- The intended release commit is on `main`.
- Required GitHub Actions checks for that commit are successful.
- `pre-commit run --all-files` succeeds locally.
- `python tools/check_test_quality_policy.py` succeeds locally.
- `python validation/validate_physics_traceability.py --registry validation/physics_traceability.json --json-out` succeeds locally.
- `python tools/capability_manifest.py --check` succeeds locally.
- `python -m tools.check_generated_traceability` succeeds locally.
- No public documentation promotes full-fidelity or facility-validation claims
  for entries still blocked by traceability.

## Residual validation debt

The release keeps the current evidence boundary explicit:

- Physics traceability still reports open external-evidence gaps.
- Full-fidelity public claims remain blocked for entries without required
  external artefacts.
- Differentiable transport and JAX GK stiffness changes remain bounded
  controller-tuning and numerical-contract surfaces until external GK,
  integrated-modelling, or facility replay artefacts validate quantitative
  transport claims.
- External materials, facility artefacts, and benchmark provenance should remain
  tracked through repository issues before future claim expansion.
