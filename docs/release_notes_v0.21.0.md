<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — v0.21.0 release notes. -->
# SCPN Control v0.21.0 Release Notes

v0.21.0 is a physics- and control-validation release on top of v0.20.7. Twenty
control and physics models gained tests that check their outputs against exact
closed-form, analytic, eigenvalue, or conservation-law references, and the
differentiable-transport facade reached near-complete statement coverage. The
release keeps every facility, target-hardware, external-code, and
production-timing claim fail-closed.

## What changed

### Closed-form and analytic validation campaign

Each model below now has a dedicated test that compares its output against an
independently derived exact reference rather than a self-consistency fixture:

- Kuramoto phase-synchronisation runtime against published synchronisation
  results.
- Grad-Shafranov equilibrium solver against the Solov'ev exact equilibrium.
- Structured-singular-value (mu) computation against exact mu identities.
- Guiding-centre orbit integrator against conservation laws.
- Transport heat-diffusion solver against analytic results.
- Neoclassical tearing-mode Modified Rutherford Equation against exact
  references.
- RZIP rigid vertical stability against exact eigenvalues.
- Resistive-wall-mode feedback against exact closed forms.
- Kadomtsev sawtooth crash against exact conservation laws.
- Two-point scrape-off-layer model against exact closed forms.
- Auxiliary current-drive model against exact closed forms.
- Ideal-MHD stability metrics against exact closed forms.
- EPED pedestal model against exact construction relations.
- ELM peeling-ballooning boundary and crash against exact forms.
- Toroidal momentum transport against exact closed forms.
- Runaway-electron avalanche model against exact closed forms.
- Fitzpatrick halo-current L/R circuit against exact closed forms.
- Volt-second flux budget against exact closed forms.
- Density-control particle balance against exact closed forms.
- DT burn-control alpha-heating algebra against exact closed forms.

### Differentiable-transport facade coverage

- Raised statement coverage of the differentiable-transport facade to 99.5%
  with module-specific tests for input-validation contracts, closure-adapter
  rejection paths, campaign-metadata mapping, gradient-audit validators,
  differentiability and full-fidelity readiness evidence guards, latency and
  runtime-metadata validators, equilibrium radial weighting, JAX-unavailable
  fail-closed contracts, and JAX compute paths.

### Typing, hardening, dependencies

- Enforced strict mypy typing for the admission, configuration, current-drive,
  and real-time EFIT modules; restored ruff lint and format compliance.
- Hardened the momentum-damping transport contract and capacitor-bank energy
  admission, and resealed multi-shot campaign evidence payload digests.
- Cleared a Rust advisory (pyo3/numpy 0.25 to 0.29) and bumped numpy, osqp,
  tornado, hypothesis, pip-audit, ruff, sha2, socket2, and codecov-action.
- Updated package, citation, Zenodo, API, changelog, release-note, and MkDocs
  navigation metadata to `0.21.0`.

## Evidence boundary

This release strengthens repository-level validation evidence. A passing
closed-form test means the model reproduces its analytic reference inside the
repository contract. It does not upgrade workstation timing, public measured-shot
validation, external-code (GENE, TGLF, GS2, CGYRO, QuaLiKiz) comparison,
target-hardware or PREEMPT_RT timing, or facility PCS claims. Benchmark reports
remain governed by their recorded host-load and isolation metadata.

## Release checklist

Do not treat `v0.21.0` as published until all items are true:

- Version metadata and documentation updates are committed.
- CI, CodeQL, Pre-commit, OpenSSF Scorecard, Docs Pages, release, and publish
  workflows are green for the `v0.21.0` commit/tag.
- Pull requests and security alerts remain clear.
- Failed or cancelled Actions/deployment records are deleted only when safe and
  only after replacement evidence is green.
- The GitHub release is created from the `v0.21.0` tag.

## Practical use and scope

Use this note to verify what changed in the 0.21.0 line.

- Use it as a checkpoint before upgrading local scripts or benchmarks.
- Confirm any referenced claims with current validation and admission surfaces.
- Keep this page as historical context for reproducibility evidence.
