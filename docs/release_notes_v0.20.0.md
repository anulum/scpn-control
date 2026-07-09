# SCPN Control v0.20.0 Release Notes

v0.20.0 is a documentation, release-readiness, and validation-admission
milestone. It consolidates the recent hardening work into a clearer public
surface: what the software is, who it is for, what can be claimed today, and
which external artefacts are still required before full-fidelity facility claims
are admissible.

## Highlights

- Public landing, onboarding, use-case, tutorial, notebook, and API surfaces now
  explain the package as a controller-facing evidence layer rather than only a
  collection of modules.
- Version metadata is bumped to `0.20.0` for the release candidate.
- The capability manifest and README capability snapshot are regenerated from
  the current tree.
- The release notes keep the evidence boundary explicit: alpha/research package,
  no commissioned plant PCS claim, no predictive EFIT/P-EFIT claim without
  strict public-reference or facility artefact admission, and no target-hardware
  real-time claim without target-hardware timing evidence.
- The repository now has a clearer collaboration and financing positioning for
  GPU campaigns, public-data curation, external-code validation, and facility
  replay access.

## What users should read first

1. [README](https://github.com/anulum/scpn-control) for the product-level description.
2. [Onboarding](onboarding.md) for the first-hour path.
3. [Use Cases and Market Value](use_cases.md) for applications and commercial
   positioning.
4. [Production Readiness](production_readiness.md) for claim boundaries.
5. [Validation and QA](validation.md) for strict admission gates.

## Release checklist

Do not treat `v0.20.0` as published until all items are true:

- Version metadata and generated capability files are committed.
- Documentation builds with MkDocs strict mode.
- GitHub Actions for `main` pass after push.
- Security alerts and open pull requests are triaged.
- Safe failed or cancelled historical Actions/deployment records are deleted
  only after the current replacement evidence is green.
- The GitHub release is created from the `v0.20.0` tag.

## Practical use and scope

Use this record to identify baseline behavior at the 0.20.0 milestone.

- Use it during migration planning for downstream workflows.
- Treat it as historical context, then confirm current behavior in the latest `docs/changelog.md` and release notes.
- Keep operational assumptions explicit when comparing against older tags.
