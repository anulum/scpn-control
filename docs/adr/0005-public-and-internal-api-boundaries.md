# ADR-0005: Public and Internal API Boundaries

## Status

Accepted.

## Context

SCPN Control exposes package APIs, CLIs, public documentation, Studio
capability manifests, generated reports, internal TODOs, and coordination logs.
Those surfaces serve different audiences. Mixing them would leak planning
detail into public docs, turn internal quality targets into public claims, or
make users depend on unstable internals.

## Decision

Keep public and internal surfaces separate:

- Public package APIs live under `src/scpn_control/` exports, CLI commands,
  generated manifests, and public docs.
- Public APIs require strict typing, public API docstrings, module-specific
  tests, and documentation coverage.
- Internal planning, audits, TODOs, handovers, incident reports, and workflow
  control stay in ignored `docs/internal/` or monorepo `.coordination/`.
- Public docs describe current evidence, bounded claim status, setup, usage,
  architecture, and validation. They do not carry internal work-control detail.
- Studio UI and manifest surfaces expose verbs, evidence schemas, and claim
  boundaries, while auth, billing, and sealing custody stay with the portal and
  keeper.

## Alternatives Considered

- **Publish internal planning as public roadmap detail.** Rejected because it
  can confuse user-facing status with work-control state.
- **Keep all API detail private and rely on source reading.** Rejected because
  users need stable public docs, mkdocstrings references, and generated
  manifests.
- **Let Studio remotes own their own auth or evidence signing.** Rejected
  because identity, entitlement, billing, and sealing custody are shared Hub
  responsibilities.

## Consequences

- Internal TODO closure records commit hashes in ignored TODO files, not public
  release notes unless the change affects users.
- Commit-ready public changes need docs updates when user-visible behaviour,
  APIs, validation claims, or generated surfaces change.
- New public symbols need docstrings and API documentation coverage.
- Tests should exercise the public API, CLI, adapter, workflow, or file format
  that users and CI rely on.
- Public docs avoid self-applied promotion language and state evidence limits
  near the relevant claim.
