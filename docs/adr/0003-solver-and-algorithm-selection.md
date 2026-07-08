<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — ADR 0003 solver and algorithm selection. -->

# ADR-0003: Solver and Algorithm Selection

## Status

Accepted.

## Context

SCPN Control includes control algorithms, bounded physics facades, phase
dynamics, and validation scripts. These surfaces have different evidence levels:
some are exact closed-form validations, some are bounded local models, and some
are blocked until external data, external codes, or target hardware exist.

The repository must choose algorithms without overstating what those algorithms
prove.

## Decision

Select algorithms by contract and evidence tier:

- Use published or exact-reference algorithms when a module claims a named
  physics or control relation.
- Use structured numerical methods before dense hand-written kernels. Match the
  mathematical structure first, then use the appropriate primitive.
- Keep broad solver development in `scpn-fusion-core` or external reference
  codes. CONTROL ports or wraps only the subset that has a control-loop
  contract.
- Treat surrogate, heuristic, and local-regression paths as bounded unless
  matched external evidence admits the stronger claim.
- Preserve fail-closed admission when a solver backend is unavailable,
  non-convergent, out of tolerance, or missing provenance.

## Alternatives Considered

- **Choose the fastest implementation by default.** Rejected because latency
  without claim admission can produce misleading facility-facing conclusions.
- **Choose the most detailed solver in every local path.** Rejected because
  CONTROL is not the broad solver laboratory, and local control workflows need
  bounded facades with explicit assumptions.
- **Treat surrogate agreement as external validation.** Rejected because
  surrogate agreement is local evidence unless it is tied to admitted external
  references.

## Consequences

- Solver-facing modules need traceability rows that state references, units,
  validation evidence, and current claim status.
- Algorithm changes that affect physics or control claims require validation
  reports and public documentation updates.
- Benchmarks compare admitted surfaces only under declared backend and host
  conditions.
- External-code and real-data blockers remain visible instead of being hidden by
  local fixtures.
- New algorithm surfaces must document their assumptions, failure modes, and
  evidence boundaries before public claims can cite them.
