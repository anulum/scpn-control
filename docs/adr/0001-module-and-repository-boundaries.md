# ADR-0001: Module and Repository Boundaries

## Status

Accepted.

## Context

SCPN Control sits between research notebooks, physics solvers, runtime control
interfaces, and validation reports. Without explicit boundaries the repository
would duplicate solver laboratories, mix runtime policy with experimental
physics, and blur which claims are supported by which evidence.

The codebase also participates in a sibling-repository ecosystem:

- `scpn-fusion-core` owns broad physics-solver development and external-code
  validation campaigns.
- `scpn-quantum-control` owns quantum execution and quantum phase-dynamics
  variants.
- `scpn-control` owns controller-facing facades, admission contracts, replay
  metadata, runtime safety boundaries, and public control APIs.

## Decision

Keep `scpn-control` as the controller-facing facade and admission repository.
Use package boundaries for responsibility boundaries:

- `scpn/` owns Petri-net structure, compiler, contracts, runtime safety
  certificates, geometry-neutral replay, and formal-verification bindings.
- `core/` owns bounded control-facing physics facades and plant models that the
  control layer can consume directly.
- `control/` owns controllers, runtime adapters, scheduler/campaign surfaces,
  digital-twin contracts, HIL-facing boundaries, and policy enforcement around
  controller actions.
- `phase/` owns Paper 27 phase dynamics, UPDE/Kuramoto runtime monitoring, and
  WebSocket phase-stream surfaces.
- `studio/` owns SCPN Studio capability, feed, evidence, and sealed-claim
  mapping for the Hub.
- `validation/` owns executable validation and benchmark admission scripts.
- `docs/` owns public user, architecture, validation, and release
  documentation. Internal planning stays in ignored `docs/internal/` or
  monorepo `.coordination/`.

Cross-repository work crosses boundaries by contract: units, shapes, digests,
version metadata, admission status, and failure modes must travel with the
surface. Shared mutable implementation files are not the integration mechanism.

## Alternatives Considered

- **Single repository for every SCPN surface.** Rejected because it would merge
  physics research, quantum execution, control APIs, and deployment policy into
  one change surface with unclear claim ownership.
- **CONTROL as a second full physics solver laboratory.** Rejected because it
  would duplicate `scpn-fusion-core` and create drift in solver mathematics.
- **Thin wrappers with no local validation.** Rejected because users need
  fail-closed admission, replay metadata, and bounded evidence at the control
  boundary, not only imports from upstream repositories.

## Consequences

- New features must land in the package that owns their operational contract.
- Solver mathematics that do not have a control-loop contract belong upstream or
  behind an explicit bounded facade.
- Public claims cite validation artefacts and traceability rows rather than
  architecture intent.
- Tests follow module ownership: each production module keeps a module-specific
  test surface, and cross-boundary features add integration or end-to-end tests.
- Documentation must state the boundary between implemented control surfaces and
  externally blocked facility, hardware, or solver claims.
