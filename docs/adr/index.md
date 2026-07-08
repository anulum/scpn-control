<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Architecture decision records index. -->

# Architecture Decision Records

This section records the standing architecture decisions that shape SCPN
Control. Each record names the decision, the reason for it, the alternatives
that were considered, and the consequences for code, tests, validation, and
public claims.

| ADR | Decision | Applies to |
| --- | --- | --- |
| [ADR-0001](0001-module-and-repository-boundaries.md) | Keep CONTROL as a controller-facing facade with explicit module and repository boundaries. | `src/scpn_control/{scpn,core,control,phase,studio}`, sibling SCPN repositories. |
| [ADR-0002](0002-python-rust-pyo3-dispatch.md) | Use Python for orchestration and Rust/PyO3 for bounded hot paths. | Python package, `scpn-control-rs`, optional native dispatch. |
| [ADR-0003](0003-solver-and-algorithm-selection.md) | Select algorithms by evidence tier and keep broad solver development upstream or external. | Physics facades, control solvers, validation reports. |
| [ADR-0004](0004-validation-data-and-evidence-strategy.md) | Treat validation artefacts as claim-admission inputs, not supporting prose. | `validation/`, `docs/physics_traceability.md`, generated reports. |
| [ADR-0005](0005-public-and-internal-api-boundaries.md) | Separate public user APIs from internal planning, coordination, and claim-governance surfaces. | Package exports, CLI, docs, Studio surfaces, internal TODOs. |

The ADRs are public documentation. Internal roadmap targets, coordination notes,
and unresolved TODO detail remain in ignored `docs/internal/` and monorepo
`.coordination/` surfaces.
