<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — ADR 0002 Python Rust PyO3 dispatch. -->

# ADR-0002: Python, Rust, and PyO3 Dispatch

## Status

Accepted.

## Context

The repository needs two different execution qualities at once:

- Python gives researchers and control engineers fast iteration, broad
  scientific-library access, readable validation scripts, and ergonomic CLI and
  documentation workflows.
- Rust gives bounded hot paths, explicit ownership, deterministic data layouts,
  and fuzzable parser/numeric surfaces.
- PyO3 connects the two without forcing every user to build native extensions
  for every workflow.

## Decision

Use Python as the orchestration and public package layer. Use Rust for hot paths
or boundary surfaces where determinism, parser safety, or measured runtime
evidence matters. Expose Rust through optional PyO3 bindings, and keep the
pure-Python path complete unless the feature is explicitly native-only.

Dispatch follows this order:

1. Validate inputs and policy in Python at the public API or CLI boundary.
2. Use a native Rust/PyO3 path when the binding is installed and the feature has
   parity evidence for the requested operation.
3. Fall back to the Python path when the native extension is absent and the
   Python path has the same admitted contract.
4. Fail closed when the native path is mandatory for the claim being requested.

## Alternatives Considered

- **Python only.** Rejected because it cannot cover every parser, fuzzing,
  latency, and deterministic hot-path requirement.
- **Rust only.** Rejected because it would slow scientific iteration, validation
  report generation, and integration with Python-native scientific libraries.
- **Always prefer native code when importable.** Rejected because an installed
  native module is not evidence by itself; dispatch must depend on parity,
  admission status, and the requested claim boundary.

## Consequences

- Every polyglot surface needs parity tests at the boundary it exposes.
- Benchmarks must identify backend, hardware, dtype, isolation context, and
  whether the result supports only local regression or a stronger claim.
- Optional native imports must have typed sentinels and fail-closed paths.
- Rust changes that affect Python-visible behaviour need PyO3 tests and public
  documentation updates.
- Python code remains the source of high-level user contracts; Rust implements
  selected execution kernels and boundary adapters.
