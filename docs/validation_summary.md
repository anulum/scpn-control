<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Validation evidence summary. -->
# Validation Summary — scpn-control

This document summarises current validation evidence and its claim boundary.
It separates bounded repository evidence from external-code, measured-shot,
target-hardware, and deployment evidence that still needs admission.

| Claim | Evidence | Script | Result |
|-------|----------|--------|--------|
| GS solver converges | Solov'ev analytic benchmark | `test_p0_regression.py` | NRMSE < 1% |
| GS solver accuracy | Mesh convergence study | `mesh_convergence_study.py` | 2nd Order ($O(h^2)$) |
| Transport scaling sanity | IPB98(y,2)-style scaling checks | validation tooling | Bounded regression evidence |
| H-inf outperforms PID | Controller comparison | `controller_comparison.py` | 30% reward improvement |
| PPO/RL research baseline | Seeded training and comparison reports | RL benchmark tooling | Bounded research evidence |
| Kernel latency | Rust kernel benchmark | Criterion and benchmark reports | 11.9 µs P50 bare kernel |
| Disruption prediction | Synthetic ROC analysis | disruption benchmark tooling | Synthetic-only evidence |
| Physical Consistency | Energy balance diagnostic | `benchmark_transport.py` | Error < 1% (Internal) |
| Native formal AOT certificate monitor | Digest-bound local-regression reports | `validation/validate_native_formal_certificate_evidence.py` | Admitted only inside declared benchmark context |

## Key Benchmarks

### 1. Equilibrium Accuracy
The Grad-Shafranov solver was benchmarked against the Solov'ev analytic solution.
A mesh convergence study confirmed that the 5-point central difference stencil
achieves the theoretical second-order spatial convergence rate.

### 2. Transport Fidelity
The 1.5D transport solver includes regression checks against confinement-scaling
contracts and internal diagnostics. Treat these as bounded repository evidence,
not as a replacement for measured-shot or external integrated-modelling
validation.

### 3. Control Performance
The control stack includes deterministic controller comparisons, stress tests,
and safety-bound checks. Treat learning-controller comparisons as research
baselines unless matched HIL, target-hardware, and measured-shot evidence exists.

### 4. Real-Time Latency
The published 11.9 us figure is a bare Rust kernel measurement. It is not an
end-to-end PCS-cycle claim. Deployment timing needs target hardware, IO,
diagnostics, actuator, queue/backpressure, and HIL replay evidence.

### 5. Native Runtime Formal Evidence

The native runtime lane now distinguishes proof sampling from strict formal
coverage. `async_drop` is diagnostic sampling and may drop saturated snapshots.
`sync_stride` measures the cost of waiting for a Rust-owned Z3 worker on selected
steps. `aot_certificate` keeps the hot path out of the SMT solver and checks a
digest-bound certificate monitor at runtime. Current workstation reports are
local-regression evidence unless the benchmark context records production-grade
core isolation, host-load, governor, runtime, and concurrent-job metadata.

## How to read this evidence summary in proposals

This page is a compact index for claim status only; it is not a substitute for the linked detailed report.

For each row, use:

- the listed script to reproduce the result,
- the evidence type (`local-regression` vs `admitted`) to decide scope,
- the strict validator gate for cross-publication or partner-facing use.

When proposing external validation, include the script, report file, and a short admission note for each claim.
