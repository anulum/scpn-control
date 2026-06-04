<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Rust pulsed-shot MPC adapter benchmark report. -->

# Rust Pulsed MPC Adapter Benchmark

- Evidence class: `local_regression`
- Production claim allowed: `false`
- Steps: `2000`
- Warmup: `200`

## Results

| Case | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| `rust_burn_feasible` | 2000 | 45.128830 | 41.718000 | 64.106000 | 102.958000 |
| `rust_burn_infeasible_safe` | 2000 | 42.162976 | 38.995000 | 63.912000 | 91.700000 |
| `rust_non_burn_mask` | 2000 | 43.922270 | 39.369000 | 72.767000 | 101.519000 |

## Claim boundary

This is local regression evidence for the Rust pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `f4a0501aacd12d45acc4cce48b1c8eadb43d674404e58d003a4d495816f17fea`
