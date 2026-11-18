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
| `rust_burn_feasible` | 2000 | 163.039883 | 184.810000 | 211.423000 | 217.758000 |
| `rust_burn_infeasible_safe` | 2000 | 140.008289 | 109.371000 | 203.653000 | 213.824000 |
| `rust_non_burn_mask` | 2000 | 176.275159 | 196.851000 | 214.611000 | 223.942000 |

## Claim boundary

This is local regression evidence for the Rust pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `7badf5219feb03cd7201b5c1e7d57b1477dadf874b2b6523cfc6fbad6e345b60`
