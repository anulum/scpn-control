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
| `rust_burn_feasible` | 2000 | 134.690865 | 123.796000 | 192.042000 | 212.926000 |
| `rust_burn_infeasible_safe` | 2000 | 123.929276 | 115.712000 | 167.849000 | 198.624000 |
| `rust_non_burn_mask` | 2000 | 155.538215 | 131.372000 | 222.741000 | 232.143000 |

## Claim boundary

This is local regression evidence for the Rust pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `ad7e576fb566995a9b4d52ae2a60adb15aac3cfd18d60a90b306c73361689e8b`
