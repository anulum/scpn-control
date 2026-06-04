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
| `rust_burn_feasible` | 2000 | 36.221027 | 34.826000 | 41.296000 | 46.760000 |
| `rust_burn_infeasible_safe` | 2000 | 36.608832 | 35.049000 | 41.814000 | 48.730000 |
| `rust_non_burn_mask` | 2000 | 37.170229 | 36.843000 | 43.140000 | 49.000000 |

## Claim boundary

This is local regression evidence for the Rust pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `9b942f4d91703951e85f15a5fcb8c67a1bdffae2f1e8f2c319cc452b112c54e8`
