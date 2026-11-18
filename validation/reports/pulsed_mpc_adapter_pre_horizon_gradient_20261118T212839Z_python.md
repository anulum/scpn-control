<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Pulsed-shot MPC adapter benchmark report. -->

# Pulsed MPC Adapter Benchmark

- Generated UTC: `2026-11-18T21:29:35Z`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- Steps: `2000`
- Warmup: `200`
- CPU affinity: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]`
- Load average start: `(59.16796875, 63.0703125, 62.544921875)`
- Load average end: `(58.44287109375, 62.35107421875, 62.33642578125)`
- PyO3 status: `available`

## Results

| Case | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| `python_non_burn_mask` | 2000 | 9185.139134 | 9213.672500 | 10110.641000 | 10389.712000 |
| `python_burn_feasible` | 2000 | 8195.884690 | 8529.587000 | 9930.470000 | 10142.777000 |
| `python_burn_infeasible_safe` | 2000 | 6568.342199 | 6193.974000 | 9696.990000 | 10309.363000 |
| `pyo3_non_burn_mask` | 2000 | 228.798002 | 246.568000 | 263.445000 | 484.818000 |
| `pyo3_burn_infeasible_safe` | 2000 | 180.290361 | 144.142000 | 253.421000 | 262.610000 |

## Claim boundary

This is local regression evidence for the pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `a4a4464b138b63d21aeeb1f825be2d7b16c14ff39c084da678cc96cd9a45e2d6`
