<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Pulsed-shot MPC adapter benchmark report. -->

# Pulsed MPC Adapter Benchmark

- Generated UTC: `2026-11-18T21:33:36Z`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- Steps: `2000`
- Warmup: `200`
- CPU affinity: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]`
- Load average start: `(57.5615234375, 59.58154296875, 61.20166015625)`
- Load average end: `(55.0439453125, 58.6279296875, 60.80810546875)`
- PyO3 status: `available`

## Results

| Case | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| `python_non_burn_mask` | 2000 | 6060.920789 | 5823.123000 | 8031.361000 | 12406.670000 |
| `python_burn_feasible` | 2000 | 6074.169961 | 5525.801000 | 8605.762000 | 11473.322000 |
| `python_burn_infeasible_safe` | 2000 | 6388.856813 | 5610.290000 | 11689.981000 | 12529.719000 |
| `pyo3_non_burn_mask` | 2000 | 145.827240 | 137.756000 | 172.609000 | 224.850000 |
| `pyo3_burn_infeasible_safe` | 2000 | 151.216386 | 138.894500 | 218.747000 | 267.213000 |

## Claim boundary

This is local regression evidence for the pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `44b69c6c832448d195d3aad872fc53bf10b99cffc77c8c7ddd7c173c24d43226`
