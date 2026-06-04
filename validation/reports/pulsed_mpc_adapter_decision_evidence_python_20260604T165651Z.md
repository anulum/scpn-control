<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Pulsed-shot MPC adapter benchmark report. -->

# Pulsed MPC Adapter Benchmark

- Generated UTC: `2026-06-04T16:56:59Z`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- Steps: `2000`
- Warmup: `200`
- CPU affinity: `[4, 5]`
- Load average start: `(2.65234375, 2.8369140625, 3.0556640625)`
- Load average end: `(2.79248046875, 2.859375, 3.060546875)`
- PyO3 status: `optional PyO3 extension is installed but was not rebuilt with pulsed decision evidence`

## Results

| Case | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| `python_non_burn_mask` | 2000 | 925.102288 | 900.629000 | 1121.022000 | 1256.894000 |
| `python_burn_feasible` | 2000 | 955.621603 | 913.227500 | 1210.088000 | 1421.172000 |
| `python_burn_infeasible_safe` | 2000 | 1152.562707 | 1104.886000 | 1565.198000 | 1815.799000 |

## Claim boundary

This is local regression evidence for the pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `c74cea4cad7d204ebcd5f96cec5243421eb1aba168e73e9b4fc205cff95b2019`
