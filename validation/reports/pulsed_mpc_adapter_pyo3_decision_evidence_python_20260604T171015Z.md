<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Pulsed-shot MPC adapter benchmark report. -->

# Pulsed MPC Adapter Benchmark

- Generated UTC: `2026-06-04T17:10:22Z`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- Steps: `2000`
- Warmup: `200`
- CPU affinity: `[4, 5]`
- Load average start: `(2.666015625, 4.2900390625, 3.8818359375)`
- Load average end: `(3.09326171875, 4.35205078125, 3.904296875)`
- PyO3 status: `available`

## Results

| Case | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| `python_non_burn_mask` | 2000 | 884.832562 | 857.219000 | 1042.758000 | 1186.337000 |
| `python_burn_feasible` | 2000 | 877.166691 | 852.062000 | 1004.590000 | 1145.755000 |
| `python_burn_infeasible_safe` | 2000 | 902.237464 | 873.722500 | 1058.292000 | 1264.718000 |
| `pyo3_non_burn_mask` | 2000 | 41.979006 | 40.112000 | 48.898000 | 57.746000 |
| `pyo3_burn_infeasible_safe` | 2000 | 43.093626 | 40.359500 | 54.422000 | 67.074000 |

## Claim boundary

This is local regression evidence for the pulsed MPC admission adapter.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `dac7ab25a0d48d67906696851a37b550ca55876857dca0b37745262e4f0b0d9f`
