<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Multi-shot campaign benchmark report. -->

# Multi-Shot Campaign Benchmark

- Generated UTC: `2026-06-04T17:25:43Z`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- CPU affinity: `[4, 5]`
- Load average start: `(3.42529296875, 3.8447265625, 3.8291015625)`
- Load average end: `(3.6318359375, 3.880859375, 3.8408203125)`

- PyO3 status: `ok`

| Surface | Samples | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|---:|
| Python | 2000 | 154.037847 | 144.769000 | 196.827000 | 244.097000 |
| PyO3 | 2000 | 11.353325 | 10.621500 | 14.885000 | 21.042000 |

This is local regression evidence for the CONTROL multi-shot campaign adapter.
Each measured campaign carries two digest-bound pulsed-MPC admission references.
It is not target-hardware timing evidence and does not admit facility PCS claims.

Payload SHA-256: `85c7c6e8343bd3c4a0adbdcc69c3fa7f49d06dec52f2e50258ca1c99d3bafb1a`
