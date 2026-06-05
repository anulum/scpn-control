<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Native runtime validation report. -->
# Native formal verification mode benchmark

Generated: 2026-06-04T08:11:35.567338+00:00
Commit: `f527dcf`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:disabled:stride_30 | 1 | 0.796310 | 0.796310 | 99.204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:async_drop:stride_1 | 1 | 0.877766 | 0.877766 | 99.122 | 1000 | 3 | 1 | 997 | 0 | 0 | 0 |
| std:async_drop:stride_30 | 1 | 1.079858 | 1.079858 | 98.920 | 33 | 3 | 1 | 30 | 0 | 0 | 0 |
| std:sync_stride:stride_1 | 1 | 17309.901811 | 17309.901811 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 29920398 |
| std:sync_stride:stride_30 | 1 | 576.005471 | 576.005471 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 24777897 |
| std:aot_certificate:stride_1 | 1 | 1.117004 | 1.117004 | 98.883 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| std:aot_certificate:stride_30 | 1 | 0.835623 | 0.835623 | 99.164 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |
| io-uring:disabled:stride_30 | 1 | 0.870836 | 0.870836 | 99.129 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| io-uring:async_drop:stride_1 | 1 | 1.263338 | 1.263338 | 98.737 | 1000 | 2 | 0 | 998 | 0 | 0 | 0 |
| io-uring:async_drop:stride_30 | 1 | 1.133501 | 1.133501 | 98.866 | 33 | 3 | 1 | 30 | 0 | 0 | 0 |
| io-uring:sync_stride:stride_1 | 1 | 17959.457218 | 17959.457218 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 32573593 |
| io-uring:sync_stride:stride_30 | 1 | 551.501530 | 551.501530 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 29673104 |
| io-uring:aot_certificate:stride_1 | 1 | 0.887008 | 0.887008 | 99.113 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| io-uring:aot_certificate:stride_30 | 1 | 0.880913 | 0.880913 | 99.119 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
