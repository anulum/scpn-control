# Native formal verification mode benchmark

Generated: 2026-06-04T08:12:27.621537+00:00
Commit: `f527dcf`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:disabled:stride_30 | 1 | 2.040826 | 2.040826 | 97.959 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:async_drop:stride_1 | 1 | 2.327547 | 2.327547 | 97.672 | 1000 | 11 | 9 | 989 | 0 | 0 | 0 |
| std:async_drop:stride_30 | 1 | 2.847419 | 2.847419 | 97.153 | 33 | 11 | 9 | 22 | 0 | 0 | 0 |
| std:sync_stride:stride_1 | 1 | 17404.260422 | 17404.260422 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 28934451 |
| std:sync_stride:stride_30 | 1 | 588.326933 | 588.326933 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 34017282 |
| std:aot_certificate:stride_1 | 1 | 2.292474 | 2.292474 | 97.708 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| std:aot_certificate:stride_30 | 1 | 1.400451 | 1.400451 | 98.600 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |
| io-uring:disabled:stride_30 | 1 | 1.429354 | 1.429354 | 98.571 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| io-uring:async_drop:stride_1 | 1 | 2.692025 | 2.692025 | 97.308 | 1000 | 11 | 9 | 989 | 0 | 0 | 0 |
| io-uring:async_drop:stride_30 | 1 | 2.590399 | 2.590399 | 97.410 | 33 | 11 | 9 | 22 | 0 | 0 | 0 |
| io-uring:sync_stride:stride_1 | 1 | 17454.856853 | 17454.856853 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 29008057 |
| io-uring:sync_stride:stride_30 | 1 | 655.073148 | 655.073148 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 33746492 |
| io-uring:aot_certificate:stride_1 | 1 | 1.333110 | 1.333110 | 98.667 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| io-uring:aot_certificate:stride_30 | 1 | 1.876749 | 1.876749 | 98.123 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
