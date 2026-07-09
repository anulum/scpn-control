# Native formal verification mode benchmark

Generated: 2026-06-04T08:15:05.378487+00:00
Commit: `f527dcf`
Workspace dirty: `True`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:disabled:stride_30 | 1 | 1.941511 | 1.941511 | 98.058 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:async_drop:stride_1 | 1 | 2.835619 | 2.835619 | 97.164 | 1000 | 9 | 7 | 991 | 0 | 0 | 0 |
| std:async_drop:stride_30 | 1 | 3.042914 | 3.042914 | 96.957 | 33 | 10 | 8 | 23 | 0 | 0 | 0 |
| std:sync_stride:stride_1 | 1 | 19203.268275 | 19203.268275 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 38849902 |
| std:sync_stride:stride_30 | 1 | 579.733770 | 579.733770 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 24379488 |
| std:aot_certificate:stride_1 | 1 | 1.783724 | 1.783724 | 98.216 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| std:aot_certificate:stride_30 | 1 | 1.930636 | 1.930636 | 98.069 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |
| io-uring:disabled:stride_30 | 1 | 1.553573 | 1.553573 | 98.446 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| io-uring:async_drop:stride_1 | 1 | 2.416258 | 2.416258 | 97.584 | 1000 | 11 | 9 | 989 | 0 | 0 | 0 |
| io-uring:async_drop:stride_30 | 1 | 2.499637 | 2.499637 | 97.500 | 33 | 10 | 8 | 23 | 0 | 0 | 0 |
| io-uring:sync_stride:stride_1 | 1 | 17981.167114 | 17981.167114 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 31407960 |
| io-uring:sync_stride:stride_30 | 1 | 547.481370 | 547.481370 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 25521352 |
| io-uring:aot_certificate:stride_1 | 1 | 2.421851 | 2.421851 | 97.578 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| io-uring:aot_certificate:stride_30 | 1 | 1.357138 | 1.357138 | 98.643 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
- aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.
