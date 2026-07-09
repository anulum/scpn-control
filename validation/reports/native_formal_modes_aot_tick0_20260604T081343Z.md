# Native formal verification mode benchmark

Generated: 2026-06-04T08:14:22.772937+00:00
Commit: `f527dcf`
Workspace dirty: `True`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:disabled:stride_30 | 1 | 0.928987 | 0.928987 | 99.071 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:async_drop:stride_1 | 1 | 0.813323 | 0.813323 | 99.187 | 1000 | 2 | 0 | 998 | 0 | 0 | 0 |
| std:async_drop:stride_30 | 1 | 0.839138 | 0.839138 | 99.161 | 33 | 3 | 1 | 30 | 0 | 0 | 0 |
| std:sync_stride:stride_1 | 1 | 17632.722322 | 17632.722322 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 31765549 |
| std:sync_stride:stride_30 | 1 | 759.472517 | 759.472517 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 34716312 |
| std:aot_certificate:stride_1 | 1 | 0.908808 | 0.908808 | 99.091 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| std:aot_certificate:stride_30 | 1 | 1.222472 | 1.222472 | 98.778 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |
| io-uring:disabled:stride_30 | 1 | 0.830792 | 0.830792 | 99.169 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| io-uring:async_drop:stride_1 | 1 | 1.160164 | 1.160164 | 98.840 | 1000 | 2 | 0 | 998 | 0 | 0 | 0 |
| io-uring:async_drop:stride_30 | 1 | 1.126766 | 1.126766 | 98.873 | 33 | 2 | 0 | 31 | 0 | 0 | 0 |
| io-uring:sync_stride:stride_1 | 1 | 18753.526148 | 18753.526148 | 0.000 | 1000 | 1000 | 1000 | 0 | 0 | 1000 | 35052277 |
| io-uring:sync_stride:stride_30 | 1 | 652.665016 | 652.665016 | 0.000 | 33 | 33 | 33 | 0 | 0 | 33 | 30158264 |
| io-uring:aot_certificate:stride_1 | 1 | 1.229034 | 1.229034 | 98.771 | 1000 | 1000 | 1000 | 0 | 0 | 0 | 0 |
| io-uring:aot_certificate:stride_30 | 1 | 1.162809 | 1.162809 | 98.837 | 33 | 33 | 33 | 0 | 0 | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
- aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.
