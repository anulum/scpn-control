# Native formal verification mode benchmark

Generated: 2026-06-04T08:40:39.277448+00:00
Commit: `3aa4132`
Workspace dirty: `True`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:sleep:disabled:stride_30 | 3 | 1.681416 | 1.988049 | 98.012 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:spin:disabled:stride_30 | 3 | 1.932146 | 2.016279 | 97.984 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:sleep:aot_certificate:stride_1 | 3 | 1.730555 | 1.945835 | 98.054 | 15000 | 15000 | 15000 | 0 | 0 | 0 | 0 |
| std:spin:aot_certificate:stride_1 | 3 | 2.382170 | 2.813493 | 97.187 | 15000 | 15000 | 15000 | 0 | 0 | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
- aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.
- spin pacing busy-waits on a native core and should only be used for short timing experiments.
