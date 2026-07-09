# Native formal verification mode benchmark

Generated: 2026-06-04T10:02:34.542512+00:00
Commit: `f83865a`
Workspace dirty: `False`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:sleep:aot_certificate:stride_1 | 5 | 1.155453 | 1.205115 | 98.795 | 50000 | 50000 | 50000 | 0 | 0 | 0 | 0 |
| std:spin:aot_certificate:stride_1 | 5 | 1.209395 | 1.233202 | 98.767 | 50000 | 50000 | 50000 | 0 | 0 | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
- aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.
- spin pacing busy-waits on a native core and should only be used for short timing experiments.
