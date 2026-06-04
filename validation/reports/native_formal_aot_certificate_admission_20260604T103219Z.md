# Native formal verification mode benchmark

Generated: 2026-06-04T10:32:28.659506+00:00
Commit: `f83865a`
Workspace dirty: `True`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Certificate admitted | Certificate SHA-256 | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| std:sleep:disabled:stride_30 | 3 | 1.709245 | 1.847833 | 98.152 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 |
| std:spin:disabled:stride_30 | 3 | 2.145228 | 2.350956 | 97.649 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 |
| std:sleep:aot_certificate:stride_1 | 3 | 2.255568 | 2.465372 | 97.535 | 15000 | 15000 | 15000 | 0 | 0 | 3 | ee058c7c918ce8eb800c03e0c6e5ae979ba01f95dc48c6da3dc3c1f63391fdfd | 0 | 0 |
| std:spin:aot_certificate:stride_1 | 3 | 2.243156 | 2.483346 | 97.517 | 15000 | 15000 | 15000 | 0 | 0 | 3 | ee058c7c918ce8eb800c03e0c6e5ae979ba01f95dc48c6da3dc3c1f63391fdfd | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
- aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.
- aot_certificate strict evidence requires certificate_admitted=true and one stable certificate_assumption_sha256.
- spin pacing busy-waits on a native core and should only be used for short timing experiments.
