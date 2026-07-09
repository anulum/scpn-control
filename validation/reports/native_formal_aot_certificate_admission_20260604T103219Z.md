# Native formal verification mode benchmark

Generated: 2026-06-04T10:57:30.673627+00:00
Commit: `32bad60`
Workspace dirty: `True`
Evidence class: `local_regression`
Production claim allowed: `False`
Isolation method: `none`
Affinity CPUs: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`
Host load before: `7.29 6.15 4.70 2/2867 1893504`
Host load after: `6.47 6.01 4.68 1/2869 1893906`

native formal mode benchmark; benchmark_context defines whether timing evidence is local regression or production benchmark evidence

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Certificate admitted | Certificate SHA-256 | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| std:sleep:disabled:stride_30 | 3 | 2.698094 | 3.282789 | 96.717 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 |
| std:spin:disabled:stride_30 | 3 | 2.085763 | 2.137472 | 97.863 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 |
| std:sleep:aot_certificate:stride_1 | 3 | 1.568554 | 2.187247 | 97.813 | 15000 | 15000 | 15000 | 0 | 0 | 3 | ee058c7c918ce8eb800c03e0c6e5ae979ba01f95dc48c6da3dc3c1f63391fdfd | 0 | 0 |
| std:spin:aot_certificate:stride_1 | 3 | 1.593288 | 1.642601 | 98.357 | 15000 | 15000 | 15000 | 0 | 0 | 3 | ee058c7c918ce8eb800c03e0c6e5ae979ba01f95dc48c6da3dc3c1f63391fdfd | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
- aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.
- aot_certificate strict evidence requires certificate_admitted=true and one stable certificate_assumption_sha256.
- spin pacing busy-waits on a native core and should only be used for short timing experiments.
