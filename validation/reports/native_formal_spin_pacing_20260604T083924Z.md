<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Native runtime validation report. -->
# Native formal verification mode benchmark

Generated: 2026-06-04T08:39:35.529793+00:00
Commit: `3aa4132`
Workspace dirty: `True`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:sleep:disabled:stride_30 | 3 | 2.038174 | 2.416682 | 97.583 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:spin:disabled:stride_30 | 3 | 2.248468 | 2.763893 | 97.236 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:sleep:aot_certificate:stride_1 | 3 | 1.738191 | 7.113521 | 92.886 | 15000 | 15000 | 15000 | 0 | 0 | 0 | 0 |
| std:spin:aot_certificate:stride_1 | 3 | 2.239795 | 2.939109 | 97.061 | 15000 | 15000 | 15000 | 0 | 0 | 0 | 0 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
- aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.
