<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Native runtime validation report. -->
# Native formal worker soft-isolated benchmark

Generated: 2026-06-04T07:13:05.596607+00:00
Commit: `f527dcf`

Classification: soft-isolated; not kernel `isolcpus` clean-room evidence.

| Case | Repeats | Median avg cycle us | Median effective step us | Formal submitted | Formal checked | Formal dropped | Formal failures | Drops | Publish failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| formal disabled | 5 | 1.909248 | 163.815953 | 0 | 0 | 0 | 0 | 0 | 0 |
| formal enabled | 5 | 3.708895 | 167.015651 | 229 | 220 | 601 | 0 | 0 | 0 |

Formal median cycle overhead: 94.259%.

Limits:
- No kernel isolated CPUs were configured.
- Process was pinned with `taskset` and `sched_setaffinity` to CPUs 7, 10, 11.
- Governor was temporarily set to `performance` where permitted and restored after the run.
- Background system load can still affect cache, memory bandwidth, and scheduler noise.
