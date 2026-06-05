<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Native runtime validation report. -->
# Native handoff comparison

Generated: 2026-06-04T03:04:04.927464+00:00

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 194.835 | 9.039 | 0 | 0 |
| native | normal | native | 5000 | 254.089 | 17.790 | 0 | 0 |

Native handoff wall-time speedup: 0.767x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
