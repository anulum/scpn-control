# Native handoff comparison

Generated: 2026-06-21T21:17:20.833247+00:00

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 184.031 | 8.850 | 0 | 0 |
| native | normal | native | 5000 | 179.687 | 4.988 | 0 | 0 |

Native handoff wall-time speedup: 1.024x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
