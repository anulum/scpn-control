# Native handoff comparison

Generated: 2026-06-21T20:05:51.078325+00:00

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 166.264 | 4.085 | 0 | 0 |
| native | normal | native | 5000 | 165.105 | 2.543 | 0 | 0 |

Native handoff wall-time speedup: 1.007x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
