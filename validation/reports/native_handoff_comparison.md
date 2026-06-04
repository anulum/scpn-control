# Native handoff comparison

Generated: 2026-06-04T02:25:26.729640+00:00

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 206.870 | 11.914 | 0 | 0 |
| native | normal | native | 5000 | 196.530 | 5.765 | 0 | 0 |

Native handoff wall-time speedup: 1.053x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
