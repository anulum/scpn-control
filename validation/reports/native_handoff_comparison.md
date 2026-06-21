# Native handoff comparison

Generated: 2026-06-21T19:51:29.728275+00:00

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 174.012 | 7.076 | 0 | 0 |
| native | normal | native | 5000 | 178.390 | 6.060 | 0 | 0 |

Native handoff wall-time speedup: 0.975x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
