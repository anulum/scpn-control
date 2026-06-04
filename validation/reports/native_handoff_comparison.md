# Native handoff comparison

Generated: 2026-06-04T03:04:04.927464+00:00

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 194.835 | 9.039 | 0 | 0 |
| native | normal | native | 5000 | 254.089 | 17.790 | 0 | 0 |

Native handoff wall-time speedup: 0.767x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
