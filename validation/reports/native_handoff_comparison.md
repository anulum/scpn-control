# Native handoff comparison

Generated: 2026-06-21T20:15:31.099725+00:00

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 185.280 | 9.692 | 0 | 0 |
| native | normal | native | 5000 | 181.053 | 5.257 | 0 | 0 |

Native handoff wall-time speedup: 1.023x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
