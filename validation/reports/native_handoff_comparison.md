# Native handoff comparison

Generated: 2026-06-21T21:17:20.833247+00:00
Source commit: `5997eed1c135608dcd04720a8287ee9c10067265`
Evidence class: `local_proxy`
Runtime admission: `fail`
Production claim allowed: `false`
Claim boundary: Dated standard loopback-UDP local/CI observation only; not fielded plant, PCS-cycle, HIL, deterministic real-time, or production evidence.

| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| python | completed | python | 5000 | 184.031 | 8.850 | 0 | 0 |
| native | normal | native | 5000 | 179.687 | 4.988 | 0 | 0 |

Native handoff wall-time speedup: 1.024x.

The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.
