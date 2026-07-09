# Workstation native formal stride matrix

Generated: 2026-06-04T07:20:33.004337+00:00
Commit: `f527dcf`

workstation soft-isolated benchmark; not kernel isolcpus clean-room evidence

| Case | Runs | Actual backend(s) | p50 cycle us | p95 cycle us | p99 cycle us | p99 headroom % | Formal submitted | Formal checked | Formal dropped | Formal failures | Drops | Publish failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:formal_disabled | 7 | std | 1.943055 | 6.670589 | 6.670589 | 93.329 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:formal_stride_1 | 7 | std | 2.475653 | 3.599710 | 3.599710 | 96.400 | 320 | 307 | 34680 | 0 | 0 | 0 |
| std:formal_stride_5 | 7 | std | 2.526574 | 2.917425 | 2.917425 | 97.083 | 340 | 326 | 6660 | 0 | 0 | 0 |
| std:formal_stride_20 | 7 | std | 2.344592 | 2.573906 | 2.573906 | 97.426 | 351 | 337 | 1399 | 0 | 0 | 0 |
| std:formal_stride_30 | 7 | std | 2.592700 | 3.221225 | 3.221225 | 96.779 | 342 | 329 | 820 | 0 | 0 | 0 |
| io-uring:formal_disabled | 7 | io-uring | 1.609187 | 1.704461 | 1.704461 | 98.296 | 0 | 0 | 0 | 0 | 0 | 0 |
| io-uring:formal_stride_1 | 7 | io-uring | 2.641997 | 3.324049 | 3.324049 | 96.676 | 332 | 318 | 34668 | 0 | 0 | 0 |
| io-uring:formal_stride_5 | 7 | io-uring | 2.738622 | 3.568203 | 3.568203 | 96.432 | 331 | 317 | 6669 | 0 | 0 | 0 |
| io-uring:formal_stride_20 | 7 | io-uring | 2.500496 | 2.699898 | 2.699898 | 97.300 | 347 | 333 | 1403 | 0 | 0 | 0 |
| io-uring:formal_stride_30 | 7 | io-uring | 2.608608 | 3.827872 | 3.827872 | 96.172 | 333 | 321 | 829 | 0 | 0 | 0 |

Limits:
- No kernel isolcpus/nohz_full configured on this boot.
- taskset/sched_setaffinity restricts this benchmark process but does not evict kernel work or unrelated processes from sibling CPUs.
- Governor was temporarily set to performance where permitted and restored by shell trap.
- p50/p95/p99 are across repeated 5000-step campaigns, not intra-campaign per-step histograms; the native loop currently exposes aggregate cycle telemetry only.
