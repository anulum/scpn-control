# Native formal verification mode benchmark

Generated: 2026-06-04T07:47:06.917115+00:00
Commit: `f527dcf`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:disabled:stride_30 | 3 | 0.839490 | 1.106604 | 98.893 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:async_drop:stride_1 | 3 | 1.167403 | 1.232226 | 98.768 | 15000 | 9 | 3 | 14991 | 0 | 0 | 0 |
| std:async_drop:stride_5 | 3 | 1.148440 | 1.169233 | 98.831 | 3000 | 9 | 3 | 2991 | 0 | 0 | 0 |
| std:async_drop:stride_20 | 3 | 0.939107 | 0.954213 | 99.046 | 750 | 9 | 3 | 741 | 0 | 0 | 0 |
| std:async_drop:stride_30 | 3 | 1.148438 | 1.174127 | 98.826 | 498 | 9 | 3 | 489 | 0 | 0 | 0 |
| std:sync_stride:stride_1 | 3 | 15639.588845 | 16083.815867 | 0.000 | 15000 | 15000 | 15000 | 0 | 0 | 15000 | 32724227 |
| std:sync_stride:stride_5 | 3 | 3126.000388 | 3290.275986 | 0.000 | 3000 | 3000 | 3000 | 0 | 0 | 3000 | 29965096 |
| std:sync_stride:stride_20 | 3 | 759.412760 | 765.132624 | 0.000 | 750 | 750 | 750 | 0 | 0 | 750 | 25580375 |
| std:sync_stride:stride_30 | 3 | 505.409290 | 507.002614 | 0.000 | 498 | 498 | 498 | 0 | 0 | 498 | 24318433 |
| io-uring:disabled:stride_30 | 3 | 1.061937 | 1.093324 | 98.907 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| io-uring:async_drop:stride_1 | 3 | 1.152584 | 1.278212 | 98.722 | 15000 | 9 | 3 | 14991 | 0 | 0 | 0 |
| io-uring:async_drop:stride_5 | 3 | 1.180638 | 1.206140 | 98.794 | 3000 | 9 | 3 | 2991 | 0 | 0 | 0 |
| io-uring:async_drop:stride_20 | 3 | 1.083519 | 1.271890 | 98.728 | 750 | 9 | 3 | 741 | 0 | 0 | 0 |
| io-uring:async_drop:stride_30 | 3 | 1.114800 | 1.293909 | 98.706 | 498 | 9 | 3 | 489 | 0 | 0 | 0 |
| io-uring:sync_stride:stride_1 | 3 | 15702.045303 | 17288.996560 | 0.000 | 15000 | 15000 | 15000 | 0 | 0 | 15000 | 38213070 |
| io-uring:sync_stride:stride_5 | 3 | 3097.415782 | 3466.457241 | 0.000 | 3000 | 3000 | 3000 | 0 | 0 | 3000 | 58219434 |
| io-uring:sync_stride:stride_20 | 3 | 750.248373 | 764.051929 | 0.000 | 750 | 750 | 750 | 0 | 0 | 750 | 26690977 |
| io-uring:sync_stride:stride_30 | 3 | 519.150226 | 540.938946 | 0.000 | 498 | 498 | 498 | 0 | 0 | 498 | 28708517 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
