# Native formal verification mode benchmark

Generated: 2026-06-04T07:58:00.098680+00:00
Commit: `f527dcf`

native formal mode benchmark; isolation depends on caller taskset/governor setup

| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| std:disabled:stride_30 | 3 | 1.450906 | 1.464193 | 98.536 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| std:async_drop:stride_1 | 3 | 2.401810 | 2.685771 | 97.314 | 15000 | 149 | 143 | 14851 | 0 | 0 | 0 |
| std:async_drop:stride_5 | 3 | 2.222997 | 2.389694 | 97.610 | 3000 | 157 | 151 | 2843 | 0 | 0 | 0 |
| std:async_drop:stride_20 | 3 | 2.267032 | 2.539247 | 97.461 | 750 | 152 | 146 | 598 | 0 | 0 | 0 |
| std:async_drop:stride_30 | 3 | 2.418636 | 2.598157 | 97.402 | 498 | 149 | 144 | 349 | 0 | 0 | 0 |
| std:sync_stride:stride_1 | 3 | 15495.819632 | 16338.570932 | 0.000 | 15000 | 15000 | 15000 | 0 | 0 | 15000 | 38615085 |
| std:sync_stride:stride_5 | 3 | 3096.944956 | 3129.345224 | 0.000 | 3000 | 3000 | 3000 | 0 | 0 | 3000 | 27342212 |
| std:sync_stride:stride_20 | 3 | 781.574638 | 789.689244 | 0.000 | 750 | 750 | 750 | 0 | 0 | 750 | 29193689 |
| std:sync_stride:stride_30 | 3 | 518.201224 | 518.291916 | 0.000 | 498 | 498 | 498 | 0 | 0 | 498 | 25561356 |
| io-uring:disabled:stride_30 | 3 | 1.514566 | 1.524470 | 98.476 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| io-uring:async_drop:stride_1 | 3 | 2.267961 | 2.401252 | 97.599 | 15000 | 156 | 150 | 14844 | 0 | 0 | 0 |
| io-uring:async_drop:stride_5 | 3 | 4.182304 | 26.938760 | 73.061 | 3000 | 131 | 125 | 2869 | 0 | 0 | 0 |
| io-uring:async_drop:stride_20 | 3 | 2.185438 | 31.557607 | 68.442 | 750 | 141 | 135 | 609 | 0 | 0 | 0 |
| io-uring:async_drop:stride_30 | 3 | 2.789061 | 3.189418 | 96.811 | 498 | 141 | 136 | 357 | 0 | 0 | 0 |
| io-uring:sync_stride:stride_1 | 3 | 16461.603891 | 16540.469449 | 0.000 | 15000 | 15000 | 15000 | 0 | 0 | 15000 | 32413595 |
| io-uring:sync_stride:stride_5 | 3 | 3153.716346 | 3260.008698 | 0.000 | 3000 | 3000 | 3000 | 0 | 0 | 3000 | 27638750 |
| io-uring:sync_stride:stride_20 | 3 | 810.751074 | 816.834398 | 0.000 | 750 | 750 | 750 | 0 | 0 | 750 | 29219479 |
| io-uring:sync_stride:stride_30 | 3 | 529.265924 | 539.809607 | 0.000 | 498 | 498 | 498 | 0 | 0 | 498 | 25043112 |

Limitations:
- p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.
- async_drop deliberately drops saturated snapshots and is not strict proof coverage.
- sync_stride blocks on designated stride steps and exposes sync wait telemetry.
