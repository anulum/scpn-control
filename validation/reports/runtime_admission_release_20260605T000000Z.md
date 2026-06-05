<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Runtime admission benchmark report. -->

# Runtime Admission Benchmark

- Generated UTC: `2026-06-05T00:33:21Z`
- Command: `benchmarks/bench_runtime_admission.py --iterations 500 --warmup 50 --core-snn 4 --core-z3 5 --core-net 6 --core-hb 7 --json-out validation/reports/runtime_admission_release_20260605T000000Z.json --md-out validation/reports/runtime_admission_release_20260605T000000Z.md`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- CPU affinity: `[4, 5, 6, 7]`
- Isolation method: `process-affinity-inherited-or-taskset`
- Load average: `[3.54541015625, 2.55517578125, 1.2998046875]` -> `[3.54541015625, 2.55517578125, 1.2998046875]`
- Samples: `500`
- Median: `140.395 us`
- p95: `182.384 us`
- p99: `216.253 us`
- Max: `275.081 us`

This is an admission-probe benchmark, not a control-loop hot-path benchmark.
