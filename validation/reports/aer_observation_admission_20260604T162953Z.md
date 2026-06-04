<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — AER observation benchmark report. -->

# AER Observation Benchmark

- JSON evidence: `validation/reports/aer_observation_admission_20260604T162953Z.json`
- Evidence class: `local_regression`
- Production claim allowed: `False`
- Claim boundary: soft-isolated local decoder benchmark evidence only; not target-hardware, neuromorphic-device, FPGA, HIL, or plant PCS timing evidence
- Command: `python benchmarks/bench_aer_observation.py --iterations 20000 --warmup 1000 --json-out validation/reports/aer_observation_admission_20260604T162953Z.json --md-out validation/reports/aer_observation_admission_20260604T162953Z.md`
- Outer command: `taskset -c 4,5 env PYTHONPATH=src .venv/bin/python benchmarks/bench_aer_observation.py --iterations 20000 --warmup 1000 --json-out validation/reports/aer_observation_admission_20260604T162953Z.json --md-out validation/reports/aer_observation_admission_20260604T162953Z.md`
- Duration: 72.361 s

## Host Context

```json
{
  "affinity": [
    4,
    5
  ],
  "cargo": "cargo 1.95.0 (f2d3ce0bd 2026-03-21)",
  "cpu_count": 12,
  "governor": "powersave",
  "loadavg_after": [
    4.09130859375,
    4.06982421875,
    3.21923828125
  ],
  "loadavg_before": [
    3.30224609375,
    3.9453125,
    3.11328125
  ],
  "platform": "Linux-6.17.0-29-generic-x86_64-with-glibc2.39",
  "python": "3.12.3",
  "rustc": "rustc 1.95.0 (59807616e 2026-04-14)"
}
```

## Results

```json
{
  "python": {
    "admission_push_report_ns": {
      "max": 160712.0,
      "mean": 20170.46925,
      "median": 19839.0,
      "min": 18282.0,
      "n": 20000.0,
      "p95": 23504.0,
      "p99": 29718.0
    },
    "admission_report_ns": {
      "max": 7643.0,
      "mean": 193.3567,
      "median": 190.0,
      "min": 168.0,
      "n": 20000.0,
      "p95": 203.0,
      "p99": 253.0
    },
    "isi_ns": {
      "max": 204919.0,
      "mean": 55147.4628,
      "median": 53805.0,
      "min": 48428.0,
      "n": 20000.0,
      "p95": 65694.0,
      "p99": 81986.0
    },
    "rate_ns": {
      "max": 207492.0,
      "mean": 39112.26845,
      "median": 37713.0,
      "min": 33828.0,
      "n": 20000.0,
      "p95": 48707.0,
      "p99": 62957.0
    },
    "temporal_ns": {
      "max": 167236.0,
      "mean": 24217.71385,
      "median": 23511.0,
      "min": 21300.0,
      "n": 20000.0,
      "p95": 29963.0,
      "p99": 37271.0
    }
  },
  "rust": {
    "admission_push_report_ns": {
      "max": 18702,
      "mean": 666.07635,
      "median": 634,
      "min": 560,
      "n": 20000,
      "p95": 796,
      "p99": 1128
    },
    "admission_report_ns": {
      "max": 26,
      "mean": 12.78195,
      "median": 13,
      "min": 12,
      "n": 20000,
      "p95": 13,
      "p99": 14
    },
    "isi_ns": {
      "max": 64119,
      "mean": 2055.4966,
      "median": 2007,
      "min": 1882,
      "n": 20000,
      "p95": 2216,
      "p99": 2552
    },
    "rate_ns": {
      "max": 43173,
      "mean": 227.2658,
      "median": 217,
      "min": 167,
      "n": 20000,
      "p95": 296,
      "p99": 412
    },
    "temporal_ns": {
      "max": 33446,
      "mean": 432.87225,
      "median": 416,
      "min": 379,
      "n": 20000,
      "p95": 511,
      "p99": 572
    }
  }
}
```
