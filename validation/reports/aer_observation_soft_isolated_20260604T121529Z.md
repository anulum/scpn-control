<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Control -->
<!-- Description: AER observation benchmark report. -->

# AER Observation Benchmark

- JSON evidence: `validation/reports/aer_observation_soft_isolated_20260604T121529Z.json`
- Claim boundary: soft-isolated local decoder benchmark evidence only; not target-hardware, neuromorphic-device, FPGA, HIL, or plant PCS timing evidence
- Command: `python benchmarks/bench_aer_observation.py --json-out validation/reports/aer_observation_soft_isolated_20260604T121529Z.json --md-out validation/reports/aer_observation_soft_isolated_20260604T121529Z.md`
- Duration: 177.042 s

## Host Context

```json
{
  "affinity": [
    0
  ],
  "cargo": "cargo 1.95.0 (f2d3ce0bd 2026-03-21)",
  "cpu_count": 12,
  "governor": "powersave",
  "loadavg_after": [
    7.28955078125,
    6.35009765625,
    5.185546875
  ],
  "loadavg_before": [
    6.712890625,
    5.44189453125,
    4.68798828125
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
    "isi_ns": {
      "max": 854510.0,
      "mean": 127977.50884,
      "median": 124470.0,
      "min": 88681.0,
      "n": 50000.0,
      "p95": 156472.0,
      "p99": 189866.0
    },
    "rate_ns": {
      "max": 589869.0,
      "mean": 64515.05296,
      "median": 52649.5,
      "min": 40019.0,
      "n": 50000.0,
      "p95": 116710.0,
      "p99": 156204.0
    },
    "temporal_ns": {
      "max": 379550.0,
      "mean": 36688.85726,
      "median": 31876.0,
      "min": 25757.0,
      "n": 50000.0,
      "p95": 57494.0,
      "p99": 73096.0
    }
  },
  "rust": {
    "isi_ns": {
      "max": 94834,
      "mean": 2402.30278,
      "median": 2138,
      "min": 1868,
      "n": 50000,
      "p95": 3770,
      "p99": 5073
    },
    "rate_ns": {
      "max": 77698,
      "mean": 190.31266,
      "median": 181,
      "min": 165,
      "n": 50000,
      "p95": 254,
      "p99": 272
    },
    "temporal_ns": {
      "max": 5422,
      "mean": 431.31394,
      "median": 435,
      "min": 356,
      "n": 50000,
      "p95": 483,
      "p99": 549
    }
  }
}
```
