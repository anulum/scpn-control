# Capacitor-Bank State Model Benchmark

- JSON evidence: `validation/reports/capacitor_bank_state_soft_isolated_20260604T115849Z.json`
- Claim boundary: soft-isolated local benchmark evidence only; not target-hardware, HIL, facility interlock, or plant PCS timing evidence
- Command: `python validation/benchmark_capacitor_bank_state.py --json-out validation/reports/capacitor_bank_state_soft_isolated_20260604T115849Z.json --md-out validation/reports/capacitor_bank_state_soft_isolated_20260604T115849Z.md`
- Duration: 138.753 s

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
    4.56005859375,
    3.92822265625,
    4.0146484375
  ],
  "loadavg_before": [
    3.2109375,
    3.61669921875,
    3.94580078125
  ],
  "platform": "Linux-6.17.0-29-generic-x86_64-with-glibc2.39",
  "processor": "x86_64",
  "python": "3.12.3",
  "rustc": "rustc 1.95.0 (59807616e 2026-04-14)"
}
```

## Results

```json
{
  "python": {
    "discharge_10_step_ns": {
      "max": 93446.0,
      "mean": 43766.164,
      "median": 42015.0,
      "min": 36406.0,
      "n": 500.0,
      "p95": 54357.0,
      "p99": 78829.0
    },
    "free_response_ns": {
      "max": 57078.0,
      "mean": 3264.45812,
      "median": 2831.0,
      "min": 2580.0,
      "n": 50000.0,
      "p95": 5457.0,
      "p99": 6843.0
    },
    "step_ns": {
      "max": 420457.0,
      "mean": 4120.02632,
      "median": 3580.5,
      "min": 2451.0,
      "n": 50000.0,
      "p95": 6264.0,
      "p99": 9712.0
    }
  },
  "rust": {
    "discharge_10_step_ns": {
      "max": 4770,
      "mean": 231.336,
      "median": 197,
      "min": 192,
      "n": 500,
      "p95": 311,
      "p99": 399
    },
    "free_response_ns": {
      "max": 13279,
      "mean": 64.36658,
      "median": 62,
      "min": 53,
      "n": 50000,
      "p95": 74,
      "p99": 82
    },
    "step_ns": {
      "max": 6732,
      "mean": 34.48338,
      "median": 33,
      "min": 27,
      "n": 50000,
      "p95": 41,
      "p99": 46
    }
  }
}
```

The Rust measurements use a temporary release-mode Cargo harness that imports the checked-in
`control-control::capacitor_bank` crate by path. The Python measurements use
`scpn_control.control.capacitor_bank_state` in the same process.
