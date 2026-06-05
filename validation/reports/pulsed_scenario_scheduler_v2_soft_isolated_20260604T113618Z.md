<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Pulsed scenario scheduler v2 soft-isolated benchmark report. -->

# PulsedScenarioScheduler v2 Soft-Isolated Benchmark

Generated: 2026-06-04T11:36:18Z
Commit under test: `850198b`
Surface: `scpn_control.control.pulsed_scenario_scheduler_v2`

## Scope

This benchmark measures one full eight-step pulsed scenario lifecycle campaign:

`idle -> ramp_up -> flat_top -> burn -> expansion -> dump -> recharge -> cool_down -> idle`

The Python surface and the Rust `control_control::pulsed_scenario` surface were
measured separately. Rust compile time was excluded from the runtime metric.

## Environment and claim boundary

This is soft-isolated workstation timing evidence only. It is not clean-room,
target-hardware, HIL, PREEMPT_RT, or plant-deployment evidence.

The benchmark used `taskset -c 0`. During the run the workstation still had
substantial background load:

| Source | Observed load |
| --- | ---: |
| `pytest tests/test_server_contracts.py` | ~141% CPU |
| `rustc` release build in SCPN-FUSION-CORE | ~99% CPU |
| `mypy` pre-commit worker | ~58% CPU |
| Browsers and code terminals | active |

Load average snapshot: `5.04, 3.94, 3.94`.

## Results

| Surface | Mean | Median | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python 8-step campaign | 34.340 us | 31.016 us | 71.658 us | 91.564 us | 2106.596 us |
| Rust 8-step campaign | 0.441 us | 0.376 us | 0.826 us | 0.905 us | 57.427 us |

Derived ratios:

| Ratio | Value |
| --- | ---: |
| Rust speed-up by mean | 77.787x |
| Rust speed-up by P99 | 101.176x |

## Interpretation

The Rust scheduler is justified for hot-path and control-loop use. The Python
surface remains appropriate for control-plane orchestration, debugging, and
campaign construction.

The large Python max outlier and Rust max outlier are attributed to workstation
contamination and OS scheduling under load. They must not be cited as release or
facility timing evidence.

## Reproduction commands

Python path:

```bash
taskset -c 0 env PYTHONPATH=src python - <<'PY'
# focused in-process benchmark; see JSON report for metric payload
PY
```

Rust path:

```bash
cargo build --release --manifest-path /tmp/scpn-control-pulsed-bench/Cargo.toml
taskset -c 0 /tmp/scpn-control-pulsed-bench/target/release/bench_control_pulsed
```

## Follow-up

1. Repeat on a quiet workstation or isolated ML350/PREEMPT_RT host before release-grade claims.
2. Persist a first-class repository benchmark harness if the scheduler becomes a gating latency surface.
3. Expose or benchmark the PyO3 path separately if Python needs to call the Rust scheduler directly.

Raw machine-readable report:
[`pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json`](pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json).
