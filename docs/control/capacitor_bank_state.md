<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Capacitor-bank state model guide. -->

# Capacitor-Bank State Model

The capacitor-bank module is a CONTROL-owned admission model for pulsed-shot
planning. It provides a bounded series-RLC state surface that can be used by the
pulsed-scenario scheduler, pulsed MPC adapter, replay reports, and optional
Rust/PyO3 runtime paths.

It is not a facility driver, relay model, switch model, insulation model, or
hardware protection system. Those claims require target hardware evidence and
plant-specific interlock validation.

## State equation

The model advances the capacitor voltage and series current with the series-RLC
state equation:

```text
d/dt [v_C, i]^T = [[0, -1/C], [1/L, -R/L]] [v_C, i]^T + [-i_load/C, 0]^T
```

Where:

| Symbol | Meaning |
|---|---|
| `C` | bank capacitance in farads |
| `L` | series inductance in henries |
| `R` | series resistance in ohms |
| `v_C` | capacitor voltage in volts |
| `i` | series current in amperes |
| `i_load` | prescribed external load current in amperes |

The Python and Rust implementations use the same Crank-Nicolson update and the
same midpoint-sampled load-current waveforms: `rect`, `half_sine`, and
`exp_decay`.

## Energy surfaces

`CapacitorBankState.energy_J` remains the scheduler-facing capacitor electric
energy:

```text
E_C = 0.5 C v_C^2
```

The discharge report uses total stored RLC energy because the inductor may still
hold magnetic energy after a pulse step:

```text
E_total = 0.5 C v_C^2 + 0.5 L i^2
```

A discharge ledger records:

| Field | Meaning |
|---|---|
| `energy_initial_J` | initial total RLC stored energy |
| `energy_remaining_J` | final total RLC stored energy |
| `capacitor_energy_remaining_J` | final capacitor electric energy |
| `inductor_energy_remaining_J` | final inductor magnetic energy |
| `energy_delivered_J` | `energy_initial_J - energy_remaining_J` |
| `resistive_loss_J` | integrated ohmic dissipation |
| `load_energy_J` | integrated energy extracted by the prescribed load |
| `energy_balance_residual_J` | ledger residual |
| `energy_balance_relative_error` | scale-normalised residual |
| `energy_balance_passed` | admission flag for the residual tolerance |

The admission residual is:

```text
energy_initial_J - energy_remaining_J
  = resistive_loss_J + load_energy_J + energy_balance_residual_J
```

This contract catches drift between the RLC state update and the energy ledger.
It does not replace hardware metrology, relay timing, or plant protection logic.

## Python example

```python
from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec, PulseSpec

spec = CapacitorBankSpec(
    capacitance_F=100e-6,
    inductance_H=100e-6,
    series_resistance_ohm=0.5,
    voltage_max_V=10_000.0,
    recharge_power_kW=20.0,
)
bank = CapacitorBank(spec, initial_voltage_V=5_000.0, initial_current_A=12.0)
report = bank.discharge(
    PulseSpec(peak_current_A=500.0, duration_s=20e-6, waveform="half_sine"),
    dt=100e-9,
    n_steps=200,
)
assert report.energy_balance_passed
```

## Rust example

```rust
use control_control::capacitor_bank::{CapacitorBank, CapacitorBankSpec, PulseSpec, PulseWaveform};

let spec = CapacitorBankSpec::new(100e-6, 100e-6, 0.5, 10_000.0, 20.0).expect("valid spec");
let mut bank = CapacitorBank::new(spec, 5_000.0, 12.0).expect("valid bank");
let pulse = PulseSpec::new(500.0, 20e-6, PulseWaveform::HalfSine).expect("valid pulse");
let report = bank.discharge(pulse, 100e-9, 200).expect("discharge evaluates");
assert!(report.energy_balance_passed);
```

## Benchmarking

Use the dedicated Python and Rust benchmark harnesses when changing the RLC
admission ledger:

```bash
PYTHONPATH=src python benchmarks/bench_capacitor_bank_energy.py \
  --steps 500 --warmup 50 --discharge-steps 200 --dt-s 1.0e-7 \
  --json-out validation/reports/capacitor_bank_energy_python.json \
  --markdown-out validation/reports/capacitor_bank_energy_python.md

cargo run --release --manifest-path scpn-control-rs/Cargo.toml \
  -p control-control --example bench_capacitor_bank_energy -- \
  --steps 500 --warmup 50 --discharge-steps 200 --dt-s 1.0e-7 \
  --json-out validation/reports/capacitor_bank_energy_rust.json \
  --markdown-out validation/reports/capacitor_bank_energy_rust.md
```

Reports generated without hard CPU isolation are local regression evidence only.
