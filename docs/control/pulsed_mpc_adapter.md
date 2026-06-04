<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Pulsed-shot MPC admission adapter guide. -->

# Pulsed MPC Adapter

The pulsed-shot MPC adapter connects the CONTROL-owned gradient MPC action to
the pulsed-shot lifecycle and capacitor-bank admission surfaces. It is designed
for pulsed fusion control campaigns where an optimizer may propose a valid
shape-control action mathematically, while the current lifecycle state or stored
bank energy still makes that action inadmissible.

This adapter does not replace the equilibrium solver. It gates the first action
from `ModelPredictiveController.plan_trajectory()` through deterministic
control-plane contracts already owned by SCPN Control.

## Contract

The adapter executes the same admission sequence for each step:

1. Compute the first gradient-MPC action from the finite-difference surrogate.
2. Resolve the scheduler state from the scheduler, an enum, a string, a mapping,
   or an object with a `state` attribute.
3. Outside `burn`, replace all configured burn-action components with the safe
   action.
4. During `burn`, convert the proposed burn action into a `PulseSpec` unless an
   explicit pulse is supplied.
5. Reject the burn action if `CapacitorBank.feasibility()` says the bank cannot
   supply the requested pulse.
6. Persist the decision rationale through `explain_last_decision()`.
7. Bind the admitted action, safe action, burn-action mask, scheduler state,
   capacitor feasibility, objective, slack, reason, and peak-current estimate
   into a SHA-256 admission digest.

The decision record includes:

- `mpc_objective`
- `constraint_slack`
- `scheduler_state`
- `bank_feasibility`
- `reason`
- `bank_feasible`
- `safe_action_applied`
- `burn_components_masked`
- `peak_current_A`
- `evidence_schema_version`
- `action_sha256`
- `safe_action_sha256`
- `burn_action_mask_sha256`
- `admission_digest`

## Python use

```python
import numpy as np

from scpn_control.control.capacitor_bank_state import (
    CapacitorBank,
    CapacitorBankSpec,
)
from scpn_control.control.fusion_sota_mpc import (
    ModelPredictiveController,
    NeuralSurrogate,
    PulsedShotMPCAdapter,
)
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
    PulsedScenarioState,
)

surrogate = NeuralSurrogate(n_coils=2, n_state=2, verbose=False)
surrogate.B = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
mpc = ModelPredictiveController(surrogate, np.array([6.0, 0.0]))

scheduler = PulsedScenarioScheduler(
    PulsedScenarioSpec(
        min_precharge_energy_J=5.0,
        ramp_current_A=10.0,
        phase_tolerance_rad=0.1,
        spatial_tolerance_m=0.01,
        burn_temperature_eV=100.0,
        min_fusion_power_W=50.0,
        expansion_velocity_m_s=5.0,
        dump_energy_floor_J=0.5,
        recharge_voltage_fraction=0.8,
        cooldown_temperature_eV=10.0,
        cooldown_current_A=1.0,
    )
)
scheduler.state = PulsedScenarioState.FLAT_TOP

bank = CapacitorBank(
    CapacitorBankSpec(
        capacitance_F=1.0,
        inductance_H=1.0,
        series_resistance_ohm=0.05,
        voltage_max_V=20.0,
    ),
    initial_voltage_V=10.0,
)

adapter = PulsedShotMPCAdapter(
    mpc,
    scheduler,
    bank,
    burn_action_mask=np.array([True, False]),
    safe_action=np.zeros(2),
)

action = adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]))
decision = adapter.explain_last_decision()
```

In this example, the scheduler is in `flat_top`, so the first action component
is replaced with the safe value and the second component remains available for
non-burn trim.

`admission_digest` is a tamper-evident digest over schema
`scpn-control.pulsed-mpc-decision-evidence.v1`. It is intended for replay,
benchmark, and campaign evidence. It is not a proof that a facility capacitor
bank can satisfy the pulse; measured hardware interlock evidence still has to be
attached before facility claims.

## Rust and PyO3 surfaces

The Rust parity surface lives in `control_control::mpc::MPController`:

```rust
let decision = mpc.plan_pulsed(
    &state,
    "flat_top",
    true,
    &[true, false],
    &safe_action,
    12.0,
)?;
```

The method rejects unknown scheduler states and mismatched action masks. Accepted
states are:

```text
idle, ramp_up, flat_top, burn, expansion, dump, recharge, cool_down
```

When the optional PyO3 extension is installed, Python can call the compiled
surface:

```python
import numpy as np
import scpn_control_rs

mpc = scpn_control_rs.PyMpcController(
    np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64),
    np.array([6.0, 0.0], dtype=np.float64),
)

action, decision = mpc.plan_pulsed(
    np.array([5.0, 1.0], dtype=np.float64),
    "burn",
    False,
    np.array([True, True], dtype=bool),
    np.zeros(2, dtype=np.float64),
    -0.5,
)
```

If the local PyO3 wheel was not rebuilt after this API landed, parity tests skip
the optional extension rather than treating a stale developer environment as a
runtime contract failure.

The Rust and PyO3 surfaces expose the same evidence field names as Python:
`evidence_schema_version`, `action_sha256`, `safe_action_sha256`,
`burn_action_mask_sha256`, `peak_current_A`, and `admission_digest`.

After rebuilding the editable PyO3 wheel from the current Rust source, the
dedicated parity tests must run without stale-wheel skips:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_fusion_sota_mpc_pulsed_adapter.py \
  tests/test_fusion_sota_mpc_pulsed_adapter_rust_parity.py -q
```

## Evidence boundary

The adapter is a CONTROL admission primitive. It is not a facility PCS driver,
capacitor hardware interlock, measured-shot validation result, or new solver.
The decision digest records the adapter's local admission evidence; it must be
carried with replay or campaign evidence if downstream systems depend on the
admitted pulsed-MPC action.

Use local regression benchmarks to catch Python/Rust adapter drift:

```bash
taskset -c 4,5 env PYTHONPATH=src python benchmarks/bench_pulsed_mpc_adapter.py \
  --steps 2000 \
  --warmup 200 \
  --evidence-class local_regression \
  --json-out validation/reports/pulsed_mpc_adapter_soft_isolated.json \
  --md-out validation/reports/pulsed_mpc_adapter_soft_isolated.md
```

Loaded workstation reports and soft-affinity reports must keep
`production_claim_allowed=false`. Production timing claims require clean host
load, explicit core isolation metadata, and target-hardware evidence.

When PyO3 is unavailable, run the native Rust benchmark directly:

```bash
cargo run --manifest-path scpn-control-rs/Cargo.toml \
  -p control-control \
  --example bench_pulsed_mpc_adapter \
  --release \
  -- \
  --steps 2000 \
  --warmup 200 \
  --json-out validation/reports/pulsed_mpc_adapter_rust_local_regression.json \
  --md-out validation/reports/pulsed_mpc_adapter_rust_local_regression.md
```
