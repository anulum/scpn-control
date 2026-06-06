<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Pulsed scenario scheduler guide. -->

# Pulsed Scenario Scheduler

The pulsed scenario scheduler is the CONTROL-owned lifecycle contract for
reusable pulsed-fusion shot admission. It models the adjacent eight-state ring:

`idle -> ramp_up -> flat_top -> burn -> expansion -> dump -> recharge -> cool_down -> idle`

The scheduler is a control-boundary primitive. It does not replace facility
machine protection, commissioned actuator drivers, capacitor-bank hardware
interlocks, or measured-shot validation. It provides a deterministic state
contract that SCPN-MIF-CORE and other pulsed-fusion callers can consume without
duplicating CONTROL logic.

## Public import surfaces

The canonical implementation is:

```python
from scpn_control.control.pulsed_scenario_scheduler_v2 import PulsedScenarioScheduler
```

The MIF-facing compatibility import is also supported:

```python
from scpn_control.control.pulsed_scenario_scheduler import PulsedScenarioScheduler
```

Both names resolve to the same Python implementation. The compiled hot-path
counterpart is `control_control::pulsed_scenario` in Rust, exposed through
`scpn_control_rs.PyPulsedScenarioScheduler` when the optional PyO3 extension is
built.

## Guard contract

| State | Transition guard | Hold reason when blocked |
| --- | --- | --- |
| `idle` | bank energy is at least `min_precharge_energy_J` | waiting for precharge energy |
| `ramp_up` | absolute coil current is at least `ramp_current_A` | waiting for ramp current |
| `flat_top` | phase, spatial, temperature, and energy guards pass | waiting for phase/spatial lock, burn temperature, or burn energy |
| `burn` | minimum dwell elapsed and fusion power reaches `min_fusion_power_W` | waiting for minimum burn dwell or fusion power |
| `expansion` | radial velocity reaches `expansion_velocity_m_s` | waiting for expansion velocity |
| `dump` | bank energy falls to `dump_energy_floor_J` or below | waiting for dump energy floor |
| `recharge` | voltage fraction reaches `recharge_voltage_fraction` | waiting for recharge voltage |
| `cool_down` | plasma temperature and coil current are below cooldown limits | waiting for plasma cooldown |

Manual transitions are allowed only to the adjacent successor. For example,
`idle -> burn` is rejected in Python, Rust, PyO3, and the Lean proof model.

## Formal liveness evidence

`lean/SCPNControl/PulsedFSM.lean` contains the finite-state transition model and
the theorem:

```lean
theorem pulsed_fsm_eventually_returns_to_idle (state : State) :
    ∃ n : Nat, n ≤ 8 ∧ stepN n state = State.idle
```

This proves that every state in the adjacent transition ring reaches `idle`
within one full cycle under the abstract transition relation. The theorem is a
structural liveness proof for the scheduler topology. It is not a proof that a
specific plasma or capacitor-bank plant satisfies the guard thresholds; those
remain runtime admission and validation obligations.

## Benchmark evidence

Existing local-regression evidence is recorded in:

- `validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json`
- `validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.md`

These measurements are soft-isolated local regression evidence only. Production
timing claims require the project benchmark-isolation protocol: reserved cores,
host-load context, scheduler/governor context, and target runtime evidence.

## Validation commands

```bash
PYTHONPATH=src python -m pytest tests/test_pulsed_scenario_scheduler_v2.py \
  tests/test_pulsed_scenario_scheduler_formal.py
```

To require the Lean proof build where `lake` is installed:

```bash
MIF_LEAN_CI=1 PYTHONPATH=src python -m pytest tests/test_pulsed_scenario_scheduler_formal.py
```

## Practical use pattern

Use this scheduler as a shared lifecycle reference between python control, native execution, and lean formal checks.

1. initialise the scheduler and bank/coil guard parameters,
2. step through intended phases in the documented order,
3. check transition reasons before accepting a phase move,
4. persist rejected transitions for later review.

For facility-like timing claims, combine this with the production-readiness and benchmark context checks.

## Practical use and scope

Use this document to map pulsed-control lifecycle transitions and transition preconditions.

- Use it before changing state ordering, dwell periods, or ring-stage behavior.
- Keep any edits aligned with admission and replay requirements in related control docs.
- Validate scheduler changes with representative campaign traces before release.

