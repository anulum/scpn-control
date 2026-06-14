<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Multi-shot campaign orchestration guide. -->

# Multi-Shot Campaign Orchestration

The multi-shot campaign orchestrator runs repeated pulsed-shot admission over
the CONTROL scheduler, capacitor-bank telemetry contract, and replay v1.1
metadata fields. It is a campaign-control adapter. It does not simulate the
plasma, replace facility shot sequencing, or add a second physics solver.

## What the adapter checks

For each shot, the adapter:

1. Resets a fresh `PulsedScenarioScheduler`.
2. Initialises a bounded `CapacitorBank` state from the declared bank
   specification.
3. Feeds timestamped plasma and bank telemetry samples through scheduler guards.
4. Records command rows and transition rows.
5. Requires the canonical lifecycle by default:
   `ramp_up -> flat_top -> burn -> expansion -> dump -> recharge -> cool_down -> idle`.
6. Emits replay-compatible pulse metadata: `pulse_id`,
   `capacitor_state_initial_J`, `trigger_timestamp_ns`, `energy_recovered_J`,
   and sorted `shot_phase_log` rows.
7. Preserves optional per-shot pulsed-MPC decision evidence through
   `pulsed_mpc_admission_digest` and
   `pulsed_mpc_evidence_schema_version` when the campaign consumes an admitted
   pulsed-MPC command.

Per-shot failures are fail-closed and do not abort the remaining campaign
unless campaign-level input is malformed, such as duplicate shot IDs.
The pulsed-MPC digest is a replay provenance binding. It does not admit a
facility interlock, target-hardware actuator path, or PCS timing claim.

## Python surface

```python
from scpn_control.control.multi_shot_campaign import (
    CampaignShotPlan,
    CampaignShotSample,
    MultiShotCampaignOrchestrator,
)

orchestrator = MultiShotCampaignOrchestrator(
    "campaign-a",
    scheduler_spec,
    bank_spec,
)

report = orchestrator.run(
    [
        CampaignShotPlan(
            shot_id="shot-001",
            samples=tuple(samples),
            initial_bank_voltage_V=5000.0,
            pulsed_mpc_admission_digest=admitted_mpc_decision.admission_digest,
        )
    ]
)
```

The returned report uses schema version
`scpn-control.multi-shot-campaign.v1` and includes a SHA-256 payload digest.
If any shot supplies `pulsed_mpc_admission_digest`, the report also records
`pulsed_mpc_admission_digest_count` and binds each digest into the report
payload hash.

## Rust and PyO3 surfaces

The Rust kernel lives in `control_control::multi_shot_campaign` and exposes:

- `CampaignShotSample`
- `CampaignShotPlan`
- `MultiShotCampaignOrchestrator`
- `MultiShotCampaignReport`

The optional PyO3 bridge exposes `PyMultiShotCampaignOrchestrator.run_table()`.
It accepts table-shaped NumPy arrays for sample index, sample time, plasma
telemetry, bank telemetry, initial bank voltages, and optional
`pulsed_mpc_admission_digests`. This keeps the bridge explicit about units,
array shapes, and evidence handoff.

## Benchmarks

Run Python local-regression evidence:

```bash
taskset -c 4,5 env PYTHONPATH=src python benchmarks/bench_multi_shot_campaign.py \
  --steps 2000 \
  --warmup 200 \
  --json-out validation/reports/multi_shot_campaign_soft_isolated.json \
  --md-out validation/reports/multi_shot_campaign_soft_isolated.md
```

Run native Rust local-regression evidence:

```bash
taskset -c 4,5 cargo run --manifest-path scpn-control-rs/Cargo.toml \
  -p control-control \
  --example bench_multi_shot_campaign \
  --release \
  -- \
  --steps 2000 \
  --warmup 200 \
  --json-out validation/reports/multi_shot_campaign_rust_soft_isolated.json \
  --md-out validation/reports/multi_shot_campaign_rust_soft_isolated.md
```

Soft-affinity workstation reports must keep `production_claim_allowed=false`.
Production timing claims require explicit core isolation, host-load context, and
target-runtime evidence.

## How to use the campaign orchestrator in a validation workflow

The orchestrator itself does not prove hardware correctness. It structures repeated shot handling so downstream validators can consume bounded evidence.

- run one short campaign first to confirm scheduler and bank handoff,
- persist campaign reports with digest fields,
- run the matching validator to convert the campaign output into admissible claims.

Do not use multi-shot campaign outputs for deployment proof unless the strict campaign validator and execution context are included.

## Practical use and scope

Use this guide when orchestrating repeated campaign runs across multiple shots.

- Review this before scheduling campaign batches or changing replay metadata expectations.
- Keep campaign orchestration consistent with control-runtime and data contracts in companion pages.
- Verify that output artifacts remain stable when scaling shot count.
