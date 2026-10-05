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
campaign="$(date -u +%Y%m%dT%H%M%S.%NZ)-multi-shot"
taskset -c 4,5 env PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family multi-shot-python \
  --campaign-id "$campaign" \
  --artifact report=validation/reports/multi_shot_campaign_soft_isolated.json \
  --artifact markdown=validation/reports/multi_shot_campaign_soft_isolated.md \
  -- python benchmarks/bench_multi_shot_campaign.py \
  --steps 2000 \
  --warmup 200 \
  --json-out validation/reports/multi_shot_campaign_soft_isolated.json \
  --md-out validation/reports/multi_shot_campaign_soft_isolated.md
```

Run native Rust local-regression evidence:

```bash
taskset -c 4,5 env PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family multi-shot-rust \
  --campaign-id "$campaign" \
  --artifact report=validation/reports/multi_shot_campaign_rust_soft_isolated.json \
  --artifact markdown=validation/reports/multi_shot_campaign_rust_soft_isolated.md \
  -- cargo run --manifest-path scpn-control-rs/Cargo.toml \
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

## Persisted benchmark report admission

`validation.validate_multi_shot_campaign_evidence.validate_multi_shot_campaign_evidence`
reads the Python/PyO3 `scpn-control.multi-shot-campaign-benchmark.v1.1` and
native Rust `scpn-control.rust-multi-shot-campaign-benchmark.v1.1` benchmark
report pair. These summaries differ from the campaign kernel report above.
The reader runs no campaign and writes no evidence. Repository defaults are
historical local-regression reports; their PASS grants no current-host readiness.

```bash
python validation/validate_multi_shot_campaign_evidence.py \
  --python-report python-report.json --rust-report rust-report.json \
  --minimum-digest-count 2 --json-out
```

Paths follow the caller's working directory and symlinks. Without `--json-out`,
the command prints status and ordered `ERROR` findings. Exit zero means reader
PASS, one refusal and two argument parsing failure. The root
`scpn-control validate` command consumes this reader by default. Its
`--multi-shot-campaign-python-report`, `--multi-shot-campaign-rust-report` and
`--multi-shot-min-digest-count` select reports/counts; the skip
option is limited to explicitly scoped import checks.

Each object must carry its expected schema, recognised `local_regression` or
`production_benchmark` class, boolean production flag, a command containing
`bench_multi_shot_campaign`, and a lowercase SHA-256 self-digest. The digest is
computed after replacing `payload_sha256` with an empty string and encoding all
fields as sorted, compact JSON. It detects inconsistent edits; a producer can
edit and reseal any declaration. Duplicate keys and nonfinite floating JSON
tokens, including overflowing exponents, are refused throughout the object.
Empty decoded objects fail required fields and retain their exact byte hashes.

Python must declare `pyo3_status="ok"` and both `result`/`pyo3_result` objects;
Rust must declare `result`. Each has positive non-boolean integer
`last_passed_count`, `stats.samples`, and
`last_pulsed_mpc_admission_digest_count` at least the requested positive integer
minimum. Individual decision bytes, sums, steps/count consistency and latency
values beyond positive sample counts are not checked or recomputed.

Python context requires a nonempty affinity list; native Rust context uses
nonblank text. Both require non-`None` `loadavg_start`/`loadavg_end`. The reader
does not inspect affinity elements, parse Linux affinity/load strings, validate
load values or host isolation, or reconcile contexts across reports. Native
producer sentinel strings are not independently qualified host observations.

The frozen result exposes status/errors, four exact-report/declared-payload
digests, the admitted surface set, declared PyO3 status, aggregate production
flag and effective minimum. `as_dict()` adds the admission v1 schema and fresh
mutable finding/surface lists. Only PASS lists `python`, `pyo3`, `rust`.
Malformed non-string digest/PyO3 declarations become `None`; invalid API minimum
arguments record FAIL and use fallback one. Successfully decoded objects retain
exact byte SHA-256 on FAIL; read/decode/non-object failure has no report digest.

Local regression cannot claim production. A production-class declaration may
keep its flag false. The aggregate flag reflects either report's literal true,
even when overall reader status is FAIL; consumers must inspect status and their
own admission evidence. No current-host qualification, producer authentication,
certified controller, machine-protection veto or facility action authority is
established. Additional metadata stays unchecked except its digest inclusion and
the decoder refusals. No filesystem containment or input size/depth budget is
provided.

## How to use the campaign orchestrator in a validation workflow

The orchestrator itself does not prove hardware correctness. It structures repeated shot handling so downstream validators can consume bounded evidence.

- run one short campaign first to confirm scheduler and bank handoff,
- preserve actual campaign reports and decision evidence with their digest fields,
- use the persisted benchmark reader for the benchmark summary pair and retain
  independent source/artifact custody evidence for any further admission.

The persisted benchmark reader does not convert campaign outputs into deployment
proof. Deployment/control admission requires its separately owned evidence and
authorisation boundaries.

## Practical use and scope

Use this guide when orchestrating repeated campaign runs across multiple shots.

- Review this before scheduling campaign batches or changing replay metadata expectations.
- Keep campaign orchestration consistent with control-runtime and data contracts in companion pages.
- Verify that output artifacts remain stable when scaling shot count.
