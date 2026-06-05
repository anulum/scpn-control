<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — PREEMPT_RT Runtime Admission -->

# PREEMPT_RT Runtime Admission

`CON-C.7` separates a fast native control loop from a production-admissible
runtime claim. The native hot path may run in microseconds, but SCPN Control only
admits production timing evidence when the host runtime contract is explicit.

The admission report records:

- Linux kernel release and PREEMPT_RT or `/sys/kernel/realtime` evidence.
- Current process CPU affinity and the requested SNN, Z3, network, and heartbeat cores.
- Scheduler policy and priority for the calling process.
- CPU governor state for requested cores.
- `RLIMIT_MEMLOCK` soft and hard limits.
- Heartbeat dead-man switch configuration.
- Native PyO3 runtime snapshot when the extension exposes it.

## Policies

`run-hardware-campaign` exposes `--runtime-admission-policy`:

| Policy | Behavior |
| --- | --- |
| `off` | Emit a schema-valid skipped admission record. Use only for local diagnostics. |
| `warn` | Run admission and continue even when the host is not production-qualified. This is the default for developer workstations. |
| `require` | Fail closed before campaign execution if PREEMPT_RT, real-time scheduler, performance governor, core affinity, heartbeat, or memory-lock requirements are not met. |

`warn` evidence must be labelled as local regression evidence. It must not be
used to claim hard real-time PCS timing.

## Strict production example

```bash
sudo chrt -f 99 taskset -c 4,5,6,7 \
  env PYTHONPATH=src python -m scpn_control.cli run-hardware-campaign \
    --execution-backend native \
    --pacing-mode spin \
    --formal-mode aot_certificate \
    --runtime-admission-policy require \
    --heartbeat-port 5556 \
    --core-snn 4 \
    --core-z3 5 \
    --core-net 6 \
    --core-hb 7 \
    --steps 5000 \
    --tick-interval-s 0.0001 \
    --json-out
```

If any runtime condition is missing, the command fails before the native loop is
entered. That fail-closed behavior is intentional.

## Python API

```python
from scpn_control.core.runtime_admission import (
    RuntimeAdmissionRequest,
    collect_runtime_admission,
)

report = collect_runtime_admission(
    RuntimeAdmissionRequest(
        execution_backend="native",
        pacing_mode="spin",
        native_backend_available=True,
        require_preempt_rt=True,
        require_realtime_scheduler=True,
        require_performance_governor=True,
        require_heartbeat=True,
        heartbeat_port=5556,
    )
)
```

The returned report is JSON-serializable and includes
`production_claim_allowed`. That field remains `false` unless strict production
requirements were requested and passed.

`NeuroCyberneticEngine.extract_slab_telemetry()` also carries the most recent
`runtime_admission` record. Emergency telemetry and post-run diagnostic dumps
therefore preserve the launch-time scheduler, affinity, governor, heartbeat,
and PREEMPT_RT assumptions that governed the campaign.

## Runtime admission role in deployments

This document defines what must pass before code moves from local execution to campaign-grade orchestration.

A safe campaign run requires:

- policy check for required artifacts,
- environment checks that match the target mode,
- and explicit handling of missing or failed admission checks.

The runtime admission surface is the final gate in the execution boundary and should be treated as non-optional in any hardening flow.
