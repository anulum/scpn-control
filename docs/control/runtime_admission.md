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
- Authenticated transport-liveness heartbeat configuration.
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
install -m 600 /secure/source/heartbeat.key /run/secrets/scpn-heartbeat.key

sudo chrt -f 99 taskset -c 4,5,6,7 \
  env PYTHONPATH=src \
    SCPN_CONTROL_HEARTBEAT_KEY_FILE=/run/secrets/scpn-heartbeat.key \
    SCPN_CONTROL_HEARTBEAT_ALLOWED_SOURCE=127.0.0.1 \
    SCPN_CONTROL_HEARTBEAT_BIND_HOST=127.0.0.1 \
    python -m scpn_control.cli run-hardware-campaign \
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

When heartbeat monitoring is enabled, the Rust bridge refuses to start unless
`SCPN_CONTROL_HEARTBEAT_KEY_FILE` names a non-symlink regular file containing
32–64 raw key bytes with no group/other permissions on POSIX hosts and
`SCPN_CONTROL_HEARTBEAT_ALLOWED_SOURCE` is one exact source IP. The bind host
defaults to `127.0.0.1`; set `SCPN_CONTROL_HEARTBEAT_BIND_HOST` explicitly for a
non-loopback interface. Raw key bytes do not enter CLI arguments or telemetry.

The accepted datagram is exactly 48 bytes: ASCII magic `SCPNHB01`, a strictly
increasing unsigned 64-bit big-endian counter, and the full HMAC-SHA256 tag over
those first 16 bytes. Use `load_transport_heartbeat_key()` and
`build_transport_heartbeat_frame()` to construct sender packets. Wrong source,
length, magic, tag, zero/repeated/reordered counter within one receiver
lifetime, or malformed configuration never refreshes liveness. Sender and
receiver must establish a new counter epoch after a receiver restart; this
one-way hint does not claim durable cross-restart replay protection.

This heartbeat is a bounded transport-liveness hint. It is not an independent
machine-protection interlock, safety function, or PCS deployment credential.

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

## Practical use and scope

Use this page to define what enters the production-timing admissible path.

- Apply these rules before running hot-path validation in native or threaded execution modes.
- Keep runtime admission checks consistent across benchmarking and campaign scripts.
- When a timing claim changes, revalidate both scheduler and hardware-admission assumptions.
