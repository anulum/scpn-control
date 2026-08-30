# Reactor semantic admission

`scpn_control.reactor_semantic_admission` is CONTROL's portable, deterministic
gate for reactor semantic evidence owned by SCPN-PHASE-ORCHESTRATOR. It admits a
handoff only for review. It never grants actuation authority and never creates,
imports, forwards, or serializes a control action.

## Boundary and ownership

The exchange has three distinct owners:

1. SCPN-FUSION-CORE produces canonical model-evidence bytes with the physical
   values, units, simulation clock, calibration declaration, numerical
   refinement uncertainty, and provenance.
2. SCPN-PHASE-ORCHESTRATOR verifies those bytes, resolves reactor identity
   through its U0 registry, and assigns nonphase bounded-feature semantics.
3. SCPN-CONTROL consumes SPO's canonical handoff bytes and emits a sealed
   admission decision.

CONTROL calls the public SPO `handoff_from_bytes` function directly. It does not
vendor the schema, decode a looser JSON representation, copy sibling source, or
construct its own reactor registry.

## Deterministic policy

Every admission call supplies a `ReactorSemanticAdmissionPolicy` containing:

- the exact expected handoff SHA-256;
- the expected FUSION schema, 40-character producer revision, and embedded
  source-envelope SHA-256;
- an explicit `ClockReference` with domain, kind, epoch, and timestamp;
- inclusive evidence-age and calibration-age limits in nanoseconds;
- complete calibration-ID and transfer-function-ID allowlists; and
- optional degraded-validity-reason and degraded-quality-flag allowlists.

No wall clock is read. Empty degradation allowlists reject degraded evidence.
Observable validity `VALID` and quality `VALID` pass. `DEGRADED` passes only
when every declared reason or flag is allowlisted. Observable validity
`UNKNOWN`, `STALE`, `OUT_OF_DISTRIBUTION`, `UNOBSERVABLE`, or `INVALID`, and
observable quality `UNKNOWN` or `INVALID`, always reject.

The semantic records are different from the observable descriptors. For this
transport exchange, every semantic record is intentionally a `bounded_feature`
with phase validity `UNOBSERVABLE`, quality `UNKNOWN`, the
`noncyclic_transport_evidence` flag, zero phase confidence and observability,
and no phase fields. This is the expected statement that no cyclic phase was
declared. It does not make a valid transport observable unusable. The UNKNOWN
regime is likewise expected because no regime classifier was supplied.

## Example

```python
from pathlib import Path

from scpn_phase_orchestrator.reactor_semantics import ClockKind, ClockReference
from scpn_control.reactor_semantic_admission import (
    ReactorSemanticAdmissionPolicy,
    admission_decision_to_bytes,
    admit_reactor_semantic_handoff,
)

handoff_bytes = Path("reactor-semantic-handoff.json").read_bytes()
expected_handoff_sha256 = Path("reactor-semantic-handoff.sha256").read_text().strip()
expected_source_revision = Path("fusion-source-revision.txt").read_text().strip()
expected_source_envelope_sha256 = Path("fusion-source-envelope.sha256").read_text().strip()
policy = ReactorSemanticAdmissionPolicy(
    expected_handoff_sha256=expected_handoff_sha256,
    expected_source_schema="scpn-fusion-core.torax-runtime-review-envelope.v1",
    expected_source_revision=expected_source_revision,
    expected_source_envelope_sha256=expected_source_envelope_sha256,
    reference_clock=ClockReference(
        domain="simulation_monotonic",
        kind=ClockKind.SIMULATION_MONOTONIC,
        epoch="scenario_start",
        timestamp_ns=20_000_000,
        sample_rate_hz=100.0,
        latency_s=0.0,
        picosecond_offset=0,
        synchronized_to=None,
    ),
    max_evidence_age_ns=0,
    max_calibration_age_ns=20_000_000,
    allowed_calibration_ids=frozenset(
        {"fusion.torax.simulation_declared_units.v1"}
    ),
    allowed_transfer_function_ids=frozenset(
        {"fusion.torax.identity_projection.v1"}
    ),
)

decision = admit_reactor_semantic_handoff(handoff_bytes, policy=policy)
portable_decision = admission_decision_to_bytes(decision)
```

The output schema is `scpn-control.reactor-semantic-admission.v1` version
`1.0.0`. It contains the admitted or rejected state, `review_only=true`,
`actionable=false`, the explicit check timestamp, safely decoded identities,
sorted unique refusal codes, a decision digest, and an outer payload seal.
Decoder failures leave upstream identity fields null rather than copying caller
expectations into evidence.

## What this does not establish

Admission does not define a plant, actuator, diagnostic, or controller. Control
and disturbance channels, saturation, slew, latency, sampled dynamics, resets,
failure semantics, facility safety, performance, and closed-loop guarantees
remain unavailable until a named FUSION plant contract supplies and validates
them. A successful review decision is evidence for human and software review,
not permission to issue a command.
