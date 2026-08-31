# Reactor regime-assessment admission

`scpn_control.reactor_semantic_admission` provides a separate, deterministic
CONTROL gate for a complete SCPN-PHASE-ORCHESTRATOR reactor-regime assessment.
The gate accepts only canonical SPO bytes and can admit an assessment only for
review. Its decision always has `review_only=true` and `actionable=false`.

This is not a regime classifier. It does not infer a label from MIF evidence,
select a controller, issue a command, or authorize a machine transition.

## End-to-end ownership

The verified reference chain has three explicit owners:

1. SCPN-MIF-CORE owns the canonical merge/compression observation envelope.
2. SCPN-PHASE-ORCHESTRATOR 1.3.1 validates that envelope, builds the semantic
   handoff, and emits an eight-axis assessment without classifying a regime.
3. SCPN-CONTROL independently checks the exact assessment identity, custody,
   freshness, and abstention policy before emitting its own sealed decision.

CONTROL calls only SPO's public `regime_assessment_from_bytes` decoder. It does
not import a sibling checkout, duplicate the upstream schema, or parse a looser
JSON representation.

The immutable reference artifacts are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| MIF source envelope | 2,475 | `c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca` |
| SPO semantic handoff | 101,652 | `c0f03b7c49346c39342598275556e8ac28c93138ba14f6e21d6739400e0edeb2` |
| SPO regime assessment | 11,943 | `3a5077b95d8b94b23a647d57a8b25f80cb798f712f00d0a34e71b95c600b154b` |

The assessment was produced by SPO tag source commit
`c2a7581d58819060806c6f173da941c822103695` using the public 1.3.1 wheel with
SHA-256
`c2d7c0a5c0ad47f420fee02e54ccc28122bf8d128eb3b80ca51ba5f034320274`.

## Exact policy

`ReactorRegimeAssessmentAdmissionPolicy` requires the caller to provide:

- the assessment SHA-256 and assessment ID;
- reactor context, configuration, and event IDs;
- producer project, revision, and artifact digest;
- source project, revision, handoff schema, handoff digest, and complete
  semantic-ID set;
- assessment schema version;
- digests over the four registry bindings, the complete clock/validity record,
  and all fields of the ordered eight-axis vector;
- the canonical axis-ID set and per-axis provenance IDs; and
- an explicit check time and maximum evidence age in nanoseconds.

No wall clock is read. The explicit check time makes replay deterministic. A
decision rejects evidence from the future, evidence older than the configured
limit, or a check outside the assessment's common validity interval.

The admission profile accepts only the abstaining output shape of SPO's
`build_abstaining_regime_assessment`; the caller's exact axis and provenance
policy plus the complete axis-custody digest bind the configuration-specific
vector. The reference `frc_compression_mif` receipt has seven `unknown` axes
and one statically `not_applicable` exhaust/boundary axis. The gate rejects
classified axes, attached classifier evidence, non-zero observability or
confidence, quality-state drift, provenance drift, and any custody mismatch.
The upstream codec owns the disposition-specific uncertainty invariants;
CONTROL's complete axis-custody digest binds those values without duplicating
the SPO schema.

## Example

```python
from pathlib import Path

from scpn_phase_orchestrator.reactor_semantics import regime_assessment_from_bytes
from scpn_control.reactor_semantic_admission import (
    ReactorRegimeAssessmentAdmissionPolicy,
    admit_reactor_regime_assessment,
    regime_assessment_admission_decision_to_bytes,
    regime_assessment_axis_custody_digest,
    regime_assessment_clock_custody_digest,
    regime_assessment_registry_custody_digest,
)

assessment_bytes = Path("reactor-regime-assessment.json").read_bytes()
assessment = regime_assessment_from_bytes(assessment_bytes)

policy = ReactorRegimeAssessmentAdmissionPolicy(
    expected_assessment_sha256="<verified assessment SHA-256>",
    expected_assessment_id=assessment.assessment_id,
    expected_reactor_context_id=assessment.reactor_context_id,
    expected_configuration=assessment.configuration,
    expected_event_id=assessment.event_id,
    expected_producer_project=assessment.producer_project,
    expected_producer_revision=assessment.producer_revision,
    expected_producer_artifact_sha256=assessment.producer_artifact_sha256,
    expected_source_project=assessment.source_project,
    expected_source_revision=assessment.source_revision,
    expected_source_handoff_schema=assessment.source_handoff_schema,
    expected_source_handoff_sha256=assessment.source_handoff_sha256,
    expected_source_semantic_ids=assessment.source_semantic_ids,
    expected_assessment_schema_version=assessment.schema_version,
    expected_registry_custody_sha256=(
        regime_assessment_registry_custody_digest(assessment)
    ),
    expected_clock_custody_sha256=(
        regime_assessment_clock_custody_digest(assessment)
    ),
    expected_axis_custody_sha256=(
        regime_assessment_axis_custody_digest(assessment)
    ),
    expected_axis_ids=tuple(axis.axis_id for axis in assessment.axes),
    expected_axis_provenance=tuple(
        (axis.axis_id, axis.provenance_id) for axis in assessment.axes
    ),
    checked_at_ns=assessment.evidence_timestamp_ns,
    max_evidence_age_ns=0,
)

decision = admit_reactor_regime_assessment(assessment_bytes, policy=policy)
portable_decision = regime_assessment_admission_decision_to_bytes(decision)
```

The example assumes that the assessment SHA-256 was obtained through an
independent trusted channel. In deployment-oriented review, persist all policy
identities and custody digests during offline provisioning and load those
pinned values rather than deriving expectations from the candidate being
admitted. The digest helpers are suitable for that provisioning step and for
auditable receipt generation.

The decision schema is
`scpn-control.reactor-regime-assessment-admission.v1`, version `1.0.0`. It has
an inner decision digest and an outer payload seal. If the SPO decoder rejects
the input, CONTROL retains only the raw byte digest and leaves every decoded
identity null rather than copying expected policy values into evidence.

## Safety boundary

Admission establishes that one exact upstream assessment is internally valid,
expected, fresh under the supplied deterministic time, and suitable for
non-actuating review. It does not establish physical diagnostic provenance,
facility-clock correlation, a trained or validated regime classifier,
closed-loop stability, actuator availability, machine protection, or a device
adapter. Those remain separate evidence and authority gates.
