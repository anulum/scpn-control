# Device diagnostic review admission

`scpn_control.reactor_semantic_admission` provides a dedicated CONTROL gate for
one exact device-diagnostic design review published by
SCPN-PHASE-ORCHESTRATOR 1.4.2 and verified unchanged under 1.4.3. The only input is the canonical sealed review
envelope accepted by SPO's public
`device_diagnostic_plan_review_from_bytes` decoder.

CONTROL does not parse or accept raw Tokamak manifests, diagnostic envelopes,
diagnostic plans, FAIR-MAST data, qualification records, or sibling-checkout
objects. The source documents embedded in the upstream review remain under
SPO's decoder and producer-custody validation; CONTROL reads only the typed
review fields returned by that public decoder.

## Exact published binding

The bundled Tokamak policy pins these immutable identities:

| Object | Identity |
|---|---|
| SPO distribution | `scpn-phase-orchestrator==1.4.3` |
| Public wheel SHA-256 | `5da94500760f9394a637f7edec044a844c12230d8d16c25a573b6e67a1ddb409` |
| Decoder module SHA-256 | `8e8c9cd3f253da1bf52983127fdb7d643822bbf14ec27e7081068e42642d3a33` |
| Review schema | `scpn-phase-orchestrator.device-diagnostic-plan-review.v1` |
| Review schema version | `1.1.0` |
| Review envelope SHA-256 | `de5573351115ae6d28787aadfec1106f037e5cf8c68206ea5074255055f5e795` |
| Review ID | `55cec93310c72602abefde47f200c0f3fb32c834fa57cd4c5ff70923c48ce53c` |
| Device producer | `SCPN-TOKAMAK-CORE` |
| Producer revision | `7402191c43e8fe57cffda1dd5b3cf4319d6d398d` |
| Producer wheel SHA-256 | `a0c6ccbf8c398d80ed65f03a82e7a313761d09dea81acb5ab8ad565997cb2720` |
| Manifest SHA-256 | `ed4dd4f86eb7a62bf9674c0bfffa341f3afed42b7754ddcd809bc0b1804a19ab` |
| Source envelope SHA-256 | `5a5aff510df9bdf35f76edfc9f147dae79e95f4edd98c3a610a60d8ea9e28a` |
| Source plan SHA-256 | `6a015adfaa2cda7ec1bf04fc685d00d6b7209ca9b78d761fff47ab0919eeec94` |
| Configurations | `conventional_tokamak`, `spherical_tokamak` |
| Clock-custody SHA-256 | `29eb15ecd40efb911b4150004fd7b0252ab2ee890c971e5aaab3d9abd3263478` |

The published wheel digest was independently verified before installation.
The installed package version and decoder source digest are checked on every
admission. The wheel archive digest is retained in the CONTROL decision as
release-custody evidence; an installed wheel cannot reproduce its original ZIP
container digest from extracted files.

## Clock custody

`device_diagnostic_review_clock_custody_digest` seals every ordered field of
all three upstream clock reviews:

- `clk_facility` is `facility_monotonic` and remains `unmapped`;
- `clk_shot` is `shot_event_epoch` and only event-relative compatible; and
- `clk_sim` is `simulation` and synthetic compatible.

All three have `mapping_evidence_claimed=false`. In particular, this contract
does not create a facility-to-shot correlation. Such a mapping needs its own
facility-owned evidence and independently versioned admission contract.

## Usage

```python
from pathlib import Path

from scpn_control.reactor_semantic_admission import (
    admit_device_diagnostic_plan_review,
    device_diagnostic_review_decision_to_bytes,
    tokamak_device_diagnostic_review_policy,
)

review_bytes = Path("tokamak-device-diagnostic-review.json").read_bytes()
decision = admit_device_diagnostic_plan_review(
    review_bytes,
    policy=tokamak_device_diagnostic_review_policy(),
)
portable_decision = device_diagnostic_review_decision_to_bytes(decision)
```

Use trusted provisioning to select policy identities. Never derive expected
digests or producer identity from the candidate bytes being admitted. A later
SPO or device release requires a new explicit policy and evidence receipt; it
must not silently widen the 1.4.3 binding.

## Refusal boundary

The public SPO decoder rejects malformed bytes, duplicate keys, noncanonical
encoding, schema-version drift, digest tamper, unknown stored clocks, embedded
source inconsistency, and contradictory authority. CONTROL converts those
failures into an identity-empty `review_decode_failed` decision.

After successful SPO decoding, CONTROL independently compares the exact review
digest and ID, device project and revision, producer artefact, all three source
digests, configuration set and clock-custody digest. Stale producer revisions,
wrong-device policies and all other mismatches remain explicit refusal codes.
An incompatible installed SPO version, missing public decoder, unreadable or
modified decoder source, or schema-constant drift also fails closed.

## Authority boundary

An accepted result means only that one exact synthetic design declaration is
eligible for review. Every CONTROL decision fixes the following fields:

```text
review_only = true
evidence_claimed = false
observation_claimed = false
measurement_claimed = false
facility_binding_claimed = false
classification_performed = false
semantic_ingress_declared = false
control_intent_created = false
actionable = false
execution_authorised = false
actuation_authorised = false
```

It does not establish physical diagnostic provenance, facility timing,
classifier validity, control-loop stability, actuator availability, machine
protection, HIL readiness, or deployment admission.

## Installation failure isolation

Public admission exports load their implementation owners on demand. The device
review gate remains importable when SPO is absent, so it can return a canonical
`spo_contract_unavailable` decision. An installed incompatible version is refused
as `spo_distribution_version_mismatch` before loading its decoder modules.
These failures never populate an unverified upstream identity.

The installation checks exercise a locally built CONTROL wheel with the actual
public SPO 1.4.3 wheel, an absent SPO distribution, and the incompatible public
1.3.1 wheel in separate environments. A wrong-device policy is refused in the
supported environment. Package-layout and hostile-facade subprocess probes are
separate tests of source-path and export corruption; they are not substitute
installation evidence. Those subprocess branches are recorded explicitly in the
coverage exception policy until subprocess coverage is merged.
