<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Runtime-bound formal safety certificate guide. -->

# Runtime-Bound Safety Certificate

The formal verifier (`scpn_control.scpn.formal_verification`) proves bounded
CTL/LTL safety and liveness obligations over a compiled `StochasticPetriNet` and
emits a `scpn-control.safety-certificate.v1` payload. That certifies the *control
logic*, but it binds only an opaque artefact digest — it does not tie the proof
to the exact thing that will run.

`scpn_control.scpn.runtime_safety_certificate` closes that gap. It binds a
holding formal certificate to a structured controller runtime identity and issues
a `scpn-control.runtime-safety-certificate.v1` wrapper that facility-facing
admission consumes. It is not a facility safety approval; it is bounded,
replayable evidence such approval would build on.

## What the certificate binds

A `ControllerRuntimeBinding` captures the exact deployable controller identity:

| Field | Meaning |
|---|---|
| `controller_id` | stable controller identifier |
| `controller_config` | the controller configuration (gains, limits, modes) |
| `petri_topology_sha256` | digest of the compiled Petri-net topology the proof ran on |
| `snn_parameters` | SNN / neuro-symbolic parameters |
| `solver_mode` | the control solver mode (for example `acados-rti`) |
| `runtime_target` | firmware / runtime target (`RuntimeTarget`) |
| `timing_envelope` | declared bounded-runtime assumptions (`TimingEnvelope`) |

`compute_petri_topology_digest` hashes the place ordering, each transition's name,
threshold and delay, and the compiled input/output incidence (arc weights,
including inhibitor arcs as negative input weights). Initial token densities are
excluded: the certificate binds the control *structure*, not a transient marking.

The `TimingEnvelope` records the bounded-runtime assumptions. It is schedulable
only when `0 < worst_case_response_us <= deadline_us <= control_period_us`, and
`proof_firing_depth` records the bounded transition depth the proof covered.

## Issuance, replay, and admission

* **Issue** — `issue_runtime_safety_certificate` issues the wrapper only after the
  binding's declared topology matches the net the proof actually ran on and the
  embedded formal certificate holds. It fails closed otherwise.
* **Replay** — `replay_runtime_safety_certificate` re-proves the obligations on the
  live net and accepts the replay only when the live topology still matches the
  binding, the fresh proof still holds, it covers the same checked specs, and it
  produces the same formal digest.
* **Admit** — `assert_runtime_certificate_admissible` fails closed unless, all on
  the declared stack: the certificate holds, its binding digest matches the live
  controller binding, its declared runtime target matches the live target, its
  timing envelope is schedulable, and the proof replay passed.

```python
from scpn_control.scpn.runtime_safety_certificate import (
    ControllerRuntimeBinding, RuntimeTarget, TimingEnvelope,
    compute_petri_topology_digest, issue_runtime_safety_certificate,
    replay_runtime_safety_certificate, assert_runtime_certificate_admissible,
)

binding = ControllerRuntimeBinding(
    controller_id="burn-ctl-1",
    controller_config={"kp": 1.2, "ki": 0.3},
    petri_topology_sha256=compute_petri_topology_digest(net),
    snn_parameters={"layers": 3},
    solver_mode="acados-rti",
    runtime_target=RuntimeTarget("rpi4-rt", "aarch64", "PREEMPT_RT", "gcc-13"),
    timing_envelope=TimingEnvelope(
        control_period_us=1000.0, worst_case_response_us=180.0,
        deadline_us=500.0, proof_firing_depth=4,
    ),
)
certificate = issue_runtime_safety_certificate(net, binding, formal_certificate=formal)
replay = replay_runtime_safety_certificate(net, certificate, reverify=lambda: reprove(net))
assert_runtime_certificate_admissible(
    certificate, live_binding=binding, live_runtime_target=binding.runtime_target, replay=replay,
)
```

## Claim boundary

The certificate's `claim_boundary` states it is not a facility safety approval or
hardware timing certificate: facility admission requires proof replay and a
target-runtime match on the declared hardware and software stack. Until the live
binding, runtime target, timing envelope, and proof replay all match on that
stack, no facility-facing safety claim is admitted.
