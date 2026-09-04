# Exact-current LIF runtime

SCPN Control can bind each compiled transition to SC-NeuroCore's versioned,
exact-current leaky-integrate-and-fire profile. This is an explicit stateful
execution mode. It does not replace or change `CompiledNet.lif_fire`, which
remains the established stateless Petri-transition gate.

## Contract boundary

The runtime accepts piecewise-constant current contributions for a duration in
milliseconds. For each transition, simultaneous contributions are summed at
the start of the tick by SC-NeuroCore. SC-NeuroCore then applies its closed-form
event-driven solver, inclusive threshold comparison, analytical event time,
hard reset, zero refractory duration, and IEEE-754 binary64 numerical contract.
No random-number generator participates in this profile.

Membrane voltage, shot-relative time, shot identity, and reset epoch persist
across `execute` calls. State changes only after every transition completes and
its packet passes strict deterministic replay. A failed call leaves every
transition at its previous state. `reset_shot` is the only reset boundary.

## Required SC-NeuroCore profile

The current CONTROL binding fails closed unless all of these identities match:

| Identity | Required value |
| --- | --- |
| Distribution | `sc-neurocore 3.16.0` |
| Profile | `sc_exact_current_hard_reset_lif_v1` |
| Profile schema | `sc-neurocore.exact-current-lif-profile.v1` |
| Profile canonical SHA-256 | `c667f3885f564dcf968febaf62125a86abaaee4758df792d5f06b0e82d1f121a` |
| Profile artefact SHA-256 | `8051be0ff173b0ff6434d3f5b54ab8a1c9f5078f62fddd3e359e6a77deb5c716` |
| Model source SHA-256 | `064be334316184e50a85fb82b1a804cdf1342bb927c39588b4d4105c7a087762` |
| Implementation commit | `bc76e5b3c217fec191534bb650685316e645ad34` |
| Contract delivery commit | `248e88a827acfe9be0d654855ae9d3b7d2dcd527` |

The profile's declared units are milliseconds, normalized voltage, normalized
current, and normalized resistance. A changed unit, schema, source, digest, or
producer identity is an incompatibility, not a conversion request.

## Compile and execute

```python
from scpn_control.scpn import (
    ExactCurrentLIFProfileBinding,
    ExactCurrentLIFTransitionTick,
    FusionCompiler,
    StochasticPetriNet,
)

net = StochasticPetriNet()
net.add_place("plasma_input", initial_tokens=1.0)
net.add_transition("shape_response", threshold=0.5)
net.add_arc("plasma_input", "shape_response", weight=1.0)

binding = ExactCurrentLIFProfileBinding.from_installed_reference()
compiled = FusionCompiler(bitstream_length=1024).compile(
    net,
    exact_current_lif_binding=binding,
    exact_current_lif_shot_id="shot-2026-001",
)

assert compiled.exact_current_lif_runtime is not None
execution = compiled.exact_current_lif_runtime.execute(
    (
        ExactCurrentLIFTransitionTick(
            duration_ms=5.0,
            transition_currents=((10.0,),),
        ),
        ExactCurrentLIFTransitionTick(
            duration_ms=20.0,
            transition_currents=((15.0, 15.0),),
        ),
    )
)

# Complete SC packets are retained, including every ordered state sample and
# threshold event. No event or state reduction is applied.
packet = execution.packets[0]
print(packet.packet_json)
print(packet.sha256)
```

The outer tuple in `transition_currents` follows `compiled.transition_names`.
Each inner tuple contains simultaneous contributions for that transition. A
tick must provide one inner tuple for every compiled transition.

## Checkpoint, restore, and reset

```python
runtime = compiled.exact_current_lif_runtime
assert runtime is not None

checkpoint = runtime.serialize_checkpoint()
continuation = runtime.execute(
    (ExactCurrentLIFTransitionTick(10.0, ((20.0,),)),)
)

runtime.restore_checkpoint(checkpoint)
replay = runtime.execute(
    (ExactCurrentLIFTransitionTick(10.0, ((20.0,),)),)
)
assert replay.to_json() == continuation.to_json()

runtime.reset_shot("shot-2026-002")
```

Checkpoint restore validates the CONTROL envelope, transition order, complete
SC state envelopes, profile digest, state schema, and a shared shot identity,
shot-relative time, and reset epoch before committing any state. A checkpoint
cannot splice transitions from different shot timelines. Unknown and duplicate
JSON members are rejected.

## Failure types

All boundary failures derive from `ExactCurrentLIFError` and carry a stable
class-level `code`:

- `ExactCurrentLIFUnavailableError`: the required installed SC API is absent;
- `ExactCurrentLIFBindingError`: profile, source, schema, unit, version, digest,
  or commit identity does not match;
- `ExactCurrentLIFInputError`: tick shape or numerical domain is invalid;
- `ExactCurrentLIFStateError`: checkpoint or explicit reset is invalid; and
- `ExactCurrentLIFExecutionError`: the bound SC executor or strict packet replay
  rejects an execution.

## Claim boundary

This contract demonstrates deterministic software execution against the exact
SC-NeuroCore reference profile and immutable multi-tick packet. It does not by
itself establish NEST or Brian2 parity, MIF AER transport, RTL equivalence,
post-route timing, hardware-in-the-loop behavior, biological fidelity, or
machine-safety admission. Stochastic or adaptive neuron models require their
own profile and evidence and must not be admitted through this deterministic
contract.
