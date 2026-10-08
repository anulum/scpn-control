# API Reference

## Shared Reference Location Declarations

`validation.reference_uri.reference_artifact_uri_error(value, field)` and
`external_executable_path_error(value, field="binary_path")` return an authored
field finding or `None`. These pure lexical checks perform no filesystem or
network reads and authenticate no reference, executable, digest or model run.
Non-string/empty input, ASCII C0/DEL and parser failures produce findings.
Surrounding ordinary spaces are ignored. Findings preserve the selected field.

Artifact locations require an explicit `file`, `https`, `s3` or `gs` scheme.
File URIs require no host and a path under `/validation/reports/` or
`/validation/reference_data/`; that prefix check alone can accept a directory.
Remote URIs require an authority and path. Literal POSIX `..` components are
refused; percent escapes, credentials, query strings and fragments are neither
decoded nor authenticated. Passing a location check proves no stable download.

Executable declarations require an absolute POSIX path, no URI authority,
literal parent component, trailing slash or final `.` component, and one of the
deployment/facility prefixes in the owner module. Mutable/system-control roots
are refused. No existence, executable permission, symlink target, binary version
or byte checksum is checked. Callers retain custody and scientific admission.
These APIs have no mutable state, cache or lock and do not resolve Windows paths.

The density declaration reader and registered `validate-density-reference`
command retain malformed URI findings under `reference_artifact_uri`; the
command emits a failing JSON report and exits 1. A malformed IPv6 authority
therefore remains a field finding instead of exposing parser exception text.

```python
from validation.reference_uri import reference_artifact_uri_error

assert reference_artifact_uri_error("https://[", "reference") == (
    "reference must be a syntactically valid URI"
)
```

## Top-Level Exports

```python
import scpn_control

scpn_control.__version__       # "0.23.0"
scpn_control.FusionKernel      # Grad-Shafranov equilibrium solver
scpn_control.RUST_BACKEND      # True if Rust acceleration available
scpn_control.TokamakConfig     # Preset tokamak geometries
scpn_control.StochasticPetriNet
scpn_control.FusionCompiler
scpn_control.CompiledNet
scpn_control.NeuroSymbolicController
scpn_control.kuramoto_sakaguchi_step
scpn_control.order_parameter
scpn_control.KnmSpec
scpn_control.build_knm_paper27
scpn_control.UPDESystem
scpn_control.LyapunovGuard
scpn_control.RealtimeMonitor
scpn_control.PhysicsDebugAssistant
scpn_control.ReactorSemanticAdmissionPolicy
scpn_control.ReactorSemanticAdmissionDecision
scpn_control.admit_reactor_semantic_handoff
```

---

## SCPN — Stateful exact-current LIF runtime

`scpn_control.scpn.exact_current_lif_runtime` consumes the immutable
SC-NeuroCore exact-current profile through its public solver classes. The
opt-in compiler binding creates one persistent session per compiled transition,
preserves complete state across calls, commits multi-transition execution
atomically, and retains every ordered SC state sample and event in canonical
packets. It is separate from the established stateless `CompiledNet.lif_fire`
gate.

Use the [Exact-current LIF runtime](guides/exact_current_lif_runtime.md) guide
for the required hashes and commits, multi-transition input shape,
checkpoint/replay workflow, typed failures, and evidence limits.

::: scpn_control.scpn.exact_current_lif_runtime

---

## Reactor Semantic Review Admission

`scpn_control.reactor_semantic_admission` consumes only the canonical portable
bytes decoded by
`scpn_phase_orchestrator.reactor_semantics.handoff_from_bytes`. The caller
supplies the expected handoff and embedded FUSION digests, exact producer
revision and schema, a reference clock, freshness limits, and calibration and
degradation allowlists. The result is deterministic, digest-sealed,
`review_only=true`, and `actionable=false`.

The SPO decoder owns strict UTF-8, duplicate-key, canonical-byte, U0, registry,
source-envelope, nonphase semantic, empty-relation, UNKNOWN-regime, and action
refusal. CONTROL then checks independent expected identities, clock freshness,
observable validity and quality, calibration declarations, and FUSION
provenance. The expected semantic `UNOBSERVABLE` state means that no cyclic
phase was declared; it does not invalidate otherwise usable transport evidence.

Use the dedicated [Reactor Semantic Admission](control/reactor_semantic_admission.md)
guide for the policy fields, refusal boundary, and non-actuation limits.

MIF merge-compression review admission is intentionally exposed only from the
reactor-semantic subpackage. It is not a root-package control API:

```python
from scpn_control.reactor_semantic_admission import (
    DeviceDiagnosticReviewAdmissionPolicy,
    MIFReactorSemanticAdmissionPolicy,
    ReactorRegimeAssessmentAdmissionPolicy,
    admit_device_diagnostic_plan_review,
    admit_reactor_regime_assessment,
    admit_mif_reactor_semantic_handoff,
    tokamak_device_diagnostic_review_policy,
)
```

The regime-assessment API is a second, separate review gate over SPO's complete
eight-axis assessment bytes. It pins exact producer, source, registry, clock,
axis, provenance, freshness, and abstention custody and emits its own canonical
sealed, non-actionable decision. See
[Reactor Regime-Assessment Admission](control/reactor_regime_assessment_admission.md).

The sealed device-diagnostic review API is a third independent gate. It uses
only SPO 1.4.2's public decoder, binds the immutable Tokamak review and producer
custody, and emits a separate canonical decision whose evidence, observation,
measurement, facility-binding, classification, semantic-ingress, intent,
actionability, execution and actuation fields are all false. See
[Device Diagnostic Review Admission](control/device_diagnostic_review_admission.md).

::: scpn_control.reactor_semantic_admission.admission

::: scpn_control.reactor_semantic_admission.decision

::: scpn_control.reactor_semantic_admission.mif_admission.MIFReactorSemanticAdmissionPolicy

::: scpn_control.reactor_semantic_admission.mif_admission.admit_mif_reactor_semantic_handoff

::: scpn_control.reactor_semantic_admission.regime_assessment_admission.ReactorRegimeAssessmentAdmissionPolicy

::: scpn_control.reactor_semantic_admission.regime_assessment_admission.admit_reactor_regime_assessment

::: scpn_control.reactor_semantic_admission.regime_assessment_admission.regime_assessment_registry_custody_digest

::: scpn_control.reactor_semantic_admission.regime_assessment_admission.regime_assessment_clock_custody_digest

::: scpn_control.reactor_semantic_admission.regime_assessment_admission.regime_assessment_axis_custody_digest

::: scpn_control.reactor_semantic_admission.regime_assessment_decision.ReactorRegimeAssessmentAdmissionStatus

::: scpn_control.reactor_semantic_admission.regime_assessment_decision.ReactorRegimeAssessmentAdmissionDecision

::: scpn_control.reactor_semantic_admission.regime_assessment_decision.regime_assessment_admission_decision_to_bytes

::: scpn_control.reactor_semantic_admission.regime_assessment_decision.regime_assessment_admission_decision_from_bytes

::: scpn_control.reactor_semantic_admission.regime_assessment_decision.regime_assessment_admission_decision_digest

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_admission.DeviceDiagnosticReviewAdmissionPolicy

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_admission.tokamak_device_diagnostic_review_policy

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_admission.admit_device_diagnostic_plan_review

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_admission.device_diagnostic_review_clock_custody_digest

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_decision.DeviceDiagnosticReviewAdmissionStatus

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_decision.DeviceDiagnosticReviewAdmissionDecision

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_decision.device_diagnostic_review_decision_to_bytes

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_decision.device_diagnostic_review_decision_from_bytes

::: scpn_control.reactor_semantic_admission.device_diagnostic_review_decision.device_diagnostic_review_decision_digest

---

## SCPN — AER Control Observation

`scpn_control.scpn.observation` adapts asynchronous Address-Event
Representation spike streams into bounded controller features while preserving
the existing mapping-based observation contract. The Python surface provides
`SpikeEvent`, `SpikeBuffer`, rate/temporal/ISI decoders, and
`AERControlObservation.to_features()`. `NeuroSymbolicController.step()` accepts
either the existing `Mapping[str, float]` observation or an
`AERControlObservation`; typed AER input is decoded through
`AERControlObservation.to_feature_mapping()` before the same feature-injection
path runs. When JSONL controller logging is enabled, the record includes the
decoded `obs` mapping plus `aer_admission` metadata from
`SpikeBuffer.admission_report()`. The matching Rust implementation lives in
`control_core::spike_buffer`, with optional PyO3 bindings exposed as
`scpn_control_rs.PySpikeBuffer` and `scpn_control_rs.aer_decode_*`.
`monotonic_input` and `out_of_order_event_count` provide bounded ingress
evidence. Observations may set `require_monotonic=True` to fail closed before
controller injection if upstream AER timestamps violate the monotonic admission
contract.

This is an ingress adapter and feature-decoding contract. It does not claim
hardware AER signal integrity, target neuromorphic-device deployment, FPGA
timing closure, or PCS admission without separate hardware evidence.

::: scpn_control.scpn.observation.SpikeEvent

::: scpn_control.scpn.observation.SpikeBuffer

::: scpn_control.scpn.observation.AERControlObservation

::: scpn_control.scpn.observation.decode_rate

::: scpn_control.scpn.observation.decode_temporal

::: scpn_control.scpn.observation.decode_isi

---

## SCPN — Geometry-Neutral Replay Schema

`scpn_control.scpn.geometry_neutral_replay` publishes deterministic
geometry-neutral replay reports and schema admission helpers. The v1.1 schema
extends the v1 report with optional pulsed-shot context: UUID pulse IDs,
capacitor initial energy, trigger timestamp, recovered energy, sorted
shot-phase logs, FRC diagnostic scalars, and digest-bound AER admission
metadata. Existing v1 reports remain loadable under the v1.1 schema bundle.

Use the dedicated [Geometry-Neutral Replay](scpn/geometry_neutral_replay.md)
guide for field semantics, example payloads, and claim boundaries.

::: scpn_control.scpn.geometry_neutral_replay.generate_report

::: scpn_control.scpn.geometry_neutral_replay.validate_report

::: scpn_control.scpn.geometry_neutral_replay.load_replay_schema

::: scpn_control.scpn.geometry_neutral_replay.register_v1_1_schema

::: scpn_control.scpn.geometry_neutral_replay.assert_v1_replay_loadable_under_v1_1_schema_bundle

::: scpn_control.scpn.geometry_neutral_replay.build_aer_admission_metadata

::: scpn_control.scpn.geometry_neutral_replay.attach_aer_admission_metadata

::: scpn_control.scpn.geometry_neutral_replay.save_geometry_neutral_replay_report

::: scpn_control.scpn.geometry_neutral_replay.load_geometry_neutral_replay_report

---

## Control — Pulsed Scenario Scheduler v2

`scpn_control.control.pulsed_scenario_scheduler_v2` owns the reusable
pulsed-fusion lifecycle contract that MIF-CORE incubated as MIF-004. The
scheduler models the adjacent state ring:

`idle -> ramp_up -> flat_top -> burn -> expansion -> dump -> recharge -> cool_down -> idle`

The Python surface provides the control-plane API and audit-log contract. The
matching Rust kernel lives in `control_control::pulsed_scenario` for the
compiled hot-path lane. When the optional extension is built,
`scpn_control_rs.PyPulsedScenarioScheduler` exposes that Rust kernel directly
to Python without changing the pure-Python API. All surfaces use the same state
names, action names, guard thresholds, monotone timestamp checks, and
transition reasons.

`scpn_control.control.pulsed_scenario_scheduler` is retained as the
SCPN-MIF-CORE compatibility import and re-exports the v2 implementation. The
finite-state topology is also captured in `lean/SCPNControl/PulsedFSM.lean`,
including the liveness theorem
`pulsed_fsm_eventually_returns_to_idle`.

The scheduler is a generic pulsed-reactor controller primitive. It is not a
facility-validated PCS implementation by itself. Hardware timing, actuator
mapping, capacitor-bank plant dynamics, and measured-shot validation remain
separate admission surfaces.

::: scpn_control.control.pulsed_scenario_scheduler

::: scpn_control.control.pulsed_scenario_scheduler_v2.PulsedScenarioState

::: scpn_control.control.pulsed_scenario_scheduler_v2.PulsedScenarioAction

::: scpn_control.control.pulsed_scenario_scheduler_v2.PulsedScenarioSpec

::: scpn_control.control.pulsed_scenario_scheduler_v2.PulsedPlasmaTelemetry

::: scpn_control.control.pulsed_scenario_scheduler_v2.CapacitorBankTelemetry

::: scpn_control.control.pulsed_scenario_scheduler_v2.PulsedScenarioTransition

::: scpn_control.control.pulsed_scenario_scheduler_v2.PulsedScenarioCommand

::: scpn_control.control.pulsed_scenario_scheduler_v2.PulsedScenarioScheduler

---

## Control — Capacitor Bank State Model

`scpn_control.control.capacitor_bank_state` owns the CONTROL-side bounded
series-RLC capacitor-bank contract for pulsed-shot admission. It mirrors the
MIF-CORE MIF-005 capacitor-bank mathematics at the control boundary: damping
regime classification, closed-form free response, Crank-Nicolson stepping,
midpoint-sampled discharge waveforms, conservative feasibility guards, and
constant-power recharge projection.

The state equation is:

```text
d/dt [v_C, i]^T = [[0, -1/C], [1/L, -R/L]] [v_C, i]^T + [-i_load/C, 0]^T
```

where `C` is capacitance in farads, `L` is inductance in henries, `R` is series
resistance in ohms, `v_C` is capacitor voltage in volts, `i` is series current
in amperes, and `i_load` is the prescribed external load current. The numerical
step uses a Crank-Nicolson update so natural-response validation can compare
against the analytical underdamped, critical, and overdamped solutions.

`CapacitorBank.telemetry()` adapts the state model to
`PulsedScenarioScheduler v2` by emitting scheduler-compatible capacitor-bank
telemetry with absolute voltage magnitude, declared voltage limit, and stored
energy. The matching Rust kernel lives in
`control_control::capacitor_bank`. When the optional extension is built,
`scpn_control_rs.PyCapacitorBankModel`, `scpn_control_rs.PyCapacitorBankSpec`,
and `scpn_control_rs.capacitor_bank_free_response()` expose the compiled
surface directly to Python.

`scpn_control.control.capacitor_bank` is retained as the SCPN-MIF-CORE
compatibility import and re-exports the state-model implementation without
duplicating the RLC mathematics.

`CapacitorBank.discharge()` now reports an explicit total RLC energy ledger for
admission and replay: initial total stored energy, remaining total stored
energy, remaining capacitor electric energy, remaining inductor magnetic energy,
integrated ohmic loss, integrated prescribed-load extraction, absolute residual,
relative residual, and a boolean pass/fail flag. The residual contract is:

```text
energy_initial - energy_remaining
  = resistive_loss + load_energy + energy_balance_residual
```

The ledger uses midpoint quantities from the same Crank-Nicolson step that
advances the state, so the residual is a numerical consistency check on the
CONTROL admission model rather than a facility hardware protection claim.
`CapacitorBankState.energy_J` remains scheduler-facing capacitor electric
energy only.

This model is a bounded control admission and scheduling primitive. It is not a
validated facility capacitor-bank driver, insulation model, switch model, or
hardware interlock implementation. Facility deployment still requires target
hardware timing, protection relays, isolation evidence, and shot-matched
validation artefacts.

::: scpn_control.control.capacitor_bank

::: scpn_control.control.capacitor_bank_state.RLCRegime

::: scpn_control.control.capacitor_bank_state.CapacitorBankSpec

::: scpn_control.control.capacitor_bank_state.CapacitorBankState

::: scpn_control.control.capacitor_bank_state.PulseSpec

::: scpn_control.control.capacitor_bank_state.EnergyReport

::: scpn_control.control.capacitor_bank_state.free_response

::: scpn_control.control.capacitor_bank_state.CapacitorBank

---

## Control — Pulsed-Shot MPC Admission Adapter

`scpn_control.control.fusion_neural_mpc.PulsedShotMPCAdapter` wraps the
CONTROL-owned gradient MPC surface and admits its first action through the
pulsed-scenario scheduler and capacitor-bank feasibility guard. It is a
control-boundary adapter over existing MPC output, not a new equilibrium solver
or a duplicate solver lane.

The adapter applies three deterministic checks:

- Non-`burn` scheduler states replace selected burn-action components with the
  configured safe action.
- `burn` state evaluates a `PulseSpec` against `CapacitorBank.feasibility()`
  unless the caller disables that policy.
- Every step records an explainable decision dictionary with scheduler state,
  capacitor feasibility text, constraint slack, MPC objective, whether a safe
  action was applied, and a digest-bound admission evidence payload.

The matching Rust kernel lives in `control_control::mpc::MPController` as
`plan_pulsed()`. When the optional PyO3 extension is rebuilt,
`scpn_control_rs.PyMpcController.plan_pulsed()` exposes the same admission
fields to Python, including `evidence_schema_version`, `action_sha256`,
`safe_action_sha256`, `burn_action_mask_sha256`, `peak_current_A`, and
`admission_digest`.

Use the dedicated [Pulsed MPC Adapter](control/pulsed_mpc_adapter.md) guide for
examples, runtime boundaries, and benchmark evidence commands.

::: scpn_control.control.fusion_neural_mpc.PulsedShotMPCDecision

::: scpn_control.control.fusion_neural_mpc.PulsedShotMPCAdapter

---

## Control — Multi-Shot Campaign Orchestrator

`scpn_control.control.multi_shot_campaign` runs repeated pulsed-shot admission
over the scheduler, capacitor-bank telemetry, and replay v1.1 metadata
contracts. It accepts explicit shot telemetry, records command and transition
logs, requires the canonical pulsed lifecycle by default, and emits a
digest-bound campaign report. When supplied, per-shot
`pulsed_mpc_admission_digest` values are validated as lowercase SHA-256
digests and preserved in the campaign report plus replay v1.1 extension fields.

The matching Rust kernel lives in `control_control::multi_shot_campaign`. When
the optional PyO3 extension is rebuilt,
`scpn_control_rs.PyMultiShotCampaignOrchestrator.run_table()` exposes the Rust
surface through table-shaped NumPy inputs, including optional per-shot
`pulsed_mpc_admission_digests`.

Use the dedicated [Multi-Shot Campaign](control/multi_shot_campaign.md) guide
for examples, claim boundaries, and benchmark commands.

Release admission is handled by
`validation.validate_multi_shot_campaign_evidence`. This standard-library
reader inspects persisted Python/PyO3 and Rust benchmark declarations. It
requires the two v1.1 schemas, recognised evidence classes, boolean production
flags, commands containing `bench_multi_shot_campaign`, consistent canonical
self-digests, declared positive passed/sample counts, and at least the requested
positive integer digest count in each Python, PyO3 and Rust result.
`pyo3_status` must be `"ok"`. Decoded empty objects fail; duplicate keys and
nonfinite floating JSON tokens fail even in extra metadata. The top-level
`scpn-control validate` command runs this gate by default before a release
evidence JSON can be admitted.

`validate_multi_shot_campaign_evidence(python_report, rust_report,
minimum_digest_count=2)` returns a frozen `MultiShotCampaignEvidenceAdmission`.
`as_dict()` includes schema
`scpn-control.multi-shot-campaign-evidence-admission.v1`, `status`, ordered
`errors`, four exact-report/declared-payload SHA-256 fields, `admitted_surfaces`,
`pyo3_status`, `production_claim_allowed` and the effective minimum count.
Only PASS lists all three surfaces. Successfully decoded objects retain their
exact byte hashes even on FAIL; read/decode/root-type failures yield `None`.
Non-string payload digests/PyO3 status become `None`. Invalid minimum arguments
produce FAIL and use fallback one. Returned mutable lists cannot change the result.

Python affinity must be a nonempty list; Rust affinity must be nonblank text.
Both contexts must record non-`None` start/end load fields. Affinity elements,
native Linux list/load syntax, latency statistics beyond positive sample counts,
count reconciliation and current-host qualification are unverified. The reader
does not reopen per-shot digests, run campaigns, authenticate producers or grant
hardware/controller/production timing admission. Self-digests cover all extra
fields but provide consistency only. Local regression cannot declare a true
production flag; production-class false is allowed. The returned flag preserves
either report's literal true even when other findings make overall status FAIL.
Consumers must check status and apply their independent admission boundaries.

The [persisted-report contract](control/multi_shot_campaign.md#persisted-benchmark-report-admission)
describes standalone JSON/text use, context serialisation and remaining nonchecks.

::: scpn_control.control.multi_shot_campaign.CampaignShotSample

::: scpn_control.control.multi_shot_campaign.CampaignShotPlan

::: scpn_control.control.multi_shot_campaign.CampaignCommandLog

::: scpn_control.control.multi_shot_campaign.CampaignShotResult

::: scpn_control.control.multi_shot_campaign.MultiShotCampaignOrchestrator

---

## Control — PREEMPT_RT Runtime Admission

`scpn_control.core.runtime_admission` emits a schema-versioned admission report
before native hardware-campaign execution. It binds the observed Linux kernel,
PREEMPT_RT evidence, current process affinity, requested execution cores,
scheduler policy, CPU governors, heartbeat configuration, and memory-lock limits
to the campaign summary. `run-hardware-campaign --runtime-admission-policy
require` fails closed before native execution if the host is not
production-admissible.

The optional PyO3 counterpart `scpn_control_rs.runtime_admission_snapshot()`
exposes native-side kernel/core parsing when the Rust extension is installed.
`validation.validate_runtime_admission_evidence` inspects persisted
runtime-admission benchmark declarations and their canonical payload
self-digest. It does not probe the current host or authenticate a producer;
resealed metadata remains a declaration, not independent realtime qualification.
The frozen `RuntimeAdmissionEvidenceAdmission` contains ordered `errors`, exact
decoded-report `report_sha256`, declared `payload_sha256`, class/production/probe
status, error-list count and positive samples. Non-string or non-boolean
declarations become `None` in their typed result fields; invalid strings remain
visible on FAIL. `as_dict()` returns a fresh JSON mapping and findings list.

Required schema is `scpn-control.runtime-admission-benchmark.v1`; the command
must contain `bench_runtime_admission.py`. A nonempty affinity list contains
nonnegative integer CPU IDs (booleans fail), host-load arrays contain exactly
three finite nonnegative numbers, and platform/Python/isolation strings are
nonempty. Samples are positive integers; six latency fields are finite and
nonnegative, min/median/p95/p99/max are monotonic and mean lies inside min/max.
No live CPU-ID validity, uniqueness, sample recomputation, latency limit,
timestamp freshness or scheduler/core qualification is supplied by this reader.

Both evidence classes require boolean production flags and string error/warning
lists. Failed local regression needs nonempty explanatory errors with a false
production flag; production declarations require a true flag, probe PASS and
empty errors. A probe PASS requires empty errors in either class. Reader PASS
can retain a failed local probe and grants no production timing claim.
Extra fields remain unvalidated but enter the canonical self-digest (sorted
compact JSON with `payload_sha256` blank). Duplicate keys and nonfinite floating
tokens at every depth fail before digest admission; empty objects fail required
fields while retaining their exact byte digest. Symlinks/caller-relative paths
are followed without containment or input-size/depth budgets.

```bash
python validation/validate_runtime_admission_evidence.py --report report.json --json-out
scpn-control validate --runtime-admission-report report.json --json-out
```

The standard-library script prints to stdout, writes no report and returns
0/1 for PASS/findings (argparse usage errors exit2). The registered root command
also evaluates its other enabled gates; scoping those out is a reader check,
not a complete release-evidence admission.

Use the dedicated [PREEMPT_RT Runtime Admission](control/runtime_admission.md)
guide for policy semantics and operator examples.

::: scpn_control.core.runtime_admission.RuntimeAdmissionRequest

::: scpn_control.core.runtime_admission.RuntimeAdmissionProbe

::: scpn_control.core.runtime_admission.collect_runtime_admission

::: scpn_control.core.runtime_admission.evaluate_runtime_admission

---

## Validation — Native Formal Certificate Report Admission

`validation.validate_native_formal_certificate_evidence.validate_native_formal_certificate_evidence`
reads one persisted native-formal benchmark object. Its public API accepts a
report path and `max_aot_p99_cycle_us=10.0`, a positive finite non-boolean numeric
bound. Invalid runtime arguments return structured FAIL before case-limit
evaluation. Required AOT summaries declare positive runs, equal admitted
certificate counts, equal positive generated/submitted/checked totals, zero
drops/failures, one expected certificate schema/id, one lowercase SHA-256 digest
and positive finite p99 at or below the bound. Non-AOT objects establish no AOT
admission. Other latency fields and per-tick proof or timing data are unchecked.

Required context declares recognised class/boolean production metadata, command
arguments identifying the producer, nonempty nonnegative CPU lists, nonblank
host/claim strings and a nonempty version object. Production also declares
explicit isolation, boolean heavy-job status and literal `workspace_dirty=false`;
true heavy-job status remains allowed. The reader does not inspect the actual
host, certificate bytes, SMT proof or producer authenticity. Context strings and
counts remain declarations.

`NativeFormalCertificateEvidenceResult.as_dict()` returns schema
`scpn-control.native-formal-certificate-evidence.v1`, `status`, `admitted_cases`,
`certificate_assumption_sha256`, `benchmark_evidence_class`,
`production_claim_allowed`, `errors` and `report_sha256`, with fresh mutable
case/error lists. Case-level admitted labels can remain on global FAIL; a single
observed valid digest can come from a rejected case. Declared boolean production
metadata is preserved on FAIL. Consumers must check overall status. Exact report
bytes are decoded and hashed once; supported read/decode/non-object failures
have no digest. Duplicate/nonfinite floating JSON tokens are refused at all
depths. Extra metadata is otherwise ignored; paths follow cwd/symlinks without
containment or size/depth budgets.

The standard-library standalone script emits JSON and exits zero on reader PASS,
one on refusal and two on argument errors. Root `validate` and tracker53 call the
same reader. See the [benchmark admission contract](benchmarks.md) for the
across-run p99 definition and claim boundaries.

## Validation — Tracker 53 Aggregate Metadata

`validation.validate_tracker53_evidence.validate_tracker53_evidence` selects six
defining module surfaces from integer issue-53 registry entries and invokes the
actual runtime/native-formal readers plus a Z3 PASS-status reader. Read-only
defaults are historical repository declarations. Optional `output_json` writes
a sorted UTF-8 manifest to an explicit path, including refused declarations;
supported output errors become FAIL findings.

`Tracker53EvidenceResult` contains status, tracker issue, ordered entries/errors,
assigned evidence classes, requested boolean production mode and an aggregate
production flag. Four classes are fixed and unqualified, so the current input
contract always keeps that flag false and refuses requested production mode.
Runtime/formal production-labelled child classes require passing real readers;
formal also requires declared Z3 PASS. Registry evidence paths and Z3 proof
internals are not reopened or authenticated. The Z3 status reader accepts paths
outside the checkout and hashes the exact decoded bytes. Non-PASS status,
ambiguous/nonfinite JSON or lower-reader failures refuse the aggregate.

`build_tracker53_manifest(result)` adds schema
`scpn-control.tracker53-evidence-gate.v1` and a SHA-256 consistency digest over
sorted compact JSON with only `manifest_sha256` removed. Nested entries and the
evidence-class mapping retain result aliases; caller mutation can invalidate
the digest. Invalid non-boolean production API arguments return FAIL and
normalise to false. No source authentication, current-host qualification,
certified controller or facility action authority is established. The
[validation guide](validation.md) documents CLI/output behaviour and nonchecks.

## Validation benchmark regression gates

`validation/validate_benchmark_regression_gates.py` admits persisted benchmark
evidence before the preflight gate accepts a regression baseline. The gate does
not run benchmarks and does not create timing evidence. It validates
`validation/reports/benchmark_regression_gates.json` against the referenced
latency reports, SHA-256 digests, metric paths, bounded thresholds, sample
counts, hardware-context metadata, and explicit non-HIL claim boundaries.

The gate is intended to catch stale, tampered, overclaimed, or regressed local
benchmark evidence before release preflight continues. Hardware-in-the-loop,
target-device, cloud-GPU, or plant real-time claims remain blocked until those
specific benchmark artefacts are generated and admitted separately.

The public function `validate_benchmark_regression_gates(manifest_path)` returns
an immutable `BenchmarkRegressionGateResult`. `as_dict()` produces the same
versioned carrier used by the CLI: status, ordered admitted ids, authored errors
and the raw manifest SHA-256. A failure clears all admitted ids. Unreadable
manifest bytes produce an empty digest; malformed readable bytes retain their
digest. Each file's hash and decoded fields use one captured byte buffer, while
separate reads have no transaction or coherent snapshot guarantee. Repeated
report references are inspected per entry.

```bash
python validation/validate_benchmark_regression_gates.py
python validation/validate_benchmark_regression_gates.py ./copied-gates.json
```

The optional path resolves from cwd; its default belongs to the script
repository. Referenced reports resolve from the repository root for an in-repo
manifest, otherwise from the resolved manifest's parent. Report URIs require
literal relative slash components: empty, dot and parent components, URLs,
absolute paths, backslashes and symlink escapes are refused. JSON must be an
unambiguous finite UTF-8 object; duplicate keys, `NaN`/`Infinity`, floating
overflow and decoding failures refuse admission even in extra metadata.

Metrics are finite numbers excluding booleans, with `us`/`ms`/`s` labels and no
unit conversion or dimensional inference. Equality uses relative tolerance
`1e-12` and absolute tolerance `1e-9`; observed values must be at most a positive
maximum threshold. Positive integer sample counters must equal the referenced
value and reach the inclusive minimum. Claim qualification uses a literal,
case-sensitive substring. Hardware context requires nonblank machine and
platform-or-kernel strings; `unknown` is still metadata, not RT qualification.
Declared optional self-digests must be hexadecimal strings even when their field
is present as `null`; their canonical hash excludes only their own field and
uses sorted compact ASCII JSON. Other domain-specific digests remain the
responsibility of their validators.

`generated_utc` is checked for nonblank text, not format, age or freshness.
Positive metric signs, authenticated provenance, host isolation and actual
sample arrays are not established by this metadata gate. It reads without
writing reports, changing argv or executing kernels. CLI status is 0 for pass,
1 for structured inspection/refusal failures and 2 for argument errors;
expected file/decoder failures print authored JSON errors without a traceback.

---

## Core — Native C++ Solver Bridge

### Stability and ownership boundary

The names exported by `scpn_control.__all__` are the admitted stable Python
package surface. Symbols shown in module-level reference pages remain directly
usable and documented, but visibility there alone is not a compatibility
promise. Protocol methods, generated helpers, CLI callbacks, shell entrypoints,
and GitHub workflow inputs are classified under their native owners rather than
being counted as Python library APIs.

The cross-language registry declares a static inventory of top-level non-private
Python classes/functions and bounded C, Lean, Rust and TypeScript declaration
patterns. The [declaration checker](#api-declaration-contract) detects changes
in selected names and their lexical classification. Its hashes do not establish
review, runtime export resolution, signatures or implementation compatibility. Python contracts use NumPy
docstrings and strict typing; Rust uses strict rustdoc and typed `Result`
values; TypeScript uses strict TypeDoc; operator scripts/workflows use their
native lint and policy gates. Parameters, returns/yields, errors, mutation and
state, units and shapes, safety boundaries, compatibility, evidence links, and
examples are required where they apply—empty headings are not accepted as
documentation.

`scpn_control.core.hpc_bridge` exposes the optional native Grad-Shafranov
solver bridge. Native compilation is disabled unless
`SCPN_ALLOW_NATIVE_BUILD=1` is set. When enabled, the bridge admits only the
package-local `solver.cpp` whose SHA-256 digest matches
`solver_manifest.json`, resolves the compiler to an absolute regular
executable, strips dynamic-loader injection variables from the build
environment, rejects symlinked solver inputs and output targets, compiles to a
temporary package-local file, and publishes the shared library atomically after
the compiler produced a regular file.

The source, normative `solver.h` ABI contract, and checksum manifest are shipped
as `scpn_control.core` package data. The bridge prefers the typed version 1 ABI
and retains the five historical symbols as a compatibility fallback. In the
versioned API, call validity and scientific convergence are separate values;
invalid dimensions, non-finite data, size mismatch, allocation failure, and
internal failure cannot be confused with non-convergence. The historical
Python argument name `j_phi` is retained for compatibility, but this kernel
accepts a pre-scaled elliptic right-hand side—not raw toroidal current density.

`HPCBridge.solve_into` and `solve_until_converged_into` require a writable,
C-contiguous `float64` output with the initialised grid shape. Read-only output
raises `ValueError` before a native sweep, preserving both the buffer and solver
state. A read-only source is accepted; source and output must not overlap.
Unavailable libraries or states retain their documented `None` result.
`tests/test_hpc_native_buffers.py` exercises these public calls against a
temporary build of the checksum-bound packaged solver, including a manufactured
elliptic solution. This native test profile requires a working C++ compiler;
missing compilation capability is a failed precondition, not native evidence.

The complete generated [C ABI and Lean proof reference](./_generated/native_api_reference.md)
records layout, shape, units, ownership, lifetime, mutation, convergence,
errors, thread safety, determinism, compatibility, and the exact scope of the
checked finite-state theorems. Compiled `bin/libscpn_solver.so` or
`bin/scpn_solver.dll` outputs are operator-local build products and are not
committed.

`tools.generate_native_api_reference.render()` reads the default UTF-8 header
and Lean corpus and returns deterministic Markdown without writing. Keyword
`header_path` and `lean_path` select local alternatives. The lexical extractor
requires ABI version 1 exactly once, five versioned and five legacy C functions,
nonempty Doxygen comments immediately before each versioned function, and nine
nonempty adjacent Lean declaration comments. An unrelated closed type or
namespace comment is excluded. Unsupported ABI, malformed C function syntax,
missing comments and unexpected counts raise `ValueError`; source IO and UTF-8
decoding errors propagate. It does not parse arbitrary C/Lean syntax, nested
Lean comments, preprocess headers, compile libraries or check proofs. Selected
source paths do not establish provenance or authentication.

The CLI `python tools/generate_native_api_reference.py --check` compares the
tracked reference with the default corpus. Optional `--header`, `--lean` and
`--output` select local files. Without `--check`, it creates the output parent
and replaces a non-alias output; direct paths, symlinks and existing hard links
to either selected input are refused before writing. Check mode only reads and
leaves stale or missing outputs untouched. Success returns 0; stale output and
supported IO/decode/contract refusals return 1 with authored diagnostics. Parser
help and usage retain exits 0 and 2. Reading multiple files and writing the
reference do not provide an atomic source snapshot or transactional publication.
Generation and comparison success cover this lexical corpus contract; compiler
ABI qualification and Lean proof checking remain separate owning gates.

External runtime solver libraries remain blocked unless
`SCPN_ALLOW_EXTERNAL_SOLVER_LIB=1` is set for a vetted absolute path. The
default path searches package-local solver locations only.

---

## Benchmark Producer Registry Audit

`tools.check_benchmark_producers.audit_registry(registry_path, repository_root)`
reads a TOML registry and an existing directory and returns ordered lexical
inventory/custody findings. Defaults select the canonical repository and its
`benchmarks/producer_registry.toml`. Both paths are caller-relative; root
resolution refuses missing paths or regular files. Categories are
`recorded_guard`, `append_stream`, `temporary_scratch`,
`stdout_or_build_product` and `custody_infrastructure`. Schema, category arrays,
duplicate ownership and classified/discovered path differences are checked.
Only discovered registered sources are decoded; unexpected paths remain
inventory findings. IO, UTF-8, TOML and Python syntax errors propagate.

Discovery includes benchmark Python files, benchmark-named scripts/tools,
validation benchmark files and three explicitly named validation owners, plus
selected Rust benches/examples and the transport binary. The auditor excludes
itself. Files are discovered independently of Git tracking; normal file-symlink
semantics apply. Python guard checks find an AST call with the literal guard
name/attribute, including unreachable calls; they do not resolve imports,
aliases, arguments or live behaviour. Rust guards and append/scratch/custody
classes use literal source markers. README and sorted public docs Markdown are
scanned for direct executable-looking commands naming guarded producers, except
changelog and internal paths. Findings retain document/line/path order. This
does not parse shell or Markdown, run producers, establish immutable output,
measure performance or admit scientific evidence. Multiple reads are not a
coherent concurrent snapshot.

`python tools/check_benchmark_producers.py --repo path/to/repository` selects
that repository's default registry; optional `--registry path/to/registry.toml`
selects a caller-relative TOML file. Success prints the discovered classification
count and returns 0. Findings and supported inspection refusals return 1 with
authored stderr; parser help/usage exits are 0/2. The CLI writes no files and runs
no benchmarks. A stdout-only producer's classification does not authenticate or
qualify its measurements; durable benchmark custody remains the recorded
runner's responsibility.

## Neural Equilibrium Reference Declarations

`validation.validate_neural_equilibrium_reference.validate_neural_equilibrium_reference(artifact_root, require_reference_artifacts=False)`
reads persisted JSON declarations. A directory selects immediate sorted JSON
files; a regular file selects itself regardless of suffix. Missing/nonfile roots
select none: optional mode returns a passing diagnostic report with zero
declarations, required mode fails. Per-file IO, UTF-8, duplicate-key/JSON and
supported value errors become ordered findings. A JSON file is captured once;
`artifact_file_sha256` hashes those exact bytes, including CRLF spelling.
Observations across multiple files are sequential and not an atomic snapshot.

The v1 declaration requires nonblank source/model/version/weight/dataset/URI/
digest/execution strings, hexadecimal SHA-256 values, an exact target schema,
two positive nonboolean integer grid counts, a positive reference-equilibrium
count and four exact unit labels. Allowed source labels are `real_pefit` and
`documented_public_reference`. P-EFIT binary paths use the shared lexical
absolute-path policy; public references require a nonblank URL or DOI string.
Neither source is executed, fetched or authenticated. Artifact URI checks use
host Path semantics: relative paths and admitted literal external prefixes are
accepted, POSIX absolute/parent/NUL paths refused; Windows drive spellings on
POSIX are not structurally validated. Unknown fields are retained in payload
hashing but do not acquire additional schema checks.

Declared `psi_rmse_Wb`, `pressure_rmse_Pa`, `q_profile_rmse`, `boundary_rmse_m`
and `axis_position_error_m` must be finite, nonnegative representable numbers
within positive finite declared tolerances. Units are Wb/rad despite the legacy
Wb field spelling, Pa, dimensionless q and metres. The validator does not read
arrays, recompute metrics, determine a boundary algorithm, verify grid shape
against data or match the declared model/version/weight values to a live model.
Unrepresentable integer-to-float conversions and malformed source/digest values
produce findings instead of escaping as overflow/type exceptions. Duplicate
model/weight/dataset triples are refused within one selected scan.

`canonical_artifact_sha256(payload)` returns an unkeyed consistency checksum of
sorted compact ASCII JSON excluding `payload_sha256`, without mutating the
dictionary. Nonfinite/nonserializable values raise `ValueError`/`TypeError`.
The mutable v2 report hashes its own canonical contents with the digest field
nulled. `reference_artifacts_admitted` denotes passing declaration metadata;
`predictive_equilibrium_claim_admitted` remains false. Referenced array/weight
bytes, provenance, timestamps and scientific qualification remain consumer and
reviewer responsibilities. The core fine-tuning caller uses report status before
reading supplied GEQDSK files; the claim-evidence caller additionally matches the
supplied weight digest. This validator alone does not authenticate those inputs
or establish facility validity. Synthetic declaration test fixtures demonstrate
schema behaviour only and do not establish executed P-EFIT evidence.

`write_neural_equilibrium_reference_report(report, output_path, artifact_root=...)`
writes sorted indented finite JSON, creates parents and replaces a non-alias
output. Resolved paths, symlinks and existing hard links to the selected root or
its current JSON inputs raise `ValueError` before writing. IO/encoding/JSON errors
propagate. Persistence and its fresh input-alias observation are not transactional
with the earlier validation. Outputs inside a corpus may be selected by later
scans; choose a separate report location.

Both `scpn-control validate-neural-equilibrium-reference` and
`python validation/validate_neural_equilibrium_reference.py` use this reader and
writer. `--artifact-root`/`--output-json` are caller-relative; absent root selects
the canonical reference directory. `--require-reference-artifacts` requires a
nonzero accepted declaration count. `--json-out` prints JSON, otherwise a summary
and findings are emitted. Passing declaration reports return 0; findings and
authored inspection/persistence failures return 1. Help/usage exits are 0/2.
No model fitting, P-EFIT execution, download or predictive admission is performed.

## Physics Traceability Registry and Reports

`validation.validate_physics_traceability.validate_physics_traceability(registry_path)`
inspects one UTF-8 JSON registry with schema `1.1`. A resolved file directly
under `validation/` uses its parent directory as repository root; every other
registry location uses cwd. Module, evidence and covered-source paths must
exist within that inferred root after resolving absolute paths, traversal and
symlinks. Files and directories can establish path existence; no evidence
content, digest, freshness or physical provenance is verified.

Required header, string/list, fidelity status and literal boolean fields are
checked. Open/bounded statuses prohibit true full-fidelity flags and require
known external tracker issues. Tracker identifiers are unique positive integers
excluding booleans, with nonblank title/scope and matching repository issue
URLs. Invalid trackers are excluded from the returned tracker list. JSON keys
must be unique and float tokens finite at every depth. Malformed inputs and
supported read/decode/path failures return FAIL with findings.

The mutable result contains status, registry spelling, raw entry total,
open-gap and literal-false-claim counts, resolved module/evidence counts,
valid tracker declarations/count, dictionary-entry summaries, findings and
source-marker coverage (`total`, `covered`, `missing`). Invalid dictionary
entries and their raw claim flags remain visible; counts are observations,
rather than admitted scientific evidence. Nonobject entries count in total but
have no summary. Optional source-marker enforcement must be boolean. Scanning
sorted Python files under `src/scpn_control` reads literal approximation-marker
words; invalid UTF-8, unreadable files or escaping symlinks cause findings.
Covered paths must belong to the resolved file/directory module scope.

`validation.generate_physics_traceability_report.generate_physics_traceability_markdown`
accepts `registry_path` and keyword `require_valid_registry=False`. Default
mode renders diagnostic FAIL reports with every displayed full-fidelity claim
blocked. Strict mode requires a passing registry and raises `ValueError` on
refusal; ill-typed policy values also raise. The output is deterministic,
newline-terminated Markdown with tracker/count metadata, a table, components,
actions and validation findings. The standalone generation CLI uses strict
mode and returns 1 before writing for invalid source; supported write/path
failures also return 1. Canonical valid report bytes remain unchanged.

`tools.check_generated_traceability.expected_traceability_markdown(registry)`
uses strict generation. `generated_traceability_is_current(registry,
report_path)` returns true only for valid source and exact readable UTF-8
report bytes; invalid source, read/decode failure, missing or stale output
returns false. Its CLI performs no generation/write. Equality of a diagnostic
FAIL report cannot clear freshness. This is a deterministic local consistency
check, with no timestamp, source-hash, remote issue or scientific validation.

All three owners have executable native examples and standard-library
standalone CLIs. They do not execute numerical models or authenticate facility,
reference or control authority. Filesystem inspection is not a coherent
concurrent snapshot. Standalone validator JSON output may contain refusals;
its output IO/path failures append findings. The Click callback retains its
separate output-path contract.

## Physics Debug Assistance

`scpn_control.physics_debug` provides a local-first advisory assistant for
physics validation gaps. The default provider policy admits loopback endpoints
only; facility or external gateways must be explicitly allowlisted.
`build_local_provider()` supplies loopback profiles for common onsite gateway
protocols: chat-completions-compatible, Ollama-style chat, direct JSON, and
text-generation endpoints. Reports are schema-versioned advisory evidence with
secret redaction, falsifiable hypothesis checks, campaign risk controls, and
risk-bound prompt-injection neutralization for untrusted evidence text before
provider prompting. Prompt-guard findings are recorded in the tamper-evident
payload digest. `build_guardrail_provider()` adds an optional hallucination
guardrail gateway with a `director-ai` default profile and explicit alternate
profiles for lab-owned guardrail solutions. Guardrail block decisions fail
closed before report persistence; allow findings are bound into the report
digest together with the SHA-256 digest of the reviewed provider draft.
The guardrail request also binds the provider metadata, safety policy, and
guardrail policy digests so reviews cannot be replayed across another provider
or relaxed policy. High-severity guardrail findings must use block actions, and
risk controls must meet the configured guardrail policy before persistence.
They are not validated physics truth, controller-parameter promotion, or
facility safety approval.
`run_provider_quorum()` runs multiple providers in local-first order and admits
only hypotheses corroborated by the required provider count over the same gap
and evidence set.
`PhysicsDebugSafetyPolicy` binds the human-review requirement, caps advisory
confidence, and rejects provider text that attempts controller promotion,
actuation, review bypass, or approval claims.

::: scpn_control.physics_debug.ProviderPolicy

::: scpn_control.physics_debug.PhysicsDebugGuardrailPolicy

::: scpn_control.physics_debug.PhysicsDebugEvidence

::: scpn_control.physics_debug.PhysicsDebugGap

::: scpn_control.physics_debug.PhysicsDebugSafetyPolicy

::: scpn_control.physics_debug.HTTPChatProvider

::: scpn_control.physics_debug.PhysicsDebugGuardrailProvider

::: scpn_control.physics_debug.PhysicsDebugAssistant

::: scpn_control.physics_debug.build_local_provider

::: scpn_control.physics_debug.build_guardrail_provider

::: scpn_control.physics_debug.build_physics_debug_report

::: scpn_control.physics_debug.run_provider_quorum

::: scpn_control.physics_debug.validate_physics_debug_report

::: scpn_control.physics_debug.validate_physics_debug_quorum_report

::: scpn_control.physics_debug.write_physics_debug_report

::: scpn_control.physics_debug.write_physics_debug_quorum_report

---

## Control — Anti-Windup PID Controller

`scpn_control.control.pid_controller` provides a single-axis position PID with an
optional saturation/slew envelope and conditional-integration anti-windup. With no
envelope configured it reproduces the ideal `kp*e + ki*Σe + kd*Δe` law exactly; an
envelope freezes the integrator for any step clamped in the error's own direction,
closing the integral-windup hazard on a saturating actuator.

The Python and native Rust PID entry points reject nonfinite gains, errors and
derived arithmetic before changing controller state. Finite inputs that would
overflow the derivative, integral or output are refused. This is a software
numerical contract; it does not certify a physical actuator or timing bound.

::: scpn_control.control.pid_controller.PIDController

---

## Control — Federated Disruption Prediction

`scpn_control.control.federated_disruption` supports FedAvg and FedProx
training across named tokamak clients. The server aggregates model updates;
the included in-process array factory does not enforce remote data isolation.
`create_facility_clients_from_arrays()` accepts per-facility `X_train`,
`y_train`, `X_test`, and `y_test` arrays already held by its caller. It
enforces the shared 8-feature disruption contract and binary labels before a
client joins the federation.

`DifferentialPrivacyConfig` enables facility-update clipping, Gaussian noise,
and a serialisable ledger of nominal Gaussian-mechanism epsilon values. The
returned client metrics expose facility test-set information, and neither the
ledger nor this API certifies end-to-end differential privacy. The shipped benchmark
`validation/benchmark_federated_disruption.py` publishes deterministic
synthetic multi-facility evidence in
`validation/reports/federated_disruption_benchmark.json` and
`validation/reports/federated_disruption_benchmark.md`. Those artefacts test
federation, heterogeneity, and update-noise mechanics; they do not
claim measured cross-facility validation without external shot databases.
The separate `differential_privacy_clip()` helper clips an aggregate gradient
dictionary; it does not perform per-example DP-SGD.

`FederatedConfig.learning_rate` controls each client's local step during a
server round. `FederatedServer.aggregate()` requires positive integer sample
counts and finite weights with the declared MLP shapes; rounds require
distinct, configured facilities. `get_state()` emits schema version 2 with
both NumPy random streams. `from_state()` checks model shape, finite values,
ledger arithmetic, and replay fields; older snapshots without the random
streams are rejected. This consistency check does not authenticate a snapshot
or prove that its training data came from an independent facility.

::: scpn_control.control.federated_disruption.DifferentialPrivacyConfig

::: scpn_control.control.federated_disruption.PrivacyLedgerEntry

::: scpn_control.control.federated_disruption.FacilityBenchmarkSummary

::: scpn_control.control.federated_disruption.FederatedConfig

::: scpn_control.control.federated_disruption.MachineClient

::: scpn_control.control.federated_disruption.FederatedServer

::: scpn_control.control.federated_disruption.create_facility_clients_from_arrays

::: scpn_control.control.federated_disruption.run_synthetic_multifacility_benchmark

---

## Control — Quantum Disruption Bridge

`scpn_control.control.quantum_disruption_bridge` is a fail-closed facade for
optional quantum-enhanced disruption prediction. Quantum circuit and provider
ownership stays in `scpn-quantum-control`; SCPN-CONTROL owns the control
feature contract, lazy optional import boundary, bounded claim metadata, and
checksum-bound advisory reports. The bridge maps the CONTROL 8-feature
disruption vector to the ITER 11-feature contract only when missing ITER fields
are either supplied explicitly or declared as bounded centre defaults. Reports
are not facility validation, controller promotion, or publication-safe evidence
without external disruption databases and benchmark artefacts.

`quantum_disruption_kernel_matrix()` emits a bounded amplitude-encoding kernel
report with symmetry, diagonal, and `[0, 1]` admission checks. The callable
quantum owner path uses `scpn_quantum_control.control.q_disruption_iter`
lazily; when that optional dependency is unavailable the report fails closed
with `status="quantum-unavailable"` and no quantum score. Every bridge report
also records advisory admission evidence: CONTROL feature digests, ITER mapping
digests, default-use reasons, and the external evidence still required before
facility or publication claims are admissible. Bridge and kernel reports carry
schema-versioned advisory certificates that bind report kind, repository
ownership, claim boundary, downstream non-admission policy, and the content
digest before the outer payload digest is accepted. The facade also publishes a
machine-readable dependency contract that names the `scpn-quantum-control`
backend module, required classifier surface, Qiskit core dependencies, optional
provider dependency families, report schemas, feature contract, and
non-admission policy for future backend hardening. Generated bridge and kernel
reports embed that dependency contract and bind its digest into the advisory
certificate so archived reports cannot be replayed against a different quantum
backend contract. When the optional backend exposes its own
`scpn_control_bridge_dependency_contract()` callable, CONTROL compares it
against the expected contract, records backend-contract attestation evidence,
and fails closed before report creation if the backend advertises a conflicting
contract. Bridge reports also include schema-versioned advisory decision
evidence that records whether the score came from the quantum backend or the
classical fallback, applies deterministic low/elevated/high risk-band
thresholds, records backend-contract validation state, fixes the control action
to `blocked`, and binds the decision digest into the advisory certificate.
The public validator also recomputes the classical score from the mapped ITER
features and requires the reported risk score to equal the selected quantum
or classical source score. It requires backend availability, quantum-score
presence, and backend attestation status to agree, even if a caller has
recomputed every digest. These
unkeyed digests provide internal consistency; they do not authenticate a
backend prediction or establish facility provenance. The facade delegates to
cohesive feature, contract, backend, report, runtime, and digest modules.

::: scpn_control.control.quantum_disruption_bridge.QuantumDisruptionBridgeConfig

::: scpn_control.control.quantum_disruption_bridge.QuantumFeatureMapping

::: scpn_control.control.quantum_disruption_bridge.map_control_features_to_iter

::: scpn_control.control.quantum_disruption_bridge.quantum_disruption_kernel_matrix

::: scpn_control.control.quantum_disruption_bridge.quantum_disruption_dependency_contract

::: scpn_control.control.quantum_disruption_bridge.run_quantum_disruption_bridge

::: scpn_control.control.quantum_disruption_bridge.validate_quantum_disruption_dependency_contract

::: scpn_control.control.quantum_disruption_bridge.validate_quantum_disruption_bridge_report

::: scpn_control.control.quantum_disruption_bridge.validate_quantum_disruption_kernel_report

The implementation references separate feature mapping, backend declarations,
contract validation, report validation and execution. Advisory reports retain
the admission boundary described above; digest checks do not authenticate an
external backend or facility witness.

::: scpn_control.control._quantum_disruption_constants

::: scpn_control.control._quantum_disruption_features

::: scpn_control.control._quantum_disruption_backend

::: scpn_control.control._quantum_disruption_contract

::: scpn_control.control._quantum_disruption_reports

::: scpn_control.control._quantum_disruption_runtime

::: scpn_control.control._quantum_disruption_utils

---

## Core — Physics Solvers

### FusionKernel

`FusionKernel` validates its JSON configuration before grid construction:
the root must be an object, duplicate JSON keys are rejected, dimensions and
grid resolution must be physical, `physics.plasma_current_target` must be
positive finite, and `physics.vacuum_permeability` must be positive finite
when supplied. Configuration models and parse/dump helpers live in a dedicated
leaf; the CONTROL `FusionKernel` product surface remains first-class and
re-exports those contracts (dual-home C).

::: scpn_control.core.fusion_kernel.FusionKernel

### Fusion Kernel Configuration

::: scpn_control.core.fusion_kernel_config.DimensionsConfig

::: scpn_control.core.fusion_kernel_config.PhysicsConfig

::: scpn_control.core.fusion_kernel_config.SolverConfig

::: scpn_control.core.fusion_kernel_config.CoilConfig

::: scpn_control.core.fusion_kernel_config.FusionKernelConfig

::: scpn_control.core.fusion_kernel_config.parse_fusion_kernel_config

::: scpn_control.core.fusion_kernel_config.fusion_kernel_config_dump

### Grad-Shafranov Green / Vacuum Helpers

Toroidal Green's function, vacuum poloidal flux from coil sets, and mutual
inductance construction live in a dedicated leaf. `FusionKernel` keeps thin
wrappers so existing free-boundary and EFIT import paths remain stable.

::: scpn_control.core.gs_green_vacuum.green_function

::: scpn_control.core.gs_green_vacuum.green_function_array

::: scpn_control.core.gs_green_vacuum.vacuum_poloidal_flux

::: scpn_control.core.gs_green_vacuum.build_mutual_inductance_matrix

### Grad-Shafranov Multigrid Primitives

Full-weighting restriction, bilinear prolongation, Red-Black SOR smoothing,
residual evaluation, and V-cycle recursion live in a dedicated leaf.
`FusionKernel` keeps thin wrappers so existing solver and Rust-parity paths
remain stable under dual-home C.

::: scpn_control.core.gs_multigrid.restrict_full_weight

::: scpn_control.core.gs_multigrid.prolongate_bilinear

::: scpn_control.core.gs_multigrid.mg_smooth

::: scpn_control.core.gs_multigrid.mg_residual

::: scpn_control.core.gs_multigrid.multigrid_vcycle

### Grad-Shafranov Elliptic Iterators

Jacobi and Red-Black SOR steps, Anderson mixing, boundary enforcement, and the
pure-Python elliptic solve path live in a dedicated leaf. `FusionKernel` keeps
thin wrappers so HPC offload and existing solver/parity paths remain stable
under dual-home C.

::: scpn_control.core.gs_elliptic_iterators.jacobi_step

::: scpn_control.core.gs_elliptic_iterators.sor_step

::: scpn_control.core.gs_elliptic_iterators.anderson_step

::: scpn_control.core.gs_elliptic_iterators.elliptic_solve_python

### Grad-Shafranov Profile and Plasma Source

Modified-tanh pedestal profiles, the normalised-flux denominator guard,
nonlinear plasma source construction, and profile Jacobians live in a dedicated
leaf. `FusionKernel` keeps thin wrappers that supply mesh, config, and external
profile tables so public solver paths remain stable under dual-home C.

::: scpn_control.core.gs_profile_source.mtanh_profile

::: scpn_control.core.gs_profile_source.mtanh_profile_derivative

::: scpn_control.core.gs_profile_source.normalised_flux_denominator

::: scpn_control.core.gs_profile_source.update_plasma_source_nonlinear

::: scpn_control.core.gs_profile_source.compute_profile_jacobian

### Free-Boundary Coil Control

Coil-current least-squares optimisation, free-boundary objective tolerances and
status evaluation, divertor configuration labels, and the public `CoilSet`
dataclass live in a dedicated leaf. `FusionKernel` re-exports `CoilSet` and
keeps thin wrappers that supply Green/mutual response operators so free-boundary
solver paths remain stable under dual-home C. Full free-boundary solve
orchestration remains on the owner class.

::: scpn_control.core.gs_free_boundary_control.CoilSet

::: scpn_control.core.gs_free_boundary_control.optimize_coil_currents

::: scpn_control.core.gs_free_boundary_control.resolve_free_boundary_objective_tolerances

::: scpn_control.core.gs_free_boundary_control.evaluate_free_boundary_objective_status

::: scpn_control.core.gs_free_boundary_control.divertor_configuration_label

### Free-Boundary Solve Orchestration

The experimental external-coil outer loop around the fixed-boundary
Grad-Shafranov solve lives in a dedicated leaf. `FusionKernel.solve_free_boundary`
is a thin wrapper that supplies mesh, equilibrium, vacuum/Green responses, and
already-extracted free-boundary control helpers under dual-home C. Phase-sync
steps and the Rust multigrid bridge remain on the owner class.

::: scpn_control.core.gs_free_boundary_solve.solve_free_boundary

### Phase-Sync Steps

Reduced-order phase-sync (`ζ sin(Ψ−θ)`) and multi-step Lyapunov tracking helpers
used by `FusionKernel` live in a dedicated leaf. Thin owner wrappers supply
`phase_sync` config defaults; Kuramoto numerics remain in
`scpn_control.phase.kuramoto` under dual-home C.

::: scpn_control.core.gs_phase_sync.phase_sync_step

::: scpn_control.core.gs_phase_sync.phase_sync_step_lyapunov

### Rust Multigrid Bridge

Python orchestration for ``solver_method=rust_multigrid`` lives in a dedicated
leaf: availability probing, boundary-constrained and missing-Rust fallbacks to
Python SOR, and state sync from the Rust accelerated kernel. `FusionKernel`
keeps a thin wrapper under dual-home C; Rust multigrid algorithm semantics are
unchanged.

::: scpn_control.core.gs_rust_multigrid_bridge.solve_via_rust_multigrid

### Global Design Scanner

`GlobalDesignExplorer` provides the bounded scalar design metrics consumed by
the disruption-contract episode path. It estimates `Q`, fusion power, neutron
wall load, and a relative cost proxy from validated tokamak design inputs.

::: scpn_control.core.global_design_scanner.GlobalDesignExplorer

::: scpn_control.core.global_design_scanner.DesignScannerConfig

### TokamakConfig

::: scpn_control.core.tokamak_config.TokamakConfig

### TransportSolver

::: scpn_control.core.integrated_transport_solver.TransportSolver

### AdaptiveTimeController

`AdaptiveTimeController` selects the next bounded time step after comparing a
full transport step with two half steps. The integrated solver keeps the
accepted two-half-step state and exposes its step and error histories.

::: scpn_control.core.adaptive_time_controller.AdaptiveTimeController

The integrated solver's public steady-state and equilibrium methods retain
their existing signatures. Their control-facing iteration is implemented in
`transport_orchestration`, separate from coefficient and diffusion formulas.

::: scpn_control.core.transport_orchestration

The state module defines replay snapshots and the thermal balance record.
The historical `integrated_transport_solver` imports for `PhysicsError` and
`ThermalEnergyBalance` remain available.

::: scpn_control.core.transport_state.PhysicsError

::: scpn_control.core.transport_state.ThermalEnergyBalance

The species runtime handles ion capacity, auxiliary heating and species
balance. Face-flux transport and Crank–Nicolson thermal evolution have their
own runtime modules; `TransportSolver` keeps the public methods.

::: scpn_control.core.transport_species_runtime

::: scpn_control.core.transport_face_runtime

::: scpn_control.core.transport_thermal_runtime

The legacy neoclassical and bootstrap formulas and coefficient selection live
behind the same `TransportSolver` facade. FUSION remains the formulation owner
for new broad solver development.

::: scpn_control.core.transport_neoclassical

::: scpn_control.core.transport_model_selection

### Plasma Power Terms

::: scpn_control.core.plasma_power_terms.bosch_hale_dt_reactivity

::: scpn_control.core.plasma_power_terms.tungsten_radiation_rate

::: scpn_control.core.plasma_power_terms.bremsstrahlung_power_density

### Radial Diffusion Numerics

The compact tridiagonal API accepts finite real float64 vectors with lengths
`n-1, n, n-1, n` for `n >= 1`. It uses the pivoted LAPACK expert driver
`dgtsvx`, whose iterative refinement lets rows moved by a pivot meet the rowwise
backward-error check, and raises distinct shape, nonfinite-input,
singular-factorisation, and numerical-failure errors.
Inputs are never mutated. The historical `thomas_solve` name is a compatibility
entry point; the Rust/PyO3 adapter accepts `n`-padded off-diagonals with zero
unused sentinels. JAX's traced no-pivot path requires strict row diagonal
dominance.

::: scpn_control.core.tridiagonal.solve_tridiagonal

::: scpn_control.core.radial_diffusion.thomas_solve

::: scpn_control.core.radial_diffusion.explicit_diffusion_rhs

::: scpn_control.core.radial_diffusion.build_cn_tridiag

### Anomalous Transport Coefficient Models

::: scpn_control.core.anomalous_transport.gyro_bohm_chi_profile

::: scpn_control.core.anomalous_transport.gk_flux_surface_transport

### Auxiliary Heating Source Deposition

::: scpn_control.core.aux_heating.aux_heating_source_profiles

### Multi-Ion Species Evolution

::: scpn_control.core.species_evolution.evolve_multi_ion_species

::: scpn_control.core.species_evolution.SpeciesEvolutionResult

### Runtime State Sanitization

::: scpn_control.core.runtime_sanitization.sanitize_with_fallback

### Transport Radial-Grid Geometry

::: scpn_control.core.transport_geometry.rho_volume_element

::: scpn_control.core.transport_geometry.estimate_plasma_surface_area_m2

::: scpn_control.core.transport_geometry.is_canonical_radial_grid

::: scpn_control.core.transport_geometry.canonical_radial_grid

### Scaling Laws

::: scpn_control.core.scaling_laws.ipb98y2_tau_e

::: scpn_control.core.scaling_laws.compute_h_factor

### GEQDSK I/O

::: scpn_control.core.eqdsk.GEqdsk

::: scpn_control.core.eqdsk.read_geqdsk

::: scpn_control.core.eqdsk.write_geqdsk

### Uncertainty Quantification

::: scpn_control.core.uncertainty.quantify_uncertainty

::: scpn_control.core.uncertainty.quantify_full_chain

::: scpn_control.core.uncertainty.UQClaimEvidence

::: scpn_control.core.uncertainty.uq_claim_evidence

::: scpn_control.core.uncertainty.assert_uq_calibrated_claim_admissible

::: scpn_control.core.uncertainty.save_uq_claim_evidence

### JAX-Accelerated Transport Primitives

Requires `pip install "scpn-control[jax]"`. GPU execution automatic when jaxlib has CUDA/ROCm.

::: scpn_control.core.jax_solvers.thomas_solve

::: scpn_control.core.jax_solvers.diffusion_rhs

::: scpn_control.core.jax_solvers.crank_nicolson_step

::: scpn_control.core.jax_solvers.batched_crank_nicolson

### DIII-D IDA Free-Boundary Adapter

Requires `pip install "scpn-control[fusion]"`. The adapter validates uniform SI
grids, PF-coil geometry and currents, normalised-flux knots, and compact
p-prime/FF-prime B-spline coefficients before calling the FUSION 4.x implicit
free-boundary solver. It requires JAX FP64, records the JAX and jaxlib versions,
hashes the exact upstream solver and profile-basis source files used at runtime,
and exposes a vector-Jacobian product for coil-current and compact-profile
sensitivities.

The Grad-Shafranov mathematics remains FUSION-owned; CONTROL does not carry a
second implementation in this adapter. Every returned evidence record keeps
scientific validation, facility validation, control admission, PCS deployment,
and safety admission false until separately bound evidence closes those gates.

::: scpn_control.core.ida_equilibrium_adapter.IDAEquilibriumRequest

::: scpn_control.core.ida_equilibrium_adapter.IDAEquilibriumEvidence

::: scpn_control.core.ida_equilibrium_adapter.IDAEquilibriumResult

::: scpn_control.core.ida_equilibrium_adapter.IDAEquilibriumVJPResult

::: scpn_control.core.ida_equilibrium_adapter.solve_ida_equilibrium

::: scpn_control.core.ida_equilibrium_adapter.ida_equilibrium_vjp

### DIII-D IDA Same-Case Evidence Admission

`scpn-control validate-ida-same-case REPORT --fusion-root FUSION_ROOT`
validates the FUSION-owned JSON report, recomputes its self-digest and every
numeric threshold projection, checks the three differentiated-input gradient
rows, and resolves the recorded source files directly from the bound FUSION
Git commit. Omitting `--fusion-root` is allowed for structural inspection but
adds `upstream_source_tree_not_verified`.

Artifact validity and admission are separate. The current v1 report uses a
DIII-D-like case already observed during integration and therefore returns
`artifact_valid=true`, `admitted=false`, and exit code `2` even when all source
bytes authenticate. Facility, PCS, safety, control, and scientific claims
remain false.

::: scpn_control.core.ida_same_case_evidence.IDASameCaseAdmission

::: scpn_control.core.ida_same_case_evidence.load_ida_same_case_report

::: scpn_control.core.ida_same_case_evidence.validate_ida_same_case_evidence

### Differentiable Transport Facade

Requires `pip install "scpn-control[jax]"` for gradient evaluation. The NumPy
path is deterministic for parity checks and non-JAX deployments, but
`transport_loss_gradient()` fails closed without JAX.
`transport_parameter_gradients()` extends the same traced Crank-Nicolson
contract to source schedules, returning JAX gradients for both transport
coefficients and additive heating, fuelling, or impurity-source inputs.
`differentiable_transport_rollout()` advances a bounded multi-step source
schedule with the same four-channel boundary contract.
`transport_rollout_source_gradients()` returns fail-closed JAX gradients for
that full source schedule so controller tuning can optimise time-distributed
heating, fuelling, and impurity-source inputs without finite differences. The
rollout gradient path keeps the loss inside the traced JAX graph and enables
JAX x64 before importing `jax.numpy`, so persisted dtype evidence is not
silently downgraded.
`audit_transport_rollout_source_gradients()` and
`assert_transport_rollout_source_gradients_consistent()` compare those rollout
gradients against sampled NumPy finite-difference perturbations.
`audit_transport_parameter_gradients()` and
`assert_transport_parameter_gradients_consistent()` compare those JAX gradients
against sampled independent finite-difference perturbations before
controller-tuning admission.
`transport_coefficients_from_neural_closure()` maps bounded neural transport
closure outputs into the four-channel coefficient order used by the facade:
electron heat, ion heat, electron particle diffusivity, and a declared impurity
diffusivity fraction.
`gyrokinetic_transport_closure_profiles()` wraps the reduced gyrokinetic
transport profile evaluator as bounded closure provenance, and
`transport_coefficients_from_gyrokinetic_closure()` maps that closure into the
same four-channel coefficient order without promoting the reduced GK model to
an externally validated transport claim.
`transport_campaign_metadata()` records backend, dtype, radial grid, timestep,
boundary conditions, closure provenance, gradient tolerance, and optional
equilibrium-grid shape for reproducible controller-tuning campaigns.
`save_transport_campaign_metadata()` and `load_transport_campaign_metadata()`
persist the same contract as schema-versioned JSON and fail closed on malformed
or physically inconsistent replay metadata.
`assert_transport_campaign_metadata_replay()` compares archived campaign
metadata with a candidate setup and raises on backend, grid, boundary, closure,
gradient-tolerance, or equilibrium-shape drift before controller tuning reruns.
`transport_differentiability_evidence()` and
`assert_transport_differentiability_claim_admissible()` add a tamper-evident
admission envelope over campaign metadata and gradient-audit results. That
envelope requires JAX backend evidence, passed sampled finite-difference
gradient audit, stable SHA-256 digests for the campaign and audit payloads, and
an optional link to the safety-critical controller proof artifact digest.
Admission revalidates finite non-negative audit losses and errors, tolerance
agreement with campaign metadata, unique in-domain sampled audit indices,
pass/fail consistency with maximum audit error, and ordered latency percentiles
before persisted controller-tuning evidence is accepted. Latency reports also
bind runtime provenance for later CPU/GPU comparison: Python version, platform,
machine class, JAX and jaxlib versions, default backend, visible JAX devices,
and x64 state.
`transport_full_fidelity_readiness_evidence()` and
`assert_transport_full_fidelity_claim_ready()` add the stricter promotion gate
for full-fidelity differentiable-transport claims. The gate binds campaign
metadata, one-step gradient-latency evidence, rollout source-gradient latency
evidence, audit digests, controller formal-proof digests, equilibrium-coupled
metadata, and an admitted external reference artifact. The repository benchmark
binds the canonical payload SHA-256 from `validation/reports/scpn_z3_formal.json`
when that bounded Z3 formal report is present and passing. Without all of those
inputs the claim remains bounded local differentiability evidence.
`equilibrium_weighted_transport_rollout_tracking_loss()` extends the optional
Grad-Shafranov flux-map weighting from one transport step to a full source
rollout. `equilibrium_weighted_transport_rollout_source_gradient()` returns
fail-closed JAX gradients with respect to both the source schedule and the
equilibrium flux map for controller-tuning studies.

### Differentiable Transport Core Step / Rollout

One-step Crank-Nicolson channel advance and multi-step source-schedule rollout
live in a dedicated numerics leaf. The numerical facade re-exports these
symbols and retains validators plus campaign metadata.

::: scpn_control.core.differentiable_transport_core.differentiable_transport_step

::: scpn_control.core.differentiable_transport_core.differentiable_transport_rollout

### Differentiable Transport Parameter AD

One-step tracking loss, chi/source parameter gradients, and sampled
finite-difference admission audits live in a dedicated leaf. The numerical
facade re-exports these symbols; step primitives remain on the core leaf.

::: scpn_control.core.differentiable_transport_parameter_ad.transport_tracking_loss

::: scpn_control.core.differentiable_transport_parameter_ad.transport_loss_gradient

::: scpn_control.core.differentiable_transport_parameter_ad.transport_parameter_gradients

::: scpn_control.core.differentiable_transport_parameter_ad.TransportParameterGradients

::: scpn_control.core.differentiable_transport_parameter_ad.audit_transport_parameter_gradients

::: scpn_control.core.differentiable_transport_parameter_ad.assert_transport_parameter_gradients_consistent

::: scpn_control.core.differentiable_transport.TransportGradientAudit

### Differentiable Transport Rollout Source AD

Multi-step tracking loss, source-schedule gradients, and sampled
finite-difference admission audits live in a dedicated leaf. The numerical
facade re-exports these symbols; core rollout primitives remain on the core leaf.

::: scpn_control.core.differentiable_transport_rollout_ad.transport_rollout_tracking_loss

::: scpn_control.core.differentiable_transport_rollout_ad.transport_rollout_source_gradients

::: scpn_control.core.differentiable_transport_rollout_ad.TransportRolloutSourceGradients

::: scpn_control.core.differentiable_transport_rollout_ad.audit_transport_rollout_source_gradients

::: scpn_control.core.differentiable_transport_rollout_ad.assert_transport_rollout_source_gradients_consistent

::: scpn_control.core.differentiable_transport.TransportRolloutGradientAudit

### Differentiable Transport Latency Benchmarks

Local timed admission of JAX gradient contracts lives in a dedicated leaf.
The numerical facade re-exports these symbols; report serialisation remains on
the evidence leaf.

::: scpn_control.core.differentiable_transport_latency.transport_runtime_metadata

::: scpn_control.core.differentiable_transport_latency.benchmark_transport_parameter_gradient_latency

::: scpn_control.core.differentiable_transport.TransportGradientLatencyReport

::: scpn_control.core.differentiable_transport.save_transport_gradient_latency_report

::: scpn_control.core.differentiable_transport_latency.benchmark_transport_rollout_source_gradient_latency

::: scpn_control.core.differentiable_transport.TransportRolloutGradientLatencyReport

::: scpn_control.core.differentiable_transport.save_transport_rollout_gradient_latency_report

### Differentiable Transport Closures

Neural and reduced gyrokinetic closure → four-channel coefficient adapters live
in a dedicated leaf module. The numerical facade re-exports these symbols so
existing `scpn_control.core.differentiable_transport` imports remain stable.

::: scpn_control.core.differentiable_transport_closures.transport_coefficients_from_neural_closure

::: scpn_control.core.differentiable_transport_closures.gyrokinetic_transport_closure_profiles

::: scpn_control.core.differentiable_transport_closures.transport_coefficients_from_gyrokinetic_closure

::: scpn_control.core.differentiable_transport_closures.GyrokineticTransportClosureResult

::: scpn_control.core.differentiable_transport.transport_campaign_metadata

::: scpn_control.core.differentiable_transport.assert_transport_campaign_metadata_replay

### Differentiable Transport Evidence

Campaign metadata types, gradient-audit certificates, latency-report
serialisation, and full-fidelity readiness claims live in a dedicated leaf
module. The numerical facade re-exports these symbols so existing
`scpn_control.core.differentiable_transport` imports remain stable.

::: scpn_control.core.differentiable_transport_evidence.TransportCampaignMetadata

::: scpn_control.core.differentiable_transport_evidence.save_transport_campaign_metadata

::: scpn_control.core.differentiable_transport_evidence.load_transport_campaign_metadata

::: scpn_control.core.differentiable_transport_evidence.TransportDifferentiabilityEvidence

::: scpn_control.core.differentiable_transport_evidence.transport_differentiability_evidence

::: scpn_control.core.differentiable_transport_evidence.assert_transport_differentiability_claim_admissible

::: scpn_control.core.differentiable_transport_evidence.TransportFullFidelityReadinessEvidence

::: scpn_control.core.differentiable_transport_evidence.transport_full_fidelity_readiness_evidence

::: scpn_control.core.differentiable_transport_evidence.assert_transport_full_fidelity_claim_ready

::: scpn_control.core.differentiable_transport_evidence.TransportGradientAudit

::: scpn_control.core.differentiable_transport_evidence.TransportRolloutGradientAudit

::: scpn_control.core.differentiable_transport_evidence.TransportGradientLatencyReport

::: scpn_control.core.differentiable_transport_evidence.TransportRolloutGradientLatencyReport

::: scpn_control.core.differentiable_transport_evidence.save_transport_gradient_latency_report

::: scpn_control.core.differentiable_transport_evidence.save_transport_rollout_gradient_latency_report

### Differentiable Transport Equilibrium Weighting

GS-flux radial weights and equilibrium-weighted one-step / multi-step losses
and JAX gradients live in a dedicated leaf. The numerical facade re-exports
these symbols; core step and rollout primitives remain on the facade.

::: scpn_control.core.differentiable_transport_equilibrium_weight.equilibrium_radial_weights

::: scpn_control.core.differentiable_transport_equilibrium_weight.equilibrium_weighted_transport_tracking_loss

::: scpn_control.core.differentiable_transport_equilibrium_weight.equilibrium_weighted_transport_loss_gradient

::: scpn_control.core.differentiable_transport_equilibrium_weight.EquilibriumWeightedTransportGradient

::: scpn_control.core.differentiable_transport_equilibrium_weight.equilibrium_weighted_transport_rollout_tracking_loss

::: scpn_control.core.differentiable_transport_equilibrium_weight.equilibrium_weighted_transport_rollout_source_gradient

::: scpn_control.core.differentiable_transport_equilibrium_weight.EquilibriumWeightedTransportRolloutGradient

### End-to-End Differentiable Scenario

`differentiable_scenario_gradient()` couples a bounded analytic Solov'ev-form
equilibrium parametrisation to the differentiable transport rollout, so a
controller-tuning loss is differentiable with respect to both the source
schedule and the equilibrium shape parameters. The flux parametrisation is a
differentiable equilibrium surface, not a Grad-Shafranov PDE solve. Gradient
APIs fail closed without JAX, and `assert_scenario_claim_admissible()` keeps a
full-fidelity claim bounded until the gradient audit, latency evidence, and
physics-traceability checks pass.
Persisted readiness reports are validated with
`validation.validate_differentiable_scenario.validate_differentiable_scenario_report()`;
the checked-in report remains blocked on physics traceability rather than
promoting the analytic surface to a facility-grade scenario claim.

::: scpn_control.core.differentiable_scenario.scenario_equilibrium_flux

::: scpn_control.core.differentiable_scenario.differentiable_scenario_loss

::: scpn_control.core.differentiable_scenario.differentiable_scenario_gradient

::: scpn_control.core.differentiable_scenario.DifferentiableScenarioGradient

::: scpn_control.core.differentiable_scenario.audit_differentiable_scenario_gradient

::: scpn_control.core.differentiable_scenario.assert_differentiable_scenario_gradient_consistent

::: scpn_control.core.differentiable_scenario.DifferentiableScenarioGradientAudit

::: scpn_control.core.differentiable_scenario.scenario_campaign_metadata

::: scpn_control.core.differentiable_scenario.ScenarioCampaignMetadata

::: scpn_control.core.differentiable_scenario.differentiable_scenario_readiness_evidence

::: scpn_control.core.differentiable_scenario.assert_scenario_claim_admissible

::: scpn_control.core.differentiable_scenario.ScenarioReadinessEvidence

### Neural Equilibrium

`NeuralEquilibriumAccelerator.pretrain_from_synthetic_equilibria()` trains
JAX-compatible PCA plus MLP weights on bounded synthetic Solovev-like
equilibria for pretraining. The corresponding real EFIT fine-tuning entry point
`fine_tune_from_efit_reconstructions()` fails closed unless the persisted
P-EFIT or documented-public-reference artefact validator passes.

::: scpn_control.core.neural_equilibrium.NeuralEquilibriumAccelerator

::: scpn_control.core.neural_equilibrium.NeuralEquilibriumClaimEvidence

::: scpn_control.core.neural_equilibrium.generate_synthetic_equilibrium_dataset

::: scpn_control.core.neural_equilibrium.neural_equilibrium_claim_evidence

::: scpn_control.core.neural_equilibrium.assert_neural_equilibrium_facility_claim_admissible

::: scpn_control.core.neural_equilibrium.save_neural_equilibrium_claim_evidence

::: scpn_control.core.neural_equilibrium.pretrain_neural_equilibrium_synthetic

::: scpn_control.core.neural_equilibrium.PretrainingResult

::: scpn_control.core.neural_equilibrium.SyntheticEquilibriumCampaign

### Equilibrium Shape & Profile Metrics

Reusable, reconstruction-agnostic extraction of macroscopic descriptors
(R0/minor radius/elongation/triangularity, `li(3)`, poloidal beta, and `q95`)
from a poloidal-flux map and the fitted `p'`/`FF'` profiles. Used by the
real-time EFIT inverse and shareable by the free-boundary kernel and kinetic
EFIT. The poloidal field uses the per-radian convention `B_pol = |grad psi| / R`
(Ampere-consistent); `q95` requires `contourpy` for the flux-surface contour.

::: scpn_control.core.equilibrium_shape.EquilibriumShape

::: scpn_control.core.equilibrium_shape.compute_equilibrium_shape

::: scpn_control.core.equilibrium_shape.boundary_geometry

::: scpn_control.core.equilibrium_shape.poloidal_field

::: scpn_control.core.equilibrium_shape.internal_inductance

::: scpn_control.core.equilibrium_shape.poloidal_beta

::: scpn_control.core.equilibrium_shape.safety_factor_q95

::: scpn_control.core.equilibrium_shape.cylindrical_q95

::: scpn_control.core.equilibrium_shape.pressure_grid

::: scpn_control.core.equilibrium_shape.plasma_boundary

::: scpn_control.core.equilibrium_shape.largest_flux_contour

### JAX-Accelerated Neural Equilibrium

Requires `pip install "scpn-control[jax]"`. GPU and autodiff via `jax.grad`.

::: scpn_control.core.jax_neural_equilibrium.jax_neural_eq_predict

::: scpn_control.core.jax_neural_equilibrium.jax_neural_eq_predict_batched

::: scpn_control.core.jax_neural_equilibrium.load_weights_as_jax

### Neural Transport

`cross_validate_neural_transport()` benchmarks the active surrogate against the
analytic critical-gradient reference across fixed regime cases and a canonical
profile, so shipped weights can be checked against a deterministic local
baseline instead of only reporting synthetic training RMSE.

`neural_transport_closure_profiles()` packages profile transport coefficients
for controller and differentiable-transport coupling.  It validates finite
strictly ordered profile inputs, fails closed when neural weights are required
but unavailable, and records whether coefficients came from loaded weights or
the analytic fallback.

::: scpn_control.core.neural_transport.NeuralTransportModel

::: scpn_control.core.neural_transport.NeuralTransportClosureResult

::: scpn_control.core.neural_transport.NeuralTransportClaimEvidence

::: scpn_control.core.neural_transport.neural_transport_closure_profiles

::: scpn_control.core.neural_transport.cross_validate_neural_transport

::: scpn_control.core.neural_transport.neural_transport_claim_evidence

::: scpn_control.core.neural_transport.assert_neural_transport_quantitative_claim_admissible

::: scpn_control.core.neural_transport.save_neural_transport_claim_evidence

### MHD Stability

::: scpn_control.core.stability_mhd.run_full_stability_check

### IMAS Adapter

The canonical data boundary is `EquilibriumSnapshot`: immutable SI arrays in
COCOS 1, with two-dimensional fields ordered `(Z, R)` and poloidal flux in
webers per radian. `export_imas_equilibrium()` and
`import_imas_equilibrium()` support real IMAS-Python Data Dictionary v3 and v4
IDS objects. The v3/v4 current leaf and COCOS differences are explicit;
`write_imas_entry()` and `read_imas_entry()` operate on a caller-owned open
`imas.DBEntry`. The OMAS pair operates on real v3 ODS objects with OMAS COCOS
conversion enabled. No adapter invents missing current density, machine
metadata, facility endpoints, or credentials.

Install one backend with `scpn-control[imas]` or `scpn-control[omas]`, or both
with `scpn-control[fusion-data]`.

::: scpn_control.core.imas_adapter.EquilibriumSnapshot

::: scpn_control.core.imas_adapter.EquilibriumDataError

::: scpn_control.core.imas_adapter.EquilibriumBackendUnavailableError

::: scpn_control.core.imas_adapter.EquilibriumBackendError

::: scpn_control.core.imas_adapter.snapshot_from_kernel

::: scpn_control.core.imas_adapter.snapshot_to_kernel_state

::: scpn_control.core.imas_adapter.snapshot_from_geqdsk

::: scpn_control.core.imas_adapter.export_imas_equilibrium

::: scpn_control.core.imas_adapter.import_imas_equilibrium

::: scpn_control.core.imas_adapter.write_imas_entry

::: scpn_control.core.imas_adapter.read_imas_entry

::: scpn_control.core.imas_adapter.export_omas_equilibrium

::: scpn_control.core.imas_adapter.import_omas_equilibrium

The following names are deprecated compatibility facades scheduled for removal
in 0.25.0. New code should use the canonical surfaces above.

::: scpn_control.core.imas_adapter.EquilibriumIDS

::: scpn_control.core.imas_adapter.from_geqdsk

::: scpn_control.core.imas_adapter.from_kernel

::: scpn_control.core.imas_adapter.to_kernel_arrays

::: scpn_control.core.imas_adapter.to_omas

::: scpn_control.core.imas_adapter.from_omas

### HPC Bridge

`HPCBridge` loads compiled Grad-Shafranov solver libraries only from absolute
dynamic-library paths. Package-local solver libraries are trusted by default.
External paths provided through `SCPN_SOLVER_LIB` require the additional
operator gate `SCPN_ALLOW_EXTERNAL_SOLVER_LIB=1`; without that gate the bridge
fails closed before calling the dynamic loader.

::: scpn_control.core.hpc_bridge.HPCBridge

::: scpn_control.core.hpc_bridge.NativeSolverStatus

::: scpn_control.core.hpc_bridge.NativeSolverError

### Gyrokinetic Transport (v0.16.0)

::: scpn_control.core.gyrokinetic_transport.GyrokineticTransportModel

### GK Solver Interface (v0.17.0)

::: scpn_control.core.gk_interface.GKSolverBase

::: scpn_control.core.gk_interface.GKLocalParams

::: scpn_control.core.gk_interface.GKOutput

### Native Linear GK Solver (v0.17.0)

::: scpn_control.core.gk_eigenvalue.solve_linear_gk

::: scpn_control.core.gk_quasilinear.quasilinear_fluxes_from_spectrum

### GK Hybrid Validation (v0.17.0)

::: scpn_control.core.gk_ood_detector.OODDetector

::: scpn_control.core.gk_scheduler.GKScheduler

::: scpn_control.core.gk_corrector.GKCorrector

### Ballooning Solver (v0.16.0)

::: scpn_control.core.ballooning_solver.BallooningEquation

::: scpn_control.core.ballooning_solver.BallooningStabilityAnalysis

::: scpn_control.core.ballooning_solver.find_marginal_stability

### Current Diffusion (v0.16.0)

::: scpn_control.core.current_diffusion.CurrentDiffusionSolver

### Current Drive (v0.16.0)

::: scpn_control.core.current_drive.ECCDSource

::: scpn_control.core.current_drive.NBISource

::: scpn_control.core.current_drive.CurrentDriveMix

### NTM Dynamics (v0.16.0)

::: scpn_control.core.ntm_dynamics.RationalSurface

::: scpn_control.core.ntm_dynamics.NTMIslandDynamics

::: scpn_control.core.ntm_dynamics.NTMController

::: scpn_control.core.ntm_dynamics.eccd_stabilization_factor

::: scpn_control.core.ntm_dynamics.find_rational_surfaces

::: scpn_control.core.ntm_dynamics.bootstrap_from_local

### Sawtooth Model (v0.16.0)

::: scpn_control.core.sawtooth.SawtoothCycler

::: scpn_control.core.sawtooth.kadomtsev_crash

### SOL Model (v0.16.0)

::: scpn_control.core.sol_model.TwoPointSOL

### Integrated Scenario (v0.16.0)

Scenario configuration presets live in a dedicated leaf and are re-exported from
the owner module:

::: scpn_control.core.integrated_scenario_presets.ScenarioConfig

::: scpn_control.core.integrated_scenario_presets.iter_baseline_scenario

::: scpn_control.core.integrated_scenario.IntegratedScenarioSimulator

::: scpn_control.core.integrated_scenario.audit_scenario_coupling

::: scpn_control.core.integrated_scenario.save_scenario_coupling_report

Coupling-audit types and report I/O live in a dedicated leaf and are re-exported
from the owner module:

::: scpn_control.core.integrated_scenario_coupling_audit.ScenarioModuleExchange

::: scpn_control.core.integrated_scenario_coupling_audit.ScenarioCouplingMetadata

::: scpn_control.core.integrated_scenario_coupling_audit.ScenarioCouplingAudit

::: scpn_control.core.integrated_scenario_coupling_audit.audit_scenario_coupling

::: scpn_control.core.integrated_scenario_coupling_audit.scenario_coupling_audit_to_dict

::: scpn_control.core.integrated_scenario_coupling_audit.save_scenario_coupling_report

Transport micro-physics helpers live in a dedicated leaf and are re-exported by
`integrated_scenario`:

::: scpn_control.core.integrated_scenario_micro_physics

::: scpn_control.core.integrated_scenario.iter_baseline_scenario

::: scpn_control.control.closed_loop_scenario.run_integrated_scenario_closed_loop

::: scpn_control.control.closed_loop_scenario.closed_loop_scenario_result_to_dict

---

## SCPN — Petri Net Compiler

### StochasticPetriNet

`verify_liveness()` reports random-campaign transition-fire coverage only when
the sampled Petri-net walk also preserves finite `[0, 1]` markings before any
controller-style clipping. If a firing step produces an out-of-range or
non-finite marking, the report includes `marking_bounds_valid=false`, records
the first violating transition, and returns `live=false`.

::: scpn_control.scpn.structure.StochasticPetriNet

### Formal Verification

`FormalPetriNetVerifier` uses exact explicit-state reachability over the
compiled Petri-net transition relation. `backend="auto"` records that
explicit-state backend and does not relabel the result as z3 just because the
optional solver package is importable. `backend="z3"` is an explicit opt-in and
requires the optional `z3-solver` package; SMT-specific report artefacts remain
on the separate z3 model-checking surface below.

::: scpn_control.scpn.formal_verification.FormalPetriNetVerifier

::: scpn_control.scpn.formal_verification.verify_formal_contracts

::: scpn_control.scpn.formal_verification.PlaceInvariant

::: scpn_control.scpn.formal_verification.CTLFormula

::: scpn_control.scpn.formal_verification.LTLFormula

::: scpn_control.scpn.formal_verification.AlwaysBounded

::: scpn_control.scpn.formal_verification.AlwaysEventuallyMarked

::: scpn_control.scpn.formal_verification.EventuallyFires

::: scpn_control.scpn.formal_verification.FireLeadsToMarking

::: scpn_control.scpn.formal_verification.NeverCoMarked

### Formal Safety Certificate

The bounded safety certificate captures a formal verification report together
with its admission policy and an optional artifact binding; the bundle
aggregates independent certificates for a controller release gate. Every payload
is schema-versioned, self-digested, and fail-closed.

::: scpn_control.scpn.formal_safety_certificate.SafetyCertificatePolicy

::: scpn_control.scpn.formal_safety_certificate.SafetyCertificateBundlePolicy

::: scpn_control.scpn.formal_safety_certificate.build_safety_certificate_payload

::: scpn_control.scpn.formal_safety_certificate.build_safety_certificate_bundle_payload

::: scpn_control.scpn.formal_safety_certificate.build_safety_certificate_bundle_artifact

::: scpn_control.scpn.formal_safety_certificate.generate_safety_certificate

::: scpn_control.scpn.formal_safety_certificate.validate_safety_certificate_payload

::: scpn_control.scpn.formal_safety_certificate.validate_safety_certificate_bundle_payload

::: scpn_control.scpn.formal_safety_certificate.validate_safety_certificate_bundle_artifact

::: scpn_control.scpn.formal_safety_certificate.admit_safety_certificate_bundle_artifact

::: scpn_control.scpn.formal_safety_certificate.write_safety_certificate

::: scpn_control.scpn.formal_safety_certificate.write_safety_certificate_bundle

### Runtime-Bound Safety Certificate

::: scpn_control.scpn.runtime_safety_certificate.RuntimeTarget

::: scpn_control.scpn.runtime_safety_certificate.TimingEnvelope

::: scpn_control.scpn.runtime_safety_certificate.ControllerRuntimeBinding

::: scpn_control.scpn.runtime_safety_certificate.CertificateReplayResult

::: scpn_control.scpn.runtime_safety_certificate.compute_petri_topology_digest

::: scpn_control.scpn.runtime_safety_certificate.issue_runtime_safety_certificate

::: scpn_control.scpn.runtime_safety_certificate.validate_runtime_safety_certificate_payload

::: scpn_control.scpn.runtime_safety_certificate.replay_runtime_safety_certificate

::: scpn_control.scpn.runtime_safety_certificate.assert_runtime_certificate_admissible

### Runtime Deadline Monitor

`scpn_control.scpn.deadline_monitor` closes the gap between the certificate's
declared `deadline_us` and the actual control-cycle duration: each measured cycle
is checked against the admitted deadline, fail-soft by default (overruns counted
and exposed) with an opt-in strict mode that raises on a single overrun.

::: scpn_control.scpn.deadline_monitor.DeadlineMonitor

::: scpn_control.scpn.deadline_monitor.DeadlineOverrunError

::: scpn_control.scpn.z3_model_checking.Z3BoundedModelChecker

### Z3 Formal Report I/O

::: scpn_control.scpn.z3_formal_report.verify_z3_formal_contracts

::: scpn_control.scpn.z3_formal_report.build_z3_formal_report_payload

::: scpn_control.scpn.z3_formal_report.build_blocked_z3_formal_report_payload

::: scpn_control.scpn.z3_formal_report.validate_z3_formal_report_payload

::: scpn_control.scpn.z3_formal_report.load_z3_formal_report

::: scpn_control.scpn.z3_formal_report.write_z3_formal_report

### FusionCompiler

::: scpn_control.scpn.compiler.FusionCompiler

### CompiledNet

::: scpn_control.scpn.compiler.CompiledNet

Inhibitor arcs remain a structure/formal-analysis feature. A caller may compile
them only with `allow_inhibitor=True`, which stores the guard as a negative
input-matrix entry for analysis paths that still inspect the original Petri-net
structure. Controller artifacts and controller runtime do not encode inhibitor
topology yet: `CompiledNet.export_artifact()`, `load_artifact()`, and
`NeuroSymbolicController` reject negative dense `w_in` weights instead of
serializing ambiguous inhibitor semantics.

### NeuroSymbolicController

`NeuroSymbolicController` rejects nonzero `sc_bitflip_rate` unless
`allow_fault_injection=True` is supplied explicitly and the process environment
sets `SCPN_ALLOW_CONTROLLER_FAULT_INJECTION=1`. Bit-flip mutation is a
double-gated fault-injection test mode, not a production control default.

Controller JSONL logging requires an explicit `log_root` whenever `log_path` is
provided. Relative and absolute log paths must resolve under that root and use a
`.jsonl` suffix before any file is opened. Log appends use a constrained append
helper that rejects symlink targets where the platform exposes no-follow open
semantics.

Runtime-bound safety admission is available through the explicit
`runtime_safety_certificate`, `runtime_safety_binding`, `runtime_safety_target`,
and `runtime_safety_replay` constructor arguments. Supplying any one of these
requires all four. The controller first checks that the runtime binding's Petri
topology digest matches the loaded `.scpnctl` artifact, then delegates to
`assert_runtime_certificate_admissible`. This keeps ordinary local experiments
unchanged while making safety-critical construction fail closed unless artifact,
binding, target, and proof replay evidence all match.

::: scpn_control.scpn.controller.NeuroSymbolicController

### Contracts

`scpn_control.scpn.contracts` owns the shared numeric kernels used by both the
public contract helpers and the live controller runtime. `extract_features()`
and `NeuroSymbolicController.step()` use the same signed-error component kernel;
`decode_actions()` and the controller readout use the same slew-rate and
absolute-limit action-vector kernel. This keeps standalone contract checks and
compiled controller execution aligned.

The `scpn_control.scpn` package root re-exports `FeatureAxisSpec`, the shared
feature/action kernels, and the runtime-safety certificate dataclasses and
helpers so installed-package consumers do not need to reach into private module
paths for controller construction and certificate admission.

::: scpn_control.scpn.contracts.ControlObservation

::: scpn_control.scpn.contracts.ControlAction

::: scpn_control.scpn.contracts.ControlTargets

::: scpn_control.scpn.contracts.extract_features

::: scpn_control.scpn.contracts.feature_error_components

::: scpn_control.scpn.contracts.decode_actions

::: scpn_control.scpn.contracts.decode_action_vector

### Artifacts

Safety-critical controller admission must call `load_artifact(...,
require_formal_verification=True)` or `validate_safety_critical_artifact()`.
That gate rejects missing, blocked, failed, malformed, or unbounded proof
evidence and accepts only hash-addressed bounded proof manifests tied to the
compiled controller artifact. The proof manifest must include the canonical
artifact payload SHA-256, report SHA-256, bounded proof depth, checked
specification names, backend/solver metadata, and a safe relative report URI.
Controller artifacts also carry `meta.firing_margin`, the default fractional
firing margin used when a transition omits its own margin. The compiler writes
this value, artifact loading validates it as finite and non-negative, and the
controller consumes it directly instead of falling back to an implicit runtime
constant. Legacy artifacts without the field still load with the historical
`0.05` default so archived evidence remains readable.
`Artifact(...)` validates direct construction by default, `load_artifact()` calls
the same public `validate_artifact()` surface after parsing, and
`NeuroSymbolicController` calls `validate_artifact()` before runtime arrays,
runtime certificates, or controller state are admitted. Validator branch tests
can pass `validate_on_init=False` only to build a deliberately malformed object;
that object is still rejected by `validate_artifact()`, `save_artifact()`, and
controller construction.
When callers provide `formal_report_root`, the loader resolves the report URI
under that root and verifies the report bytes against the declared SHA-256.
Z3 and Lean report files reached through an artifact manifest are loaded through
duplicate-key-safe and schema-strict public report loaders.
Z3 reports are additionally schema-versioned as
`scpn-control.z3-formal-report.v2`, carry a canonical payload SHA-256 over the
proof payload, reject unknown top-level and proof-section fields, schema-check
serialized counterexample records, enforce solver-status/holds/counterexample
consistency, reject counterexamples on `unknown` solver sections, and must
match the manifest status, solver, proof depth, and
checked specification list before a safety-critical artifact is admitted.
Temporal sections distinguish a witness-producing `sat` result from an
`unsat` counterexample search, publish `mixed` when both occur in one admitted
contract, and publish `not-run` when no temporal obligation was submitted.
Existential firing obligations require exact unit transition weights; a
parametric or fractional weight envelope is rejected instead of being treated
as a discrete firing witness. Version 1 reports are not admitted under the new
contract and must be regenerated, because they cannot distinguish a successful
existential SAT witness from a universal UNSAT proof.
Each Z3 proof section must also carry unique non-empty `checked_specs`; duplicate
section obligations are rejected before top-level report/spec matching.
Blocked Z3 reports are not proof evidence: they must use the unavailable solver
label, zero proof depth, and only the `z3_solver_available` checked
specification. Pass/fail Z3 reports must identify `z3-solver` and must not
use the unavailable-solver label; a missing `z3-solver` dependency is
admitted only as blocked SMT evidence.
Lean 4 reports are admitted only through the bounded `lean4` manifest path:
the manifest must bind a solver string that identifies Lean and includes the
declared Lean version, Lake file SHA-256, proof-source SHA-256, theorem names,
theorem modules, proved contracts, linked production module references,
safety-case identifiers, checked specification list, report SHA-256, and
compiled artifact SHA-256. Production module references should use importable
module names such as `scpn_control.scpn.controller` so installed wheels and
sdists do not depend on a repository `src/` checkout; legacy safe relative
source paths remain accepted for existing reports. The Lean report schema is
`scpn-control.lean4-formal-report.v1`; when a report root is supplied, every
manifest field above must match the report before admission. The current
required Lean proof-contract surface covers PID actuator-saturation preservation
and SNN/neuro-symbolic marking-bound preservation, and reports cannot list
unsupported proved contracts to imply wider formal coverage. This is an evidence
admission contract; it does not claim certification unless the referenced
machine-checked proof artefacts are present and verified.
The admitted PID/SNN surface is also exact-linked: theorem modules, theorem
names, linked production module references, and safety-case identifiers must
remain inside the expected PID/SNN namespaces and module references instead of
padding a valid report with unrelated evidence.
`get_artifact_json_schema()` returns the current `.scpnctl.json` Draft-07
schema from the same payload sections and dataclass field sets used by
`save_artifact()` and `load_artifact()`. The schema declares the serialized
`meta.firing_margin`, closed dense-weight and readout objects, the raw
`data_u64` packed-weight form, the compact `u64-le-zlib-base64` packed-weight
form, and the closed `formal_verification` manifest fields admitted by the
runtime validator.

::: scpn_control.scpn.artifact.Artifact

::: scpn_control.scpn.artifact.FormalVerificationEvidence

Topology/meta/weight/readout model dataclasses live in a dedicated leaf and are
re-exported from the owner module:

::: scpn_control.scpn.artifact_model.Topology

::: scpn_control.scpn.artifact_model.Weights

Structural validation and safety-critical admit live in a dedicated leaf and are
re-exported from the owner module:

::: scpn_control.scpn.artifact_validate.validate_artifact

::: scpn_control.scpn.artifact_validate.validate_safety_critical_artifact

Compact packed-weight codec helpers live in a dedicated leaf and are re-exported
from the owner module:

::: scpn_control.scpn.artifact_codec.encode_u64_compact

::: scpn_control.scpn.artifact_codec.decode_u64_compact

Load/save and payload hashing live in a dedicated leaf and are re-exported from
the owner module:

::: scpn_control.scpn.artifact_io.load_artifact

::: scpn_control.scpn.artifact_io.save_artifact

::: scpn_control.scpn.artifact_io.compute_artifact_payload_sha256

JSON Schema emission lives in a dedicated leaf and is re-exported from the owner
module:

::: scpn_control.scpn.artifact_schema.get_artifact_json_schema

::: scpn_control.scpn.artifact.compute_artifact_payload_sha256

::: scpn_control.scpn.artifact.validate_artifact

::: scpn_control.scpn.artifact.get_artifact_json_schema

::: scpn_control.scpn.artifact.save_artifact

::: scpn_control.scpn.artifact.load_artifact

::: scpn_control.scpn.artifact.validate_safety_critical_artifact

::: scpn_control.scpn.lean_verification.LeanFormalVerificationReport

::: scpn_control.scpn.lean_verification.build_lean_formal_report_payload

::: scpn_control.scpn.lean_verification.validate_lean_formal_report_payload

::: scpn_control.scpn.lean_verification.write_lean_formal_report

::: scpn_control.scpn.lean_verification.load_lean_formal_report

`load_lean_formal_report()` and the validation executable
`validation/validate_scpn_lean_formal.py` validate Lean report JSON with
duplicate-key rejection and can additionally call
`load_artifact(..., require_formal_verification=True)` against a supplied
artifact and report root. This is the release-gate path for admitting Lean proof
evidence without running long proof jobs inside the Python test suite.
When an artifact is supplied, this validator also requires its backend to be
Lean and its report digest to match the explicitly named report bytes. A
separately valid report cannot substitute for that artifact reference. These
are schema/declaration/byte-digest checks; the validator executes no Lean/Lake
process and authenticates no proof-source bytes. Reads are sequential without
snapshot locks. With no formal-report root, the artifact loader validates its
manifest without opening its report URI.

The Lean report and artifact manifest validators also enforce namespace
coverage for required controller contracts: PID actuator-saturation evidence
must include a `ScpnControl.PID` theorem module and theorem name, while
SNN/neuro-symbolic marking-bound evidence must include a `ScpnControl.SNN`
theorem module and theorem name.
Lean evidence must also declare explicit bounded `proof_assumptions` and a
canonical `assumption_sha256`; report and artifact admission reject unbounded
assumptions, certification overclaims, malformed assumption digests, or
manifest/report assumption mismatches.
Admission also rejects non-Lean solver declarations, Lean solver strings that
do not include the reported `lean_version`, and any `proved_contracts` outside
the currently admitted PID/SNN proof surface.
Reports also fail closed when `theorem_names`, `theorem_modules`,
`module_paths`, or `safety_case_ids` include unrelated entries outside the
admitted PID/SNN proof boundary. Despite the historical field name,
`module_paths` admits importable module references for installed packages and
legacy safe relative source paths for existing reports.
The same checks run on the `.scpnctl` artifact manifest itself, even without a
report root, so safety-critical artifact loading cannot admit a stale or padded
Lean manifest before report-byte comparison is available.
Both Lean report payloads and artifact `formal_verification` manifests are
closed schemas: unknown proof fields are rejected rather than ignored.
External Lean reports must also carry the canonical `payload_sha256` self-digest;
reports that omit it are rejected before safety-critical artifact admission.

---

## Phase — Paper 27 Dynamics

These APIs implement and transport an example oscillator model. They do not
identify oscillator states from reactor observations, connect to a plant
solver, map outputs to physical actuators, or establish machine-protection
authority. The model-local guard and network admission states must not be
interpreted as reactor safety verdicts.

### Kuramoto-Sakaguchi Step

::: scpn_control.phase.kuramoto.kuramoto_sakaguchi_step

`kuramoto_sakaguchi_step()` may dispatch to the optional Rust backend for
wrapped phase updates. Treat latency as benchmark-context evidence; source
comments and docstrings do not make fixed timing claims.

::: scpn_control.phase.kuramoto.order_parameter

::: scpn_control.phase.kuramoto.lyapunov_v

::: scpn_control.phase.kuramoto.lyapunov_exponent

`lyapunov_exponent()` validates a positive finite `dt` and finite,
non-negative `V(t)` samples. Its heuristic floors only the initial and final
sample at `LYAPUNOV_VALUE_FLOOR` before the log ratio, then divides by
`(n_samples - 1) * dt` because the input is a sampled state history.

::: scpn_control.phase.kuramoto.wrap_phase

::: scpn_control.phase.kuramoto.GlobalPsiDriver

::: scpn_control.phase.kuramoto.KuramotoRuntimeEvidence

::: scpn_control.phase.kuramoto.kuramoto_runtime_evidence

::: scpn_control.phase.kuramoto.assert_kuramoto_runtime_claim_admissible

::: scpn_control.phase.kuramoto.save_kuramoto_runtime_evidence

::: scpn_control.phase.kuramoto.load_kuramoto_runtime_evidence

### Knm Coupling Matrix

::: scpn_control.phase.knm.KnmSpec

::: scpn_control.phase.knm.build_knm_paper27

### UPDE Multi-Layer Solver

::: scpn_control.phase.upde.UPDESystem

`UPDESystem.step()` returns a single output-state snapshot contract across the
NumPy fallback and the optional Rust/PyO3 path. `theta1`, `dtheta`, `R_layer`,
`Psi_layer`, `R_global`, `Psi_global`, `V_layer`, and `V_global` describe the
same completed tick. The input `psi_driver` still drives the derivative term,
but the returned `Psi_global` is the mean phase of `theta1`. Stale Rust bindings
that cannot provide the full contract fall back to the NumPy implementation
instead of returning partial snapshots.

### Lyapunov Guard

::: scpn_control.phase.lyapunov_guard.LyapunovGuard

### Realtime Monitor

::: scpn_control.phase.realtime_monitor.RealtimeMonitor

::: scpn_control.phase.realtime_monitor.TrajectoryRecorder

### Adaptive Knm Engine

::: scpn_control.phase.adaptive_knm.AdaptiveKnmEngine

::: scpn_control.phase.adaptive_knm.AdaptiveKnmConfig

::: scpn_control.phase.adaptive_knm.DiagnosticSnapshot

The adaptive Knm engine uses every required diagnostic field in
`DiagnosticSnapshot`: `R_layer` and `V_layer` drive the diagonal coherence
channel, while `lambda_exp`, `q95`, `disruption_risk`, and `mirnov_rms`
contribute to a bounded MHD-pair risk drive. The configuration defaults are
dimensionless local-control gains except `lambda_risk_gain_s`, which converts a
Lyapunov exponent in `1/s` into a dimensionless stress contribution. These
settings are bounded model heuristics, not facility-calibrated stability claims
or an identified reactor feedback law.

### Plasma Knm

The plasma-labelled example is a separate ontology from the abstract Paper-27
Knm construction. Its built-in positional mapping supports ordered reduced
prefixes with `L=1..8` and one explicit refined hierarchy with `L=16`.
Unsupported implicit layer counts fail before matrix or frequency construction.
Custom `layer_names` are display aliases and do not remap the coupling indices.
These labels and frequencies remain hand-selected model inputs, not identified
reactor signals.

::: scpn_control.phase.plasma_knm.build_knm_plasma

::: scpn_control.phase.plasma_knm.build_knm_plasma_from_config

::: scpn_control.phase.plasma_knm.plasma_omega

### WebSocket Stream

`PhaseStreamServer` binds to loopback by default and requires authenticated
clients by default.  Operators must supply an API key or explicitly disable
client authentication for local development.  Non-loopback binds require an API
key, command frames are capped by `max_payload_bytes`, accepted commands are
rate-limited with token buckets per connection and per network peer, and
production remote exposure should enable TLS with `require_tls=True`.
Authentication, origin, payload, rate-limit, capacity, and command-authority
rejections emit structured security audit log events without logging tokens.
Query-string token authentication and plaintext
non-loopback binds are disabled by default and require explicit operator
opt-ins for constrained development or isolated lab environments.  Browser
clients that send an `Origin` header are rejected unless the origin is
allowlisted, and deployments may restrict command authority with
`allowed_actions`.
`websocket_runtime_evidence()` emits a tamper-evident network-runtime artifact that
binds WebSocket configuration, authenticated sessions, accepted commands,
successful broadcasts, audit counters, TLS enforcement, payload caps, and
backpressure state without storing API-key material. Qualified network-runtime
admission requires client authentication, configured TLS enforcement, observed
commands, observed broadcasts, no query-token authentication, no insecure remote
binding, and zero backpressure disconnects. It does not admit the streamed
oscillator model for reactor or facility control.

::: scpn_control.phase.ws_phase_stream.PhaseStreamServer

::: scpn_control.phase.ws_phase_stream.WebSocketRuntimeEvidence

::: scpn_control.phase.ws_phase_stream.websocket_runtime_evidence

::: scpn_control.phase.ws_phase_stream.assert_websocket_runtime_claim_admissible

::: scpn_control.phase.ws_phase_stream.save_websocket_runtime_evidence

::: scpn_control.phase.ws_phase_stream.load_websocket_runtime_evidence

---

## Control — Controllers

### H-infinity (normalized continuous-time DGKF)

::: scpn_control.control.h_infinity_controller.HInfinityController

::: scpn_control.control.h_infinity_controller.get_radial_robust_controller

### Model Predictive Control

::: scpn_control.control.fusion_neural_mpc.NeuralSurrogate

::: scpn_control.control.fusion_neural_mpc.ModelPredictiveController

### Optimal Control

::: scpn_control.control.fusion_optimal_control.OptimalController

### Digital Twin

`run_digital_twin()` now supports persistent sensor calibration bias and drift
in addition to dropout and white-noise corruption, and it can now stress the
command path with deterministic actuator bias, drift, first-order lag, and
rate limiting. The returned summary exposes both commanded and applied actions
plus actuator-lag telemetry so replay tests can see what the plant actually
received. Density and effective-charge knobs are explicit model-update
parameters.

`digital_twin_online_update` adds fail-closed TRANSP/TSC simulator artifact
metadata validation and deterministic Bayesian optimisation over bounded twin
parameters. The shipped benchmark is synthetic online-update evidence only;
external simulator replay claims require validated artifact metadata and the
strict digital-twin reference gate.
`digital_twin_update_evidence()` and
`assert_digital_twin_update_claim_admissible()` bind a bounded Bayesian update
to TRANSP and TSC simulator metadata digests, observation and prior digests,
result digest, baseline-improvement evidence, and an optional
safety-critical controller proof artifact digest. Admission also revalidates
source binding, finite non-negative loss history, minimum-loss consistency,
best-parameter bounds, strict integer campaign settings, and simulator unit
coverage for every observation target.

::: scpn_control.control.tokamak_digital_twin.run_digital_twin

::: scpn_control.control.digital_twin_online_update.validate_external_simulator_artifact

::: scpn_control.control.digital_twin_online_update.bayesian_update_digital_twin

::: scpn_control.control.digital_twin_online_update.DigitalTwinUpdateEvidence

::: scpn_control.control.digital_twin_online_update.digital_twin_update_evidence

::: scpn_control.control.digital_twin_online_update.assert_digital_twin_update_claim_admissible

::: scpn_control.control.digital_twin_online_update.synthetic_online_update_benchmark

### Controller Safety-Case Evidence

`control.safety_case` defines the bounded safety-case workflow contract that
links a passing safety-critical controller proof manifest, audited JAX
differentiable-transport evidence, and TRANSP/TSC-backed bounded digital-twin
online-update evidence. The bundle is tamper-evident and fails closed unless
all evidence items bind to the same canonical controller artifact SHA-256. This
is a repository safety-package admission boundary, not an external
certification claim. `save_controller_safety_case_evidence()` persists the
bundle with a manifest integrity digest, and
`load_controller_safety_case_evidence()` rejects schema drift, malformed fields,
or edited evidence payloads before replay admission.
`evaluate_controller_safety_case_readiness()` separates linked bounded evidence
from promotion readiness: external physics validation, target-hardware timing
evidence, qualified HIL replay evidence, qualified CODAC/EPICS runtime
evidence, qualified WebSocket runtime evidence, qualified HDL export evidence,
and independent safety-review digests are necessary for evidence completeness.
They do not make the package admissible for promotion by themselves.
`ReadinessArtifactEvidence` and
`evaluate_controller_safety_case_readiness_from_artifacts()` check the supplied
files: each required readiness input must be a typed artifact with a
known kind, SHA-256 digest, safe relative artifact URI, producer, and generation
timestamp. The evaluator also requires
an explicit `artifact_root`: each URI must resolve below that root and match the
declared bytes. `target_hardware_timing` artifacts must additionally pass the
schema-versioned E2E latency evidence validator with nonplaceholder declared
target labels and the configured p95 limit. That reader checks declarations,
not hardware origin or operator qualification. `hil_replay_evidence` artifacts use schema v2.
Caller-supplied replay metrics, target identifiers, and hashes cannot qualify
target hardware. The HIL loader refuses deployment claims without independently
verified hardware provenance, so local replay blocks promotion readiness.
`codac_runtime_evidence` artifacts use schema v3 and are local-only when built
from caller-supplied timing and interlock counts. The admission loader rejects
facility claims without independently verified runtime and hardware origin,
including re-sealed artifacts with a qualified status. CODAC therefore blocks
readiness until an independent evidence source is integrated.
`websocket_runtime_evidence` artifacts must pass the schema-versioned WebSocket
runtime admission loader with authenticated command evidence, TLS enforcement,
token-bucket and payload-cap configuration, successful broadcast counters, and
zero backpressure disconnects.
`hdl_export_evidence` artifacts must pass the schema-versioned FPGA export
admission loader with controller-artifact binding, generated project file
digests, synthesis-report digest binding, and non-negative timing slack.
External physics validation and independent safety review currently have no
signed, distinct-identity attestation verifier. Their file hashes establish
custody only, so `promotion_admissible` remains false even if every artifact
file is present. `assert_controller_safety_case_readiness_admissible()` refuses
promotion until those contracts are implemented and independently verified.
`save_controller_safety_case_readiness()` and
`load_controller_safety_case_readiness()` persist that readiness decision with
the same schema-versioned integrity-digest semantics as the safety-case bundle.

::: scpn_control.control.safety_case.ControllerSafetyCaseEvidence

::: scpn_control.control.safety_case.SafetyCaseReadinessEvidence

::: scpn_control.control.safety_case.ReadinessArtifactEvidence

::: scpn_control.control.safety_case.controller_safety_case_evidence

::: scpn_control.control.safety_case.assert_controller_safety_case_admissible

::: scpn_control.control.safety_case.save_controller_safety_case_evidence

::: scpn_control.control.safety_case.load_controller_safety_case_evidence

::: scpn_control.control.safety_case.evaluate_controller_safety_case_readiness

::: scpn_control.control.safety_case.evaluate_controller_safety_case_readiness_from_artifacts

::: scpn_control.control.safety_case.assert_controller_safety_case_readiness_admissible

::: scpn_control.control.safety_case.save_controller_safety_case_readiness

::: scpn_control.control.safety_case.load_controller_safety_case_readiness

### Flight Simulator

::: scpn_control.control.tokamak_flight_sim.IsoFluxController

::: scpn_control.control.tokamak_flight_sim.run_flight_sim

### Free-Boundary Tracking

Experimental closed-loop free-boundary tracking that keeps the full
`FusionKernel` in the loop and re-identifies the local coil-response map from
repeated solves. The configuration must declare at least one explicit
free-boundary flux, X-point, or divertor target; the repository's generic
`iter_config.json` is not a tracking configuration. Safe-current fallback
targets can be supplied through the
`free_boundary_tracking.fallback_currents` config block when supervisor
rejection should ramp the coils toward a predefined safe state. Persistent
objective residuals can also be accumulated with the config-driven
`free_boundary_tracking.observer_gain` and `observer_max_abs` settings. When
free-boundary objective tolerances are configured, the controller also uses
them directly in its correction and accept/reject logic so tighter X-point or
divertor targets take precedence over looser shape goals, and it refuses trial
steps that would push an already-satisfied objective back outside tolerance. If
the identified coil-response map loses authority entirely, the controller marks
that degraded state explicitly and drops into the safe-state recovery path
instead of silently accepting a zero-action step. Residuals already inside the
configured tolerances are also treated as deadband, so the controller stops
chattering the coils once the protected objectives are met. Coil allocation is
also headroom-aware, so the regularized solve prefers actuators that still have
current authority instead of leaning equally on a nearly saturated coil.
Deterministic objective-space sensor bias and per-step drift can be injected
through `free_boundary_tracking.measurement_bias` and
`measurement_drift_per_step`, and known calibration corrections can be applied
with `measurement_correction_bias` and `measurement_correction_drift_per_step`.
The run summary reports both measured and hidden true objective errors so
calibration faults cannot masquerade as control success in acceptance tests.
Response identification computes all coil columns privately and publishes the
matrix only after its perturbation solves and return-to-baseline solve report
convergence and the candidate matrix passes SVD diagnostics. A raised solve,
diagnostic failure, or malformed/nonfinite observation restores the
controller's original coil currents and actuator state without publishing a
partial matrix; the exception stops that tracking attempt. This protects the
controller state at this boundary. Initial, step, trial, and recovery solves
must also report convergence before their observations or actions are accepted.
If a trial raises or reports failure, the controller restores baseline coil
currents and actuator state and aborts the shot without recording that step.
This does not certify inner equilibrium
convergence, an immutable accepted kernel state, or a physical safety action;
those require the separate solver-status and facility evidence contracts.

```python
from scpn_control.control.free_boundary_tracking import run_free_boundary_tracking

summary = run_free_boundary_tracking(
    "reviewed_free_boundary_config.json",
    shot_steps=5,
    gain=0.8,
    verbose=False,
    coil_slew_limits=2.5e5,
    supervisor_limits={"x_point_position": 0.15, "max_abs_actuator_lag": 1.0e5},
    hold_steps_after_reject=2,
)

print(summary["shape_rms"], summary["objective_converged"], summary["supervisor_intervention_count"])
```

::: scpn_control.control.free_boundary_tracking.FreeBoundaryTrackingController

::: scpn_control.control.free_boundary_tracking.run_free_boundary_tracking

The response-identification owner constructs a complete candidate matrix from
bounded current perturbations. Failed columns are not published; the caller
retains responsibility for restoring the plant state after the callbacks.

::: scpn_control.control.free_boundary_response_identification

### Free-Boundary Tracking Limit Resolvers

Pure objective-tolerance, supervisor-limit, coil-slew, and scalar config
resolvers for free-boundary tracking live in a dedicated leaf. The controller
keeps thin wrappers; claim evidence remains a separate module.
`hold_steps_after_reject` accepts only a non-negative integer; fractional,
boolean and text values are rejected instead of being converted to a step count.

::: scpn_control.control.free_boundary_tracking_limits.resolve_objective_tolerances

::: scpn_control.control.free_boundary_tracking_limits.resolve_supervisor_limits

::: scpn_control.control.free_boundary_tracking_limits.resolve_coil_slew_limits

### Free-Boundary Tracking Observation Vectors

Objective-block topology, target-vector construction from a coil set,
measurement-offset resolution, and control-objective weighting live in a
dedicated leaf. The controller keeps thin wrappers; latency-state and kernel
observation remain on the owner. A tracking shot rejects nonfinite true,
measured, delayed or effective observations before accepting a step. If a
later observation stage fails, it restores controller-owned latency and EKF
state. This does not certify the upstream equilibrium state.

::: scpn_control.control.free_boundary_tracking_observation.ObjectiveBlock

::: scpn_control.control.free_boundary_tracking_observation.require_finite_observation

::: scpn_control.control.free_boundary_tracking_observation.build_target_vector

::: scpn_control.control.free_boundary_tracking_observation.resolve_measurement_vector

::: scpn_control.control.free_boundary_tracking_observation.build_control_objective_weights

### Free-Boundary Tracking Control Law

Response-matrix SVD diagnostics, objective-block activation masks, coil
headroom penalties, and Tikhonov-regularised coil correction live in a
dedicated leaf. The controller keeps thin wrappers; kernel-coupled response
identification and actuator application remain on the owner. The correction
law refuses nonfinite inputs and derived arithmetic before returning a coil
command. This is a simulated control boundary, not physical action admission.

::: scpn_control.control.free_boundary_tracking_control_law.ResponseDiagnostics

::: scpn_control.control.free_boundary_tracking_control_law.compute_response_diagnostics

::: scpn_control.control.free_boundary_tracking_control_law.build_control_activation_mask

::: scpn_control.control.free_boundary_tracking_control_law.build_coil_penalties

::: scpn_control.control.free_boundary_tracking_control_law.compute_coil_correction

### Free-Boundary Tracking Objective Metrics

Shape, X-point, divertor, weighted-control and convergence metrics are
computed in a dedicated leaf and exposed through the controller's existing
`evaluate_objectives()` method. RMS and norms use scaled arithmetic for large
finite errors; an unrepresentable metric raises an error before a tracking
decision is returned.

::: scpn_control.control.free_boundary_tracking_metrics.evaluate_objective_metrics

### Free-Boundary Tracking Claim Evidence

::: scpn_control.control.free_boundary_tracking_claims.free_boundary_tracking_claim_evidence

::: scpn_control.control.free_boundary_tracking_claims.assert_free_boundary_tracking_facility_claim_admissible

::: scpn_control.control.free_boundary_tracking_claims.save_free_boundary_tracking_claim_evidence

### Disruption Predictor

`predict_disruption_risk_safe()` still returns a bounded scalar risk, but its
metadata now includes deterministic sigma-point uncertainty summaries
(`risk_p05`, `risk_p50`, `risk_p95`, `risk_std`, `risk_interval`) for both
fallback and checkpoint inference paths. `DisruptionTransformer.predict()`
returns the same bounded scalar-risk shape expected by `evaluate_predictor()`,
so trained transformer instances can be evaluated directly without wrapper
adapters.

Checkpoint integrity, train/load, physics proxies, fault campaigns, and claim
boundaries live in dedicated leaves re-exported by the owner module.

::: scpn_control.control.disruption_predictor.DisruptionTransformer

::: scpn_control.control.disruption_predictor.predict_disruption_risk

::: scpn_control.control.disruption_predictor.predict_disruption_risk_safe

::: scpn_control.control.disruption_checkpoint_integrity.DisruptionCheckpointIntegrityError

::: scpn_control.control.disruption_checkpoint_integrity.verify_checkpoint_integrity

::: scpn_control.control.disruption_checkpoint_integrity.verified_checkpoint_snapshot

::: scpn_control.control.disruption_checkpoint.train_predictor

::: scpn_control.control.disruption_checkpoint.load_or_train_predictor

::: scpn_control.control.disruption_physics_proxies.simulate_tearing_mode

::: scpn_control.control.disruption_physics_proxies.build_disruption_feature_vector

::: scpn_control.control.disruption_physics_proxies.predict_disruption_risk

::: scpn_control.control.disruption_physics_proxies.disruption_warning_time

::: scpn_control.control.disruption_fault_campaigns.apply_bit_flip_fault

::: scpn_control.control.disruption_fault_campaigns.run_fault_noise_campaign

::: scpn_control.control.disruption_fault_campaigns.HybridAnomalyDetector

::: scpn_control.control.disruption_fault_campaigns.run_anomaly_alarm_campaign

::: scpn_control.control.disruption_risk_claims.DisruptionRiskClaimBoundary

::: scpn_control.control.disruption_risk_claims.disruption_risk_claim_boundary

### Disruption Contracts

`disruption_contracts` preserves the public imports for the synthetic episode
physics, episode runtime and labelled-shot replay. Replay requires at least one
sample after its predictor window; this matches the shared risk-series scorer.
Its disruption label must be a scalar Boolean, and its scalar integer index must
be within the shot for a disruptive label or `-1` for a safe label. The replay
keeps the historical RL-agent parameter but uses a deterministic SPI decision.
The synthetic signal and proxy functions reject invalid numeric inputs and
nonfinite derived results. The mitigation outputs remain bounded software-model
evidence.

::: scpn_control.control.disruption_contracts.run_disruption_episode

::: scpn_control.control.disruption_contracts.predict_disruption_risk

Episode physics proxies, synthetic episode orchestration and labelled-shot
replay have separate implementation owners beneath this facade.

::: scpn_control.control._disruption_episode_physics

::: scpn_control.control._disruption_episode_runtime

::: scpn_control.control._disruption_shot_replay

### Disruption ROC

`scpn_control.control.disruption_roc` scores a fixed-weight risk series over a
shot, sweeps alarm thresholds, and reports bounded internal ROC/AUC plus
warning-time recall. The scoring core is a bounded model (the n=3 toroidal
amplitude approximates n=2), so its metrics stay internal and admission-blocked.
Scoring requires equal-length finite one-dimensional channels, a full sample
window and at least one sample after it; the public import delegates to a
dedicated scoring module. Shot metrics
require finite increasing times, risks in `[0, 1]`, and both safe and disruptive
shots for ROC/AUC. Invalid cohorts or thresholds raise `ValueError` instead of
producing an AUC. This is a software input contract, not predictive validation.

::: scpn_control.control.disruption_roc.score_risk_series

The per-window scorer used by replay and ROC analysis is documented at its
implementation location as well as through the public facade.

::: scpn_control.control._disruption_risk_series

::: scpn_control.control.disruption_roc.ShotEvaluation

::: scpn_control.control.disruption_roc.first_alarm_index

::: scpn_control.control.disruption_roc.confusion_at_threshold

::: scpn_control.control.disruption_roc.roc_curve

::: scpn_control.control.disruption_roc.roc_auc_from_curve

::: scpn_control.control.disruption_roc.warning_time_recall

::: scpn_control.control.disruption_roc.disruption_metrics

### SPI Mitigation

::: scpn_control.control.spi_mitigation.ShatteredPelletInjection

::: scpn_control.control.spi_mitigation.run_spi_mitigation

### Fusion Control Room

::: scpn_control.control.fusion_control_room.run_control_room

::: scpn_control.control.fusion_control_room.TokamakPhysicsEngine

### Gymnasium Environment

`TokamakEnv.step` rejects derived nonfinite model values before committing
actuator, plasma, reward or episode state. A failed noisy observation also
preserves the random stream. `reset` preserves the preceding episode when a
new observation cannot be formed. These are reduced-order software-model
contracts, not plasma-plant safety guarantees.

::: scpn_control.control.gym_tokamak_env.TokamakEnv

### Analytic Solver

::: scpn_control.control.analytic_solver.AnalyticEquilibriumSolver

### Bio-Holonomic Controller

::: scpn_control.control.bio_holonomic_controller.BioHolonomicController

### Digital Twin Ingest

Synthetic planning accepts one canonical machine and strictly ordered telemetry
timestamps. Repeated plans replay from the same accepted buffer and seed; wall
latency is measured separately. These outputs are local emulation evidence.
If no plan is produced, mean risk and p95 latency are absent (`None`).

::: scpn_control.control.digital_twin_telemetry.TelemetryPacket

::: scpn_control.control.digital_twin_telemetry.generate_emulated_stream

::: scpn_control.control.digital_twin_ingest.RealtimeTwinHook

::: scpn_control.control.digital_twin_ingest.run_realtime_twin_session

### Director Interface

::: scpn_control.control.director_interface.DirectorInterface

### Fueling Mode Controller

The public `IcePelletFuelingController.step` rejects nonfinite or negative
density, nonpositive or nonfinite time intervals, invalid or repeated step
indices, and overflowing PI arithmetic before changing its integrator. The
synthetic `simulate_iter_density_control` path accepts an exact integer count of at
least eight; it does not silently truncate fractional counts. Its results are
normalised software-model evidence, not a measured ITER fueling result.

::: scpn_control.control.fueling_mode.IcePelletFuelingController

### Halo RE Physics

The public module delegates halo circuits, runaway-electron dynamics, ensemble
simulation, and claim evidence to focused implementation modules. Short
simulations end at the requested duration, including a shortened final step.
Validated mitigation admission requires an independently verified reference
comparison; reference metadata alone does not grant it.

::: scpn_control.control.halo_re_physics.HaloCurrentModel

::: scpn_control.control.halo_re_physics.DisruptionMitigationClaimEvidence

::: scpn_control.control.halo_re_physics.disruption_mitigation_claim_evidence

::: scpn_control.control.halo_re_physics.assert_disruption_mitigation_claim_admissible

::: scpn_control.control.halo_re_physics.save_disruption_mitigation_claim_evidence

The halo circuit and runaway-electron models feed the bounded ensemble runner.
Their claim-evidence owner retains the matched-reference admission checks.

::: scpn_control.control._halo_current_model

::: scpn_control.control._runaway_electron_model

::: scpn_control.control._disruption_ensemble

::: scpn_control.control._disruption_claims

### HIL Test Harness

The public `hil_harness` facade is backed by separate I/O, timed-loop, FPGA,
replay-evidence, benchmark, and demo modules. Its schema-v2 replay builder
creates local evidence; setting `deployment_claim_allowed` cannot grant a
target-hardware claim without independent provenance. Version 1 artifacts are
rejected and must be regenerated as local evidence.

::: scpn_control.control.hil_harness.HILControlLoop

::: scpn_control.control.hil_harness.HILBenchmarkResult

::: scpn_control.control.hil_harness.hil_replay_evidence

::: scpn_control.control.hil_harness.assert_hil_replay_evidence_admissible

::: scpn_control.control.hil_harness.save_hil_replay_evidence

::: scpn_control.control.hil_harness.load_hil_replay_evidence

The implementation references distinguish simulated ADC/DAC and FPGA register
interfaces from the timed software loop, evidence checks and benchmark runners.
The demo runner simulates the register-mapped controller; its execution alone
does not establish target-hardware deployment.

::: scpn_control.control.hil_io

::: scpn_control.control.hil_fpga

::: scpn_control.control.hil_loop

::: scpn_control.control.hil_evidence_contracts

::: scpn_control.control.hil_evidence

::: scpn_control.control.hil_benchmark

::: scpn_control.control.hil_demo

### JAX Traceable Runtime

Requires `pip install "scpn-control[jax]"`.

::: scpn_control.control.jax_traceable_runtime.TraceableRuntimeSpec

### LIF+NEF SNN Controller

::: scpn_control.control.nengo_snn_wrapper.NengoSNNController

### Neuro-Cybernetic Controller

::: scpn_control.control.neuro_cybernetic_controller.NeuroCyberneticController

### Synthetic hybrid-control example

The historical TORAX-named API executes local illustrative update equations.
It neither invokes an external TORAX solver nor measures wall-clock latency.
Serialise its result with `dataclasses.asdict`; retain the provenance fields
alongside the legacy numeric fields when exporting JSON.

| Legacy metric | Actual interpretation |
| --- | --- |
| `torax_parity_pct` | Mean per-episode synthetic beta-trajectory agreement: clipped `100 * (1 - RMSE / RMS(baseline beta))`. |
| `p95_loop_latency_ms` | P95 of the analytical proxy `0.24 + 0.12 * clip(disturbance, 0, 1) + 0.08 * abs(snn_corr)`, in nominal milliseconds. No hardware calibration is supplied. |
| `passes_thresholds` | Synthetic regression checks only; no physical or performance qualification. |

Detached results carry schema `scpn-control.synthetic-hybrid-campaign.v1`,
`torax_parity_kind`, `latency_kind`, `threshold_scope`, and explicit false
`external_torax_executed`, `wall_clock_measured`, `production_claim_allowed`.
Existing metric names and numerical calculations remain compatible. These
annotations describe provenance; they do not authenticate an arbitrary payload.

Both branches start each episode from the same sampled state and receive the
same deterministic disturbance envelope. Their stochastic state updates consume
successive draws from one RNG, so they do not use paired identical noise. Beta
is assigned numerically to `R_axis_m` for this example; it is not a physical
beta-to-position calibration. Plasma state and risk history reset each episode;
the controller and RNG persist, and controller step indices remain campaign-wide.
A high-risk streak can end an episode early, so episodes need not have equal
numbers of executed steps. External parity and measured latency require a
separate executable solver contract and actual measurement evidence.

::: scpn_control.control.torax_hybrid_loop.ToraxHybridCampaignResult

::: scpn_control.control.torax_hybrid_loop.run_nstxu_torax_hybrid_campaign

### Advanced SOC Learning

::: scpn_control.control.advanced_soc_fusion_learning.run_advanced_learning_sim

### NMPC Controller (v0.16.0)

`NonlinearMPC` validates the NMPC quadratic program contract before
optimization: `Q`, `R`, and optional terminal `P` must be finite symmetric
positive-definite matrices with tokamak state/input dimensions; state, input,
and slew bounds must be finite and ordered; and plant-model evaluations must
return finite state vectors. Invalid math contracts fail closed instead of
propagating undefined SQP or PGD iterates.
The public `compute_cost()` evaluator includes the finite-horizon terminal
penalty, using configured `P` when supplied and the controller's conservative
fallback terminal weight otherwise.
Production plant models may provide an analytic `linearization_model(x, u)`
contract returning finite `(6, 6)` state and `(6, 3)` input Jacobians. The
controller validates those matrices before use and records
`last_linearization_source == "analytic"`. If no analytic provider is supplied,
the controller can use `linearization_backend="jax"` for JAX-traceable plant
models and records `last_linearization_source == "jax"`; otherwise it falls
back to bounded central finite differences and records
`last_linearization_source == "finite_difference"`. Quadratic weights use a
strict symmetry gate before positive-definite projection so near-zero
off-diagonal asymmetry cannot pass as a valid cost matrix.
DARE-derived terminal matrices are accepted only when finite, symmetric, and
positive definite; invalid solver output falls back to the conservative terminal
weight. The terminal weight alone does not prove recursive feasibility for a
constrained nonlinear plant; the `10 Q` fallback is a cost heuristic.
Explicit terminal state sets are configured with paired `terminal_x_min` and
`terminal_x_max` vectors. These bounds must lie inside the configured physics
state envelope and currently require `qp_backend="scipy"`, `qp_backend="osqp"`,
`qp_backend="casadi"`, or `qp_backend="acados"` so the coupled terminal-state
inequality is enforced inside the constrained QP solve rather than checked
after the fact.  `casadi` is a repository-local optional dependency path.
The `acados` backend is a full optional OCP interface: deployments may inject a
pre-built acados OCP/solver factory, or provide `symbolic_dynamics_model(ca, x,
u)` so the controller builds a discrete augmented-state acados model. The
augmented state stores the previous actuator vector, making `|Δu| <= du_max`
a native acados path constraint instead of a post-solve clamp. The default
builder configures SQP, partial-condensing HPIPM, exact Hessian mode, linear
least-squares stage/terminal costs, state/input bounds, terminal state bounds,
warm starts, fail-closed solver-status handling, and a runtime plant-consistency
gate. The returned acados state trajectory must start from the commanded state,
remain inside configured state bounds, satisfy any terminal admissible set, and
match `plant_model` transitions within `acados_dynamics_residual_tol` before the
first actuator command is admitted.
The previous input supplied to `step()` must already satisfy actuator bounds so
the slew-rate projection cannot propagate an unsafe actuator state.
`rti_residual_tol` must be positive and finite. Both `step()` and `step_rti()`
return detached control arrays so caller changes cannot alter the controller's
warm-start state. If a plant or solver callback raises during either tick, the
previous control and state trajectories are restored before the exception is
propagated; a failed tick does not become the next warm start.
The accepted `horizon=1` case is handled as a valid one-step receding-horizon
controller and warm-starts from the bounded previous input.
Each QP solve records `last_qp_iterations` and `last_qp_converged`, making
projection-tolerance convergence distinguishable from iteration-budget
exhaustion.
The projected-gradient QP iteration budget is configured by `qp_max_iter`
instead of being an unobservable hard-coded loop bound.
Linearization perturbations are clipped to the configured state/input domain:
interior points use central differences, while boundary points use one-sided
finite differences.

::: scpn_control.control.nmpc_controller.NonlinearMPC

The implementation references expose configuration and result records,
linearisation, Hessian audits, QP construction and dispatch, control ticks and
the optional acados lifecycle. Optional solver requirements and admission
checks are the same as for `NonlinearMPC` above.

::: scpn_control.control.nmpc_types

::: scpn_control.control.nmpc_linearization

::: scpn_control.control.nmpc_hessian

::: scpn_control.control.nmpc_qp_solver

::: scpn_control.control.nmpc_qp_core

::: scpn_control.control.nmpc_runtime

::: scpn_control.control.nmpc_acados

### NMPC Transport-Model Tuning (v0.16.0)

The transport-model tuning entry points live in their own
`scpn_control.control.nmpc_transport_tuning` module: fitting the transport model
the controller tracks against is a distinct responsibility from receding-horizon
tracking, so it is separated from `nmpc_controller`. The entry-point signatures
and fail-closed semantics are unchanged.
`tune_transport_coefficients_for_tracking()` connects NMPC controller tuning to
the differentiable transport facade. It updates four-channel transport
coefficients from the JAX gradient of the transport tracking loss, applies
non-negative coefficient bounds and fractional update caps, and fails closed
when JAX gradients are unavailable. By default, coefficient tuning also runs the
differentiable-transport finite-difference gradient audit before admission and
stores the audit result beside the validated transport campaign metadata for
backend, dtype, radial grid, boundary conditions, closure provenance, and
gradient tolerance.
`tune_neural_transport_closure_for_tracking()` initialises the same tuning path
from a bounded neural transport closure, preserving the differentiable facade's
four-channel coefficient contract, the explicit JAX-gradient requirement, and
the default gradient-audit admission gate.
`tune_transport_sources_for_tracking()` applies the audited JAX gradient path to
additive heating, fuelling, and impurity-source schedules. Source lower and
upper bounds are explicit because replay studies may include physically valid
sink terms, and every accepted update carries campaign metadata plus the
gradient-audit result.
`tune_transport_source_rollout_for_tracking()` extends that admission boundary
from a single transport step to a complete `(n_steps, 4, n_rho)` source
schedule. It uses JAX for the multi-step rollout gradient, requires a sampled
NumPy finite-difference audit by default, clips per-entry source updates when
configured, and records bounded campaign metadata before the schedule can enter
NMPC tuning. The default audit-failure mode is fail-closed. The explicit
`gradient_audit_failure_mode="warn"` mode exists only for advisory, non-control
analysis and preserves the failed audit evidence in the returned result.
All three gradient-update paths require a finite tracking loss and reject
unrepresentable update arithmetic before applying configured bounds. Their
`step_norm` uses scaled arithmetic, so a large but representable control update
does not become an infinite diagnostic. Rollout audit sample coordinates must
be integers; fractional values are rejected instead of being truncated to a
different location. These checks also apply when the gradient audit is
explicitly disabled for an offline study.

::: scpn_control.control.nmpc_transport_tuning.TransportCoefficientTuningResult

::: scpn_control.control.nmpc_transport_tuning.TransportSourceScheduleTuningResult

::: scpn_control.control.nmpc_transport_tuning.TransportSourceRolloutGradientAudit

::: scpn_control.control.nmpc_transport_tuning.TransportSourceRolloutTuningResult

::: scpn_control.control.nmpc_transport_tuning.tune_transport_coefficients_for_tracking

::: scpn_control.control.nmpc_transport_tuning.tune_transport_sources_for_tracking

::: scpn_control.control.nmpc_transport_tuning.tune_transport_source_rollout_for_tracking

::: scpn_control.control.nmpc_transport_tuning.tune_neural_transport_closure_for_tracking

The result records, bounded-update checks and sampled rollout-gradient audit
used by these tuning entry points live in the following contract module.

::: scpn_control.control.nmpc_transport_contracts

### Riccati State Feedback with Static Mu Analysis

::: scpn_control.control.static_mu_analysis.RiccatiStateFeedbackController

::: scpn_control.control.static_mu_analysis.StaticMuAnalysisResult

::: scpn_control.control.static_mu_analysis.StructuredUncertainty

::: scpn_control.control.static_mu_analysis.UncertaintyBlock

::: scpn_control.control.static_mu_analysis.compute_static_mu_upper_bound

::: scpn_control.control.static_mu_analysis.design_riccati_state_feedback_with_static_mu_analysis

::: scpn_control.control.static_mu_analysis.StaticMuAnalysisClaimEvidence

::: scpn_control.control.static_mu_analysis.static_mu_analysis_claim_evidence

::: scpn_control.control.static_mu_analysis.assert_static_mu_analysis_validated_claim_admissible

::: scpn_control.control.static_mu_analysis.save_static_mu_analysis_claim_evidence

::: scpn_control.control.static_mu_analysis.load_static_mu_analysis_claim_evidence

The implementation modules separate uncertainty-block structure, static
D-scaling bounds, Riccati feedback and claim-evidence persistence. These are
the same static DC analysis owners used by the facade.

::: scpn_control.control._static_mu_structure

::: scpn_control.control._static_mu_bounds

::: scpn_control.control._static_mu_riccati

::: scpn_control.control._static_mu_claims

The historical `scpn_control.control.mu_synthesis` names are deprecated
compatibility aliases scheduled for removal in version 0.25.0. They do not
perform frequency-dependent D-K synthesis.

::: scpn_control.control.mu_synthesis

### Real-Time EFIT (v0.16.0)

`RealtimeEFIT` reports `iteration_status` as `geometric_solve`,
`picard_converged`, or `picard_limit`; only Picard results report
`final_relative_change`. Directly constructed results remain `unreported`.
The public module preserves historical imports while the data contracts,
claim evidence, diagnostic response, grid solver, and inverse runtime live in
cohesive `realtime_efit_*` modules.
`find_xpoint` estimates an interior flux saddle from local grid derivatives;
flat or O-point fields return `None`. This local estimate does not certify a
separatrix or facility equilibrium.
The inverse rejects nonfinite observations, invalid iteration and tolerance
settings, and unrepresentable diagnostic weights before least squares.

EFIT-lite claim evidence schema 2 records this termination status. Caller
reference labels, arrays, and tolerances remain unverified metadata, even
when they numerically match the result. The builder, assertion, and persistence
path refuse facility admission until independent source bytes, provenance,
and a fixed comparison convention are bound and verified.

::: scpn_control.control.realtime_efit.RealtimeEFIT

::: scpn_control.control.realtime_efit.EFITLiteClaimEvidence

::: scpn_control.control.realtime_efit.efit_lite_claim_evidence

::: scpn_control.control.realtime_efit.assert_efit_lite_facility_claim_admissible

::: scpn_control.control.realtime_efit.save_efit_lite_claim_evidence

The implementation references cover reconstruction records, synthetic
diagnostic response, grid operators, local saddle estimation, inverse runtime
and claim evidence. Local topology estimates retain the limits described above.

::: scpn_control.control.realtime_efit_contracts

::: scpn_control.control.realtime_efit_diagnostics

::: scpn_control.control.realtime_efit_solver

::: scpn_control.control.realtime_efit_topology

::: scpn_control.control.realtime_efit_runtime

::: scpn_control.control.realtime_efit_claims

### Gain-Scheduled Controller (v0.16.0)

The controller rejects nonfinite or wrong-size diagnostics and publishes PID
and regime state only after a finite output is available. Gain interpolation
does not certify bumpless actuator output or facility operation. The historical
scenario imports remain available here; their waveform implementation lives in
`scpn_control.control.gain_scheduled_scenario` and accepts finite, strictly
increasing knots with linear interpolation only.

::: scpn_control.control.gain_scheduled_controller.GainScheduledController

Waveform validation, scenario schedules and the baseline scenario factory are
documented at their implementation location.

::: scpn_control.control.gain_scheduled_scenario

### Safe RL Controller (v0.16.0)

::: scpn_control.control.safe_rl_controller.LagrangianPPO

### Sliding-Mode Vertical (v0.16.0)

`SuperTwistingSMC` refuses nonfinite or overflowing derived controls before
changing its integral state. `VerticalStabilizer` requires finite geometry and
a representable model force coefficient. The historically named
`lyapunov_certificate` and `estimate_convergence_time` check idealized
sign-law expressions only; smoothing, sampling and actuator limits prevent
those values from serving as runtime convergence certificates. The emitted
correction is in caller-defined units and has no coil-voltage calibration.

::: scpn_control.control.sliding_mode_vertical.SuperTwistingSMC

::: scpn_control.control.sliding_mode_vertical.lyapunov_certificate

::: scpn_control.control.sliding_mode_vertical.estimate_convergence_time

::: scpn_control.control.sliding_mode_vertical.VerticalStabilizer

### Scenario Scheduler (v0.16.0)

Waveform interpolation requires finite, equal-length, strictly increasing
knots. A feedforward control step checks the schedule, state, cycle time and
three-value feedback trim before emitting a finite command. Offline trajectory
optimisation requires a finite positive horizon/timestep and projects heating
power and plasma current candidates to non-negative values; it does not claim
a globally optimal or facility-qualified trajectory.

::: scpn_control.control.scenario_scheduler.ScenarioOptimizer

### Closed-Loop Integrated Scenario Demo (v0.22.1)

`scpn_control.control.closed_loop_scenario` wires the reusable
`ScenarioSchedule` / `FeedforwardController` surface into
`IntegratedScenarioSimulator` for the `scpn-control demo --scenario combined`
path. The exported result carries controller commands, bounded auxiliary-power
application, and a replay coupling audit. It is a deterministic repository
wiring contract; measured-discharge validation remains gated by the physics
traceability registry.
The loop requires a positive integer `max_steps` and rejects nonfinite
controller power commands before applying actuator bounds or advancing the
plant. Finite commands continue to use the configured lower and upper bounds.

::: scpn_control.control.closed_loop_scenario.ClosedLoopScenarioStep

::: scpn_control.control.closed_loop_scenario.ClosedLoopScenarioResult

::: scpn_control.control.closed_loop_scenario.run_integrated_scenario_closed_loop

### Fault-Tolerant Control (v0.16.0)

::: scpn_control.control.fault_tolerant_control.ReconfigurableController

::: scpn_control.control.fault_injector.FaultType

::: scpn_control.control.fault_injector.FaultInjector

### RZIp Model (v0.16.0)

::: scpn_control.control.rzip_model.RZIPModel

::: scpn_control.control.rzip_model.RZIPController

::: scpn_control.control.rzip_model.RZIPCalibrationEvidence

::: scpn_control.control.rzip_model.rzip_calibration_evidence

::: scpn_control.control.rzip_model.assert_rzip_facility_claim_admissible

::: scpn_control.control.rzip_model.save_rzip_calibration_evidence

The calibration decoder checks the consistency and numerical domains of the
declared metrics. It does not authenticate the declared reference source.

::: scpn_control.control._rzip_calibration

### RWM Feedback (v0.16.0)

::: scpn_control.control.rwm_feedback.RWMFeedbackController

::: scpn_control.control.rwm_feedback.RWMClaimEvidence

::: scpn_control.control.rwm_feedback.rwm_claim_evidence

::: scpn_control.control.rwm_feedback.assert_rwm_facility_claim_admissible

::: scpn_control.control.rwm_feedback.save_rwm_claim_evidence

---

## Complete Module Index

This index covers Python source modules under `src/scpn_control/`, including
implementation owners whose public names are reexported by a facade. Domain
pages above describe primary entry points. Module references render signatures
and docstrings at their implementation locations; underscore-prefixed modules
remain internal implementation paths.

### Top-Level CLI

#### Cli

::: scpn_control.cli

#### CLI Reference Validators

::: scpn_control.cli_reference_validators

The facade retains registration order and the hidden compatibility command.
Each family owns the existing command options and source-validator contracts.

::: scpn_control.cli_reference_kinetic

::: scpn_control.cli_reference_equilibrium

::: scpn_control.cli_reference_transport

::: scpn_control.cli_reference_engineering

::: scpn_control.cli_reference_instabilities

::: scpn_control.cli_reference_tracking

::: scpn_control.cli_reference_static_mu

::: scpn_control.cli_reference_paths

Report destination conversion rejects NUL with a fixed usage finding before
filesystem calls. The GK cross-code and neural equilibrium commands share this
conversion and protect their selected inputs during report persistence.

#### CLI Evidence Validators

::: scpn_control.cli_evidence_validators

#### Shared Typing Aliases

::: scpn_control._typing

#### Shared NPZ Writer

::: scpn_control._npz

### Control Modules

#### Burn Controller

::: scpn_control.control.burn_controller

#### Codac Interface

`CODACInterface.run_cycle()` is a fail-closed software-adapter boundary. Every
configured external interlock PV and every hard-limit process signal must be
present, numeric, finite, nominal, and within its declared hard range before
the controller is invoked. External binary interlocks use `0.0` for clear and
any non-zero value for trip. A blocked cycle does not call the controller and
returns the complete fail-stationary zero-output packet. Controller outputs are
rejected if non-numeric or non-finite and otherwise clamped to the envelopes in
`_OUTPUT_CHANNELS`; generated analog EPICS records carry matching `DRVH` and
`DRVL` drive limits.

The controller-consumed `R_axis` and `Z_axis` PVs are required, finite and
within their declared channel ranges. Their `_m` aliases are accepted when
equal; conflicting names block the cycle. `pack_observation()` raises for an
invalid axis, while `run_cycle()` sends the stationary packet without calling
the controller. Each cycle uses one snapshot of the input PV mapping.

CODAC runtime evidence uses `scpn-control.codac-runtime-evidence.v3` and records
that output limits, EPICS drive limits, and the fail-closed interlock path are
active. Versions 1 and 2 are rejected and must be regenerated. These software
guards do not constitute an independent machine-protection interlock, hardware
commissioning, or facility qualification.

Runtime evidence validation and axis observation admission live in separate
modules; `codac_interface` keeps the established public API names.

::: scpn_control.control.codac_interface

::: scpn_control.control.codac_interface.CODACRuntimeEvidence

::: scpn_control.control.codac_interface.codac_runtime_evidence

::: scpn_control.control.codac_interface.assert_codac_runtime_claim_admissible

::: scpn_control.control.codac_interface.save_codac_runtime_evidence

::: scpn_control.control.codac_interface.load_codac_runtime_evidence

The observation owner validates the controller-consumed axis channels. The
evidence owner checks persisted runtime payloads independently of the live
adapter, retaining the software-only admission boundary described above.

::: scpn_control.control.codac_observation

::: scpn_control.control.codac_evidence

#### Controller Tuning

`tune_pid` uses real environment rollouts when Optuna is available. For the
public `TokamakEnv`, it tunes the heating channel against `T_target - T_axis`
while commanding zero change to the current channel. Scalar-error
environments retain their one-dimensional action contract. Without Optuna,
PID tuning raises `ImportError`; fixed fallback gains are not labelled as
optimised. `tune_hinf(plant)` uses the normalised DGKF plant matrices through
`HInfinityController` and returns only the feasible near-infimum attenuation
`gamma`. It refuses missing or non-normalised plant data. The former
`n_trials` argument and synthetic `bandwidth` output are removed; neither
represented a plant-derived optimisation result. The attenuation applies to
the unsaturated linear continuous-time model, not clipped runtime action or
facility performance.

::: scpn_control.control.controller_tuning

#### Density Controller

The public module is a stable facade over transport, PI control, claim evidence,
Kalman estimation, and pellet scheduling modules. Density claim evidence schema
version 2 records whether supplied Greenwald and inventory references agree.
Caller supplied reference values and provenance labels cannot independently
establish a facility witness, so `facility_density_claim_allowed` remains false
and the facility admission guard rejects promotion.

::: scpn_control.control.density_controller

::: scpn_control.control.density_controller.KalmanDensityEstimator

::: scpn_control.control.density_controller.DensityControlClaimEvidence

::: scpn_control.control.density_controller.density_control_claim_evidence

::: scpn_control.control.density_controller.assert_density_control_facility_claim_admissible

::: scpn_control.control.density_controller.save_density_control_claim_evidence

The implementation references separate the radial transport plant, PI actuator
commands, Kalman estimator, pellet scheduler and claim-evidence admission.

::: scpn_control.control._density_transport

::: scpn_control.control._density_control_runtime

::: scpn_control.control._density_estimator

::: scpn_control.control._density_fueling

::: scpn_control.control._density_claim_evidence

#### Detachment Controller

The public controller computes a finite software PI candidate and publishes its
state only after the command is valid. Multi-impurity stepping restores every
controller if a later species fails. The scalar has no established actuator
conversion or physical seeding-rate unit. `target_DOD` is retained for API
compatibility and is not part of the current PI law; facility use remains
unadmitted pending the detachment formulation and actuator contract.

::: scpn_control.control.detachment_controller

The stateful PI candidate and all-species rollback coordinator live in the
runtime module below, with the same actuator and admission limits.

::: scpn_control.control.detachment_control_runtime

#### Federated Disruption

::: scpn_control.control.federated_disruption

The implementation references separate client data and training, the eight-input
MLP, dataset-weighted FedAvg, server settings and rounds, and replayable state.
Privacy records report nominal clipping/noise arithmetic; the synthetic
benchmark does not independently qualify facility privacy or prediction.

::: scpn_control.control._federated_config

::: scpn_control.control._federated_clients

::: scpn_control.control._federated_model

::: scpn_control.control._federated_aggregation

::: scpn_control.control._federated_privacy

::: scpn_control.control._federated_state

::: scpn_control.control._federated_server

::: scpn_control.control._federated_benchmark

#### State Estimator

::: scpn_control.control.state_estimator

#### Volt Second Manager

The historical module reexports the flux-budget core, profile proxy, online
monitor, scenario analysis and claim-evidence API. The monitor publishes state
only after all derived values are finite. Caller-declared reference metadata,
including digest-shaped text and error metrics, remains `reference_metadata_unverified`;
the facility-claim assertion refuses it until independent source bytes and
comparison metrics are bound. These software checks do not qualify a pulse
design or central solenoid for facility operation.

::: scpn_control.control.volt_second_manager

The implementation references distinguish budget planning, pressure-profile
bootstrap proxies, online consumption and scenario accounting from claim
evidence and persistence.

::: scpn_control.control.volt_second_core

::: scpn_control.control.volt_second_profiles

::: scpn_control.control.volt_second_runtime

::: scpn_control.control.volt_second_claims

### Benchmark Record Custody

`scpn_control.benchmark_records` provides the public immutable-run contract used
by benchmark producers. It reserves a campaign before execution, retains legacy
fixed-name artifacts, seals complete or failed runs, and exposes a digest-bound
`latest` index without treating that index as evidence custody.

::: scpn_control.benchmark_records

The output lease reserves cooperating campaigns' destinations until finalisation
or explicit recovery. See [benchmark custody](benchmarks.md) for lifecycle and
interrupted-run handling.

::: scpn_control.benchmark_output_lease.BenchmarkOutputLease

Directory artefact inspection binds node types, empty directories and
case-sensitive POSIX relative-name ordering with an explicit digest algorithm.
The latest reader verifies stored payload bytes and the canonical manifest
binding; it does not authenticate a producer or establish scientific validity.

New stored names use the zero-based output declaration index, followed by the
source suffix for files (`.bin` if absent); directory names use the index alone.
Role labels remain separate manifest fields, so file `report` and directory
`report.json` cannot share a stored name. Resolve artefacts through their role
and explicit `immutable_path_in_run`; do not infer a filename from the role.
Schema and digest algorithms are unchanged, and legacy role-named records remain
readable without mutation. Missing outputs retain their declared index slots.

New predecessor archives use `legacy/<digest-algorithm>/<digest>/artifact`.
Reuse requires both the expected filesystem kind and matching content digest;
equal file/tree digests cannot alias across the named algorithms. Role labels
do not select archive names. Prior and failed outputs use zero-based declaration
indices without suffixes; `invocation.json` records each role and exact prior
path. This preserves case-distinct roles on Windows. Existing manifests retain
their explicit legacy paths unchanged; no historical archive is renamed. The
verified-latest reader checks sealed artifacts, not predecessor archives.

Family and campaign identifiers are native filename components and retain their
supplied case. Case-distinct family names do not provide independent carriers
on a case-insensitive filesystem; a later successful campaign can select the
same latest path, after which the reader refuses a different declared family.
Use one consistent spelling for each family. Native filename restrictions,
including Windows trailing-dot and reserved-name handling, still apply.

::: scpn_control.benchmark_artifacts

::: scpn_control.benchmark_record_integrity

The source-checkout runner executes an argument vector without a shell, from the
resolved repository root, and passes the reserved campaign through
`SCPN_BENCHMARK_CAMPAIGN_ID`. Each repeated `--artifact ROLE=PATH` describes one
output; relative paths use that repository root. The records root must resolve
inside the repository. Command options remain producer arguments after `--`.

`main(argv)` returns the native producer code, `127` for a launch failure or
`130` for an interrupt caught while waiting. A zero-exit command that does not
recreate every declared output returns `1`. Help and malformed wrapper arguments
raise argparse's `SystemExit` before reservation. Reservation and finalisation
errors propagate; an unresolved custody failure can retain a recovery lease.
See the [runner lifecycle](benchmarks.md#recorded-command-lifecycle) for native
process boundaries and immutable failure records.

Measurement labels inferred from command spelling can be overridden by
`--measurement-json`. They do not measure execution, validate sample counts or
grant scientific or production admission. The source HEAD field does not bind
uncommitted file bytes.

::: tools.run_recorded_benchmark

### Core Support and Physics Modules

#### Rust Compatibility

`RustPIDController` is available through the historical `_rust_compat` import.
When the optional native `PyPIDController` binding is absent, its Python
fallback uses the same finite-input law as `control.pid_controller.PIDController`.
The fallback radial and vertical presets use the native Rust gains (2.0, 0.1,
0.5) and (5.0, 0.2, 2.0), respectively. The selected backend appears in
`repr`; a Python fallback result is not native execution evidence.

::: scpn_control.core._rust_compat

The PID adapter selects the optional native binding or canonical Python fallback.

::: scpn_control.core._rust_pid_compat

#### Statistics Helpers

::: scpn_control.core._statistics

#### Validators

::: scpn_control.core._validators

#### Alfven Eigenmodes

::: scpn_control.core.alfven_eigenmodes

#### Blob Transport

::: scpn_control.core.blob_transport

#### Checkpoint

::: scpn_control.core.checkpoint

#### Disruption Sequence

::: scpn_control.core.disruption_sequence

#### Elm Model

::: scpn_control.core.elm_model

#### Eped Pedestal

::: scpn_control.core.eped_pedestal

#### GK CGYRO

::: scpn_control.core.gk_cgyro

#### GK GENE

::: scpn_control.core.gk_gene

#### GK Geometry

::: scpn_control.core.gk_geometry

#### GK GS2

::: scpn_control.core.gk_gs2

#### GK Nonlinear

::: scpn_control.core.gk_nonlinear

#### GK Online Learner

`OnlineLearner` admits finite nonnegative transport targets only when the
caller-supplied OOD score is inside the configured threshold. Retraining uses a
validation holdout, rolls back on non-improvement, and can persist an auditable
JSON report containing every accepted or rejected update decision.

::: scpn_control.core.gk_online_learner

#### GK QuaLiKiz

::: scpn_control.core.gk_qualikiz

#### GK Species

::: scpn_control.core.gk_species

#### GK TGLF

::: scpn_control.core.gk_tglf

#### GK TGLF Native

::: scpn_control.core.gk_tglf_native

#### GK Verification Report

::: scpn_control.core.gk_verification_report

#### Impurity Transport

::: scpn_control.core.impurity_transport

#### JAX GK Nonlinear

::: scpn_control.core.jax_gk_nonlinear

#### JAX GK Solver

The linear JAX GK solver includes a schema-versioned parity artifact producer
for backend reproducibility. `build_jax_gk_parity_artifact()` and
`write_jax_gk_parity_artifact()` bind the native local-dispersion comparison,
backend metadata, dtype/X64 state, solver kwargs, tolerances, and canonical
payload SHA-256 digest while preserving the backend-parity-only claim boundary.
Artifacts also bind case-parameter digests, native/JAX mode spectra, dominant
mode labels, and case acceptance limits for CBC, kinetic-electron TEM, and
low-drive stable-mode parity evidence. `validation/validate_jax_gk_parity.py`
can require named cases and named backends before admitting an evidence
directory, so archived single-case artifacts cannot be replayed as full parity
coverage.

The persisted reader is available separately as
`validation.validate_jax_gk_parity.validate_jax_gk_parity(artifact_root,
require_parity_artifacts=False, require_cases=None, require_backends=None)`.
It uses only the standard library. A directory selects immediate sorted JSON
children; a file selects that file regardless of suffix. Missing inputs pass
without requirements, while required artifacts or missing named coverage fail.
Names are stripped and deduplicated; unsupported names raise `ValueError`.
Both nonempty requirement sets demand every Cartesian pair. Invalid runtime
policy booleans produce a FAIL finding and normalise to false.

The mutable result includes status, displayed root, admitted artifact count,
normalised requirements, admitted entries, path/field/error findings, sorted
case/backend counts and pair lists, required-pair coverage (`None` without
pairs), admitted maximum gamma/frequency drift (`None` when empty), entry-digest
multiset SHA-256 and report self digest. Entries remain visible when another
file or requirement fails; duplicate files count separately. Finite JSON float
tokens and unique keys are enforced at every depth. Scalars exclude booleans,
nonnumeric values and integers that overflow float conversion.

Gamma drift is `abs(jax-native)/max(abs(native),1e-12)` and frequency drift is
absolute. Growth must be nonnegative and both declared tolerances positive.
Ordered stripped mode lists must agree; dominant labels must agree and occur
in both lists. Declared required modes and any finite non-null growth bound
must hold. Mode names are compared as strings; no physical classifier is run.
Solver kwargs and case parameters require nonempty objects and their declared
canonical digests, but their internal physical domains and consistency with
the named case are not revalidated.

Artifact/report self digests use sorted compact ASCII JSON and omit both
`payload_sha256` and `report_payload_sha256` at the top level. Nested digests
include every key. Comparisons are case-sensitive even though hexadecimal
syntax accepts uppercase letters. These digests bind declarations rather than
original file bytes or authenticated provenance. No device/timestamp/version
verification, native/JAX execution, external-code validation or control
admission occurs. The standalone CLI maps supported output write/path errors
into FAIL and refreshes the report digest; it can write refused reports.
Public `write_jax_gk_parity_report` protects root/immediate selected direct,
resolved, symlink and existing hardlink input aliases. JSON is sorted UTF8+LF
and nonfinite report values refuse serialisation. API path/IO errors propagate;
other destinations may replace. The standalone output failure report retains
its `output_json` finding and updated digest. The registered root command uses
fixed operational refusal one, and Click NUL output is usage two before IO.
Decoder failures use fixed authored findings and refuse nonzero decimal
underflow without exposing raw exceptions or private duplicate member names.
Sequential reads/checks provide no pathname lock or concurrent snapshot.

::: scpn_control.core.jax_gk_solver

::: scpn_control.core.jax_gk_solver.gk_stiffness_chi_i_profile_jax

::: scpn_control.core.jax_gk_solver.build_jax_gk_parity_artifact

::: scpn_control.core.jax_gk_solver.write_jax_gk_parity_artifact

#### JAX GS Solver

::: scpn_control.core.jax_gs_solver

#### Kinetic EFIT

::: scpn_control.core.kinetic_efit

::: scpn_control.core.kinetic_efit.KineticEFITClaimEvidence

::: scpn_control.core.kinetic_efit.kinetic_efit_claim_evidence

::: scpn_control.core.kinetic_efit.assert_kinetic_efit_facility_claim_admissible

::: scpn_control.core.kinetic_efit.save_kinetic_efit_claim_evidence

#### L-H Transition

::: scpn_control.core.lh_transition

#### Locked Mode

::: scpn_control.core.locked_mode

#### MARFE

::: scpn_control.core.marfe

#### MDSplus Acquisition

::: scpn_control.core.mdsplus_acquisition

#### Momentum Transport

::: scpn_control.core.momentum_transport

#### Neoclassical

::: scpn_control.core.neoclassical

#### Neural Turbulence

::: scpn_control.core.neural_turbulence

::: scpn_control.core.neural_turbulence.NeuralTurbulenceClaimEvidence

::: scpn_control.core.neural_turbulence.cross_validate_neural_turbulence

::: scpn_control.core.neural_turbulence.neural_turbulence_claim_evidence

::: scpn_control.core.neural_turbulence.assert_neural_turbulence_quantitative_claim_admissible

::: scpn_control.core.neural_turbulence.save_neural_turbulence_claim_evidence

#### Orbit Following

::: scpn_control.core.orbit_following

::: scpn_control.core.orbit_following.OrbitFollowingClaimEvidence

::: scpn_control.core.orbit_following.orbit_following_claim_evidence

::: scpn_control.core.orbit_following.assert_orbit_following_external_claim_admissible

::: scpn_control.core.orbit_following.save_orbit_following_claim_evidence

#### Pedestal

::: scpn_control.core.pedestal

#### Pellet Injection

::: scpn_control.core.pellet_injection

#### Plasma Startup

::: scpn_control.core.plasma_startup

#### Plasma Wall Interaction

::: scpn_control.core.plasma_wall_interaction

#### Real Data Manifest

::: scpn_control.core.real_data_manifest

`load_real_data_manifest(path, verify_artifact=False)` validates finite,
unique-key UTF-8 JSON and translates supported read/decode/path failures into
`RealDataManifestError`. `validate_real_data_manifest(payload)` inspects schema
declarations; direct dataclass construction bypasses that check. Shot identity
accepts trimmed strings or integers, excluding booleans. Signal names and
artifact URI spellings must be unique. Every provided checksum uses lowercase
64-hex. Unknown mapping metadata is not interpreted by schema validation.

`resolve_manifest_artifact(uri, manifest_path=...)` exposes the local lookup
policy used by checksum verification and directory coverage: ordered manifest,
containing evidence and nearest-repository roots, with canonical containment
and no cwd fallback. It returns a file path without hashing or decoding its
contents. `verify_manifest_artifact` returns the resolved path for a single
local source, or `None` for either an artifact list or an unchecked remote
source; `None` alone does not establish verification. Artifact lists take
precedence over source URI/root checksum. These module APIs do not expand the
package's stable root exports.

The stdlib directory API `validation.validate_data_manifests.validate_manifest_directory`
returns status, discovered/admitted counters, DIII-D coverage, acquisition
linkage and findings. Its two policy keywords require booleans. A realised
MDSplus spec requires matching identity, source/access/licence and the exact
requested signal subset, plus local artifact declarations. Disabling byte
verification preserves this as a metadata observation. Duplicate identities,
invalid discovery containment, read/decode errors and unmatched declarations
fail the report. Invalid specs are findings rather than pending records; empty
manifest discovery ends before spec decoding. No result authenticates facility
access, measured arrays, licence validity or scientific/control readiness.
See the [manifest validation contract](validation.md) for CLI output semantics.

The public `validation.plan_neural_equilibrium_training_campaign` module exposes
`CampaignInputs`, `CampaignPlanError`, `build_plan`, `write_report`, `parse_args`
and `main`. It requires public acquisition PASS before returning a prepared
plan. Consumed dataset metadata, finite budgets and optional producer digest
are validated. Present selected storage bytes must match SHA-256; remote
operator attestation remains explicit and never claims local hashing. Report
writers refuse stale/nonfinite self-digests, malformed render shapes and output
aliases before persistence; supported IO failure returns authored CLI failure.
Reports describe planning only. See the [campaign contract](validation.md) for
storage fields, sequential write limits, explicit output paths and CLI codes.

`validation.neural_equilibrium_campaign_inputs` owns
`read_campaign_dataset_report`, `inspect_storage_payload`,
`summarise_campaign_public_data` and `canonical_campaign_digest` for that
planner. The storage inspector expects validated dataset metadata and literal
boolean controls; direct calls do not validate a complete campaign. Native
module/API examples exercise actual canonical metadata without training.

The public `validation.train_mast_efm_neural_equilibrium` facade retains
`TrainingInputs`, `build_training_report`, `build_result_templates`,
`validate_training_report`, `validate_result_templates`, `write_report`,
`write_result_templates`, `parse_args` and `main`. Controls and source/plan
bindings live in `validation.neural_equilibrium_training_inputs`; tensor and
PCA/ridge work in `validation.neural_equilibrium_training_arrays`; declaration
validators in `validation.neural_equilibrium_training_evidence`; persistence
and output custody in `validation.neural_equilibrium_training_rendering`.
These helpers add no stable package-root exports.

Present selected NPZ bytes must match dataset SHA-256 and the complete tensor
contract even for dry-run. Missing tensors allow preparation with FAIL
pre-run admission. Execution requires exact consumed plan bindings, canonical
plan/source digests and source/compute PASS. The baseline uses training rows
for normalisation, filling and PCA; its unpenalised ridge intercept count avoids
large-alpha cancellation. Nonfinite numerical fits refuse before weights.
Generated reports never self-admit predictive or facility claims.

`validate_training_report(..., require_executed=True)` additionally checks
actual local weight bytes against SHA-256. Other report validation checks
declarations; it does not authenticate physical source measurements.
Result templates require their specific supported section schemas, distinct
required field names and launch/dataset bindings. Writers render before
sequential JSON/Markdown writes and refuse output aliases; second-write
failure can leave JSON. See the [trainer contract](validation.md) for the
complete shape, source, CLI and claim limits.

The offline `validation.validate_public_data_acquisition` APIs inspect a separate
Zenodo file-manifest schema. Loader/mapping APIs return frozen acquisition/file
records or `PublicDataAcquisitionError`; directory inspection returns admitted
counters and ordered findings. Download URLs must match the numeric DOI record
and decoded unique safe key. Selected local mirrors bind SHA-256, MD5 and size
to one byte stream, using repository-relative then full adjacent-relative
lookup. Raw record bytes are checked only when an adjacent record exists; absent
records leave an unverified digest declaration. Metadata PASS neither downloads
nor authenticates remote records, licences, tensors or scientific admission.
The native source docstrings describe parsing, lookup and output failure limits.

#### Runaway Electrons

::: scpn_control.core.runaway_electrons

#### Stellarator Geometry

::: scpn_control.core.stellarator_geometry

#### Tearing Mode Coupling

::: scpn_control.core.tearing_mode_coupling

#### Vessel Model

::: scpn_control.core.vessel_model

#### VMEC Lite

::: scpn_control.core.vmec_lite

::: scpn_control.core.vmec_lite.VMECLiteClaimEvidence

::: scpn_control.core.vmec_lite.vmec_lite_claim_evidence

::: scpn_control.core.vmec_lite.assert_vmec_lite_full_vmec_claim_admissible

::: scpn_control.core.vmec_lite.save_vmec_lite_claim_evidence

### Phase Modules

#### GK UPDE Bridge

::: scpn_control.phase.gk_upde_bridge

### SCPN Compiler and Replay Modules

#### FPGA Export

`scpn_control.scpn.fpga_export` writes bounded HDL project files and
tamper-evident export evidence. It does not claim to emit a deployable FPGA
bitstream. Facility or hardware claims require qualified synthesis report
evidence through `assert_hdl_export_claim_admissible`.

::: scpn_control.scpn.fpga_export

::: scpn_control.scpn.fpga_export.HDLExportEvidence

::: scpn_control.scpn.fpga_export.hdl_export_evidence

::: scpn_control.scpn.fpga_export.assert_hdl_export_claim_admissible

::: scpn_control.scpn.fpga_export.save_hdl_export_evidence

::: scpn_control.scpn.fpga_export.load_hdl_export_evidence

#### Geometry Neutral Contracts

::: scpn_control.scpn.geometry_neutral_contracts

#### Geometry Neutral Replay

::: scpn_control.scpn.geometry_neutral_replay

::: scpn_control.scpn.geometry_neutral_replay.GeometryNeutralReplayEvidence

::: scpn_control.scpn.geometry_neutral_replay.geometry_neutral_replay_evidence

::: scpn_control.scpn.geometry_neutral_replay.assert_geometry_neutral_replay_claim_admissible

::: scpn_control.scpn.geometry_neutral_replay.save_geometry_neutral_replay_evidence

::: scpn_control.scpn.geometry_neutral_replay.load_geometry_neutral_replay_evidence

### Studio Vertical

CONTROL's SCPN STUDIO vertical expresses its verbs and evidence on the locked
`scpn-studio-platform` contract (installed via the optional `studio` extra). It
consumes the platform SDK rather than forking it: verbs are declared as platform
`Verb` records, results map to platform `EvidenceBundle` records, and the
capability manifest is the platform `CapabilityManifest`.

#### Studio Verbs

::: scpn_control.studio.verbs

#### Studio Evidence Bundles

::: scpn_control.studio.evidence

#### Studio Capability Manifest

::: scpn_control.studio.manifest

#### Studio Live-Emitter Adapters

::: scpn_control.studio.adapters

#### Studio Panel Feed

The federated panel reads CONTROL's verbs and claims from the wire feed this module
emits (`studio.control-feed.v1`), so the UI never holds a second, drifting copy of
the contract. Claim summaries include the platform freshness axis: the safety
certificate emits `verified-at-source` after the mapper re-checks proof coverage,
while the other representative claims emit `traceable-unchecked` and render at their
boundary unless fresh source verification is supplied. Regenerate the standalone artefact with
`python -m scpn_control.studio.feed > studio-web/public/studio-feed.json`.

::: scpn_control.studio.feed

#### Studio Sealed Safety Claim

The Hub's transparency log verifies artefacts with an RFC-8785 JCS
canonicaliser that rejects non-integer JSON numbers, so the sealed
safety-certificate claim is emitted float-free: integers stay within the
exact-interoperability range and exact decimals travel as strings. The module
is deliberately SDK-free so the artefact can be produced from a checkout
without the optional `studio` extra.

::: scpn_control.studio.sealed_claim

---

## CLI

```bash
scpn-control demo --scenario combined --steps 1000
scpn-control benchmark --n-bench 5000 --json-out
scpn-control validate --json-out
scpn-control validate-release-evidence artifacts/release_evidence_report.json --json-out
scpn-control info --json-out
scpn-control live --port 8765 --zeta 0.5 --layers 16
scpn-control hil-test --shots-dir path/to/shots
```

| Command | Description |
|---------|------------|
| `demo` | Closed-loop control demonstration (PID, SNN, combined) |
| `benchmark` | PID vs SNN timing benchmark with JSON output option |
| `validate` | Transport/import hygiene and six enabled evidence readers; import or gate findings emit FAIL and exit one, including explicitly scoped skips |
| `validate-release-evidence` | Declaration checks and exact input-byte digest for manifest, JAX parity, traceability, multi-shot, runtime and native formal summaries; referenced artifacts are not reopened |
| `info` | Version, Rust backend status, weight provenance, Python/NumPy versions |
| `live` | Real-time WebSocket phase sync server |
| `hil-test` | Hardware-in-the-loop test campaign against shot data |

---

## Rust Acceleration

When `scpn-control-rs` is built via maturin, all core solvers use Rust backends automatically:

```python
from scpn_control import RUST_BACKEND
print(RUST_BACKEND)  # True if Rust available

# Transparent acceleration — same Python API, Rust execution
kernel = FusionKernel(R0=6.2, a=2.0, B0=5.3)
```

Build Rust bindings:

```bash
cd scpn-control-rs/crates/control-python
maturin develop --release
```

### PyO3 Bindings

| Python Class | Rust Binding | Crate |
|-------------|-------------|-------|
| `FusionKernel` | `PyFusionKernel` | control-core |
| `RealtimeMonitor` | `PyRealtimeMonitor` | control-math |
| `SnnPool` | `PySnnPool` | control-control |
| `MpcController` | `PyMpcController` | control-control |
| `Plasma2D` | `PyPlasma2D` | control-core |
| `TransportSolver` | `PyTransportSolver` | control-core |

---

## Core — Native Rust Engine Wrapper

`scpn_control.core.rust_engine` is the Python control-plane wrapper for the
optional PyO3 native execution bridge. It configures native campaign execution,
formal-verification mode, runtime admission, transport backend selection, and
emergency telemetry handoff while keeping timing-critical execution inside the
compiled Rust data plane when the extension is available.

This wrapper is an execution boundary, not a physics solver. It does not turn a
local workstation run into target-hardware PCS evidence unless the matching
runtime-admission and benchmark-context reports pass.

The optional UDP heartbeat is an authenticated transport-liveness hint, not a
safety interlock. When enabled, the Rust receiver uses an explicit bind host
(loopback by default), exact source-IP allowlist, fixed `SCPNHB01` frame magic,
strictly increasing per-receiver 64-bit counter, and full HMAC-SHA256
verification before it refreshes liveness. The secret is loaded from a private
key file named by `SCPN_CONTROL_HEARTBEAT_KEY_FILE`; it is never accepted as a
CLI value. Python
senders can use `load_transport_heartbeat_key()` and
`build_transport_heartbeat_frame()` for the same exact 48-byte protocol.

::: scpn_control.core.rust_engine

## API surface usage model

This page is the binding map for practical integration, not a promise of stable semantics for every release.

- Use top-level exports for workflow composition.
- Use submodules for module-specific interfaces and keep import paths explicit.
- Use documented Rust bindings for hot-path execution only after matching environment checks pass.

When uncertain, start in Python for reproducibility and then switch to native paths only for timing and production-oriented experiments.

## API usage expectation for enterprise workflows

This API surface is intentionally split by risk and execution boundary:

- **Safe exploratory usage**: top-level Python entry points with explicit argument
  checks and deterministic defaults.
- **Research extension usage**: module-level APIs where validation still controls the
  admissible claim level.
- **Deployment-oriented usage**: PyO3-backed primitives where timing and transport
  behavior are benchmarked under explicit host context.

When preparing an integration PR, include both:

- one reproducible usage example against the public API,
- one admission-visible benchmark or validation record for the same path.

That pairing is the minimum contract for external-facing confidence.

## Practical use and scope

Use this page as the public contract boundary for `scpn_control` exports and import-level behavior.

- Validate import and symbol usage here before changing interface signatures in `src/scpn_control`.
- Use the API map to decide whether integration changes are safe for third-party callers.
- Pair API updates with corresponding validation and release notes for any public call-flow changes.


## Signed external flux and conservative transport

These APIs preserve raw signed provider moments, explicit SI reference scales
and conservative face-flux evolution. See [TGLF fluxes](tglf_flux.md) and
[face transport](transport_flux.md) for units, boundaries and remaining
physical admission requirements. The package reads retained provider output;
the launcher that executes the provider is a validation command of the
repository and is described on the TGLF page.

::: scpn_control.core.tglf_flux

::: scpn_control.core.tglf_units

::: scpn_control.core.transport_flux

::: scpn_control.core.tglf_miller.TGLFSpecies
::: scpn_control.core.tglf_miller.TGLFMillerGeometry
::: scpn_control.core.tglf_miller.miller_tglf_deck

::: scpn_control.core.tglf_miller.miller_volume_metric

::: scpn_control.core.transport_flux.TransportFaceGeometry

## Validation current-drive reference

`validation.validate_current_drive_reference.validate_current_drive_reference(artifact_root,
require_reference_artifacts=False)` inspects local declarations without running
current-drive physics. The direct script and registered
`scpn-control validate-current-drive-reference` command call this API.

| Contract | Behaviour |
| --- | --- |
| Discovery | Sorted, non-recursive directory `*.json` matches, or one explicit file of any suffix. Caller-relative paths; symlinks followed. |
| Empty input | Optional mode returns `pass`, zero entries. Required mode returns `fail`; an optional pass is not reference evidence. |
| JSON | UTF-8; unique keys at every depth; finite floating-point values, including otherwise unselected fields. |
| Identity | Version `1.0`; nonblank source/model/version/dataset/date strings; digest checked only for 64 hex characters. |
| Provenance | Declared public URL/DOI, measured shot/diagnostic URI, or a named external code and artifact URI. No retrieval or authentication. |
| Units | Exact W, A, A/m^2, 10^19 m^-3, keV, dimensionless rho, s and keV labels; no conversion. |
| Grid metadata | Positive power and rho_points, with `0 < rho_min < rho_max <= 1`; no integral grid-count or array-shape validation. |
| Comparison | Five finite non-negative declared errors, each at most its positive declared tolerance; no metric recomputation. |
| Report | Fresh `status`, `root`, count, requirement flag, `entries` and `errors`. Failed reports can retain accepted entries as diagnostics. |
| State | Local reads only in the API; no cache, lock, FFI, actuation or coherent concurrent-read snapshot guarantee. |

The current default reference directory has no artefacts. This actual persisted
bounded report demonstrates a refusal (the command exits 1):

```bash
python validation/validate_current_drive_reference.py \
  --artifact-root validation/reports/current_drive_claims.json \
  --require-reference-artifacts --json-out
```

The Python docstring contains executable examples against the same repository
corpus. A passing declaration does not verify referenced bytes, freshness,
external-code results, experiment provenance or facility admission.

Both commands support `--artifact-root`, `--require-reference-artifacts`,
`--json-out` and `--output-json`. Public `write_current_drive_reference_report` protects the root and immediate selected
direct/resolved/symlink/existing hardlink input aliases, then creates parents and
replaces unrelated output with sorted UTF8 JSON plus LF, before stdout is
emitted. This is not an atomic or durable-write promise. Expected input read and
decode failures use structured authored errors. Output filesystem failure prints
`could not write current-drive reference report`; the script returns 2 and the
registered Click command returns 1 with its `Error:` prefix. Click first rejects
an existing directory destination with its authored usage error and exit 2.
Argument-parser help and usage retain their usual 0/2 behaviour.

## Python lint contract

`tools.check_python_lint_contract.lint_contract_errors(repo)` reads the four
repository-relative `SurfaceContract` declarations for `Makefile`,
`.github/workflows/ci-static-governance.yml`, `tools/preflight.py` and
`.pre-commit-config.yaml`. It returns a fresh list of errors in surface order,
then required/forbidden fragment order. Each missing or non-file surface yields
one error; expected read and UTF-8 failures yield fixed authored text and
inspection continues. Symlinks are followed without containment checks.

`SurfaceContract(path, required, forbidden=())` is a frozen declaration object.
Its fields store literal substrings without constructor validation. Matching
uses case and whitespace sensitive UTF-8 text with universal newline handling;
comments and inactive strings can satisfy a fragment. This is a declaration
check, not an execution or semantic parser verdict. It does not inspect Git,
run linters/workflows, mutate generated ledgers, validate tool versions or prove
native documentation completeness. There is no cache, lock or coherent
concurrent-file snapshot guarantee.

The pre-commit declaration binds the actual local checker entry and its filter
to both the entry workflow and static-governance workflow. The package command
retains Ruff D rules; the broad test/tool/validation command excludes D and the
separate owner/docstring gates enforce their stated scopes. The generated
coverage-ledger spelling exclusion is required unchanged.

```bash
python tools/check_python_lint_contract.py
python tools/check_python_lint_contract.py --repo ./copied-checkout
```

`main(argv=None)` accepts process arguments or an argument list without the
executable name. Default repository root comes from the script; `--repo` uses
caller-relative spelling and is resolved before inspection. Return codes are
0 for matching fragments, 1 for declaration/read failures, and 2 for an
unresolvable root. Authored diagnostics use stdout. ArgumentParser retains
help exit 0 and usage exit 2. The owning-language docstring executes a genuine
repository example and can be rendered with pydoc.

## Public surface hygiene

`tools.check_public_surface_hygiene.scan_repository(repo)` enumerates Git index
paths and inspects their current UTF-8 worktree content. It returns a fresh list
of `Finding(path, line, category, detail)` values in Git path order, followed by
the text scanner's rule order. Paths are relative to the supplied root; line is
one-based for content or zero for a path-name finding. Frozen fields do not
validate constructor input. Details retain source text and are not redacted.

`iter_scanned_files(repo)` is lazy. Git `ls-files -z` preserves native filenames
without quoting or newline splitting. Relative roots remain caller-relative;
Git subdirectories enumerate their own index-relative scope. The yielded paths
are absolute only when the supplied root is absolute. Case-sensitive suffix or
workflow selection, private/cache exclusions and the guard's own fixture
exclusions apply before regular-file inspection. Missing/non-file indexed
paths are skipped, symlinks to regular files are followed, and no containment
check is made. Untracked files, staged blob bytes and history are not inspected.

`scan_text(path, text)` applies path-dependent patterns directly to a Unicode
payload without filesystem access or repository selection exclusions. Logical
paths use POSIX spelling; no normalisation or semantic Markdown/JSON/source
parsing occurs. Findings first report path identity, then a first-eight-line
Markdown preamble or validation producer, then line rules. Each family takes
its first match; independent families may both report on the same line.
Markdown fence-looking lines toggle state without matching delimiter or length.
Only planning checks are suppressed inside fences. Bounded context matches
suppress promotion, path-specific and planning rules after identifier checks;
reviewed planning exceptions require the exact path. A clean lexical scan does
not authenticate or validate the underlying claims.

Invalid UTF-8 selected payloads are skipped as binary. Other selected-file read
failures, selected-path inspection failures or Git enumeration failures raise
`PublicSurfaceScanError` with fixed
authored text; a failed scan does not return successful partial findings. Text
uses universal newline handling. There is no write, cache, locking, subprocess
timeout, coherent index/worktree snapshot, renderer or deployment check.

```bash
python tools/check_public_surface_hygiene.py
python tools/check_public_surface_hygiene.py --repo ./repository
```

`main(argv=None)` accepts process arguments or an argument list without its
executable name. The default root comes from the script; `--repo` resolves
caller-relative spelling. Exit 0 means no findings in inspected text, 1 means
policy findings, and 2 means a root, Git or read refusal. Reports use stdout;
finding details retain payload text while operational refusals omit exception
diagnostics. Argparse retains help exit 0 and invalid-argument exit 2. Native
docstrings inspect the actual maintained development guide and render with
pydoc; this is separate from the ordinary all-definition enforcement gate.


## API declaration contract

`tools.api_contract_inventory.build_inventory(repo=ROOT)` reads an actual local
source graph without importing it. Relative API roots use the caller's working
directory. `Candidate` records a lexical qualified name, class/callable kind,
repository-relative POSIX path, one-based AST definition line and exact document
directive membership. It is frozen; construction does not validate field values.
The result contains fresh mutable mappings/lists with integer counts and SHA-256
name digests, without units, timing samples or numerical accuracy claims.

Python scans sorted `src/scpn_control/**/*.py` for non-private top-level classes,
functions and async functions. Methods, nested declarations, imports and reexports
are not candidates. Package initialiser names omit `__init__`. Standalone `:::`
lines in `docs/api.md` provide exact qualified-name membership, without rendering.
The package root must declare literal `__all__` as unique identifier strings in
a list and `_EXPORT_MODULES` as identifier-to-dotted-module strings in a table.
Stable ownership uses an exported short name and owner module prefix; an
aggregator owner may therefore cover a declaration in a leaf module. The special
`__version__` and `RUST_BACKEND` names remain counted exports but do not classify
Python candidates. Remaining candidates prefer exact directive membership, then
nonstable status. This does not resolve actual lazy imports or prove stability.

C matches `SCPN_SOLVER_API` prototypes in `core/solver.h`; Lean matches documented
`inductive`, `def` and `theorem` ASCII names in `PulsedFSM.lean`. Their lists retain
source order. Rust matches line-start `pub` declarations of struct/enum/trait/fn/type,
with optional async and indentation, under `scpn-control-rs/crates`, excluding
relative paths containing `target`. TypeScript matches column-zero `export`
function/class/interface/type/const/enum declarations, optionally default/async,
in `studio-web/src/**/*.ts*`. These regexes do not implement compiler parsers,
comment handling, configuration, transitive exports or proof checking. Each
selected Python/Rust/TypeScript family must contain source files; valid files
with no selected regex declarations still produce zero symbols.

Name digests sort strings, retain duplicate declarations, join with LF and hash
UTF-8 bytes. Rust/TypeScript strings include relative file owners; Python name
digests do not hash implementation bytes, line positions, types, signatures or
docstring content. Reads follow symlinks without containment, Git tracking,
locking, cache or a coherent snapshot guarantee. Inputs use UTF-8 and universal
newlines. Expected read/parse/container failures raise authored
`ApiContractInspectionError`; no successful partial inventory is returned.

`tools.check_api_contracts.check_contracts(repo=ROOT, registry_path=None)` compares
the inventory against `tools/api_contract_registry.toml`, or a cwd-relative explicit
registry. It requires the v1 schema, nonnegative integer counts (excluding booleans),
64 lowercase hexadecimal digests, an exact three-class partition, unique native
identifier lists and nonempty renderer fragment declarations. Extra metadata and
policy prose are not interpreted. Renderer paths must be relative without parent
traversal; symlinks can still point outside the root. Fragments are case- and
whitespace-sensitive text membership, and comments/inactive declarations can
satisfy them. An empty error list means these declarations match; no renderer,
compiler, source authentication, independent review or scientific admission ran.
Diagnostics retain declared paths/fragments without redaction.

```bash
python tools/check_api_contracts.py --repo .
python tools/check_api_contracts.py --repo . --print-inventory
python tools/check_api_contracts.py --repo . --registry tools/api_contract_registry.toml
```

The CLI resolves its root; status 0 means matching declarations/fragments or a
printed inventory, 1 means declaration/fragment drift, and 2 means an authored
inspection refusal. Diagnostics use stdout; argparse retains help 0 and usage 2.
`--print-inventory` does not read the registry or renderer inputs. This command
never rewrites a baseline. A current working graph can differ from the declared
registry; aligning a copied candidate is not a canonical classification review.


## Release-evidence declarations

`validation.validate_release_evidence.validate_release_evidence(path)` returns
the frozen `ReleaseEvidenceAdmission` with `status`, ordered `errors`, the
exact decoded report-byte `report_sha256`, and required sections declaring
PASS in `admitted_gates`. That last field is a declaration list and may remain
populated on overall FAIL. Unreadable, malformed UTF-8/JSON, non-object roots,
duplicate keys and nonfinite floating tokens fail before digest admission.
Malformed declaration types produce findings rather than set/comparison errors.

All six required gate summaries are checked. Manifest coverage needs equal
nonnegative integer counts and empty missing list; JAX string lists/entry
objects cover the three case/two backend campaign; traceability blocks all
declared gaps; multi-shot declares Python/PyO3/Rust and bounded digest syntax;
runtime/native formal summaries enforce their local/production claim flags.
Local runtime admission failure remains permitted with no production claim.
The full [declaration contract](validation.md) describes fields and boundaries.
The reader does not reopen or authenticate artifacts, recompute any referenced
digest/metric, verify an AOT proof, or grant realtime/facility/control readiness.
Extra metadata is unvalidated beyond duplicate/nonfinite JSON rules.

The standard-library standalone script accepts caller-relative/symlink paths,
prints JSON or text, and returns 0/1 for PASS/findings. Argparse errors exit2;
no output report is written. The registered Click entry point checks readable
files and uses stderr for text findings. Neither reader imposes filesystem
containment or an input size/depth budget.

```bash
python validation/validate_release_evidence.py report.json --json-out
scpn-control validate-release-evidence report.json --json-out
```

## Density-reference declarations

`validation.validate_density_reference.validate_density_reference(artifact_root,
require_reference_artifacts=False)` inspects local JSON metadata and declared
metrics. The direct `python validation/validate_density_reference.py` and
registered `scpn-control validate-density-reference` commands use this reader.
The direct entry works from another directory without an installed package and
uses only the standard library. Native examples inspect the actual Python owner
as invalid JSON; they create no reference data or passing physical results.

| Contract | Actual behaviour |
| --- | --- |
| Selection | A directory supplies sorted immediate `*.json` paths; one file is inspected regardless of suffix. Relative paths use the caller's working directory. Symlinks are followed; no containment or Git-tracking check occurs. |
| JSON | Unique keys at every depth; finite floating-point tokens including unused metadata; nonzero tokens rounded to binary64 zero are refused. Required numbers must have representable finite float values; booleans are refused. Other unused fields are not schema-validated. |
| Identity | Schema `1.0`, six nonblank identity/source strings, exactly 64 hexadecimal digest characters, allowed source labels and source-specific string/URI declarations. DOI/URL/shot/code/digest authenticity is not checked. |
| Grid and actuators | Integer `n_rho >= 2`, positive declared radii, nonnegative finite actuator values and recycling fraction at most one. No radial samples, `a < R`, physical geometry or actuator execution is established. |
| Units and comparison | Seven exact unit labels, positive integer declared case count, four finite nonnegative declared errors no greater than positive declared tolerances. No conversions, density model or metric recomputation occurs. |
| Report | Fresh `status`, `root`, `reference_artifacts`, `require_reference_artifacts`, `entries` and `errors`. Count includes accepted declarations. Optional empty/missing scope passes with zero entries; strict absence fails. Any finding makes status fail. |
| Failures | Candidate read, UTF-8, parse and declaration failures become per-path findings. Root selection/enumeration failures may propagate `OSError`. No fallback reference is substituted. |
| Commands | Original options and help retained; pass exits 0, findings or supported operational refusal exit 1, argparse help/usage retain 0/2. Public writer protects selected input aliases, creates parents and replaces unrelated output with sorted UTF8 JSON+LF. API errors propagate; commands use fixed operational stderr. Click directory/NUL output uses usage 2. Text findings use stderr. |
| State | Local reads only in the reader API, no network fetch, digest/metric recomputation, model, cache, lock or concurrent snapshot. A passing report supplies no external-code, measured-campaign, action or facility admission. |

The inherited owning tests contain illustrative schema declarations; their
labels and numbers are not an authenticated density-reference corpus. Negative
cases exercise the actual reader, real files, standard-library entry and root
Click registration. The ordinary all-definition documentation gate covers this
source and its owning tests without certifying scientific reference acceptance.

## MAST supervised dataset producer

`validation.build_mast_efm_neural_equilibrium_dataset` preserves the public
`DatasetInput`, `build_dataset`, `build_feature_matrix`, hash/JSON/path helpers,
`validate_feature_matrix`, `validate_dataset_report`, `write_report`,
`parse_args` and `main` entry points.
The facade delegates candidate/control custody, feature derivation, target
assembly and report validation/rendering to four cohesive Python modules.
Schema and feature/target constants have one shared owner for producer,
planner and trainer imports.

`DatasetInput` requires local `Path` selections, an explicit `.npz` output and
nonempty disjoint tuples of genuine positive shot IDs. All selected files and
the dataset destination must resolve inside `storage_root`. Candidate JSON
requires finite numbers, unique keys, the actual converter schema, a matching
canonical self-digest, reference-only blocked flags, and exact selected shot,
count and partition declarations. Each reference NPZ is captured once as compressed-file bytes. The candidate
SHA is checked against that capture, and `allow_pickle=False` decoding reads
the same capture. A pathname change cannot replace those verified input bytes;
separate before/after hashes could miss a change followed by restoration.
One compressed bundle is retained temporarily alongside its decoded arrays.
This checks local byte custody, not physical source authenticity or a coherent
multi-file campaign snapshot. The reference loader and supervised trainer
both call `validation.neural_equilibrium_dataset_tensors.load_verified_npz`;
its same captured bytes are verified and decoded before arrays are admitted.
The trainer records the declared SHA only after this successful verification.

Required arrays enforce per-row shot/time/shape custody, genuine boolean masks,
finite valid observations and strictly monotonic coordinate grids. Descending
coordinates are flipped together with flux and masks. Cross-shot grids and
profile dimensions must match; no interpolation is supplied. LCFS valid points
are compacted in their original order, counts record real valid points, and
ragged coordinates/masks use NaN/False padding.

`build_feature_matrix` returns the existing 12 ordered columns. Missing scalar
or profile inputs retain explicit defaults; unsupported scalar/higher-rank
profile/geometry inputs refuse. Ip/Bt/FF-prime source provenance is declared
only when every selected shot supplies the key. Pressure mean/median and
campaign RMS median avoid intermediate overflow for large finite values.
These reference-derived features do not establish independent predictions of
the same profiles or geometry.
`validate_feature_matrix(values, row_count=...)` requires the declared N × 12
real array shape and finite values after conversion to float64. The producer
uses it for its completed matrix, and the actual NPZ trainer uses it when
loading that matrix. Row count must be a genuine positive integer, excluding
booleans and coerced values; column-name identity remains with its callers. A value finite in a wider stored dtype can overflow in float64
and must refuse before training. The returned array can be a view; this helper
does not freeze caller-owned mutable memory.

`validate_dataset_report` returns the mapping unchanged after checking its
finite canonical digest, blocked flags, supported feature/target/source policy,
relative storage declarations, split/shot counts, time ranges and grid bounds.
It does not retrieve the remote dataset or authenticate measurements. Writers
validate and render before sequential JSON/Markdown writes. Pair aliases,
including symlink/hardlink aliases, refuse; second-file failure can leave the
first JSON. The CLI checks every output against selected inputs before dataset
persistence; supported domain/IO failures print authored `FAIL:` and return 1,
while success returns 0 and argparse retains help 0/usage 2.

```bash
python validation/build_mast_efm_neural_equilibrium_dataset.py --help
```

The [storage-host rebuilding command](validation.md) requires actual local
selected reference bytes. Retained converter-shaped NPZ regressions exercise
producer → planner → trainer and source/custody refusals; they provide
engineering format evidence, not authenticated physical MAST performance.

## MAST converted feature-source audit

`validation.audit_mast_efm_feature_provenance` exposes `build_audit`,
`validate_audit_report`, `write_report`, `parse_args` and `main`. Converted
source capture and all-shot aggregation live in
`validation.mast_efm_feature_audit_inputs`; declaration validation and paired
report persistence live in `validation.mast_efm_feature_audit_reporting`.

The auditor reads finite, duplicate-free JSON and validates the actual dataset
producer contract. Its input-file SHA binds that same captured JSON. Each
declared reference SHA binds the exact compressed bytes decoded by the shared
verified NPZ reader. Shot IDs, row counts, ordered time bounds and matching
coordinate grids must agree with the producer report. The supported source
vectors require real dtypes, exact row shapes, finite float64 values and
strictly positive FF-prime RMS. A union of inventory keys cannot resolve a
feature: its canonical channel must be complete on every selected shot.
Alternative aliases remain inventory hints. The observed fallback set must
match the dataset producer declaration.

PASS means complete converted source channels. This audit does not inspect the
supervised dataset tensors, revalidate target observations, authenticate original
acquisition or grant predictive admission. JSON and reference captures are
sequential. Writers validate finite self-digests and exact per-shot/aggregate
declarations before rendering and IO; they protect selected inputs against
output aliases. A second-file error may leave the first JSON. Direct validator
and writer calls validate declarations without reopening source files.

The CLI retains its existing paths and accepts explicit argument sequences.

The trainer uses the same complete dataset declaration reader and captures its
JSON bytes once. `validate_audit_dataset_bindings` requires a complete converted
audit whose raw report SHA, producer payload SHA, dataset NPZ SHA, dataset and
candidate locators, reference identity/count, and each shot's ID/path/SHA/rows
match that selected declaration. A matching dataset name or self-consistent
audit digest alone does not pass source admission. Even a whitespace-only JSON
rewrite requires a new audit of the selected declaration bytes. Missing legacy
bindings remain FAIL. Dry-run retains these diagnostics; execution refuses
before fitting or writing weights. Copied compute datasets may have a different
local path, while their bytes must match the declared storage dataset SHA.

Output alias custody now lives in the shared dataset-contract module, retaining
the trainer-rendering public re-export and its existing path semantics. This
keeps source declaration validation independent of training/report persistence
dependencies. The trainer compares declared reference bindings without reopening
the reference corpus or authenticating acquisition.
A completed pass or blocked inspection exits 0; invalid/missing local inputs
print an authored `FAIL` and exit 1; argparse help and usage retain 0 and 2.

```bash
python validation/audit_mast_efm_feature_provenance.py --help
```

The [source-audit command and preserved scientific report](validation.md)
describe the storage boundary. Scientific reports require authentic source
custody before regeneration; the current engineering NPZ regressions provide
format and error-handling evidence.

## MAST original feature-source audit

`validation.audit_mast_efm_original_feature_sources` exposes
`load_zarr_candidate_metadata`, `classify_feature_sources`,
`build_original_feature_source_audit`, `validate_original_audit_report`,
`write_report`, `parse_args` and `main`. Candidate descriptor inspection and
classification alone cannot establish readiness. The full builder validates
the producer declaration and embeds the actual converted feature audit.

For each selected shot, the builder captures the original consolidated Zarr
metadata and physical files, then decodes that captured byte mapping through
the public reference extractor. Required observation chunks must exist;
metadata and fill values cannot replace missing observations. Each selected
reference SHA is checked before decoding. All 20 reference arrays, including
times, coordinates, targets and masks, must exactly match the original
conversion. An ordered selected time subset is supported. Comparison retains
ordinary float64 RMS values exactly; scaling protects extreme finite profiles
against intermediate overflow and underflow.

The preferred channels are `plasma_current_x` in A, `bphi_rmag` in T and
`ffprime` in T-rad. Alternative names require a separate source policy. A
scalar successful status guards the entire shot; positive convergence flags
still select actual time rows. Negative pretrigger times may precede selected
nonnegative observations, but the full source clock must increase strictly.

The v2 report records per-file SHA-256 and size, a canonical snapshot digest,
conversion results and complete selected producer/reference bindings.
`validate_original_audit_bindings` checks those bindings against the selected
dataset declaration. The trainer requires this full v2 audit together with
the converted audit. Legacy v1 reports, stale self-digests and a changed raw
producer JSON SHA refuse source admission. `source_ready` means local
conversion equivalence; it does not validate supervised tensor layout,
authenticate acquisition or establish predictive admission.

Capture, reference reads and writes are sequential observations, without a
campaign transaction or source signature. The captured compressed store and
decoded arrays occupy memory. Writers validate and render before IO, protect
selected inputs and original stores from output aliases, and can leave the
first JSON if the Markdown write fails. A complete ready or blocked inspection
exits 0; ordinary read, conversion or write refusals exit 1. Argparse help and
usage retain exits 0 and 2.

```bash
python validation/audit_mast_efm_original_feature_sources.py --help
```

See the [original source audit and rebuilding procedure](validation.md) for
storage selection and preserved scientific artifacts.

## Git path exposure inspector

`tools.check_history_exposure` exposes `Exposure`, `collect_current_paths`,
`collect_history_paths`, `find_first_commit`, `collect_exposures` and `main`.
The aggregate requires a Git worktree root, reads literal NUL-delimited index
names and optionally adds all locally reachable ref history. Pattern exceptions,
Git log ordering, index-only commit IDs and operational CLI exits are described
in the [owning-language contract](history_exposure.md). This inspects path names
and declared Git history, without reading blobs or certifying secret contents.

## MAST diagnostic model evaluation

`validation.evaluate_mast_efm_neural_equilibrium` retains `FeatureProjection`,
`build_feature_projection`, `masked_rmse`, `evaluate_flux_geometry`,
`load_reference_bundle`, `evaluate_reference_bundle`, `write_report`,
`sha256_file`, `sha256_json`, `parse_args` and `main`.
`validation.mast_efm_evaluation_features` owns source projection;
`validation.mast_efm_evaluation_geometry` owns metre-grid axis/contour metrics.
These remain Python interfaces using actual accelerator inference with existing
weights. The [validation guide](validation.md) specifies provenance and custody.

`build_feature_projection(data, ffprime_reference=VALUE)` accepts an open NPZ
mapping or dictionary of arrays. It uses the supervised producer's twelve-column
definitions. Source Ip/Bt observations replace wholly absent-channel defaults.
FF-prime requires a positive finite campaign normalisation reference; absent
provenance retains explicit neutral fallback. Caller-owned inputs remain
unchanged. Missing axes/scalars and unobserved profiles retain declared defaults.

`evaluate_reference_bundle(reference_path, weights_path, prediction_path,
ffprime_reference=VALUE)` verifies captured reference bytes and loads captured
existing weights from a temporary snapshot. It handles the accelerator's
squeezed single-row result. Prediction output must not alias either input;
`main` checks all three outputs before inference, and `write_report` protects
report-declared inputs/prediction. Writes are not transactional; non-alias outputs
may be replaced. Reports retain `admission_ready=false` and
`strict_artifact_emitted=false`.

Flux RMSE is in Wb/rad and scales finite residuals before squaring. Geometry
derives an extremal grid point for the axis and unique rounded boundary edge
crossings; nearest-point distances are directed. Inferred grids use a padded
reference LCFS/axis envelope with explicit provenance. A point cloud does not
prove a closed, connected plasma boundary. Legacy metric aliases and last-sample
`q95` are documented in reports and the guide. Source/shape/model failures refuse;
no predictive tolerance or independent model qualification is implied.

```bash
python validation/evaluate_mast_efm_neural_equilibrium.py --help
```

## Orbit reference declaration inspection

`validation.validate_orbit_reference.validate_orbit_reference` reports persisted
metadata consistency, including exact unit labels, source/citation declarations,
SHA-256 spelling, case counts and declared error/score bounds. It authenticates
no referenced bytes, DOI, execution timestamp, model or orbit calculation.
Missing optional roots pass with zero entries; required mode fails. Read/JSON/
UTF-8/duplicate-key errors become fixed per-file findings. Source-type and
unrepresentable-number errors remain authored field findings.

`write_orbit_reference_report` protects selected input aliases before creating
parents and replacing other output. Direct script and registered
`validate-orbit-reference` use this same writer, with supported operational
failures returning exit 1 and fixed authored text. Neither inspection nor
persistence is a concurrent filesystem snapshot or transaction.

::: validation.validate_orbit_reference

Declaration predicates are maintained in `validation.orbit_reference_contracts`;
they are consumed through the public persisted reader. Real source inspection,
refusal and persistence tests use declaration carriers and actual API/CLI paths;
passing those carriers is no external orbit-code or facility validation.


## Uncertainty reference declaration API

`validation.validate_uncertainty_reference.validate_uncertainty_reference` reads
selected persisted JSON declarations and reports schema, source, unit and
declared-tolerance findings. It does not authenticate referenced bytes, citations,
model identity, execution time or the UQ propagation chain. Real campaign URI
fields retain nonblank-string presence semantics; no lexical or remote URI
validation is performed by this owner.

`write_uncertainty_reference_report` preserves selected input aliases before
creating parents and writing sorted UTF-8 JSON. Operational failures propagate
from this API and become fixed authored refusal in the standalone script and
registered `validate-uncertainty-reference` command. Directory observations and
output writes do not constitute a transaction against concurrent changes.

::: validation.validate_uncertainty_reference

Declaration predicates are maintained in
`validation.uncertainty_reference_contracts`; they compare declared bounds
without running a physical uncertainty model.


## VMEC reference declaration API

`validation.validate_vmec_reference.validate_vmec_reference` reports persisted
schema, source, identity, unit, Fourier-truncation and declared-error findings.
It computes no stellarator equilibrium and authenticates no referenced artifact,
citation, model identity, execution timestamp or reported force/geometry metric.
Campaign locations retain shared lexical URI policy without fetching content.

`write_vmec_reference_report` protects selected input aliases before creating
parents and writing sorted UTF-8 JSON. API failures propagate; the standalone
script and registered `validate-vmec-reference` command turn supported operational
failures into fixed authored refusal. Sequential inspection/write observations
do not guarantee a transaction against concurrent pathname changes.

::: validation.validate_vmec_reference

Declaration predicates live in `validation.vmec_reference_contracts`; Fourier
integer domains and declared error limits do not prove numerical convergence.


### EPED reference declarations

`validation.validate_eped_reference.validate_eped_reference` reports persisted
schema `scpn-control.eped-reference.v1` identity, provenance, unit, grid, shape
and declared tolerance findings. `canonical_artifact_sha256` retains the
original sorted ASCII compact body hash excluding `payload_sha256`; comparison
binds the supplied body without producer or referenced-byte authentication.
Four artifact URI fields retain prefix/relative-path lexical checks, without
URL parsing or retrieval. No pedestal physics is computed.

`write_eped_reference_report` protects selected input aliases before creating
parents and writing sorted UTF8 JSON. The actual script and registered
`validate-eped-reference` command return authored findings and fixed operational
refusals. Declaration identity/provenance and geometric/metric checks live in
`eped_reference_contracts` and `eped_reference_geometry`. See the EPED section
of [validation](validation.md) for exact numeric domains and examples.

::: validation.validate_eped_reference


### MARFE reference declarations

`validation.validate_marfe_reference.validate_marfe_reference` checks persisted
schema `scpn-control.marfe-reference.v1` identity, unit, source, scan, impurity,
geometry, power and declared error-bound consistency. The original public
`canonical_artifact_sha256` uses sorted compact ASCII body serialisation
excluding `payload_sha256`. A matching body hash does not authenticate its
producer or four referenced artifacts. URI prefix/relative-path checks retain
the original lexical rules without URL parsing or retrieval.

`write_marfe_reference_report` protects selected input aliases and writes sorted
UTF8 output through API, actual script and registered
`validate-marfe-reference` CLI. Identity/provenance checks and declared numeric
domains live in `marfe_reference_contracts` and `marfe_reference_domains`.
See [validation](validation.md) for precise domains, operational refusals and
executable examples. No radiation model or density-limit metric is recomputed.

::: validation.validate_marfe_reference


### NTM reference declarations

`validation.validate_ntm_reference.validate_ntm_reference` checks persisted
schema `scpn-control.ntm-reference.v1` identity, source, units, rho/q profiles,
rational-surface metadata, seed widths, ECCD and declared error bounds.
`canonical_artifact_sha256` retains original sorted compact ASCII body hashing
excluding `payload_sha256`; it does not authenticate its producer or four
referenced artifacts. URI prefix/relative-path checks are lexical, without
parsing or retrieval. No island trajectory, rational-mode equality, q
interpolation or ECCD effect is computed.

`write_ntm_reference_report` protects selected input aliases and writes sorted
UTF8 output through API, actual script and registered `validate-ntm-reference`
CLI. Cyclic output paths receive fixed CLI operational refusal. Predicates live
in `ntm_reference_contracts` and `ntm_reference_domains`. See
[validation](validation.md) for exact domains, limits and executable examples.

::: validation.validate_ntm_reference

## Volt-second reference declarations

The public reader validates original schema `1.0` metadata and declared error
bounds. Reference SHA format and URI/citation presence do not authenticate
external bytes or physics. The writer protects selected input aliases; both
registered and script CLIs provide authored operational refusal. See the
[validation guide](validation.md) for original numeric domains, unit labels,
selection and exit behaviour, and an executable report-persistence example.

::: validation.validate_volt_second_reference

::: validation.volt_second_reference_contracts

## Burn-control reference declarations

The reader checks original schema `1.0` identity, provenance presence and declared
numeric bounds. Format-only reference digests authenticate no bytes or burn
physics. The public writer protects selected input aliases; both CLI surfaces
provide authored operational refusal. The [validation guide](validation.md)
describes original units, domains, selection, exits and executable persistence.

::: validation.validate_burn_reference

::: validation.burn_reference_contracts


### RZIP reference declarations

The reader checks original schema `1.0` identity, distinct source URI policies
and declared numeric bounds, including signed vertical field index. Reference
SHA syntax authenticates no bytes or RZIP physics. The public writer protects
selected input aliases; both CLI surfaces provide authored operational
refusal. The [validation guide](validation.md) describes original units,
domains, selection, exits and executable persistence.

::: validation.validate_rzip_reference

::: validation.rzip_reference_contracts


### Disruption reference declarations

The reader checks original schema `1.0` identity and source presence, signal
sample/timing domains, nonnegative inventories, bounded mitigation strength
and declared error bounds. Format-only SHA declarations authenticate no bytes
or mitigation physics. The public writer protects selected input aliases;
both CLI surfaces provide authored operational refusal. The
[validation guide](validation.md) describes units, domains and executable
persistence.

::: validation.validate_disruption_reference

::: validation.disruption_reference_contracts


### Digital twin reference declarations

The reader checks original identity/source presence, grid and actuator domains,
units and declared error bounds. SHA declarations authenticate no reference
bytes, IDS export or physical twin simulation. The writer protects selected
input aliases and both CLI surfaces return authored operational refusal. The
[validation guide](validation.md) supplies domains and executable persistence.

::: validation.validate_digital_twin_reference

::: validation.digital_twin_reference_contracts

::: validation.digital_twin_reference_domains


### Free-boundary reference declarations

The persisted reader checks original identity and source presence, uncapped
equilibrium counts, positive control interval/slew, units and declared tracking
bounds. Format-only SHA declarations authenticate no reference bytes or
facility control. The writer protects selected inputs; both CLI surfaces
provide authored operational refusal. The [validation guide](validation.md)
provides domains and executable persistence.

::: validation.validate_free_boundary_reference

::: validation.free_boundary_reference_contracts


### SOL blob reference declarations

The reader retains the named schema, lexical artifact URI policy, canonical
body consistency hash, geometry/coordinate/zero-lower pair domains and declared
error bounds. Hash consistency authenticates no producer or referenced bytes.
The input-protected writer and both CLI surfaces expose authored operational
refusal; the [validation guide](validation.md) includes executable persistence.

::: validation.validate_blob_transport_reference

::: validation.blob_transport_reference_contracts

::: validation.blob_transport_reference_domains


### Neural transport reference declarations

The persisted reader retains the named schema, exact QLKNN feature/target
order, lexical artifact and executable policies, canonical body consistency,
uncapped sample count and declared error/score domains. Hash syntax and body
consistency authenticate no producer, weights or referenced bytes. The writer
protects selected input aliases and both CLI surfaces return authored
operational refusal. The [validation guide](validation.md) includes executable
persistence and the distinct numeric and source domains.

::: validation.validate_neural_transport_reference

::: validation.neural_transport_reference_contracts

::: validation.neural_transport_reference_domains

### Stored neural turbulence reference declarations

The public reader retains gyroBohm units, critical-gradient score domains and
campaign/public citation presence. Hash declarations check exact hex shape; the
separate admission builder additionally matches supplied weight bytes without
recomputing independent metrics. The public writer protects selected aliases.
See the [validation guide](validation.md) for executable persistence and limits.

::: validation.validate_neural_turbulence_reference

::: validation.neural_turbulence_reference_contracts

::: validation.neural_turbulence_reference_domains

### Stored SOC lattice and learning declarations

The reader retains source citation presence, lattice/learning metadata and
declared error bounds without running learning or fetching reference bytes.
Metadata zero critical slope remains distinct from runtime constructibility.
The public writer protects selected input aliases; the [validation guide](validation.md)
includes executable persistence and exact scalar/count/provenance domains.

::: validation.validate_soc_reference

::: validation.soc_reference_contracts

::: validation.soc_reference_domains

### Stored ELM crash and RMP declarations

The reader preserves lexical URI and provenance presence rules, compact ASCII
body consistency, grids, independent time windows and inclusive Type-I energy
fractions. Public report persistence protects selected inputs. The
[validation guide](validation.md) includes an executable example and the boundary
between stored declarations and measured physical comparison.

::: validation.validate_elm_reference

::: validation.elm_reference_contracts

::: validation.elm_reference_domains

### Stored static structured-mu declarations

Original zero-frequency labels, provenance presence and declared error bounds
remain. The current/deprecated public entrypoints share input-protected report
persistence; the [validation guide](validation.md) includes an executable example
and the boundary to independently validated claim admission.

::: validation.validate_static_mu_analysis_reference

::: validation.validate_mu_synthesis_reference

::: validation.static_mu_reference_contracts

::: validation.static_mu_reference_domains


### Stored Miller geometry reference comparisons

The reader binds exact reference bytes and retains the original four-angle
local Miller comparison, required cases, numeric domains and tolerances.
Invalid physical/numeric inputs become authored findings. The public writer
protects single-reference aliases; both CLI surfaces return authored operational
refusal. The [validation guide](validation.md) includes executable persistence,
report metadata and the boundary with independent numerical comparison. Both
public comparisons use `jacobian=m`, `g_rt=m-1` and `g_tt=m-2`; corrected
metadata changes fresh report digests while historical reports retain their bytes.

::: validation.validate_gk_geometry_reference

::: validation.gk_geometry_reference_contracts

::: validation.gk_geometry_reference_cases

### Stored species and collision reference comparisons

The public reader and writer bind exact input bytes, protect reference aliases
and preserve bounded species/operator comparisons. Fresh larmor coefficient
metadata uses `m*T`; physical equations and historical report bytes remain.
See the [validation guide](validation.md) for scalar/coercion domains, resource
limits, report digests and executable persistence.

::: validation.validate_gk_species_reference

::: validation.gk_species_reference_contracts

::: validation.gk_species_reference_cases

::: validation.gk_species_reference_operators

::: validation.gk_species_reference_numeric


### Stored linear GK cross-code declarations

The public reader checks author-declared identities, digests, units and scalar
discrepancies. Body consistency grants no binary execution or physical/control
admission. The public writer protects selected input aliases before persistence;
script and registered CLI routes give fixed operational refusals. The
[validation guide](validation.md) specifies original bounds, selection, decimal
representability and concurrent-path limitations, with an executable example.

::: validation.validate_gk_crosscode

::: validation.gk_crosscode_reference_contracts

### Stored GK OOD campaign declarations

The reader binds exact captured bytes and canonical author metadata. It checks
finite distribution values, positive detector thresholds and inclusive `[0, 1]`
rate domains before the original acceptance comparisons. Covariance identity
and positive-definite status are declarations; no matrix is fetched or checked.
The original `deployment_calibration_admitted` flag records accepted metadata,
and installs no detector calibration or physical/control deployment. The writer
protects selected input aliases. The [validation guide](validation.md) specifies
selection, fixed refusals, duplicate campaigns and the executable inspection.

::: validation.validate_gk_ood_calibration

::: validation.gk_ood_reference_contracts

::: validation.gk_ood_reference_domains


## Declared GK interface artifacts

The reader inspects declared interface metadata, the public canonical hash
preserves author-body consistency, and the writer protects selected input
aliases. Accepted metadata grants no independently verified solver/parser run
or full cross-code claim. The [validation guide](validation.md) specifies
selection, numeric/URI limits, actual-byte binding and timestamped reports.

::: validation.validate_gk_interface_artifacts

::: validation.gk_interface_reference_contracts


## Persisted JAX GK parity reader and protected reports

The declaration domains, original artifact comparisons and required-coverage
aggregates are documented separately from numerical solver execution. The
[validation guide](validation.md) gives an executable empty-input example.

::: validation.validate_jax_gk_parity

::: validation.jax_gk_parity_contracts

::: validation.jax_gk_parity_domains

::: validation.jax_gk_parity_summary


## Density declaration domains and protected report API

`write_density_reference_report(report, output_path, artifact_root=...)` protects
root/immediate selected direct, resolved, symlink and existing hardlink aliases
before any write. Other output may replace; nonfinite report JSON is refused.
Checks are sequential without locks, snapshots or an atomic write guarantee.
Fixed decoder findings never expose private duplicate member or exception text.
The [validation guide](validation.md#density-declaration-report-persistence) has
an executable absence example; no reference or model result is fabricated.

::: validation.validate_density_reference

::: validation.density_reference_contracts

::: validation.density_reference_domains


## Current-drive declaration domains and protected report API

The declaration reader retains its original schema, identities, inclusive metrics,
unit labels and positive source/grid rules. Nonzero binary64 underflow refuses
globally. Original authored duplicate/nonfinite/read/JSON findings remain unchanged.
No referenced digest authentication, external solver or genuine positive physics
reference is added. Public writer refuses nonfinite report serialisation and selected
input aliases before writing; API failures propagate. Original script write refusal
remains2 with fixed text; root ClickException1 retains Error prefix. Directory/NUL
output is Click usage2. Checks are sequential without locks or snapshots.

::: validation.validate_current_drive_reference

::: validation.current_drive_reference_contracts

::: validation.current_drive_reference_domains


## Vacuum software diagnostic reports

`validation.benchmark_free_boundary.run_free_boundary_benchmark()` creates three
fresh real kernels on a fixed 65x65 grid and returns raw diagnostics. Its
single-coil expression copies the current vacuum solver's grouping, so equality
checks implementation consistency. It provides no independent physical
normalisation witness. The Helmholtz field sample is at R=0.5, Z=0 while the
reference is on R=0, which the grid excludes. No same-point tolerance is applied;
the API preserves a legacy unconditional qualitative `pass=True` marker.

The public report writer `main()` records that unassessed Helmholtz comparison
as `pass=null` and `assessment="diagnostic_only_off_axis_sample"`. Markdown shows
N/A. Existing numeric values and other flags retain their original arithmetic.
The X-point entry is a whole-grid gradient-norm minimum with a Z-only threshold;
it does not verify a saddle/null and its expected R=0 is outside the grid.

`main()` parses no CLI parameters and writes caller-relative
`validation/reports/free_boundary_benchmark.json` and `.md` sequentially. The
module-relative campaign persistence scope requires an identifier for guarded
paths, while outside-root scratch paths are allowed. IO/model errors propagate
and can leave partial reports. The diagnostic API uses a temporary config and
cleans it after the computation; allocation/initial-write errors precede that
cleanup block. These APIs do not establish axis agreement, validated topology
or facility admission. See the [software example](benchmarks.md#vacuum-diagnostic-report-software-check).

::: validation.benchmark_free_boundary

## Bounded burn, volt-second and orbit report producers

The three fixed `main()` producers below use fresh local model state, parse no
CLI overrides and write sequential UTF-8 JSON/Markdown under their module's
`reports` directory after the campaign ID guard. Their guard checks identifier
presence/syntax; the actual recorded wrapper separately preserves custody.
The writes have no transaction, atomic replacement or producer locks, so errors
can leave partial output. Burn and volt-second write the same JSON both before
and after their Markdown write. See the [temporary copied-output example](benchmarks.md#bounded-particle-report-software-check).

Burn fixes 48 radial samples, density in 1e20 m^-3, ion/electron temperatures
in keV, confinement time 3.7 s and auxiliary power 50 MW. Its defining builder
computes profile diagnostics and a single controller step with dt=0.1 s; no
burn trajectory is advanced. `reactor_claim_allowed` remains false.

Volt-second fixes a 120 V s budget, plasma inductance 1.2 microhenry, resistance
0.08 microohm, 80/400/60 s ramp/flat/down durations and 15/4 MA plasma/bootstrap
currents. Its output records scalar scenario accounting. It does not integrate
measured voltage or evolve a radial current profile; facility admission stays
closed.

Orbit calls a first-loss estimate at the fixed geometry and separately declares
q=2, rho_L=0.05 m and epsilon=0.25 for a banana-width estimate. Its six passing,
two trapped and two lost particles are a fixed `EnsembleResult` fixture with
loss fraction 0.2. No trajectory ensemble or collision simulation produces those
counts. `external_orbit_claim_allowed` remains false. Provenance labels identify
declared fixtures without authenticating physical reference bytes.

::: validation.benchmark_burn_control_claims

::: validation.benchmark_volt_second_claims

::: validation.benchmark_orbit_following_claims

## Bounded density and current-drive report producers

`validation.benchmark_density_control_claims.main()` and
`validation.benchmark_current_drive_claims.main()` produce fixed repository
declarations. Both use fresh model state, parse no CLI parameters, require a
recorded-campaign identifier for their fixed module-relative report paths, and
propagate model and filesystem errors. The identifier check proves presence
and syntax; the recorded wrapper separately captures immutable output custody.
Neither establishes authentic facility references or independently recomputed
external comparisons.

The density producer advances one imposed gas-source model step with requested
dt=1 s and the defining CFL clamp. It separately computes a controller command
from the initial profile; that command is not applied to the resulting profile.
The report grants no facility density claim. The current-drive producer samples
80 normalised radial points using density in 1e19 m^-3 and temperatures in keV,
with 8/3/14 MW ECCD/LHCD/NBI powers. It grants no external deposition claim.

Reports use UTF-8 sequential JSON/Markdown writes under `validation/reports`.
Current-drive writes the same JSON both before and after its Markdown write.
Producer writes have no transaction, atomic replacement or locks; partial output
can remain after failure. The wrapper's reservation/custody behaviour is distinct
from these writes. See the [temporary output example](benchmarks.md#bounded-particle-report-software-check)
and the [recorded current-drive command](validation.md).

::: validation.benchmark_density_control_claims

::: validation.benchmark_current_drive_claims

## Manual nonlinear JAX GK experiments

`tools.em_and_dimits.run_jax(label, **overrides)` creates a fresh solver and
returns saved diagnostics. Its defaults request a large 5000-step run; supply
explicit small grids and step counts for a software boundary check. The runner
uses the configured JAX backend with NumPy fallback disabled. An empty history
raises `RuntimeError`; a nonfinite transport display becomes `None`.

The reported `converged` flag means more than one finite flux sample. Endpoint
`late_growth` is a fractional difference per solver code time, rather than a
logarithmic growth rate. Neither field establishes a physical Dimits shift or
asymptotic convergence. The API prints diagnostics and writes no file. Its
fixed five-case `main()` requests three 5000-step and two 10000-step experiments,
then replaces a caller-relative raw JSON report under `gpu_results` using the
legacy serialiser. It provides no campaign, digest or atomic-write custody.

See the [short CPU example](benchmarks.md#manual-nonlinear-gk-software-check).

::: tools.em_and_dimits

`tools.dimits_256_fixed.run(label, rlt)` exposes the separate fixed 256-kx,
10000-step comparison with hyper coefficient 0.02; only its label and drive
are parameters. Its `main()` runs drives 3.0 and 6.9, writes
`gpu_results/dimits_256_fixed.json`, then prints a comparison. An empty history
can make that comparison fail after the report write. The diagnostic kx print
depends on the current solver's private NumPy grid implementation.

`tools.dimits_long.main()` fixes a 128-kx, 20000-step grid with hyper coefficient
0.2 and the same two drives. It writes `gpu_results/dimits_long_3_vs_69.json`
after both cases; an empty history causes endpoint indexing to fail before
that final write. These two entry points provide no short-grid CLI overrides.
Both use the legacy JSON/nonfinite convention and caller-relative replacement,
without campaign custody or physical admission. Their histories, drive ratio,
wall-clock versus solver time, and finite-sample flag have the diagnostic
limits described above.

::: tools.dimits_256_fixed

::: tools.dimits_long

## Manual native component timing probes

`tools.benchmark_full_stack.benchmark_full_stack()` prints three independent
in-memory probes and returns `None`. It has no registered root command or
JSON evidence output; the benchmark producer registry classifies it as
`stdout_or_build_product`. It requires an importable real `scpn_control_rs`
extension, NumPy and the Python compiler/controller dependencies; no build,
package installation or backend fallback occurs. Missing extension import fails.

The first probe reads caller-relative `iter_config.json`, prints native solve
wall time/iterations/residual and does not assert convergence. Its exception is
printed while the subsequent probes continue. The controller exports an actual
`actions` readout with the original positive/negative places, one gain/limit/slew
entry and explicit Rust backend. Ten warmups precede 1000 fixed synthetic input
ticks. The last probe calls the native Kuramoto binding once with800 phases and
frequencies from NumPy's global RNG; its L16/N50 label denotes total count.
The three components exchange no solved state or feedback. Controller/compiler/
oscillator errors propagate; no artifact is saved or actuator invoked. Timings
include call overhead and give no controlled regression or physical/facility/
closed-loop acceptance. See the [validation guide](validation.md#manual-native-component-timing).

::: tools.benchmark_full_stack


## Metadata report destinations and selected inputs

The source-tree JSON commands 'validate-data-manifests' and
'validate-physics-traceability', their standalone Python scripts, and the
physics traceability Markdown generator check the report destination before
creating its parent directories or writing bytes. Direct/resolved paths,
symbolic links including linked parents, and existing hard links must not alias
selected input paths. Unrelated existing report files may still be replaced.

Manifest reports protect the root and discovered manifest/specification/required
DIII-D artifact paths. Metadata-valid manifests also contribute local artifact
URIs through the defining contained evidence-root resolver, without hashing or
fetching remote bytes. Invalid metadata still protects its own manifest path;
unresolvable artifact names add no path. This additional sequential discovery
provides no coherent snapshot, locking, atomic replacement or protection against
concurrent pathname changes. Physics writers protect the selected registry;
this does not extend to every source/evidence path declared inside that registry.

Output refusals retain standalone exit1. JSON scripts/data Click output retain
FAIL diagnostics; the physics Click command now reports a fixed operational
'Error: could not write physics traceability report' with exit1. CLI options and
help text are preserved. This boundary protects input bytes; it authenticates
neither report custody, artifact provenance nor physical/scientific admission.
The [validation guide](validation.md#metadata-report-input-custody) binds actual
public alias tests to these source-tree entry points.

::: validation.report_output_paths

The repository-local `validation` initialiser exposes no eager submodule API.
Select the defining reader, writer or campaign entry point for its documented
inputs, output/error behaviour and evidence limitations.

::: validation

### Declared benchmark regression policy and command

`tools.benchmark_regression_gate` preserves the Python comparison API while
separating ratio/domain policy from diagnostic verdict assembly. It compares
already recorded declarations; no source authentication, host measurement,
cross-language execution or physical admission follows from its passed flag.
The command protects selected report/baseline/policy input aliases and refuses
supported read/write failures even in evidence-only mode. See
[baseline inspection and command boundaries](onboarding.md#interpreting-benchmark-gate-verdicts)
for executable usage and [regression policy](benchmarks.md) for declared metrics.

::: tools.benchmark_regression_gate

::: tools.benchmark_gate_policy

::: tools.benchmark_gate_verdict

## Bounded UQ, kinetic-EFIT and tracking report producers

The public UQ build_reference_scenario() returns fresh mutable inputs in MA,
T, MW, 1e19 m^-3 and metres, with dimensionless geometry/fuel fractions and
ion mass in AMU. main() samples 256 model-proxy draws with local seed 31 and
writes JSON then Markdown. It supplies no calibration witness or PDE solve.

Kinetic build_reference_case() returns a fresh reconstruction, constraints
and fast-ion model; the reconstruction references the same supplied objects.
Diagnostics at (6, 1) m are inside its 33-by-33 grid. main() supplies empty
magnetic measurements, computes 50-point synthetic pressure profiles in Pa
and derives q from prescribed MSE pitch. Temperatures/fast-ion energy use keV,
density uses 1e19 m^-3 and pitch uses degrees. The defining reconstruction's
fixed chi-squared, iteration-count and wall-time fields are not measurements.

Tracking main() uses the shipped fixed 8-by-4 response fixture, running five
actual controller steps without early stopping. Its zero Psi grid is not a
magnetic equilibrium; stored flux/current units follow consumer conventions
without a physical calibration witness. The factory ignores the config path.
No reference artifact is supplied by any of these producers and their
calibrated/facility flags remain False.

All three parse no CLI parameters and write fixed paths beside their source
under reports. Kinetic and tracking write JSON, Markdown, then the same JSON
again; UQ writes each once. UTF-8 sequential writes may leave earlier output
after an OSError, with no producer lock or transaction. Invalid campaign IDs
and defining evidence inputs raise ValueError; an absent persistent campaign
ID raises RuntimeError before report creation. The campaign ID presence/syntax
guard is separate from the recorded wrapper's reservation and byte custody.
The [temporary copied-output example](benchmarks.md#bounded-uq-and-equilibrium-report-software-check)
exercises their real public commands without replacing scientific reports.

## Formal validation command contracts

The Lean command accepts a required report and optional --artifact and
--formal-report-root, prints sorted JSON and returns zero only for pass/no
errors. Missing initial report I/O becomes fail/no digest; later report or
artifact I/O and native parsing failures may propagate. It writes no files.
The named report digest hashes the first raw observation; semantic loading
rereads the path. Both report and artifact are data declarations unless their
proof/source generation has a separately verified witness.

The Z3 publisher builds the fixed source/sink/move net and actually checks
two-firing bounded marking and temporal obligations with the installed solver.
Explicit --json-out/--markdown-out paths use caller cwd; defaults point into
checkout validation/reports. JSON then Markdown are sequential overwrites,
without alias checks, locks or campaign authentication. Caught RuntimeError
writes a blocked report; --require-z3 returns nonzero afterward. A permitted
blocked result returns zero without proving the model. Output errors can leave
the earlier file. Bounded holds is separate from hardware, PCS and unbounded
proof admission.


## Local E2E latency declaration reader

`validation.validate_e2e_latency_evidence` retains the public builder,
validator, result class and compatibility helper aliases. Canonical payload
fields/numeric checks live in `validation.e2e_latency_payload`; context and
explicit UTC checks live in `validation.e2e_latency_context`. The builder
returns a shallow copy, fixes schema and local boundary strings, preserves
explicit class/production declarations and recomputes a parsed-payload checksum.
It does not measure timing or qualify the declarations.

`validate_e2e_latency_evidence(path, require_target_hardware=True,
max_e2e_p95_us=None)` reads caller-relative UTF-8 JSON and returns frozen
`LatencyEvidenceReport`. Report field failures return status `fail` and
ordered diagnostics; p95 and trimmed labels remain available when valid.
The optional budget is finite, nonnegative microseconds, inclusive at equality;
None omits that comparison and zero rejects every valid positive p95.
Invalid budgets, including booleans, NaN, infinities and conversion overflow,
raise ValueError before report I/O. UTC must be timezone-aware with zero offset
(Z or +00:00); old timestamps are allowed because freshness is not checked.

Target strings reject known placeholders; nonplaceholder strings remain
declarations. A pass does not authenticate hardware, samples, clock, isolation,
source revision or operator approval. production_claim_allowed must remain
False. The digest is canonical sorted compact ASCII JSON without its own field,
not a raw-file hash or signature. Duplicate JSON keys retain json.load's last
value; unknown fields remain checksum inputs. Native I/O/UTF-8/JSON and legacy
large-integer/serialisation errors propagate. The reader does not mutate files.

The standalone command takes a report, `--allow-local-unqualified`,
`--max-e2e-p95-us` and `--json-out`; it prints the existing six result
fields or text diagnostics, writes no report, and exits 0/pass, 1/field refusal,
2/argparse syntax. Uncaught API exceptions produce script failure. The temporary
real-producer example is maintained in the E2E section of docs/benchmarks.md.


## Manufactured mesh and differentiable latency readers

`validation.mesh_convergence_study.run_solovev_benchmark(nr, nz,
max_iter=25000, tol=1e-10)` solves its own manufactured Dirichlet problem on
R=[1,3], Z=[-1.5,1.5]: psi=R^4/8+Z^2/2 and source=R^2+1. Independent
row-vector SOR uses omega1.2, exact edges and zero interior. It does not invoke
FusionKernel. nr/nz need a nonempty interior (>=3); native errors propagate for
invalid grids. max_iter is a Python range cap; zero/negative performs no sweep
and reports iterations=0. Residual is inspected at sweep indices0,200,... and
compared strictly to tol. Cap exhaustion has no separate convergence flag.
The result carries interior RMSE/max error, NRMSE normalised by full-grid exact
range, radial h, count and sweep-only wall seconds; these are mathematical
benchmark scales, not discharge calibration.

Mesh `main()` has no parser: it runs17/33/65/129 with defaults, computes
adjacent-grid rates and sequentially overwrites caller-relative JSON/Markdown
under validation/reports. First row has no rate. No campaign guard or atomic
pair write exists; use isolated cwd and retain software-only custody.
The fixed public study and original17/33 convergence regression are exercised;
legacy NaN termination is not forced with mocks for coverage.

`validation.validate_differentiable_transport_latency` keeps its public API,
constants and helper aliases. fields/audit/context/reports/readiness modules
under `validation.differentiable_latency_*` own JSON, domains, sampled
declarations and status construction. The public reader takes one_step_report,
optional rollout_report, keyword readiness_report and require_admitted.
Paths use caller cwd; CLI defaults use repository validation/reports.

Reports are local declarations: schema1, backendjax, dtypefloat64, four channels,
positive counters, finite nonnegative ordered p50/p95/max milliseconds, runtime
metadata and sampled audit fields. Fixed integer schema/counter fields refuse
boolean/float coercion; huge nonrepresentable numeric declarations become
field refusals. Each invalid entry has statusfail and is excluded from admitted
counts. Valid authored no-backend declarations are blocked and count only when
their own fields pass. require_admitted refuses blocked latency reports; it
does not require readiness or full_fidelity_ready=True. Aggregate readiness is
False on any aggregate error; the separately returned readiness entry describes
its own field validation.

A true readiness declaration additionally requires declared external admission\nand non-null external-reference/controller digest declarations, matching the\ndefining producer's prerequisites. Readiness SHA fields are syntax checks, not\nbinding to supplied file/audit/
campaign/reference/proof bytes. full_fidelity_ready describes validated metadata
and never authenticates promotion. Runtime strings/device/x64/timestamps are
also declarations; no host verification or freshness check occurs. The reader
executes no JAX audit or benchmark, writes no artifacts and contacts no source.
Duplicate keys fail at all JSON depths. I/O/UTF-8/JSON/root errors become per-file
json diagnostics; excessive nesting/native path errors can propagate.

The command prints JSON or status/counts/error lines and exits0/pass,1/fail,
2/parser syntax. Actual installed-JAX observation tests and exact temporary
examples are maintained in docs/benchmarks.md; authored derivatives are clearly
reader inputs and never new timing/reference/proof measurements.


### Synthetic risk campaigns and phase runtime producer

::: validation.benchmark_kuramoto_runtime_evidence.main

::: validation.control_resilience_campaign.generate_campaign_report

::: validation.control_resilience_campaign.render_markdown

::: validation.control_resilience_campaign.main

::: validation.disruption_roc_analysis.generate_scenario_batch

::: validation.disruption_roc_analysis.evaluate_batch

::: validation.disruption_roc_analysis.main

Kuramoto CLI phases are radians and frequencies/couplings are rates with a
seconds timestep. It writes one Python one-step/two-half-step refinement
comparison and checks the actually imported optional Rust kernel. Persistent
evidence destinations require recorded-runner custody; the default claim
remains bounded. A deployment request additionally requires native parity and
target count/refinement checks.

Resilience normalises the legacy integer/float inputs before its actual
synthetic fault campaign. Bit flips affect binary64 mantissa bits; risk errors
and tolerances are dimensionless, while recovery offsets are samples.
Generation timestamps and measured duration vary; fixed-seed campaign metrics
repeat. The renderer formats an existing mapping without validation or I/O.
Strict exit two follows both report writes, whereas non-strict exit zero can
accompany failed synthetic thresholds.

The synthetic ROC generator records the absolute index of its simulator's
observed terminal threshold crossing. The simulator's separate elapsed
trigger offset is not an absolute event index. Evaluation samples128-point
past windows every20 steps and only accepts alarms strictly before the
declared event; no validated physical warning time follows. Missing-class
denominators yield zero rates. Literal threshold comparisons accept NaN as
no detections, so callers must provide a meaningful finite threshold.


### Declared transport profile comparisons

::: validation.code_to_code_benchmark.main

::: validation.code_to_code_local.run_local_transport

::: validation.code_to_code_scenario.validate_scenario

::: validation.code_to_code_scenario.initial_profiles

::: validation.code_to_code_comparison.compare_transport_profiles

::: validation.code_to_code_comparison.verify_payload_digest

::: validation.code_to_code_reports.build_comparison_report

::: validation.code_to_code_torax.write_torax_config

These source-checkout APIs use the actual local transport solver and separate
profile arithmetic, report declarations and optional-provider preparation.
R0/a are metres, B0 tesla, I_p amperes, P_aux MW, temperature keV and local
density 10^19 m^-3; TORAX configuration density is converted to m^-3.
Strict integer n_rho sets the actual radial length. Fixed dt is seconds and
t_final must equal n_steps*dt; zero steps/zero final time observes initialisation.

Both adapters now receive linear axis-to-edge temperature/density inputs.
D/T each start at half the local electron density and helium at zero.
The local adapter does not apply B0/delta, and its gyro-Bohm transport, boxed
current/geometry and source channels differ from TORAX's configured models.
These concrete differences keep physical-reference admission blocked even
when declared profile metrics are numerically valid.

Comparison interpolates the reference on the actual local rho coordinates,
including equal-length grids with different coordinates. Rho must be finite,
strictly ordered in [0,1], match each supplied profile length and cover the
local comparison domain. Missing coordinates are refused; endpoint
extrapolation is refused. RMSE is equal-weighted per local sample, in keV.
Numeric strings, booleans and nonrepresentable scalars/differences are refused.
Optional ion metrics require both ion profiles; full reference payload checks
require both temperature channels.

Schema v3 binds normalised scenario/report declarations to SHA-256 and exposes
diagnostic_comparison_available separately from always-blocked physical
admission under the current configured model contract. Recomputed self-digests
cannot authenticate an external provider. Reader fixtures derived from local
observations are declared arithmetic inputs and never TORAX execution evidence.

main checks output aliases against each other, its defining sources and the
legacy optional configuration before computation. Persistent evidence roots
require recorded-runner custody. JSON/Markdown use sequential UTF-8 overwrites
without a pairwise transaction; required-external exit one follows both writes.
The actual local config uses a unique caller-cwd directory and is cleaned up.
Standalone TORAX configuration export uses exclusive file creation and refuses
an existing destination. The legacy optional runtime path remains dependent
on the actual installed provider API; missing TORAX returns an explicit block.

### Synthetic disturbance rejection API

The source-checkout facade in validation/benchmark_disturbance_rejection.py
retains PIDController, MPCController, SNNControllerWrapper, LinearPlant,
SCENARIOS, ScenarioMetrics, TraceData, run_scenario and the public report
renderers. Defining contracts live in the
[controller module][validation.disturbance_controllers],
[runtime module][validation.disturbance_runtime],
[input module][validation.disturbance_inputs] and
[report module][validation.disturbance_reports].

::: validation.disturbance_controllers

::: validation.disturbance_runtime

::: validation.disturbance_inputs

::: validation.disturbance_reports

The plant evolves position metres and velocity metres/s using explicit Euler:
z'' = 100² z - 10 z' + u + 0.5 d, with u/d in m/s². Scenario labels select
acceleration forcings; no ITER geometry, density, beta_N or energy state is
implemented. Gains, state vectors, outputs and derived metrics must remain
finite. Boolean/string numeric declarations and truncated count inputs refuse.
A duration is a positive integer number of fixed dt intervals. Controller
instances are mutable and must not be stepped concurrently.

run_scenario resets a real controller and returns aligned initial/terminal
samples of length completed_steps+1. An unstable trajectory stops at its first
|z|>10 m; no fabricated tail is returned. The last control slot repeats the
last held action for plotting, without adding an interval. ISE integrates
recorded squared error in m² s; effort sums |u| over executed intervals in m/s;
peak_overshoot is maximum absolute error in metres. The settling band is
threshold × max(initial error, first 100 observed absolute errors, 0.001 m).
settled is separate from stable and means only settling over the observed
finite horizon. Requested/completed durations and stop reasons accompany rows.

MPCController retains the original q_weight, r_weight and iterations arguments
and mutable cost/iteration attributes. Weights and iteration counts are checked
at construction; callers remain responsible for valid later attribute changes.
HInfinityErrorController converts target-minus-position error to the actual
DGKF positive measurement convention. Its nonzero-target tracking is not
admitted. The local MPC retains an approximate one-step-sensitivity gradient
and zero action initialisation on each call; no optimal MPC claim is made.
SNN reset reconstructs the real SC-NeuroCore pool, including cells and RNG;
legacy NumPy fallback and quantum entropy are disabled. Plant dt does not
control the provider neuron clock, and the provider uses its own neuron seeds.
The independent Rust PID/MPC/SNN components are different numerical contracts;
this Python benchmark is not their native performance comparison.

main(output_dir=None, *, duration_scale=1.0, require_complete=False) retains
its legacy None return. Temporary caller outputs are allowed; persistent
artifacts require recorded campaign custody. JSON/Markdown/PNG source or output
aliases refuse before computation. Writes remain sequential, without a lock
or atomic multi-file transaction. Plot figures are closed when rendering or
writing fails; plots written before that failure remain. Schema v2 names actual/missing controllers,
actual runtime backend and observed horizons. cli(argv=None) exposes the real
parser. Ordinary exit zero means reports were written; --require-complete
exits two after reports for missing controllers or incomplete bounded runs.
No physical-reference, facility, safety or controlled speedup admission follows.

## Free-boundary tracking acceptance diagnostic

`validation.free_boundary_tracking_acceptance` preserves `run_campaign()`,
`generate_report()`, `render_markdown(report)`, shared mutable threshold
dictionaries and fixed sweep declarations. The command facade delegates configuration, comparison,
generic/topology sweeps, campaign assembly and rendering to their owning modules.

The full real `FusionKernel` cohort has eighteen four-step scenarios and twelve
sweeps. Its 12-by-12 fixtures set permeability and plasma-current target to 1.0,
sample targets from the same solver and subtract exact supplied measurement
errors in corrected cases. Tracking norms combine configured objectives; flux
residuals follow this normalisation. These outcomes establish local regression
behaviour without independent equilibrium, calibration, actuator or safety proof.

`generate_report()` returns schema v2 with the legacy timestamp, elapsed runtime
and campaign mappings plus an explicit model contract and false physical
reference admission. Elapsed time includes campaign and UTC timestamp
construction, excluding rendering and writes. Numerical exceptions propagate.
Temporary configuration paths in summaries name files removed after execution.
Nested mappings and the shared public thresholds remain mutable; callers must
serialise access and changes themselves. Reports do not authenticate origin.

`render_markdown()` accepts a complete legacy or schema-v2 declaration and
returns Markdown ending in a newline. It neither executes a campaign nor
writes files. Missing/malformed declarations propagate their field/type errors.

```python
from validation.free_boundary_tracking_acceptance import generate_report, render_markdown

report = generate_report()
assert report["physical_reference_admitted"] is False
assert len(report["free_boundary_tracking_acceptance"]["scenarios"]) == 18
assert len(report["free_boundary_tracking_acceptance"]["sweeps"]) == 12
assert "not calibrated SI flux" in render_markdown(report)
```

The compatible `main(argv=None)` checks source/output aliases and recorded
custody before computation. JSON rejects nonfinite values. Outputs replace
unrelated files sequentially without an atomic pair. Temporary destinations
need no campaign; persistent evidence uses the
[recorded runner][tools.run_recorded_benchmark]. The
[validation guide](validation.md) describes its arguments, exit codes and
scientific limits.


## Local source/data ZIP export

The Python tool `tools/export_zenodo_dataset.py` exposes `collect_files(root=ROOT)`,
`create_archive(output=None, *, root=ROOT)` and `main(argv=None)`. It reads the
current working directory contents of the selected checkout, including eligible
untracked public modules. It does not export a Git ref or certify a release.
The default root is the checkout containing the tool; a supplied root is resolved.

`collect_files` returns sorted, unique absolute paths matching `INCLUDE_PATTERNS`:
Python source, tests and tools; the listed examples and validation JSON/CSV;
Markdown documentation; the listed Rust sources/manifests; and the listed root
metadata/configuration files. Python tooling is included because exported tests
import it. Only regular files without symlink path components are selected.
`docs/internal` and `EXCLUDE_DIRS` are excluded explicitly. The actual Git ignore
engine applies current local/global ignore rules with `--no-index`, including
tracked files; inherited Git-root/index environment variables cannot redirect
that engine. An empty initial selection returns `[]`; a nonempty selection needs
working Git. Selection is not an allow-list for all possible confidential content:
only those stated exclusions and current ignore rules are enforced.

`create_archive` returns `(absolute_output_path, member_count)`. The prefix is
`scpn-control-{version}/`, using the nonempty ASCII version declared in a regular
`.zenodo.json`; the rest of each name is a checkout-relative POSIX path. Versions
must start with an ASCII letter/digit and contain only letters, digits, `.`, `_`,
`+` or `-`. This is a naming check, not release, licence or metadata validation.
The function refuses an empty selection and destinations resolving into
`root/.git`. It exclusively creates the output: an existing archive, source,
symlink or hard link is preserved. Output parents must already exist.

Source reads and ZIP writes are sequential without a lock or consistent snapshot.
The archive uses `ZIP_DEFLATED` and source timestamps/permissions; byte-for-byte
reproducibility is not promised. A write failure after creation may leave a partial
candidate. Expected filesystem errors, invalid metadata/version and unavailable
Git propagate from the API as `OSError`/`ValueError`/`RuntimeError`; an existing
output raises `FileExistsError`. The CLI maps these to a fixed refusal and exit 2.
Creation returns 0; argparse help returns 0 and usage errors return 2. Help is
processed before reading metadata. Caller-relative output paths and the default
`scpn-control-{version}.zip` resolve from the caller's cwd.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from tools.export_zenodo_dataset import collect_files, create_archive

selected = collect_files()
assert selected
with TemporaryDirectory(prefix="scpn-control-source-export-") as directory:
    output, count = create_archive(Path(directory) / "candidate.zip")
    with ZipFile(output) as archive:
        assert count == len(selected) == len(archive.namelist())
        assert not any("/docs/internal/" in name for name in archive.namelist())
        assert archive.testzip() is None
```

This fixed-pattern subset does not bundle all runtime dependencies, native
binaries, weights, recovered ML350 arrays, videos, or a complete release/build
configuration. No scientific report admission, training or upload occurs.
The ZIP file-selection surface has no Rust/TypeScript counterpart; the included
numerical implementations are unchanged, and export adds no performance claim.


## Phase-model video

`tools.generate_phase_video.generate(n_ticks, L, N_per, zeta, fps, *,
output_dir=Path("phase-video"), gif_only=False, ffmpeg_path=None)` captures the
actual `RealtimeMonitor.from_paper27` with seed 42, external driver 0, PAC 0 and
model dt 0.001. Positive nonboolean integer counts, finite nonnegative gain and
integer playback FPS in [1,100] are required. Layers beyond 16 repeat the factory
frequency table modulo 16. The numerical model and its optional Rust UPDE remain
unchanged; this Python visualisation has no Rust/TypeScript renderer counterpart.

The output directory is caller-relative or absolute, must be new and must have
an existing directory parent. Existing files, directories and symlinks refuse
before model capture. Defaults require both real Pillow GIF and H.264-capable
FFmpeg MP4. `gif_only=True` explicitly omits MP4 and ignores `ffmpeg_path`;
otherwise the supplied executable path/name or `ffmpeg` on PATH is resolved.
There is no automatic format fallback or encoder installation.

`tools.phase_video_model.capture_trajectory` returns `PhaseTrajectory`, whose
public snapshots are mutable borrowed mappings with consecutive ticks 1..n,
R_global, R_layer of length L, V_global, lambda_exp, boolean guard_approved and
nonnegative latency_us. Capture retains every actual tick in memory. `validate()`
checks finite display fields, R in [0,1] and V in [0,2] with 1e-12 endpoint
roundoff tolerance, positive finite dt and model-error markers. Model guard
HALT is valid diagnostic data. Supplied labels/data are not authenticated or
recomputed by this validation. R/V are dimensionless; the finite-history
exponent is per model-time unit, rather than a full Lyapunov spectrum. Tick
latency measures monitor model/guard work and excludes rendering/encoding.

`tools.phase_video_rendering.render_trajectory(trajectory, output_dir, *, fps=20,
gif_only=False, ffmpeg_path=None)` revalidates snapshots and options, then returns
`VideoResult(gif_path, mp4_path, metadata_path, frame_indices)` with absolute
paths. The original selection is stride=max(1,n_ticks//(fps*10)), starting at
index 0 and appending n_ticks-1 when absent. This targets about ten playback
seconds. Traces use actual ticks 1..n, with visible markers for single samples.
Saved frames include the model-value/guard footer. GIF durations are quantised to centiseconds;
MP4 uses requested FPS, H.264 and bitrate 2000 with Matplotlib encoder arguments.

`phase_video.json` schema `scpn-control.phase-video.v1` records all displayed
metrics, model labels/dt, playback FPS, selected ticks, ideal playback duration,
model elapsed time, filenames and `physical_reference_admitted=false`. Displayed
model PASS/HALT does not establish reactor protection or actuation authority.
The current Python default at tick 500 gives R≈0.154, V≈0.846 and λ≈−0.458;
the retained documentation GIF/MP4 are historical and their earlier R=0.92,
V→0 caption is not reproduced. Monitor latency varies by environment.

Public validation raises `ValueError`; filesystem/encoder lookup or writes raise
`OSError`, and native encoder failure can raise `subprocess.CalledProcessError`.
Writes are sequential GIF, optional MP4, then JSON. A later failure leaves the
new partial bundle; there is no atomic bundle, locking or consistent concurrent
snapshot. Unmanaged Agg figures are released and temporary rc/style settings
restore on success and failure. Matplotlib global rc contexts make concurrent
rendering unsupported. Existing documentation media are preserved.

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.phase_video_model import capture_trajectory
from tools.phase_video_rendering import render_trajectory

trajectory = capture_trajectory(8, 2, 3, 0.5)
trajectory.validate()
with TemporaryDirectory() as directory:
    result = render_trajectory(trajectory, Path(directory) / "model-video", fps=5, gif_only=True)
    payload = json.loads(result.metadata_path.read_text())
    assert result.gif_path.is_file() and result.mp4_path is None
    assert result.frame_indices == tuple(range(8))
    assert payload["sampled_ticks"] == list(range(1, 9))
    assert payload["model_elapsed_time"] == 0.008
    assert payload["ideal_playback_seconds"] == 1.6
    assert payload["physical_reference_admitted"] is False
```

The CLI [validation recipe](validation.md#phase-model-video) states exit codes
and encoder requirements. No numerical comparison benchmark is changed by
this renderer, and its displayed tick latency is not an export benchmark.


## Manual JAX CBC grid study

`tools.gk_convergence_benchmark` exposes `run_benchmark(name, config)`,
`save(results)` and the original fixed `main()`. Import eagerly initialises the
configured JAX backend and prints its devices. Missing JAX refuses import;
CPU is permitted. The caller must arrange the checkout/package dependencies.
`RESULTS_FILE` is a mutable string, initially `/tmp/gk_convergence.json`; an
assigned relative path follows cwd and the parent must exist.

`run_benchmark` borrows a mutable `NonlinearGKConfig` without adding domain/grid
validation and constructs a fresh `JaxNonlinearGKSolver`. Initialisation retains
seed 42 and configured JAX precision/device; no NumPy fallback is requested.
The state axes are species,kx,ky,theta,vpar,mu. `BenchmarkResult` contains four
raw diagnostic fields: `chi_i_gB`, `converged`, `wall_s`, and `Q_i`.
`Q_i` is a list of saved post-step ion heat-flux values in the solver's code
normalisation, with neither times/config nor authenticated physical reference.
`chi_i_gB` is the provider's second-half saved mean divided by
max(R_L_Ti,0.01). Empty histories return 0; one saved sample has an empty second
half and produces NaN, which becomes None. Infinity and nonfinite flux samples
retain their original conventions. `converged` is copied directly: more than
one finite flux sample, without proving saturation, completed integration,
final-state validity or grid convergence. This report cannot admit CBC transport.

The system wall clock brackets construction and `run()`, including allocation,
first-use compilation and synchronised diagnostics; import, result formatting,
printing and writes are excluded. `wall_s` is rounded to one decimal; clock
adjustments, compilation caches, precision, configured device and shared load
prevent treating it as a controlled speed comparison. Provider/configuration,
allocation and backend errors propagate unchanged. This API prints but writes
no file and retains no solver for reuse.

`save` returns None, prints the target and writes a caller-supplied mapping with
legacy `json.dump(indent=2)`, platform text encoding and allow_nan=True. Thus
None becomes JSON null while arbitrary supplied NaN/Infinity remain nonstandard
JSON floats. No input/authenticity/alias/ownership check is added. The existing
file is truncated before serialisation; unsupported or circular input can leave
a partial file. OSError/TypeError/ValueError propagate. Writes are sequential
without atomic replacement or locking. Callers must own their scratch target
and serialise module-global changes. The producer registry classifies this
output as temporary scratch, never canonical validation evidence.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from scpn_control.core.gk_nonlinear import NonlinearGKConfig
from tools import gk_convergence_benchmark as runner

config = NonlinearGKConfig(
    n_kx=8, n_ky=4, n_theta=8, n_vpar=4, n_mu=2,
    n_steps=2, save_interval=1,
)
result = runner.run_benchmark("short actual software check", config)
assert len(result["Q_i"]) == 2 and result["chi_i_gB"] is not None
original_path = runner.RESULTS_FILE
try:
    with TemporaryDirectory() as directory:
        runner.RESULTS_FILE = str(Path(directory) / "raw-gk.json")
        runner.save({"software_check": result})
        assert json.loads(Path(runner.RESULTS_FILE).read_text())["software_check"] == result
finally:
    runner.RESULTS_FILE = original_path
```

`main()` retains its original large ten-step calibration and four cases. It has
no parsed arguments/help/reduced mode and returns None. All cases use dt 0.02,
CFL adaptation and remaining config defaults, including theta 64/species 2.
Calibration at 128kx/32ky/16vpar/8mu, adiabatic/beta0, save10 estimates 2000 steps
from construction/run wall time divided by 10. An estimate above 120 minutes
selects 500 steps for the adiabatic beta0 and kinetic beta0.01 cases; otherwise
2000. Both save=max(count//10,1). Electromagnetic remains False. The kx64/256
grid cases each use 500 steps,save50 and adiabatic beta0. This estimate does not
separate compilation or early stopping from throughput. Each stage calls `save`,
so later failure leaves earlier scratch progress. A completed script exits 0;
unhandled provider/I/O/JSON errors give a Python failure. The short example
exercises the configurable API and does not execute or qualify this long campaign.
No matching grid-study wrapper exists in Rust/TypeScript; provider mathematics
remain unchanged. See the [benchmark scope](benchmarks.md#manual-jax-cbc-grid-study).


## Fixed CBC model comparison

`tools.gpu_cbc_benchmark` exposes `run_linear_benchmark()`,
`run_tglf_native()`, `run_nonlinear_numpy(n_steps=500, label="numpy")`,
`run_nonlinear_jax(n_steps=500, label="jax")`, `main()` and `install_jax()`.
Import itself performs neither installation nor a model run. There is no parsed
CLI/help/reduced campaign, and JAX CPU is permitted. Available reports are raw
in-memory dictionaries; model/provider errors propagate.

The linear call uses deuterium plus adiabatic electrons at T2keV,n5e19m^-3,
R/L_T6.9,R/L_n2.2,R0=2.78m,a1m,B2T,q1.4,shear0.78,16 ion-scale bins,
theta64,periods2. `LinearBenchmark` exposes `gamma_max`, `k_y_max`, `gamma`,
`k_y`, `mode_type` and `elapsed_s`. Growth is provider-normalised and ky is
k_y*rho_s. There is no comparison with an independent CBC reference.

The native TGLF call uses SAT1,16 bins,theta64,epsilon0.18, both temperatures
2keV,n_e5e19m^-3 and the same drives/geometry. Its underlying linear model uses
kinetic electrons and period1. `TGLFBenchmark` exposes `chi_i`, `chi_e`, `D_e`
in m^2/s, `dominant_mode`, raw normalised `gamma`/`k_y` and `elapsed_s`.
The native model needs no external TGLF executable; model output does not establish
physical agreement with that code.

Both nonlinear calls retain grid16kx/16ky/64theta/16vpar/8mu/species2,dt0.02,
save50,seed42,CBC drives6.9/6.9/2.2,geometry2.78/1/2,q1.4/shear0.78,
nonlinear/collisions enabled,nu0.01,hyper0.1 and CFL adaptation disabled.
Other defaults remain, including adiabatic electrons. The wrapper adds no count
validation. `NonlinearBenchmark` has `label`, raw code-unit `chi_i`/`chi_e`,
`converged`, `elapsed_s`, requested `n_steps`, `phi_rms`, `zonal_rms`, `Q_i`
and saved normalised model `time`; it omits final state and electron flux history.
Samples are post-step at indices0,50,..., with no appended terminal sample.
Zero steps give empty histories/zero means; one saved sample produces NaN means.
Raw NaN/Infinity are preserved. The requested count and finite-sample flag do not
prove complete integration or saturation. NumPy also checks detected divergence;
JAX's flag only checks multiple finite saved ion fluxes, with no separate
final-state/divergence condition. No provider parity or reference test is implied.

JAX runs on its configured device and precision without requesting a NumPy fallback.
Provider import `ImportError` or false `jax_available()` returns an
`UnavailableJaxBenchmark` containing only `label`, `skipped=True`, and respectively
`reason="import failed"` or `reason="JAX not available"`; other errors propagate.
This reported unavailability is neither a measured result nor a successful GPU run.

Each `elapsed_s` uses a monotonic clock around the actual provider call. Linear
excludes imports/species construction. TGLF includes solver construction and solve
but excludes imports/parameter construction. Nonlinear excludes config/solver
construction and includes run initialisation, diagnostics and, for JAX, first-use
compilation/synchronisation. Printing and report writes are excluded throughout.
The differing model/timing scopes, cache/device/precision and shared host load
prevent treating these fields as a controlled speedup comparison.

```python
from tools import gpu_cbc_benchmark as study

linear = study.run_linear_benchmark()
tglf = study.run_tglf_native()
assert len(linear["gamma"]) == len(linear["k_y"]) == 16
assert set(tglf) == {"chi_i", "chi_e", "D_e", "dominant_mode", "elapsed_s", "gamma", "k_y"}
empty = study.run_nonlinear_numpy(n_steps=0, label="actual allocation without stepping")
assert empty["n_steps"] == 0 and empty["Q_i"] == [] and empty["converged"] is False
```

This example exercises native reports and actual fixed-grid allocation; it does
not execute the large comparison campaign. `main()` retains six sequential stages:
linear,TGLF,NumPy500,JAX500,NumPy2000,JAX2000. Each nonlinear call starts fresh at
the same resolution; no separately measured warmup or higher-resolution stage is
added. Main returns None and writes once after all calls to caller-relative
`gpu_results/gk_nonlinear_cbc_gpu.json`, with one host UTC start timestamp.
`require_recorded_campaign` resolves the destination against source `REPO_ROOT`:
canonical persistent output requires the recorded runner, while its outside-root
scratch exemption remains. This guard checks custody, not result validity.
Missing JAX produces skipped maps and can still yield script exit0.
Provider errors before the write leave no new report; write failure can leave a
partial file. Existing output is overwritten with legacy indent2/allow_nan=True
JSON and platform text encoding. The producer adds no atomicity, output locking,
source/config/backend digest, alias protection or finite/reference validation.
The recorded runner supplies separate declared-artifact custody.

`install_jax()` explicitly invokes this interpreter's pip to install unpinned
`jax[cuda12]`, inheriting cwd/environment/console, then imports JAX and prints
actual devices. It returns True after success; pip/import/backend failures
propagate, without rollback or timeout. This may modify the environment and use
the network. The standalone import/install block precedes main's campaign guard;
calling main itself does not install. No matching fixed CBC comparison wrapper
exists in Rust/TypeScript. See the [benchmark scope](benchmarks.md#fixed-cbc-model-comparison).


## JarvisLabs training workflow and native transport

`tools.jarvislabs_train` retains the public `setup_jarvislabs`, `get_balance`,
`list_instances`, `create_instance`, `wait_for_ready`, `destroy_instance`,
`run_ssh_command`, `scp_upload`, `scp_download` and `main` imports. The SDK and
transport implementations are in `tools.jarvislabs_client` and
`tools.jarvislabs_transport`. Imports perform no provider request or installation.

`setup_jarvislabs(token)` lazily imports optional `jlclient.jarvisclient` and
sets its process-global token; empty/whitespace tokens refuse before import.
This does not authenticate the credential. `get_balance(client)` returns the
opaque actual SDK response; `list_instances(client)` requires a list. Standard
SDK stdout/stderr are discarded during requests, and exceptions propagate from
the API. Token configuration and stream redirection require one request stream
per process. Missing JLClient raises an import error; no SDK is installed here.

`create_instance(client, name)` requests one A5000 GPU, 20 GB storage and the
`pytorch` template. Resource availability, prices and quota are provider choices.
`wait_for_ready(instance, timeout=300)` uses `User.get_instance(instance_id=...)`,
a monotonic loop and up to ten seconds between polls. Running status plus a
nonempty SSH string is required; Failed/Destroyed status refuses. The positive,
finite timeout bounds the loop, not a blocking SDK request. Creation can leave
an allocated resource without a returned usable handle; automatic cleanup cannot
cover that case. `destroy_instance(instance)` returns a boolean: only the actual
SDK mapping `success is True` acknowledges the request. Exceptions, None and
malformed replies return false with a fixed cleanup notice. Acknowledgement does
not independently prove resource disappearance or stopped billing.

`run_ssh_command(endpoint, command, timeout=1800)`,
`scp_upload(endpoint, local_path, remote_path, timeout=300)` and
`scp_download(endpoint, remote_path, local_path, timeout=300)` run the installed
native executables and return captured text `CompletedProcess` only for exit
zero. Nonzero exits raise `CalledProcessError`; deadline expiration raises
`TimeoutExpired`; launch errors propagate. Endpoints accept only
`ssh user@host` or `ssh -p PORT user@host`, including bracketed IPv6. Extra flags,
invalid ports, malformed hosts and control characters refuse. Host keys must
already be trusted and authentication must be noninteractive; the functions do
not enroll hosts or configure keys. Commands are interpreted by the remote shell.
Transfer paths accept plain POSIX characters, and timeouts must be finite and
positive. Caller-supplied command/output content is not printed by the wrapper.

Uploads require an existing local file/directory and retain recursive SCP's
remote replacement semantics. Downloads require an existing parent and a new
local target; files, directories and dangling symlinks refuse before transport.
Concurrent path changes are not locked, transfer failures can leave partial
candidates, and neither successful transfer nor file existence authenticates
contents or provides atomic publication.

`main(argv=None)` returns 1 for missing credentials before SDK import. The API
parses an empty argument list when `argv` is omitted; the standalone CLI passes
its actual arguments. `--output-dir PATH` selects nine flat downloads into an
existing directory. Omitting it retains the legacy local paths. Before any
provider request, main checks all seven declared upload sources, nine distinct
fresh targets with existing parents, and recorded-campaign input for the canonical
benchmark. A campaign flag is a custody input, not authenticated provenance.
Setup, readiness, transport and artifact/metric failures return 1 with fixed text.
Every returned creation handle is retained for cleanup even when readiness fails;
unconfirmed cleanup also returns 1, and Ctrl-C executes cleanup then propagates.

The remote recipe installs dependencies, reuses or clones `scpn-control`, uploads
the model, trainer, shell, benchmark and three recipe helpers, and invokes the
500000-step shell recipe with CPU training forced. Candidates use
`artifacts/rl/<campaign>` remotely. Other remote source and dependencies are not
pinned to the local checkout. Downloads must be nine nonempty regular files;
`MPC/PID/PPO` summaries must include finite reward, nonnegative population deviation,
mean episode length of at least one, disruption fraction in [0,1], and equal
positive integer episode counts. File/schema checks do not authenticate weights.
The corrected shell forwards seeds 42/123/456 to the actual PPO constructor.
Retained legacy seed-labelled artifacts predate that correction: the old recipe
used seed 42 for all three labels and does not prove independent training.

This local example exercises the actual CLI's missing-token refusal without
provisioning, SDK installation, training or artifact writes:

```python
import os
import subprocess
import sys
from tools import jarvislabs_train

env = dict(os.environ)
env.pop("JARVISLABS_TOKEN", None)
result = subprocess.run(
    [sys.executable, str(jarvislabs_train.REPO_ROOT / "tools/jarvislabs_train.py")],
    cwd=jarvislabs_train.REPO_ROOT, env=env, capture_output=True, text=True,
    timeout=20, check=False,
)
assert result.returncode == 1
assert result.stdout == "Set JARVISLABS_TOKEN environment variable\n"
assert result.stderr == ""
```

::: tools.jarvislabs_train

::: tools.jarvislabs_client

::: tools.jarvislabs_transport


## Seeded PPO recipe, evaluation and candidate results

`tools/train_rl_tokamak.py` retains the `GymTokamakEnv`, `PIDController` and
`evaluate_agent` imports. The implementations reside in
`tools.rl_tokamak_evaluation`; configuration and report checks reside in
`tools.rl_training_config` and `tools.rl_training_results`. Importing these
modules does not train a model or write weights.

Install the `rl` or `dev` extra for the actual Gymnasium/SB3 runtime; they require
Gymnasium >=1.2.3 and Stable-Baselines3 >=2.8. The
[CI dependency profile](development.md#rl-test-dependencies) supplies hash-pinned
SDKs and platform-specific PyTorch wheels before the matrix's first Test step.

`TrainingConfig(timesteps, eval_episodes, output, seed=42, dry_run=False)` requires
positive integer counts, a uint32 seed and a `.zip` output. `--ci` caps requested
steps at 5000 and episodes at five. `--dry-run` prints JSON containing the same
`ppo_parameters()` mapping used for construction: MlpPolicy, learning rate
3e-4, rollout 256, batch 64, ten epochs, gamma 0.99, GAE lambda 0.95, clip 0.2,
caller seed and CPU device. It does not import PPO, create a model/directory or
establish execution readiness. This example exercises the native shell and its
three child CLIs without learning or output writes:

```bash
SCPN_RL_PYTHON=.venv/bin/python bash tools/train_rl_upcloud.sh --dry-run --output-dir artifacts/rl/example 321 4
```

Real trainer execution checks the ZIP and adjacent `.metrics.json` destinations
before constructing PPO. Existing files, directories and dangling links refuse.
Learning may complete a rollout beyond the requested count; metrics record both
`timesteps` and `actual_timesteps`, the training seed, evaluation seeds, learning
time and PPO/PID summaries. Saving ZIP and metrics is not an atomic transaction;
a failure can leave a partial candidate. CLI execution failures return 1 with
fixed text; argparse help and malformed arguments retain exits 0 and 2.

`GymTokamakEnv` uses the unchanged float64 reduced-order `TokamakEnv` behind
float32 six-component observations and two-component actions. The positive episode
horizon defaults to 500. Reset seeds both wrappers; reset options are accepted
and unused. `PIDController` retains the legacy proportional heating/current
corrections: its name does not imply integral/derivative terms.
`evaluate_agent(env, policy, n_episodes=20)` uses paired seeds starting at 1000,
calls controller functions or deterministic `predict`, and stops on termination
or truncation. It counts termination as disruption and returns finite reward
mean/population deviation, mean length, disruption fraction and episode count.
Evaluation advances the supplied model; it does not certify physical safety.

`rl_training_results.py preflight DIRECTORY` checks all nine shell destinations
and canonical benchmark campaign input before learning. `select DIRECTORY` reads
actual seed identities and finite PPO metrics, selects the greatest reward without
a numeric sentinel, and keeps the first seed on ties. Legacy metrics lacking a
seed field refuse. `summarize REPORT` validates the actual uppercase MPC/PID/PPO
schema and shared count. These operations never authenticate, train or copy weights.

The shell's execution mode forwards each seed, copies the declared best candidate,
and invokes `benchmarks/rl_vs_classical.py --episodes N --agent ZIP --output REPORT`.
The benchmark loads an existing policy on CPU and evaluates PPO, the unchanged
proportional baseline and the original 11x5 one-step temperature-cost MPC on paired
seeds. Its report must be fresh, and persistent destinations require recorded
campaign input. No training or weight save occurs in the benchmark.
`examples/tutorial_03_ppo_rl_agent.py --episodes N` runs all six default sections
with stored-policy inference and the retained report table. Learning requires
explicit `--train-demo`, which selects the original 5000-step demonstration.
Both commands return 1 with fixed text on execution/input delivery failure.

::: tools.rl_training_config

::: tools.rl_tokamak_evaluation

::: tools.rl_training_results

::: tools.train_rl_tokamak

::: benchmarks.rl_vs_classical

::: examples.tutorial_03_ppo_rl_agent


The trainer, benchmark and tutorial also guard CLI dependency loading. With a
missing scientific dependency they return 1 and fixed dependency-unavailable
text before model work or output writes. Importing them as library modules still
propagates the import failure. Help/syntax exits 0/2 apply when the required import
profile is available; dependency failure can occur before argument parsing.


## Native oscillator capacity sweep

`tools.stress_test_oscillators.stress_test() -> None` calls the installed
`scpn_control_rs.upde_tick` binding on a fixed 16-layer allocation. It prints a
latency table and a completion or capacity-stop message. The function consumes
the process-global NumPy RNG and returns no report value.

The eleven per-layer sizes are 10, 50, 100, 256, 512, 1024, 2048, 4096, 8192,
16384 and 32768. One invocation draws a coupling matrix uniformly in [0,0.5);
each size gets fresh phases in [0,2π). Frequencies come from `plasma_omega(16)`.
Phase lags are zero, zeta is 0.5, the global driver is 0.3, PAC gain is 1 and
the model timestep is 0.001.

Each size performs five warm-up calls and then 50 measured calls below 8192
oscillators per layer, or ten at larger sizes. Measured calls reuse the input
arrays and discard returned ticks. Mean call latency uses `perf_counter`; the
printed frequency is its reciprocal in calls per second. A frequency below ten
ends the sweep with a capacity-stop message and a normal return.

```bash
python tools/stress_test_oscillators.py
```

The CLI returns zero after normal completion or a capacity stop. Missing
dependencies return one with `Oscillator capacity benchmark dependencies are
unavailable.`; other execution failures return one with `Oscillator capacity
benchmark failed.`. Library calls propagate native import and execution errors.
Importing the tool in an environment with NumPy and the Python package leaves
the native extension import until `stress_test()` is called.

The native boundary is the [Rust Python binding](#pyo3-bindings), implemented in
`scpn-control-rs/crates/control-python/src/lib.rs`.
See [measurement semantics](benchmarks.md#native-oscillator-capacity-sweep) before
interpreting the table as a capacity or scheduling result.

## Selected Python preflight command

`tools.run_python_preflight.main(argv=None) -> int` runs metadata, Golden
notebook, Task 5/6 smoke and strict-typing checks in that order, followed by the
unconditional docstring gate. Four independent skip flags omit the first four
categories. Children run in the owning repository with the current interpreter;
caller working-directory changes do not change the inspected checkout.

Zero means every selected child succeeded. The first nonzero child status is
returned, with later checks left unexecuted. Help exits zero and invalid
arguments exit two; subprocess launch errors propagate. Child stdout and stderr
are inherited, and the final success line names the selected checks.

The default notebook and Task 5/6 test targets are absent from this extraction,
so a default call currently fails at the notebook check. A successful limited
selection does not qualify those missing profiles or supply scientific
admission. The [development reference](development.md#selected-python-preflight-checks)
lists the exact target files and selection flags. There is no separate native
preflight implementation or benchmark timing result from this command.

## Native fuzz campaign evidence

The campaign runner builds and invokes the five Rust harnesses described in
[fuzzing](validation/fuzzing.md). It is separate from the fast Python preflight.
Its public record validator accepts only nonempty, complete, unique registered
campaign evidence with positive execution counts and finite positive durations.
A returned verdict validates submitted records; it does not authenticate the
binary, source, sanitiser or complete evolving corpus.

```python
from tools.run_fuzz_campaign import triage

passed, failures = triage([], [])
assert passed is False
assert failures
```

The CLI refuses invalid target and resource selections before compilation.
Campaign metadata queries also require successful native commands, nonempty
version output and one nonempty host line; unavailable metadata refuses before
compilation. These observations do not authenticate the binary or source.
Build-only success means the native build succeeded and supplies no run report.
Campaign JSON always carries `production_claim_allowed: false`.

::: tools.run_fuzz_campaign

## Legacy gyrokinetic campaign reporters

The following scripts retain their fixed CBC campaign settings. Their JAX
cases use the available device without guaranteeing GPU execution; their
implicit electron cases use NumPy. They parse no CLI options, so passing
`--help` does not avoid launching a campaign. Importing the modules and rendering
their API documentation does not run a solver.

| Reporter | Original campaign | Output relative to caller cwd |
|---|---|---|
| `kinetic_e_dual_test` | Explicit JAX, mass 1/25, 5,000 steps; implicit NumPy, mass 1/400, 200 steps | `gpu_results/kinetic_e_dual.json` |
| `kinetic_electron_test` | Adiabatic and kinetic JAX, mass 1/400, 5,000 steps each | `gpu_results/kinetic_electron_comparison.json` |
| `sugama_comparison` | Krook/adiabatic and Sugama/adiabatic JAX; Sugama/kinetic implicit NumPy, 5,000 steps each | `gpu_results/sugama_comparison.json` |

All use the original grid `128 × 16 × 32 × 16 × 8`, CFL-adaptive nominal step
`0.05` and diagnostic interval 100. Their wall times cover `solver.run()` and
exclude construction. The cases change physics and sometimes backend/work,
so these records do not establish a paired backend speed comparison.
Reported `chi_i_gB` divides the producer's mean ion flux by `R_L_Ti`; histories
remain in solver units. Scalar nonfinite chi becomes `None`, while histories
retain their original values. The copied `converged` flag supplies no nonlinear
saturation or finite-final-state certificate. Existing output files are
overwritten after the cases return, and JSON retains default nonfinite-number
behaviour. The two summary printers can fail on `None` after writing their files.

::: tools.kinetic_e_dual_test

::: tools.kinetic_electron_test

::: tools.sugama_comparison

## Legacy QLKNN preparation and trainer

`tools.train_neural_transport_qlknn` is the retained NumPy prototype. Its loader
selects an unsorted first NPZ match before CSV and validates neither dataset
provenance nor units, shape or finiteness. Stored NPZ dtypes remain unchanged;
CSV skips one header and slices ten feature and three target columns.

The synthetic generator prepares proxy arrays without training or writing
weights:

```python
from tools.train_neural_transport_qlknn import generate_synthetic_qlknn_data

features, targets = generate_synthetic_qlknn_data()
assert features.shape == (5000, 10)
assert targets.shape == (5000, 3)
```

`python tools/train_neural_transport_qlknn.py --help` displays actual CLI options
without learning. Supplying `--synthetic` or a loadable `--data-dir` invokes
the learner and creates weights. Synthetic targets use a scale-one
critical-gradient proxy and supply no real QuaLiKiz provenance. The original
learner normalises all rows before splitting, retains the last weights after
early stopping and returns metrics that share preprocessing with test rows.
Those metrics are not an independent holdout certificate.

The default weights destination resolves from the script to
`weights/neural_transport_qlknn.npz`; explicit relative paths resolve from caller
cwd. NumPy may append `.npz`, while metrics independently replace the supplied
suffix with `.metrics.json`. Files are overwritten through separate writes,
without atomic publication or dataset/adoption admission.

::: tools.train_neural_transport_qlknn

## Documentation graph and HTTP observation

`tools.document_link_audit` exposes indexed-source selection, supported link
extraction, local/site inspection and optional sequential HTTP/cache observation.
The local graph checks existing relative targets, default index membership,
Markdown heading anchors and MkDocs navigation. It reads current disk contents
after index enumeration, without a coherent snapshot or network requests.

```python
from pathlib import Path
from tools.document_link_audit import extract_links

source = Path("README.md").resolve()
references = extract_links(source, source.parent)
assert all(reference.line >= 1 for reference in references)
```

`python tools/document_link_audit.py --list-external` lists screened HTTP URLs
without requesting them. Actual availability requires explicit `--external`.
HTTP/cache classifications, restricted/transient zero-status behaviour, report
overwrite/provenance limits and site-build prerequisites are described in the
[development contract](development.md#documentation-links). These observations
do not authenticate documentation contents or certify remote cited claims.

::: tools.document_link_audit

## Public docstring floor and ledger

`tools.run_docstring_gate` checks the configured public Ruff rule set and its
default all-definition owners before reading or updating a count snapshot.
Missing docs return one; inspection/ledger failure returns two; a refused
baseline increase returns three. Extra AST paths add to the ordinary scope.

```python
from tools.run_docstring_gate import DocstringDebtLedger, evaluate_ratchet, parse_ruff_diagnostics

baseline = DocstringDebtLedger.from_dict({
    "schema": "scpn-control.docstring-debt.v1",
    "total": 0,
    "per_module": {},
})
assert evaluate_ratchet(0, {}, baseline).ok
assert parse_ruff_diagnostics("[]") == []
```

Counts are nonnegative integers, booleans are refused, total/module sums agree,
and optional legacy `rules` must match the selected rule order. Files additionally
require unique JSON object keys. Direct constructors remain unchecked values;
native reconstruction/writes enforce the ledger contract. The
[development contract](development.md#public-api-docstrings) describes path,
update and error semantics. A passing floor measures documentation presence.

`parse_ruff_diagnostics` is the same pure JSON decoder used by the native Ruff
probe. It accepts an array of objects with string filenames and one of the five
selected rule codes, preserving other fields. Empty/malformed/non-array JSON and
invalid entries raise fixed authored `RuntimeError` messages. A decoded filename
does not establish file existence or the authenticity of a Ruff execution.
Native numeric-conversion refusals, including the interpreter's integer digit
limit, use the same authored JSON-decoding error.

::: tools.run_docstring_gate

## Strict typing debt and native gate

`tools.run_mypy_strict` runs the configured source/validation check and a native
strict package probe, then compares coherent per-module counts with a persisted
ledger. The [development contract](development.md#type-checking) describes the
explicit advisory mode, update flags and failure statuses.

```python
from tools.run_mypy_strict import StrictDebtLedger, evaluate_ratchet

baseline = StrictDebtLedger.from_dict({
    "schema": "scpn-control.mypy-strict-debt.v1",
    "mypy_version": "example-only",
    "total": 2,
    "per_module": {"historical.module": 2},
})
result = evaluate_ratchet(0, {}, baseline)
assert result.ok and result.total_delta == -2
assert result.improvements == {"historical.module": (2, 0)}
```

`from_dict` refuses booleans, negative/coerced counts, empty/non-string labels,
non-string versions and unequal total/module sums with authored `ValueError`
messages. Missing legacy version and extra fields remain supported. File loading
additionally rejects duplicate keys and non-object JSON. Direct constructors
remain unchecked and their dictionaries mutable; pure ratchet comparisons assume
caller-supplied counts and do not authenticate a Mypy run. Version strings are
observations, not provider certificates. File writes are caller-selected and
non-atomic; CLI refusal preserves an existing malformed ledger.

::: tools.run_mypy_strict

## Rigid vertical validation reports

The RZIP validator compares the production rigid model with exact no-wall
references and a passive-wall case. Finite positive tolerances apply to relative
errors and marginal growth in s^-1. The CLI exposes the same tolerance overrides,
emits text or sealed JSON and optionally replaces a JSON/Markdown report pair.

The report decoder requires the exact v1 structure, finite domains and literal
boolean verdicts consistent with the declared metrics and scaling laws. A
well-formed failing report returns `False`; invalid or contradictory reports
raise `ValueError`. A self-seal does not authenticate a producer, measured
reference or facility admission. The writer validates before writing, requires
an existing parent and can leave a JSON-only pair on a later IO failure.

```python
from validation.validate_rzip_vertical_stability import (
    build_evidence, validate_evidence_payload, validate_rzip_vertical_stability,
)

report = build_evidence(validate_rzip_vertical_stability(), target_id="rigid-example")
assert validate_evidence_payload(report) is True
failed = build_evidence(
    validate_rzip_vertical_stability(exact_tol=1e-30), target_id="tight-tolerance",
)
assert validate_evidence_payload(failed) is False
```

::: validation.validate_rzip_vertical_stability

::: validation.rzip_vertical_evidence

::: validation.rzip_vertical_models

## Bounded normalized DGKF reports

The H-infinity validator runs the actual normalised flight-simulator factory,
Riccati/controller identities and a fixed 20002-frequency sweep. The result is
a frozen local observation; finite sampling corroborates the sampled gain and
does not prove an exact norm or facility-control behaviour.

The decoder requires the exact v1 structure, finite domains, original numerical
thresholds, peak/gamma agreement and literal consistent verdicts. It checks the
three declared runtime source files against current bytes, including the report
decoder. The capture-time Git label is checked for object-ID format; it need not
equal a later HEAD with unchanged source bytes. Matching hashes and a self-seal
authenticate no producer or actual run.

Coherent failing results serialise with local `scientific_admission=False` and
are refused by `validate_evidence_payload`. `public_claim_allowed` and
`production_admission` must always be literal `False`; the fixed model,
exclusions and finite-sweep classification cannot be widened by resealing.

```python
from validation.validate_h_infinity_control import (
    build_evidence, validate_evidence_payload, validate_h_infinity_control,
)

report = build_evidence(validate_h_infinity_control())
assert validate_evidence_payload(report) is True
assert report["claim_boundary"]["production_admission"] is False
```

`--check-report PATH` reads unique-key UTF-8 JSON, validates it and prints an
accepted payload without replacing it or writing outputs. Ordinary producer
mode writes the requested JSON/Markdown pair unless `--no-write` is supplied.
The writer creates parents and replaces files directly, rejects output aliases
and overwriting declared runtime sources, and can leave a JSON-only pair after
a later IO failure. Check mode returns 0 for acceptance or 1 with a diagnostic
for refusal; malformed CLI arguments use argparse status 2. Old source-bound
reports remain historical and may fail the current source-owner check.

::: validation.validate_h_infinity_control

::: validation.h_infinity_evidence

## Declared RZIP calibration

`rzip_calibration_evidence` captures the actual bounded plant's growth rate
[s^-1], growth time [ms] and inertia [kg], together with caller-declared wall
time [s], reference growth [s^-1] and dimensionless error/tolerance. The frozen
`RZIPCalibrationEvidence` constructor itself is unchecked. Builder, writer and
admission API require consistent metrics, literal version/boolean fields,
declared-source status and the historical ASCII JSON digest. Positive growth
time must equal `1000/gamma`; zero growth retains positive infinity as historical
extended-real JSON and never sets the facility flag.

The facility function returns the same observation only for a declared external
source with finite positive model growth/time and a matching reference within
tolerance. A label, quantitative self-comparison and self-seal authenticate no
external result, acquisition, freshness or deployment approval. Local, missing
reference, stable and coherent tolerance-failing observations remain writable
and refuse at the facility gate. The writer validates before creating parents
or replacing JSON; later IO errors propagate without an atomicity guarantee.

The fixed calibration benchmark uses the actual historical symmetric two-loop
plant. Its default persistent outputs still require the recorded-campaign
wrapper. These real CLI variants print checked JSON or write a temporary pair:

```bash
python -m validation.benchmark_rzip_calibration --no-write
python -m validation.benchmark_rzip_calibration --json-out "$RZIP_REPORT_DIR/rzip-calibration.json" --markdown-out "$RZIP_REPORT_DIR/rzip-calibration.md"
```

Set `RZIP_REPORT_DIR` to a caller-owned temporary directory for the second
command. Output/source aliases refuse before writing. JSON then Markdown are replaced
directly and a later failure can leave a partial pair. CLI success is 0,
inspection/model/IO/custody refusal is 1 with a diagnostic, and argparse uses
0/2 for help or malformed arguments. This produces bounded local regression
evidence without facility admission or an authenticated external reference.

::: validation.benchmark_rzip_calibration

## Static test ownership linkage

::: tools.check_test_module_linkage

The four public APIs inspect UTF-8 files and native ASTs without importing or
executing the inspected package/tests. Direct API paths use the caller's working
directory; command-relative paths use the script checkout. The source package
spelling remains `scpn_control` for custom roots. Missing/non-directory roots
raise `NotADirectoryError`; existing empty directories are supported.

Top-level `test_` functions and `test_` methods of top-level `Test` classes supply
entry points. Calls and assertion references resolve visible imports, named
called local helpers and visible import re-exports in initialisers and ordinary
module files. A referenced file facade and its selected export target both count;
unused facade imports do not. Export cycles terminate and traversed files are
parsed once per inspection. Scope-wide rebindings and conflicting aliases are
refused conservatively.
Named, unambiguous class declarations follow imported or locally declared base
identities, without crediting an unreferenced class. Class decorators/keywords,
wildcard imports and rebinding prevent class attribution; dynamic bases are ignored.
This establishes type references without evaluating class initialisation or inherited
method dispatch. A selected plain method follows named lexical body references;
decorated, conflicting or source-mutated methods do not. A single direct assignment
from a visible constructor name identifies a receiver for later references in that
scope. Unrelated data-field writes preserve the receiver binding. Explicit writes
or deletions to a selected member or its parent path refuse that edge. Receiver
type/dictionary changes and imported constructor-hook mutations refuse inferred
instances; imported member writes resolve through visible aliases. Source methods
that overwrite the selected receiver member and classes with custom attribute
hooks or type/dictionary writes are opaque. Bare class identities remain static
references. Parameters, other name writes/imports, conditional assignments and
annotated/chained/tuple assignments prevent receiver binding.
Aliases, inline constructors and factory return values are not inferred. Selecting
an overridden method does not follow the base method body. Class namespaces and
implicit `self`/`cls` helper dispatch are not evaluated.
A selected, unambiguous top-level function also follows named calls and assertion
references in its body through lexical imports and module helpers. Unused functions,
defaults, annotations, decorated/rebound declarations and function-object attributes
do not establish body-call edges. This does not prove that a function referenced in
an assertion actually executes. Recursive function references terminate.
Fixture injection, decorators, dynamic imports, subprocesses, arbitrary export
assignments, dynamic mutator calls, inherited descriptors and argument/return flow
are outside this scan. Unreachable syntactic calls
may count, and attribute/importability checks are absent. This establishes
bounded static references, not runtime reachability, execution or coverage.

```python
from pathlib import Path
from tools.check_test_module_linkage import collect_source_modules, collect_unlinked_modules

source = Path("src/scpn_control")
assert collect_source_modules(source) == sorted(collect_source_modules(source))
missing = collect_unlinked_modules(source_root=source, test_root=Path("tests"))
assert missing == sorted(missing)
```

```bash
python tools/check_test_module_linkage.py
python tools/check_test_module_linkage.py --source-root src/scpn_control --test-root tests
```

The CLI prints four counts and sorted missing/stale paths; zero means that the
configured static contract passes. Unexpected owners return one before the stale
allowlist decision. `--allow-stale-allowlist` permits stale exemptions only.
Malformed arguments return two; help returns zero before file reads. Native
decode/AST/IO exceptions propagate (normally script exit one with traceback).
The JSON allowlist retains exact nonempty path strings; duplicates collapse and
extra metadata is ignored. Neither input files nor policy bytes are written.
Concurrent files are not an atomic snapshot and no path confinement is promised.

## Native coverage declaration guard

`tools.native_coverage_matrix.validate_native_coverage_matrix` reads workflow,
threshold and prose inputs without rewriting them or executing their shell.
`workflow_path=None` selects the live distributed policy. A selected YAML path
is a monolithic workflow; a selected `.json` path is a distributed policy whose
relative workflow paths resolve from the parent of its containing directory.
For the live `tools/ci_workflow_policy.json`, that directory is the repository
root. `pyproject_path` and `docs_paths` select the other readable UTF-8 inputs.

The guard requires enabled Python/native collection, the expected saved data
paths, immutable artifact-action pins and hidden-file uploads. It binds the
coordinator calls to physical reusable owners and checks ordered downloads,
direct guard invocation, combination, XML output and the 100-percent report
command. Literal workflow/job/step environments and run defaults are included;
the Ubuntu/Python 3.12 selector must match a non-excluded matrix lane. Parsed
TOML `tool.coverage.report.fail_under` must be numeric 100.

Only the maintained direct shell profile is admitted. Comments, `echo` and
heredoc bodies do not supply commands. Conditional blocks, failure masking,
early exit, environment mutation, non-root working directories and unsupported
shells refuse the affected contract. This inspects local declarations; it does
not authenticate artifact contents, measure coverage or prove hosted execution.

`NativeCoverageFinding(path, check, detail)` is a frozen authored diagnostic.
`NativeCoverageMatrix(findings)` is an unchecked frozen collection; `passed`
means only that it contains no findings. `to_jsonable()` returns a fresh
`scpn-control.native-coverage-matrix.v1` mapping with `passed` and `findings`.
Input IO/decoding, malformed or duplicate-key YAML/JSON/TOML, and incomplete
shell declarations become fixed authored findings. Empty `docs_paths` fails
the prose contract. No caught interpreter exception text is exposed.

These source-tree entry points use the same `main(argv)`:

```bash
python tools/native_coverage_matrix.py --json
python -m tools.native_coverage_matrix
```

`--workflow`, `--pyproject` and `--docs` select inputs; policy snapshots retain
their repository directory layout. JSON and text success return 0, findings
return 1, and argparse help/malformed arguments retain exits 0/2. The command
uses the pinned CI tooling profile, including PyYAML, and has no installed
console-script alias.

## JOSS local editorial and citation guard

`tools.check_joss_submission.check_repository() -> list[str]` reads three fixed
UTF-8 files under the repository discovered from the resolved script path:
the canonical submission's `manuscript.md` and `references.bib`, and
`docs/joss_paper.md`. It returns a fresh list of findings, in input, marker,
title, then bibliography/citation order. The caller owns the returned list;
mutating it has no effect on later checks. Files are read without writes,
downloads, simulation state, numeric units, array shapes, or timing inputs.

Missing files and empty/whitespace-only files receive `MISSING:` and `EMPTY:`
findings respectively. Present-file read failures and invalid UTF-8 raise the
native `OSError` or `UnicodeError`. Symlinks retain ordinary pathlib behaviour;
this tool does not establish a sandbox or authenticate input provenance.

Required manuscript and pointer markers are case-sensitive substrings after
Unicode whitespace normalisation. The first regex-matched `title:` line must
be within the manuscript's initial literal `---`-delimited block; its normalised
text must appear in the pointer. This is lexical title extraction: YAML escaping,
metadata types, duplicate metadata keys, and the complete JOSS schema are outside
the check. The bibliography must have at least one recognised `@word{key,`
entry, no repeated case-sensitive keys, and all manuscript keys recognised
inside Pandoc-style brackets. Keys begin with an ASCII letter and can contain
letters, digits, underscores, colons, or hyphens. Bare narrative citations,
documentation citations, and full BibTeX syntax are outside the scan. Comments
and code fences are not parsed separately.

`main() -> int` prints findings and their count to standard output and returns
one, or prints the scoped local-success message and returns zero. Read/decode
exceptions propagate. The standalone `python tools/check_joss_submission.py`
command uses the same fixed paths, independently of caller cwd, and has no option
parser. Existing preflight and static-governance lint invoke that command. This
guard does not render the PDF, resolve hyperlinks, authenticate citations or
scientific reports, submit a paper, or establish review acceptance.

```python
from tools.check_joss_submission import check_repository, main

findings = check_repository()
assert findings == []
findings.append("caller-owned annotation")
assert check_repository() == []
assert main() == 0
```

The example uses the current canonical draft and reads no external service.
Regression fixtures copy the production script byte for byte into real physical
trees and exercise these same public functions and the actual standalone command.

## Coverage exception ownership ledger

::: tools.coverage_exception_ledger

`build_ledger()` reads paths relative to the physical script's repository,
independently of caller cwd. It inventories lexical package-source no-cover
pragmas, configured exclusion patterns and AST calls spelled
`pytest.mark.skipif`, `pytest.skip` or `pytest.mark.xfail` under tests/validation.
Aliases, `pytest.mark.skip` and `pytest.importorskip` are outside this scope.
String/docstring pragma spellings are lexical matches; xfail strictness is not
checked. Entries use one-based lines and first-match policy descriptions.

The schema-v1 TOML loader requires a nonnegative integer count (excluding
booleans), a lowercase SHA-256 string, exact calendar review labels and a
nonempty ordered rule array. Required metadata are nonempty strings;
identifiers are unique, statuses come from the maintained vocabulary and
patterns compile. Empty workflow evidence declares no CI text binding.
Unknown keys and valid string spellings are preserved. The last rule is a
fallback candidate and can fail to match. Validation checks syntax and shape;
competent review of ownership rationale remains a separate requirement.

```python
from tools.coverage_exception_ledger import build_ledger, main

ledger = build_ledger()
assert ledger["schema"] == "scpn-control.coverage-exception-ledger.v1"
assert ledger["entry_count"] == len(ledger["entries"])
assert sum(ledger["counts"].values()) == ledger["entry_count"]
assert main(["--check"]) == 0
```

Entry IDs bind kind/path/line/condition/reason; the full-entry digest also binds
metadata. Sequential reads have no atomic snapshot or clock. Workflow
substrings and lane labels declare evidence requirements, without proving
executed variants, dependency availability, coverage or scientific readiness.
The frozen `ExceptionEntry` constructor itself preserves values without
validation. No physical units or controller state are processed.

`main(argv=None)` uses native argparse arguments. `--print-summary` takes
precedence over `--check`, reads no output file and reports unreviewed inventory.
Other modes refuse count/digest drift before output IO; check then compares
exact sorted-key, indent-two JSON plus a final newline. Successful summary,
write or check returns zero; stale/missing output or changed seal returns one.
Help exits zero and unknown options exit two. Inspection, decoding and
filesystem exceptions propagate; the CLI normally exits one with their native
traceback. Default writing replaces the fixed output directly, with no atomic
replacement, symlink confinement or partial-IO rollback.

## Coverage pragma reason guard

`tools.check_coverage_pragmas.CoveragePragmaViolation(path: str, line: int,
text: str)` is a frozen dataclass. Its fields are a resolved POSIX diagnostic
path (repository-relative when possible, absolute otherwise), a one-based line
number and the matching line with surrounding whitespace removed. Construction
does not validate values or establish containment; native missing/unexpected
argument errors and frozen-field assignment errors remain observable.

`iter_python_files(paths: Sequence[Path]) -> list[Path]` returns sorted
case-sensitive `.py` file requests and native `Path.rglob('*.py')` file
descendants of requested directories. Paths keep their supplied spelling;
overlapping requests retain duplicates. Relative API paths use the caller's
working directory. Empty sequences and directories return empty lists. A missing
explicit path or broken symlink raises `FileNotFoundError`; an existing request
that is neither a Python file nor a directory raises `ValueError`.

`find_unreasoned_pragmas(paths: Sequence[Path]) -> list[CoveragePragmaViolation]`
reads these files as UTF-8 and searches each entire line for the case-sensitive
`pragma:\s*no cover` marker. Leading whitespace and `-:;.,#) ]–—` separators
after the marker do not count as a reason. Remaining text counts lexically;
its scientific validity and exception ownership are not judged. Strings and
docstrings are searched too: this tool does not parse Python syntax or coverage
configuration. Diagnostics follow sorted file order, then source line order,
and duplicate requests produce duplicate diagnostics.

`main(argv: list[str] | None = None) -> int` accepts explicit argparse tokens or
process arguments. Omitted paths select the script repository's
`src/scpn_control`; relative CLI spellings resolve against that repository.
`--json` emits `{"unreasoned": [{"path": ..., "line": ..., "text": ...}]}`.
The normal output prints a summary and every finding to standard output. Zero
means no unreasoned markers among files actually enumerated; one means findings.
Argparse help exits zero and invalid options exit two. Missing/unsupported
requests, native filesystem errors and `UnicodeDecodeError` propagate before
any success summary or JSON. Files are read without changing sources or policy;
enumeration and reading do not provide an atomic concurrent-edit snapshot.

The maintained command is `python tools/check_coverage_pragmas.py`, also invoked
by preflight. This local lexical gate supplies no coverage measurement, compiler
variant, containment or scientific-readiness proof. The broader exception ledger
remains a separate ownership and CI-variant policy check.

```python
from pathlib import Path
from tools.check_coverage_pragmas import (
    CoveragePragmaViolation,
    find_unreasoned_pragmas,
    iter_python_files,
    main,
)

paths = [Path("src/scpn_control")]
assert iter_python_files(paths)
assert find_unreasoned_pragmas(paths) == []
record = CoveragePragmaViolation("example.py", 1, "# pragma: no cover")
assert (record.path, record.line) == ("example.py", 1)
assert main([]) == 0
```

## Tracked source-header policy contract

`tools.check_source_headers` reads schema-v1 TOML and checks Git-index paths
against current worktree headers. The public frozen dataclasses are
`Finding(path: str, category: str, detail: str)`,
`Exemption(category: str, reason: str, suffixes: frozenset[str],
names: frozenset[str], paths: frozenset[str] = frozenset())` and
`Policy(schema: str, enforced_suffixes: frozenset[str],
enforced_names: frozenset[str], exemptions: tuple[Exemption, ...])`.
Construction preserves supplied values and freezes fields; validation occurs
in `load_policy(path: Path) -> Policy`. `FindingPayload` and `AuditResult` are
static dictionary contracts, with no runtime constructor validation.

The loader requires `scpn-control.source-header-policy.v1`, an enforced table
and, when supplied, an array of exemption tables. Suffix/name/path collections
must be arrays of strings. Omitted collections are empty; duplicates collapse.
Suffixes case-fold; names and exact paths retain case. Exemptions require a
nonblank category and a reason with at least 20 stripped characters. Enforced
and exempt suffix/name sets cannot overlap. Exact exemption paths must be
canonical relative POSIX spellings, cannot repeat and cannot overlap any
suffix or basename declaration. Unknown keys are ignored; syntactically valid
rationales are not evaluated for legal or scientific sufficiency.

`tracked_paths(root: Path) -> list[Path]` runs `git ls-files -z`, decodes UTF-8
names and sorts native paths.
`classify(path: Path, policy: Policy) -> tuple[str, str]` returns
`("enforced", "")`, `("exempt", category)` or
`("unclassified", "")`, using enforced precedence and then exemption order.
Classification performs no I/O, root resolution or policy revalidation.

`expected_header(path: Path, purpose: str) -> list[str]` renders the six exact
identity fields and a purpose. Lean uses a nine-line block; HTML uses seven
individual comments; C/C++/JavaScript/Rust/TypeScript use slash comments;
other suffixes use hash comments. The renderer does not validate the purpose.
`header_finding(root: Path, path: Path) -> Finding | None` reads current UTF-8
bytes, accepts one leading shebang and native newline spellings, and requires
exact identity plus nonblank purpose with native closing syntax. Later content
is ignored. Invalid text produces `header_mismatch`; non-UTF-8 source content
produces `non_utf8_enforced`; native file errors propagate.

`audit(root: Path, policy_path: Path) -> AuditResult` validates policy before
Git access. Its `scpn-control.source-header-audit.v1` result contains
`policy_schema`, post-scan `source_head`, `passed`, observed classification and
exemption counts with sorted keys, and findings in tracked-path order. Zero
count categories are absent. An empty tracked repository can pass. Exempt
files are not read. Counts have no physical units or time axis.

`main(argv: list[str] | None = None) -> int` runs the maintained command
`python tools/check_source_headers.py`. Defaults use the script repository and
its policy; explicit relative options and API policy paths use the process
working directory. Exit zero means no findings, one means findings, and two
means caught policy, decode, filesystem or Git errors. `--json` prints the
complete audit only for zero/one. Text mode prints success to stdout or findings
to stderr. Errors print an authored prefix to stderr without success JSON;
argparse help/invalid options exit zero/two before auditing.

The audit modifies no policy, source, Git index or report. Index names,
worktree bytes and HEAD are separate observations, without atomic snapshot,
filesystem containment, license-content review or scientific admission.

```python
from pathlib import Path
from tools.check_source_headers import classify, expected_header, load_policy

policy = load_policy(Path("tools/source_header_policy.toml"))
assert classify(Path("probe.PY"), policy) == ("enforced", "")
assert classify(Path("README.md"), policy) == ("exempt", "prose-or-venue-source")
header = expected_header(Path("probe.lean"), "Native header example.")
assert len(header) == 9
assert header[0] == "/-" and header[-1] == "-/"
```

## Rust toolchain declaration contract

`tools.check_rust_toolchain_contract.check_rust_toolchain_contract(root: Path = ROOT)
-> list[str]` checks one repository's `rust-toolchain.toml` and seven fixed
workflow files. The default root resolves from this script; explicitly supplied
relative paths resolve from caller cwd. Reads retain ordinary symlink behaviour.
The fresh returned list belongs to the caller. No files are written, compiler or
network process is launched, or numerical/controller state is inspected.

The local TOML must contain only `[toolchain]`, with stable channel `1.98.0`,
ordered components `["clippy", "rustfmt"]`, and profile `"minimal"`. Six workflows
declare seven stable Rust action steps; `fuzz-nightly.yml` declares two
`nightly-2026-08-18` steps. Counts, literal channel strings, exact stable
`"rustfmt, clippy"` component strings, absent nightly components, and lowercase
40-hexadecimal action refs are checked. Ref syntax does not authenticate action
content or installed compiler versions.

Workflow YAML is composed into nodes without constructing tags. Only actual
`jobs → steps` action mappings count; each action's own `with` mapping supplies
its inputs. Named steps and ordinary aliases work. Run-block text, comments,
environment values and subsequent steps do not supply Rust settings. Relevant
mappings require unique ordinary string keys without merge keys, step lists
must be sequences, and relevant action values must be literal string scalars.
Malformed YAML, multiple documents, ambiguous structure, TOML decoding errors
and unreadable/missing inputs produce findings. Invalid UTF-8 propagates its
native `UnicodeError`. This declaration policy does not validate the complete
Actions schema or prove conditional reachability, execution, installation,
native parity, benchmark performance, or scientific admission.

`main(argv: list[str] | None = None) -> int` exposes `--root`, prints `FAIL:`
findings and returns one, or prints the pinned-policy success line and returns
zero. None reads process arguments. Argparse help exits zero and malformed or
unknown options exit two; decode errors propagate. The maintained standalone
command is `python tools/check_rust_toolchain_contract.py`. Preflight and the
static-governance capability job invoke it. That job installs hash-pinned
`requirements/ci-lint.txt` before policy checks so PyYAML is available.

```python
from pathlib import Path
from tools.check_rust_toolchain_contract import check_rust_toolchain_contract, main

assert check_rust_toolchain_contract() == []
findings = check_rust_toolchain_contract(Path("."))
assert findings == []
findings.append("caller annotation")
assert check_rust_toolchain_contract() == []
assert main(["--root", "."]) == 0
```

The example reads current declarations from repository cwd; it installs or runs
no Rust toolchain. Regression tests exercise actual public API roots and the
byte-identical standalone script in physical fixture trees.

## Private documentation declaration guard

`tools.check_docs_internal_private` supplies source-tree public inspectors and
`main(argv=None)`. The inspectors accept text or exact repository-relative path
lists and return fresh ordered `list[str]` findings; an empty list accepts that
input. They do not mutate caller inputs or retain state. There are no numeric
units, shapes, or clock contracts.

`check_gitignore_rules(text)` requires an exact root rule that ignores
`docs/internal` as a directory: the name followed by `/` or by `/**`.
Leading whitespace is significant; trailing
unescaped spaces are ignored. Literal negations mentioning `docs/internal` are
forbidden anywhere. All other negations must precede the final protective rule.
This conservative policy avoids proving arbitrary glob intersections.

`check_mkdocs_excludes_internal(text)` composes one YAML document into nodes
without constructing custom tags. It requires an ordinary top-level mapping
with unique ordinary string keys and no YAML merges. `exclude_docs` must be an
ordinary string containing an exact `internal/**` line after every negation.
Anchors referring to ordinary strings are accepted; unrelated custom-tagged
values are parsed without execution. Comments, unrelated fields, nested keys,
a pattern that repeats the `docs` prefix before `internal/**`, and inline
pattern comments do not supply the required
exclusion. Malformed YAML and refused declarations return authored findings.
The policy deliberately requires the canonical pattern even where other
patterns would exclude the same paths. MkDocs interprets these patterns
relative to `docs_dir` using ordered Gitignore syntax; see its
[configuration reference](https://www.mkdocs.org/user-guide/configuration/#exclude_docs).

`check_no_tracked_internal_paths(paths)` and
`check_no_history_internal_paths(paths)` inspect the literal `docs/internal`
prefix and its descendants. Similar names such as `docs/internal-extra` are
outside the prefix. Findings retain duplicates and sort exact names, including
Unicode and embedded newlines. The history inspector reports at most 50 names
and an ellipsis when truncated; it performs no rewrite.

```python
from tools.check_docs_internal_private import (
    check_gitignore_rules,
    check_mkdocs_excludes_internal,
    check_no_tracked_internal_paths,
)

assert check_gitignore_rules("!docs/public.md\ndocs/internal/\n") == []
assert check_mkdocs_excludes_internal("exclude_docs: |\n  !.assets\n  internal/**\n") == []
assert check_no_tracked_internal_paths(["docs/internal-extra/public.md"]) == []
assert check_mkdocs_excludes_internal("exclude_docs: |\n  internal/**\n  !internal/private.md\n")
```

`collect_errors(check_history=True)` reads the repository containing the script,
using UTF-8 for `.gitignore`, `mkdocs.yml`, and Git output. It inspects actual NUL-delimited
Git index paths and, by default, paths across every locally available history
reference. Findings are ordered by ignore, index, history, and MkDocs checks.
Missing configuration files are findings. Native filesystem, decoding, and
Git-process failures propagate; they never become PASS. Files and Git state
are unchanged.

```bash
python tools/check_docs_internal_private.py
python tools/check_docs_internal_private.py --skip-history
```

CLI findings print to stdout and return 1. Acceptance returns 0 and names the
checked scope; `--skip-history` explicitly says history was not checked.
Argparse retains help/invalid-option exits 0/2. PyYAML is a development dependency,
already pinned in the CI lint and test tooling profiles. Static-governance CI,
local preflight, the always-running local pre-commit hook, and the Pages build
invoke the full-history command. The two GitHub jobs fetch complete history.
The ordinary docstring gate also checks every definition in the guard and its
public test module. This guards local declarations and paths; it does not inspect
plugins, alternate configurations,
built-site bytes, distribution archives, or remote publication. No
installed-package console-script alias is provided.

## Independent Collision Coefficient Reference

`validation.gk_collision_independent_reference` is an importable source-tree
reference API. It has no installed command or CLI and performs no file I/O.
`chandrasekhar_g` and `deflection_shape` accept float64-convertible arrays of
dimensionless `x = v/v_th`, preserving arbitrary shape, including scalars and
empty arrays. The former accepts zero, the latter requires strictly positive
elements; both reject nonfinite elements with authored `ValueError`. Outputs
are fresh float64 arrays; strided and read-only inputs retain their bytes.

The scalar rate APIs use these units and domains:

| Input or result | Unit and contract |
| --- | --- |
| `mass_amu`, `field_mass_amu` | Positive finite multiples of proton mass; the historical name does not use the atomic mass constant. |
| `charge_e` | Finite multiples of elementary charge; signed and zero values are supported through the fourth power. |
| `temperature_keV`, `T_e_keV` | Positive finite test/field temperatures in keV. |
| `n_field_19`, `n_e_19` | Positive finite field density in 10^19 m^-3. |
| `z_eff`, `ln_lambda` | Positive finite dimensionless effective charge and Coulomb logarithm. |
| `n_quad`, `x_max` | Python integer >= 2 excluding bool; finite positive dimensionless speed limit. Defaults are 64 and 10. |
| Collision rates | Scalars in s^-1, without `v_th/R` normalisation. |
| Elastic efficiency, field-temperature factor | Dimensionless; the default field mass is the electron/proton mass ratio. |

`basic_collision_frequency` returns the retained normalising rate;
`thermal_deflection_rate` multiplies it by effective charge and a fixed
Gauss–Legendre Maxwellian average. `braginskii_collision_rate` provides the
separate retained closed-form coefficient. `elastic_energy_transfer_efficiency`
uses the supplied two masses. `independent_collision_rates` assembles the
three rates and two factors in a fresh frozen `IndependentCollisionRates`.
Direct dataclass construction is unchecked; frozen only prevents reassignment.

Invalid finite/positive domains raise authored `ValueError`. Scalar `float`
conversion, resource failures and extreme arithmetic retain native behaviour.
There is no adaptive quadrature error estimate, upper count/resource bound or
guarantee of representable results for every finite input. Truncation and
resolution must be assessed together. Functions do not retain mutable state
or modify caller inputs; callers must prevent concurrent mutation of inputs.

```python
from validation.gk_collision_independent_reference import independent_collision_rates

rates = independent_collision_rates(
    mass_amu=2.0, charge_e=1.0, temperature_keV=8.0, n_e_19=10.0, T_e_keV=4.0
)
assert rates.thermal_deflection_rate > 0.0  # s^-1
assert rates.braginskii_rate > 0.0         # s^-1
assert rates.energy_relaxation_rate > 0.0  # s^-1
```

`tests/test_gk_collision_independent_reference.py` exercises this example,
array ownership, scalar domains and both retained rate routes. Structural
independence from the production implementation is a local cross-check; cited
literature is not a checksum-bound external numerical case. This API does not
establish a conserving collision operator, quantitative damping, training or
facility admission. The separate report producer
`validation.validate_gk_collision_independent` retains its own API/CLI contract.

## MAST candidate channel and dataset workflow

The maintained source-tree modules are
`validation.disruption_channel_recipes`,
`validation.build_disruption_replay_channels` and
`validation.build_mast_disruption_dataset`. They provide a local
mirror-to-channel-to-proxy-label workflow; they do not download data or train
a model. The [MAST source-authority gates](validation.md#mast-toroidal-field-authority)
remain separate requirements.

| Surface | Inputs and numerical behaviour | Results and limits |
| --- | --- | --- |
| `amperes_to_megamperes`, `per_1e19` | Finite arrays, converted to float64 and divided by 10^6 or 10^19. Signed values and arbitrary shapes are retained. | MA or 10^19 m^-3; zero-dimensional inputs follow NumPy's scalar behaviour. |
| `toroidal_harmonic`, `n_mode_amplitude` | Finite `(samples, coils)` fields in T and a matching angle vector in rad; positive Python integer mode excluding bool, strictly below coils/2. Projection is `(2/N) sum b exp(-i n phi)`. | Fresh complex128 harmonic or float64 magnitude in T. Empty sample rows are supported. Supplied coil spacing is not authenticated. |
| `locked_mode_envelope` | Positive Python integer window no longer than the trace; the pure recipe also permits even windows. The n=1 phasor is boxcar averaged. | Fresh T magnitude, with NumPy `same` convolution and zero-padded edges. Window selection is a caller decision, not evidence of stationarity. |
| `dbdt_gauss_per_s` | Matching finite 1-D field in T and time in s, at least two samples and a strictly increasing clock. NumPy's default gradient is multiplied by 10^4. | Fresh G/s trace; no smoothing or source-quantity attestation. A source already measuring T/s must not be differentiated again. |
| `q_at_psi_norm` | Finite 1-D profile or 2-D profile rows; nonempty finite unique flux knots, sorted internally; finite target default 0.95. | Fresh dimensionless values from linear interpolation, including endpoint clamping, single-knot and empty-row support. Flux normalisation is supplied, not verified. |
| `vacuum_toroidal_field` | Finite current in A, positive finite radius in m and positive Python integer turns excluding bool. No machine geometry default. | T with the input shape, using the retained mu0 N I/(2 pi R) formula. The caller must verify geometry provenance. |

Recipe inputs retain their bytes, including read-only and strided views.
Authored `ValueError` rejects invalid domains; conversion, resource and extreme
arithmetic retain native behaviour. Finite inputs do not guarantee
representable outputs for every extreme value.

`derive_replay_channels(mirror, locked_window=201)` consumes the acquisition's
dotted array keys. Summary, equilibrium and saddle clocks must be finite and
strictly increasing; the summary clock needs at least two samples. Summary
current/density use A and m^-3; equilibrium values use their declared units;
saddle geometry uses degrees. The first poloidal probe and the matching prefix
of the Mirnov clock are used by the historical derivative recipe.

The returned eleven fresh float64 vectors share `time_s`: Ip in MA, field in T,
dimensionless beta/q95, density in 10^19 m^-3, modal/envelope fields in T,
derivative in G/s and vertical position in m. Scalar equilibrium values use
linear interpolation and endpoint clamping. A value/time length mismatch
retains the historical uniform-grid alignment assumption. Fast candidates use
per-bin peak magnitudes and interpolation over empty bins. Historical missing
row replacement and final nonfinite-to-zero handling remain; neither grants a
canonical physical binding. Missing keys, invalid dimensions/clocks and
unusable signals are refused. There is no I/O in this function.

`build_channels` reads numerically ordered `shot_<positive-int64>.npz` mirrors
without pickle, rejects duplicate numeric identities (including 1/01), and
records malformed shots with a fixed authored failure reason. The builder
requires an odd positive Python integer window and a nonempty generated-at
label; the pure recipe's even-window support does not widen this builder
contract. The label is not parsed as a timestamp. The output directory must
resolve outside the immutable material directory.

A temporary uncompressed archive is reopened and checked before exclusive
hardlink publication as `channels.npz`; an existing destination is refused and
the temporary file is removed. An empty or entirely failed material inventory
can still produce an empty candidate archive. The v2 report's scientific,
training, facility and control admission flags remain false.

`inspect_replay_archive` and `inspect_replay_archive_bytes` require exactly
one `shot_ids` member plus eleven `<id>:<channel>` members per shot, with no
duplicate names or extra members. IDs have integer dtype, are sorted, unique
and positive within signed int64. Vectors have floating dtype, are finite,
nonempty and aligned, with strictly increasing time. Empty ID inventories are
supported. Optional expected IDs are compared directly. The result binds raw
bytes and canonical channel values, without authenticating the producer;
there is no archive-size limit. File inspection reads once and follows
symlinks. Invalid/read/decode inputs raise an authored `ValueError` subtype.

The dataset CLI also preserves historical floating ID vectors when every
value is exactly integral, finite and within positive signed int64; it never
truncates fractions. Channel and member rules otherwise match inspection.
`build_shot_npz` writes eleven measured vectors, three scalar legacy labels
and a self-digested `shot_label_record_json`. It ignores extra input mapping
keys. `derive_ip_quench_label` projects the full proxy record to a tuple;
the detector uses absolute Ip, the last sample at least 0.8 times its maximum,
and the first subsequent threshold crossing. Defaults are a drop fraction
0.8 and a window 5 ms, with finite fraction in (0, 1) and positive finite
window. No-current ambiguity is retained by the full label record.

`build_dataset` validates the entire nonempty batch before the first write,
including distinct positive Python int64 IDs, chronological vectors and proxy
domains. `dataset_id` is a single ASCII filename identifier beginning with a
letter/digit and containing only letters/digits/dot/underscore/hyphen.
The nonempty retrieval label and generated-at label are copied without clock
parsing. It writes shot archives and a validated local manifest, then returns
a v2 report; it does not write that report itself. Its dataset fingerprint
hashes the sorted shot-file checksums. Every label has `ip_proxy` authority;
`independent_label_count=0`, `status="blocked"` and
`admission_ready=False`. The `synthetic:false` field is the builder's input
declaration, not authentication of the input's origin.

The following manufactured software example writes temporary candidate files.
It demonstrates neither acquired MAST data nor admission:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from validation.build_mast_disruption_dataset import build_dataset
from validation.disruption_channel_recipes import (
    dbdt_gauss_per_s, n_mode_amplitude, vacuum_toroidal_field,
)

time = np.linspace(0.0, 0.03, 8, dtype=np.float64)
angles = np.arange(12, dtype=np.float64) * np.pi / 6.0
saddle = np.cos(angles)[None, :].repeat(time.size, axis=0)
channels = {
    "time_s": time,
    "Ip_MA": np.full(time.size, 0.6, dtype=np.float64),
    "BT_T": vacuum_toroidal_field(
        np.full(time.size, 1.0e6, dtype=np.float64), 0.8, n_turns=100
    ),
    "beta_N": np.full(time.size, 1.5, dtype=np.float64),
    "q95": np.full(time.size, 3.8, dtype=np.float64),
    "ne_1e19": np.full(time.size, 3.0, dtype=np.float64),
    "n1_amp": n_mode_amplitude(saddle, angles, 1),
    "n2_amp": n_mode_amplitude(saddle, angles, 2),
    "locked_mode_amp": n_mode_amplitude(saddle, angles, 1),
    "dBdt_gauss_per_s": dbdt_gauss_per_s(3.0 * time, time),
    "vertical_position_m": np.zeros(time.size, dtype=np.float64),
}
with TemporaryDirectory(prefix="mast-candidate-") as temporary:
    report = build_dataset(
        [{"shot_id": 101, "channels": channels}],
        dataset_id="candidate", out_dir=Path(temporary),
        retrieved_at="2026-07-10T00:00:00+00:00",
        generated_at="2026-07-10T00:00:00+00:00",
    )
    assert report["status"] == "blocked"
    assert report["independent_label_count"] == 0
```

From the repository root, the native module commands are:

```bash
python -m validation.build_disruption_replay_channels --help
python -m validation.build_mast_disruption_dataset --help
```

Each command returns 0 after successful candidate assembly, including a blocked
dataset report; help exits 0 and argparse errors exit 2. Direct Python
`main(argv)` calls preserve input/path and native I/O exceptions. The module
commands map caught value/I/O/type/key failures to fixed stderr sentences and
exit 2, without exposing interpreter exception text.

Replay report publication is exclusive and attempts rollback of its own
archive/report after a report-write failure. Dataset publication uses direct
writes, can overwrite existing destinations and has no rollback after later
I/O failure. Its CLI checks selected-input, module/helper-source and
report/artifact aliases through direct/resolved/symbolic/existing hardlink
identity before writing. The dataset Python writer does not enforce those CLI
path checks. These are sequential checks, not a race-proof filesystem
transaction or protection of every dependency.


## FAIR-MAST source policy

`validation.fair_mast_source_policy.fair_mast_provenance()` takes no arguments
and returns `dict[str, str | list[str]]`: `licence`, `licence_url`, `citation`,
`citations` and `source_policy_url`. The licence identifier is `CC-BY-SA-4.0`;
the citation list contains the 2025 IEEE TPS service paper followed by the 2024
SoftwareX FAIR-MAST paper. The combined citation string joins those entries
with `"; "`. The public `FAIR_MAST_*` constants hold the same recorded values.

Each call creates a new dictionary and citation list, so caller edits cannot
change subsequent blocks. There are no numeric units, array shapes, clock
fields or hidden I/O: the function copies local constants and neither fetches
the catalogue nor checks an asset's licence, source bytes or scientific
admission. The [catalogue policy](https://mastapp.site/#license) has exceptions
for individually noted assets. Acquisition and lineage producers copy this
default declaration; consumers must retain any asset-specific policy evidence.
There is no command-line interface or maintained foreign-language counterpart.

```python
import json
from validation.fair_mast_source_policy import fair_mast_provenance

first = fair_mast_provenance()
second = fair_mast_provenance()
citations = first["citations"]
assert isinstance(citations, list)
citations.clear()
assert len(second["citations"]) == 2
assert second == fair_mast_provenance()
assert json.loads(json.dumps(second, allow_nan=False)) == second
```

## MAST native source acquisition

The source-tree `validation.acquire_mast_disruption_shots` APIs retain exact
selected sample values in derived NPZ files and construct a source-object v2
manifest. `parse_shots(text)` expands comma/whitespace-separated ASCII decimal
identities and inclusive ascending ranges, preserving supplied order. The
selection must contain 1 to 100000 unique positive int64 identities. Blank,
descending, duplicated, fractional and oversized selections raise `ValueError`
before acquisition. The historical `_parse_shots` name remains an alias.

`SourceGenerationPin` is a frozen checked declaration: exact FAIR-MAST shot URI,
lowercase SHA-256, positive Python byte count at most 16 MiB and string/None HTTP
headers. Construction and `to_dict()` read no source and authenticate no data.
`read_source_generation(shot_id)` performs an uncached fixed-origin HTTPS request
with a 30-second socket timeout and a 16 MiB root-metadata bound. It hashes the exact
raw bytes, requires UTF-8 JSON without duplicate keys/nonfinite numeric literals,
and checks integer Zarr version 3 and inline consolidated metadata. It does not
validate every node or establish immutable chunk bytes. Advisory ETag/Last-Modified
values are excluded from generation equality.
`decode_source_generation(shot_id, raw, etag=None, last_modified=None)` replays
the same selected metadata checks against exact captured bytes without network
I/O. The uncached reader calls this decoder; successful decoding authenticates
neither the supplied byte source nor any chunk inventory.

`make_filesystem(cache_dir)` configures fsspec simplecache with anonymous S3 and
the fixed FAIR-MAST endpoint; it creates the directory, but does not prove network
availability or Zarr-v3 support. The default opener bridges synchronous
simplecache through fsspec's async wrapper so cached S3 operations stay on their
own I/O loop when called by Zarr 3. Other filesystem protocols retain their
mapper behaviour. `mirror_shot(fs, shot_id, open_group=...,
metadata_out=None)` reads four selected xarray groups without resampling, clock
alignment or numerical calibration. Present values keep source dtype/shape and
may alias provider buffers. Object dtype and absent/empty/non-2D saddle arrays
refuse before archive publication. Available dimensions, units, time-dimension
names, attributes and chunks are recorded separately; unsupported attributes or
keys that collide after JSON conversion refuse. NumPy scalars whose `item()`
still returns a NumPy scalar, including extended-precision metadata, raise
`TypeError` without recursive conversion or silent precision loss.
Caller metadata can be partially
populated on error; opener/connection lifetimes remain the provider's responsibility.

`acquire(shot_ids, out_dir=..., cache_dir=..., generated_at=..., retrieved_at=...,
make_fs=None, open_group=None, read_generation=None)` copies/preflights the
selection and both nonempty string labels before any output. Each shot gets a
new cache namespace bound to its labels and root hash; repeated namespaces refuse.
The root URI must identify the requested shot and its content identity must match
before/after collection. Source errors become failed-shot records. The explicit
`SourceGenerationError` refusal type retains authored messages without underlying
transport/JSON exception text; other caught failures use the fixed sentence
"Could not acquire the requested MAST shot." Export and final artifact/manifest
errors propagate. Numeric/nonobject NPZ output is written
directly, overwriting unrelated same-shot files. Earlier shot files and cache
namespaces remain on later failure. Selected output/report paths cannot alias
the acquisition owners or selected NPZs through direct/symlink/hardlink paths;
these sequential checks do not prevent concurrent replacement or protect every
dependency. The source module CLI returns 0 for complete acquisition, 1 for a
partial/empty report, and authored stderr and exit 2 for caught value/I/O/type failures.

The defining owners are `validation.mast_replay_contracts._acquisition_source`,
`_acquisition_arrays`, `_acquisition_selection` and `_acquisition_report`; the
historical source-module names remain explicit reexports. The report's fixed
`synthetic=False`, source and licence fields describe the declared acquisition
policy. Caller-supplied native adapters, matching digests, successful software
checks and a complete report do not authenticate original data or admit training.
The retained DEFUSE access/range text is historical policy rather than a current
network observation. Install the `mast-acquisition` extra in a separate source
checkout environment for Zarr 3, xarray, fsspec and s3fs. The `mast-data`, `all`
and `dev` extras constrain Zarr 2 for the captured-store conversion profile and
cannot be combined with `mast-acquisition` in one environment. Package resolution
selects a Zarr release compatible with the supported Python interpreter; recent
Zarr releases require NumPy 2. Installation does not establish live source
availability or scientific training admission.

```python
from validation.acquire_mast_disruption_shots import parse_shots, SourceGenerationPin

assert parse_shots("30424,30419-30421") == [30424, 30419, 30420, 30421]
pin = SourceGenerationPin(
    "s3://mast/level2/shots/30421.zarr", "a" * 64, 64, None, None
)
assert pin.to_dict()["bytes"] == 64  # Caller declaration; no source read.
```

```bash
python -m pip install '.[mast-acquisition]'
python -m validation.acquire_mast_disruption_shots --help
```

## SOL algebraic diagnostic

`validation.validate_sol_two_point` preserves its public configuration/result
dataclasses, scalar error checks, `validate_sol_two_point`, `build_evidence`,
`validate_evidence_payload`, schema constant and `main` imports. These are
source-checkout validation interfaces. The cohesive owners are
`validation.sol_two_point_contracts.models` (inputs and numerical checks),
`validation.sol_two_point_contracts.evidence` (complete v1 content and self-digest)
and `validation.sol_two_point_contracts.command` (CLI and readable report).

`SOLConfig(r0, a, q95, b_pol)` is immutable and requires positive finite Python
numbers, excluding booleans and strings, with `0 < a < r0`. Radius uses metres,
field uses tesla, and q95/epsilon are dimensionless. `model()` constructs a fresh
production model. The illustrative default R0=1.7 m/a=0.5 m is not a declared
facility geometry. `validate_sol_two_point(config=None, operating_points=...,
detachment_q_par_mw_m2=100.0, detachment_l_par=20.0, exact_tol=1e-9)` accepts
a nonempty sequence of `(power_MW, upstream_density_1e19_m^-3)` pairs, positive
finite parallel flux in MW/m^2 and length in metres, and a positive finite
dimensionless strict tolerance. Malformed inputs or nonfinite diagnostics raise
`ValueError`; extreme production arithmetic can raise `OverflowError` or
`ZeroDivisionError`. Every call computes fresh, timestamp-free observations.

The connection, mapping, conduction, pressure, peak-flux and scaling error
checks return dimensionless errors against shared production formulas. Scaling
observations keep four ordered exponent names; epsilon doubles below 0.5 and
halves otherwise so probes respect the production domain. Detachment observes
0.99/1.01 of the analytic critical density, stored in `1e19 m^-3`. Result flags
use strict `error < exact_tol`, the two detachment probes and their conjunction.
This is a local algebraic diagnostic using shared constants/Eich width; no
independent physical reference, facility, safety, controller or training
admission follows from a passing result.

`build_evidence(result, target_id=...)` returns a fresh complete v1 mapping with
the caller's nonempty descriptive target string and a UTC wall-clock timestamp
at second resolution. It checks finite numeric domains and flag consistency
before sealing. `validate_evidence_payload(mapping)` checks the exact v1 field
set and shapes, digest, timestamp format, ordered scaling ratios/arithmetic,
maximum error and every strict gate. It returns the literal outcome (`False`
for a coherent failure) or raises `ValueError`. A self-digest does not
authenticate a producer or source. V1 does not record all detachment input
parameters; its reader checks declared observations without re-running physics,
proving clock freshness or admitting scientific data. A caller decoding JSON
must reject duplicate keys before decoding loses them.

```python
from validation.validate_sol_two_point import (
    SOLConfig, build_evidence, validate_evidence_payload, validate_sol_two_point,
)

result = validate_sol_two_point(
    config=SOLConfig(r0=1.7, a=1.0, q95=3.5, b_pol=0.4),
    operating_points=((10.0, 3.0),),
)
evidence = build_evidence(result, target_id="local-algebraic-diagnostic")
assert result.passed and validate_evidence_payload(evidence)
failed = build_evidence(validate_sol_two_point(exact_tol=1e-30), target_id="strict")
assert validate_evidence_payload(failed) is False
```

`main(argv=None)` supports module/direct-script checkout entry points;
`--json-out` selects JSON stdout and `--exact-tol` exercises the public threshold.
`--report` selects caller-relative JSON and sibling `.md` outputs. Selected
sources and output/output aliases, including symbolic/hard links, refuse before
solving; persistent evidence roots require recorded campaign custody. Unrelated
files are replaced JSON then Markdown, without atomic-pair/locking/snapshot
guarantees; an I/O refusal can leave an earlier JSON file. CLI returns zero for
pass, one for diagnostic failure and two for supported input/custody/arithmetic/
I/O refusal. Parser help/usage retain exits zero/two. `render_markdown` from the
command owner renders checked passing or failing content and escapes target
line breaks as JSON text.

## Release-artifact builder

`tools.build_release_artifacts` is a source-checkout tool. Its public interfaces
are `ArtifactSummary(path, entries, sha256)`, `normalise_sdist(path, epoch)`,
`validate_artifact(path)`, `build_release_artifacts(outdir, *, epoch,
sdist_only=False)` and `main(argv=None)`. The frozen dataclass records the
caller-spelled path, archive entry count and compressed SHA-256; constructing
it directly does not validate fields or authenticate a producer.

`normalise_sdist` rewrites an existing gzip tar with filename-sorted members,
canonical ownership, cleared PAX overrides and an integer Unix epoch from zero
through `2**32 - 1`. Booleans and other types refuse. Payloads are buffered in
memory; modes and duplicate names are retained. Member inspection finishes
before rewriting. An exclusively created sibling temporary file is closed
before native replacement and cleaned afterward. Existing legacy fixed-name
temporary files are preserved. Native filesystem/symlink behaviour applies;
callers coordinate directories, and no fsync or directory lock is provided.

`validate_artifact` accepts `.whl` and `.tar.gz`. Both refuse absolute, parent,
Windows drive/backslash and private/build-only member names. Tar members must
be regular files or directories. Wheels require exactly one UTF-8 METADATA
with the literal canonical license line and one entry-point declaration;
declared console modules must exist as a Python module or package. Missing
console sections return an empty declaration set. Counts include directories
and duplicate names; empty tar inventories are accepted. Inspection neither
extracts/imports payloads nor verifies callable existence, full wheel schemas,
all CRC/RECORD entries, signatures or release admission.

`build_release_artifacts` validates the epoch before creating outputs and
refuses existing `.whl`/`.tar.gz` files. It launches `sys.executable -m build`
with native argv, the script repository as cwd and the inherited environment
with `SOURCE_DATE_EPOCH` replaced. The real backend may install isolated build
requirements and execute project hooks. The result must contain two artifacts,
or one with `sdist_only=True`; produced sdists are normalised and every
artifact is inspected. Summaries follow filename order. No build timeout,
coherent source snapshot, directory lock or cross-backend/platform byte
reproducibility guarantee is supplied. Failure retains produced artifacts.

```python
from pathlib import Path
from tools.build_release_artifacts import normalise_sdist, validate_artifact

# Inspect an existing caller-owned source distribution.
archive = Path("dist/package.tar.gz")
normalise_sdist(archive, 1_700_000_000)
summary = validate_artifact(archive)
print(summary.path, summary.entries, summary.sha256)
```

```bash
python tools/build_release_artifacts.py --outdir dist --source-date-epoch 1700000000
python tools/build_release_artifacts.py --outdir source-dist --sdist-only
```

The CLI chooses the explicit epoch, `SOURCE_DATE_EPOCH`, then actual Git HEAD
timestamp in that order. It prints tab-separated filename, entry count, SHA-256
and epoch, returning zero on success. Help exits zero; usage and epoch-range
errors exit two. Invalid environment/Git data, absent Git, native build/I/O and
archive/declaration exceptions propagate. Public API refusal uses `ValueError`;
native `OSError`, archive, decoding/configuration and subprocess errors remain
visible to callers. A passing result does not publish an artifact.
