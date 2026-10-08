## Validation entry map

Shared reference-location checks are lexical declaration policies. See
[Shared Reference Location Declarations](api.md#shared-reference-location-declarations)
for scheme/prefix rules, malformed/control-character refusals and limits.
They do not resolve symlinks, decode percent escapes, download references or
authenticate executable bytes. Passing URI/path syntax supplies no scientific
admission; the owning consumer remains responsible for actual byte custody.

This page is the admission surface for all bounded and measurable claims.
Use it as the source of truth before publishing any benchmark, timing, or
physics improvement statement.

- **Local evidence**: deterministic fixtures, repository benchmarks, and local
  manifests.
- **Admitted evidence**: outputs that pass strict validators and match declared
  claim boundaries.
- **Blocked claims**: outputs that remain local-only, synthetic, or missing
  external-code / facility-level evidence.

The practical review workflow is:

1. run the relevant checker for the changed surface,
2. inspect the report for required digests, context, and boundaries,
3. only then promote the result into planning or investor-facing material.

The [public MAST evidence manifest](mast_evidence_manifest.md) binds the six
MAST method pages to exact report schemas and digests without exposing raw
traces, private storage, or operational coordination paths.

Public documentation links are admitted in two stages. The deterministic local
gate resolves source files, anchors, MkDocs navigation, manuscript references,
public metadata, and rendered HTML without network access. The scheduled
external gate then records bounded HTTP checks with cache and policy provenance;
transient or access-restricted responses remain distinct from confirmed
permanent failures.

## MAST toroidal-field authority

The toroidal-field admission gate is documented in
[MAST toroidal-field authority](mast_toroidal_field_authority.md). It separates
the total `bphi_rmag` source from the vacuum `bvac_rmag` candidate, requires the
paired magnetic-axis radius, and fails closed until physical sign and
one-standard-deviation uncertainty are source-attested. It never promotes a
field from its name, units, observed sign, or approximate magnitude.

## MAST normalised-beta authority

The normalised-beta admission gate is documented in
[MAST normalised-beta authority](mast_normalised_beta_authority.md). It records
the conflict between the live `EFM_BETAN` unit and the current IMAS definition,
forbids a fabricated numerical unit conversion, and keeps canonical `beta_N`
blocked until formula inputs, sign and scale, reconstruction quality,
negative-value validity, and one-standard-deviation uncertainty are
source-authoritative.

## MAST saddle-modal authority

The saddle-modal admission gate is documented in
[MAST saddle-modal authority](mast_saddle_modal_authority.md). It pins the
twelve `ASM_SAD/M01..M12` candidates and all three source geometry surfaces,
verifies the observed polygon centres and finite row coverage, and performs no
modal reduction until the field-row join, vertical set, released geometry,
calibration uncertainty, baseline, and bad-channel policies are source-attested.

## MAST locked-mode authority

The locked-mode admission gate is documented in
[MAST locked-mode authority](mast_locked_mode_authority.md). It binds the MAST
description of a growing stationary n=1 radial perturbation on the outer-midplane
saddle array to the saddle row/geometry contract. It measures the historical
201-sample boxcar's physical time scale but performs no filter or locked-mode
reduction until component, frame, filter/edge, background/pickup/vessel correction,
calibration uncertainty, and estimator evidence are source-attested.

## MAST dB/dt source authority

The dB/dt source-quantity admission gate is documented in
[MAST dB/dt source authority](mast_dbdt_authority.md). It records the conflict
between the official `T` mapping and live `Tesla/sec` label, measures the five
centre-column Mirnov rows and their 500 kHz timebase, and executes no signal
transform. A content-digested source contract must select exactly one path:
differentiate an attested `T` magnetic field once, or convert an attested `T/s`
derivative without differentiating it. Ordered row/geometry identity, scale
semantics, component/orientation/sign, reduction and missing-data policies,
filter/edge policy, calibration uncertainty, and released geometry remain
mandatory.

## MAST candidate replay and proxy-label assembly

The [candidate workflow contract](api.md#mast-candidate-channel-and-dataset-workflow)
documents the numerical recipes, archive schema, identity/clock rules,
native commands and file-publication limits for
`validation.build_disruption_replay_channels` and
`validation.build_mast_disruption_dataset`.

A valid local candidate still carries false scientific/training/facility/control
admission flags. Proxy labels derive from Ip, an input feature, and are neither
independent facility outcomes nor calibrated probabilities. Historical
missing-row replacement, uniform-grid alignment fallback and per-bin reduction
do not satisfy the toroidal-field, normalised-beta, modal, stationary-estimator
or dB/dt source-authority gates above. The builder's `synthetic:false` declaration
does not authenticate acquired input origins. The API example uses explicitly
manufactured data only.

## Local-first physics debug assistance

`scpn_control.physics_debug` is an advisory triage boundary for physics
validation gaps. `ProviderPolicy` defaults to loopback-only model gateways;
remote or facility gateways require an explicit endpoint allowlist.
`build_local_provider()` provides onsite profiles for chat-completions-compatible,
Ollama-style chat, direct JSON, and text-generation gateways while keeping the
host loopback-only by default. Evidence is redacted before prompting, provider
output must cite supplied evidence, prompt-injection findings are neutralized
before provider prompting, every hypothesis must include a falsification test,
and campaign suggestions must declare measurements, stop conditions, and risk
controls. Persisted reports use
`scpn-control.physics-debug-report.v1` with a canonical SHA-256 payload digest.
Optional hallucination guardrail review uses `build_guardrail_provider()` with
a `director-ai` default profile and explicit alternate profiles for lab-owned
guardrail solutions. Guardrail block decisions fail closed before report
persistence, while allow findings are recorded in the same tamper-evident
report digest with the reviewed provider-draft SHA-256. High-severity
guardrail findings require block actions, and admitted guardrail reviews must
meet the configured risk-control minimum. Guardrail request metadata binds the
provider, safety policy, and guardrail policy digests so admitted reviews cannot
be replayed across a different provider or a relaxed policy.
`PhysicsDebugSafetyPolicy` binds mandatory human review, caps advisory
confidence, and rejects provider text that attempts controller promotion,
actuation, review bypass, or approval claims before evidence can be persisted.
`run_provider_quorum()` records every provider report digest and emits
`scpn-control.physics-debug-quorum-report.v1` only when enough providers
corroborate the same gap and evidence set while meeting the required local
provider count. These reports are not validated physics truth,
controller-parameter promotion, or facility safety approval.

## Lean proof evidence admission

Lean 4 formal-verification reports use the
`scpn-control.lean4-formal-report.v1` schema and are admitted only as bounded
evidence for the current PID actuator-saturation and SNN marking-bound proof
surface. The public report loader and validator reject duplicate JSON keys,
non-Lean solver declarations, Lean solver
strings that do not include the declared `lean_version`, unsupported
`proved_contracts`, unsafe report paths, malformed theorem identifiers, missing
PID/SNN namespace coverage, unbounded proof assumptions, and certification
overclaims. It also rejects unrelated theorem namespaces, production module
references, and safety-case IDs instead of accepting padded reports.
Safety-critical `.scpnctl` artifact admission must still bind the report
digest, compiled artifact digest, Lake file digest, proof-source digest, checked
specifications, theorem namespaces, production module references, and bounded
proof assumptions before the artifact can be loaded with
`require_formal_verification=True`. The `module_paths` report field should use
importable module names such as `scpn_control.control.pid_controller` so
installed packages do not require repository `src/` paths; legacy safe relative
source paths remain accepted for existing reports. The
artifact manifest gate applies the same Lean solver/version and exact-link
checks before optional report-root byte comparison runs, and report-root Lean
files use the same duplicate-key-safe loader as direct report validation. Lean
reports and artifact formal-verification manifests reject unknown proof fields
instead of silently ignoring stale or foreign evidence. External Lean report
payloads must carry the canonical `payload_sha256` self-digest; reports that
omit it are not admissible safety-case evidence.

## Z3 proof evidence admission

Z3 formal-verification reports use the
`scpn-control.z3-formal-report.v2` schema and are admitted only as bounded SMT
evidence for compiled Petri-net transition relations. The public report loader
rejects duplicate JSON keys, unknown top-level fields, unknown proof-section
fields, malformed counterexample records, duplicate or malformed section
`checked_specs`, blocked reports that carry proof depth or live solver labels,
pass/fail reports that do not identify `z3-solver`, pass/fail reports that
reuse the unavailable-solver label, and inconsistent solver-state
combinations. The safety section is a single counterexample query: `unsat`
must hold without a counterexample and `sat` must fail with one. Temporal
sections may combine universal counterexample searches with existential witness
searches, so they publish `mixed` when successful obligations return both
statuses and `not-run` when no temporal obligation was submitted. `unknown`
never holds and carries no counterexample because it is not a discovered
violation path. An `EventuallyFires` obligation is accepted only under exact
unit transition weights; non-unit bounds are rejected because fractional token
flow is not a discrete firing witness. Version 1 reports must be regenerated and
are rejected by the v2 loader because their aggregate status does not preserve
that distinction. Safety-critical `.scpnctl` artifact admission applies the same
Z3 report loader before matching the manifest status, solver, bounded depth,
checked specifications, report digest, and compiled artifact digest.

## Quantum disruption bridge

`scpn_control.control.quantum_disruption_bridge` keeps quantum circuit,
Qiskit/PennyLane, and provider-specific execution in `scpn-quantum-control`.
SCPN-CONTROL exposes only a control-grade facade with lazy optional imports,
strict CONTROL-to-ITER feature mapping, explicit centre-default provenance,
bounded amplitude-kernel reports, and checksum-bound advisory disruption
reports. The facade fails closed when the optional quantum owner dependency is
unavailable, records `status="quantum-unavailable"`, and never admits a
control action. Missing ITER fields must be supplied explicitly unless
`allow_center_defaults=True` is set for bounded fallback evidence. Public
facility-validation or publication claims remain blocked until external
disruption databases and benchmark artefacts are supplied. Bridge reports carry
admission evidence with CONTROL-feature, ITER-feature, and feature-mapping
digests, explicit default-use reasons, and required external evidence entries
for measured disruption databases, quantum backend benchmarks, and classical
baseline comparisons. Each bridge or kernel report also carries a
schema-versioned advisory certificate that binds report kind, CONTROL facade
ownership, quantum backend ownership, claim-boundary digest, downstream
non-admission policy, and report-content digest before the outer tamper seal is
accepted. The matching dependency contract names the expected
`scpn-quantum-control` module, classifier API, feature ordering, Qiskit core
dependencies, optional provider families, and downstream non-admission policy
so backend work can evolve without silently drifting from the CONTROL facade.
Each report embeds the dependency contract used for that evaluation and binds
the contract digest into the advisory certificate before payload validation
continues. If the optional quantum backend exposes a bridge-contract callable,
CONTROL records whether the backend contract matched, was not exposed, or was
unavailable; an exposed mismatching backend contract is treated as a fail-closed
runtime error. Bridge reports additionally carry advisory decision evidence
with score-basis provenance, deterministic risk-band thresholds, backend
contract-validation state, blocked control action, and a certificate-bound
decision digest so downstream tooling cannot treat a risk score as admitted
control evidence.
The report validator recomputes the classical score from the mapped features
and checks that the risk score equals its selected source score after checking
the digest chain. It also cross-checks backend availability against quantum-score
presence and backend attestation. Recomputing the unkeyed digests cannot bypass these numerical
relationships, but the digests alone do not authenticate a quantum backend or
independent disruption data.

## Federated disruption synthetic multi-facility benchmark

Run:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family federated-disruption \
  --artifact report=validation/reports/federated_disruption_benchmark.json \
  --artifact markdown=validation/reports/federated_disruption_benchmark.md \
  -- python validation/benchmark_federated_disruption.py
```

Outputs:

- `validation/reports/federated_disruption_benchmark.json`
- `validation/reports/federated_disruption_benchmark.md`

Scope: deterministic synthetic DIII-D/JET/KSTAR/EAST facility distributions,
FedProx aggregation, and a nominal Gaussian update-noise calculation. The
in-process array factory does not enforce remote data isolation; returned
facility metrics and this nominal calculation do not establish end-to-end
differential privacy.
The public state replay tests require schema version 2, both random streams,
finite model parameters, and internally consistent nominal ledger records.
Malformed snapshots and duplicate or undeclared clients are rejected before
they can enter another training round. These checks do not authenticate the
source of a snapshot or supply an independent facility-data provenance proof.
This is not measured cross-facility validation; measured claims remain blocked
until external facility shot databases and provenance manifests are supplied.

# Validation and QA

## Python tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/ -q
```

Coverage gate (matches CI threshold):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin -p pytest_cov tests/ --cov=scpn_control --cov-report=term --cov-fail-under=100
```

The **100% configured package coverage gate for admitted CI contexts** keeps
statement and branch `fail_under = 100` in `pyproject.toml` and CI. It applies
to the merged baseline and Rust-present contexts, not to unavailable hardware,
facility data, external services, or optional packages without an executing
lane. New recovery work
must add module-specific behavioural tests for concrete production surfaces
rather than synthetic line-hit tests.

## Rust workspace checks

```bash
cd scpn-control-rs
cargo build --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

## Rust/Python interop checks (PyO3 + maturin)

```bash
python -m venv .venv
. .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip maturin pytest hypothesis
cd scpn-control-rs/crates/control-python
python -m maturin develop --release
cd ../../..
python -m pip install -e .
python -c "import importlib.util; from scpn_control.core._rust_compat import _rust_available; assert importlib.util.find_spec('scpn_control_rs') and _rust_available()"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin \
  tests/test_aer_observation_rust_parity.py tests/test_boris_pyo3_bridge.py \
  tests/test_capacitor_bank_state_pyo3.py tests/test_controller_advanced_paths.py \
  tests/test_fusion_neural_mpc_pulsed_adapter_rust_parity.py \
  tests/test_multi_shot_campaign.py tests/test_multi_shot_campaign_pyo3.py \
  tests/test_pyo3_control_bridge.py tests/test_rust_compat_wrapper.py \
  tests/test_rust_python_parity.py tests/test_rust_realtime_parity.py \
  tests/test_snn_pyo3_bridge.py -v
```

Native-dependent coverage uses two CI data files and a merged report:

1. `python-tests` writes `coverage-data-python` from the rust-absent Python
   coverage job.
2. `rust-python-interop` builds `scpn_control_rs`, runs every ledger-recorded
   Rust/PyO3 conditional test-file owner with `COVERAGE_FILE=.coverage.rust`,
   and writes `coverage-data-rust`.
3. `native-coverage-combine` downloads both artifacts, runs
   `coverage combine --keep artifacts/coverage/python artifacts/coverage/rust`,
   emits the merged `coverage-report-combined`, and gates it with
   `coverage report --fail-under=100`.

Run `python tools/native_coverage_matrix.py --json` to inspect the physical
workflow owners and public docs against the
`scpn-control.native-coverage-matrix.v1` declaration contract. Enabled producer
steps must collect and save the expected data files; pinned uploads must include
hidden data files. The consumer must download both artifacts before the direct
guard, combination, XML output and numeric 100-percent gate. Workflow inputs are
parsed with unique keys and the TOML threshold is read as a number. A passing
declaration check does not establish executed CI or authenticate coverage data.

`python tools/coverage_exception_ledger.py --check` independently seals every
current lexical pragma, configured exclusion pattern, and the three recognised
AST calls `pytest.skip`, `pytest.mark.skipif` and `pytest.mark.xfail`.
It does not inventory aliases, `pytest.mark.skip` or `pytest.importorskip`, or
validate xfail strictness; see the
[ownership contract](api.md#coverage-exception-ownership-ledger).
Its generated JSON records owner, reason, dependency, declared CI lane or explicit
blocker, last review, and removal condition. The ledger therefore reports
variant breadth separately from the configured coverage percentage instead of
silently treating skipped variants as covered.

## Local acceptance campaigns

### FAIR-MAST source-object binding readiness

The acquisition and lineage emitters use the shared
`validation.fair_mast_source_policy.fair_mast_provenance()` declaration. It
records the [catalogue's default CC-BY-SA-4.0 policy and requested paper
citations](https://mastapp.site/#license), without reading source objects or
verifying the licence of an individual asset. Preserve separately noted
asset-specific exceptions. Each returned block owns its citation list; edits
by one producer do not change another producer's block. See the
[source-policy API](api.md#fair-mast-source-policy) for the exact fields and
executable example. This declaration supplies no data or training admission.

Acquire each requested shot into an off-repository material root and use a
separate off-repository cache root:

```bash
python -m validation.acquire_mast_disruption_shots \
  --shots "30421 30424" \
  --out-dir /path/to/material \
  --cache-dir /path/to/cache \
  --manifest-out /path/to/source_object_manifest.json \
  --generated-at 2026-07-22T00:00:00Z \
  --retrieved-at 2026-07-22T00:00:00Z
```

The acquisition reads each shot's exact upstream Zarr-v3 root `zarr.json`
outside `simplecache`, requires inline consolidated metadata, and records its
SHA-256, byte count, ETag, and Last-Modified header. It creates a new empty
cache namespace bound to the shot, both acquisition labels, and the root
metadata digest. An existing namespace is never reused. After all selected
arrays are read, acquisition fetches the root metadata again and writes no NPZ
when the source-byte generation changed. The accepted generation record is
bound into the derived artifact's selected-array parent digest. Older valid v2
or migrated manifests may omit this pin because their historical generation
cannot be reconstructed safely; they do not gain a generation-pinned fidelity
claim.

The FAIR-MAST disruption extractor accepts acquisition output only through
`scpn-control.source-object-manifest.v2.0.0`. The reader validates the manifest
self-digest, local NPZ bytes, member set, dtype, shape, array-value digests,
parent snapshot including any source-generation pin, and transform descriptor
before exposing exact
`<group>.<array_name>` keys.

```bash
python validation/extract_mast_disruption_channels.py \
  --manifest /path/to/source_object_manifest.json \
  --artifact-root /path/to/material \
  --json-out /path/to/binding_readiness.json
```

The resulting `scpn-control.mast-disruption-binding-readiness.v1` report proves
only the acquisition-to-extractor transport contract. It remains
`status="blocked"` and `channel_extraction_admissible=false`: direct toroidal
field or sourced TF current, magnetic-axis Z, poloidal-probe reduction,
saddle-angle units and probe alignment, and cross-group timebase alignment are
not yet bound. The gate does not accept old `shots[].local_path` manifests or
invent aliases for top-level names. Eleven-channel extraction remains disabled
until a versioned MAST signal-binding specification supplies those physical
contracts.

For regenerated proxy-labelled datasets, run
`validation.verify_mast_lineage_bound_regeneration` after two fixed-time fresh
builds. The verifier reopens the SourceObjectManifest-v2 artifact set, exact
producer-bound replay bytes, both producer-lineage manifests, and every emitted
dataset file. It requires native source-generation pins and byte-identical
complete tree inventories, then emits a self-digested
`scpn-control.mast-lineage-bound-regeneration-verification.v1.0.0` report with
status `reproducible_blocked`. Reproducibility does not supply independent
outcome labels, an admitted cohort, or any training, scientific, facility,
prediction, reuse, or control claim.

Real-data manifest provenance gate:

```bash
scpn-control validate --json-out
scpn-control validate-release-evidence artifacts/release_evidence_report.json --json-out
scpn-control validate-manifest validation/reference_data/diiid/manifests/diiid_hmode_1p5MA.geqdsk.manifest.json --verify-artifact
scpn-control validate-manifest validation/reference_data/diiid/manifests/shot_163303_hmode.npz.manifest.json --verify-artifact --json-out
scpn-control validate-manifest validation/reference_data/diiid/manifests/mock_diiid_ci.manifest.json --json-out
```

Transport import failure or already loaded Matplotlib/Torch/Streamlit makes
the combined status FAIL and exits one, even when evidence gates were explicitly
skipped for a scoped check. Enabled reader findings also produce exit one;
passing checks exit zero. JSON is emitted before failure exit, while text
findings use stderr.

The top-level `validate` command includes repository data-manifest
validation, strict persisted JAX GK parity evidence admission, physics
traceability validation, multi-shot pulsed-MPC campaign evidence admission,
runtime-admission evidence validation, and native formal certificate admission
by default, so routine validation cannot pass while ignoring data provenance,
backend parity evidence drift, bounded-claim registry drift, campaign replay
evidence drift, runtime claim-boundary drift, or native certificate drift. The
local `tools/preflight.py` path now runs this top-level
release-evidence gate as a non-test gate, including in `make preflight-fast`,
and validates the generated JSON report with `validate-release-evidence`, so
release preflight cannot skip provenance, parity, claim-boundary drift,
multi-shot campaign evidence, runtime-admission evidence, or
artifact-admission drift when tests are intentionally omitted. Use
`--data-manifest-root` for staged facility drops,
`--jax-gk-parity-root` for staged parity campaigns,
`--physics-traceability-registry` for staged claim-boundary registries,
`--multi-shot-campaign-python-report` and `--multi-shot-campaign-rust-report`
for staged campaign benchmark reports,
`--runtime-admission-report` for staged PREEMPT_RT admission benchmark reports,
`--no-verify-artifacts` for metadata-only manifest checks, and
`--no-data-manifests`, `--no-jax-gk-parity`, or `--no-physics-traceability` only
for explicitly scoped import-hygiene checks. `--no-multi-shot-campaign-evidence`
is limited to scoped CLI/import checks and must not be used for release
evidence. `validate-release-evidence` checks the resulting JSON declarations by
rejecting duplicate keys, nonfinite floating tokens, skipped or failing mandatory
gates, incomplete CPU/GPU JAX GK parity case coverage, traceability reports that
do not block every open fidelity gap, multi-shot campaign evidence that lacks
Python, PyO3, Rust, digest-count, SHA-256, or benchmark-context admission, and
runtime-admission evidence that lacks benchmark context, payload sealing, or
fail-closed production-claim boundaries, and native formal certificate evidence
that attempts to promote production-class AOT timing without an explicit
production-claim boundary and an empty validator-error list.

This reader hashes the exact UTF-8 report bytes. It does not reopen referenced
reports, recompute their digests or metrics, authenticate a producer, run a
solver, or grant scientific, facility, or control admission. `admitted_gates`
lists required sections declaring `status="pass"`, even when another field in
that section makes overall admission fail. PASS depends on earlier producer
validation and report custody.

Manifest `total` is a positive integer; `real` and `synthetic` are nonnegative
integers. Coverage `expected` and `covered` must be nonnegative integers,
equal, and accompanied by `missing=[]`. Boolean or omitted counts fail. The
reader does not reconcile manifest totals or reopen covered artifacts. JAX
case/backend string lists must include the three named cases and CPU/GPU;
entry objects must have string `case`/`backend` fields covering all six pairs.
Additional string labels and duplicates are allowed; artifact counts are not
reconciled with entry counts. Traceability declares a positive total,
nonnegative gap/blocked counts, and enough blocked claims for every gap; its
registry is not reopened. Multi-shot summaries require exactly the
Python/PyO3/Rust surface set, `pyo3_status="ok"`, four lowercase SHA-256
spellings, a positive minimum digest count, a boolean production flag, and
empty errors. Digests are not recounted and this flag is not cross-checked
against benchmark context.

The lower persisted campaign reader validates both Python/PyO3 and Rust objects
even when decoded empty. It refuses malformed class types, duplicate keys and
nonfinite floating JSON tokens, checks each canonical payload self-digest, and
requires positive integer passed/sample counts and the configured minimum
declared decision-digest count. It does not reopen the decision chain. Python
and Rust context serialisation differs; affinity shape and load-field presence
are checked, while their contents and actual host qualification are unverified.
See the [persisted campaign contract](control/multi_shot_campaign.md#persisted-benchmark-report-admission).
Neither lower-reader PASS nor top-level summary admission grants deployment,
certified control, physical timing or producer authenticity.

Runtime/formal classes are `local_regression` or `production_benchmark`, with
boolean production flags consistent with the class. Local runtime summaries
may declare admission failure and nonnegative admission errors with a false
production flag; these declarations grant no realtime readiness. Production
runtime summaries must declare PASS and zero admission errors. Runtime
summaries require positive samples, lowercase report/payload digest spellings,
and empty reader errors. Native formal summaries require nonempty string case
labels containing `:aot_certificate:`, lowercase assumption/report digests, and
empty errors; certificate or case-label truth is not proved.

The standalone standard-library command accepts caller-relative paths and
follows symlinks, prints JSON/text to stdout, and exits zero for PASS, one for
findings, or two for argparse errors. It writes no report. Read/decode failures
yield no digest; decoded JSON objects retain their exact digest on field
failure. Extra fields are accepted beyond duplicate/nonfinite JSON rejection
at all depths. No filesystem containment or input size/depth budget is
provided. The registered Click command retains readable-file argument checks
and prints text errors to stderr.

```bash
python validation/validate_release_evidence.py artifacts/release_evidence_report.json --json-out
```

`--no-runtime-admission-evidence` is
limited to scoped CLI/import checks and must not be used for release evidence.

The lower persisted runtime reader also recomputes the canonical payload
self-digest before its result enters that summary. Empty objects fail required
fields; malformed types return findings. CPU IDs must be nonnegative integers,
load arrays must have three finite nonnegative values, and latency means must
lie inside min/max alongside monotonic percentiles. The actual host is not
re-probed and payload self-consistency does not authenticate a producer. A
resealed report remains a declaration; local FAIL with explanatory errors and
false production flag can pass this reader without qualifying realtime timing.
See the [persisted runtime report contract](control/runtime_admission.md#persisted-admission-probe-reports)
for fields, default historical input and CLI boundaries.

The gate separates experimental
validation evidence from CI fixtures. A manifest claiming real-shot validation
must include a non-synthetic source kind, machine, shot, signal paths, physical
units, retrieval timestamp, checksum, and licence or facility data policy. Local
real-data manifests can additionally verify the referenced artefact checksum
with `--verify-artifact`; local artefact URIs must be relative and resolve under
the manifest evidence tree or repository root, not arbitrary absolute paths.
Synthetic manifests remain allowed for CI, but require generator and seed
metadata and are reported as `kind: synthetic`. Manifest and acquisition-spec
JSON is parsed with duplicate-key rejection so provenance fields cannot be
overwritten silently by ambiguous objects.

Tracker #53 hardware/runtime evidence is aggregated by a dedicated bounded
gate:

```bash
python validation/validate_tracker53_evidence.py --output-json validation/reports/tracker53_evidence_gate.json
python validation/validate_tracker53_evidence.py --require-production-claim --json-out
```

The first command refreshes the local bounded manifest for checkpoint replay,
Kuramoto runtime parity, formal proof packaging, FPGA HDL export, package-level
runtime markers, and PREEMPT_RT runtime admission. The second command is
expected to fail until every tracker #53 surface supplies qualified hardware,
synthesis, PREEMPT_RT, replay, or external safety-review evidence. A passing
local tracker #53 manifest is therefore not a production hardware claim. The
current input contract assigns fixed unqualified classes to checkpoint,
Kuramoto, FPGA and package surfaces, so aggregate `production_claim_allowed`
always remains false and requested production mode always fails.

The registry selector accepts exactly integer issue 53 entries with unique,
nonempty module paths and requires all six defining surfaces. Unrelated entries
are ignored; other registry metadata and referenced evidence paths are copied
without reopening artifacts. Actual runtime/native-formal reader findings
propagate. A production-labelled child class is assigned only when its real
reader passes; formal metadata also requires a declared Z3 PASS. The Z3 status
reader hashes the exact decoded bytes and accepts paths outside the checkout.
It does not validate the Z3 schema, holds fields or self-digest, or replay SMT.
Missing, malformed or non-PASS Z3 status refuses aggregate admission.

Registry/Z3 duplicate keys and nonfinite floating JSON tokens are rejected at
all depths. An invalid runtime `require_production_claim` argument produces FAIL
and normalises to false. Explicit manifest output creates parent directories
and writes sorted UTF-8 JSON, including refused declarations. Output failures
return FAIL with an output finding; a partial/failed write is not valid custody.
The manifest digest hashes sorted compact JSON with only `manifest_sha256`
removed. It binds declarations, not authorship or hardware qualification.
The frozen result has mutable nested metadata; the public builder retains
nested entry/evidence-class aliases. Caller mutation can invalidate its digest.

The lower native-formal reader inspects required context and AOT summary counts,
certificate spellings and positive finite p99 bounds. Production context must
declare literal `workspace_dirty=false`. Its metadata PASS does not rerun a
proof or timing campaign. See the [native benchmark admission contract](benchmarks.md)
for field limits, case-level results on overall FAIL and provenance boundaries.
Both standalone readers work with the standard library; no native extension or
hardware loop is required to inspect historical reports.

CI validates the full manifest set and writes a JSON evidence report:

```bash
scpn-control validate-data-manifests --json-out
python validation/validate_data_manifests.py --output-json artifacts/data_manifest_report.json
```

Public neural-transport acquisition metadata is validated separately from
facility-shot manifests:

```bash
python validation/validate_public_data_acquisition.py --json-out
python validation/validate_public_data_acquisition.py --output-json artifacts/public_data_acquisition_report.json
```

The report covers the mirrored normalised Zenodo file metadata for QLKNN10D,
QLKNN11D, and QuaLiKiz JET spectra, plus the deferred byte count for multi-GB
numeric tensors. This is acquisition
readiness evidence only. It does not satisfy neural-transport reference
validation until the tensor payloads are downloaded on an admitted storage
target and converted into strict reference-artifact evidence.

The offline public acquisition inspector requires a positive numeric Zenodo
DOI, unique POSIX/Windows-safe relative file keys, and exact DOI-record plus
decoded file-key linkage in HTTPS download URLs. Queries, fragments and malformed
percent escapes are refused. Local mirror declarations require both path and
SHA-256; selected bytes must also match advertised MD5 and positive size in one
observed stream. Repository-relative mirrors take precedence over a full
relative-path adjacent fallback, both canonically inside the defining repository
root. An unrelated basename cannot substitute for a missing nested path.

Optional adjacent raw `record.json` is hash-bound when present and must stay
inside its manifest directory. It is not decoded or authenticated. Canonical
raw records are deliberately absent, so their record digests are unverified
declarations. Direct mapping validation ignores unknown metadata; JSON loading
separately refuses duplicate keys and nonfinite numbers at every depth.
Supported read, UTF-8, path, JSON/depth, scan-containment and output-write failures
return domain errors or structured directory/CLI FAIL. Accepted counters on FAIL
are partial observations; separate manifest DOI identities are not deduplicated.
No remote availability, licence validity, tensor format, numeric agreement or
scientific/control readiness follows from this report. There is no atomic
filesystem snapshot, download or training execution.

The report also enforces DIII-D artefact coverage: every discovered DIII-D GEQDSK
and disruption-shot NPZ under `validation/reference_data/diiid/` must be covered
by a manifest entry with a local SHA-256 checksum. It reports acquisition-spec
readiness as `realised` or `pending`; by default pending facility pulls are
visible but do not break normal metadata validation.

Strict facility-campaign gate:

```bash
scpn-control validate-data-manifests --require-real-acquisition --json-out
python validation/validate_data_manifests.py --require-real-acquisition --output-json artifacts/data_manifest_report.json
```

This strict mode fails when a valid acquisition specification has no matching
real `mdsplus` declaration with a local artifact list. Matching requires the
expected dataset ID, exact tree/machine, integer-normalised shot, source URI,
access policy and licence, and every requested signal's name, node/path, units
and timebase. Extra signals are allowed. A matching ID alone cannot realise a
spec. Duplicate manifest/spec dataset identities are findings. Invalid specs
fail the report and are excluded from its pending/realised linkage counts.

The default checks referenced local bytes against their SHA-256 digests.
`--no-verify-artifacts` disables these byte checks and records
`artifact_verification=false`; a realised entry in that mode is a metadata
observation. Coverage and hashing use the same ordered contained manifest,
evidence and nearest-repository roots. The process working directory cannot
shadow a verified artifact. Absolute, drive-qualified, parent-traversing and
escaping-symlink artifact references are refused by the defining resolver.
Discovered manifests, specs and required DIII-D artifacts must also remain
inside the selected scan root. Coverage inspects file existence, without
decoding arrays or asserting that a file is tracked in Git.

Manifest/spec JSON requires unique keys and finite floating values at every
depth. Manifest shot identity accepts strings or integers, excluding booleans,
nulls, floats and containers. Duplicate signal names and artifact URI spellings
are refused, and every provided root/artifact checksum must use lowercase
64-hex. Manifest loading translates supported JSON, UTF-8, IO, path and decoder
depth failures into `RealDataManifestError`. Directory inspection returns
structured findings; supported report-write failures return FAIL. Empty
manifest discovery ends before acquisition specs are decoded. Counts retained
on FAIL are diagnostic and do not establish admission.

The single-manifest CLI's `artifact_verified` field is true only when selected
local bytes were checked. Selecting `--verify-artifact` for a remote provenance
record without local artifacts reports false. Declarations, digest equality
and `realised` do not authenticate a facility, retrieval time, licence, measured
signal contents or control readiness. These checks do not replay artifact
formats and do not provide an atomic filesystem snapshot.

Optional MDSplus acquisition writes both the acquired NPZ and a validated
manifest at retrieval time. Use a checked acquisition specification for
repeatable facility pulls. Install `scpn-control[facility]` for the pure-Python
`mdsthin.MDSplus` compatibility client, or install the native MDSplus Python
client from a facility MDSplus distribution when local tree access is required:

```bash
pip install "scpn-control[facility]"
```

```bash
scpn-control acquire-mdsplus-shot \
  --spec-json validation/reference_data/diiid/acquisition_specs/shot_163303_mdsplus.json \
  --output-npz validation/reference_data/diiid/disruption_shots/shot_163303_mdsplus.npz \
  --manifest-json validation/reference_data/diiid/manifests/shot_163303_mdsplus.manifest.json \
  --json-out
```

Inline signal requests remain available for ad hoc facility sessions:

```bash
scpn-control acquire-mdsplus-shot \
  --tree DIII-D \
  --shot 163303 \
  --signal '{"name":"plasma_current","node":"\\\\IP","units":"A","timebase":"time_s"}' \
  --signal '{"name":"normalised_beta","node":"\\\\BETAN","units":"1","timebase":"time_s"}' \
  --output-npz validation/reference_data/diiid/disruption_shots/shot_163303_mdsplus.npz \
  --manifest-json validation/reference_data/diiid/manifests/shot_163303_mdsplus.manifest.json \
  --json-out
```

The command fails before writing validation claims if the optional client is not
installed, a signal is empty or non-finite, or the generated manifest cannot
verify the local artefact checksum.

Kuramoto phase-runtime evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family kuramoto-runtime \
  --artifact report=artifacts/kuramoto_runtime_evidence.json \
  -- python validation/benchmark_kuramoto_runtime_evidence.py \
    --output-json artifacts/kuramoto_runtime_evidence.json
```

The produced JSON uses `scpn-control.kuramoto-runtime-evidence.v1` and binds
the input phase/frequency arrays by SHA-256 instead of storing the arrays in
the evidence payload. Deployment-claim admission requires optional Rust parity
against the Python reference, deployment-target oscillator count coverage, and
timestep-refinement convergence under the declared tolerance. Python-only
reports remain bounded runtime evidence and do not satisfy deployment claims.

The producer draws fresh local-seed phase/frequency vectors, runs one wrapped
Euler step and two half steps, and writes sorted UTF-8 JSON. The default claim
is bounded even when native parity is available; enabling the deployment
flag requests the core's additional admission checks. This flag does not
authorise a device, facility controller or safety claim. The numerical
Python/Rust/PyO3 kernels are shared unchanged.

### Synthetic resilience and ROC diagnostics

`validation/control_resilience_campaign.py` generates synthetic signals,
additive noise and mantissa-bit faults through the actual public risk campaign.
Its four threshold checks use mean/p95 absolute dimensionless risk errors,
p95 recovery offset in samples and a recovery success fraction. The finite
recovery search is capped at the trace tail. A threshold pass is a local
synthetic diagnostic, not proven physical controller recovery.
`--strict` exits two on failed metrics after JSON, Markdown and stdout
are produced; a non-strict failed campaign exits zero. Output files are
sequential UTF-8 writes and have no campaign/alias/transaction guard.

`validation/disruption_roc_analysis.py` creates100 synthetic shots with
a50/50 class quota, using at most500 samples per simulated shot and51 thresholds.
Its former use of the simulator's trigger-relative offset as an absolute
disruption index misclassified early alarms. It now records the observed
threshold-crossing last sample as an absolute index. The actual40-shot
regression has20 disruptive cases; all alarm in time at a zero threshold.
The corrected full100-shot/51-threshold software-custody run reports AUC0.32
and prints FAIL while exiting zero. This observed score remains a synthetic
diagnostic; the threshold and historical scientific originals are unchanged.

The generator may reject an unbounded number of draws to fill quotas.
Evaluation uses128 past samples every20 steps, applies mechanism-specific
observable scaling, and compares risk strictly with the threshold. The
shared AUC helper adds endpoint pairs and sorts/integrates the curve; the
reported arrays retain the raw swept points. `PASS` means AUC >0.85
and `FAIL` still exits normally. Neither the mixed synthetic signals/
observable scales nor this score establishes held-out facility discrimination
or physical lead time. Fixed outputs are caller-relative sequential writes
with the platform codec; isolate cwd for software checks.

Grad-Shafranov solver evidence against the exact Solov'ev analytic equilibrium
can be regenerated with:

```bash
python -m validation.validate_grad_shafranov_solovev \
  --report validation/reports/grad_shafranov_solovev.json
```

The produced JSON uses `scpn-control.grad-shafranov-solovev-validation.v2` and
binds its own payload by SHA-256. It confirms that the production discrete
operator `FusionKernel._apply_gs_operator` (sharing the `_mg_residual` stencil),
the production Red-Black SOR smoother `_sor_step`, and the production Python
multigrid V-cycle `_multigrid_vcycle` converge at second order in the mesh
spacing to the exact field `ψ = c1 R⁴/8 + c2 Z²`, `Δ*ψ = c1 R² + 2 c2`. The
Rust `scpn_control_rs.py_multigrid_solve` binding is recorded for transparency
and reproduces the same Solov'ev field under the shared solver-stack sign
convention. This validates the equilibrium discretisation, SOR solver, and
multigrid solver against an analytic benchmark; facility-grade EFIT/GEQDSK
reconstruction claims still require matched external equilibria.

Structured-singular-value (mu) evidence against the exact closed-form mu
identities can be regenerated with:

```bash
python -m validation.validate_mu_structured_singular_value \
  --report validation/reports/mu_structured_singular_value.json
```

The produced JSON uses
`scpn-control.mu-structured-singular-value-validation.v1` and binds its own
payload by SHA-256. It confirms the production D-scaled upper bound
`compute_mu_upper_bound` reproduces the analytic structured singular value where
it is known in closed form: a single full block gives `mu = sigma_max(M)`, a
diagonal plant with diagonal uncertainty gives `mu = max|M_ii|`, a rank-one
plant gives `mu = sum|u_i v_i|`, and every diagonal case satisfies the spectral
sandwich `rho(M) <= mu(M) <= sigma_max(M)`. D-scaling invariance is recorded as
a diagnostic only: the bound minimises `sigma_max(D M D^{-1})` with a finite
finite-difference descent, so its invariance holds only up to the descent's
local-minimum spread. This validates the static mu upper bound against analytic
identities; frequency-dependent D-K synthesis and facility robust-control claims
still require an external validated backend.

Guiding-centre orbit-integrator evidence against exact conservation laws can be
regenerated with:

```bash
python -m validation.validate_guiding_centre_conservation \
  --report validation/reports/guiding_centre_conservation.json
```

The produced JSON uses
`scpn-control.guiding-centre-conservation-validation.v1` and binds its own
payload by SHA-256. It integrates the production `GuidingCenterOrbit` RK4 stepper
in a static analytic axisymmetric tokamak field and confirms the exact
guiding-centre invariants: kinetic energy `E = (1/2) m v_par^2 + mu B` and the
canonical toroidal momentum `p_phi = m R v_par (B_phi/B) + q psi(R, Z)` are both
conserved to better than `1e-4` relative over passing and trapped (banana)
orbits for deuterons and 3.5 MeV alphas, and the parallel speed never exceeds the
total speed. This validates the orbit integrator and drift physics against
analytic conservation laws; external orbit-code, banana-width, and measured
fast-ion loss claims still require matched reference artefacts.

Transport heat-diffusion evidence against exact analytic references can be
regenerated with:

```bash
python -m validation.validate_transport_diffusion \
  --report validation/reports/transport_diffusion.json
```

The produced JSON uses `scpn-control.transport-diffusion-validation.v1` and binds
its own payload by SHA-256. It confirms the production cylindrical diffusion
operator `TransportSolver._explicit_diffusion_rhs` reproduces the exact Bessel
eigenvalue `L[J0(lambda rho)] = -(chi lambda^2 / a^2) J0` at second order, and
the Crank-Nicolson tridiagonal solve (`_build_cn_tridiag` + `_thomas_solve`)
recovers the manufactured steady state `T* = 1 - rho^3` (source
`S = 9 chi rho / a^2`) at second order. The polyglot leg checks that the Rust
`scpn_control_rs.py_thomas_solve` — the compute primitive used by the Rust
`transport_step` — reproduces the Python `_thomas_solve` solution of the
identical Crank-Nicolson system to machine precision. This validates the
diffusion discretisation and linear-solver chain against analytic references;
facility-calibrated integrated-modelling claims still require a measured
discharge or published benchmark.

NTM island-dynamics evidence against exact Modified Rutherford Equation
references can be regenerated with:

```bash
python -m validation.validate_ntm_island_dynamics \
  --report validation/reports/ntm_island_dynamics.json
```

The produced JSON uses `scpn-control.ntm-island-dynamics-validation.v1` and binds
its own payload by SHA-256. It confirms that, in the classical-only limit
(bootstrap, polarisation, diamagnetic, and ECCD terms off), the production
`NTMIslandDynamics.dw_dt` and RK4 `evolve` reproduce the exact separable solution
`w(t) = -2 r_s + 2 sqrt((r_s + 0.5 w0)^2 + K t)` with
`K = Delta'_0 r_s^2 / tau_R` to integrator precision, and that the closed-form
classical+bootstrap saturated width
`w_sat = -a1 (j_bs/j_phi) r_s / (Delta'_0 r_s + 0.5 a1 (j_bs/j_phi))` zeroes
`dw_dt` and acts as a stable attractor for the RK4 evolution from both below and
above. This validates the island-evolution ODE right-hand side and integrator
against analytic references; facility-qualified NTM forecasting or suppression
claims still require a measured campaign or documented public reference.

RZIP rigid vertical stability evidence against exact eigenvalue references can be
regenerated with:

```bash
python -m validation.validate_rzip_vertical_stability \
  --report validation/reports/rzip_vertical_stability.json
```

For the separate declared RZIP calibration observation, the actual fixed
two-loop benchmark can run without persistent output:

```bash
python -m validation.benchmark_rzip_calibration --no-write
```

`--json-out` and `--markdown-out` support explicit temporary destinations;
default persistent evidence still requires the recorded-campaign wrapper.
Calibration builder/writer/admission APIs check growth-time and reference-error
consistency, literal flags/version, declared-source status and self-seal. Zero
growth uses positive infinite growth time and never admits the facility flag.
Caller-declared external-source labels and a matching comparison do not
authenticate the external result or physical deployment. Invalid observations
refuse before writer IO; output aliases refuse, while a later Markdown failure
can leave only JSON. See [the API contract](api.md#declared-rzip-calibration).

The vertical-stability JSON uses `scpn-control.rzip-vertical-stability-validation.v1` and
binds its own payload by SHA-256. The public decoder checks exact v1 fields,
finite numerical domains, the three scaling laws and agreement between metrics,
individual boolean checks and the aggregate boolean verdict. A coherent failing
report returns `False`; resealing a malformed or contradictory report does not
admit it. The seal establishes local consistency, without authenticating a
producer, source freshness or a measured facility result.

`--exact-tol` sets a finite positive relative-error threshold and
`--marginal-tol` sets a finite positive growth threshold in s^-1. Omitted options
retain the API defaults of `1e-9` and `1e-6`. CLI verdicts use exit 0 for pass,
1 for a coherent failure and argparse exit 2 for invalid tolerances. `--json-out`
emits the sealed report to stdout; execution without `--report` writes no files.
The report parent must exist. JSON and same-stem Markdown are overwritten in
that order; an IO error can leave a JSON-only pair. Report writing provides no
rollback or facility admission.

In the no-wall limit the rigid vertical mode
reduces to the exact 2x2 block with eigenvalues `+/- sqrt(-K/M_eff)`, so the
production `RZIPModel.vertical_growth_rate` is checked against
`gamma = sqrt(-n mu_0 Ip^2 / (4 pi R_0 M_eff))` for a destabilising index
`n < 0`, the oscillation frequency `sqrt(K/M_eff)` for a stabilising `n > 0`, the
growth-time identity, and the exact `Ip`, `sqrt(-n)`, and `1/sqrt(M_eff)` scaling
laws — all to about `1e-16` relative. A passive resistive wall is shown to reduce
the growth rate below the no-wall value, confirming the eddy-current circuit
coupling is stabilising. This validates the rigid-mode physics and eigenvalue
machinery. `RZIPController` computes continuous-time LQR gains when the local
SciPy Riccati path is usable and falls back to a bounded NumPy discrete-Riccati
iteration if SciPy validation fails; zero gain is reserved for fail-closed cases
where both designs fail. Facility-validated vertical-control claims still
require a matched RZIP/CREATE-L/TSC or measured vertical-displacement benchmark.

The normalised DGKF H-infinity report producer and checker use the bounded
flight-simulator factory with the original numerical residual, feasibility,
stability and sampled-gain thresholds. Run without replacing stored reports:

```bash
python -m validation.validate_h_infinity_control --no-write
python -m validation.validate_h_infinity_control --check-report validation/reports/h_infinity_control.json
```

Check mode reads unique-key UTF-8 JSON and never reseals or writes the supplied
file. Exit 0 admits a consistent passing local report; exit 1 reports refusal,
and argparse uses exit 2 for malformed arguments. Exact v1 fields, finite metric
domains, the 20002-sample declaration, peak/gamma ratio and literal verdicts are
checked. The three current owner hashes include the controller, validator and
decoder. The Git capture label is format-checked rather than forced to a later
HEAD with unchanged source bytes; neither the label nor hashes authenticate a run.

Public and production admission must remain `False`, and exclusions cannot be
removed by resealing. A coherent failing result carries local scientific
admission `False` and fails the passing-report gate. The frequency sweep remains
finite corroboration without an exact norm or facility-control claim. Old reports
with earlier source ownership remain historical and are not silently migrated.
The public writer creates parents and replaces distinct JSON/Markdown files;
output/source aliases refuse before writes, and later IO failure can leave a
partial pair. See the [API contracts](api.md#bounded-normalized-dgkf-reports).

Resistive-wall-mode feedback evidence against exact closed forms can be
regenerated with:

```bash
python -m validation.validate_rwm_feedback \
  --report validation/reports/rwm_feedback.json
```

The produced JSON uses `scpn-control.rwm-feedback-validation.v1` and binds its
own payload by SHA-256. It checks the production `RWMPhysics` and
`RWMFeedbackController` against their exact closed forms: the Bondeson-Ward
growth rate `gamma_wall = (1/tau_eff)(beta_N - beta_nw)/(beta_w - beta_N)`, the
wall-gap `tau_eff = tau_wall (b/d)^2`, the Fitzpatrick rotation term, the
critical-rotation marginality (total growth rate is exactly zero at
`Omega_crit`), and the feedback marginalisation (the closed-loop growth rate is
exactly zero at the required gain `(1 + gamma tau_ctrl)/M_coil`) — all to about
`1e-16` relative — plus the no-wall/ideal-kink window boundaries and the
`1/tau_wall` scaling. This validates the RWM stability-window and feedback
physics against analytic references; facility-validated MHD-stability or
hardware-control claims still require measured RWM shots or an external MHD
stability reference.

Kadomtsev sawtooth-crash evidence against exact conservation laws can be
regenerated with:

```bash
python -m validation.validate_sawtooth_kadomtsev \
  --report validation/reports/sawtooth_kadomtsev.json
```

The produced JSON uses `scpn-control.sawtooth-kadomtsev-validation.v1` and binds
its own payload by SHA-256. It confirms the production `kadomtsev_crash`
conserves the volume integrals `integral T rho drho` and `integral n rho drho`
over the mixing region to machine precision (energy and particle conservation),
that the helical-flux proxy vanishes at the mixing radius
(`psi*(rho_mix) = 0`), that the temperature and density flatten inside the mixing
radius while `q` is reset to one and outer profiles stay invariant, that the
grid-interpolated `q = 1` radius converges at second order to the analytic
`rho_1 = sqrt((1 - q0)/(qa - q0))`, and that a `q > 1` profile triggers no crash.
This validates the full-reconnection redistribution against analytic references;
full nonlinear MHD sawtooth-crash or measured-shot claims still require a measured
or published reference.

The two-point scrape-off-layer algebraic diagnostic runs without writing files:

```bash
python -m validation.validate_sol_two_point --json-out
```

The produced JSON retains `scpn-control.sol-two-point-validation.v1` and a
SHA-256 self-digest. Its reader requires the complete structure, finite numeric
domains, literal boolean flags, UTC timestamps at second resolution, ordered
scaling observations and internally consistent errors, maxima and strict gates.
Resealing malformed or contradictory content cannot turn it into a passing
report. A coherent failing report returns `False` normally. The self-digest
authenticates neither a producer nor the source or physical observations.

The diagnostic checks production `TwoPointSOL` and divertor helpers against
the same algebraic formulas and shared constants: the connection length
`L_par = pi q95 R0`, the parallel heat-flux mapping, the Spitzer-Härm upstream
conduction integral `q_par = kappa_0 T_u^{7/2}/((7/2) L_par)`, the pressure
balance `n_u T_u = 2 n_t T_t`, the Eich regression exponents
(`P^{-0.02}`, `R^{0.04}`, `B_pol^{-0.92}`, `eps^{0.42}`), the peak target heat
flux, and the sheath-limited detachment density boundary. Geometry is in metres,
field in tesla, power in MW, upstream density in `1e19 m^-3`, and errors/ratios
are dimensionless. The illustrative default is R0=1.7 m, a=0.5 m, q95=3.5 and
B_pol=0.4 T; it carries no ITER or facility provenance. Epsilon doubles below
0.5 and halves otherwise, keeping each scaling probe in `0 < epsilon < 1`.

`--exact-tol` requires a positive finite threshold and uses strict `<` gates;
`--exact-tol 1e-30` exercises a real numerical diagnostic failure. CLI exits are
zero for pass, one for diagnostic failure and two for supported input, custody,
arithmetic or I/O refusal. Standard argparse help/usage retain exits zero/two.
`--report` writes selected JSON and sibling `.md` files after checking both for
direct, symbolic and hard-link aliases of selected sources or each other.
Persistent repository evidence destinations require the recorded runner;
existing scientific reports are not silently replaced by ordinary invocations.
JSON then Markdown are sequential writes without an atomic pair or filesystem
snapshot; a later failure can leave the JSON file.

These checks establish local algebraic consistency. Independent numerical,
experimental, facility, safety, controller and training admission remain absent;
measured probe-campaign or independent published reference artefacts are still
needed. The Python contract and executable example are in [API documentation](api.md#sol-algebraic-diagnostic).

Auxiliary current-drive evidence against exact closed forms can be regenerated
with:

```bash
python -m validation.validate_current_drive \
  --report validation/reports/current_drive.json
```

The produced JSON uses `scpn-control.current-drive-validation.v1` and binds its
own payload by SHA-256. It checks the production ECCD, LHCD, and NBI sources and
efficiency helpers against their exact closed forms: grid-normalised deposition
power conservation (`integral P drho = P_source`) for all three sources, the
deposition centroid, the Stix critical energy `E_crit = 14.8 T_e (A_b/A_i)^{2/3}`
and the slowing-down-time scalings (`T_e^{3/2}`, `1/n_e`, `1/Z_eff`), the Prater
ECCD efficiency with the launch-angle factor maximised at `N_parallel = 1`, the
driven-current proportionality `j_cd = eta_cd P_abs/(n_e T_e)`, and the
neutral-beam fast-ion current chain `j_cd = e n_fast v_par/Z_beam` — all to
machine precision. This validates the deposition and efficiency physics against
analytic references; external current-drive claims still require ray-tracing,
Fokker-Planck, or measured-deposition artefacts.

Ideal-MHD stability-metric evidence against exact closed forms can be regenerated
with:

```bash
python -m validation.validate_mhd_stability \
  --report validation/reports/mhd_stability.json
```

The produced JSON uses `scpn-control.mhd-stability-validation.v1` and binds its
own payload by SHA-256. It checks the production stability metrics against their
exact closed forms: the Troyon limit `beta_N = 100 beta_t a B0 / Ip` with its
`beta_t`, `a`, `B0`, and `1/Ip` scaling and no-wall/ideal-wall boundaries, the
Mercier interchange index `D_M = s(s-1) + alpha(1-s/2)` with hand-evaluated
marginal cases, the Connor-Hastie-Taylor ballooning boundary
`alpha_crit = s(1-s/2)` for `s<1` and `0.6 s` for `s>=1`, and the
Kruskal-Shafranov `q_edge > 1` external-kink criterion — all to machine precision
with consistent stability flags. This validates the analytic stability metrics;
full ideal- or resistive-MHD eigenmode claims still require an independent MHD
stability code or benchmark profiles.

EPED pedestal-model evidence against its exact construction relations can be
regenerated with:

```bash
python -m validation.validate_eped_pedestal \
  --report validation/reports/eped_pedestal.json
```

The produced JSON uses `scpn-control.eped-pedestal-validation.v1` and binds its
own payload by SHA-256. It checks the production `eped1_predict` against the exact
construction relations — the `q95 = a B0/(R0 B_pol)(1+kappa^2)/2` formula, the
alpha-inversion pedestal pressure `p_ped = alpha_crit B0^2 a Delta/(2 mu0 q95^2
R0)`, the poloidal beta `beta_p = 2 mu0 p/B_pol^2`, the ideal-gas temperature
`T_ped = p/(2 n_e e)`, the collisionality width narrowing (with the `nu*=0`
identity), and the shaping-factor reference (unity at the ITER reference shape) —
all to machine precision, plus the KBM width constraint
`Delta_KBM = C_KBM sqrt(beta_p)` satisfied at the converged collisionless width
within the fixed-point iteration tolerance. The Rust `control-core/src/pedestal.rs`
is a separate simplified ELM-trigger proxy with a different width scaling and is
not a parity counterpart, so no cross-language parity is asserted. This validates
the EPED construction; externally validated EPED-database claims still require
measured pedestal data or published benchmark points.

ELM peeling-ballooning and crash evidence against exact closed forms can be
regenerated with:

```bash
python -m validation.validate_elm_peeling_ballooning \
  --report validation/reports/elm_peeling_ballooning.json
```

The produced JSON uses
`scpn-control.elm-peeling-ballooning-validation.v1` and binds its own payload by
SHA-256. It checks the production `PeelingBallooningBoundary` and `ELMCrashModel`
against their exact closed forms: the ballooning `alpha_crit` and peeling
`j_crit` limits with their `1/q95`, `1/sqrt(n_mode)`, and `R0/a` scalings, the
elliptical stability margin `1 - sqrt((j/j_crit)^2 + (alpha/alpha_crit)^2)` (zero
on the unit ellipse, sign-consistent with `is_unstable`, stable interior /
unstable exterior), and the Type-I ELM crash energy conservation `Delta_W = f
W_ped` with `W_post = (1 - f) W_ped` and the pedestal-region `n T` product
dropping by `(1 - f)` while the core stays unchanged — all to machine precision.
This validates the ELM stability and crash physics; facility ELM/RMP claims still
require measured H-mode campaign data or published ELM cases.

Toroidal momentum-transport evidence against exact closed forms can be
regenerated with:

```bash
python -m validation.validate_momentum_transport \
  --report validation/reports/momentum_transport.json
```

The produced JSON uses `scpn-control.momentum-transport-validation.v1` and binds
its own payload by SHA-256. It checks the production momentum-transport functions
against their exact closed forms: the NBI torque `P_NBI R0 sin(theta)/v_beam`
(and zero torque for a non-positive beam), the Hinton-Hazeltine radial electric
field `E_r = (1/(e n_i)) dp_i/dr + R0 omega_phi B_theta` (exact for constant and
linear pressure, where `np.gradient` is exact), the Burrell E×B shearing rate
`|R0 B_theta/B domega_phi/dr|` (exact for a linear rotation profile), the
Biglari-Diamond-Terry suppression factor `1/(1 + (omega_ExB/gamma)^2)`, the Rice
intrinsic velocity `3.5 W_p/I_p` with its scaling, and the toroidal Mach number —
all to machine precision. This validates the rotation and torque diagnostics;
facility momentum-transport claims still require measured NBI rotation cases.

Runaway-electron avalanche evidence against exact closed forms can be regenerated
with:

```bash
python -m validation.validate_runaway_electron \
  --report validation/reports/runaway_electron.json
```

The produced JSON uses `scpn-control.runaway-electron-validation.v1` and binds its
own payload by SHA-256. It checks the production `RunawayElectronModel` against
its exact closed forms: the Connor-Hastie critical field `E_c = n_e e^3 lnL /
(4 pi eps0^2 m_e c^2)` (with the total free-plus-bound electron density), the
Dreicer field `E_D`, the collision time, the avalanche time constant `tau_av`
(with `Z_eff` enhancement), the impurity-aware critical field, and the
Rosenbluth-Putvinski avalanche rate `gamma_av = n_RE (E/E_c - 1)/(tau_av lnL)`
(zero below `E_c`, linear in `n_RE` and `(E/E_c - 1)`, with the 0.001 RMP
deconfinement factor above 0.3 mol of neon) — all to machine precision. This
validates the runaway-generation physics; facility disruption-mitigation claims
still require measured disruption-campaign data.

Halo-current L/R circuit evidence against exact closed forms can be regenerated
with:

```bash
python -m validation.validate_halo_current \
  --report validation/reports/halo_current.json
```

The produced JSON uses `scpn-control.halo-current-validation.v1` and binds its
own payload by SHA-256. It checks the production `HaloCurrentModel` against its
exact closed forms: the halo resistance `R_h = eta 2 pi R0 / (d_wall a
f_contact)`, the halo inductance `L_h = mu0 R0 (ln(8 R0/a) - 1.5)`, the mutual
inductance `M = f_contact sqrt(L_p L_h)`, and the time constant `tau_h = L_h/R_h`,
together with the `R_h` scaling laws (linear in `eta` and `R0`, inverse in
`f_contact` and `d_wall`), the simulated electromagnetic wall force `F = mu0
I_h,peak I_p0 / (2 pi a)`, and the toroidal-peaking product — all to machine
precision — plus the fast-circuit quasi-static limit in which the halo current
tracks `M |dI_p/dt| / R_h` with an error that decreases monotonically as
`tau_h/tau_cq -> 0`. This validates the halo-circuit physics; facility
disruption-mitigation claims still require measured disruption-campaign data.

Disruption-sequence phase-ordering evidence can be regenerated with:

```bash
python -m validation.validate_disruption_sequence \
  --output-json validation/reports/disruption_sequence.json \
  --output-md validation/reports/disruption_sequence.md
```

The produced JSON uses `scpn-control.disruption-sequence-validation.v1` and
binds its own payload by SHA-256. It checks the bounded production sequence
against exact repository-owned identities: total duration equals thermal-quench
plus current-quench duration, wall heat load equals thermal-quench deposition
plus runaway termination load, the current trace starts at the configured plasma
current and decays monotonically, the vessel force is the vertical halo force,
the post-TQ temperature remains below the pre-TQ state, and the SPI density
branch changes the current-quench phase without pretending to validate measured
disruption windows. The payload keeps `production_claim_allowed=false`; facility
claims still require labelled disruption-window and mitigation-campaign
artifacts through the strict reference gate.

Volt-second flux-budget evidence against exact closed forms can be regenerated
with:

```bash
python -m validation.validate_volt_second \
  --report validation/reports/volt_second.json
```

The produced JSON uses `scpn-control.volt-second-validation.v1` and binds its
own payload by SHA-256. It checks the production `FluxBudget`,
`ScenarioFluxAnalysis`, `FluxConsumptionMonitor`, and `VoltSecondOptimizer`
against their exact closed forms: the inductive flux `L_p I_p`, the Ejima startup
flux `C_E mu0 R0 I_p` (with their linear scalings in `I_p`, `L_p`, and `R0`), the
resistive ramp integral `sum R_p I_p dt`, the flat-top budget closure in which
the flat-top resistive consumption `R_p (I_p - I_bs) tau_flat` exactly equals the
remaining flux at `tau_flat`, the ramp/flat-top/ramp-down scenario decomposition
and the budget margin, the `V_loop dt` consumption integrator, and the uniform
linear ramp optimiser — all to machine precision. The bootstrap-current proxy
remains a documented rough scaling outside this exact-closed-form scope. This
checks the declared bounded model's algebra; facility pulse-design or central-solenoid
commissioning claims still require independent measured references and their
source/metric qualification.

The v1 consumer now requires the complete schema, valid circuit configuration,
finite nonnegative errors, positive tolerances, known unique scaling laws and
literal boolean verdicts consistent with every metric. A well-formed failing
report returns `False`; missing, contradictory or malformed fields raise an
authored `ValueError` even when their content hash is valid. The SHA-256 field
checks content consistency; it does not establish a producer, source revision,
freshness, experimental provenance or facility/control admission.

For zero analytical phase flux, relative errors use the available positive flux
budget as their reference scale. Ordinary nonzero references keep their prior
normalisation. Zero loop voltage similarly uses the flux budget for consumption
error and unit scale for the dimensionless fraction. The zero phase and
zero-consumption cases are computed, rather than omitted. Iteration counts and
ramp segment counts must define a nonempty integration and a two-point ramp.

The existing API tolerances are also available as `--exact-tol` (default `1e-9`)
and `--margin-abs-tol` (default `1e-6` volt-seconds). Both must be finite and
positive. Choosing a stricter tolerance can produce a genuine failed report;
these declarations do not replace any external acceptance threshold. CLI status
is 0 for agreement, 1 for a failed declared gate and 2 for invalid arguments or
report publication refusal.

JSON and Markdown reports are validated before guarded publication. Source,
configuration and reference-data paths are protected; the documented
`validation/reports/` output directory remains available. Destinations must be
distinct regular files. Each replacement is atomic; handled failures recover
unchanged predecessors, without a crash-transaction or concurrent-writer claim.
`validation.volt_second_models`, `volt_second_evidence` and `volt_second_report`
own the records, report checks and publication; the original
`validation.validate_volt_second` public exports remain available.

Density-control particle-balance evidence against exact closed forms can be
regenerated with:

```bash
python -m validation.validate_density_control \
  --report validation/reports/density_control.json
```

The produced JSON uses `scpn-control.density-control-validation.v3` and binds its
own payload by SHA-256. It checks the production `ParticleTransportModel` and
`DensityController` against their exact closed forms: the Greenwald limit
`n_GW = I_p/(pi a^2)` (with linear `I_p` and inverse-square `a` scaling), the
volume-averaged Greenwald fraction `<n>/n_GW`, the circular flux-surface volume
elements `V' = 4 pi^2 R0 a^2 rho` and `V = 2 pi^2 R0 (a rho)^2`, the gas-puff,
neutral-beam, and recycling source normalisation (the source profiles integrate
to their requested particle rate, with the neutral-beam rate `P/E_beam/e`), the
cryopump edge sink, and the finite-volume diffusion operator vanishing on a
spatially uniform interior — all to machine precision. The pellet
neutral-gas-shielding ablation profile remains a separate bounded model outside
this exact-closed-form scope. This validates the particle-balance physics;
facility-calibrated fuelling or exhaust claims still require independent measured
particle-balance references and their source/metric qualification.

Density v3 requires the complete configuration, finite nonnegative errors,
positive tolerances, both known Greenwald scaling laws and literal stage/aggregate
verdicts consistent with every metric. A valid content hash alone cannot admit
missing or contradictory data. Earlier v2 declarations are refused: v3 records
the additional checker and guarded publisher source digests. These labels and
hashes are declarations, not producer authentication, verified current source,
freshness, independent physical provenance or facility/control admission.

The existing tolerances are available through `--exact-tol` (default `1e-9`)
and `--invariance-tol` (default `1e-12`), both finite and positive. CLI status0
means agreement with the declared bounded model, status1 a failed declared
tolerance and status2 invalid arguments or report publication refusal. The
original numerical functions and particle-balance model remain unchanged.

JSON/Markdown output pairs are checked before guarded per-file atomic
publication. Source, configuration and reference-data namespaces are protected;
`validation/reports/` remains the documented output directory. Handled failures
recover unchanged predecessors, without a crash-transaction or concurrent
hostile-writer guarantee. Boolean recycling coefficients are refused as invalid
numbers rather than treated as one.

The density claim evidence helper records numerical comparison separately from
facility admission in schema version 2. A caller can supply its own reference
values and provenance labels, so a match remains bounded evidence. Facility
admission requires an independently verified reference witness and currently
fails closed.

DT burn-control alpha-heating evidence against exact closed forms can be
regenerated with:

```bash
python -m validation.validate_burn_control \
  --report validation/reports/burn_control.json
```

The produced JSON uses `scpn-control.burn-control-validation.v1` and binds its
own payload by SHA-256. It checks the production `AlphaHeating`,
`BurnStabilityAnalysis`, `lawson_triple_product`, and `burn_fraction` against
their exact closed forms: the alpha-energy partition `E_fus/E_alpha = 5`, the
alpha power density `(n_e/2)^2 <sigma v> E_alpha`, the alpha-power volume integral
`p_alpha 2 pi^2 R0 a^2 kappa` for a constant power density, the energy gain
`Q = 5 P_alpha/P_aux` with its `P_aux = 0` ignition limits, the Lawson triple
product `n tau_E T` and the `3e21` ignition margin, the burn fraction
`a^2 n_DT <sigma v> / (4 v_th)`, and the reactivity exponent
`d ln<sigma v>/d ln T` reproduced by the centred finite difference — all to
machine precision. The Bosch-Hale DT reactivity is validated separately
(`scpn_control.core.uncertainty.bosch_hale_reactivity`) and held as the shared
input. This validates the burn-control algebra; reactor burn-control claims still
require integrated-transport or measured burn references.

Geometry-neutral stellarator replay reports now have a separate
schema-versioned evidence envelope,
`scpn-control.geometry-neutral-replay-evidence.v1`. The envelope binds the
validated replay report, scenario, trace, metrics, thresholds, magnetic
configuration provenance, actuator calibration, latency model, and fault model
by SHA-256 digest. Synthetic W7-X-like replay remains bounded evidence; device
control claims require a measured or benchmark stellarator artefact digest and
non-synthetic magnetic-configuration provenance.

Physics traceability validates that high-risk physics surfaces are bounded to
their current evidence status before full-fidelity or facility-validation claims
are made:

```bash
scpn-control validate-physics-traceability --json-out
python validation/validate_physics_traceability.py --output-json artifacts/physics_traceability_report.json
python validation/generate_physics_traceability_report.py --output-md docs/physics_traceability.md
python tools/check_generated_traceability.py
python tools/evidence_gap_matrix.py --output-json artifacts/evidence_gap_matrix.json --output-md artifacts/evidence_gap_matrix.md
python tools/validation_report_freshness.py --output-json artifacts/validation_report_freshness.json --output-md artifacts/validation_report_freshness.md
```

This is a declaration/path/marker consistency gate, rather than an
independent scientific admission. Registry files directly under a resolved
`validation/` directory use its parent as root; other registry locations use
cwd. All module/evidence/covered paths must resolve within that root, including
absolute paths and symlinks. Existence does not verify evidence bytes,
provenance, freshness or physical truth. Source marker scanning uses literal
words in local Python text; unreadable/invalid UTF-8/escaping paths cause
findings. Optional enforcement and claim policy require literal booleans.

Invalid status/header/container/duplicate/nonfinite declarations return FAIL,
with invalid entry summaries/raw flags and observed counts retained. Tracker
metadata is returned only when its positive nonboolean issue number, unique
identity, strings and matching URL are valid. These local URLs are not queried
for current state. Report flags on FAIL are diagnostic declarations.

The Markdown API can render diagnostics but displays every full-fidelity
claim as blocked on FAIL. Its optional `require_valid_registry=True` mode,
the generation CLI and the freshness checker require valid source; an exact
copy of a refused report cannot produce a green freshness result. Invalid
source is refused before the generator writes, and supported output IO/path
failures return refusal. A passing registry with exact canonical Markdown
establishes consistency only. The existing canonical valid report bytes and
scientific evidence status are preserved by these reader/consumer corrections.

The evidence gap matrix uses schema `scpn-control.evidence-gap-matrix.v1` and
groups blocked or bounded physics-traceability entries by their external
validation tracker. It is the planning input for promotion campaigns: first
clear the tracker work package with real external-code, facility, benchmark, or
hardware evidence, then update `validation/physics_traceability.json` and
regenerate `docs/physics_traceability.md`.

`build_evidence_gap_matrix(registry)` validates the fields consumed by planning:
known fidelity statuses, positive nonboolean issue numbers, unique tracker IDs,
boolean claim declarations, and nonblank strings. JSON with repeated member
names or nonfinite numbers is refused. Finite unused metadata remains supported.
Missing or unresolved positive tracker links remain visible through
`untracked_open_entries`; planning does not execute the full source/path/marker
validator or independently authenticate evidence or physical fidelity.
The public record imports and stored pickle addresses remain in
`tools.evidence_gap_matrix`.

The gap CLI protects its consumed registry, the command and selected repository's
lifecycle registry and claim ledger, and their report/refresh namespaces.
Symlink or nonregular outputs, aliases to protected files and overlapping output
paths are refused before publication. Distinct JSON/Markdown files share the
same staged publication and handled-failure recovery described below. JSON
stdout takes precedence when both stdout modes are requested. Metadata and
output refusals use authored field-level messages; other caught read/decode or
publication failures return a fixed sentence without interpreter details.

The validation report freshness inventory uses
`scpn-control.validation-report-freshness.v2` and consumes the versioned
`scpn-control.validation-report-lifecycle.v1` registry in
`validation/report_lifecycle_registry.json`. The registry binds every audited
JSON report to its SHA-256 digest, Git-storage class, evidence timestamp,
lifecycle bucket, evidence class, refresh state, provenance fields, and an
explicit fail-closed claim boundary. The immutable source payload can therefore
remain historical even when it predates embedded claim metadata; the registry
records that source limitation separately instead of rewriting the artifact.

Inventory generation reads report, refresh and registry bytes without changing
them. A relative reports root is first anchored to the working directory. Its
lexical containing directory determines the refresh repository root for both
reading and output protection. Linking the reports directory to another corpus
does not relocate the refresh tree to that link's target repository.
Both file-output options reject those input files, their hard links and
the report/refresh namespaces, as well as symlinks, nonregular destinations
and two options resolving to the same file. Freshness output also protects
`validation/public_claim_ledger.json` in both the command's repository and the
selected reports root's containing repository, including an absent ledger's
reserved path and hard links to an existing ledger. Existing regular inventory outputs
outside those namespaces may be replaced. Both requested formats are staged
before replacement. Directory identities also protect aliased input namespaces.
The writer checks again after creating output parents and before replacing a
later destination. This detects aliases that become visible only when the first
new output exists, including names on case-insensitive volumes. A handled
publication failure restores predecessor bytes
or removes newly published files. If recovery cannot finish or another writer
changes a published output, staging and predecessor files are retained for
inspection. Replacement is atomic per file; this does not provide a transaction
across a power loss or hostile concurrent namespace changes. Concurrent callers
must coordinate publication themselves.
The shared byte publisher can also create selected immutable output names
exclusively from complete staging files. Existing/racing names remain intact;
unsupported filesystem operations fail without an overwrite fallback.

`--max-age-days` selects a nonnegative integer advisory window, including broad
historical windows. It does not rewrite or waive the registry's recorded
21-day policy. Invalid timestamps, including an explicitly blank `--as-of`,
are refused before output publication. Authored lifecycle/output refusals may
describe the rejected declaration; other caught input and output exceptions
return fixed messages without native exception text. With `--fail-on-stale`,
an otherwise valid inventory is published before the stale exit status is
returned.

The reader verifies available report/refresh digests and consistency of
declared provenance and claim boundaries. It does not resolve declared Git
objects, attest host identity, execute preserved commands, recompute a producer
payload seal or independently establish scientific admission. Missing
owner-local artifacts remain indexed by their frozen declarations.

Classification is declarative, not inferred from filenames or report prose.
The three lifecycle buckets are `rerunnable_local`,
`external_artifact_blocked`, and `historical_only`. An owner-local ignored
artifact can be represented by its frozen digest without being presented as a
tracked file on a clean clone. Missing tracked reports, extra reports, digest
drift, unknown evidence classes, future timestamps, ambiguous refreshed-host
metadata, stale current evidence, and unsupported admission promotion all fail
closed.

Rebenchmarking is append-only. Existing files below `validation/reports/`
remain byte-for-byte source evidence; a later run is written below
`validation/report_refreshes/<UTC-date>/` using schema
`scpn-control.validation-report-refresh.v1`. Each refresh names and hashes its
source report, records the exact source commit, producer command and digests,
dependency lock, host and load context, sample design, raw-capture custody,
claim boundary, failures, and a canonical payload seal. The lifecycle registry
then binds the refresh artifact by path, SHA-256, and evidence timestamp. This
preserves side-by-side historical comparison and prevents a new workstation
run from silently replacing or promoting an older result.

Rerunnable local reports also include a refresh plan with a status and command
list: `ready_exact_command` when the registry preserves the complete command,
and `manual_reconstruction_required` when only partial or no command metadata
exists. Public consumers must select only the
matrix's fresh `current_admitted_reports`; historical and external-blocked
entries are never eligible. Use `--fail-on-stale` only in release or promotion
campaigns after the affected reports have been refreshed or deliberately kept
as historical evidence.

The canonical `scpn-control.public-claim-ledger.v1` ledger makes that selection
machine-readable. It contains only fresh reports whose lifecycle boundary
explicitly enables current evidence, scientific admission, and public claims;
every entry is bound to the report digest and commit, registry digest, source
commit, dependency lock, and any refresh artifact. An empty ledger means no
scientific claim is currently admitted for public use and must not be promoted
through documentation or release prose.

```bash
python tools/public_claim_ledger.py
python tools/public_claim_ledger.py --check
```

The ledger CLI validates one matrix and publishes its complete UTF-8 payload
through the same guarded writer as the freshness inventory. Its output cannot
replace consumed registry/report/refresh files, their aliases or the reserved
report/refresh namespaces. Existing regular ledger outputs elsewhere may be
replaced with the shared staging, backup and handled-failure recovery described
above. `--check` only reads and compares; it does not create or replace output.
Authored lifecycle and output refusals return status one with their deliberate
messages. Other caught input/output errors return status one with
`Public claim ledger inputs or output could not be inspected`.

The ledger records validated admission declarations; it does not establish
independent experimental truth. It rereads the registry for its byte digest
after validating the matrix. Callers must coordinate input changes because
these reads are not an atomic snapshot of concurrent registry updates.

Z3-backed SCPN formal evidence is published as schema-versioned JSON and
Markdown. The JSON uses `scpn-control.z3-formal-report.v2`, binds the proof
payload with `payload_sha256`, and records pass, fail, or blocked status. A
missing optional `z3-solver` dependency produces a blocked report in normal
publication mode; strict mode fails so release campaigns cannot mistake missing
SMT evidence for a successful proof:

The same Petri-net transition relation now exposes bounded CTL/LTL formula
facades for certification workflows. `CTLFormula` covers bounded `AG`, `EF`,
and `AG EF` obligations; `LTLFormula` covers bounded `G`, `F`, and
`G(trigger -> F<=n target)` obligations. `generate_safety_certificate` resolves
one verifier backend, runs base safety/liveness plus optional CTL/LTL
obligations, binds optional controller artifact bytes by SHA-256, and persists
schema-versioned `scpn-control.safety-certificate.v1` JSON and Markdown
artifacts with a canonical digest. `build_safety_certificate_payload` and
`write_safety_certificate` remain available for callers that already hold
validated report objects. Certificate admission also revalidates section
status, depth, backend, and checked-specification consistency, so an internally
inconsistent certificate remains rejected even if its digest is recomputed. The
optional `SafetyCertificatePolicy` gate can additionally require minimum proof
depth, controller artifact binding, CTL evidence, LTL evidence, and named
checked specifications before certificate artifacts are emitted or admitted. The
`write_safety_certificate_bundle` path persists a schema-versioned
`scpn-control.safety-certificate-bundle.v1` bundle for release gates that need
multiple independent certificates tied to the same controller artifact, backend,
and certificate policy. Bundle admission revalidates every embedded certificate
before checking bundle-level policy and digest integrity. Bundle artifact
admission uses `build_safety_certificate_bundle_artifact`,
`validate_safety_certificate_bundle_artifact`, and
`admit_safety_certificate_bundle_artifact` to require safe relative bundle URIs
and SHA-256 byte matches under a caller-supplied artifact root, plus a canonical
artifact metadata digest and non-future UTC creation timestamp, before replay
validation. The
certificate is evidence for bounded model checking only; it is not a facility
safety approval or an unbounded proof.
Z3 formal report files reached through safety-critical artifact manifests are
loaded through the duplicate-key-safe and schema-strict public
`load_z3_formal_report()` path before manifest/report field matching runs.
Unknown Z3 top-level or proof-section fields are rejected even when a foreign
producer recomputes the report payload digest.
Serialized Z3 counterexample records must carry only the admitted
`property_name`, `message`, `marking`, `path`, `place`, and `transition` fields,
with finite numeric marking values.
Z3 proof sections also enforce solver-status consistency: `unsat` sections must
hold and carry no counterexamples, `sat` sections must not hold and must carry
counterexamples, and `unknown` sections must not be admitted as holding.
Section-level Z3 `checked_specs` must contain unique non-empty strings so a
report cannot hide duplicate proof obligations behind the top-level
de-duplicated checked-spec list.
Blocked Z3 reports are limited to solver-availability evidence only: the
admitted blocked shape has solver `z3-solver unavailable`, `max_depth` equal to
zero, and exactly `z3_solver_available` as its checked specification.

```bash
python validation/validate_scpn_z3_formal.py
python validation/validate_scpn_z3_formal.py --require-z3
```

This publisher checks its fixed two-place source/sink net at depth two using
the installed Z3 solver. A permitted blocked report returns process status zero
without proving obligations; --require-z3 returns nonzero after blocked output.
Explicit output paths use caller cwd and sequential JSON/Markdown writes can
leave partial output. These reports establish bounded compiled-net evidence,
not hardware timing, PCS certification or unbounded liveness.

Outputs:

- `validation/reports/scpn_z3_formal.json`
- `validation/reports/scpn_z3_formal.md`

Nonlinear Cyclone Base Case saturation claims are gated separately from quick
smoke runs. The validator requires a long enough campaign, finite gyro-Bohm
ion heat flux, agreement with the documented CBC reference band, and a flat
tail heat-flux trace before a run can support saturated-transport claims:

```bash
python validation/gk_nonlinear_cyclone.py
```

The generated `validation/reports/gk_nonlinear_cyclone.json` and Markdown
summary use the `scpn-control.gk-nonlinear-cyclone.v2` schema and bind the
report payload with SHA-256. Short finite traces remain useful diagnostics, but
they are reported as insufficient saturation evidence rather than quantitative
nonlinear CBC validation. The current local run passed the linear, energy, and
zonal-flow diagnostics, but kept the saturated `chi_i` claim blocked because
the V4 campaign used `200` steps, returned `chi_i_gB=1.6568813509166032e-09`,
fell outside the `1.0..5.0` CBC reference band, and had tail relative drift
`0.30041712853638713` above the `0.10` saturation threshold.

Linear GK cross-code agreement claims require immutable real external-code run
evidence. Parser fixtures and published reference numbers are useful readiness
checks, but they do not prove quantitative agreement against actual binaries:

```bash
scpn-control validate-gk-crosscode --require-external-runs --json-out
python validation/validate_gk_crosscode.py --require-external-runs --output-json artifacts/gk_crosscode_report.json
```

The reader inspects self-declared comparison metadata. Required mode refuses
when no accepted declarations are present. A directory selects immediate sorted
`*.json` entries; a regular file selects itself regardless of suffix. Optional
absence passes with zero declarations. `external_runs` counts accepted files,
including repeated run IDs. A pass authenticates no binary execution, input deck,
external output, source bytes, version or timestamp; it grants no independent
scientific, facility or control admission.

The unchanged `scpn-control.gk-crosscode.v1` schema requires nonblank identities,
source `real_binary`, units `c_s/a` and four exact 64-character ASCII hex digests.
The body digest checks compact sorted ASCII JSON excluding `payload_sha256`.
It proves author consistency; the three source digests are declarations.
`binary_path` is checked lexically against admitted absolute executable roots;
existence and execution are not verified. URI, relative, traversal, temporary
and system-control paths are rejected.

Six scalars must be finite representable nonboolean numbers. Growth rates must
be nonnegative, dominant wavenumbers positive, and frequencies may be signed.
The original relative error is `abs(native - external) / max(abs(external), 1e-12)`:
growth permits at most `0.20`, frequency at most `0.30`. Absolute wavenumber error
permits at most `0.10`; equality passes. No metric or threshold is fitted.

Each declaration reads one captured byte sequence. IO, invalid UTF-8/JSON,
duplicate keys, nonfinite numbers and nonzero decimal underflow produce fixed
findings without raw decoder exceptions or duplicate member names. The public
`write_gk_crosscode_report` and both commands refuse direct, resolved, symlink
and existing hardlink aliases of the root or selected inputs before writing.
Other destinations may be replaced; sorted UTF-8 JSON ends with a newline.
Reads and alias checks are sequential and provide no concurrent snapshot or
pathname lock. Operational command failures return one with fixed text; Click
NUL outputs fail as usage with code two.

An optional empty-root inspection can be reproduced without external execution:

```python
from tempfile import TemporaryDirectory
from validation.validate_gk_crosscode import validate_gk_crosscode_evidence

with TemporaryDirectory() as directory:
    report = validate_gk_crosscode_evidence(directory)
assert report["status"] == "pass" and report["external_runs"] == 0
```

The stored Miller reader reads one reference file once and hashes its exact
bytes, including CRLF. Invalid UTF-8/JSON, duplicate keys, IO and nonzero decimal
tokens collapsed to binary64 zero yield authored findings. Input schema `1.0`,
seven nonblank header declarations, circular/shaped/high-shear required cases
and their original numeric domains remain. Eight required parameters refuse
boolean/nonnumber values; recognised optional parameters retain original float
coercion, including numeric strings and booleans, while unknown extras are ignored.
Actual Miller physical-domain failures become fixed parameter findings.

The comparison retains `n_theta=4, n_period=1`, nearest-grid theta selection,
nine sample fields, `atol=1e-11, rtol=1e-10` and unchanged Miller equations.
Sample and computed comparison values must be finite. Report schema v2 retains
its case digests and canonical payload hashing. Both readers label the two-dimensional
Jacobian in `m`, the mixed metric `g_rt` in `m-1` and `g_tt` in `m-2`:
radial derivatives are dimensionless and angle derivatives have units of metres.
The independent finite-difference validator uses its own 128-angle grid and five
cases. Corrected unit metadata changes fresh payload digests without changing
numerical comparisons. Historical science reports retain their original labels
and hashes. A bounded-local pass keeps full-equilibrium admission false and
authenticates no independent producer.

`write_gk_geometry_reference_report` creates parents and writes sorted UTF-8,
refusing direct/resolved/symbolic/existing-hardlink aliases of the single input
before mutation. Unrelated output may replace. API errors propagate; script and
registered CLI return 0 for pass, 1 for findings/fixed operational refusal, and
parser help/usage 0/2. Path observations are sequential, without a coherent
snapshot or lock against concurrent changes.

```python
from tempfile import TemporaryDirectory
from pathlib import Path
from validation.validate_gk_geometry_reference import (
    validate_gk_geometry_reference,
    write_gk_geometry_reference_report,
)
with TemporaryDirectory() as directory:
    reference = Path(directory) / "absent.json"
    report = validate_gk_geometry_reference(reference)
    assert report["status"] == "fail" and report["cases"] == 0
    write_gk_geometry_reference_report(report, Path(directory) / "report.json", reference_path=reference)
```

Miller geometry validation compares repository flux-tube geometry output against
immutable circular, shaped, and high-shear reference cases:

```bash
scpn-control validate-gk-geometry-reference --json-out
python validation/validate_gk_geometry_reference.py --output-json artifacts/gk_geometry_reference_report.json
```

The strict report uses the `scpn-control.gk-geometry-reference.v2` schema,
records the immutable reference-file SHA-256, per-case digests, SI units,
absolute and relative tolerances, and the canonical payload SHA-256. Current
local evidence in `validation/reports/gk_geometry_reference.json` admits the
bounded local Miller-geometry reference with three cases and sub-`1e-15`
maximum absolute drift. It does not admit a full equilibrium-reconstruction
claim; independent Miller-geometry implementation evidence or external
equilibrium-code evidence remains required.

The species reader reads one reference once and hashes the inspected raw bytes.
Unreadable/invalid UTF-8/JSON, duplicate keys and nonzero decimals collapsed to
binary64 zero return authored findings. Original schema `1.0`, four required
species and nonblank header declarations remain. Required numerical inputs and
expected values must be finite nonboolean numbers; failed/overflow conversion
and actual physical-domain errors produce fixed findings. Original finite
fractional quadrature/sparsity declarations retain integer truncation;
`is_adiabatic` retains bool coercion. Quadrature counts have no declared upper
resource bound; inspecting arbitrary very large grids can require substantial
resources. This reader does not establish resource admission for such grids.

The historical `larmor_radius_per_tesla_m` key retains its name and numbers;
its coefficient is `rho * B` with units `m*T`. Divide by field `B` in tesla to
obtain the radius in metres. Collision coefficients are in `s^-1`, without
`v_th/R` normalisation. Original physics equations and tolerances are unchanged.
Fresh corrected metadata changes report digests; historical reports retain
their original bytes and unit labels. `verify_payload_digest` checks body
consistency, without authenticating a producer or independent execution.

`write_gk_species_reference_report` creates parents and writes sorted UTF-8,
refusing direct/resolved/symbolic/existing hardlink input aliases before mutation.
Unrelated output may replace. API operational errors propagate; script and
registered CLI use fixed stderr and exit 1, with pass/findings 0/1 and parser
help/usage 0/2. Path observations are sequential rather than a locked snapshot.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from validation.validate_gk_species_reference import (
    validate_gk_species_reference, verify_payload_digest, write_gk_species_reference_report,
)
with TemporaryDirectory() as directory:
    reference = Path(directory) / "absent.json"
    report = validate_gk_species_reference(reference)
    assert report["status"] == "fail" and verify_payload_digest(report)
    write_gk_species_reference_report(report, Path(directory) / "result.json", reference_path=reference)
```

Gyrokinetic species validation compares mass, charge, thermal speed,
Larmor-radius normalisation, gyroaverage Bessel values, diamagnetic-drive
signs, velocity-grid quadrature normalisation, pitch-angle operator sparsity
and nullspace behaviour, and collision-frequency coefficients against
immutable electron, main-ion, impurity, and extreme-temperature reference cases:

```bash
scpn-control validate-gk-species-reference --json-out
python validation/validate_gk_species_reference.py --output-json artifacts/gk_species_reference_report.json
```

The strict report uses the `scpn-control.gk-species-reference.v3` schema,
records the immutable reference-file SHA-256, per-case digests, SI units,
absolute and relative tolerances, and the canonical payload SHA-256. Current
local evidence in `validation/reports/gk_species_reference.json` admits the
bounded species, gyroaverage, diamagnetic-drive, velocity-grid,
pitch-angle-operator, and test-particle collision-coefficient reference with
four species cases and zero relative drift. It does not admit a full
collision-operator claim; field-particle momentum-conservation evidence and an
external Fokker-Planck or equivalent reference remain required.

The source-tree `validation.gk_collision_independent_reference` API is a
separate local structural cross-check. Its vector functions reject nonfinite
speed ratios before calculation and preserve caller arrays, including
read-only views. Scalar rate inputs use positive finite proton-mass multiples,
keV and density in 10^19 m^-3; charge may be signed or zero. The fixed
Gauss–Legendre average has no adaptive error estimate or resource bound.
See the [collision reference API contract](api.md#independent-collision-coefficient-reference)
for units, errors, container limits and an executable example. This local
comparison does not supply a downloaded external numerical reference,
conservation proof or quantitative collisional-damping admission.

JAX GK parity claims require persisted native-vs-JAX parity artifacts with
backend metadata, dtype, X64 setting, device kind, and pinned tolerances:

```bash
scpn-control validate-jax-gk-parity --require-parity-artifacts --require-cases cyclone_base_case,tem_kinetic_electron,stable_mode --require-backends cpu,gpu --json-out
python validation/validate_jax_gk_parity.py --require-parity-artifacts --output-json artifacts/jax_gk_parity_report.json
python validation/validate_jax_gk_parity.py --require-parity-artifacts --require-cases cyclone_base_case,tem_kinetic_electron,stable_mode --require-backends cpu,gpu
```

Strict mode now admits the persisted CPU and GPU parity campaign in
`validation/reports/jax_gk_parity/` for CBC, kinetic-electron TEM, and
low-drive stable-mode cases. Live smoke tests remain useful diagnostics, but
they do not replace persisted CPU/GPU/TPU parity evidence, and parity evidence
does not replace external-code GK validation.

The reader enforces finite floating JSON tokens and duplicate-key refusal at
all depths, safe numeric conversion, string case/backend membership, literal
policy booleans and canonical metadata/payload digests. Malformed case lists,
very large scalar or case-bound integers and nonfinite extra fields return
structured refusal. It does not execute JAX or the native solver. The two
mode spectra are ordered string comparisons, with dominant membership and
declared acceptance modes/bounds; these checks do not authenticate physical
classification, backend/device/version/timestamp metadata or case inputs.

File inputs are read directly; directory inputs include every immediate JSON
child and ignore nested directories. Admitted duplicate files count separately.
Admitted entries remain in a failed overall report, and named coverage uses
only admitted entries. Missing paths with no requirements may pass empty;
unsupported required names raise a configuration `ValueError`. Artifact and
report canonical digests omit both top-level self-digest keys; nested metadata
digests include all keys. The raw file bytes are not digested by this reader.
The standalone script handles supported output IO/invalid-path failures as
FAIL findings and updates its report digest; refused output is not admission.
Its empty-directory native example runs without optional numerical packages.

Stored parity declarations also refuse nonzero decimal tokens that collapse
to zero in binary64. Decode/read findings use fixed authored IO/UTF8/JSON/
duplicate/nonfinite/underflow text, with no raw exception or private duplicate
member name. The original digest algorithms, drift formulas, ordered modes,
case/backend requirement normalisation and declared bounds remain unchanged.

Public `write_jax_gk_parity_report` and both commands protect the root and
immediate selected direct/resolved/symlink/existing hardlink aliases before
writing sorted UTF8 JSON plus LF with nonfinite serialisation refused. Other
destinations may replace. Reads and checks are sequential, not pathname locks
or snapshots. API failures propagate. The standalone CLI retains its original
output failure contract: append an `output_json` finding, set FAIL, update the
report digest, and emit JSON/text. The root command uses fixed operational
refusal one; Click NUL output is usage two before filesystem operations.
Neither path launches a solver, refreshes scientific evidence or admits control.

```python
from tempfile import TemporaryDirectory
from validation.validate_jax_gk_parity import validate_jax_gk_parity

with TemporaryDirectory() as directory:
    report = validate_jax_gk_parity(directory)
assert report["parity_artifacts"] == 0
assert report["complete_required_case_backend_coverage"] is None
```

The strict validator now emits aggregate case/backend coverage, backend counts,
case counts, maximum gamma and real-frequency drift, an entries payload digest,
and a report payload digest. The benchmark producer writes local timing evidence
to `validation/reports/jax_gk_parity_benchmark.json` and
`validation/reports/jax_gk_parity_benchmark.md` outside the parity-artifact
directory, preserving strict artifact admission. The recorded workstation CPU
benchmark regenerated the three CPU cases in `2.963800` seconds and the
persisted CPU/GPU gate still passes with six artifacts and complete
case/backend coverage.

GK OOD detector deployment claims require persisted calibration artefacts with a
declared 10D feature schema, training-distribution metadata, threshold
provenance, and false-positive / false-negative acceptance metrics:

```bash
scpn-control validate-gk-ood-calibration --require-campaign-artifacts --json-out
python validation/validate_gk_ood_calibration.py --require-campaign-artifacts --output-json artifacts/gk_ood_calibration_report.json
```

The reader inspects author-declared campaigns using the original
`scpn-control.gk-ood-calibration-artifact.v2` and report-v2 schemas. A directory
selects immediate sorted JSON entries; any regular file selects itself. Optional
missing/nonfile roots pass zero; required absence refuses. Duplicate campaign IDs
refuse after the first accepted declaration. It launches no external code or
calibration fitting and authenticates no published/facility/source provenance.

Each entry binds the exact captured UTF-8 bytes, including CRLF, and the decoded
compact sorted ASCII JSON. The report's original canonical hash uses its
`payload_sha256=None`. Fixed IO/UTF-8/JSON/duplicate/nonfinite/nonzero-underflow
findings omit raw exception text and duplicate member names. Selection, reads
and path checks are sequential, with no concurrent snapshot or pathname lock.

The declared ten-feature order remains exact. Means must be finite signed
representable nonboolean values; standard deviations must be finite and
nonnegative. Zero deviation remains valid descriptive metadata. Sample count
is an uncapped positive nonboolean integer, not an allocation request. Three
detector thresholds are finite and positive, matching the public `OODDetector`
constructor. All six observed/maximum/minimum rate values lie in inclusive
`[0, 1]`. Original false-positive/negative `<= maximum` and recall `>= minimum`
comparisons are unchanged; equality passes. Rates are not recomputed against
held-out cases. A covariance method, exact 64-character ASCII hex digest,
literal `positive_definite=true` and feature order are metadata checks: no
covariance bytes are fetched or matrix definiteness independently established.

The original `deployment_calibration_admitted` flag becomes true only for a
passing metadata report with at least one accepted campaign. It installs no
calibration and grants no measured, physical, facility, operator or control
admission; `full_gk_operating_envelope_admitted` remains false. The historical
`validation/reports/gk_ood_calibration.json` remains untouched and blocked with
zero campaigns and payload SHA-256
`1d81ac7337eaa3370dc7dd8e003b394fcb0684cdc41b60b74f5e4e6f87a39f70`.

Public `write_gk_ood_calibration_report` and both commands protect root/selected
input direct/resolved/symlink/existing hardlink aliases before writing sorted
UTF-8 JSON with a trailing newline. Other output paths may be replaced.
Operational failures return one with fixed text; root output NUL is Click
usage two. No source evidence is resealed during report persistence.

```python
from tempfile import TemporaryDirectory
from validation.validate_gk_ood_calibration import validate_gk_ood_calibration

with TemporaryDirectory() as directory:
    report = validate_gk_ood_calibration(directory)
assert report["campaign_artifacts"] == 0
assert report["public_claims"]["deployment_calibration_admitted"] is False
```

External GK interface parser claims require persisted artefacts from real
solver executables or documented public reference outputs. Mock subprocess
fixtures remain parser-readiness checks only:

```bash
scpn-control validate-gk-interface-artifacts --require-interface-artifacts --json-out
python validation/validate_gk_interface_artifacts.py --require-interface-artifacts --output-json artifacts/gk_interface_artifacts_report.json
```

The strict report uses the `scpn-control.gk-interface-artifact-report.v2`
schema and binds the canonical report payload by SHA-256. Current local
evidence in `validation/reports/gk_interface_artifacts.json` remains blocked
with zero admitted interface artefacts and payload SHA-256
`141d89e3b413b58b62af84b39ed95b5b8d9ef43425b9b232e6aebd0ed06d6f85`.

Strict mode fails until `validation/reports/gk_interfaces/` contains interface
artefacts using schema `scpn-control.gk-interface-artifact.v1` with code
identity, source provenance, version, run id, execution timestamp, safe deck,
raw-output, and parsed-output artefact URIs, SHA-256 hashes for each of those
artefacts, a canonical payload SHA-256 hash, parser version, explicit `m^2/s`,
`c_s/a`, and `k_y*rho_s` units, finite transport coefficients, growth rate,
real frequency, and dominant wavenumber. Real-executable artefacts must also
declare an admitted absolute `binary_path`; URI, relative, traversal,
temporary, or system-control paths are not accepted as executable provenance.

The interface reader inspects author-declared metadata. The original
`external_interface_artifacts_admitted` flag means the report passed with at
least one declaration; it does not authenticate any executable, deck, raw or
parsed output bytes, parser execution or public-reference publication. The
full cross-code claim stays false. Referenced resources are not fetched.

Selection is sorted immediate `*.json` in a directory, or a single regular
file of any suffix. Optional missing/nonfile roots pass with zero entries;
required absence fails. Duplicate code/run pairs fail after the first accepted
entry. The original lexical URI-prefix rules remain, including bare prefixes;
there is no remote resource admission or download. All six numeric fields must
be finite, representable and nonboolean. Transport coefficients and growth
rate are nonnegative, dominant wavenumber positive, and real frequency signed.
Unit tokens retain the original substring checks. SHA-256 fields contain
exactly 64 ASCII hex characters; case is accepted, trailing newlines refused.

Captured input bytes bind `artifact_file_sha256`, including CRLF line endings.
Public `canonical_artifact_sha256` preserves the original sorted compact ASCII
JSON hash over a shallow copy excluding only `payload_sha256`. The report
hash includes its actual `generated_at_utc`; separate real calls can have
different timestamps and report hashes. Stored declarations refuse duplicate
keys, nonfinite JSON numbers and nonzero decimal underflow at every depth;
fixed IO/UTF8/JSON findings expose no raw exception or duplicate member name.

Public `write_gk_interface_artifacts_report` and both commands protect the root
and immediate selected direct/resolved/symlink/existing hardlink aliases before
writing sorted UTF8 JSON plus LF with nonfinite serialisation refused. Other
destinations may replace. Reads and alias checks are sequential; they are not
pathname locks or snapshots. API path/IO failures propagate; script/root emit
fixed operational refusal one. Click NUL output is usage two before filesystem
operations. No scientific source report is refreshed or resealed by inspection.

```python
from tempfile import TemporaryDirectory
from validation.validate_gk_interface_artifacts import validate_gk_interface_artifacts

with TemporaryDirectory() as directory:
    report = validate_gk_interface_artifacts(directory)
assert report["interface_artifacts"] == 0
assert report["public_claims"]["external_interface_artifacts_admitted"] is False
assert report["public_claims"]["full_gk_cross_code_claim_admitted"] is False
```

Neural equilibrium cross-validation claims require persisted P-EFIT or
documented public reference artefacts for the same surrogate weights and
equilibrium cases. Synthetic training runs and local smoke tests do not count as
matched equilibrium-reference evidence:

Public MAST Level 1 EFM measured-shot campaigns can be converted into
reference-candidate arrays on storage-host dataset storage with:

```bash
ssh storage-operator@storage-host '/data/SCPN-CONTROL/.venv/bin/python /path/to/SCPN-CONTROL/validation/convert_mast_efm_neural_equilibrium_reference.py --dataset-root /data/SCPN-CONTROL --campaign-manifest /data/SCPN-CONTROL/manifests/mast_level1_efm_campaign_30419_30424.json --output-root /data/SCPN-CONTROL/converted/neural_equilibrium_reference --report-out /data/SCPN-CONTROL/converted/neural_equilibrium_reference/mast_efm_neural_equilibrium_reference_candidate.json'
```

Install the `mast-data` extra in the selected project environment before
conversion (`python -m pip install '.[mast-data]'`). The declared runtime reads
consolidated Zarr v2 through xarray with `chunks=None`, so Dask is optional.
The converter requires observed, finite, strictly increasing source time;
selected reconstruction times must be nonnegative. Negative pretrigger rows
remain observations and cannot enter the output without positive convergence.
An observed scalar scheduler status is a whole-shot guard; per-time status must
have the exact time dimension. Other required variables retain explicit time
dimensions; successful status **and** positive convergence flags;
and source units `A`, `T`, and `T-rad` for current, axis field, and FF-prime.
It preserves the exact spatial coordinates, including descending grids, and
computes finite profile RMS while preserving ordinary float64 reduction bytes.
Extreme magnitudes use scaling to avoid overflow and retain positive tiny RMS.
Missing time, failed
convergence, text/boolean/complex coercion, and empty selected target rows are
refused. Source and report paths are checked before writes. An existing selected
reference output can be replaced by a subsequent conversion; callers retain
immutable copies when they need archival custody.

The converter writes compressed `.npz` reference arrays and a
schema-versioned candidate report only. It deliberately does not emit a
passing `scpn-control.neural-equilibrium-reference.v1` artefact until
pressure reconstruction, exact-weight predictions, reference/prediction
SHA-256 digests, metrics, tolerances, and strict admission evidence are
present.

Current storage host conversion evidence from the acquired campaign produced 527
finite converged reference-candidate equilibria across shots 30419-30424
with candidate report payload SHA-256
`8d173f423440243c4362256480e7ec40a8ca16244ac862b727428d6f28f747e5`.
Converted bundles now persist exact public EFM `profile_r` and `profile_z`
coordinate grids as `r_grid_m` and `z_grid_m` with lengths 129 and 65.
The report remains `admission_ready=false` and is intentionally not a
passing predictive EFIT/P-EFIT admission artefact.

Current-model prediction evidence can be generated with:

```bash
python validation/evaluate_mast_efm_neural_equilibrium.py --reference-path /data/SCPN-CONTROL/converted/neural_equilibrium_reference/mast_efm_shot_30419_reference.npz --weights-path /data/SCPN-CONTROL/converted/neural_equilibrium_reference/neural_equilibrium_synthetic_65x129_weights.npz --prediction-path /data/SCPN-CONTROL/converted/neural_equilibrium_reference/evaluation_predictions/mast_efm_shot_30419_prediction.npz --json-out /data/SCPN-CONTROL/converted/neural_equilibrium_reference/evaluation_predictions/mast_efm_shot_30419_evaluation.json --report-out /data/SCPN-CONTROL/converted/neural_equilibrium_reference/evaluation_predictions/mast_efm_shot_30419_evaluation.md
```

The evaluator uses the supervised producer's twelve feature definitions.
Present `Ip_MA` and `Bt_T` arrays supply acquired observations; wholly absent
keys retain explicit 8 MA and 5 T diagnostic defaults. Malformed present
observations refuse. Supply `--ffprime-reference VALUE` using the
`feature_source_policy.ffprime_scale.campaign_reference` from the dataset report
for the model's actual training campaign. A single evaluation shot cannot define
the campaign median. Without both source RMS and the explicit reference,
FF-prime retains neutral fallback 1. This option does not certify that selected
weights were trained with that campaign or those feature definitions.

Reference arrays are decoded from checksum-verified captured NPZ bytes. Existing
weights are captured once and loaded from a temporary snapshot beside the
selected prediction output; the snapshot is removed after loading. The report
binds those captured reference and weight bytes. No training or transactional
snapshot across input files occurs. Prediction, JSON and Markdown outputs must
be distinct from one another and either input, including resolved symlinks and
existing hard links. Non-alias output files may be replaced. The typed NPZ writer
uses the exact selected filename, even without a `.npz` suffix. CLI success is
exit 0; supported IO/model/value failures give an authored refusal and exit 1;
argparse help/usage retain exits 0/2.

Flux RMSE uses jointly finite masked points and is measured in Wb/rad. Historical
`psi_rmse_Wb` is an alias without `2*pi` conversion. `boundary_rmse_m` aliases
the directed predicted-to-reference mean nearest-point distance; it is not root
mean square or symmetric. Contours collect rounded edge crossings and do not
establish connectedness or enclosure. The feature named `q95` uses the last
finite masked profile sample, not interpolation at 95% flux. Geometry/profile
features derived from the same reference are not independent prediction evidence
for those quantities. Reports record fallback names, feature notes, explicit
normalisation and metric-unit notes. Pressure/q predictions, independent
tolerances and predictive admission remain open.

The following historical scores used the earlier fallback projection. They do
not describe source-derived evaluation or an unchanged-input model benchmark.

Scoped 2026-06-01 evaluation over shots 30419-30424 used full 65 x 129
reference grids, exact public EFM coordinates, and matching-grid synthetic-domain
weights to exercise the current model prediction path. Flux masked RMSE values
were 1.574623069235, 1.643688910187, 1.565222714156, 1.486059078976,
1.499524077369, and 1.561932368275 Wb/rad for shots 30419-30424 respectively.
Derived magnetic-axis RMSE values were 0.800979524200, 0.783302289712,
0.797510280021, 0.725725041659, 0.724815492014, and 0.797042852897 m. Derived
LCFS mean-distance values were 0.594233082526, 0.490801237843,
0.592467676720, 0.479744649844, 0.484618427508, and 0.593385388932 m. These
reports remain `admission_ready=false` and `strict_artifact_emitted=false`
because the model path does not yet produce pressure or q-profile predictions
and some required diagnostic inputs are represented by documented fallback
features.

The repository-published campaign summary is checked in as
`validation/reports/mast_efm_neural_equilibrium_campaign.json` and
`validation/reports/mast_efm_neural_equilibrium_campaign.md`. The compact
report aggregates all six shot evaluations, records storage-relative paths and
SHA-256 digests for the internal `.npz` payloads, and keeps the admission state
blocked until the full-output predictive contract is satisfied. The current
aggregate flux RMSE mean is 1.5551750363663988 Wb/rad, the magnetic-axis RMSE
mean is 0.7715625800838742 m, and the LCFS mean-distance mean is
0.5392084105619522 m across 527 evaluated equilibria.

The supervised MAST EFM neural-equilibrium dataset can be rebuilt on admitted
compute where the original stores and campaign manifest are staged. Select that
storage root explicitly and generate a new reference candidate for its actual
paths. The following commands create a separate derivative; retain historical
references, tensors and repository-published reports:

```bash
SCPN_MAST_STORAGE=/data/SCPN-CONTROL
SCPN_MAST_DERIVED="$SCPN_MAST_STORAGE/derived/current-contract"

python validation/convert_mast_efm_neural_equilibrium_reference.py \
  --dataset-root "$SCPN_MAST_STORAGE" \
  --campaign-manifest "$SCPN_MAST_STORAGE/manifests/mast_level1_efm_campaign_30419_30424.json" \
  --output-root "$SCPN_MAST_DERIVED/reference" \
  --report-out "$SCPN_MAST_DERIVED/reference/candidate.json"

python validation/build_mast_efm_neural_equilibrium_dataset.py \
  --candidate-report "$SCPN_MAST_DERIVED/reference/candidate.json" \
  --storage-root "$SCPN_MAST_STORAGE" \
  --output-npz "$SCPN_MAST_DERIVED/dataset.npz" \
  --json-out "$SCPN_MAST_DERIVED/dataset.json" \
  --report-out "$SCPN_MAST_DERIVED/dataset.md"
```

Choose an unused derivative directory for another rebuild. The candidate embeds
the selected paths and new reference digests; do not rewrite its locators or
reseal it to relocate a campaign. Generate new source audits and a new plan from
this derivative's dataset report, then explicitly select all of them in the
trainer dry-run. Current LCFS counts are the actual valid-point counts, with
contiguous valid prefixes and NaN/False padding. The producer's overflow-safe
pressure mean can also change ordinary rounding in `pprime_scale`; the new
dataset SHA binds the complete resulting tensor. Passing this preparation chain
does not fit a model or establish predictive admission.

The repository-published dataset report is checked in as
`validation/reports/mast_efm_neural_equilibrium_dataset.json` and
`validation/reports/mast_efm_neural_equilibrium_dataset.md`. The large numeric
dataset is declared on storage-host at
`processed/neural_equilibrium/mast_efm_supervised_dataset.npz` with SHA-256
`3206bd530efdd6fc73bae57b2ac18646aff39e130533c7d5167abe1ae7d136f3`.
The deterministic shot-held-out split contains 340 training equilibria from
shots 30419-30422, 80 validation equilibria from shot 30423, and 107 test
equilibria from shot 30424. LCFS boundary targets are preserved with padded
coordinate arrays, `False` padded validity masks, and per-slice
`lcfs_point_count` metadata up to 157 boundary points. The former fallback
feature columns are now sourced from public MAST EFM metadata: `Ip_MA` comes
from `plasma_current_x` with A-to-MA conversion, `Bt_T` comes from `bphi_rmag`,
and `ffprime_scale` comes from per-time-slice `ffprime` RMS magnitude
normalised by the campaign median and clipped to `[0.25, 4.0]`. These historical declarations do not establish current payload availability
or source admission. Predictive EFIT/P-EFIT admission remains blocked until
a full-output model passes declared tolerances
for flux, pressure, q-profile, LCFS geometry, and magnetic-axis outputs.

The current producer verifies the actual converter candidate self-digest and
selected NPZ byte digests before accepting shape, shot, time, mask and grid
custody. Hash verification and NPZ decoding use the same captured compressed
bytes, so pathname mutation cannot substitute decoded input after hashing.
The capture retains one compressed bundle temporarily; it does not freeze
all campaign files or authenticate physical sources. The supervised trainer
uses the same verified-NPZ reader for its selected dataset, so both sides bind
actual decoded arrays to the verified byte capture. Descending grids are normalised with their flux/masks, and LCFS holes
are compacted into contiguous valid prefixes with true point counts. Ip/Bt/FF
provenance requires complete selected-shot keys; absent keys stay declared
fallbacks. Finite large pressure/RMS values use stable mean/median arithmetic.
Reports preserve blocked admission and validate storage-relative paths,
source units/transforms, partition totals, time ranges and grid declarations.
The [public API contract](api.md) describes sequential-write and CLI failures.

The canonical 527-equilibrium NPZ is external to this checkout. Select the
actual mounted dataset storage explicitly; the default `/data/SCPN-CONTROL`
path is not remote-storage discovery. Both retained feature-source audit
snapshots have stale self-digests; the retained original audit also uses the
descriptor-only v1 schema. Current original auditing requires v2 evidence.
Legacy supervised tensors must satisfy the current contiguous LCFS mask/count
and NaN padding contract before fitting. Rebuild incompatible tensors from
their verified converted references, preserving old artifacts and declaring new
digests. Engineering format tests do not regenerate canonical scientific reports
or establish predictive admission.

Training is prepared as an explicit campaign plan rather than launched during
documentation or release work:

```bash
python validation/plan_neural_equilibrium_training_campaign.py \
  --mast-dataset-report "$SCPN_MAST_DERIVED/dataset.json" \
  --storage-root "$SCPN_MAST_STORAGE" \
  --require-storage-payload \
  --json-out "$SCPN_MAST_DERIVED/plan.json" \
  --report-out "$SCPN_MAST_DERIVED/plan.md"
```

Historical plan snapshots are checked in as
`validation/reports/neural_equilibrium_training_campaign_plan.json` and
`validation/reports/neural_equilibrium_training_campaign_plan.md`. A fresh plan
records the MAST EFM dataset declaration, deferred QLKNN/QuaLiKiz payloads,
external dataset requirements, run order and GPU-hour planning ranges. These
ranges are assumptions, not measured throughput or reserved GPU capacity.
The planner does not launch training or download data.

`build_plan(CampaignInputs(...))` now requires the real public acquisition
inspector to PASS; an empty, invalid or partially failing manifest tree raises
`CampaignPlanError` with the original diagnostic entries. Dataset UTF-8 JSON
must have unique keys and finite numbers. The consumed reference identity,
lowercase SHA-256, positive integer count/grid, exact train/validation/test
counts with matching total, fallback strings and ragged-target declarations
are validated without boolean/numeric coercion. A provided producer
`payload_sha256` must match canonical JSON with that field set to null. This
checks metadata consistency; it does not verify target units or split membership.
Dataset and candidate-report paths must be safe relative spellings and remain
canonically within the selected storage root, including through symlinks.

A present local dataset must be a regular file whose streamed SHA-256 matches
the declaration, even with `--verified-storage-payload`. A missing required
payload raises `FileNotFoundError` unless that flag explicitly records remote
operator attestation. The flag does not inspect remote storage. The payload
fields `availability_basis`, `exists_on_this_host` and
`sha256_verified_on_this_host` distinguish local byte verification, remote
attestation and unobserved storage. The historical `prepared_on_storage` lane
label describes declared storage, not proof that the file is mounted locally.
Neither a digest nor an attestation admits predictive or facility claims.

JSON and Markdown output paths must be distinct, including symlink/hardlink
aliases. The writer checks schema/status, finite canonical self-digest and
renderability before persistence. Writes are sequential: a second-file IO
failure can leave the JSON file, and the CLI still returns 1 with authored
`FAIL:` text. Success returns 0; argparse help/usage retain 0/2. For inspection
without updating checked-in snapshots, supply both outputs explicitly:

```bash
python validation/plan_neural_equilibrium_training_campaign.py \
  --json-out /tmp/control-campaign-plan.json \
  --report-out /tmp/control-campaign-plan.md
```

CLI commands rendered in the plan quote storage paths as shell arguments.
Planning never creates weights, reads numerical tensor formats or grants
execution authority.

Audit the derivative's converted channels and original public MAST Level 1 EFM
Zarr observations using the same dataset declaration and storage selection:

```bash
python validation/audit_mast_efm_feature_provenance.py \
  --dataset-report "$SCPN_MAST_DERIVED/dataset.json" \
  --storage-root "$SCPN_MAST_STORAGE" \
  --json-out "$SCPN_MAST_DERIVED/feature.json" \
  --report-out "$SCPN_MAST_DERIVED/feature.md"

python validation/audit_mast_efm_original_feature_sources.py \
  --dataset-report "$SCPN_MAST_DERIVED/dataset.json" \
  --storage-root "$SCPN_MAST_STORAGE" \
  --json-out "$SCPN_MAST_DERIVED/original.json" \
  --report-out "$SCPN_MAST_DERIVED/original.md"
```

The repository-published original-source audit is checked in as
`validation/reports/mast_efm_original_feature_source_audit.json` and
`validation/reports/mast_efm_original_feature_source_audit.md`. That historical
v1 snapshot reads only consolidated metadata. The current producer emits
`scpn-control.mast-efm-original-feature-source-audit.v2`: it captures each actual
source file into immutable bytes, hashes those bytes, decodes the same capture
through xarray/Zarr, and requires physical observation chunks. It compares all
20 converted time/grid/channel/target/mask arrays exactly against the SHA-bound
selected references, allowing ordered subsets of observed times. Missing or
unreadable observations, unsupported aliases and unrelated targets cannot grant
readiness from descriptors. Preferred channel metadata and conversion/reference
checks must pass on every shot, together with a complete actual converted-feature
audit bound to the selected full producer declaration.

Captures read files sequentially and temporarily retain compressed source bytes
plus decoded arrays; they are not a coherent directory/campaign transaction or
an acquisition signature. Owners control concurrent source changes. Report
validation checks declared consistency; it does not reopen the original corpus.
The writer protects selected dataset/candidate/reference/source inputs and
refuses output aliases or outputs inside a source group. JSON then Markdown
persistence is sequential, so a second-file failure does not imply rollback.
The CLI returns 0 for a complete ready or blocked report, 1 for ordinary input
or write refusal, and argparse help/usage retain 0/2.

The preserved audit declares `source_ready` and names `plasma_current_x` with an `A_to_MA`
conversion for `Ip_MA`, `bphi_rmag` as the total toroidal field at the magnetic
axis for `Bt_T`, and `ffprime` with the declared RMS plus campaign-median
normalisation policy for `ffprime_scale` across all six shots. The supervised
dataset report declares no fallback features. The current converted-feature
and original-source audit bytes both have stale canonical self-digests, so the
trainer refuses their source admission. V1 descriptor-only reports are not
silently upgraded or resealed. Generate and explicitly select fresh audits from
actual verified observations before execution; historical declarations do not
prove current source custody or numerical tensor compatibility.

The dry-run-first full-output baseline trainer can be prepared with:

```bash
python validation/train_mast_efm_neural_equilibrium.py \
  --dataset-report "$SCPN_MAST_DERIVED/dataset.json" \
  --campaign-plan "$SCPN_MAST_DERIVED/plan.json" \
  --dataset-path "$SCPN_MAST_DERIVED/dataset.npz" \
  --feature-provenance-report "$SCPN_MAST_DERIVED/feature.json" \
  --original-source-report "$SCPN_MAST_DERIVED/original.json" \
  --compute-host-kind workstation \
  --compute-host-label local-staged-MAST-dry-run \
  --weights-out "$SCPN_MAST_DERIVED/UNWRITTEN-weights.npz" \
  --json-out "$SCPN_MAST_DERIVED/dry-run.json" \
  --report-out "$SCPN_MAST_DERIVED/dry-run.md" \
  --templates-json-out "$SCPN_MAST_DERIVED/templates.json" \
  --templates-report-out "$SCPN_MAST_DERIVED/templates.md"
```

With the fresh dataset, both source audits and campaign plan generated above,
this writes launch and template reports to the explicit derivative outputs
without creating weights. Inspect its `pre_run_admission` result; CLI success
alone does not establish source/compute readiness. Default
output paths still name checked-in snapshots; those snapshots are historical
preparation declarations. The historical launch preserves the expected dataset SHA-256
`3206bd530efdd6fc73bae57b2ac18646aff39e130533c7d5167abe1ae7d136f3`, records
that the storage-host dataset payload is not mounted on this workstation, and remains
fail-closed until the data are mounted read-only or copied to admitted compute
storage. The launch report payload digest is
`fc8724dc72801e8a92126a4e5cd46fd574f33eb320cb6889fd37bc6ae90d2b7d`. The
companion result-template report is
`validation/reports/mast_efm_neural_equilibrium_result_templates.json` with
payload digest `ca3c80f970e63ca50ace0186caf7555de2d0476a0374716cfbd8940a20d04d28`.
The storage host is storage-only: the exact `--execute` command must be run only on this
workstation or external cloud compute with the storage-host dataset mounted read-only or
copied to admitted compute storage. The trainer now validates launch and result
template reports before persistence, rejects tampered payload digests, and
performs a strict pre-run admission check before `--execute`: campaign metadata
and self-digest must match all consumed dataset bindings, the acquisition
summary must declare PASS, and both source audits must have current canonical
self-digests and matching full dataset declarations. The v2 original audit
consumes complete converted-source bindings, including raw dataset-report bytes,
payload and tensor digests, locators, reference identity/count and each shot's
path/SHA/rows. Changing only report whitespace requires fresh byte captures in
both audits. The selected dataset
SHA-256 must match the published supervised-dataset report, the converted
feature-provenance audit must have no blocked features, the original
public-source audit must be `source_ready`, the compute host must be declared as
`workstation` or `external_cloud`, and `weights_out` must not be under
storage-host dataset storage. Execution mode trains deterministic ridge/PCA
baseline heads for flux,
pressure-gradient profile, q-profile, LCFS geometry, and magnetic-axis outputs,
then writes weights and compact train, validation, and test metrics. Predictive
admission still requires an executed training artefact, holdout metrics, exact
weight checksum validation, and the strict reference admission gate.

The historical default result-schema template paths are:

```text
validation/reports/mast_efm_neural_equilibrium_result_templates.json
validation/reports/mast_efm_neural_equilibrium_result_templates.md
```

These templates define the required holdout-metric, latency, GPU-cost, and
admission-certificate fields for the later workstation or cloud compute run.
They are not executed training evidence.

Feature provenance for the current converted public MAST EFM bundles can be
audited with:

```bash
python validation/audit_mast_efm_feature_provenance.py \
  --dataset-report validation/reports/mast_efm_neural_equilibrium_dataset.json \
  --storage-root /data/SCPN-CONTROL
```

The generated audit is checked in as
`validation/reports/mast_efm_feature_provenance_audit.json` and
`validation/reports/mast_efm_feature_provenance_audit.md`. Its preserved status
declares PASS and lists flux, masks, pressure-gradient, q-profile, LCFS, axis,
grid, shot, time, `Ip_MA`, `Bt_T` and `ffprime_rms_T_rad` arrays. Its
self-digest currently does not match the content, so source admission remains
FAIL. Listed keys do not establish source custody or predictive admission.

The current auditor validates the complete dataset producer declaration and
captures each selected reference NPZ once, verifies its declared SHA-256, then
decodes those same bytes without pickle. It checks shot/time/grid bindings and
requires the actual producer's canonical `Ip_MA`, `Bt_T` and positive
`ffprime_rms_T_rad` vectors on every selected shot. Each channel must have the
declared row count, a real numeric dtype and finite float64 values. One missing
shot remains blocked; alternative key aliases are inventory hints until the
converter supports their units and transformations. Changed container bytes
require a new binding even when all decoded arrays are identical.

Audit PASS means complete converted channels. Original acquisition authenticity,
target validity and supervised tensor contents have separate checks. Writers
validate finite declarations, self-digests and per-shot aggregate status before
rendering or persistence, and refuse output aliases of selected inputs. Pair
writes remain sequential: a second-output failure may leave the first JSON file.
The CLI exits 0 for a completed pass or blocked inspection, 1 with an authored
`FAIL` for invalid or missing local inputs, and 2 for invalid arguments. The
preserved scientific reports have not been regenerated or resealed.

The trainer consumes the complete converted audit through the same validator.
It requires the exact captured dataset JSON SHA, producer payload SHA, declared
dataset NPZ SHA, dataset/candidate locators, reference identity/count and every
shot's source path/SHA/row binding. An audit for different bytes or references
cannot pass source admission even if its own digest and PASS label are valid.
Whitespace-only dataset JSON changes require a newly captured audit. Dry-run
records FAIL diagnostics and execution stops before fitting or weight writes.
The consumer compares declarations without reopening source NPZ or authenticating
original acquisition; those separate scientific gates remain required.

The trainer validates the complete selected NPZ layout before fitting:
finite real `features` of shape N × 12 in the declared feature order;
train/validation/test labels and exact metadata counts; positive integer shot
IDs without cross-split shot leakage; finite nonnegative times; increasing
grids of the declared dimensions; flux grids and boolean masks; finite axis
and flux-boundary scalars; nonempty pressure/q profiles; and LCFS targets with
matching integer counts and contiguous valid prefixes. Masked target values
remain unobserved. The public builder's schema, feature list and final feature
matrix validator are shared. Features, scalar/grid fields and observed target
values must be finite in the actual float64 computation representation, not
merely in a wider stored dtype. Wider-dtype overflow refuses before training;
unobserved masked values retain their existing missing-data treatment.

Normalisation, masked-value filling and PCA use training rows only. Ridge
regularisation is finite and positive, and its unpenalised intercept count is
assigned directly to avoid cancellation at large alpha. At least two training
rows are required. Numerical failures and nonfinite coefficients/predictions
refuse before weight persistence. Null holdout metrics mean no observed data,
not a passing tolerance. Schema regression fixtures are engineering evidence,
not authenticated physical MAST training or benchmark evidence.

Launch validation requires finite canonical JSON, typed pre-run diagnostics,
literal blocked admission flags and complete required metrics. Execute-mode
`require_executed=True` additionally verifies the actual selected weight file
SHA-256. Result-template validation enforces supported section schemas, typed
policy text, distinct required members and exact launch/dataset bindings; it
does not validate measured future results.

JSON/Markdown outputs must be distinct, including symlink/hardlink aliases.
The CLI checks all four outputs against each other and selected input/weight
paths before training. Writers validate and render before sequential writes;
a second-file failure may leave JSON. Supported domain/IO failures return 1
with `FAIL:` text, success returns 0, and argparse help/usage retain 0/2.
Paths rendered into execution commands are shell quoted. Reads and writes are
sequential observations, not an atomic filesystem snapshot.

Synthetic neural-equilibrium pretraining evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family neural-equilibrium-pretraining \
  --artifact weights=validation/reports/neural_equilibrium_synthetic_pretrain.npz \
  --artifact report=validation/reports/neural_equilibrium_pretraining.json \
  --artifact markdown=validation/reports/neural_equilibrium_pretraining.md \
  -- python validation/benchmark_neural_equilibrium_pretraining.py
```

This writes `validation/reports/neural_equilibrium_pretraining.json`,
`validation/reports/neural_equilibrium_pretraining.md`, and JAX-compatible
synthetic pretraining weights. These artefacts demonstrate pretraining and
inference plumbing only; real EFIT/P-EFIT fine-tuning remains gated by the
strict reference-artefact validator below.

```bash
scpn-control validate-neural-equilibrium-reference --require-reference-artifacts --json-out
python validation/validate_neural_equilibrium_reference.py --require-reference-artifacts --output-json artifacts/neural_equilibrium_reference_report.json
```

The strict report uses the `scpn-control.neural-equilibrium-reference-report.v2`
schema and binds the canonical report payload by SHA-256. Current local
evidence in `validation/reports/neural_equilibrium_reference.json` remains
blocked with zero admitted reference artefacts and payload SHA-256
`bf6b89baaf1a81e7e93c1e7d9625da81d6ef8b134d339407905893b0ff1491d4`.

Strict mode fails until `validation/reports/neural_equilibrium_reference/`
contains artefacts using schema `scpn-control.neural-equilibrium-reference.v1`
with source provenance, surrogate identity, trained-weight SHA-256, safe
reference and prediction artefact URIs, reference/prediction/payload SHA-256
hashes, grid shape, target schema, psi/pressure/q/boundary unit contracts,
reference-equilibrium count, and error metrics inside declared tolerances. Real
P-EFIT artefacts must declare an admitted absolute `binary_path`; URI,
relative, traversal, temporary, or system-control paths are rejected before the
artefact can support predictive equilibrium claims.

These checks validate declarations and declared tolerances. The validator does
not read referenced arrays/weights, execute P-EFIT, fetch public references or
recompute the five errors. Its `reference_artifacts_admitted` flag denotes
accepted metadata; `predictive_equilibrium_claim_admitted` stays false. Source
authenticity, actual numerical evidence and model compatibility require their
own consumer/reviewer checks. The core fine-tuning consumer subsequently reads
supplied GEQDSK inputs; the claim-evidence consumer additionally checks the
selected weight digest. A validator pass alone does not establish facility
validity, and schema fixtures do not constitute executed P-EFIT evidence.

The reader captures each JSON file once and hashes its exact bytes, preserving
CRLF spelling. Malformed source/digest fields and unrepresentable scalar metrics
are findings. The shared writer used by both CLI surfaces refuses direct,
symbolic and existing hard-link input aliases before writing. It creates output
parents and replaces ordinary non-alias output; choose a report location outside
the input corpus. Optional empty scans return diagnostic pass/zero declarations,
while `--require-reference-artifacts` rejects them. Passing declaration reports
return 0, findings or authored operational refusals return 1, and parser usage
returns 2. Full units, JSON/checksum, input selection and persistence limits are
in the [API contract](api.md#neural-equilibrium-reference-declarations).

The neural-transport reader selects sorted immediate JSON files or a regular
file of any suffix. Optional absent roots pass with zero declarations; required
mode refuses. Invalid JSON/UTF-8, duplicate keys, IO and nonzero decimal tokens
rounded to binary64 zero yield authored findings. Reads are sequential.

Original schema `scpn-control.neural-transport-reference.v1` requires eleven
nonblank identities and exact feature order `R_LTi, R_LTe, R_Ln, q, s_hat, alpha,
Ti_Te, Zeff, collisionality, beta_e`; targets are `chi_i, chi_e, D_e,
unstable_branch`. Transport coefficients use `m^2/s`, input gradients are
`dimensionless`. Sample count is an uncapped positive nonboolean integer.
Four errors are finite nonnegative with positive finite inclusive bounds;
branch accuracy and its declared minimum both accept the inclusive interval
`[0,1]`, with accuracy at least its minimum. No metrics are recomputed.

Canonical body SHA uses original compact sorted ASCII-escaped JSON excluding
`payload_sha256`; this is caller-recomputable consistency, without producer
or referenced-byte authentication. Weight/reference/prediction SHA fields are
format-only. Artifact URIs retain lexical relative/http/https/doi/s3/gs-prefix
admission without fetching; NUL, absolute paths and parent components refuse,
but bare prefixes remain admitted. Public provenance requires URL or DOI
presence. `real_qualikiz` instead uses the existing shared absolute POSIX
executable declaration policy; existence/executability/version are untested.
No weight load, QuaLiKiz execution, training or physical admission occurs.

The existing core claim consumer separately requires an exact supplied weight
path and compares its actual SHA with the admitted declaration. Its matched
reference boolean records those declaration and byte-identity checks; it does
not independently read the reference/prediction arrays or recompute their
errors. Independent quantitative evidence remains a separate review obligation.

`write_neural_transport_reference_report` creates parents, writes sorted UTF-8
JSON and refuses direct/resolved/symbolic/existing-hardlink aliases of the root
or selected inputs before mutation. Unrelated output may be replaced. Registered
and script CLIs return 0 for declaration pass, 1 for findings or fixed authored
operational refusal, and usage/help 2/0. API IO/path/encoding errors propagate;
concurrent path changes are outside the sequential protection guarantee.

```python
from tempfile import TemporaryDirectory
from validation.validate_neural_transport_reference import (
    validate_neural_transport_reference,
    write_neural_transport_reference_report,
)
with TemporaryDirectory() as directory:
    report = validate_neural_transport_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    write_neural_transport_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

Neural transport surrogate validation claims require persisted QuaLiKiz or
documented public reference artifacts for the same QLKNN-style feature schema
and trained weights:

Bounded local neural-transport claim evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family neural-transport-claims \
  --artifact report=validation/reports/neural_transport_claims.json \
  --artifact markdown=validation/reports/neural_transport_claims.md \
  -- python validation/benchmark_neural_transport_claims.py
```

This writes `validation/reports/neural_transport_claims.json` and
`validation/reports/neural_transport_claims.md`. These artefacts demonstrate
local fallback-regression and claim-admission plumbing only; quantitative
QuaLiKiz, QLKNN, or measured transport validation remains gated by the strict
reference-artifact validator below.

```bash
scpn-control validate-neural-transport-reference --require-reference-artifacts --json-out
python validation/validate_neural_transport_reference.py --require-reference-artifacts --output-json artifacts/neural_transport_reference_report.json
```

Strict mode fails until `validation/reports/neural_transport_reference/`
contains artifacts using schema `scpn-control.neural-transport-reference.v1`
with source provenance, surrogate identity, weight SHA-256, safe reference and
prediction artifact URIs, reference/prediction/payload SHA-256 hashes,
QLKNN-10D feature ordering, target schema, target-unit contracts,
reference-sample count, and chi_i/chi_e/D_e plus branch-accuracy metrics inside
declared tolerances. Real QuaLiKiz artifacts must declare an admitted absolute
`binary_path`; URI, relative, traversal, temporary, or system-control paths are
rejected before the artifact can support quantitative transport claims.

The persisted neural-turbulence reader selects sorted immediate JSON files or
one regular file regardless of suffix. Optional absent/nonfile roots pass with
zero entries; required mode refuses. Each file is read once. UTF-8/JSON/IO,
duplicate keys and nonzero decimal underflow return authored findings without
input exception text. Observations are sequential, without a directory snapshot.

Schema `1.0` retains seven nonblank identity declarations and two exact 64-hex
hash shapes. Hashes are format-only, with no reference or weight bytes fetched.
`real_gk_campaign` needs nonblank `campaign_artifact_uri`; public references need
nonblank URL or DOI. These are presence-only declarations, without URI parsing.
The exact feature order is `R_LTi`, `R_LTe`, `R_Ln`, `q`, `s_hat`, `alpha_MHD`,
`Ti_Te`, `nu_star`, `Z_eff`, `epsilon`. `Q_i`, `Q_e`, `Gamma_e` use `gyroBohm`,
with dimensionless input gradients. Reference sample count is an uncapped
positive nonboolean integer. Four nonnegative finite errors must be within
positive finite inclusive bounds; critical-gradient accuracy and its minimum
are both in `[0, 1]`, with equality admitted. Huge metric/bound integers whose
float conversion overflows become findings. No model is instantiated or fitted.

The separate `neural_turbulence_claim_evidence` admission builder calls this
reader and additionally hashes the supplied weight file. Its matching-reference
path can admit declared metrics; it does not independently load prediction or
reference arrays or recompute them. Reader pass alone does not prove quantitative
turbulence validation. The benchmark below constructs and fits a synthetic model;
reading stored artifacts does not execute that campaign.

The public writer creates parents and writes sorted UTF-8 reports, refusing
direct/resolved/symbolic/existing hardlink aliases of selected inputs before
writing. Unrelated output may replace. API operational errors propagate;
registered Click and script use fixed stderr and exit 1 for operational refusal,
0/1 for pass/findings and 0/2 for parser help/usage. Sequential alias observations
do not protect against concurrent pathname replacement.

```python
from tempfile import TemporaryDirectory
from validation.validate_neural_turbulence_reference import (
    validate_neural_turbulence_reference,
    write_neural_turbulence_reference_report,
)
with TemporaryDirectory() as directory:
    report = validate_neural_turbulence_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    write_neural_turbulence_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

Neural turbulence surrogate validation claims require persisted gyrokinetic
campaign or documented public reference artifacts for the same QLKNN-class
feature schema and trained weights:

Bounded local neural-turbulence claim evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family neural-turbulence-claims \
  --artifact report=validation/reports/neural_turbulence_claims.json \
  --artifact markdown=validation/reports/neural_turbulence_claims.md \
  -- python validation/benchmark_neural_turbulence_claims.py
```

This writes `validation/reports/neural_turbulence_claims.json` and
`validation/reports/neural_turbulence_claims.md`. These artefacts demonstrate
local analytic-target regression and claim-admission plumbing only; quantitative
gyrokinetic, QuaLiKiz, or measured turbulence validation remains gated by the
strict reference-artifact validator below.

```bash
scpn-control validate-neural-turbulence-reference --require-reference-artifacts --json-out
python validation/validate_neural_turbulence_reference.py --require-reference-artifacts --output-json artifacts/neural_turbulence_reference_report.json
```

Strict mode fails until `validation/reports/neural_turbulence_reference/`
contains artifacts with source provenance, surrogate identity, weight and
reference SHA-256 hashes, feature ordering, gyro-Bohm flux target units,
reference-sample count, and Q_i/Q_e/Gamma_e plus critical-gradient metrics
inside declared tolerances.

The persisted SOL blob reader selects sorted immediate `*.json` directory
entries or a regular file regardless of suffix. Optional absent/nonfile roots
pass with zero entries; required mode refuses. Invalid UTF-8/JSON, duplicate
keys, unreadable files and nonzero decimal tokens rounded to binary64 zero
produce authored findings. Reads are sequential observations.

Original schema `scpn-control.blob-transport-reference.v1` requires ten nonblank
source/dataset/time/URI/hash strings. There is no model or case-count requirement.
Three reference/profile/detector hashes require exact 64-hex format and fetch
no referenced bytes. The public `canonical_artifact_sha256` retains compact,
sorted, ASCII-escaped JSON excluding `payload_sha256`; matching the declared
body hash establishes caller-recomputable consistency, without producer or
reference-byte authentication. Uppercase body digests are normalised; malformed
or non-ASCII digests refuse before comparison without raising an exception.

Reference/profile/detector URI text retains lexical NUL/absolute/parent-path
refusal, with original relative or `http://`, `https://`, `doi:`, `s3://`, `gs://`
prefix admission. Text is not fetched or resolved; bare prefixes and lexical
relative-looking `file:///` text remain admitted. Public source needs URL or DOI
presence; measured probe campaign needs machine and shot OR campaign presence.
Units remain `m`, `s`, `m/s`, `m^-3`, `eV`, `T`, `m^-2 s^-1`, without conversion.
SOL coordinates require at least two finite nonnegative strictly increasing
values. Detector time AND blob-size ranges require two finite ordered values
with nonnegative lower and positive upper: both admit lower zero, preserving
the original implementation despite its former positive-size wording. Five
R0/B0/parallel-length/Te/density declarations are positive finite; five declared
velocity/profile/wall/duration/size errors are finite nonnegative with positive
finite inclusive bounds. Overflow refuses. No SOL evolution, blob velocity,
wall flux, detector reconstruction or physical campaign comparison is run.

`write_blob_transport_reference_report` writes sorted UTF-8 JSON, creates
parents and refuses direct/resolved/symlink/hardlink selected root/input aliases
before mutation; unrelated output is replaced. Discovery is sequential without
locks or concurrent pathname guarantees. API errors propagate. Argparse and
registered `validate-blob-transport-reference` return authored stderr and exit
one for supported inspection/write failures, including cyclic paths; parser
help/usage retain zero/two and declaration pass/findings return zero/one. The
cold script runs from an unrelated directory without `PYTHONPATH`.

```python
from tempfile import TemporaryDirectory
from validation.validate_blob_transport_reference import (
    canonical_artifact_sha256,
    validate_blob_transport_reference,
    write_blob_transport_reference_report,
)

assert canonical_artifact_sha256({"value": 1}) == canonical_artifact_sha256(
    {"value": 1, "payload_sha256": "ignored"}
)
with TemporaryDirectory() as directory:
    report = validate_blob_transport_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail"
    write_blob_transport_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

Blob transport validation claims require measured probe-campaign or documented
public reference artifacts for SOL filament velocity, spreading, wall-flux, and
event-domain checks:

```bash
scpn-control validate-blob-transport-reference --require-reference-artifacts --json-out
python validation/validate_blob_transport_reference.py --require-reference-artifacts --output-json artifacts/blob_transport_reference_report.json
```

Strict mode fails until `validation/reports/blob_transport_reference/` contains
artifacts using schema `scpn-control.blob-transport-reference.v1` with source
provenance, safe reference/profile/detector artifact URIs, SHA-256 hashes for
each artifact and the canonical payload, SOL unit contracts, strictly ordered
separatrix-to-wall coordinates, positive detector-time and blob-size domains,
positive magnetic-geometry metadata, and velocity/profile/wall-flux/event
metrics inside declared tolerances. Synthetic blob regressions remain
module-behaviour checks only.

The stored ELM reader selects immediate sorted JSON entries or one regular
file regardless of suffix. Optional absent/nonfile roots pass with zero entries;
required mode refuses. Invalid UTF-8/JSON, duplicate keys, unreadable entries
and nonzero decimals rounded to binary64 zero return authored findings without
exception or input-key text. Each entry is read once; this is no coherent snapshot.

Schema `scpn-control.elm-reference.v1` retains twelve nonblank text fields,
five exact 64-hex digest shapes and four artifact URI declarations. The four
reference-byte digests are format checks without fetching or authentication.
The public `canonical_artifact_sha256` retains compact sorted ASCII JSON,
excluding only `payload_sha256`. The body digest provides consistency supplied
by the author, without authenticating a reference or measurement.

Artifact URI policy remains lexical: nonblank text, no NUL, no absolute local
path and no parent path components; `http://`, `https://`, `doi:`, `s3://`,
`gs://` prefixes are admitted, including bare prefixes. Other relative text,
including relative-looking `file:///...`, remains admitted. No URI is fetched
or parsed as a resource. Measured metadata requires a nonblank machine and
shot or campaign ID; public metadata requires URL or DOI presence only.

Seven unit labels are retained. Pedestal rho has at least two finite
nonboolean coordinates strictly increasing in `[0,1]`. Event and RMP windows
each contain two finite nonboolean times, starting at zero or later and
increasing strictly; the two windows need no relation to each other. Energy
fraction endpoints retain `0.04 <= lower <= upper <= 0.15`, including equal
endpoints. Six finite nonnegative declared errors must not exceed positive
finite declared tolerances. Equality and representable subnormals remain
admitted; enormous scalar conversions become findings. No simulation runs.

The actual analytic ELM validation has no dependency on this persisted reader.
Reader pass alone does not establish measured comparison or physical admission.
`write_elm_reference_report` creates parents and writes sorted UTF-8 JSON,
refusing direct, resolved, symbolic and existing hardlink aliases of the supplied
root or selected immediate JSON entries before writing. Unrelated outputs may
replace. API operational errors propagate; Click/script use authored fixed stderr
with pass/findings 0/1, operational refusal 1 and parser help/usage 0/2.
Sequential path checks do not prevent concurrent pathname changes.

```python
from tempfile import TemporaryDirectory
from validation.validate_elm_reference import validate_elm_reference, write_elm_reference_report
with TemporaryDirectory() as directory:
    report = validate_elm_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    write_elm_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

ELM crash and RMP suppression validation claims require measured H-mode
campaign or documented public reference artifacts for ELM frequency, crash
energy fraction, pedestal profile drops, RMP suppression windows, and peak heat
flux:

```bash
scpn-control validate-elm-reference --require-reference-artifacts --json-out
python validation/validate_elm_reference.py --require-reference-artifacts --output-json artifacts/elm_reference_report.json
```

Strict mode fails until `validation/reports/elm_reference/` contains artifacts
using schema `scpn-control.elm-reference.v1` with declared source provenance,
lexical pre-crash/post-crash/event/RMP URI checks, format-only reference-byte
hashes and canonical body consistency, ELM/RMP unit labels, strictly ordered
pedestal rho grids, independently increasing nonnegative event/RMP windows,
inclusive Type-I energy-fraction endpoints, and declared error bounds.
Synthetic ELM-cycle regressions remain module-behaviour checks only; this
metadata inspection does not recompute a measured comparison.

EPED pedestal validation claims require measured pedestal-database or
documented public reference artifacts for pedestal height, pedestal width,
peeling-ballooning pressure limit, bootstrap-current coupling, collisionality
width ordering, and shaping input provenance:

```bash
scpn-control validate-eped-reference --require-reference-artifacts --json-out
python validation/validate_eped_reference.py --require-reference-artifacts --output-json artifacts/eped_reference_report.json
```

`validate_eped_reference` reads immediate sorted `*.json` entries of a directory,
or one regular file regardless of suffix. Missing roots produce no entries:
optional mode passes, required mode fails. Each entry is read once; this is a
sequential observation rather than a coherent filesystem snapshot. Malformed
JSON, duplicate keys, UTF8 and IO failures return authored `json` findings without
raw exception details. Nonzero JSON decimal tokens rounded to binary64 zero are
refused before checksum admission; large integers become numeric findings.

Schema `scpn-control.eped-reference.v1` admits
`measured_pedestal_database` with nonblank machine and shot/campaign identity,
or `documented_public_reference` with nonblank URL/DOI presence. Dataset and
time are strings without authentication. Four artifact SHA256 strings require
exact 64-character hexadecimal formatting. `payload_sha256` is compared
case-insensitively against SHA256 of the body excluding that field, serialised
with sorted keys, compact separators and ASCII escaping. The public
`canonical_artifact_sha256` retains this original serialisation. This binds the
supplied body without authenticating its producer or referenced artifact bytes.

The four URI fields retain EPED's lexical contract: trimmed nonblank strings
without NUL; `http://`, `https://`, `doi:`, `s3://` and `gs://` prefixes are
admitted, as are relative paths without literal parent segments. URI syntax,
resolution and referenced byte hashes are not checked. No artifact is
retrieved or executed. In particular, a prefix alone does not establish a
usable external address.

Units are width `psi_N`, pressure `Pa`, temperature `eV`, density `m^-3`,
current `A`, and dimensionless beta/shape. The rho grid has at least two finite
numbers strictly increasing in inclusive `[0, 1]`. Width endpoints satisfy
`0 < lower <= upper < 1`; beta endpoints satisfy `0 < lower < upper`.
Shaping requires positive kappa, major/minor radii, `abs(delta) < 1` and
`minor < major`. Five finite nonnegative declared errors must not exceed their
strictly positive declared bounds; equality is admitted. These comparisons do
not compute pedestal, peeling-ballooning or bootstrap physics. A passing
report checks declaration consistency and does not establish experimental
EPED validation. Synthetic EPED regressions remain module behaviour checks.

API, script and registered root CLI use `write_eped_reference_report` for
sorted UTF8 output, creating parents and replacing unrelated output while
refusing direct, resolved, symbolic and existing hard-link aliases of the root
or selected JSON inputs before writing. Alias discovery is a fresh observation;
concurrent pathname changes are outside this protection. Supported operational
failures become fixed CLI stderr with exit1; parser help/usage retain exits0/2.
Predicate ownership is split between `eped_reference_contracts` and
`eped_reference_geometry`, while the original validator and canonical hash
remain public.

```python
from tempfile import TemporaryDirectory
from validation.validate_eped_reference import validate_eped_reference

with TemporaryDirectory() as directory:
    report = validate_eped_reference(directory, require_reference_artifacts=True)
assert report["status"] == "fail"
assert report["errors"][0]["field"] == "artifact_root"
```

MARFE radiation-condensation and density-limit validation claims require
measured MARFE campaign or documented public reference artifacts for onset
temperature, density-limit, Greenwald fraction, front-temperature, and
radiative-growth checks:

```bash
scpn-control validate-marfe-reference --require-reference-artifacts --json-out
python validation/validate_marfe_reference.py --require-reference-artifacts --output-json artifacts/marfe_reference_report.json
```

`validate_marfe_reference` reads immediate sorted `*.json` entries of a
directory, or one regular file regardless of suffix. Missing/nonfile roots
select nothing: optional mode passes and required mode fails. Each entry is
read once; no coherent concurrent filesystem snapshot is claimed. Unsupported
JSON/UTF8, duplicate keys and IO errors become fixed authored `json` findings.
Nonzero decimal tokens rounded to binary64 zero are refused before body-hash
admission; huge integers produce numeric findings without conversion crashes.

Schema `scpn-control.marfe-reference.v1` admits `measured_marfe_campaign`
with nonblank machine and shot/campaign identity, or
`documented_public_reference` with nonblank URL/DOI presence. Dataset/time
strings and citation presence are unauthenticated. Four artifact SHA256 fields
require exactly 64 hexadecimal characters. The canonical `payload_sha256`
compares case-insensitively with SHA256 of the body excluding that field, using
original sorted compact ASCII-escaped JSON serialisation. The public hash
function retains this original format. Body consistency does not authenticate
the producer or referenced artifact bytes.

Four artifact URI fields retain the MARFE lexical contract: trimmed nonblank
strings without NUL; `http://`, `https://`, `doi:`, `s3://`, `gs://` prefixes,
or relative paths without literal parent segments. Prefix acceptance does not
validate URL syntax, retrieve a source or establish its referenced checksum.

Units are temperature `eV`, density `m^-3`, power `W`, current `A`,
dimensionless impurity fraction, length `m`, and growth rate `s^-1`.
Temperature and density scans have at least two finite, positive, strictly
increasing numbers, without an upper unit-grid bound. Impurity fraction obeys
`0 < lower <= upper <= 1`; equality and upper one remain allowed. Geometry
requires positive major/minor radii, q95 and connection length, with
`minor < major`. Power requires positive `P_SOL_W` and nonnegative
`q_perp_W_m2`; impurity is a nonblank string. Five finite nonnegative declared
errors must not exceed their positive declared bounds; equality passes. No
radiation-condensation, density limit, front temperature or growth metric is
computed. A passing declaration report does not establish experimental MARFE
validation. Synthetic regressions remain module behaviour checks.

API, script and registered root CLI use `write_marfe_reference_report` to
protect direct, resolved, symbolic and existing hard-link aliases of the root
or selected JSON inputs before writing sorted UTF8 output. Parents are created
and unrelated outputs replaced. Path discovery is a fresh sequential
observation; concurrent pathname changes are outside this protection.
Supported operational failures produce fixed CLI stderr with exit1; parser
help/usage retain exits0/2. Identity/hash and numeric predicates live in
`marfe_reference_contracts` and `marfe_reference_domains` while the original
validator and canonical hash remain public.

```python
from tempfile import TemporaryDirectory
from validation.validate_marfe_reference import validate_marfe_reference

with TemporaryDirectory() as directory:
    report = validate_marfe_reference(directory, require_reference_artifacts=True)
assert report["status"] == "fail"
assert report["errors"][0]["field"] == "artifact_root"
```

The repository-owned bounded MARFE algebra is validated separately against
exact closed forms:

```bash
python validation/validate_marfe_onset.py --report validation/reports/marfe_onset.json
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family marfe-onset-claims \
  --artifact report=validation/reports/marfe_onset_claims.json \
  --artifact markdown=validation/reports/marfe_onset_claims.md \
  -- python validation/benchmark_marfe_onset_claims.py
```

The local validator covers the Greenwald limit, Greenwald scaling, bounded
MARFE density-limit scaling, edge connection length, radiation-condensation
critical-density bracket, cooling-slope onset membership, density-power scan
boundary, and front-temperature detector thresholds. Its benchmark report is
local-regression evidence only and sets `production_claim_allowed=false`.

The neural-equilibrium, orbit, uncertainty, VMEC, EPED, MARFE and NTM reference
commands reject cyclic report-output paths with an authored operational failure
and exit 1 through both the actual script and registered root CLI. Their public
report writers propagate path-resolution errors; declaration consistency still
does not qualify referenced physics or model behaviour.

NTM island-dynamics validation claims require measured NTM campaign or
documented public reference artifacts for q-profile reconstruction,
rational-surface location, island-width growth and saturation, suppression
time, seed-island domain, and ECCD alignment:

```bash
scpn-control validate-ntm-reference --require-reference-artifacts --json-out
python validation/validate_ntm_reference.py --require-reference-artifacts --output-json artifacts/ntm_reference_report.json
```

`validate_ntm_reference` reads immediate sorted `*.json` directory entries or
one regular file regardless of suffix. Missing/nonfile roots select nothing:
optional mode passes and required mode fails. Each entry is read once, without
a coherent concurrent filesystem snapshot. Unsupported JSON/UTF8, duplicate
keys and IO errors become fixed authored `json` findings. Nonzero decimal tokens
rounded to binary64 zero are refused before checksum admission; huge numeric
conversions become field findings rather than interpreter errors.

Schema `scpn-control.ntm-reference.v1` admits `measured_ntm_campaign` with
nonblank machine and shot/campaign identity, or `documented_public_reference`
with nonblank URL/DOI presence. Dataset/time/citation identity is unauthenticated.
Four artifact SHA256 fields require exactly 64 hexadecimal characters. The
canonical `payload_sha256` compares case-insensitively against SHA256 of the
body excluding that field, using original sorted compact ASCII-escaped JSON.
The public hash function retains this serialisation; matching it establishes
body consistency without authenticating the producer or referenced bytes.

Four artifact URI fields retain trimmed nonblank/no-NUL lexical checks:
`http://`, `https://`, `doi:`, `s3://`, `gs://` prefixes, or relative paths
without literal parent segments. Prefix acceptance is not URL parsing,
resolution, retrieval or referenced checksum verification.

Units are island/deposition width `m`, time `s`, current `A`, dimensionless
q/rho, power `W`, and growth rate `m/s`. The rho grid has at least two finite,
nonnegative, strictly increasing numbers in inclusive `[0, 1]`; q values are
positive, finite and length-matched. q monotonicity/interpolation is not checked.
The declared rational surface requires positive nonboolean integer m/n,
positive q, surface radius and major/minor radii, finite signed shear,
`0 < rho < 1`, `minor < major` and `surface radius < minor`. Integer modes
retain their original uncapped domain; no `q = m/n` or rho-radius relationship
is computed. Positive seed width endpoints allow equality. ECCD power, current
and alignment error are nonnegative; deposition width is positive. Five finite
nonnegative declared errors compare against strictly positive declared bounds,
with equality admitted. A passing report does not establish measured island
evolution, suppression, ECCD effect or metric authenticity. Synthetic NTM
regressions remain module behaviour checks.

API, actual script and registered root CLI use `write_ntm_reference_report` to
protect direct, resolved, symbolic and existing hard-link aliases of the root
or selected JSON inputs before writing sorted UTF8 output. Parents are created,
unrelated outputs replaced and actual cyclic output paths produce fixed CLI
operational refusal. API path/IO/serialisation errors propagate; supported CLI
failures return exit1 and fixed stderr without exception details. Parser
help/usage retain exits0/2. Path discovery is sequential; concurrent changes are
outside this protection. Identity/hash and numeric predicates live in
`ntm_reference_contracts` and `ntm_reference_domains`, while the original
validator and canonical hash remain public.

```python
from tempfile import TemporaryDirectory
from validation.validate_ntm_reference import validate_ntm_reference

with TemporaryDirectory() as directory:
    report = validate_ntm_reference(directory, require_reference_artifacts=True)
assert report["status"] == "fail"
assert report["errors"][0]["field"] == "artifact_root"
```

Orbit-following validation claims require persisted published-reference or
real-campaign artifacts for banana-width, first-orbit-loss, and
passing/trapped/lost classification checks:

```bash
scpn-control validate-orbit-reference --require-reference-artifacts --json-out
python validation/validate_orbit_reference.py --require-reference-artifacts --output-json artifacts/orbit_reference_report.json
```

Strict mode fails until `validation/reports/orbit_reference/` contains artifacts
with source provenance, model identity, SHA-256 reference hash, case count,
orbit/loss/energy/field units, and declared error or classification metrics
inside tolerance. Real-campaign artifact URIs must use an admitted scheme
(`file`, `https`, `s3`, or `gs`); local `file://` URIs must stay under
`/validation/reports/` or `/validation/reference_data/`.

The orbit gate checks declarations. It neither opens referenced campaign bytes
nor recomputes banana-width, loss or classification metrics. A SHA-256 field
declares exactly 64 hexadecimal characters; it is not compared with referenced
bytes. Public URL/DOI presence, source labels, execution time and model identity
are unauthenticated metadata. Passing does not establish measured fast-ion
agreement or qualify a facility claim.

The reader selects sorted immediate JSON files from a directory, or one regular
file regardless of suffix. Optional empty/missing roots pass with zero entries;
required mode fails. JSON, UTF-8, duplicate-key and IO failures become authored
per-file findings. Malformed source types and binary64-unrepresentable numeric
values fail their fields. A nonzero JSON decimal token that decodes to zero is
refused as a JSON finding; exact zeros and representable subnormals remain
valid. Both error metrics must be nonnegative and within
strictly positive declared bounds; classification score and minimum are in
inclusive `[0, 1]`, with equality passing. Units remain exactly `m`, `1`, `keV`
and `T`. Extra declaration fields are permitted.

`write_orbit_reference_report` is shared by the standalone script and registered
command. It creates output parents but rejects direct, resolved, symbolic and
existing hard-link aliases of the supplied root or its current selected JSON
inputs. Other existing output is replaced. Supported inspection/write failures
return exit 1 with fixed authored stderr; parser help/usage retain exits 0/2.
Inspection and writing are sequential observations, without locking or a
transaction protecting concurrent path replacement. The script also runs from
an unrelated working directory; explicit paths remain caller-relative.

Uncertainty quantification claims require persisted published-reference or
campaign artifacts for the full propagation chain:

```bash
scpn-control validate-uncertainty-reference --require-reference-artifacts --json-out
python validation/validate_uncertainty_reference.py --require-reference-artifacts --output-json artifacts/uncertainty_reference_report.json
```

The validator checks persisted declarations, including source labels, nonblank
identity strings, exact 64-hex SHA-256 spelling, positive case counts,
`tau_E=s`, `P_fusion=MW`, `Q=1`, and `sigma=same_as_quantity` units. Three
nonnegative relative errors must not exceed their declared positive bounds;
percentile monotonicity and its declared minimum must be in inclusive [0, 1].
Equality passes. Booleans, nonfinite values and integers too large for binary64
are rejected; nonzero JSON decimals rounded to zero are refused during decoding.

A passing declaration does not authenticate the model, execution time,
referenced checksum, citation or metric computation. Public references require
a nonblank URL or DOI; real UQ campaigns retain the nonblank-string requirement
for `campaign_artifact_uri`. This command does not parse that URI, fetch or hash
referenced bytes, or propagate physical uncertainty. Strict mode fails when no
selected reference declaration is accepted.

Directories select immediate sorted `*.json` entries; regular files are read
regardless of suffix. Missing roots pass with zero entries in optional mode.
Malformed UTF-8/JSON, duplicate keys and selected-file IO failures return fixed
JSON findings. The public `write_uncertainty_reference_report` is shared by the
standalone script and registered CLI. It refuses direct, resolved, symbolic and
existing hard-link aliases of selected inputs before writing a report. Other
outputs are replaced, with missing parents created; no locking or coherent
concurrent directory snapshot is promised. Operational inspection/write failure
uses authored stderr and exit 1; declaration pass/failure use exits 0/1, and
standalone parser help/usage retain 0/2. The standalone script also works from an
unrelated working directory without inherited `PYTHONPATH`.

```python
from tempfile import TemporaryDirectory
from validation.validate_uncertainty_reference import validate_uncertainty_reference

with TemporaryDirectory() as directory:
    report = validate_uncertainty_reference(directory, require_reference_artifacts=True)
assert report["status"] == "fail" and report["reference_artifacts"] == 0
```

VMEC-lite stellarator-equilibrium validation claims require persisted
published-reference or real-VMEC-run artifacts for surface geometry,
rotational-transform, Fourier truncation, and force-residual checks:

```bash
scpn-control validate-vmec-reference --require-reference-artifacts --json-out
python validation/validate_vmec_reference.py --require-reference-artifacts --output-json artifacts/vmec_reference_report.json
```

This gate checks persisted declarations, including schema 1.0, admitted source
labels, nonblank model/version/dataset/time strings, exact 64-hex SHA-256 spelling,
positive case counts and exact Fourier/pressure/iota unit labels. Fourier
truncation requires positive integer `m_pol` and `n_fp` and nonnegative integer
`n_tor`, excluding booleans. The four declared surface R/Z, iota and force errors
must be finite and nonnegative and must not exceed their positive declared
bounds; equality passes. Oversized integers, nonfinite values and wrong numeric
types yield field findings; nonzero JSON decimal tokens rounded to zero are
refused. No coefficient arrays or equilibrium are evaluated by this gate.

Public references require a nonblank URL or DOI declaration. Real-VMEC artifact
URIs retain the shared lexical policy: schemes `file`, `https`, `s3`, or `gs`,
with local `file://` paths under `/validation/reports/` or
`/validation/reference_data/`. URI policy does not resolve symlinks, authenticate
remote content or download/hash the referenced file. Declared SHA-256, model,
time, source, citation and error values are not verified against a real VMEC
execution; declaration pass does not establish stellarator equilibrium,
convergence, force balance or calibrated facility evidence.

Immediate sorted `*.json` directory entries are selected; a regular file is read
regardless of suffix. Optional missing roots pass with zero entries; required
empty roots fail. Malformed UTF-8/JSON, duplicate keys and selected input IO
failures return authored findings. Standalone script and registered CLI share
`write_vmec_reference_report`, protecting direct, resolved, symbolic and
existing hard-link input aliases before creating parents/writing sorted UTF-8
reports. Other output is replaced. Discovery/write observations are sequential,
without concurrent locking or transaction guarantees. Operational failure uses
fixed stderr and exit 1; declaration pass/failure use 0/1, standalone parser
help/usage use 0/2. The actual standalone script works from an unrelated cwd
without inherited `PYTHONPATH`.

```python
from tempfile import TemporaryDirectory
from validation.validate_vmec_reference import validate_vmec_reference

with TemporaryDirectory() as directory:
    report = validate_vmec_reference(directory, require_reference_artifacts=True)
assert report["status"] == "fail" and report["reference_artifacts"] == 0
```

The repository-owned bounded VMEC-lite spectral geometry is validated
separately against exact local forms:

```bash
python validation/validate_vmec_lite_geometry.py --report validation/reports/vmec_lite_geometry.json
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family vmec-lite-claims \
  --artifact report=validation/reports/vmec_lite_claims.json \
  --artifact markdown=validation/reports/vmec_lite_claims.md \
  -- python validation/benchmark_vmec_lite_claims.py
```

The local validator covers spectral mode count, direct Fourier-basis
evaluation, axisymmetric boundary coefficients, fixed-boundary radial scaling,
q/iota reciprocity, B-coefficient construction, and positive sampled major
radius. Its benchmark report binds the claim-admission evidence to the sealed
geometry-validation payload and keeps `production_claim_allowed=false`.

The RZIP reference reader selects sorted immediate `*.json` directory entries
or a regular file regardless of suffix. Optional missing/nonfile roots pass
with zero entries; required mode refuses. Unreadable files, invalid UTF-8,
malformed JSON, duplicate keys and nonzero decimal tokens rounded to binary64
zero produce authored findings. Each read is a sequential observation rather
than a coherent snapshot.

Schema `1.0` requires nonblank identity strings, an exact 64-character hex
reference SHA declaration, original units (vertical displacement `m`, growth
rate `s^-1`, growth time `ms`, coil current `A`, time `s`) and six positive
finite physical parameters: major/minor radius, elongation, plasma current,
toroidal field and wall time constant. `vertical_field_index` is finite and
signed: negative, zero and positive values remain admitted. No radius ordering
or minimum elongation is imposed. Case count is a positive nonboolean integer
without an artificial cap. Growth-rate, displacement and pole errors must be
finite and nonnegative with positive finite bounds; equality passes. Huge
unrepresentable numbers produce findings.

Public references require nonblank URL or DOI presence. Measured discharges
require shot and diagnostic URI presence; these do not use the external
artifact URI policy. External benchmarks require a named `CREATE-L`,
`CREATE-NL`, `TSC` or `reference_rzip` code and the existing shared lexical URI
policy: hostless `file` paths under the validation report/reference prefixes,
or `https`, `s3`, `gs` addresses with authority and path. ASCII controls,
malformed URI syntax and literal parent components refuse. Percent escapes,
queries, fragments and credential text are not decoded or authenticated.
No referenced bytes are fetched, hashed or executed. SHA syntax, identities,
declared errors and a passing report authenticate no RZIP dynamics or facility
measurement, and do not recompute growth rates, vertical motion or pole values.

`write_rzip_reference_report` persists sorted UTF-8 JSON, creates parents and
refuses selected root/input aliases (direct, resolved, symlink or hard link)
before mutation. Unrelated output is replaced. Discovery is a fresh sequential
observation without locking or concurrent pathname guarantees. API operational
errors propagate; argparse and registered `validate-rzip-reference` return
fixed authored stderr and exit one for supported inspection/write failures,
including cyclic paths. Parser help/usage retain zero/two; declaration
pass/findings return zero/one. The script also runs from an unrelated working
directory without `PYTHONPATH`.

```python
from tempfile import TemporaryDirectory
from validation.validate_rzip_reference import (
    validate_rzip_reference,
    write_rzip_reference_report,
)

with TemporaryDirectory() as directory:
    report = validate_rzip_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail"
    write_rzip_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

RZIP vertical-stability validation claims require persisted public-reference,
external-code, or measured-discharge artifacts for vertical growth rates,
vertical displacement, and closed-loop pole checks:

```bash
scpn-control validate-rzip-reference --require-reference-artifacts --json-out
python validation/validate_rzip_reference.py --require-reference-artifacts --output-json artifacts/rzip_reference_report.json
```

Strict mode fails until `validation/reports/rzip_reference/` contains artifacts
with source provenance, model identity, SHA-256 reference hash, RZIP physical
parameters, unit contracts, case count, and vertical-stability metrics inside
declared tolerances. External-code artifact URIs must use an admitted scheme
(`file`, `https`, `s3`, or `gs`); local `file://` URIs must stay under
`/validation/reports/` or `/validation/reference_data/`.

Density-control and particle-source validation claims require persisted
public-reference, measured-fuelling, or external integrated-modelling artifacts
for Greenwald fraction, pellet deposition, recycling, and density-profile
checks:

```bash
scpn-control validate-density-reference --require-reference-artifacts --json-out
python validation/validate_density_reference.py --require-reference-artifacts --output-json artifacts/density_reference_report.json
```

Strict mode fails until `validation/reports/density_reference/` contains
accepted declarations of source, model identity, a 64-character hexadecimal
reference digest, radial-grid metadata, actuator settings, exact unit labels,
positive case count and four finite nonnegative errors within positive declared
tolerances. External integrated-modelling declarations also need an allowed code
label and admitted artifact URI syntax.

The reader checks local metadata and caller-supplied metrics. It does not fetch
references, authenticate provenance, recompute digests or density metrics, or
run a model. A passing declaration is not measured-campaign, external-code or
facility admission. Optional inspection can pass with zero declarations;
`--require-reference-artifacts` makes absence fail. Duplicate keys and nonfinite
floating-point JSON tokens are rejected at every depth, including unused
metadata; unsupported source/code containers and unrepresentable required
numbers produce findings. The native [density-reference contract](api.md#density-reference-declarations)
describes scope, outputs, errors and the direct/registered command surfaces.

The burn-control reference reader inspects original schema `1.0` metadata.
Directories select sorted immediate `*.json` entries; a regular file is read
regardless of suffix. Optional absent/nonfile roots pass with zero entries;
required mode fails. Invalid UTF-8, JSON, duplicate keys, IO and nonzero decimal
fraction tokens rounded to binary64 zero produce authored findings. Inspection
is sequential and establishes no concurrent snapshot.

The six source/model/version/dataset/reference-SHA/time fields must be nonblank
strings. The reference SHA requires exactly 64 hex characters, with no payload
body digest or reference-byte authentication. Public URL/DOI, measured replay
shot/diagnostic URI and integrated-transport TRANSP/TSC/JINTRAC/ASTRA plus
artifact URI are presence declarations; URI strings are not parsed, resolved
or downloaded. Relative parent segments and nonblank NUL-containing values
retain the original presence-only domain.

Units remain density `m^-3`, temperature `keV`, power `MW`, time `s`, reactivity
`m^3/s`, triple product `m^-3 s keV` and dimensionless `1`. Plasma metadata
requires positive finite `major_radius_m`, `minor_radius_m`, `elongation`,
`tau_E_s` and `P_aux_MW`; no geometric ordering or elongation minimum is inferred.
Case count is a positive nonboolean integer without an artificial cap. The
five declared alpha-power, Q, Lawson margin, burn-fraction and reactivity-exponent
errors are nonnegative finite numbers with positive finite bounds; equality
passes. Huge unrepresentable numbers refuse. No burn evolution, Lawson,
reactivity, Q or alpha-heating calculation is performed or authenticated.

`write_burn_reference_report` persists sorted UTF-8 JSON with parent creation,
refusing selected root/input aliases (direct, resolved, symlink or hard link)
before mutation. Unrelated output is replaced; discovery is a fresh sequential
observation. API operational errors propagate, while argparse and registered
`validate-burn-reference` use fixed authored stderr and exit one for supported
inspection/write failures, including cyclic output paths. Parser help/usage
retain zero/two; declaration pass/findings return zero/one. Diagnostic saved
reports do not establish a healthy burn or facility claim.

```python
from tempfile import TemporaryDirectory
from validation.validate_burn_reference import (
    validate_burn_reference,
    write_burn_reference_report,
)

with TemporaryDirectory() as directory:
    report = validate_burn_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail"
    write_burn_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

DT burn-control validation claims require persisted documented public,
integrated transport benchmark, or measured burn replay artifacts for alpha
power, Q, Lawson margin, burn fraction, and reactivity-exponent checks:

Bounded local burn-control claim evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family burn-control-claims \
  --artifact report=validation/reports/burn_control_claims.json \
  --artifact markdown=validation/reports/burn_control_claims.md \
  -- python validation/benchmark_burn_control_claims.py
```

This writes `validation/reports/burn_control_claims.json` and
`validation/reports/burn_control_claims.md`. These artefacts demonstrate
deterministic burn-control claim-admission plumbing only; reactor-control
claims remain gated by the strict reference-artifact validator below.

```bash
scpn-control validate-burn-reference --require-reference-artifacts --json-out
python validation/validate_burn_reference.py --require-reference-artifacts --output-json artifacts/burn_reference_report.json
```

Strict mode fails until `validation/reports/burn_reference/` contains artifacts
with source provenance, model identity, SHA-256 reference hash, plasma metadata,
unit contracts, case count, and burn-control metrics inside declared
tolerances.

The volt-second reference reader selects sorted immediate `*.json` directory
entries or a regular file regardless of suffix. An optional missing/nonfile
root passes with zero entries; required mode fails. Schema `1.0` requires
nonblank model/version/dataset/time and reference SHA strings, exact 64 hex
characters, the declared unit labels and five positive finite machine values
(`Phi_CS_Vs`, `L_plasma_H`, `R_plasma_Ohm`, `Ip_MA`, `R0_m`). Case counts are
positive nonboolean integers without an artificial upper limit. Five declared
errors must be finite and nonnegative, with positive finite bounds; equality
passes. Huge unrepresentable numeric values return findings. Nonzero JSON
fraction tokens rounded to binary64 zero, duplicate keys, unreadable files,
malformed JSON and invalid UTF-8 produce fixed authored JSON findings.

Public references require URL or DOI presence; measured replay requires shot
and diagnostic URI presence; external scenarios require a named TRANSP, TSC,
ASTRA, JINTRAC or PROCESS string plus artifact URI presence. These strings are
not parsed, downloaded or resolved: a relative `../presence-only` declaration
is admitted. The SHA is format-only, with no required payload digest or
reference-byte authentication. No MA conversion, flux-budget calculation,
flat-top duration, Ejima term, bootstrap or margin comparison is recomputed.

`write_volt_second_reference_report` writes sorted UTF-8 JSON and creates
parents, refusing direct, resolved, symbolic or hard-link aliases of the root
or selected inputs before mutation. Unrelated output is replaced. Inspection
and output discovery are sequential observations, without a concurrent
snapshot guarantee. API path/IO/encoding/serialisation failures propagate;
argparse and registered `validate-volt-second-reference` translate supported
operational failures, including cyclic output paths, to fixed stderr and exit
one. Parser help/usage retain exits zero/two, and declaration pass/findings
return zero/one. A saved report is diagnostic output rather than physical
evidence or an admitted reference artifact.

```python
from tempfile import TemporaryDirectory
from validation.validate_volt_second_reference import (
    validate_volt_second_reference,
    write_volt_second_reference_report,
)

with TemporaryDirectory() as directory:
    report = validate_volt_second_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail"
    write_volt_second_reference_report(
        report, directory + "/report.txt", artifact_root=directory
    )
```

Volt-second scenario validation claims require persisted documented public,
measured loop-voltage replay, or external scenario benchmark artifacts for total
flux, flat-top duration, Ejima flux, bootstrap current, and budget-margin
checks:

Bounded local volt-second claim evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family volt-second-claims \
  --artifact report=validation/reports/volt_second_claims.json \
  --artifact markdown=validation/reports/volt_second_claims.md \
  -- python validation/benchmark_volt_second_claims.py
```

This writes `validation/reports/volt_second_claims.json` and
`validation/reports/volt_second_claims.md`. These artefacts demonstrate
deterministic scenario-accounting claim-admission plumbing only; pulse-duration
or solenoid-commissioning claims remain blocked. The validator below checks
reference metadata and declared error limits; it does not hash independent
source bytes or recompute the comparison metrics.

```bash
scpn-control validate-volt-second-reference --require-reference-artifacts --json-out
python validation/validate_volt_second_reference.py --require-reference-artifacts --output-json artifacts/volt_second_reference_report.json
```

Strict mode fails until `validation/reports/volt_second_reference/` contains
artifacts with source provenance, model identity, SHA-256-shaped reference
hash, machine metadata, unit contracts, case count, and declared volt-second
metrics inside tolerances. Even a passing structural report cannot enable the
public facility claim without byte-bound reference evidence and independently
recomputed comparisons.

Auxiliary current-drive validation claims require persisted documented public,
ray-tracing, Fokker-Planck, or measured-deposition artifacts for absorbed power,
driven current, deposition centroid, peak current density, and NBI slowing-down
checks:

Bounded local current-drive claim evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family current-drive-claims \
  --artifact report=validation/reports/current_drive_claims.json \
  --artifact markdown=validation/reports/current_drive_claims.md \
  -- python validation/benchmark_current_drive_claims.py
```

This writes `validation/reports/current_drive_claims.json` and
`validation/reports/current_drive_claims.md`. These artefacts demonstrate
deterministic current-drive claim-admission plumbing only; ray-traced,
Fokker-Planck, or measured-deposition claims remain gated by the strict
reference-artifact validator below.

```bash
scpn-control validate-current-drive-reference --require-reference-artifacts --json-out
python validation/validate_current_drive_reference.py --require-reference-artifacts --output-json artifacts/current_drive_reference_report.json
```

The current default reference directory is empty. Optional mode passes with zero
entries; `--require-reference-artifacts` makes absence fail. The persisted
`current_drive_claims.json` bounded report is refused by this reference schema.
Directories contribute sorted immediate `*.json` paths; explicit files are read
regardless of suffix, and relative paths use the caller's working directory.

This validator checks declarations: version `1.0`, nonblank identity/provenance,
64 hexadecimal digest characters, exact unit labels, positive source metadata,
a positive integer case count, and five finite non-negative errors no greater
than their positive declared tolerances. Its radial metadata contract is
`0 < rho_min < rho_max <= 1`; `rho_points` is positive but need not be integral.
JSON must be valid UTF-8 with unique keys and finite floating-point values,
including fields outside the selected metrics. Invalid source/code containers,
unrepresentable numeric scalars and expected read failures produce report errors.

A passing report does not fetch references, verify digest bytes, authenticate
provenance, parse dates, recompute comparisons or admit an external/facility claim.
Accepted declarations remain diagnostic entries when another file fails the
inspection. The report writer creates parents and directly replaces its output;
filesystem refusals use fixed text without traceback. See the
[current-drive reference API contract](api.md#validation-current-drive-reference).

The stored static-mu reader selects sorted immediate JSON directory entries or
one regular file of any suffix. Optional absent/nonfile roots pass zero entries;
required mode refuses. Unsupported UTF-8/JSON, duplicate keys, unreadable entries
and nonzero decimal tokens rounded to binary64 zero produce fixed authored
findings without interpreter or input-key text. Each file is read once; this is
a sequential inspection, without a coherent directory snapshot.

Schema `1.0` retains six nonblank identities and one exact 64-hex reference-hash
shape, without fetching or authenticating bytes. Public references need URL/DOI
presence; measured control replay needs nonblank `shot_id` and `diagnostic_uri`;
external references need one of `MATLAB_MU_TOOLBOX`, `ROBUST_CONTROL_TOOLBOX`,
`SLICOT`, `JULIA_ROBUSTANDOPTIMALCONTROL` and nonblank artifact URI. Citation and
URI policy remains presence-only, without parsing or dereferencing.

Plant `state_dimension`, `control_dimension`, `output_dimension` and
`uncertainty_total_size`, plus case count, retain positive uncapped nonboolean
integers. No matrix allocation uses these metadata counts. Five original unit
labels remain; historical `robustness_margin` denotes the reciprocal STATIC
zero-frequency upper bound. It does not certify frequency-dependent stability.
Five declared errors are finite/nonnegative and must not exceed positive finite
bounds, with equality and representable subnormal bounds admitted. Huge scalar
conversion failures become field findings. This reader designs no controller,
runs no D-K iteration or frequency sweep and grants no validated claim.

The actual static claim builder/loader has no dependency on this persisted
reader. Its public admission assertion and required loader reject the existing
bounded claim; declared metrics do not open independent reference admission.

`write_static_mu_analysis_reference_report` creates parents, writes sorted UTF-8
JSON and refuses direct/resolved/symbolic/existing hardlink aliases of supplied
root or selected inputs before writing. Other outputs may replace. API IO/path/
serialisation failures propagate; script and both registered Click routes use
fixed authored operational stderr, pass/findings 0/1 and parser help/usage 0/2.
Sequential alias checks do not guard concurrent pathname changes.

The deprecated Python reader preserves `DeprecationWarning`; its script emits
the original authored notice and delegates. Both actual scripts bootstrap the
source checkout without editable installation. The hidden Click compatibility
route retains its legacy default directory; the canonical route retains its
static directory. All delegate to the same protected inspection/persistence.

```python
from tempfile import TemporaryDirectory
from validation.validate_static_mu_analysis_reference import (
    validate_static_mu_analysis_reference, write_static_mu_analysis_reference_report,
)
with TemporaryDirectory() as directory:
    report = validate_static_mu_analysis_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    write_static_mu_analysis_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

Static mu-analysis validation claims require persisted documented public,
external mu-toolbox, or measured control replay artifacts for mu upper bound,
reciprocal static upper bound, controller gain, D-scaling, and closed-loop spectral
abscissa checks:

Bounded local static mu-analysis evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family static-mu-analysis-claims \
  --artifact report=validation/reports/static_mu_analysis_claims.json \
  --artifact markdown=validation/reports/static_mu_analysis_claims.md \
  -- python validation/benchmark_static_mu_analysis_claims.py
```

This writes `validation/reports/static_mu_analysis_claims.json` and
`validation/reports/static_mu_analysis_claims.md`. These artefacts demonstrate
deterministic static mu-analysis claim-admission plumbing only. The persisted
JSON carries a canonical payload SHA-256 digest and
`load_static_mu_analysis_claim_evidence()`
rejects duplicate keys, schema drift, edited metric fields, and bounded
evidence presented as a validated robust-control claim. Full frequency-dependent
D-K synthesis and frequency-dependent stability are outside this static reader
and require independent scientific reference and comparison contracts.

The claim builder and loader reject validated admission from caller-supplied
reference dictionaries and self-reported error metrics, even when their
payload digest is intact. The reference-artefact validator checks artifact
metadata and declared metric bounds; it does not independently recompute the
comparison or enable validated claim admission. That path requires a verified
reference producer and comparison contract before it can be opened.

```bash
scpn-control validate-static-mu-analysis-reference --require-reference-artifacts --json-out
python validation/validate_static_mu_analysis_reference.py --require-reference-artifacts --output-json artifacts/static_mu_analysis_reference_report.json
```

Strict mode fails until `validation/reports/static_mu_analysis_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash, plant
metadata, unit contracts, case count, and mu-analysis metrics inside declared
tolerances.

The former command, validator script, and Python module remain deprecated
compatibility entrypoints through version 0.24.x; they forward to this static
analysis contract and do not execute D-K iteration.

The disruption reference reader selects sorted immediate `*.json` directory
entries or a regular file regardless of suffix. Optional absent/nonfile roots
pass with zero entries; required mode refuses. Unsupported UTF-8/JSON,
duplicate keys, unreadable files and nonzero decimal tokens rounded to binary64
zero produce authored findings. Reads are sequential observations.

Original schema `1.0` requires six nonblank identity strings and an exact
64-character hex reference SHA declaration. This is format-only and hashes no
referenced bytes or payload. Documented public references require URL or DOI
presence; measured disruption campaigns require shot and diagnostic URI
presence; external benchmarks require a named `JOREK`, `M3D-C1`, `NIMROD` or
`TSC` code and nonblank artifact URI presence. URI text is not parsed, fetched
or authenticated, preserving original relative/NUL presence semantics.

Signal `sample_count` is a nonboolean integer at least eight, without a cap;
sample period, pre-disruption duration, current-quench duration and thermal-
quench duration are positive finite values. No duration relation is imposed.
Neon/argon/xenon/total impurity inventories, mitigation strength and reference
TBR are nonnegative finite values. Strength is at most one, including both
zero and one. No inventory sum or positive-TBR rule is imposed. Units are
`s`, `ms`, `MA`, `MJ`, `mol` and dimensionless risk/TBR `1`, without conversion.
Reference case count is a positive nonboolean integer without a cap. Five
declared risk, lead-time, halo-current, runaway-beam and TBR errors must be
finite nonnegative numbers with positive finite bounds; equality passes.
Overflowing numbers refuse. Passing declarations authenticate no mitigation,
physical campaign, producer identity, quench dynamics or TBR calculation.

`write_disruption_reference_report` persists sorted UTF-8 JSON, creates parents
and refuses selected root/input aliases (direct, resolved, symlink or hardlink)
before mutation; unrelated output is replaced. Discovery is a fresh sequential
observation without locks or concurrent pathname guarantees. API failures
propagate. Argparse and registered `validate-disruption-reference` return fixed
authored stderr and exit one for supported inspection/write failures, including
cyclic output paths; parser help/usage retain zero/two and declaration
pass/findings return zero/one. Actual cold script execution from an unrelated
working directory without `PYTHONPATH` is supported.

```python
from tempfile import TemporaryDirectory
from validation.validate_disruption_reference import (
    validate_disruption_reference,
    write_disruption_reference_report,
)

with TemporaryDirectory() as directory:
    report = validate_disruption_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail"
    write_disruption_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

Disruption-mitigation contract validation claims require persisted
public-reference, measured-disruption, or external benchmark artifacts for
warning lead time, mitigation outcome, halo current, runaway beam, and TBR
equivalence checks:

Bounded local disruption-mitigation claim evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family disruption-mitigation-claims \
  --artifact report=validation/reports/disruption_mitigation_claims.json \
  --artifact markdown=validation/reports/disruption_mitigation_claims.md \
  -- python validation/benchmark_disruption_mitigation_claims.py
```

This writes `validation/reports/disruption_mitigation_claims.json` and
`validation/reports/disruption_mitigation_claims.md`. These artefacts
demonstrate deterministic halo/runaway ensemble and claim-admission plumbing
only. The claim builder, assertion and saver reject promoted mitigation claims
from caller-supplied reference metadata. The reference-artifact validator below
checks provenance fields and declared tolerances; it does not independently
recompute the ensemble comparison or establish mitigation admission.

```bash
scpn-control validate-disruption-reference --require-reference-artifacts --json-out
python validation/validate_disruption_reference.py --require-reference-artifacts --output-json artifacts/disruption_reference_report.json
```

Strict mode fails until `validation/reports/disruption_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash,
disruption-window timing, mitigation-cocktail metadata, unit contracts, case
count, and disruption-mitigation metrics inside declared tolerances. An
independent source-data and comparison contract remains required before a
validated mitigation claim can be admitted.

Bounded local differentiable-transport gradient evidence can be regenerated
with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family differentiable-transport-latency \
  --artifact parameter-report=validation/reports/differentiable_transport_latency.json \
  --artifact parameter-markdown=validation/reports/differentiable_transport_latency.md \
  --artifact rollout-report=validation/reports/differentiable_transport_rollout_latency.json \
  --artifact rollout-markdown=validation/reports/differentiable_transport_rollout_latency.md \
  --artifact readiness-report=validation/reports/differentiable_transport_full_fidelity_readiness.json \
  --artifact readiness-markdown=validation/reports/differentiable_transport_full_fidelity_readiness.md \
  -- python validation/benchmark_differentiable_transport_latency.py
python validation/validate_differentiable_transport_latency.py --require-admitted --json-out
```

This writes `validation/reports/differentiable_transport_latency.json`,
`validation/reports/differentiable_transport_latency.md`,
`validation/reports/differentiable_transport_rollout_latency.json`, and
`validation/reports/differentiable_transport_rollout_latency.md`, plus
`validation/reports/differentiable_transport_full_fidelity_readiness.json` and
`validation/reports/differentiable_transport_full_fidelity_readiness.md`. The
reports exercise the audited JAX gradient-admission path when JAX is available and
otherwise publish a blocked-backend status. Persisted evidence fails closed on
non-finite or negative audit losses/errors, tolerance drift from campaign
metadata, duplicate or out-of-domain sampled audit indices, inconsistent
pass/fail flags, malformed latency run counts, and unordered latency
percentiles. The same admission path now binds runtime provenance for CPU/GPU
comparison campaigns, including Python version, operating platform, machine
class, JAX and jaxlib versions, default backend, visible JAX devices, and x64
state. The standalone validator checks declared fields, sampled indices and internal
status/count consistency. It does not replay audits, authenticate runtime or
bind the readiness digest declarations to files; its pass alone does not grant
release or full-fidelity promotion. The rollout source-gradient loss remains inside the traced JAX
graph, and the module enables JAX x64 before importing `jax.numpy` so benchmark
dtype evidence matches the requested differentiable transport precision.
Full-fidelity differentiable-transport claims additionally require the
`transport_full_fidelity_readiness_evidence()` promotion gate to bind campaign
metadata, one-step and rollout latency reports, audit digests, controller
formal-proof evidence, equilibrium coupling, and an independently admitted
external reference artefact. When `validation/reports/scpn_z3_formal.json`
contains a passing bounded Z3 Petri-net formal report, the benchmark binds that
report's canonical payload SHA-256 into differentiable-transport readiness.
Missing external reference evidence leaves the
claim explicitly blocked rather than promoted from local differentiability
evidence.

Bounded coupled differentiable-scenario evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family differentiable-scenario \
  --artifact report=validation/reports/differentiable_scenario_readiness.json \
  --artifact markdown=validation/reports/differentiable_scenario_readiness.md \
  -- python validation/benchmark_differentiable_scenario.py
python validation/validate_differentiable_scenario.py --json-out
```

This writes `validation/reports/differentiable_scenario_readiness.json` and
`validation/reports/differentiable_scenario_readiness.md`. The report binds the
analytic Solov'ev-form equilibrium parameters, R/Z flux grid, four-channel
transport rollout, sampled finite-difference gradient audit, campaign digest,
and local non-isolated timing context. The validator admits the persisted
evidence only as bounded local scenario-gradient evidence and requires
`claim_admissible=false` until physics traceability is satisfied by external
equilibrium or integrated-modelling evidence.

The TORAX code-to-code transport benchmark publishes its own strict
external-reference evidence boundary:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family torax-code-to-code \
  --artifact report=validation/reports/code_to_code_benchmark.json \
  --artifact markdown=validation/reports/code_to_code_benchmark.md \
  -- python validation/code_to_code_benchmark.py --with-torax

PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family torax-code-to-code-required \
  --artifact report=validation/reports/code_to_code_benchmark.json \
  --artifact markdown=validation/reports/code_to_code_benchmark.md \
  -- python validation/code_to_code_benchmark.py --with-torax --require-external
```

The generated JSON and Markdown use schema
`scpn-control.code-to-code-benchmark.v3`. Digests bind normalised declared
scenario/report bytes; they authenticate no provider. The initial temperature/
density profiles, radial count and fixed dt now share explicit mappings.
Comparison uses actual rho coordinates and refuses malformed vectors, missing
coordinate coverage and nonrepresentable arithmetic, including equal-length
coordinate mismatches that formerly compared by index.

The configured gyro-Bohm versus constant transport, current/geometry and
source-channel differences remain explicit physical-admission blockers.
diagnostic_comparison_available denotes valid declared arithmetic, not a
trusted external run. Missing/not-requested TORAX remains blocked/not_requested;
a successful local diagnostic writes both reports before optional required-
external exit one. Genuine tests execute the real local API/script/imported
main, configuration export, output alias and custody refusals. Authored reader
inputs and the retained legacy mocked extraction test are not TORAX evidence.
Installed-provider execution and version/API compatibility remain a separate
unfulfilled runtime qualification wherever TORAX is absent.

The digital-twin reference reader selects sorted immediate `*.json` directory
entries or a regular file regardless of suffix. Optional absent/nonfile roots
pass with zero entries; required mode refuses. Invalid UTF-8/JSON, duplicate
keys, unreadable files and nonzero decimal tokens rounded to binary64 zero
produce authored findings. Reads are sequential observations.

Original schema `1.0` requires six nonblank identity strings and an exact
64-character hex reference SHA declaration. This is format-only: referenced
bytes and the payload are not hashed or authenticated. Public provenance needs
URL or DOI presence; measured discharge replay needs shot and diagnostic URI
presence; external integrated modelling needs a named `ASTRA`, `IMAS`,
`JINTRAC`, `TRANSP` or `TSC` code and nonblank artifact URI presence. URI text is
not parsed, fetched or authenticated; original relative/NUL presence semantics
remain admitted.

Grid size is a nonboolean integer at least four, time steps a positive integer
and seed a nonnegative integer, all without an upper cap. State variables form
a nonempty list of nonblank strings; unknown and duplicate names are admitted.
The IDS export flag must be boolean and admits both true and false. Actuator
lag is an uncapped nonboolean nonnegative integer; bias is signed finite, rate
and sensor noise are finite nonnegative, and dropout includes both zero and
one. There is no calibration, topology catalog or IDS export authenticity
check. Units are `keV`, `m^-3`, dimensionless q/action/IDS `1` and time `step`,
without conversion. Case count is an uncapped positive nonboolean integer.
Five declared temperature, q-profile, lag, IDS and island errors are finite
nonnegative values with positive finite bounds; equality passes. Overflowing
numeric declarations refuse. No twin simulation or physical comparison occurs.

`write_digital_twin_reference_report` writes sorted UTF-8 JSON, creates parents
and refuses selected root/input aliases (direct, resolved, symlink or hardlink)
before mutation; unrelated output is replaced. Discovery is sequential, with
no locks or concurrent pathname guarantee. API failures propagate. Argparse
and registered `validate-digital-twin-reference` return authored stderr and
exit one on supported inspection/write failures, including cyclic paths;
parser help/usage retain zero/two and declaration pass/findings return zero/one.
The cold script runs from an unrelated directory without `PYTHONPATH`.

```python
from tempfile import TemporaryDirectory
from validation.validate_digital_twin_reference import (
    validate_digital_twin_reference,
    write_digital_twin_reference_report,
)

with TemporaryDirectory() as directory:
    report = validate_digital_twin_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail"
    write_digital_twin_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

Tokamak digital-twin validation claims require persisted public-reference,
measured-discharge replay, or external integrated-modelling artifacts for grid
topology, q-profile evolution, actuator latency, IDS export, and island-mask
checks:

Bounded synthetic online model-update evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family digital-twin-online-update \
  --artifact report=validation/reports/digital_twin_online_update.json \
  --artifact markdown=validation/reports/digital_twin_online_update.md \
  -- python validation/benchmark_digital_twin_online_update.py
```

This writes `validation/reports/digital_twin_online_update.json` and
`validation/reports/digital_twin_online_update.md`. The benchmark exercises
Bayesian updating of density, effective charge, and actuator dynamics against a
synthetic reference and publishes the bounded evidence digests for simulator
metadata, observation targets, priors, and Bayesian-update results. Admission
revalidates finite non-negative loss histories, best-parameter bounds, source
binding, and simulator unit coverage for every observation target. TRANSP/TSC
coupling requires validated external simulator metadata and the strict
reference gate below before measured replay claims.

```bash
scpn-control validate-digital-twin-reference --require-reference-artifacts --json-out
python validation/validate_digital_twin_reference.py --require-reference-artifacts --output-json artifacts/digital_twin_reference_report.json
```

Strict mode fails until `validation/reports/digital_twin_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash, grid
metadata, actuator and sensor metadata, unit contracts, case count, and
digital-twin replay metrics inside declared tolerances.

The stored SOC reader selects sorted immediate JSON entries or one regular file
regardless of suffix. Optional absent/nonfile roots pass with zero entries;
required mode refuses. Invalid UTF-8/JSON, duplicate keys, unreadable inputs and
nonzero decimals rounded to binary64 zero return authored findings without
exception/input-key text. Each file is inspected once, without a coherent
snapshot or runtime/learning execution.

Schema `1.0` retains six nonblank identities and one exact 64-hex reference-hash
shape. Reference bytes are not fetched or authenticated. Public references
need nonblank URL or DOI; measured declarations need nonblank `shot_id` and
`diagnostic_uri`; external references need one of `CGYRO`, `GENE`, `GS2`, `TGLF`,
`QuaLiKiz` and nonblank artifact URI. Citation and URI policy is presence-only.

Lattice metadata keeps nonboolean uncapped integer `size >= 8`, positive
`time_steps`/`max_sub_steps` and nonnegative `seed`. Declared `z_crit_base`,
`flow_generation` and `shear_efficiency` are finite and nonnegative;
`flow_damping` lies in `[0, 1)`. Learning `alpha`, `gamma`, `epsilon` are in
`[0, 1]`; state/action counts are positive nonboolean uncapped integers and
reward text is nonblank. Six units are dimensionless and time uses `step`.
Five finite nonnegative errors must not exceed positive finite declared bounds;
equality and representable subnormals remain admitted. Huge scalar conversion
failures become findings, without allocating the declared lattice or Q-table.

These are stored declaration contracts. The actual `CoupledSandpileReactor`
constructor requires positive `z_crit_base`, while original stored metadata
admits zero. Reader pass does not establish runtime constructibility or
physical comparison. The runtime has no dependency on this persisted reader;
this command performs no sandpile simulation or Q-learning.

`write_soc_reference_report` creates parents and writes sorted UTF-8 reports,
refusing direct/resolved/symbolic/existing hardlink aliases of selected inputs
before writing. Unrelated output can replace. API operational errors propagate;
Click and script use fixed authored stderr with pass/findings 0/1, operational
refusal 1 and parser help/usage 0/2. Sequential alias checks do not protect
against concurrent pathname changes.

```python
from tempfile import TemporaryDirectory
from validation.validate_soc_reference import validate_soc_reference, write_soc_reference_report
with TemporaryDirectory() as directory:
    report = validate_soc_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    write_soc_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

SOC turbulence-learning validation claims require persisted public-reference,
measured-turbulence replay, or external gyrokinetic-reference artifacts for the
sandpile lattice, flow coupling, shear suppression, Q-learning policy, and
reward-behaviour checks:

```bash
scpn-control validate-soc-reference --require-reference-artifacts --json-out
python validation/validate_soc_reference.py --require-reference-artifacts --output-json artifacts/soc_reference_report.json
```

Strict mode fails until `validation/reports/soc_reference/` contains artifacts
with source provenance, model identity, SHA-256 reference hash, lattice
metadata, Q-learning metadata, unit contracts, case count, and SOC replay
metrics inside declared tolerances.

Free-boundary tracking diagnostics on the real `FusionKernel` path:

```bash
python tools/run_recorded_benchmark.py \
  --family free-boundary-tracking-acceptance \
  --artifact report=validation/reports/free_boundary_tracking_acceptance.json \
  --artifact markdown=validation/reports/free_boundary_tracking_acceptance.md \
  -- python validation/free_boundary_tracking_acceptance.py --require-thresholds
```

The acceptance report covers nominal tracking, external coil-current kicks,
topology-aware X-point/divertor tracking under kick disturbance,
measurement-fault exposure and correction for both generic shape tracking and
topology-aware X-point/divertor tracking, supervisor/fallback safety under a
large kick, and severity sweeps for generic disturbance, topology-aware
disturbance, generic measurement faults, delayed-measurement latency and
latency compensation, topology-aware measurement faults, and actuator limits,
topology-aware measurement-plus-latency scenarios, plus combined topology
disturbance-and-calibration-fault scenarios including actuator-constrained
supervisor/fallback lanes and their measurement-severity sweeps.

The fixed cohort contains eighteen four-step scenarios and twelve sweeps on a
12-by-12 grid. Permeability and plasma-current target are configured as 1.0.
Shape, X-point and divertor targets are sampled from the same solver; corrected
measurements subtract the exact injected bias/drift. Flux residuals use that
normalisation, and tracking norms combine configured objectives. This is
same-model regression evidence with no independent physical-reference,
observer-calibration, facility-control or safety admission. Schema v2 records
these limits and retains the existing timestamp/runtime/campaign fields.
Runtime includes campaign and timestamp construction, excluding rendering and
writes. Temporary configurations are deleted after execution. Shared public
threshold dictionaries remain mutable and affect later runs; no concurrent
snapshot or lock exists.

Temporary output paths can be selected directly. Persistent evidence requires
the recorded runner. Selected source and output/output aliases refuse before
computing. JSON refuses nonfinite values; JSON and Markdown writes are
sequential. Ordinary exit zero reports completion; require-thresholds returns
one after both outputs if a diagnostic check fails. Input/custody/execution/IO
refusals return two with authored stderr. A threshold pass leaves independent
physical admission false.

This writes:

- `validation/reports/free_boundary_tracking_acceptance.json`
- `validation/reports/free_boundary_tracking_acceptance.md`

Bounded free-boundary claim-admission evidence can be regenerated with:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family free-boundary-tracking-claims \
  --artifact report=validation/reports/free_boundary_tracking_claims.json \
  --artifact markdown=validation/reports/free_boundary_tracking_claims.md \
  -- python validation/benchmark_free_boundary_tracking_claims.py
```

This writes `validation/reports/free_boundary_tracking_claims.json` and
`validation/reports/free_boundary_tracking_claims.md`. These artefacts
demonstrate deterministic claim-admission plumbing using a fixed 8-by-4 linear
fixture with a zero Psi grid, rather than a magnetic equilibrium solve. Its
config path is a report label and is not opened by the fixture; facility-control
claims remain gated by strict reference artefacts.

The persisted free-boundary reference reader selects sorted immediate `*.json`
directory entries or a regular file regardless of suffix. Optional absent or
nonfile roots pass with zero entries; required mode refuses. Invalid UTF-8/JSON,
duplicate keys, unreadable files and nonzero decimal tokens rounded to binary64
zero produce authored findings. Reads are sequential observations.

Original schema `1.0` requires six nonblank identity strings and an exact
64-character hex reference SHA declaration. This format-only SHA authenticates
no referenced bytes or payload. Public provenance needs URL or DOI presence;
measured free-boundary replay needs shot and diagnostic URI presence; external
equilibrium benchmarks need a named `EFIT`, `P-EFIT`, `CREATE-NL` or `TSC` string
and nonblank artifact URI presence. URI text is not parsed, fetched or
verified; original relative/parent/NUL presence text remains admitted.

Coil, boundary-point and divertor-point counts are positive nonboolean integers
without an upper cap or cross-count relation. Control interval and coil slew
are finite positive values, admitting representable subnormals. Units remain
`m`, `Wb/rad`, `MA`, `s` and dimensionless tracking error `1`, without flux or
current conversion. Reference case count is an uncapped positive nonboolean
integer. Five declared shape, X-point position/flux, divertor and coil-current
errors are finite nonnegative with positive finite bounds; equality passes.
Overflowing numbers refuse. No equilibrium solution, coil dynamics, replay,
physical comparison or facility-control admission is established.

`write_free_boundary_reference_report` writes sorted UTF-8 JSON, creates parents
and refuses direct/resolved/symlink/hardlink selected root/input aliases before
mutation. Unrelated output is replaced; discovery is sequential without locks
or concurrent pathname guarantees. API errors propagate. Argparse and
registered `validate-free-boundary-reference` return fixed authored stderr and
exit one for supported inspection/write failures, including cyclic paths;
help/usage retain zero/two and declaration pass/findings return zero/one. The
cold script runs from an unrelated directory without `PYTHONPATH`.

```python
from tempfile import TemporaryDirectory
from validation.validate_free_boundary_reference import (
    validate_free_boundary_reference,
    write_free_boundary_reference_report,
)

with TemporaryDirectory() as directory:
    report = validate_free_boundary_reference(directory, require_reference_artifacts=True)
    assert report["status"] == "fail"
    write_free_boundary_reference_report(report, directory + "/report.txt", artifact_root=directory)
```

Free-boundary tracking validation claims require persisted public-reference,
measured free-boundary replay, or external equilibrium benchmark artifacts for
shape, X-point, divertor, and coil-current agreement:

```bash
scpn-control validate-free-boundary-reference --require-reference-artifacts --json-out
python validation/validate_free_boundary_reference.py --require-reference-artifacts --output-json artifacts/free_boundary_reference_report.json
```

Strict mode fails until `validation/reports/free_boundary_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash, unit
contracts, equilibrium metadata, case count, and free-boundary metrics inside
declared tolerances.

## CODAC/EPICS boundary evidence

CODAC runtime reports use `scpn-control.codac-runtime-evidence.v3`. Admission
requires boolean declarations for finite/clamped output enforcement, generated
analog `DRVH`/`DRVL` drive limits, and fail-closed interlock execution. A
facility claim also requires independently verified runtime and hardware origin;
the caller-supplied timing samples, interlock counters, and flag cannot provide
that origin. Facility admission currently fails closed, including for re-sealed
artifacts that set the flag. Version 1 and 2 reports are rejected and must be
regenerated as local-only v3 reports.

At runtime, every configured external interlock PV and every hard-limit process
signal is mandatory. Missing, non-numeric, non-finite, tripped, or out-of-range
input returns a complete zero-output packet without calling the controller;
external binary interlocks define `0.0` as clear and every non-zero value as a
trip.
Finite controller actions are clamped to the declared output channel envelope;
non-finite actions are rejected before a packet is returned. This validates a
bounded software adapter only, not independent plant protection or a deployed
PCS.

## CI workflows

- Core CI coordinator: `.github/workflows/ci.yml`
- Core CI responsibility owners: `.github/workflows/ci-*.yml`, governed by
  `tools/ci_workflow_policy.json` and checked by
  `python tools/check_ci_workflow_modularity.py`
- Docs and Pages deployment: `.github/workflows/docs-pages.yml`
- PyPI publish workflow: `.github/workflows/publish-pypi.yml`

## CI quality gates in the distributed core CI

- `python-tests` (3.11/3.12/3.13 Ubuntu + 3.12 Windows + 3.12 macOS; mypy + coverage on 3.12)
- `native-coverage-combine` (combines rust-absent `coverage-data-python` with
  Rust-present `coverage-data-rust` and gates `coverage-report-combined`)
- `python-lint` (ruff check + ruff format)
- `python-security` (bandit SAST)
- `python-audit` (`pip_audit`)
- `data-manifest-gate` (data manifest provenance and local artefact checksum report)
- `release-evidence-gate` (`scpn-control validate --json-out` over data
  provenance, strict persisted JAX GK CPU/GPU parity evidence, and physics
  traceability; validates the generated JSON with `scpn-control
  validate-release-evidence`; uploads `release-evidence-report` with the raw
  report and admission report)
- `python-benchmark` (E2E control latency)
- `python validation/validate_e2e_latency_evidence.py <report> --max-e2e-p95-us 1000 --json-out`
  admits only schema-versioned, digest-bound latency reports with preserved
  benchmark command, generated timestamp, host-load/isolation context, bounded
  local-evidence claim metadata, and qualified target-hardware fields for
  real-time evidence.
- `python validation/validate_benchmark_regression_gates.py` admits only
  persisted benchmark-gate manifests whose referenced reports match their file
  SHA-256, whose observed metrics and sample counts match the report payloads,
  and whose embedded `payload_sha256` / `report_payload_sha256` self-digests
  validate recursively.
  Each file is hashed and decoded from the same captured bytes. Relative
  external manifest paths resolve reports from their resolved parent;
  literal URI components and resolved symlink containment are checked before
  reading reports. Duplicate/nonfinite/overflowing JSON, declared null
  self-digests and expected file failures produce structured refusals.
  This checks persisted metadata, not timestamp freshness, authenticated
  hardware origin, host isolation or new timing measurements; the full
  [API contract](api.md#validation-benchmark-regression-gates) describes its
  tolerances, units and filesystem/error semantics.
- Controller safety-case readiness resolves typed readiness artifacts under an explicit artifact root and verifies their SHA-256 bytes. HIL replay schema v2 and CODAC runtime schema v3 refuse caller-authored qualified claims without independent hardware/runtime origin; these artifacts currently block promotion admission. The `target_hardware_timing`, `hdl_export_evidence`, and `websocket_runtime_evidence` gates retain their own strict validators.
- `notebook-smoke` (executes CI notebook set; full neuro notebook only if `sc_neurocore` is available)
- `package-quality` (`build` + `twine check`)
- `rmse-gate` consumes `scpn-control.rmse-dashboard.v1` and the bounded
  reference-evidence v1 report. It requires populated, available confinement
  tau RMSE (seconds), SPARC grid-axis consistency RMSE (metres), beta_N RMSE
  (dimensionless) and synthetic disruption FPR with a positive safe-shot count.
  Missing, skipped, nonfinite, malformed or unversioned legacy input fails;
  none of these comparisons grants held-out physics or facility validation.
  Missing historical burn models produce an unavailable beta lane with a null
  metric rather than using reference targets as predictions. Report generation
  may succeed while this regression gate fails. `scpn-control validate-rmse`
  writes the report and puts optional figures beside its JSON destination.
  The historical `validation/rmse_dashboard.py` script also exposes the same
  public functions. Calculations belong to `rmse_dashboard_metrics.py`, report
  presentation to `rmse_dashboard_rendering.py`, and argument parsing and local
  output generation to `rmse_dashboard_command.py`. The renderer accepts supplied
  comparison mappings; populated beta rows do not prove a model run. Its display
  thresholds remain separate from regression admission. PNG generation closes
  each figure even if preparing rows or writing its file fails; earlier output
  files can remain after a later write failure. Missing optional Matplotlib
  produces no figures while Markdown generation remains available.
- `e2e-diiid` (end-to-end DIII-D replay plumbing with synthetic fixtures, not public physics evidence)
- `synthetic-diiid-reference` (DIII-D-like synthetic fixture replay plumbing,
  not public physics evidence)
- `jax-parity` (JAX transport, neural equilibrium, GS solver parity tests, and
  strict persisted CPU/GPU JAX GK parity evidence admission)
- `nengo-loihi` (LIF+NEF SNN wrapper emulator tests)
- `rust-tests` (`cargo test --workspace` + clippy + fmt)
- `rust-python-interop` (maturin build + PyO3 parity)
- `rust-benchmarks` (Criterion, uploads `bench-results` artifact)
- `rust-audit` (cargo-audit vulnerability scan)
- `cargo-deny` (license + advisory supply-chain policy)
## Federated disruption synthetic multi-facility benchmark

Run:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family federated-disruption \
  --artifact report=validation/reports/federated_disruption_benchmark.json \
  --artifact markdown=validation/reports/federated_disruption_benchmark.md \
  -- python validation/benchmark_federated_disruption.py
```

Outputs:

- `validation/reports/federated_disruption_benchmark.json`
- `validation/reports/federated_disruption_benchmark.md`

Scope: deterministic synthetic DIII-D/JET/KSTAR/EAST facility distributions,
FedProx aggregation, and facility-update differential privacy accounting.
This is not measured cross-facility validation; measured claims remain blocked
until external facility shot databases and provenance manifests are supplied.

## Practical use and scope

Use this as the admission boundary for all measurable scientific and software claims.

- Route every timing, physics, and safety statement through this evidence surface.
- Use this page as the source of truth before updating external-facing claims.
- Keep this section synchronized with any new validator, benchmark, or experiment workflow.


### Density declaration report persistence

The public density reader checks original schema `1.0`, geometry/actuator/units,
source-specific declarations and four errors within declared tolerances. Source,
DOI/shot/code/URI/digest fields are metadata; no fetch, model execution, referenced
byte authentication, metric recomputation or facility/control admission occurs.
Optional absence still passes zero; required absence fails. Decoder read/UTF8/JSON/
duplicate failures have fixed findings. Nonfinite floating tokens and nonzero
binary64 underflow are refused globally, including unused fields. The original
`density reference JSON numbers must be finite` refusal remains unchanged.

`write_density_reference_report` and both commands protect the root and immediate
selected direct/resolved/symlink/existing hardlink aliases before sorted UTF8 JSON
plus LF. Nonfinite report serialisation refuses. Other output may replace and API
errors propagate; commands use fixed operational refusal one. Click directory/NUL
output remains usage two. Checks are sequential without locks or snapshots.

```python
from tempfile import TemporaryDirectory
from validation.validate_density_reference import validate_density_reference

with TemporaryDirectory() as directory:
    report = validate_density_reference(directory, require_reference_artifacts=True)
assert report["status"] == "fail"
assert report["reference_artifacts"] == 0
```


### Current-drive declaration report persistence

The public reader retains original metadata-only schema1.0 and five inclusive
error/tolerance checks, eight exact units and positive source/grid metadata. Source
labels, digest, DOI/shot/code/URI remain declarations; no referenced bytes, solver
result or physics reference is authenticated. Nonzero binary64 underflow refuses
at every JSON depth. Original authored decoding findings remain unchanged.

`write_current_drive_reference_report` and both commands protect root/immediate
selected direct/resolved/symlink/existing hardlink inputs before sorted UTF8 JSON+LF,
with nonfinite serialisation refused. Other output may replace; API failures
propagate. Script write refusal2 and root ClickException1 with its Error prefix
remain original; Click directory/NUL output uses usage2. Reads and checks are
sequential without locks, snapshots or atomic writes. The actual bounded analytic
corpus stays refused as external reference evidence.

```python
from validation.validate_current_drive_reference import ROOT, validate_current_drive_reference

report = validate_current_drive_reference(ROOT / "validation/reports/current_drive_claims.json")
assert report["status"] == "fail"
assert report["reference_artifacts"] == 0
```


### Manual native component timing

With the existing real native package available, run from the checkout:

```bash
python tools/benchmark_full_stack.py
```

This stdout-only tool probes Rust equilibrium, a synthetic controller tick and
one Kuramoto call separately. Caller-relative `iter_config.json` is needed for
the equilibrium probe; a solver exception is printed and the next probes continue.
Missing `scpn_control_rs` fails at import, with no automatic installation/build.
The controller uses original two-place topology, one `ctrl` action, explicit Rust
backend, ten warmups and1000 measured ticks. Its corrected public exporter uses
the canonical `actions` list instead of obsolete parallel-name/place keys.

The three probes exchange no solved state, have no actuator or saved evidence
artifact, and establish no physical correctness, closed-loop or facility admission.
Global random oscillator inputs and uncontrolled host state prevent a reproducible
statistical speedup claim. Python/native overhead is included. Existing baseline
failed at artifact export before controller timing, so no controller/oscillator
speedup or comparable full-stack baseline is inferred from the repair. Real public
script/API tests use an existing extension explicitly without installing it or
substituting a backend; the loaded binary is hash pinned, not assumed built from
current source. Original core equations, native binding calls and workload unchanged.


### Metadata report input custody

Choose a report path distinct from selected source inputs. This executable
example inspects the actual default corpora and writes separate reports:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from validation.validate_data_manifests import main as data_main
from validation.validate_physics_traceability import main as physics_main
from validation.generate_physics_traceability_report import main as markdown_main

with TemporaryDirectory() as output:
    folder = Path(output)
    assert data_main(["--output-json", str(folder / "manifests.json"), "--json-out"]) == 0
    assert physics_main(["--output-json", str(folder / "traceability.json"), "--json-out"]) == 0
    assert markdown_main(["--output-md", str(folder / "traceability.md")]) == 0
```

The commands refuse direct, resolved, symbolic-link and existing hard-link aliases
of their protected input paths before creating parents or writing bytes. A
manifest report also protects its selected root, discovered manifest/acquisition
specification/required DIII-D files and resolvable local artifacts in
metadata-valid manifests. Invalid/unresolvable artifact declarations add no
additional paths. A physics report protects its selected registry, rather than
all source/evidence files referenced by that registry. Unrelated report files
retain ordinary replacement behaviour. These checks are sequential and supply
no concurrent snapshot, lock, atomic write or authenticated report custody.

The registered 'scpn-control validate-data-manifests' and
'scpn-control validate-physics-traceability' commands use the same path predicates.
Their options and help remain unchanged. Output refusal is exit1; JSON/data
commands preserve diagnostic FAIL output, while the physics Click command gives
a fixed operational error. The Markdown command still validates the registry
before rendering/writing. No schema, source-validation rule, mathematical
formula, threshold or admission flag changes here.

'tests/test_report_output_paths.py' exercises actual normal and stdlib-only
scripts, public Python APIs and registered commands using copied canonical
registry bytes and existing synthetic DIII-D metadata. Direct, normalised,
symbolic-link, hard-link and linked-parent aliases preserve copied input bytes.
Non-pattern artifacts such as the original FreeGS coilset JSON are protected;
unrelated existing destinations receive actual UTF-8 JSON or Markdown. No
mocked producer, new physical reference or acquisition authenticity is inferred.


### E2E latency report boundary tests

The E2E reader's public tests execute the installed Python producer with three
measured iterations, one warmup and the shipped 16x16 grid. The actual recorded
runner retains command/status and verifies an immutable raw-report copy as
software_boundary_test. That wrapper custody does not grant hardware or source
authentication. The original observation has unqualified labels and production
permission False; it passes only with allow-local-unqualified.

Reader boundary cases derive clearly authored field declarations from that
actual report and rebuild them through the public builder. They are not new
timing measurements. API/script/imported-main tests cover finite inclusive
budgets, invalid-budget refusal before I/O, explicit UTC, metadata/checksum and
native JSON errors. Normal docs enforcement includes all three reader modules
and their fixtures/tests, including private helpers; genuine removed-doc cases
exercise the ordinary command. No mocks, skipped cases, private production
calls or hardware claims are used. The tiny observation establishes software
behaviour, not a statistically useful p95 or a comparative speedup.


### Manufactured mesh and differentiable reader checks

The mesh study runs an independent manufactured polynomial Dirichlet SOR
stencil. Its17/33 regression and complete17/33/65/129 command check spatial
error reduction for that fixture, not canonical FusionKernel fidelity. A zero
or negative iteration cap now reports zero completed sweeps; the numerical
update/error/residual formulas are preserved. Scientific report destinations
are left unchanged by running the actual command under temporary software-only
record custody.

Differentiable reader tests run the full byte-identical shipped21-point/four-step
producer on installed CPU JAX, including public finite-difference audit and
one warmup/five timed runs. They retain immutable one-step/rollout/readiness
reports; missing external/formal evidence keeps readiness blocked. Field-error
cases are authored derivatives of those actual observations, not new timings.
They verify per-entry fail/count consistency, readiness clearing on errors,
integer schema domains, overflow refusal and actual script/imported-main CLI
outcomes. Authored blocked declarations test metadata without pretending the
installed backend is absent. All private reader contracts and test helpers
are included in ordinary removed-doc enforcement.

A true differentiable-readiness declaration also requires external_reference_admitted=True
and non-null valid external-reference/controller digest declarations. This checks
producer-defined internal consistency; it does not authenticate those artifacts.

### Synthetic disturbance benchmark public checks

The actual checks in `tests/test_disturbance_runtime.py` and
`tests/test_disturbance_commands.py` exercise the
[synthetic disturbance API](api.md#synthetic-disturbance-rejection-api). They execute
public PID/MPC/DGKF/SC-NeuroCore APIs, the direct script, imported cli and the
compatible imported main. They use actual installed SC-NeuroCore cells;
provider availability is not mocked, installed or replaced. The default-duration
strict command writes real reports/plots before its incomplete-cohort exit.
Bounded complete commands use duration-scale 0.001 and the same forcings/dt.

Independent quadrature checks real terminal traces and interval effort. A
zero-gain PID verifies actual instability truncation. The real defining DGKF
controller and plant reproduce the benchmark's corrected measurement sign.
Fresh actual SNN pools verify complete reset. Input-domain, overflow,
source/report/plot alias and persistent-custody refusals preserve prior bytes.
The owning zero-floor doc gate includes private definitions, constructors and
new test helpers, with removed-doc probes against the actual copied gate.
These are software diagnostics, not physical validation or controlled native
performance measurements. Optional unavailable-provider branches remain
explicit qualification limits rather than substituted execution evidence.


## Local source/data archive candidate

Create a new candidate outside the checkout, with an existing parent directory:

```bash
python tools/export_zenodo_dataset.py --output /existing/output/directory/candidate.zip
```

The exporter includes current eligible Python source, tests and tools, the listed
validation/example files, Markdown docs, Rust sources/manifests and root metadata.
It excludes `docs/internal`, Git-ignored files and symlink inputs. Nonignored
untracked public source is included; the archive is not tied to a commit. The
metadata version only names the ZIP prefix. The recovered 527-row ML350 NPZ,
weights, native binaries and videos are outside its selection, and this command
does not validate scientific provenance or authorise publication/training.

Exit 0 means creation completed. Exit 2 means an input, Git selection or output
refusal; argparse usage errors also return 2. Existing outputs are preserved and
missing output parents are not created. Creation is sequential, and failed writes
can leave a partial candidate. See the [native API and executable example](api.md#local-sourcedata-zip-export).


## Phase-model video

Set `PHASE_VIDEO_OUTPUT` to a new directory under an existing parent, then run
the real seeded model frontend from the checkout:

```bash
python tools/generate_phase_video.py --ticks 8 --layers 2 --n-per 3 --zeta 0.5 --fps 5 --output-dir "$PHASE_VIDEO_OUTPUT"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames -of json "$PHASE_VIDEO_OUTPUT/phase_sync_live.mp4"
```

The bundle contains `phase_sync_live.gif`, `phase_sync_live.mp4` and
`phase_video.json`. Matplotlib and Pillow are required, and MP4 additionally
requires a real H.264-capable FFmpeg. `--ffmpeg-path PATH` selects its executable;
`--gif-only` explicitly omits MP4, with no automatic fallback. The test suite uses
real FFmpeg and ffprobe to verify H.264, dimensions, FPS and frame count, plus
Pillow to decode GIF frame count/timing and an independent actual monitor replay
to verify displayed numerical values. Decoded GIF frames also verify that the
model-value/guard footer is present. Tests do not skip absent encoders.

Defaults are 500 ticks, 16 layers, 50 oscillators per layer, ζ0.5, FPS20 and a new
caller-relative `phase-video` directory. The parent must exist. Existing targets,
including dangling symlinks, refuse before capture. Invalid inputs or actual
render/encoder failure give fixed stderr and exit 2; a complete bundle gives 0.
Argparse help gives 0 and usage errors2. API exceptions remain available to callers.
A failure after directory creation may retain partial new outputs; inspect them
before choosing a different new destination.

Model dt 0.001 and playback FPS describe different clocks. The original floor
stride and appended final sample remain; trace coordinates use actual ticks 1..n.
JSON retains all displayed tick metrics and explicit false physical-reference
admission. Model guard HALT remains visible rather than becoming a successful
reactor protection claim. Current default R≈0.154,V≈0.846 at tick 500 does not
reproduce the old R=0.92,V→0 caption in the retained historical media.
See the [API contract](api.md#phase-model-video) for validation ranges, memory,
mutable input, output sequencing and concurrent-rendering limits.


The [MAST native source acquisition contract](api.md#mast-native-source-acquisition)
now rejects empty/descending/duplicate selections and invalid reproducibility
labels before outputs, reserves isolated per-shot caches, and refuses source
identity drift/object arrays before NPZ publication. Direct writes retain explicit
partial-I/O behaviour. API/CLI refusals and self-consistent source-object declarations
do not establish a live Zarr-v3/S3 acquisition or scientific training admission.
The `mast-acquisition` extra supplies Zarr 3, xarray, fsspec and s3fs in a separate
source-checkout environment. `mast-data`, `all` and `dev` use the Zarr 2 conversion
profile and cannot be combined with it. Follow the linked API's installation and
source-module command; dependency installation alone does not prove upstream
availability, an immutable chunk snapshot or original acquisition provenance.

The checked-in `validation/reports/density_control.json` remains a historical
v2 artifact with public claims disabled in the lifecycle registry. Generate a
separate v3 report with the current validator; the v3 consumer refuses that
older schema. A successful bounded calculation alone does not admit it as
current scientific or control evidence.
