<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Validation and QA -->

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

## Quantum disruption bridge

`scpn_control.control.quantum_disruption_bridge` keeps quantum circuit,
Qiskit/PennyLane, and provider-specific execution in `scpn-quantum-control`.
SCPN-CONTROL exposes only a control-grade facade with lazy optional imports,
strict CONTROL-to-ITER feature mapping, explicit centre-default provenance,
bounded amplitude-kernel reports, and tamper-evident advisory disruption
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

## Federated disruption synthetic multi-facility benchmark

Run:

```bash
python validation/benchmark_federated_disruption.py
```

Outputs:

- `validation/reports/federated_disruption_benchmark.json`
- `validation/reports/federated_disruption_benchmark.md`

Scope: deterministic synthetic DIII-D/JET/KSTAR/EAST facility distributions,
FedProx aggregation, and facility-update differential privacy accounting.
This is not measured cross-facility validation; measured claims remain blocked
until external facility shot databases and provenance manifests are supplied.

# Validation and QA

## Python tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/ -q
```

Coverage gate (matches CI threshold):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin -p pytest_cov tests/ --cov=scpn_control --cov-report=term --cov-fail-under=93
```

The `93` gate is temporary debt aligned to the latest full CI measurement
(`93.74%` on the May 31, 2026 main-branch run). New recovery work must add
module-specific behavioural tests for concrete production surfaces rather than
synthetic line-hit tests.

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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/test_rust_python_parity.py tests/test_rust_compat_wrapper.py tests/test_snn_pyo3_bridge.py -v
```

## Local acceptance campaigns

Real-data manifest provenance gate:

```bash
scpn-control validate --json-out
scpn-control validate-release-evidence artifacts/release_evidence_report.json --json-out
scpn-control validate-manifest validation/reference_data/diiid/manifests/diiid_hmode_1p5MA.geqdsk.manifest.json --verify-artifact
scpn-control validate-manifest validation/reference_data/diiid/manifests/shot_163303_hmode.npz.manifest.json --verify-artifact --json-out
scpn-control validate-manifest validation/reference_data/diiid/manifests/mock_diiid_ci.manifest.json --json-out
```

The top-level `validate` command now includes repository data-manifest
validation, strict persisted JAX GK parity evidence admission, and physics
traceability validation by default, so routine validation cannot pass while
ignoring data provenance, backend parity evidence drift, or bounded-claim
registry drift. The local `tools/preflight.py` path now runs this top-level
release-evidence gate as a non-test gate, including in `make preflight-fast`,
and validates the generated JSON report with `validate-release-evidence`, so
release preflight cannot skip provenance, parity, claim-boundary drift, or
artifact-admission drift when tests are intentionally omitted. Use
`--data-manifest-root` for staged facility drops,
`--jax-gk-parity-root` for staged parity campaigns,
`--physics-traceability-registry` for staged claim-boundary registries,
`--no-verify-artifacts` for metadata-only manifest checks, and
`--no-data-manifests`, `--no-jax-gk-parity`, or `--no-physics-traceability` only
for explicitly scoped import-hygiene checks. `validate-release-evidence` admits
the resulting JSON report as a release artifact by rejecting duplicate keys,
skipped or failing mandatory gates, incomplete CPU/GPU JAX GK parity case
coverage, and traceability reports that do not block every open fidelity gap.
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

CI validates the full manifest set and writes a JSON evidence report:

```bash
scpn-control validate-data-manifests --json-out
python validation/validate_data_manifests.py --output-json artifacts/data_manifest_report.json
```

The report also enforces DIII-D artefact coverage: every tracked DIII-D GEQDSK
and disruption-shot NPZ under `validation/reference_data/diiid/` must be covered
by a manifest entry with a local SHA-256 checksum. It reports acquisition-spec
readiness as `realised` or `pending`; by default pending facility pulls are
visible but do not break normal metadata validation.

Strict facility-campaign gate:

```bash
scpn-control validate-data-manifests --require-real-acquisition --json-out
python validation/validate_data_manifests.py --require-real-acquisition --output-json artifacts/data_manifest_report.json
```

This strict mode fails if an acquisition specification, such as the DIII-D
MDSplus shot spec, does not have a corresponding real `mdsplus` manifest with
the expected dataset id and checksum-covered acquired artefact.

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
python validation/benchmark_kuramoto_runtime_evidence.py \
  --output-json artifacts/kuramoto_runtime_evidence.json
```

The produced JSON uses `scpn-control.kuramoto-runtime-evidence.v1` and binds
the input phase/frequency arrays by SHA-256 instead of storing the arrays in
the evidence payload. Deployment-claim admission requires optional Rust parity
against the Python reference, deployment-target oscillator count coverage, and
timestep-refinement convergence under the declared tolerance. Python-only
reports remain bounded runtime evidence and do not satisfy deployment claims.

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
scpn-check-generated-traceability
```

Z3-backed SCPN formal evidence is published as schema-versioned JSON and
Markdown. The JSON uses `scpn-control.z3-formal-report.v1`, binds the proof
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

```bash
python validation/validate_scpn_z3_formal.py
python validation/validate_scpn_z3_formal.py --require-z3
```

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

Short finite traces remain useful diagnostics, but they are reported as
insufficient saturation evidence rather than quantitative nonlinear CBC
validation.

Linear GK cross-code agreement claims require immutable real external-code run
evidence. Parser fixtures and published reference numbers are useful readiness
checks, but they do not prove quantitative agreement against actual binaries:

```bash
scpn-control validate-gk-crosscode --require-external-runs --json-out
python validation/validate_gk_crosscode.py --require-external-runs --output-json artifacts/gk_crosscode_report.json
```

Strict mode fails until `validation/reports/gk_crosscode/` contains real-binary
evidence with code identity, version, run id, execution timestamp, units, native
and external growth rates, real frequencies, and dominant wavenumber agreement
inside the declared tolerances. Evidence reports are schema-versioned as
`scpn-control.gk-crosscode.v1` and must bind the external input deck, external
output, native input, and canonical report payload by SHA-256 digest. External-code
`binary_path` provenance must be an absolute filesystem path under an admitted
deployment or facility executable root; URI, relative, traversal, temporary, or
system-control paths are rejected.

Miller geometry validation compares repository flux-tube geometry output against
immutable circular, shaped, and high-shear reference cases:

```bash
scpn-control validate-gk-geometry-reference --json-out
python validation/validate_gk_geometry_reference.py --output-json artifacts/gk_geometry_reference_report.json
```

Gyrokinetic species validation compares mass, charge, thermal speed,
Larmor-radius normalisation, and collision-frequency coefficients against
immutable electron, main-ion, impurity, and extreme-temperature reference cases:

```bash
scpn-control validate-gk-species-reference --json-out
python validation/validate_gk_species_reference.py --output-json artifacts/gk_species_reference_report.json
```

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

GK OOD detector deployment claims require persisted calibration artifacts with a
declared 10D feature schema, training-distribution metadata, threshold
provenance, and false-positive / false-negative acceptance metrics:

```bash
scpn-control validate-gk-ood-calibration --require-campaign-artifacts --json-out
python validation/validate_gk_ood_calibration.py --require-campaign-artifacts --output-json artifacts/gk_ood_calibration_report.json
```

Strict mode fails until `validation/reports/gk_ood_calibration/` contains real,
external-code, facility, or published GK campaign calibration evidence.

External GK interface parser claims require persisted artifacts from real
solver executables or documented public reference outputs. Mock subprocess
fixtures remain parser-readiness checks only:

```bash
scpn-control validate-gk-interface-artifacts --require-interface-artifacts --json-out
python validation/validate_gk_interface_artifacts.py --require-interface-artifacts --output-json artifacts/gk_interface_artifacts_report.json
```

Strict mode fails until `validation/reports/gk_interfaces/` contains interface
artifacts using schema `scpn-control.gk-interface-artifact.v1` with code
identity, source provenance, version, run id, execution timestamp, safe deck,
raw-output, and parsed-output artifact URIs, SHA-256 hashes for each of those
artifacts, a canonical payload SHA-256 hash, parser version, explicit `m^2/s`,
`c_s/a`, and `k_y*rho_s` units, finite transport coefficients, growth rate,
real frequency, and dominant wavenumber. Real-executable artifacts must also
declare an admitted absolute `binary_path`; URI, relative, traversal,
temporary, or system-control paths are not accepted as executable provenance.

Neural equilibrium cross-validation claims require persisted P-EFIT or
documented public reference artifacts for the same surrogate weights and
equilibrium cases. Synthetic training runs and local smoke tests do not count as
matched equilibrium-reference evidence:

Synthetic neural-equilibrium pretraining evidence can be regenerated with:

```bash
python validation/benchmark_neural_equilibrium_pretraining.py
```

This writes `validation/reports/neural_equilibrium_pretraining.json`,
`validation/reports/neural_equilibrium_pretraining.md`, and JAX-compatible
synthetic pretraining weights. These artefacts demonstrate pretraining and
inference plumbing only; real EFIT/P-EFIT fine-tuning remains gated by the
strict reference-artifact validator below.

```bash
scpn-control validate-neural-equilibrium-reference --require-reference-artifacts --json-out
python validation/validate_neural_equilibrium_reference.py --require-reference-artifacts --output-json artifacts/neural_equilibrium_reference_report.json
```

Strict mode fails until `validation/reports/neural_equilibrium_reference/`
contains artifacts with source provenance, surrogate identity, weight and
reference SHA-256 hashes, grid shape, psi/pressure/q/boundary unit contracts,
reference-equilibrium count, and error metrics inside declared tolerances.

Neural transport surrogate validation claims require persisted QuaLiKiz or
documented public reference artifacts for the same QLKNN-style feature schema
and trained weights:

Bounded local neural-transport claim evidence can be regenerated with:

```bash
python validation/benchmark_neural_transport_claims.py
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

Neural turbulence surrogate validation claims require persisted gyrokinetic
campaign or documented public reference artifacts for the same QLKNN-class
feature schema and trained weights:

Bounded local neural-turbulence claim evidence can be regenerated with:

```bash
python validation/benchmark_neural_turbulence_claims.py
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

Uncertainty quantification claims require persisted published-reference or
campaign artifacts for the full propagation chain:

```bash
scpn-control validate-uncertainty-reference --require-reference-artifacts --json-out
python validation/validate_uncertainty_reference.py --require-reference-artifacts --output-json artifacts/uncertainty_reference_report.json
```

Strict mode fails until `validation/reports/uncertainty_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash, case
count, tau_E/P_fusion/Q/sigma unit contracts, and relative-error plus percentile
monotonicity metrics inside declared tolerances.

VMEC-lite stellarator-equilibrium validation claims require persisted
published-reference or real-VMEC-run artifacts for surface geometry,
rotational-transform, Fourier truncation, and force-residual checks:

```bash
scpn-control validate-vmec-reference --require-reference-artifacts --json-out
python validation/validate_vmec_reference.py --require-reference-artifacts --output-json artifacts/vmec_reference_report.json
```

Strict mode fails until `validation/reports/vmec_reference/` contains artifacts
with source provenance, model identity, SHA-256 reference hash, Fourier
truncation, unit contracts, case count, and surface/iota/residual metrics inside
declared tolerances. Real-VMEC artifact URIs must use an admitted scheme
(`file`, `https`, `s3`, or `gs`); local `file://` URIs must stay under
`/validation/reports/` or `/validation/reference_data/`.

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
artifacts with source provenance, model identity, SHA-256 reference hash, radial
grid metadata, actuator settings, unit contracts, case count, and
admitted external-artifact URI syntax when external integrated-modelling
artifacts are cited.
fuelling-profile metrics inside declared tolerances.

DT burn-control validation claims require persisted documented public,
integrated transport benchmark, or measured burn replay artifacts for alpha
power, Q, Lawson margin, burn fraction, and reactivity-exponent checks:

Bounded local burn-control claim evidence can be regenerated with:

```bash
python validation/benchmark_burn_control_claims.py
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

Volt-second scenario validation claims require persisted documented public,
measured loop-voltage replay, or external scenario benchmark artifacts for total
flux, flat-top duration, Ejima flux, bootstrap current, and budget-margin
checks:

Bounded local volt-second claim evidence can be regenerated with:

```bash
python validation/benchmark_volt_second_claims.py
```

This writes `validation/reports/volt_second_claims.json` and
`validation/reports/volt_second_claims.md`. These artefacts demonstrate
deterministic scenario-accounting claim-admission plumbing only; pulse-duration
or solenoid-commissioning claims remain gated by the strict reference-artifact
validator below.

```bash
scpn-control validate-volt-second-reference --require-reference-artifacts --json-out
python validation/validate_volt_second_reference.py --require-reference-artifacts --output-json artifacts/volt_second_reference_report.json
```

Strict mode fails until `validation/reports/volt_second_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash,
machine metadata, unit contracts, case count, and volt-second metrics inside
declared tolerances.

Auxiliary current-drive validation claims require persisted documented public,
ray-tracing, Fokker-Planck, or measured-deposition artifacts for absorbed power,
driven current, deposition centroid, peak current density, and NBI slowing-down
checks:

Bounded local current-drive claim evidence can be regenerated with:

```bash
python validation/benchmark_current_drive_claims.py
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

Strict mode fails until `validation/reports/current_drive_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash, source
metadata, unit contracts, case count, and current-drive metrics inside declared
tolerances.

Static mu-analysis validation claims require persisted documented public,
external mu-toolbox, or measured control replay artifacts for mu upper bound,
robustness margin, controller gain, D-scaling, and closed-loop spectral
abscissa checks:

Bounded local mu-synthesis claim evidence can be regenerated with:

```bash
python validation/benchmark_mu_synthesis_claims.py
```

This writes `validation/reports/mu_synthesis_claims.json` and
`validation/reports/mu_synthesis_claims.md`. These artefacts demonstrate
deterministic static mu-analysis claim-admission plumbing only. The persisted
JSON carries a canonical payload SHA-256 digest and `load_mu_synthesis_claim_evidence()`
rejects duplicate keys, schema drift, edited metric fields, and bounded
evidence presented as a validated robust-control claim. Full frequency-dependent
D-K synthesis claims remain gated by the strict reference-artefact validator
below.

```bash
scpn-control validate-mu-synthesis-reference --require-reference-artifacts --json-out
python validation/validate_mu_synthesis_reference.py --require-reference-artifacts --output-json artifacts/mu_synthesis_reference_report.json
```

Strict mode fails until `validation/reports/mu_synthesis_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash, plant
metadata, unit contracts, case count, and mu-analysis metrics inside declared
tolerances.

Disruption-mitigation contract validation claims require persisted
public-reference, measured-disruption, or external benchmark artifacts for
warning lead time, mitigation outcome, halo current, runaway beam, and TBR
equivalence checks:

Bounded local disruption-mitigation claim evidence can be regenerated with:

```bash
python validation/benchmark_disruption_mitigation_claims.py
```

This writes `validation/reports/disruption_mitigation_claims.json` and
`validation/reports/disruption_mitigation_claims.md`. These artefacts
demonstrate deterministic halo/runaway ensemble and claim-admission plumbing
only; mitigation validation remains gated by the strict reference-artifact
validator below.

```bash
scpn-control validate-disruption-reference --require-reference-artifacts --json-out
python validation/validate_disruption_reference.py --require-reference-artifacts --output-json artifacts/disruption_reference_report.json
```

Strict mode fails until `validation/reports/disruption_reference/` contains
artifacts with source provenance, model identity, SHA-256 reference hash,
disruption-window timing, mitigation-cocktail metadata, unit contracts, case
count, and disruption-mitigation metrics inside declared tolerances.

Bounded local differentiable-transport gradient evidence can be regenerated
with:

```bash
python validation/benchmark_differentiable_transport_latency.py
python validation/validate_differentiable_transport_latency.py --require-admitted --json-out
```

This writes `validation/reports/differentiable_transport_latency.json`,
`validation/reports/differentiable_transport_latency.md`,
`validation/reports/differentiable_transport_rollout_latency.json`, and
`validation/reports/differentiable_transport_rollout_latency.md`. The reports
exercise the audited JAX gradient-admission path when JAX is available and
otherwise publish a blocked-backend status. Persisted evidence fails closed on
non-finite or negative audit losses/errors, tolerance drift from campaign
metadata, duplicate or out-of-domain sampled audit indices, inconsistent
pass/fail flags, malformed latency run counts, and unordered latency
percentiles. The standalone validator admits those persisted reports before
they are used as release evidence. The rollout source-gradient loss remains inside the traced JAX
graph, and the module enables JAX x64 before importing `jax.numpy` so benchmark
dtype evidence matches the requested differentiable transport precision.

Tokamak digital-twin validation claims require persisted public-reference,
measured-discharge replay, or external integrated-modelling artifacts for grid
topology, q-profile evolution, actuator latency, IDS export, and island-mask
checks:

Bounded synthetic online model-update evidence can be regenerated with:

```bash
python validation/benchmark_digital_twin_online_update.py
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

Free-boundary tracking acceptance on the real `FusionKernel` path:

```bash
python validation/free_boundary_tracking_acceptance.py
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

This writes:

- `validation/reports/free_boundary_tracking_acceptance.json`
- `validation/reports/free_boundary_tracking_acceptance.md`

Bounded free-boundary claim-admission evidence can be regenerated with:

```bash
python validation/benchmark_free_boundary_tracking_claims.py
```

This writes `validation/reports/free_boundary_tracking_claims.json` and
`validation/reports/free_boundary_tracking_claims.md`. These artefacts
demonstrate deterministic claim-admission plumbing only; facility-control
claims remain gated by strict reference artefacts.

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

## CI workflows

- Core CI: `.github/workflows/ci.yml`
- Docs and Pages deployment: `.github/workflows/docs-pages.yml`
- PyPI publish workflow: `.github/workflows/publish-pypi.yml`

## CI quality gates in `.github/workflows/ci.yml`

- `python-tests` (3.10/3.11/3.12/3.13 Ubuntu + 3.12 Windows + 3.12 macOS; mypy + coverage on 3.12)
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
  admits only schema-versioned, digest-bound, qualified target-hardware latency
  reports for real-time evidence.
- Controller safety-case readiness resolves typed readiness artifacts under an explicit artifact root, verifies their SHA-256 bytes, and admits `target_hardware_timing`, `hil_replay_evidence`, `hdl_export_evidence`, `codac_runtime_evidence`, and `websocket_runtime_evidence` only after their strict evidence gates pass.
- `notebook-smoke` (executes CI notebook set; full neuro notebook only if `sc_neurocore` is available)
- `package-quality` (`build` + `twine check`)
- `rmse-gate` (SPARC and DIII-D GEQDSK regression bounds)
- `e2e-diiid` (end-to-end DIII-D replay plumbing with synthetic fixtures, not public physics evidence)
- `real-diiid` (DIII-D reference disruption-shot archive)
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
python validation/benchmark_federated_disruption.py
```

Outputs:

- `validation/reports/federated_disruption_benchmark.json`
- `validation/reports/federated_disruption_benchmark.md`

Scope: deterministic synthetic DIII-D/JET/KSTAR/EAST facility distributions,
FedProx aggregation, and facility-update differential privacy accounting.
This is not measured cross-facility validation; measured claims remain blocked
until external facility shot databases and provenance manifests are supplied.
