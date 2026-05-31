<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Validation and QA -->


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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin -p pytest_cov tests/ --cov=scpn_control --cov-report=term --cov-fail-under=99
```

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
scpn-control validate-manifest validation/reference_data/diiid/manifests/diiid_hmode_1p5MA.geqdsk.manifest.json --verify-artifact
scpn-control validate-manifest validation/reference_data/diiid/manifests/shot_163303_hmode.npz.manifest.json --verify-artifact --json-out
scpn-control validate-manifest validation/reference_data/diiid/manifests/mock_diiid_ci.manifest.json --json-out
```

The top-level `validate` command now includes repository data-manifest
validation by default, so routine validation cannot pass while ignoring data
provenance. Use `--data-manifest-root` for staged facility drops,
`--no-verify-artifacts` for metadata-only checks, and `--no-data-manifests` only
for explicitly scoped import-hygiene checks. The gate separates experimental
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

Physics traceability validates that high-risk physics surfaces are bounded to
their current evidence status before full-fidelity or facility-validation claims
are made:

```bash
scpn-control validate-physics-traceability --json-out
python validation/validate_physics_traceability.py --output-json artifacts/physics_traceability_report.json
python validation/generate_physics_traceability_report.py --output-md docs/physics_traceability.md
scpn-check-generated-traceability
```

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
inside the declared tolerances.

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
scpn-control validate-jax-gk-parity --require-parity-artifacts --json-out
python validation/validate_jax_gk_parity.py --require-parity-artifacts --output-json artifacts/jax_gk_parity_report.json
```

Strict mode fails until `validation/reports/jax_gk_parity/` contains parity
artifacts for the declared backend campaign. Live smoke tests remain useful
diagnostics, but they do not replace persisted CPU/GPU/TPU parity evidence.

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
artifacts with code identity, source provenance, version, run id, execution
timestamp, input and output SHA-256 hashes, parser version, units, finite
transport coefficients, growth rate, real frequency, and dominant wavenumber.

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
contains artifacts with source provenance, surrogate identity, weight and
reference SHA-256 hashes, QLKNN-10D feature ordering, target-unit contracts,
reference-sample count, and chi_i/chi_e/D_e plus branch-accuracy metrics inside
declared tolerances.

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
deterministic static mu-analysis claim-admission plumbing only; full
frequency-dependent D-K synthesis claims remain gated by the strict
reference-artifact validator below.

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
synthetic reference. TRANSP/TSC coupling requires validated external simulator
metadata and the strict reference gate below before measured replay claims.

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
- `python-benchmark` (E2E control latency)
- `python validation/validate_e2e_latency_evidence.py <report> --max-e2e-p95-us 1000 --json-out`
  admits only qualified target-hardware latency reports for real-time evidence.
- `notebook-smoke` (executes CI notebook set; full neuro notebook only if `sc_neurocore` is available)
- `package-quality` (`build` + `twine check`)
- `rmse-gate` (SPARC and DIII-D GEQDSK regression bounds)
- `e2e-diiid` (end-to-end DIII-D replay plumbing with synthetic fixtures, not public physics evidence)
- `real-diiid` (DIII-D reference disruption-shot archive)
- `jax-parity` (JAX transport, neural equilibrium, GS solver parity tests)
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
