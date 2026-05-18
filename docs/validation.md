<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Validation and QA -->

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
with `--verify-artifact`. Synthetic manifests remain allowed for CI, but require
generator and seed metadata and are reported as `kind: synthetic`. Manifest and
acquisition-spec JSON is parsed with duplicate-key rejection so provenance fields
cannot be overwritten silently by ambiguous objects.

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
- `notebook-smoke` (executes CI notebook set; full neuro notebook only if `sc_neurocore` is available)
- `package-quality` (`build` + `twine check`)
- `rmse-gate` (SPARC GEQDSK + synthetic DIII-D regression bounds)
- `e2e-diiid` (end-to-end mock DIII-D shot test)
- `real-diiid` (17 real DIII-D disruption shots)
- `jax-parity` (JAX transport, neural equilibrium, GS solver parity tests)
- `nengo-loihi` (LIF+NEF SNN wrapper emulator tests)
- `rust-tests` (`cargo test --workspace` + clippy + fmt)
- `rust-python-interop` (maturin build + PyO3 parity)
- `rust-benchmarks` (Criterion, uploads `bench-results` artifact)
- `rust-audit` (cargo-audit vulnerability scan)
- `cargo-deny` (license + advisory supply-chain policy)
