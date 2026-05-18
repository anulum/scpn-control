<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

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
scpn-control validate-manifest validation/reference_data/diiid/manifests/diiid_hmode_1p5MA.geqdsk.manifest.json --verify-artifact
scpn-control validate-manifest validation/reference_data/diiid/manifests/shot_163303_hmode.npz.manifest.json --verify-artifact --json-out
scpn-control validate-manifest validation/reference_data/diiid/manifests/mock_diiid_ci.manifest.json --json-out
```

The gate separates experimental validation evidence from CI fixtures. A manifest
claiming real-shot validation must include a non-synthetic source kind, machine,
shot, signal paths, physical units, retrieval timestamp, checksum, and licence
or facility data policy. Local real-data manifests can additionally verify the
referenced artefact checksum with `--verify-artifact`. Synthetic manifests remain
allowed for CI, but require generator and seed metadata and are reported as
`kind: synthetic`.

CI validates the full manifest set and writes a JSON evidence report:

```bash
scpn-control validate-data-manifests --json-out
python validation/validate_data_manifests.py --output-json artifacts/data_manifest_report.json
```

The report also enforces DIII-D artefact coverage: every tracked DIII-D GEQDSK
and disruption-shot NPZ under `validation/reference_data/diiid/` must be covered
by a manifest entry with a local SHA-256 checksum.

Optional MDSplus acquisition writes both the acquired NPZ and a validated
manifest at retrieval time. Use a checked acquisition specification for
repeatable facility pulls; it requires the facility MDSplus Python client in the
runtime environment:

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
