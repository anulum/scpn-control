# Validation and QA

## Python tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/ -q
```

Coverage gate (matches CI threshold):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin -p pytest_cov tests/ --cov=scpn_control --cov-report=term --cov-fail-under=50
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

## CI workflows

- Core CI: `.github/workflows/ci.yml`
- Docs and Pages deployment: `.github/workflows/docs-pages.yml`
- PyPI publish workflow: `.github/workflows/publish-pypi.yml`

## CI quality gates in `.github/workflows/ci.yml`

- `python-tests` (3.9/3.10/3.11/3.12, mypy + coverage on 3.12)
- `notebook-smoke` (executes CI notebook set; full neuro notebook only if `sc_neurocore` is available)
- `package-quality` (`build` + `twine check`)
- `python-audit` (`pip_audit`)
- `rmse-gate`
- `rust-tests`
- `rust-python-interop`
- `rust-benchmarks` (uploads `bench-results` artifact from `scpn-control-rs/target/criterion/`)
- `rust-audit`
