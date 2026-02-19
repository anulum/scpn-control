# Validation and QA

## Python tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/ -q
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
python -m maturin develop --release --manifest-path scpn-control-rs/crates/control-python/Cargo.toml
python -m pip install -e .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/test_rust_python_parity.py tests/test_rust_compat_wrapper.py tests/test_snn_pyo3_bridge.py -v
```

## CI workflows

- Core CI: `.github/workflows/ci.yml`
- Docs and Pages deployment: `.github/workflows/docs-pages.yml`
